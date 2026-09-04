"""FastAPI application for the healing service.

Why this exists
---------------
Locator repair needs two things that do not belong on the same machine: a
browser (which every bot has) and an accelerator (which almost none of them
should have to). Putting the model-backed half behind an API means one GPU
host serves a whole fleet, the weights are provisioned and versioned in one
place, and a bot needs nothing heavier than Playwright to benefit from it.

What actually crosses the wire
------------------------------
Element fingerprints for the shortlist the bot already believes in - tag,
attributes, label text, geometry, ancestor chain. Not the page, not the DOM,
not a screenshot, and not the values the bot is typing.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse

from bis_core.adjudicator import (PROTOCOL_VERSION, AdjudicationRequest,
                                  LocalAdjudicator)
from bis_core.config import HealingConfig
from bis_core.js_harvest import (DESCRIBE_ONE_JS, HARVEST_JS, PRESENCE_JS,
                                 PROBE_JS)
from .schemas import (AdjudicateRequest, AdjudicateResponse,
                      HealthResponse, JsBundleResponse)

log = logging.getLogger("bis.service")

def _js_bundle() -> dict:
    """The JS payload plus a content hash clients cache against."""
    body = {"harvest": HARVEST_JS, "probe": PROBE_JS,
            "describe_one": DESCRIBE_ONE_JS, "presence": PRESENCE_JS}
    digest = hashlib.sha256()
    for name in sorted(body):
        digest.update(name.encode("utf-8"))
        digest.update(body[name].encode("utf-8"))
    body["version"] = digest.hexdigest()[:16]
    return body


JS_BUNDLE = _js_bundle()

SERVICE_NAME = "bot-immune-system-healing"
SERVICE_VERSION = "1.1.0"


class ServiceState:
    """Process-wide state: one loaded model, shared by every request."""

    def __init__(self, config: HealingConfig, api_key: str = ""):
        self.config = config
        self.api_key = api_key
        self.adjudicator = LocalAdjudicator(config)
        self.started_at = time.time()
        self.requests = 0
        self.warm = {}


def create_app(config: Optional[HealingConfig] = None,
               api_key: str = "", warm: bool = True) -> FastAPI:
    """Build the ASGI app. Loading happens at startup, not on first request."""
    config = config or HealingConfig()
    # Environment wins, so a container can be configured without editing code.
    config.model_path = os.environ.get("BIS_MODEL_PATH", config.model_path)
    config.gpu_layers = int(os.environ.get("BIS_GPU_LAYERS", config.gpu_layers))
    api_key = api_key or os.environ.get("BIS_API_KEY", "")
    # A service that phoned another service would defeat the point.
    config.service_url = ""

    state = ServiceState(config, api_key)

    app = FastAPI(
        title="Bot Immune System - Healing Service",
        version=SERVICE_VERSION,
        summary="Ranks and adjudicates replacement locators for RPA bots "
                "whose selectors have broken.",
    )
    app.state.bis = state

    def require_key(x_api_key: str = Header(default="")) -> None:
        """Optional shared-secret auth.

        Unset means open, which is right for a demo on localhost and wrong for
        anything reachable from a network. The startup log says which it is.
        """
        if state.api_key and x_api_key != state.api_key:
            raise HTTPException(status_code=401, detail="invalid or missing API key")

    @app.on_event("startup")
    def _startup() -> None:
        if state.api_key:
            log.info("API key auth enabled")
        else:
            log.warning("no API key set - this service is open to anyone who "
                        "can reach it (set BIS_API_KEY to require one)")
        if warm:
            # Pay the load cost now. Otherwise the first bot to hit a broken
            # locator also pays for loading the weights, and the latency
            # numbers in its run report are a lie.
            log.info("loading models ...")
            state.warm = state.adjudicator.warmup()
            log.info("ready: %s", state.warm)

    @app.get("/v1/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        info = state.adjudicator.describe()
        ai = info.get("ai", {})
        return HealthResponse(
            ready=bool(ai.get("loaded")) or not state.config.use_ai,
            service=SERVICE_NAME,
            version=SERVICE_VERSION,
            protocol=PROTOCOL_VERSION,
            encoder_ready=bool(info.get("encoder_ready")),
            ai=ai,
            uptime_s=round(time.time() - state.started_at, 1),
            requests_served=state.requests,
        )

    @app.get("/")
    def root() -> JSONResponse:
        return JSONResponse({
            "service": SERVICE_NAME,
            "version": SERVICE_VERSION,
            "protocol": PROTOCOL_VERSION,
            "endpoints": ["/v1/health", "/v1/js", "/v1/adjudicate", "/docs"],
        })

    @app.get("/v1/js", response_model=JsBundleResponse)
    def js_bundle() -> JsBundleResponse:
        """Serve the in-page JavaScript the bot executes.

        This is what keeps a thin client thin. The harvester, the probe and
        the single-element fingerprinter are the parts most likely to need
        improving as sites change, and serving them means that work ships
        centrally instead of as a redeploy to every bot.

        The bot still runs the JavaScript itself and still binds its own
        selectors - the service never sees the page and never returns a
        selector.
        """
        return JsBundleResponse(
            protocol=PROTOCOL_VERSION,
            version=JS_BUNDLE["version"],
            harvest=JS_BUNDLE["harvest"],
            probe=JS_BUNDLE["probe"],
            describe_one=JS_BUNDLE["describe_one"],
            presence=JS_BUNDLE["presence"],
        )

    @app.post("/v1/adjudicate", response_model=AdjudicateResponse,
              dependencies=[Depends(require_key)])
    def adjudicate(body: AdjudicateRequest) -> AdjudicateResponse:
        """Rank the supplied candidates and choose one.

        Returns an *index* into the list the bot sent, never a selector: the
        bot owns selector binding because only it can prove a selector
        resolves on the page in front of it.
        """
        if body.protocol != PROTOCOL_VERSION:
            raise HTTPException(
                status_code=409,
                detail="protocol %s not supported (service speaks %s)"
                       % (body.protocol, PROTOCOL_VERSION))
        if not body.candidates:
            raise HTTPException(status_code=422, detail="no candidates supplied")

        state.requests += 1
        started = time.time()
        request = AdjudicationRequest(
            locator=body.locator.model_dump(),
            action=body.action,
            page=body.page.model_dump(),
            candidates=body.candidates,
            baseline=body.baseline,
            shortlist_size=body.shortlist_size,
            budget_s=body.budget_s,
        )
        decision = state.adjudicator.adjudicate(request)

        # An optional locator is held to a stricter floor. The bot cannot
        # apply this itself without the policy drifting between fleets, so
        # it declares intent and the service decides - which is also what
        # lets the threshold be retuned without touching a single bot.
        if body.optional and decision.chosen >= 0:
            band = state.config.decide_optional(decision.confidence)
            if band != state.config.APPLY:
                log.info("optional locator %s: %.0f%% is below the optional "
                         "floor of %.0f%% - reporting absent",
                         body.locator.key, decision.confidence * 100,
                         state.config.optional_min_confidence * 100)
                decision.chosen = -1
                decision.rationale = (
                    "optional locator below the %.0f%% floor; treated as "
                    "absent rather than repaired"
                    % (state.config.optional_min_confidence * 100))

        payload = decision.to_dict()
        payload["band"] = (state.config.decide(decision.confidence)
                           if decision.chosen >= 0 else state.config.REFUSE)

        log.info("adjudicated %s (%s) -> candidate %d, %.0f%% via %s in %.1fs",
                 body.locator.key, body.action, decision.chosen,
                 decision.confidence * 100, decision.strategy or "none",
                 time.time() - started)
        return AdjudicateResponse(**payload)

    return app


app = None


def main() -> int:
    """Console entry point: python -m bot_immune_system.service"""
    import argparse
    import uvicorn

    ap = argparse.ArgumentParser(description="Bot Immune System healing service")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8077)
    ap.add_argument("--model", default=None, help="path to the GGUF model file")
    ap.add_argument("--gpu-layers", type=int, default=None,
                    help="-1 offloads everything, 0 forces CPU")
    ap.add_argument("--context", type=int, default=None)
    ap.add_argument("--api-key", default="")
    ap.add_argument("--no-ai", action="store_true",
                    help="serve heuristic + semantic ranking only")
    ap.add_argument("--no-warm", action="store_true")
    ap.add_argument("--log-level", default="info")
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-7s %(name)-16s %(message)s",
        datefmt="%H:%M:%S")

    config = HealingConfig()
    if args.model:
        config.model_path = args.model
    if args.gpu_layers is not None:
        config.gpu_layers = args.gpu_layers
    if args.context is not None:
        config.ai_context = args.context
    config.use_ai = not args.no_ai

    application = create_app(config, api_key=args.api_key, warm=not args.no_warm)
    uvicorn.run(application, host=args.host, port=args.port,
                log_level=args.log_level)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
