"""The adjudication seam.

This is where the system splits across the network.

    on the bot            |  over HTTP  |  on the inference host
    ----------------------+-------------+------------------------------
    harvest the live DOM  |             |
    lexical ranking       |  ---------> |  semantic ranking (embeddings)
    (no model needed)     |             |  AI adjudication (GGUF on GPU)
    bind + validate       |  <--------- |  ranked decision
    apply / escalate      |             |

The split is chosen so that everything needing a GPU is on one side and
everything needing a browser is on the other. The bot ships only element
*fingerprints* for the shortlist it already believes in - roughly 45 of them,
tens of kilobytes - never the page, never a screenshot, never the DOM.

Two implementations satisfy the same contract:

``LocalAdjudicator``   runs in-process. Used by the service, and available to
                      a bot that genuinely has its own accelerator.
``RemoteAdjudicator``  posts to the service and returns its decision. Used by
                      every bot in a fleet, none of which needs a GPU.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from . import runtime
from .config import HealingConfig
from .descriptors import ElementDescriptor, LocatorSpec
from .similarity import HealTarget, score_candidates

log = logging.getLogger("bis.adjudicator")

PROTOCOL_VERSION = "1"


@dataclass
class AdjudicationRequest:
    """What the bot asks the service to decide."""

    locator: Dict[str, Any]
    action: str
    page: Dict[str, str]
    candidates: List[Dict[str, Any]]
    baseline: Optional[Dict[str, Any]] = None
    shortlist_size: int = 8
    budget_s: float = 25.0
    protocol: str = PROTOCOL_VERSION

    @classmethod
    def build(cls, spec: LocatorSpec, action: str, url: str, title: str,
              candidates: List[ElementDescriptor],
              baseline: Optional[ElementDescriptor],
              shortlist_size: int, budget_s: float) -> "AdjudicationRequest":
        return cls(
            locator={
                "key": spec.key,
                "selector": spec.selector,
                "description": spec.description,
                "tag": spec.tag,
                "role": spec.role,
                "expected_text": spec.expected_text,
                "multiple": spec.multiple,
            },
            action=action,
            page={"url": url, "title": title or ""},
            # `selectors` is stripped: only the bot binds selectors, so the
            # service has no use for them and they are the bulk of the payload.
            candidates=[_strip(c.to_dict()) for c in candidates],
            baseline=_strip(baseline.to_dict()) if baseline else None,
            shortlist_size=shortlist_size,
            budget_s=budget_s,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _strip(raw: Dict[str, Any]) -> Dict[str, Any]:
    raw.pop("selectors", None)
    return raw


def spec_from_payload(raw: Dict[str, Any]) -> LocatorSpec:
    return LocatorSpec(
        key=raw.get("key", "locator"),
        selector=raw.get("selector", ""),
        description=raw.get("description", "") or "",
        tag=raw.get("tag"),
        role=raw.get("role"),
        expected_text=raw.get("expected_text"),
        multiple=bool(raw.get("multiple", False)),
    )


@dataclass
class ShortlistEntry:
    """One ranked candidate, as the service saw it."""

    index: int                       # position in the candidate list sent
    score: float
    signals: Dict[str, Optional[float]] = field(default_factory=dict)
    summary: str = ""
    chosen: bool = False


@dataclass
class AdjudicationDecision:
    """What the service decided. Contains no selector - the bot binds that."""

    chosen: int = -1                 # index into the candidates sent, -1 = none
    confidence: float = 0.0
    strategy: str = ""               # heuristic | ai-confirmed | ai-override
    rationale: str = ""
    shortlist: List[ShortlistEntry] = field(default_factory=list)
    ai_consulted: bool = False
    ai_choice: int = -1
    ai_confidence: float = 0.0
    ai_reason: str = ""
    ai_latency_s: float = 0.0
    ai_device: str = ""
    duration_s: float = 0.0
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "AdjudicationDecision":
        entries = [ShortlistEntry(**e) for e in raw.get("shortlist", [])]
        known = {f for f in cls.__dataclass_fields__} - {"shortlist"}
        return cls(shortlist=entries,
                   **{k: v for k, v in raw.items() if k in known})


class LocalAdjudicator:
    """Ranks and adjudicates in-process.

    Owns the two model-backed stages - the sentence encoder and the AI
    adjudicator - and nothing else. It never touches a browser, which is what
    lets it run on a host that has a GPU and no display.
    """

    def __init__(self, config: HealingConfig, model=None, encoder=None):
        self.config = config
        self._model = model
        self._encoder = encoder
        # Identical requests get identical verdicts, and on CPU a verdict costs
        # ~25 seconds. Workers sharding one job arrive here within seconds of
        # each other holding the same shortlist, so this collapses N of those
        # into one.
        self._cache: Dict[str, Any] = {}
        self._cache_lock = threading.Lock()
        self.cache_hits = 0

    @staticmethod
    def _cache_key(request: AdjudicationRequest) -> str:
        """Everything that could change the verdict, and nothing else."""
        payload = json.dumps({
            "key": request.locator.get("key"),
            "selector": request.locator.get("selector"),
            "description": request.locator.get("description"),
            "action": request.action,
            "url": (request.page or {}).get("url", ""),
            "candidates": request.candidates,
            # The baseline drives the structure and geometry signals, so two
            # requests that differ only in whether one was available are not
            # the same question. Leaving it out replays a cold verdict at a
            # cold confidence over a warm heal that would have scored higher.
            "baseline": request.baseline,
        }, sort_keys=True, default=str)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def _cached(self, key: str) -> Optional["AdjudicationDecision"]:
        ttl = getattr(self.config, "decision_cache_s", 0.0)
        if ttl <= 0:
            return None
        with self._cache_lock:
            hit = self._cache.get(key)
            if not hit or time.time() - hit[0] > ttl:
                return None
            self.cache_hits += 1
            # A copy: a caller that mutates the decision must not poison the
            # next worker to ask the same question.
            return AdjudicationDecision.from_dict(hit[1].to_dict())

    def _remember(self, key: str, decision: "AdjudicationDecision") -> None:
        ttl = getattr(self.config, "decision_cache_s", 0.0)
        if ttl <= 0 or decision.chosen < 0:
            return
        limit = max(1, getattr(self.config, "decision_cache_size", 256))
        with self._cache_lock:
            self._cache[key] = (time.time(), decision)
            while len(self._cache) > limit:
                self._cache.pop(next(iter(self._cache)))

    # ------------------------------------------------------------- lazy deps
    @property
    def encoder(self):
        if self._encoder is None and self.config.use_embeddings:
            self._encoder = runtime.make_encoder(self.config.embedding_model)
        return self._encoder

    @property
    def model(self):
        if self._model is None and self.config.use_ai:
            self._model = runtime.make_model(
                model_path=self.config.model_path,
                gpu_layers=self.config.gpu_layers,
                context=self.config.ai_context,
                max_tokens=self.config.ai_max_tokens,
            )
        return self._model

    def warmup(self) -> Dict[str, Any]:
        status = {"encoder": False, "ai": False, "device": "none"}
        if self.encoder is not None:
            status["encoder"] = self.encoder.warmup()
        if self.model is not None:
            status["ai"] = self.model.warmup()
            status["device"] = self.model.device
        return status

    def describe(self) -> Dict[str, Any]:
        out = {"encoder_ready": bool(self.encoder and self.encoder.available)}
        out["ai"] = self.model.describe() if self.model else {"loaded": False}
        out["cache"] = {"entries": len(self._cache), "hits": self.cache_hits}
        return out

    # -------------------------------------------------------------- decide
    def adjudicate(self, request: AdjudicationRequest) -> AdjudicationDecision:
        """Rank the candidates and pick one. Never raises."""
        started = time.time()
        cache_key = self._cache_key(request)
        cached = self._cached(cache_key)
        if cached is not None:
            log.info("serving an identical adjudication from cache")
            cached.duration_s = round(time.time() - started, 3)
            return cached

        decision = AdjudicationDecision()
        spec = spec_from_payload(request.locator)
        # coerce, not from_dict: the full agent sends serialised descriptors
        # and a thin client sends the harvester's own output. Decoding the
        # second shape with from_dict drops id, type, aria-label, data-* and
        # the accessible name, which is most of what identity and attributes
        # are scored on.
        candidates = [ElementDescriptor.coerce(c) for c in request.candidates]
        if not candidates:
            decision.error = "no candidates supplied"
            decision.duration_s = time.time() - started
            return decision

        baseline = (ElementDescriptor.coerce(request.baseline)
                    if request.baseline else None)
        target = HealTarget(spec=spec, baseline=baseline)

        # The bot has already applied the lexical signals to narrow the field.
        # Re-scoring here is cheap and keeps the service self-contained: it
        # never has to trust a score computed on a machine it does not own.
        semantic = None
        if self.encoder is not None and self.encoder.available:
            semantic = self.encoder.similarities(
                spec.intent_sentence(),
                [c.semantic_sentence() for c in candidates],
            )
        ranked = score_candidates(target, candidates, self.config.weights, semantic)
        shortlist = ranked[: max(1, request.shortlist_size)]

        chosen = shortlist[0]
        strategy = "heuristic"
        confidence = chosen.score * 0.92
        rationale = "top-ranked by similarity (%s)" % chosen.explain()

        # ---- AI adjudication -------------------------------------------
        remaining = request.budget_s - (time.time() - started)
        model = self.model
        if model is not None and remaining > 2:
            verdict = model.adjudicate(
                intent=spec.intent_sentence(),
                broken_selector=spec.selector,
                action=request.action,
                page_title=request.page.get("title", ""),
                shortlist=[c.descriptor.compact(i)
                           for i, c in enumerate(shortlist)],
                heuristic_pick=0,
                required_text=spec.expected_text,
            )
            decision.ai_consulted = True
            decision.ai_choice = verdict.choice
            decision.ai_confidence = round(verdict.confidence, 3)
            decision.ai_reason = verdict.reason
            decision.ai_latency_s = round(verdict.latency_s, 2)
            decision.ai_device = verdict.device

            if verdict.ok and 0 <= verdict.choice < len(shortlist):
                picked = shortlist[verdict.choice]
                w = self.config.ai_weight
                fused = (1.0 - w) * picked.score + w * verdict.confidence
                if verdict.choice == 0:
                    # Two independent methods agreeing is real evidence, so the
                    # agreement bonus sits on top of the fused score.
                    strategy = "ai-confirmed"
                    confidence = min(1.0, fused + 0.08)
                else:
                    strategy = "ai-override"
                    confidence = fused
                chosen = picked
                chosen.llm_bonus = confidence - chosen.score
                chosen.llm_reason = verdict.reason
                rationale = "%s: %s" % (strategy,
                                        verdict.reason or "no reason given")
            elif verdict.ok and verdict.choice < 0:
                # An explicit abstention is information: the model looked and
                # saw nothing suitable. Trust the heuristics less, not more.
                strategy = "heuristic-unconfirmed"
                confidence = chosen.score * 0.75
                rationale = "AI found no suitable candidate; " + rationale
            elif verdict.error:
                rationale = "AI unavailable (%s); %s" % (
                    verdict.error.split(":")[0], rationale)

        decision.chosen = chosen.index
        decision.confidence = round(max(0.0, min(1.0, confidence)), 4)
        decision.strategy = strategy
        decision.rationale = rationale
        decision.shortlist = [
            ShortlistEntry(
                index=c.index,
                score=round(c.score, 4),
                signals={k: (round(v, 3) if v is not None else None)
                         for k, v in c.signals.items()},
                summary=c.descriptor.compact(rank),
                chosen=(c is chosen),
            )
            for rank, c in enumerate(shortlist)
        ]
        decision.duration_s = round(time.time() - started, 2)
        self._remember(cache_key, decision)
        return decision


class RemoteAdjudicator:
    """Calls the healing service over HTTP.

    This is what a production bot uses. It has no model, no GPU and no heavy
    dependencies - just the standard library - so rolling the agent out to a
    fleet costs a pip install of Playwright and nothing else.

    Every failure mode returns an empty decision rather than raising, because
    a healing service being down must degrade a bot to "this step failed",
    never to "this bot crashed".
    """

    def __init__(self, base_url: str, api_key: str = "", timeout_s: float = 90.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_s = timeout_s
        self._healthy: Optional[bool] = None

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        return headers

    def health(self) -> Dict[str, Any]:
        try:
            req = urllib.request.Request(self.base_url + "/v1/health",
                                         headers=self._headers())
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            self._healthy = bool(data.get("ready"))
            return data
        except Exception as exc:
            self._healthy = False
            log.warning("healing service unreachable at %s (%s)",
                        self.base_url, exc)
            return {"ready": False, "error": str(exc)}

    def warmup(self) -> Dict[str, Any]:
        info = self.health()
        return {
            "encoder": bool(info.get("encoder_ready")),
            "ai": bool((info.get("ai") or {}).get("loaded")),
            "device": (info.get("ai") or {}).get("device", "remote"),
            "service": self.base_url,
        }

    def describe(self) -> Dict[str, Any]:
        return self.health()

    def adjudicate(self, request: AdjudicationRequest) -> AdjudicationDecision:
        payload = json.dumps(request.to_dict()).encode("utf-8")
        started = time.time()
        try:
            req = urllib.request.Request(self.base_url + "/v1/adjudicate",
                                         data=payload, headers=self._headers(),
                                         method="POST")
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
            return AdjudicationDecision.from_dict(raw)
        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8", "ignore")[:200]
            except Exception:
                pass
            log.error("healing service returned %s: %s", exc.code, body)
            return AdjudicationDecision(
                error="service HTTP %s: %s" % (exc.code, body),
                duration_s=round(time.time() - started, 2))
        except Exception as exc:
            log.error("healing service call failed: %s", exc)
            return AdjudicationDecision(
                error="service unreachable: %s" % exc,
                duration_s=round(time.time() - started, 2))


_ADJUDICATORS: Dict[str, Any] = {}


def build_adjudicator(config: HealingConfig):
    """Pick the adjudicator this process should use, once.

    A service URL in the config means "there is an accelerated host somewhere,
    use it". Without one, this process runs the models itself.

    The result is memoised per target because a local adjudicator owns several
    gigabytes of weights: an agent per browser tab must not mean a copy of the
    model per browser tab.
    """
    key = config.service_url or ("local:" + config.model_path)
    cached = _ADJUDICATORS.get(key)
    if cached is not None:
        return cached
    if config.service_url:
        log.info("using healing service at %s", config.service_url)
        built = RemoteAdjudicator(config.service_url, config.service_api_key,
                                  config.service_timeout_s)
    else:
        log.info("running the adjudicator in-process (%s)", config.model_path)
        built = LocalAdjudicator(config)
    _ADJUDICATORS[key] = built
    return built


def reset_adjudicators() -> None:
    """Drop the cache - for tests, and for reloading swapped-out weights."""
    _ADJUDICATORS.clear()
