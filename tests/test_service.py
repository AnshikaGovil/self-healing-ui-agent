"""Contract tests for the healing service and the network seam.

These run against the FastAPI app in-process with the AI adjudicator disabled,
so they need no GPU, no weights and no network. What they check is the part
that would break silently in production: the shape of the contract between a
bot and the service, and the promise that an unreachable service degrades a
step rather than crashing a bot.

    python tests/test_service.py
"""

import os
import sys

ROOT_ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_)

from bis_core.adjudicator import (AdjudicationRequest,      # noqa: E402
                                  LocalAdjudicator,
                                  RemoteAdjudicator,
                                  build_adjudicator,
                                  reset_adjudicators)
from bis_core.config import HealingConfig                   # noqa: E402
from bis_core.descriptors import (ElementDescriptor,        # noqa: E402
                                  LocatorSpec)


def _config(**kw) -> HealingConfig:
    cfg = HealingConfig()
    cfg.use_ai = False            # no weights needed for contract tests
    cfg.use_embeddings = False
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def _spec() -> LocatorSpec:
    return LocatorSpec(
        key="plp.product_card", selector=".pd19-product-card__item",
        description="every smartphone product card in the listing grid",
        tag="div", role="text", multiple=True)


def _candidates():
    def el(classes, text):
        return ElementDescriptor(
            tag="div", classes=list(classes), text=text, visible=True,
            rect={"x": 10, "y": 10, "width": 300, "height": 700},
            selectors=["div." + classes[0]])
    return [
        el(["pd21-product-card__options-wrap"], "Colour : Mint"),
        el(["pd21-product-card__item", "js-pfv2-product-card"], "Galaxy F70 Pro"),
        el(["pd21-product-card__rating-wrap"], "Ratings 4.9"),
    ]


def _request(spec=None, candidates=None) -> AdjudicationRequest:
    return AdjudicationRequest.build(
        spec=spec or _spec(), action="count",
        url="https://www.samsung.com/in/smartphones/all-smartphones/",
        title="Samsung India", candidates=candidates or _candidates(),
        baseline=None, shortlist_size=8, budget_s=20.0)


def test_request_strips_selectors_from_the_payload():
    """Selector proposals are the bulk of a descriptor and only the bot uses
    them, so they must not be shipped across the network."""
    req = _request()
    assert req.candidates
    for cand in req.candidates:
        assert "selectors" not in cand
    assert req.locator["multiple"] is True


def test_payload_stays_small():
    """A repair request has to be cheap enough to make inline, mid-process."""
    import json
    size = len(json.dumps(_request().to_dict()))
    assert size < 20_000, "payload grew to %d bytes" % size


def test_local_adjudicator_ranks_without_any_model():
    """With AI and embeddings off, the deterministic signals must still pick
    the renamed component."""
    decision = LocalAdjudicator(_config()).adjudicate(_request())
    assert decision.chosen == 1
    assert decision.strategy == "heuristic"
    assert decision.ai_consulted is False
    assert decision.confidence > 0.4
    assert [e.index for e in decision.shortlist][0] == 1


def test_build_adjudicator_picks_by_configuration():
    reset_adjudicators()
    assert isinstance(build_adjudicator(_config()), LocalAdjudicator)
    remote = build_adjudicator(_config(service_url="http://gpu-host:8077"))
    assert isinstance(remote, RemoteAdjudicator)
    reset_adjudicators()


def test_adjudicator_is_shared_per_process():
    """A local adjudicator owns gigabytes of weights. One agent per browser
    tab must not mean one copy of the model per browser tab."""
    reset_adjudicators()
    a = build_adjudicator(_config())
    b = build_adjudicator(_config())
    assert a is b
    reset_adjudicators()


def test_unreachable_service_degrades_instead_of_raising():
    """A healing service being down must fail the step, never the bot."""
    remote = RemoteAdjudicator("http://127.0.0.1:9", timeout_s=2.0)
    decision = remote.adjudicate(_request())
    assert decision.chosen == -1
    assert decision.error
    assert remote.health()["ready"] is False


def test_service_endpoints_round_trip():
    """The HTTP contract itself: health, auth, protocol check, adjudication."""
    from fastapi.testclient import TestClient
    from healing_service import create_app

    app = create_app(_config(), api_key="secret", warm=False)
    with TestClient(app) as client:
        health = client.get("/v1/health").json()
        assert health["service"] == "bot-immune-system-healing"
        assert health["protocol"] == "1"

        body = _request().to_dict()

        # auth is enforced
        assert client.post("/v1/adjudicate", json=body).status_code == 401

        headers = {"X-API-Key": "secret"}
        res = client.post("/v1/adjudicate", json=body, headers=headers)
        assert res.status_code == 200, res.text
        data = res.json()
        assert data["chosen"] == 1
        assert data["shortlist"][0]["index"] == 1
        assert "selector" not in data      # the bot binds selectors, not us

        # an old client must be told, not silently mis-served
        stale = dict(body, protocol="0")
        assert client.post("/v1/adjudicate", json=stale,
                           headers=headers).status_code == 409

        empty = dict(body, candidates=[])
        assert client.post("/v1/adjudicate", json=empty,
                           headers=headers).status_code == 422


def test_confidence_bands_are_defined_once():
    """apply / review / refuse, at the exact boundaries.

    Regression: the agent used to have no branch for the refuse band, so a
    repair scoring below the review threshold was applied anyway whenever a
    selector happened to resolve. The policy now lives in one place and both
    the engine and the agent read it from there.
    """
    cfg = HealingConfig()
    cfg.auto_apply_threshold = 0.62
    cfg.review_threshold = 0.42
    assert cfg.decide(0.95) == cfg.APPLY
    assert cfg.decide(0.62) == cfg.APPLY          # inclusive lower bound
    assert cfg.decide(0.619) == cfg.REVIEW
    assert cfg.decide(0.42) == cfg.REVIEW         # inclusive lower bound
    assert cfg.decide(0.419) == cfg.REFUSE
    assert cfg.decide(0.0) == cfg.REFUSE


# --------------------------------------------------- the locator contract


def test_the_service_derives_role_from_the_payload_tag():
    """Both sides of the seam must agree without the payload growing a field."""
    from bis_core.adjudicator import spec_from_payload
    assert spec_from_payload({"selector": "a.buy"}).role == "link"
    assert spec_from_payload({"selector": "a.buy", "role": "text"}).role == "text"


def test_unknown_locator_keys_warn_instead_of_disappearing():
    """A typo must be visible - but must never take a running bot down."""
    import logging

    records = []

    class Capture(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    log = logging.getLogger("bis.descriptors")
    handler = Capture()
    log.addHandler(handler)
    try:
        spec = LocatorSpec.from_dict("login.user",
                                     {"selector": ".a", "discription": "oops"})
    finally:
        log.removeHandler(handler)

    assert spec.selector == ".a"          # fail open: the spec still built
    assert any("discription" in m for m in records)
    assert any("description" in m for m in records)   # suggests the right key


def test_a_locator_without_a_selector_names_itself():
    try:
        LocatorSpec.from_dict("plp.price", {"description": "the price"})
    except KeyError as exc:
        assert "plp.price" in str(exc)
    else:
        raise AssertionError("a selector-less locator must not load")


def test_the_service_serves_the_harvesting_javascript():
    """A thin client downloads its JS rather than shipping it.

    This is what lets the harvester improve without redeploying a fleet, so
    the endpoint existing - and carrying a content hash to cache against -
    is part of the contract, not an extra.
    """
    from fastapi.testclient import TestClient
    from healing_service import create_app

    client = TestClient(create_app(_config(), warm=False))
    body = client.get("/v1/js").json()

    assert body["protocol"] == "1"
    assert len(body["version"]) >= 8, "no content hash to cache against"
    for part in ("harvest", "probe", "describe_one", "presence"):
        assert body[part].strip().startswith("("), "%s is not a function" % part
    # Same bundle, same hash - or every bot re-downloads it on every call.
    assert client.get("/v1/js").json()["version"] == body["version"]


def test_an_optional_locator_is_reported_absent_rather_than_repaired_weakly():
    """The asymmetry that makes try/except-with-pass safe to replace.

    A weak repair on a required locator is still worth reporting - the bot
    decides what to do with it. The same repair on an optional one must come
    back as "absent", because nothing raises when an optional step proceeds,
    so a wrong element would simply be clicked in silence.
    """
    from fastapi.testclient import TestClient
    from healing_service import create_app

    client = TestClient(create_app(_config(), warm=False))
    candidates = [
        {"tag": "button", "elementId": "promo-close",
         "classes": ["promo__close"], "text": "Close", "visible": True,
         "interactive": True, "rect": {"x": 10, "y": 10, "w": 30, "h": 30},
         "selectors": ["#promo-close"]},
    ]

    def ask(optional):
        return client.post("/v1/adjudicate", json={
            "locator": {"key": "promo.close", "selector": "#gone",
                        "description": "dismiss the promo popup"},
            "action": "check",
            "page": {"url": "https://x.test/", "title": "X"},
            "candidates": candidates, "optional": optional,
            "protocol": "1"}).json()

    required, optional = ask(False), ask(True)
    assert required["confidence"] == optional["confidence"],         "the flag must change the verdict, not the score"
    assert required["confidence"] < HealingConfig().optional_min_confidence
    assert required["chosen"] == 0, "a required locator still gets an answer"
    assert optional["chosen"] == -1, "a weak optional repair must not be applied"
    assert "floor" in optional["rationale"]


def _run_all():
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print("  PASS  %s" % name)
        except AssertionError as exc:
            failures += 1
            print("  FAIL  %s  %s" % (name, exc))
        except Exception as exc:
            failures += 1
            print("  ERROR %s  %s: %s" % (name, type(exc).__name__, exc))
    print("\n%s" % ("all tests passed" if not failures
                    else "%d test(s) failed" % failures))
    return failures


if __name__ == "__main__":
    raise SystemExit(_run_all())
