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
sys.path.append(os.path.join(os.path.dirname(ROOT_), "bis-core"))

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


# ------------------------------------------------------------- baselines
#
# Structure, geometry, tag and text all compare a candidate against the
# element the step used while it still worked. A client with no memory has
# none, those signals return "no opinion", and the repair is scored on the
# weight that is left. The service keeps baselines so that a thin client
# does not have to become a stateful thing in order to get them.


def _harvested(element_id="firstName"):
    """One candidate in the shape the browser-side harvester emits."""
    return {
        "tag": "input", "id": element_id, "classes": ["form-control"],
        "name": "", "type": "text", "role": "", "ariaLabel": "",
        "placeholder": "First Name", "title": "", "alt": "", "href": "",
        "dataAttrs": {"data-qa": "fname"}, "text": "", "accName": "First Name",
        "xpath": "/html[1]/body[1]/div[1]/input[1]", "cssPath": "#" + element_id,
        "ancestors": ["div.form-group"], "depth": 6, "sibIndex": 2,
        "rect": {"x": 416, "y": 255, "width": 220, "height": 38},
        "visible": True, "enabled": True, "interactive": True,
        "neighborText": "First Name", "selectors": ["#" + element_id],
    }


def test_a_harvester_payload_is_not_decoded_as_a_stored_descriptor():
    """coerce(), not from_dict(), or two thirds of the evidence disappears.

    The full agent runs from_js() in the bot and sends a serialised
    descriptor; a thin client has no descriptor class and sends the
    harvester's own output, which names the same facts differently.
    Decoding the second with from_dict() drops id, type, aria-label, data-*
    and the accessible name - exactly the fields identity and attributes are
    scored on - and does it silently: the request succeeds, the ranking is
    just built on a fraction of the evidence.

    Measured on the live demoqa form, that was the difference between a
    heuristic score of 0.100 and 0.524 for the same correct element.
    """
    raw = _harvested()

    dropped = ElementDescriptor.from_dict(raw)
    assert dropped.element_id == "", "from_dict is supposed to be strict"
    assert "first" not in dropped.attribute_tokens()

    kept = ElementDescriptor.coerce(raw)
    assert kept.element_id == "firstName"
    assert kept.input_type == "text"
    assert kept.data_attrs == {"data-qa": "fname"}
    assert kept.accessible_name == "First Name"
    assert kept.sibling_index == 2
    assert "first" in kept.attribute_tokens()

    # ... and the serialised shape still round-trips, or the full agent breaks.
    assert ElementDescriptor.coerce(kept.to_dict()) == kept
    assert ElementDescriptor.coerce({}).tag == ""


def test_the_store_refuses_a_baseline_that_is_not_on_screen():
    """No baseline beats a false one.

    Structure and geometry are the signals that make repairs better the
    longer a fleet runs. Seeding them from a hidden or zero-sized element
    makes them worse forever, and nothing later in the pipeline can tell.
    """
    import tempfile
    from healing_service.baselines import BaselineStore

    with tempfile.TemporaryDirectory() as folder:
        store = BaselineStore(os.path.join(folder, "b.json"))
        url = "https://demoqa.com/automation-practice-form?utm=x"

        hidden = dict(_harvested(), visible=False)
        assert store.record("form.first_name", url, hidden) is False
        assert store.get("form.first_name", url) is None

        flat = dict(_harvested(),
                    rect={"x": 0, "y": 0, "width": 0, "height": 0})
        assert store.record("form.first_name", url, flat) is False

        assert store.record("form.first_name", url, _harvested()) is True
        got = store.get("form.first_name", url)
        assert got and got["element_id"] == "firstName"


def test_a_baseline_is_found_again_however_the_page_was_reached():
    """Keyed by the toolkit's page_scope, so query strings do not fragment it.

    Two bots reaching the same page with different tracking parameters must
    share one baseline rather than seed two.
    """
    import tempfile
    from healing_service.baselines import BaselineStore

    with tempfile.TemporaryDirectory() as folder:
        store = BaselineStore(os.path.join(folder, "b.json"))
        store.record("form.first_name",
                     "https://demoqa.com/automation-practice-form?a=1",
                     _harvested())
        assert store.get("form.first_name",
                         "https://demoqa.com/automation-practice-form#top")
        assert store.get("form.first_name", "https://demoqa.com/other") is None


def test_baselines_outlive_the_process_that_recorded_them():
    """One good run has to seed the next one, or nothing has been learned."""
    import tempfile
    from healing_service.baselines import BaselineStore

    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "b.json")
        BaselineStore(path).record("form.first_name", "https://demoqa.com/f",
                                   _harvested())
        assert BaselineStore(path).get("form.first_name",
                                       "https://demoqa.com/f")


def test_a_corrupt_store_costs_accuracy_not_availability():
    """Fail open, like everything else. Losing baselines is not losing bots."""
    import tempfile
    from healing_service.baselines import BaselineStore

    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "b.json")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("{not json")
        store = BaselineStore(path)
        assert store.stats()["entries"] == 0
        assert store.record("k", "https://x.test/p", _harvested()) is True


def test_a_reported_baseline_reaches_the_next_adjudication():
    """The whole point, end to end over HTTP.

    A thin client posts what worked; the next repair for that locator on
    that page is scored with structure and geometry available, without the
    client having sent - or stored - anything itself.
    """
    import tempfile
    from fastapi.testclient import TestClient
    from healing_service import create_app

    with tempfile.TemporaryDirectory() as folder:
        app = create_app(_config(), warm=False,
                         baseline_path=os.path.join(folder, "b.json"))
        with TestClient(app) as client:
            url = "https://demoqa.com/automation-practice-form"
            body = {
                "locator": {"key": "form.first_name",
                            "selector": "#hr-v2-firstName",
                            "description": "the First Name box on the form"},
                "action": "interact",
                "page": {"url": url, "title": "Practice Form"},
                "candidates": [_harvested("lastName"), _harvested("firstName")],
                "use_stored_baseline": True,
                "protocol": "1",
            }

            cold = client.post("/v1/adjudicate", json=body).json()
            cold_signals = cold["shortlist"][0]["signals"]
            assert cold_signals["structure"] is None
            assert cold_signals["geometry"] is None

            posted = client.post("/v1/baseline", json={
                "key": "form.first_name", "selector": "#firstName",
                "url": url, "descriptor": _harvested(), "protocol": "1"}).json()
            assert posted["stored"] is True and posted["entries"] == 1
            assert client.get("/v1/health").json()["baselines"] == 1

            # The same request, with no baseline field - the service supplies
            # what the fleet learned.
            warm = client.post("/v1/adjudicate", json=body).json()
            warm_signals = warm["shortlist"][0]["signals"]
            assert warm_signals["structure"] is not None
            assert warm_signals["geometry"] is not None
            assert warm["confidence"] > cold["confidence"], (
                "a warm heal that scores no better has learned nothing: "
                "%.3f vs %.3f" % (warm["confidence"], cold["confidence"]))

            # An old client must be told, not silently ignored.
            assert client.post("/v1/baseline", json={
                "key": "k", "url": url, "descriptor": _harvested(),
                "protocol": "0"}).status_code == 409


def test_a_client_with_its_own_memory_is_not_handed_the_fleet_s():
    """An absent baseline from the full agent is a fact, not a gap.

    The agent keeps its own memory, so `baseline: null` from it means "my
    memory has none for this locator" - not "I have no memory". Substituting
    the fleet's would overrule a client that manages its own, and would
    silently warm the benchmark, which wipes its memory per suite precisely
    so that every row is a cold heal. A measurement that stops being true
    without a line of it changing is the worst kind.
    """
    import tempfile
    from fastapi.testclient import TestClient
    from healing_service import create_app

    with tempfile.TemporaryDirectory() as folder:
        app = create_app(_config(), warm=False,
                         baseline_path=os.path.join(folder, "b.json"))
        with TestClient(app) as client:
            url = "https://demoqa.com/automation-practice-form"
            body = {
                "locator": {"key": "form.first_name",
                            "selector": "#hr-v2-firstName",
                            "description": "the First Name box on the form"},
                "action": "interact",
                "page": {"url": url, "title": "Practice Form"},
                "candidates": [_harvested("lastName"), _harvested("firstName")],
                "protocol": "1",
            }
            client.post("/v1/baseline", json={
                "key": "form.first_name", "selector": "#firstName",
                "url": url, "descriptor": _harvested(), "protocol": "1"})

            # No use_stored_baseline: the store is full and stays out of it.
            quiet = client.post("/v1/adjudicate", json=body).json()
            assert quiet["shortlist"][0]["signals"]["structure"] is None

            asked = client.post("/v1/adjudicate",
                                json=dict(body, use_stored_baseline=True)).json()
            assert asked["shortlist"][0]["signals"]["structure"] is not None


def test_the_baseline_is_part_of_the_decision_cache_key():
    """Two requests differing only in the baseline are not the same question.

    The decision cache replays a verdict for an identical request. Leaving
    the baseline out of the key replays a cold verdict, at its cold
    confidence, over a warm heal that would have scored higher - and the
    symptom is a fix that appears not to have worked.
    """
    cold = _request()
    warm = _request()
    warm.baseline = ElementDescriptor.coerce(_harvested()).to_dict()
    adjudicator = LocalAdjudicator(_config())
    assert adjudicator._cache_key(cold) != adjudicator._cache_key(warm)


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
