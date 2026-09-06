"""Offline unit tests for the deterministic ranker.

These run without a browser, without the healing service and without
the sentence encoder, because the whole point of the heuristic layer is
that it holds up when none of those are available.

    python -m pytest tests -q       (or)      python tests/test_similarity.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bis_core.config import ScoringWeights                # noqa: E402
from bis_core.descriptors import (ElementDescriptor,      # noqa: E402
                                  LocatorSpec, collapse_digits,
                                  tokenize)
from bis_core.descriptors import (intent_tokens,           # noqa: E402
                                  normalize_selector,
                                  selector_syntax,
                                  selector_tag,
                                  selector_uses_id)
from bis_core.similarity import (HealTarget,              # noqa: E402
                                 _path_similarity,
                                 geometry_signal,
                                 identity_signal,
                                 neighbor_text_signal,
                                 parse_selector_hints,
                                 score_candidates,
                                 selector_hints)


def el(tag="div", classes=(), text="", eid="", role="", aria="",
       visible=True, interactive=False, rect=None, **kw):
    return ElementDescriptor(
        tag=tag, classes=list(classes), text=text, element_id=eid, role=role,
        aria_label=aria, visible=visible, interactive=interactive,
        rect=rect or {"x": 10, "y": 10, "width": 120, "height": 40}, **kw)


def target(selector, description="", **kw):
    return HealTarget(spec=LocatorSpec(key="t", selector=selector,
                                       description=description, **kw))


def test_digit_collapse_makes_version_bumps_identical():
    assert collapse_digits("pd19-product-card__item") == \
           collapse_digits("pd21-product-card__item")
    assert collapse_digits("nv00-gnb-v3__utility") == \
           collapse_digits("nv00-gnb-v4__utility")


def test_tokenizer_handles_bem_and_camel_case():
    assert "product" in tokenize("pd21-product-card__item")
    assert "search" in tokenize("gnbSearchKeyword")


def test_selector_hints_are_mined_from_css():
    toks, tag, text = parse_selector_hints("a.pd19-product-card__title")
    assert tag == "a"
    assert "product" in toks and "title" in toks
    assert text is None


def test_identity_signal_scores_a_version_bump_as_exact():
    t = target(".pd19-product-card__item")
    assert identity_signal(t, el(classes=["pd21-product-card__item"])) == 1.0
    assert identity_signal(t, el(classes=["cookie-bar__close"])) < 0.35


def test_identity_signal_reports_no_evidence_without_identifiers():
    assert identity_signal(target("text=Accept"), el(classes=["x"])) is None


def test_ranker_picks_the_renamed_component_over_its_siblings():
    """The regression this signal was added for.

    Every candidate here lives in the same redesigned component, so token
    overlap alone leaves them within a hair of each other; only the surviving
    BEM element name separates the right one.
    """
    t = target(".pd19-product-card__item",
               "every smartphone product card tile in the listing grid")
    cands = [
        el(classes=["pd21-product-card__options-wrap"], text="Colour : Mint"),
        el(classes=["pd21-product-card__name-wrap"], text="Galaxy F70 Pro 5G"),
        el(classes=["pd21-product-card__item", "js-pfv2-product-card"],
           text="New Galaxy F70 Pro 5G"),
        el(classes=["pd21-product-card__rating-wrap"], text="Ratings 4.9"),
    ]
    ranked = score_candidates(t, cands, ScoringWeights(), None)
    assert ranked[0].descriptor.classes[0] == "pd21-product-card__item"


def test_ranker_finds_the_renamed_nav_button():
    t = target("button.nv00-gnb-v3__utility-search",
               "the magnifier icon in the global nav that opens site search",
               tag="button", role="button")
    cands = [
        el(tag="button", classes=["nv00-gnb-v4__utility-search"], text="Search",
           interactive=True),
        el(tag="a", classes=["nv00-gnb-v4__logo"], aria="Samsung"),
        el(tag="button", classes=["cookie-bar__close"], text="close",
           interactive=True),
    ]
    ranked = score_candidates(t, cands, ScoringWeights(), None)
    assert ranked[0].descriptor.classes == ["nv00-gnb-v4__utility-search"]


def test_hidden_duplicates_lose_to_the_visible_one():
    """samsung.com renders the nav twice - mobile and desktop. Only one is
    on screen, and clicking the other would hang the process."""
    t = target("button.nv00-gnb-v3__utility-search", "site search button",
               tag="button", role="button")
    hidden = el(tag="button", classes=["nv00-gnb-v4__utility-search"],
                text="Search", visible=False, interactive=True)
    shown = el(tag="button", classes=["nv00-gnb-v4__utility-search"],
               text="Search", visible=True, interactive=True)
    ranked = score_candidates(t, [hidden, shown], ScoringWeights(), None)
    assert ranked[0].descriptor.visible is True


def test_missing_signals_are_dropped_not_scored_as_zero():
    """A cold heal has no baseline, so structure and geometry have nothing to
    say. They must be excluded from the mean rather than dragging it down."""
    t = target(".pd19-product-card__item", "a product card")
    cand = el(classes=["pd21-product-card__item"], text="Galaxy")
    ranked = score_candidates(t, [cand], ScoringWeights(), None)
    signals = ranked[0].signals
    assert signals["structure"] is None and signals["geometry"] is None
    assert signals["identity"] == 1.0
    assert ranked[0].score > 0.5


def test_expected_text_drives_the_text_signal():
    t = target(".cookie-bar__accept-button", "the accept button",
               tag="a", role="button", expected_text="^Accept$")
    accept = el(tag="a", classes=["cta", "cta--contained"], text="Accept",
                interactive=True)
    close = el(tag="button", classes=["cookie-bar__close"], text="close",
               interactive=True)
    ranked = score_candidates(t, [accept, close], ScoringWeights(), None)
    assert ranked[0].descriptor.text == "Accept"


# ---------------------------------------------------------------- XPath
# An RPA team standardised on XPath must get the same quality of repair as one
# using CSS. Before these, every XPath locator healed blind: none of the CSS
# patterns matched //div[@class='...'], so the identity signal - the
# highest-weighted one - reported "no evidence" on every single repair.


def test_bare_xpath_is_given_its_prefix():
    assert normalize_selector("//button[@id='go']") == "xpath=//button[@id='go']"
    assert normalize_selector("(//a)[2]").startswith("xpath=")
    assert normalize_selector(".card") == ".card"
    assert selector_syntax("//div") == "xpath"
    assert selector_syntax("div.card") == "css"


def test_xpath_attribute_predicates_are_mined():
    ids, toks, tag, _text = selector_hints("//div[@class='pd19-product-card__item']")
    assert tag == "div"
    assert ids == ["pd#-product-card__item"]
    assert "product" in toks and "card" in toks


def test_xpath_and_css_yield_the_same_identity_score():
    """The same intent written two ways must score identically."""
    css = target(".pd19-product-card__item")
    xp = target("//div[@class='pd19-product-card__item']")
    cand = el(classes=["pd21-product-card__item"])
    assert identity_signal(css, cand) == identity_signal(xp, cand) == 1.0


def test_xpath_contains_and_starts_with_are_mined():
    for sel in ("//a[contains(@class,'pd19-product-card__title')]",
                "//a[starts-with(@class,'pd19-product-card__title')]"):
        ids, _toks, tag, _text = selector_hints(sel)
        assert tag == "a"
        assert ids == ["pd#-product-card__title"]


def test_xpath_text_predicates_become_a_text_hint():
    for sel in ("//button[text()='Accept']",
                "//button[normalize-space()='Accept']",
                "//button[contains(text(),'Accept')]",
                "//button[.='Accept']"):
        _ids, _toks, _tag, text = selector_hints(sel)
        assert text == "Accept", sel


def test_identifiers_come_from_the_last_step_only():
    """//div[card]//a[title] selects the anchor. Scoring a candidate against
    the ancestor's class name would be measuring the wrong element."""
    sel = ("//div[@class='pd19-product-card__item']"
           "//a[@class='pd19-product-card__title']")
    ids, toks, tag, _text = selector_hints(sel)
    assert tag == "a"
    assert ids == ["pd#-product-card__title"]
    # Ancestor tokens are deliberately excluded: a candidate cannot carry its
    # ancestors' words, so including them only inflates the denominator of the
    # attribute signal and pushes good candidates below the apply threshold.
    assert "item" not in toks


def test_css_descendant_paths_target_the_last_compound():
    _ids, _toks, tag, _text = selector_hints("div.card > a.name")
    assert tag == "a"


def test_xpath_quotes_do_not_break_mining():
    _ids, _toks, tag, text = selector_hints('//button[@aria-label="Search"]')
    assert tag == "button"
    _ids2, toks2, _t, _x = selector_hints('//input[@name="search"]')
    assert "search" in toks2


def test_the_same_intent_scores_the_same_in_either_language():
    """A CSS repository and an XPath repository must rank identically.

    Regression: XPath forms used to tokenise the attribute *name* ("class")
    and to absorb ancestor tokens, both of which the CSS form never did. The
    cookie-consent locator ended up at 0.61 in XPath and 0.68 in CSS - either
    side of the auto-apply threshold, so the same repair applied in one
    language and was refused in the other.
    """
    css = target(".cookie-bar__accept-button", "the accept button",
                 tag="a", role="button", expected_text="^Accept$")
    xp = target("//div[contains(@class,'cookie-bar')]"
                "//a[contains(@class,'cookie-bar__accept-button')]",
                "the accept button", tag="a", role="button",
                expected_text="^Accept$")
    assert sorted(set(css.selector_tokens)) == sorted(set(xp.selector_tokens))
    assert css.target_identifiers() == xp.target_identifiers()

    cands = [
        el(tag="a", classes=["cta", "cta--contained", "cta--emphasis"],
           text="Accept", interactive=True),
        el(tag="button", classes=["cookie-bar__close"], text="close",
           interactive=True),
        el(classes=["cookie-bar__manage"], text="Accept"),
    ]
    a = score_candidates(css, cands, ScoringWeights(), None)
    b = score_candidates(xp, cands, ScoringWeights(), None)
    assert a[0].descriptor.text == b[0].descriptor.text == "Accept"
    assert abs(a[0].score - b[0].score) < 0.02


# ------------------------------------------------- description as evidence
# The description reaches the ranker twice - whole, for the semantic signal and
# the AI prompt, and tokenised, in the lexical bag. Only the second use needs
# filtering, and getting that wrong is expensive: the unfiltered descriptions
# were costing the top candidate ~0.13 of final score against a 0.62 threshold.


def test_english_function_words_stay_out_of_the_lexical_bag():
    toks = intent_tokens("the product title link inside every smartphone card")
    for junk in ("the", "inside", "every"):
        assert junk not in toks
    for real in ("product", "title", "link", "smartphone", "card"):
        assert real in toks


def test_filtering_recovers_score_a_prose_description_would_have_cost():
    """The defect, stated as an invariant.

    Note what this does *not* claim: filtering cannot rescue a genuinely
    verbose description, because thirteen content words really are thirteen
    entries in the denominator. That is the author's problem and the linter
    warns about it (W102). What is fixed here is that function words - which
    no DOM element could ever carry - no longer cost anything.
    """
    from bis_core.similarity import attribute_signal
    from bis_core.descriptors import tokenize

    cand = el(tag="a", classes=["pd21-product-card__name"],
              text="Galaxy F70 Pro", interactive=True)
    spec = LocatorSpec(key="k", selector="a.pd19-product-card__title",
                       description="the product title link inside every card",
                       tag="a", role="link")
    filtered = HealTarget(spec=spec)

    unfiltered = HealTarget(spec=spec)
    unfiltered.selector_tokens = (
        selector_hints(spec.selector)[1] + tokenize(spec.description))

    assert attribute_signal(filtered, cand) > attribute_signal(unfiltered, cand)


def test_the_cookie_banner_ranking_survives_the_description_change():
    """Kept because dropping description tokens outright inverts this pair.

    The component was renamed wholesale, so the *correct* element carries none
    of the target's identity tokens while the close button carries two of four
    ('cookie', 'bar'). The attribute signal is actively misleading here, which
    is why the fix filters function words rather than removing the description
    from the lexical bag altogether.
    """
    t = target(".cookie-bar__accept-button",
               "the Accept button that dismisses the cookie consent banner",
               tag="a", role="button", expected_text="^Accept$")
    accept = el(tag="a", classes=["cta", "cta--contained", "cta--emphasis"],
                text="Accept", interactive=True)
    close = el(tag="button", classes=["cookie-bar__close"], text="close",
               interactive=True)
    manage = el(classes=["cookie-bar__manage"], text="Accept")
    ranked = score_candidates(t, [accept, close, manage], ScoringWeights(), None)
    assert ranked[0].descriptor is accept


# ------------------------------------------------------- derived role / tag


def test_selector_tag_reads_the_last_step():
    assert selector_tag("div.card > a.name") == "a"
    assert selector_tag("//div[@class='c']//a[@id='n']") == "a"
    assert selector_tag(".card") is None


def test_role_is_derived_from_the_tag_when_omitted():
    cases = [("a.buy", "link"), ("input#q", "textbox"),
             ("//select[@id='x']", "dropdown"), ("img.hero", "image"),
             ("//h2[@class='t']", "text")]
    for sel, expected in cases:
        assert LocatorSpec(key="k", selector=sel).role == expected, sel
    assert LocatorSpec(key="k", selector=".x", tag="h2").role == "text"


def test_an_explicit_role_is_never_overridden():
    assert LocatorSpec(key="k", selector="a.x", role="text").role == "text"


def test_an_unmappable_tag_leaves_the_role_unset():
    """A div could be anything. Better one signal fewer than a guess."""
    assert LocatorSpec(key="k", selector=".card").role is None
    assert LocatorSpec(key="k", selector="div.card").role is None


def test_a_derived_role_scores_like_an_explicit_one():
    cand = el(tag="a", classes=["pd21-product-card__name"], text="Galaxy",
              interactive=True)
    explicit = HealTarget(spec=LocatorSpec(
        key="k", selector="a.pd19-product-card__title", role="link"))
    derived = HealTarget(spec=LocatorSpec(
        key="k", selector="a.pd19-product-card__title"))
    w = ScoringWeights()
    assert (score_candidates(explicit, [cand], w, None)[0].score
            == score_candidates(derived, [cand], w, None)[0].score)


def test_the_dropdown_role_recognises_a_select():
    """Without _ROLE_TAGS['dropdown'] the derivation would be a regression."""
    t = target("//select[@id='year']", "the year dropdown")
    assert t.spec.role == "dropdown"
    cand = el(tag="select", classes=["year"], interactive=True)
    from bis_core.similarity import tag_signal
    assert tag_signal(t, cand) > 0.8


def test_the_core_imports_nothing_outside_the_standard_library():
    """What keeps this installable on an RPA machine that has nothing.

    Every bot in the fleet installs bis-core. The moment something here
    imports torch, or numpy, or a GGUF runtime, that cost lands on every
    automation VM to no purpose - the models run on the healing service.
    The accelerator half plugs in through ``bis_core.runtime`` instead, so
    there is never a reason for a third-party import to appear here.
    """
    import ast
    import os

    core = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "bis_core")
    allowed = set(sys.stdlib_module_names) | {"bis_core"}
    offenders = []

    for name in sorted(os.listdir(core)):
        if not name.endswith(".py"):
            continue
        with open(os.path.join(core, name), encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                # level > 0 is a relative import - inside the package.
                mods = [] if node.level else [node.module or ""]
            else:
                continue
            for mod in mods:
                root = mod.split(".")[0]
                if root and root not in allowed:
                    offenders.append("%s: %s" % (name, mod))

    assert not offenders, "bis-core reached outside stdlib: %s" % offenders


# ------------------------------------------- what a baseline unlocks
#
# Six of the seven signals compare a candidate against the element the step
# used while its locator still worked. These pin the three that were either
# newly built or found to be measuring the wrong thing.


def test_neighbour_text_scores_the_copy_around_the_element():
    """The one piece of evidence a wholesale rename cannot touch.

    A design-system bump rewrites every class on the page and leaves the words
    beside the button where they were. The harvester has always collected that
    text; until this signal existed nothing scored it.
    """
    base = el(tag="input", eid="firstName",
              neighbor_text="First Name Enter your legal first name")
    t = HealTarget(spec=LocatorSpec(key="t", selector="#hr-v2-firstName"),
                   baseline=base)

    same = el(tag="input", eid="firstName",
              neighbor_text="First Name Enter your legal first name")
    assert neighbor_text_signal(t, same) == 1.0

    unrelated = el(tag="input", neighbor_text="Payment card expiry month")
    assert neighbor_text_signal(t, unrelated) < 0.3

    # Long words carry more evidence than short ones: sharing "legal" and
    # "first" says far more than sharing "your".
    long_words = el(tag="input", neighbor_text="legal first name")
    short_words = el(tag="input", neighbor_text="your")
    assert neighbor_text_signal(t, long_words) > \
        neighbor_text_signal(t, short_words)


def test_neighbour_text_says_nothing_without_a_baseline():
    """Invariant 1: no evidence reports None, never zero.

    A cold heal has no recorded surroundings to compare against. Scoring that
    as zero would punish a repair for being the first one.
    """
    cold = target("#hr-v2-firstName")
    assert neighbor_text_signal(cold, el(neighbor_text="anything")) is None

    blank = HealTarget(spec=LocatorSpec(key="t", selector="#x"),
                       baseline=el(neighbor_text=""))
    assert neighbor_text_signal(blank, el(neighbor_text="something")) is None


def test_geometry_survives_a_scroll_and_a_different_screen():
    """rect is viewport-relative, so position alone is not a fact about a page.

    The baseline is captured at the top of the page and the candidate is
    harvested after scrolling to it. Same element, same place in the document,
    900 pixels apart on screen. Before the scroll offset travelled with the
    fingerprint, this scored as though the element had moved half a page.
    """
    top = el(rect={"x": 100, "y": 200, "width": 120, "height": 40},
             scroll_x=0, scroll_y=0, viewport_w=1600, viewport_h=900)
    scrolled = el(rect={"x": 100, "y": -700, "width": 120, "height": 40},
                  scroll_x=0, scroll_y=900, viewport_w=1600, viewport_h=900)
    t = HealTarget(spec=LocatorSpec(key="t", selector="#x"), baseline=top)
    assert geometry_signal(t, scrolled) == 1.0

    # And the same element on a smaller screen is still the same element: the
    # distance is normalised by the viewport that was actually in use, not by
    # a constant that only ever described one developer's window.
    small = el(rect={"x": 100, "y": 200, "width": 120, "height": 40},
               scroll_x=0, scroll_y=0, viewport_w=1280, viewport_h=720)
    t_small = HealTarget(spec=LocatorSpec(key="t", selector="#x"),
                         baseline=small)
    assert geometry_signal(t_small, small) == 1.0

    # A fingerprint recorded before those fields existed reports zero for all
    # of them, and must degrade rather than divide by nothing.
    old = el(rect={"x": 100, "y": 200, "width": 120, "height": 40})
    t_old = HealTarget(spec=LocatorSpec(key="t", selector="#x"), baseline=old)
    assert geometry_signal(t_old, old) == 1.0


def test_paths_are_compared_by_step_not_by_character():
    """Edit distance over an absolute XPath measures mostly punctuation.

    It was also 47% of the entire ranking cost. Steps are the unit that means
    something, and matching from the leaf back up is how a redesign actually
    moves an element: the wrappers near the root get rewritten, the immediate
    surroundings usually do not.
    """
    a = "/html[1]/body[1]/div[2]/form[1]/div[3]/input[1]"
    assert _path_similarity(a, a) == 1.0

    # One wrapper renumbered near the root: most of the path still agrees.
    reshuffled = "/html[1]/body[1]/div[5]/form[1]/div[3]/input[1]"
    assert _path_similarity(a, reshuffled) > 0.6

    # A different element entirely, sharing only the document root.
    elsewhere = "/html[1]/body[1]/header[1]/nav[1]/a[4]"
    assert _path_similarity(a, elsewhere) < 0.3
    assert _path_similarity("", a) == 0.0


def test_the_identity_short_circuit_changes_speed_and_not_answers():
    """difflib's cheap bounds are exact, so skipping on them is safe.

    real_quick_ratio() >= quick_ratio() >= ratio(), so a pair whose upper
    bound is already below the best seen can never win. Getting the *pair
    order* wrong while optimising this silently changed every score, because
    ratio() is not symmetric - hence this test.
    """
    from difflib import SequenceMatcher

    def naive(t, cand):
        wanted = t.target_identifiers()
        if not wanted:
            return None
        have = [collapse_digits(cand.element_id.lower())] if cand.element_id else []
        have += [collapse_digits(c.lower()) for c in cand.classes]
        have = [h for h in have if len(h) > 2]
        if not have:
            return 0.0
        best = 0.0
        for w in wanted:
            for h in have:
                if w == h:
                    return 1.0
                best = max(best, SequenceMatcher(None, w, h).ratio())
        return best ** 3

    t = target(".pd19-product-card__item")
    for cand in (el(classes=["pd21-product-card__item"]),
                 el(classes=["pd21-product-card__title", "js-card"]),
                 el(eid="product-card-item"),
                 el(classes=["totally-unrelated-banner"]),
                 el(classes=[])):
        assert identity_signal(t, cand) == naive(t, cand), cand.classes


def test_a_repair_keeps_the_shape_the_developer_wrote():
    """An author who avoided ids has told us something about the application.

    Usually that its ids are framework-generated and do not survive a deploy.
    Handing that developer an id-based repair overrides knowledge they have
    and we do not, and it is also the code they will have to review.
    """
    assert selector_uses_id("#firstName")
    assert selector_uses_id("input#firstName")
    assert selector_uses_id("div#main .row")
    assert selector_uses_id("xpath=//input[@id='firstName']")

    # A '#' inside an attribute value is a fragment, not an id.
    assert not selector_uses_id('a[href="#top"]')
    assert not selector_uses_id("div.card > a.title")
    assert not selector_uses_id(".pd19-product-card__item")
    assert not selector_uses_id("xpath=//div[@class='card']")
    assert not selector_uses_id("")


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
    print("\n%s" % ("all tests passed" if not failures
                    else "%d test(s) failed" % failures))
    return failures


if __name__ == "__main__":
    raise SystemExit(_run_all())
