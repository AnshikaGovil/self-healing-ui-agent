"""Deterministic multi-signal similarity between a healing target and a
candidate element.

Design principle: the LLM is an *adjudicator*, never the primary ranker.  This
module alone must be able to find the right element for the common failure
modes (renamed classes, version-bumped design systems, moved nodes), so that
the system degrades gracefully to pure heuristics if the model is offline.

Eight independent signals are combined, seven of them deterministic:

    identity   - survival of the exact normalised id or component class name
    tag        - tag / ARIA-role compatibility
    attributes - id, name, class and data-* token overlap (digit-normalised)
    text       - visible and accessible text similarity
    structure  - DOM path and ancestor-chain similarity
    neighbor   - the page copy around the element, vs the baseline's
    geometry   - document position and size vs a recorded baseline
    semantic   - embedding cosine between business intent and the element

Seven of the eight draw on a *baseline* - a fingerprint of the element taken
while the locator still worked. Only ``semantic`` is independent of one, which
is why a cold heal is scored on a fraction of the total weight and why a
successful step is worth recording.

Signals that have no evidence to work from (e.g. geometry on a first run)
report ``None`` and are dropped from the weighted mean rather than scored as
zero - a missing signal must not look like a mismatch.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

from .config import ScoringWeights
from .descriptors import (CSS_TAG_RE, SYNTAX_TEXT, SYNTAX_XPATH,
                          XPATH_TAG_RE, ElementDescriptor, LocatorSpec,
                          collapse_digits, intent_tokens, normalize_text,
                          selector_syntax, split_selector_steps,
                          strip_syntax_prefix, tokenize)

# Selector fragments we can mine for identity tokens when there is no
# successful baseline to compare against - the broken selector is itself the
# developer's best description of what they were aiming at.
#
# Both selector languages are mined, because an RPA team standardised on XPath
# would otherwise heal blind: none of the CSS patterns below match
# //div[@class='pd19-product-card__item'], and the identity signal - the
# highest-weighted one - would report "no evidence" on every repair.

# --- CSS ---------------------------------------------------------------
_ID_RE = re.compile(r"#([\w\-]+)")
_CLASS_RE = re.compile(r"\.([\w\-]+)")
_ATTR_RE = re.compile(r"\[\s*([\w\-]+)\s*(?:[~^$*|]?=\s*[\"']?([^\"'\]]*)[\"']?)?\s*\]")

# --- XPath -------------------------------------------------------------
# @class='x'  |  @data-testid="y"
_XP_ATTR_RE = re.compile(r"@([\w:.\-]+)\s*=\s*[\"']([^\"']*)[\"']")
# contains(@class,'x') | starts-with(@id,'y') | ends-with(@href,'z')
_XP_FN_ATTR_RE = re.compile(
    r"(?:contains|starts-with|ends-with|matches)\s*\(\s*@([\w:.\-]+)\s*,"
    r"\s*[\"']([^\"']*)[\"']")
# text()='Accept' | .='Accept' | normalize-space()='Accept'
_XP_TEXT_EQ_RE = re.compile(
    r"(?:text\(\)|normalize-space\(\s*(?:\.|text\(\))?\s*\)|\.)\s*=\s*"
    r"[\"']([^\"']*)[\"']")
# contains(text(),'Accept') | contains(normalize-space(),'Accept') | contains(.,'x')
_XP_TEXT_FN_RE = re.compile(
    r"contains\s*\(\s*(?:text\(\)|normalize-space\(\s*\)|\.)\s*,"
    r"\s*[\"']([^\"']*)[\"']")

# --- engine-prefixed text selectors ------------------------------------
_PW_TEXT_RE = re.compile(r"(?:^text=|:has-text\(|:text\()[\"']?([^\"')]+)")


@dataclass
class HealTarget:
    """Everything the engine knows about the element it is looking for."""

    spec: LocatorSpec
    baseline: Optional[ElementDescriptor] = None
    """Fingerprint captured the last time this locator worked. May be None."""

    selector_identifiers: List[str] = field(default_factory=list)
    selector_tokens: List[str] = field(default_factory=list)
    selector_tag: Optional[str] = None
    selector_text: Optional[str] = None

    def __post_init__(self) -> None:
        identifiers, toks, tag, text = selector_hints(self.spec.selector)
        self.selector_identifiers = identifiers
        self.selector_tokens = toks
        self.selector_tag = self.spec.tag or tag
        self.selector_text = text
        # The description carries intent tokens too ("cookie accept button"),
        # but only its content words. attribute_signal scores containment as
        # matched/total, so English function words - which no DOM element can
        # carry - would inflate the denominator and drag every candidate down.
        # The semantic signal and the AI prompt still get the description whole.
        self.selector_tokens += intent_tokens(self.spec.description)

    def expected_tag(self) -> Optional[str]:
        if self.baseline and self.baseline.tag:
            return self.baseline.tag
        return self.selector_tag

    def reference_text(self) -> str:
        """Best available notion of what the element should read as.

        Deliberately returns "" when the only thing available is the free-text
        description: matching a sentence of English prose against an element
        label with an edit-distance ratio produces noise, not evidence. Intent
        is the semantic signal's job, not the text signal's.
        """
        if self.baseline and self.baseline.label_text():
            return self.baseline.label_text()
        if self.selector_text:
            return self.selector_text
        if self.spec.expected_text:
            return re.sub(r"[\^$.*+?()\[\]{}|]", " ", self.spec.expected_text).strip()
        return ""

    def target_identifiers(self) -> List[str]:
        """The normalised id / class names the step was written against.

        These are the strongest evidence available on a redesign: a front-end
        team renames `pd19-product-card__item` to `pd21-product-card__item` far
        more often than it renames the component itself. Mined from CSS and
        XPath alike, so the signal does not go dark on an XPath repository.
        """
        out = list(self.selector_identifiers)
        if self.baseline:
            if self.baseline.element_id:
                out.append(collapse_digits(self.baseline.element_id.lower()))
            for c in self.baseline.classes:
                out.append(collapse_digits(c.lower()))
        return [x for x in out if len(x) > 2]

    def target_tokens(self) -> List[str]:
        if self.baseline:
            return self.baseline.attribute_tokens() + self.selector_tokens
        return self.selector_tokens


# Attributes whose values name the element itself rather than describing it.
_IDENTITY_ATTRS = {"id", "class", "name"}


def _identity_attr(attr: str) -> bool:
    attr = attr.lower()
    return attr in _IDENTITY_ATTRS or attr.startswith("data-")


def _mine_step(step: str, syntax: str):
    """Pull (identifiers, tokens, tag, text) out of one selector step."""
    identifiers, tokens, tag, text = [], [], None, None

    def take(attr, value):
        # The *name* of an identity attribute says nothing about the element -
        # every element has a class. Tokenising it would add a "class" token
        # to XPath locators that the CSS form never contributes, and the two
        # languages would stop scoring alike. Descriptive attribute names
        # (data-omni-type, aria-label) do carry meaning, so those are kept.
        if not _identity_attr(attr):
            tokens.extend(tokenize(attr))
        tokens.extend(tokenize(value))
        if _identity_attr(attr):
            for part in str(value).split():
                if len(part) > 2:
                    identifiers.append(collapse_digits(part.lower()))

    if syntax == SYNTAX_XPATH:
        head = step.lstrip("./")
        m = XPATH_TAG_RE.match(head)
        if m and m.group(1).lower() not in {"and", "or", "not", "text"}:
            tag = m.group(1).lower()
        for attr, value in _XP_ATTR_RE.findall(step):
            take(attr, value)
        for attr, value in _XP_FN_ATTR_RE.findall(step):
            take(attr, value)
        for hit in _XP_TEXT_EQ_RE.findall(step) + _XP_TEXT_FN_RE.findall(step):
            text = normalize_text(hit) or text
    else:
        m = CSS_TAG_RE.match(step)
        if m:
            tag = m.group(1).lower()
        for hit in _ID_RE.findall(step):
            identifiers.append(collapse_digits(hit.lower()))
            tokens.extend(tokenize(hit))
        for hit in _CLASS_RE.findall(step):
            identifiers.append(collapse_digits(hit.lower()))
            tokens.extend(tokenize(hit))
        for attr, value in _ATTR_RE.findall(step):
            take(attr, value or "")
        hit = _PW_TEXT_RE.search(step)
        if hit:
            text = normalize_text(hit.group(1))
    return identifiers, tokens, tag, text


def selector_hints(selector: str):
    """Mine a selector for everything the ranker can use.

    Returns ``(identifiers, tokens, tag, text)`` where *identifiers* come from
    the final step only - they name the target element - while *tokens* are
    gathered from the whole expression, because ancestor names are still
    useful context for the attribute signal.

    Understands CSS, XPath (prefixed or bare) and Playwright text engines, so
    a repository written in either language heals with the same evidence.
    """
    if not selector:
        return [], [], None, None
    syntax = selector_syntax(selector)
    body = strip_syntax_prefix(selector)
    if syntax == SYNTAX_TEXT:
        return [], tokenize(body), None, normalize_text(body)

    steps = split_selector_steps(body, syntax) or [body]

    # Text can be pinned on an ancestor step, so it is gathered from all of
    # them. Identifiers and tokens come from the last step only.
    #
    # Ancestor tokens look like useful context but measurably are not: the
    # attribute signal scores how much of the *target's* vocabulary a
    # candidate carries, and a candidate cannot carry its ancestors' words
    # unless the naming scheme already repeats them (as BEM does, in which
    # case the last step has them anyway). Including them only inflates the
    # denominator and pushes good candidates below the auto-apply threshold.
    text = None
    for step in steps:
        _ids, _toks, _tag, txt = _mine_step(step, syntax)
        if txt:
            text = txt
    identifiers, tokens, tag, last_text = _mine_step(steps[-1], syntax)
    if last_text:
        text = last_text
    if not tokens and not identifiers:
        tokens = tokenize(body)
    return identifiers, tokens, tag, text


def parse_selector_hints(selector: str) -> Tuple[List[str], Optional[str], Optional[str]]:
    """Backwards-compatible view of :func:`selector_hints`."""
    _identifiers, tokens, tag, text = selector_hints(selector)
    return tokens, tag, text


# --------------------------------------------------------------------------
# Individual signals.  Each returns a float in [0, 1] or None ("no evidence").
# --------------------------------------------------------------------------

_ROLE_TAGS = {
    "button": {"button", "a", "input", "summary", "div", "span"},
    "link": {"a", "button"},
    "textbox": {"input", "textarea"},
    "searchbox": {"input"},
    "text": {"span", "p", "div", "h1", "h2", "h3", "h4", "a", "li", "strong", "em"},
    "image": {"img", "picture", "svg"},
    # Roles are now derived from the tag when a locator omits one, so every
    # value _ROLE_BY_TAG can produce must appear here. A missing entry would
    # score the role half of tag_signal at 0.0 and make the derivation a
    # regression rather than a convenience.
    "dropdown": {"select"},
}


def identity_signal(target: HealTarget, cand: ElementDescriptor) -> Optional[float]:
    """Did the element's own name survive the change?

    This is the highest-yield signal against real front-end releases, which
    overwhelmingly *version* component names rather than replace them:

        .pd19-product-card__item   ->  .pd21-product-card__item     exact
        .nv00-gnb-v3__utility      ->  .nv00-gnb-v4__utility        exact
        #gnb-search-keyword-input  ->  #gnb-search-keyword          0.93

    Digit runs are already collapsed on both sides, so a version bump scores a
    clean 1.0 while a genuinely different component scores on edit distance.
    Scores are sharpened (cubed) below the exact-match point so that "nearly
    the same name" does not sit as close to "the same name" as raw edit
    distance would suggest.
    """
    wanted = target.target_identifiers()
    if not wanted:
        return None
    have = [collapse_digits(cand.element_id.lower())] if cand.element_id else []
    have += [collapse_digits(c.lower()) for c in cand.classes]
    have = [h for h in have if len(h) > 2]
    if not have:
        return 0.0
    # Only the best pair matters, so a pair whose *upper bound* is already
    # below the best seen never needs its real ratio computed. difflib
    # guarantees real_quick_ratio() >= quick_ratio() >= ratio(), so this is a
    # short circuit and not an approximation - the answer is identical.
    # Worth the three lines: this signal is called once per candidate for up
    # to 900 candidates on every heal, on the machine that has no accelerator.
    # Only the best pair matters, so a pair whose *upper bound* is already
    # below the best seen never needs its real ratio computed. difflib
    # guarantees real_quick_ratio() >= quick_ratio() >= ratio(), so this is a
    # short circuit, not an approximation - the answer is identical.
    #
    # The loops are the other way round from the obvious reading on purpose.
    # ratio() is *not* symmetric, so the pair has to stay (wanted, have); but
    # the index difflib builds is over the second sequence, so iterating
    # `have` outside is what lets set_seq2 be paid once instead of once per
    # pair. Getting this backwards silently changes every score, which is how
    # it was caught.
    best = 0.0
    matcher = SequenceMatcher(None, "", "")
    for h in have:
        matcher.set_seq2(h)
        for w in wanted:
            if w == h:
                return 1.0
            matcher.set_seq1(w)
            if matcher.real_quick_ratio() <= best or                     matcher.quick_ratio() <= best:
                continue
            ratio = matcher.ratio()
            if ratio > best:
                best = ratio
    return best ** 3


def tag_signal(target: HealTarget, cand: ElementDescriptor) -> Optional[float]:
    """Tag / role compatibility, with partial credit for the same family."""
    role = target.spec.role
    expected = target.expected_tag()
    if not role and not expected:
        return None
    score = 0.0
    hits = 0
    if expected:
        hits += 1
        if cand.tag == expected:
            score += 1.0
        elif {cand.tag, expected} <= {"a", "button", "input"}:
            score += 0.65          # clickable family: a/button/input swap often
        elif {cand.tag, expected} <= {"div", "span", "p", "li"}:
            score += 0.5           # generic containers are interchangeable
        else:
            score += 0.05
    if role:
        hits += 1
        allowed = _ROLE_TAGS.get(role, set())
        if cand.role == role:
            score += 1.0
        elif cand.tag in allowed:
            score += 0.8 if cand.interactive or role == "text" else 0.5
        else:
            score += 0.0
    return score / hits if hits else None


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def attribute_signal(target: HealTarget, cand: ElementDescriptor) -> Optional[float]:
    """Token overlap over id / name / class / data-* after digit collapsing.

    Combines Jaccard (penalises bloat) with containment (rewards a candidate
    that carries *all* of the target's tokens even if it adds new ones - the
    typical shape of a redesign that appends utility classes).
    """
    t = set(target.target_tokens())
    c = set(cand.attribute_tokens())
    if not t:
        return None
    if not c:
        return 0.0
    jac = _jaccard(t, c)
    containment = len(t & c) / len(t)
    # Character-level ratio catches near-miss tokens (searchbtn vs search-btn).
    fuzzy = SequenceMatcher(None, " ".join(sorted(t)), " ".join(sorted(c))).ratio()
    return max(0.0, min(1.0, 0.30 * jac + 0.50 * containment + 0.20 * fuzzy))


def text_signal(target: HealTarget, cand: ElementDescriptor) -> Optional[float]:
    """Similarity of the human-readable label."""
    ref = normalize_text(target.reference_text()).lower()
    got = normalize_text(cand.label_text()).lower()
    if not ref:
        return None
    if not got:
        return 0.0
    if ref == got:
        return 1.0
    ratio = SequenceMatcher(None, ref, got).ratio()
    # Substring containment matters a lot for labels ("Accept" inside
    # "Accept all cookies"), and SequenceMatcher under-rates it.
    if ref in got or got in ref:
        ratio = max(ratio, 0.82)
    ref_w, got_w = set(ref.split()), set(got.split())
    return max(0.0, min(1.0, 0.7 * ratio + 0.3 * _jaccard(ref_w, got_w)))


def _path_similarity(a: str, b: str) -> float:
    """Agreement between two absolute XPaths, compared step by step.

    This was a character-level SequenceMatcher over the two path strings,
    which was 47% of the entire ranking cost - the single most expensive
    thing the bot does on a heal - and it was measuring the wrong thing:
    edit distance over "/html[1]/body[1]/div[3]" is dominated by the tags
    and the punctuation that every path on the page shares.

    Steps are the unit that means something. Matching from the leaf back up
    also matches how a redesign moves an element: the wrapper divs near the
    root are rewritten, the immediate surroundings usually are not.
    """
    if not a or not b:
        return 0.0
    pa = [s for s in a.split("/") if s]
    pb = [s for s in b.split("/") if s]
    if not pa or not pb:
        return 0.0
    common = 0.0
    for x, y in zip(reversed(pa), reversed(pb)):
        if x == y:
            common += 1.0
        elif x.split("[")[0] == y.split("[")[0]:
            common += 0.5      # same tag, different position among siblings
        else:
            break
    return max(0.0, min(1.0, common / max(len(pa), len(pb))))


def neighbor_text_signal(target: HealTarget,
                         cand: ElementDescriptor) -> Optional[float]:
    """Agreement between the copy *surrounding* the baseline and the candidate.

    The one piece of evidence a wholesale rename cannot touch. A design-system
    bump rewrites every class on the page and leaves the words next to the
    button exactly where they were, which is precisely the case this system
    exists for - and the harvester has been collecting the text for it all
    along without anything scoring it.

    Word containment weighted by word *length*, after Similo, which ranks this
    property in its top band alongside tag, name, id and visible text. Length
    weighting is what stops "the" and "a" from carrying the same evidence as
    "Delivery" - a long word shared by two elements is a far stronger claim
    that they play the same role than a short one.
    """
    base = target.baseline
    if not base:
        return None
    ref = normalize_text(base.neighbor_text).lower()
    got = normalize_text(cand.neighbor_text).lower()
    if not ref:
        return None
    if not got:
        return 0.0
    ref_words = [w for w in ref.split() if w]
    got_words = set(got.split())
    if not ref_words:
        return None
    matched = sum(len(w) for w in ref_words if w in got_words)
    total = sum(len(w) for w in ref_words)
    return max(0.0, min(1.0, matched / total)) if total else None


def structure_signal(target: HealTarget, cand: ElementDescriptor) -> Optional[float]:
    """How similar the element's position in the DOM tree is to the baseline.

    Compares the ancestor chain (tag + first stable class per level) and the
    absolute XPath.  Only meaningful when a baseline exists, so it returns
    None on a cold heal rather than pretending to know.
    """
    base = target.baseline
    if not base:
        return None
    a, b = base.ancestors, cand.ancestors
    if not a and not b:
        chain = 0.5
    else:
        # Longest common suffix of the two chains, normalised.
        common = 0
        for x, y in zip(reversed(a), reversed(b)):
            xt, yt = x.split(".")[0], y.split(".")[0]
            if x == y:
                common += 1.0
            elif xt == yt:
                common += 0.6     # same tag, restyled container
            else:
                break
        chain = common / max(len(a), len(b), 1)
    xp = _path_similarity(base.xpath, cand.xpath) if base.xpath else 0.0
    depth_pen = 1.0 - min(abs(base.depth - cand.depth) / 8.0, 1.0)
    return max(0.0, min(1.0, 0.55 * chain + 0.30 * xp + 0.15 * depth_pen))


def geometry_signal(target: HealTarget, cand: ElementDescriptor) -> Optional[float]:
    """Screen-position and size agreement with the recorded baseline.

    A redesign usually moves an element a little; it rarely teleports it to the
    other side of the viewport.  Distance is normalised by the viewport
    diagonal so the signal is resolution-independent.
    """
    base = target.baseline
    if not base or not base.rect or not cand.rect:
        return None
    bw, bh = base.rect.get("width", 0), base.rect.get("height", 0)
    cw, ch = cand.rect.get("width", 0), cand.rect.get("height", 0)
    if bw <= 0 or bh <= 0 or cw <= 0 or ch <= 0:
        return None
    # Document coordinates, not viewport coordinates. `rect` comes from
    # getBoundingClientRect, so a baseline captured at the top of a page and a
    # candidate harvested after scrolling to the footer would otherwise look
    # like the element had moved the height of the page.
    bx = base.rect.get("x", 0) + base.scroll_x + bw / 2.0
    by = base.rect.get("y", 0) + base.scroll_y + bh / 2.0
    cx = cand.rect.get("x", 0) + cand.scroll_x + cw / 2.0
    cy = cand.rect.get("y", 0) + cand.scroll_y + ch / 2.0
    # The real viewport, when the fingerprint carries one. A fixed 1600x900
    # was here for a long time under a docstring promising resolution
    # independence, which meant every bot not running at that size was scored
    # against a window it did not have. Fingerprints recorded before the
    # fields existed report zero, and fall back to the old constant rather
    # than to a guess.
    vw = max(base.viewport_w, cand.viewport_w) or 1600.0
    vh = max(base.viewport_h, cand.viewport_h) or 900.0
    diag = (vw ** 2 + vh ** 2) ** 0.5
    dist = ((bx - cx) ** 2 + (by - cy) ** 2) ** 0.5
    pos = max(0.0, 1.0 - dist / (diag * 0.55))
    size = min(bw, cw) / max(bw, cw) * min(bh, ch) / max(bh, ch)
    return max(0.0, min(1.0, 0.6 * pos + 0.4 * size))


def viability_penalty(target: HealTarget, cand: ElementDescriptor) -> float:
    """Multiplicative penalty for candidates that could not serve the step.

    Kept separate from the similarity signals so that the reason a candidate
    lost stays legible in the audit trail: it can look perfect on paper and
    still be unusable because it is hidden behind a collapsed menu.
    """
    factor = 1.0
    role = (target.spec.role or "").lower()
    interactive_role = role in {"button", "link", "textbox", "searchbox",
                                "dropdown"}
    if not cand.visible:
        factor *= 0.25 if interactive_role else 0.55
    if not cand.enabled:
        factor *= 0.5
    if interactive_role and not cand.interactive:
        factor *= 0.7
    rect = cand.rect or {}
    if interactive_role and (rect.get("width", 0) < 4 or rect.get("height", 0) < 4):
        factor *= 0.4
    return factor


@dataclass
class ScoredCandidate:
    """A candidate element plus the full breakdown of why it scored as it did."""

    index: int
    descriptor: ElementDescriptor
    signals: Dict[str, Optional[float]]
    penalty: float
    score: float
    llm_bonus: float = 0.0
    llm_reason: str = ""

    @property
    def final_score(self) -> float:
        return max(0.0, min(1.0, self.score + self.llm_bonus))

    def explain(self) -> str:
        parts = [
            "%s=%.2f" % (k, v) for k, v in self.signals.items() if v is not None
        ]
        if self.penalty < 1.0:
            parts.append("viability x%.2f" % self.penalty)
        if self.llm_bonus:
            parts.append("llm%+.2f" % self.llm_bonus)
        return ", ".join(parts)


def score_candidates(
    target: HealTarget,
    candidates: List[ElementDescriptor],
    weights: ScoringWeights,
    semantic_scores: Optional[List[float]] = None,
) -> List[ScoredCandidate]:
    """Rank every candidate against the target.

    ``semantic_scores`` is supplied by the embedding stage (or None when
    embeddings are disabled).  Absent signals are excluded from the weighted
    mean so that a cold heal is not silently punished for lacking a baseline.
    """
    out: List[ScoredCandidate] = []
    for i, cand in enumerate(candidates):
        signals: Dict[str, Optional[float]] = {
            "identity": identity_signal(target, cand),
            "tag": tag_signal(target, cand),
            "attributes": attribute_signal(target, cand),
            "text": text_signal(target, cand),
            "structure": structure_signal(target, cand),
            "geometry": geometry_signal(target, cand),
            "neighbor": neighbor_text_signal(target, cand),
            "semantic": semantic_scores[i] if semantic_scores else None,
        }
        weight_map = {
            "identity": weights.identity,
            "tag": weights.tag,
            "attributes": weights.attributes,
            "text": weights.text,
            "structure": weights.structure,
            "geometry": weights.geometry,
            "neighbor": weights.neighbor,
            "semantic": weights.semantic,
        }
        total_w = sum(weight_map[k] for k, v in signals.items() if v is not None)
        if total_w <= 0:
            base = 0.0
        else:
            base = sum(
                weight_map[k] * v for k, v in signals.items() if v is not None
            ) / total_w
        penalty = viability_penalty(target, cand)
        out.append(ScoredCandidate(
            index=i, descriptor=cand, signals=signals,
            penalty=penalty, score=base * penalty,
        ))
    out.sort(key=lambda c: c.score, reverse=True)
    return out
