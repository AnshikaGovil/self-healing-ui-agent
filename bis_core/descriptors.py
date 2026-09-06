"""Data model: what the bot knows about an element it is trying to reach.

Two objects matter:

LocatorSpec
    The *design-time* contract authored by the RPA developer - the selector to
    try, plus a natural-language statement of business intent.  Intent is what
    lets the agent heal an element it has never successfully seen before.

ElementDescriptor
    A rich, run-time fingerprint of one concrete DOM element harvested from the
    live page.  Used both as a healing *candidate* and, once a step succeeds,
    as the *baseline* stored in memory for future comparisons.
"""

from __future__ import annotations

import difflib
import logging
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

log = logging.getLogger("bis.descriptors")

# Digits are the single biggest source of false negatives when a front-end team
# ships a redesign: pd19-product-card becomes pd21-product-card,
# nv00-gnb-v3__search becomes nv00-gnb-v4__search.  Collapsing digit runs to a
# placeholder makes those pairs identical under comparison while leaving the
# rest of the token intact.
_DIGITS = re.compile(r"\d+")
_SPLIT = re.compile(r"[^a-zA-Z0-9]+")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_WS = re.compile(r"\s+")

# Tokens carrying no discriminating power; they only add noise to overlap
# scores. Kept deliberately small - over-filtering hurts more than it helps.
_STOP_TOKENS = {
    "js", "ng", "is", "el", "ui", "wrap", "wrapper", "inner", "outer",
    "container", "div", "span", "active", "show", "hide", "hidden", "clearfix",
}

# English function words, stripped from a prose *description* before it joins
# the lexical token bag - see intent_tokens() for why.
#
# Deliberately closed-class only: articles, pronouns, prepositions,
# conjunctions, auxiliaries, quantifiers. No domain nouns. Words like "button",
# "page", "item", "card", "text" and "search" appear in real class names all
# the time and must survive, or the description stops being able to name the
# thing it is describing.
_PROSE_STOP = {
    "a", "an", "the", "this", "that", "these", "those", "it", "its", "which",
    "who", "whose", "what", "there", "here",
    "of", "in", "on", "at", "to", "for", "from", "by", "with", "within",
    "into", "onto", "over", "under", "above", "below", "beside", "next",
    "near", "inside", "outside", "between", "across", "through", "up", "down",
    "and", "or", "but", "as", "if", "then", "than", "so", "because", "when",
    "while", "where", "after", "before",
    "is", "are", "was", "were", "be", "been", "being", "am",
    "has", "have", "had", "do", "does", "did", "can", "could", "will",
    "would", "shall", "should", "may", "might", "must",
    "all", "any", "each", "every", "some", "one", "both", "more", "most",
    "other", "another", "same", "only", "own", "such", "very", "not", "no",
}


def collapse_digits(token: str) -> str:
    """Map pd21 -> pd#, so version-bumped class names stay comparable."""
    return _DIGITS.sub("#", token)


def tokenize(value):
    """Split an attribute value into normalised comparison tokens.

    Handles kebab-case, snake_case, BEM (block__elem--mod) and camelCase, then
    lowercases and version-normalises each piece.
    """
    if not value:
        return []
    value = _CAMEL.sub(" ", str(value))
    out = []
    for part in _SPLIT.split(value):
        if not part:
            continue
        part = collapse_digits(part.lower())
        if part in _STOP_TOKENS or len(part) < 2:
            continue
        out.append(part)
    return out


SYNTAX_CSS = "css"
SYNTAX_XPATH = "xpath"
SYNTAX_TEXT = "text"


def selector_syntax(selector) -> str:
    """Which selector language a locator is written in.

    Recognises an explicit ``xpath=`` prefix and also bare XPath, because RPA
    teams migrating from Selenium or UiPath habitually paste expressions that
    start with ``//`` and expect them to work.
    """
    if not selector:
        return SYNTAX_CSS
    body = str(selector).strip()
    low = body.lower()
    if low.startswith("xpath="):
        return SYNTAX_XPATH
    if low.startswith("text=") or low.startswith("css="):
        return SYNTAX_TEXT if low.startswith("text=") else SYNTAX_CSS
    if body.startswith(("//", "(/", "(//", "./", "/")):
        return SYNTAX_XPATH
    return SYNTAX_CSS


def normalize_selector(selector):
    """Give bare XPath its explicit prefix.

    Everything downstream - the in-page probe, the binder, Playwright itself -
    switches on that prefix, so normalising once here keeps a bare ``//button``
    from being handed to ``querySelectorAll`` and failing as a syntax error.
    """
    if not selector:
        return selector
    body = str(selector).strip()
    if selector_syntax(body) == SYNTAX_XPATH and not body.lower().startswith("xpath="):
        return "xpath=" + body
    return body


def strip_syntax_prefix(selector) -> str:
    """The selector body, without its engine prefix."""
    body = str(selector or "").strip()
    for prefix in ("xpath=", "css=", "text="):
        if body.lower().startswith(prefix):
            return body[len(prefix):]
    return body


CSS_TAG_RE = re.compile(r"^([a-zA-Z][\w\-]*)")
XPATH_TAG_RE = re.compile(r"^(?:[\w\-]+::)?([a-zA-Z][\w\-]*)")

# Node tests that are XPath syntax rather than element names.
_XPATH_NON_TAGS = {"and", "or", "not", "text"}


def split_selector_steps(body: str, syntax: str):
    """Break a selector into its steps, respecting quotes and brackets.

    The **last** step is the element being selected; everything before it
    describes ancestors. That distinction matters: for
    ``//div[@class='card']//a[@class='title']`` the target is the anchor, and
    scoring a candidate against the card's class name would be wrong.
    """
    seps = "/" if syntax == SYNTAX_XPATH else " >+~" + chr(9) + chr(10)
    steps, buf, depth, quote = [], [], 0, ""
    for ch in body:
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = ""
            continue
        if ch in "\"'":
            quote = ch
            buf.append(ch)
        elif ch in "([":
            depth += 1
            buf.append(ch)
        elif ch in ")]":
            depth = max(0, depth - 1)
            buf.append(ch)
        elif depth == 0 and ch in seps:
            if buf:
                steps.append("".join(buf))
                buf = []
        else:
            buf.append(ch)
    if buf:
        steps.append("".join(buf))
    return [s for s in (x.strip() for x in steps) if s]


def selector_tag(selector):
    """The tag a selector targets, read from its final step.

    Returns None when the selector names no tag (``.card``, ``//*[@id='x']``).
    """
    if not selector:
        return None
    syntax = selector_syntax(selector)
    if syntax == SYNTAX_TEXT:
        return None
    steps = split_selector_steps(strip_syntax_prefix(selector), syntax)
    if not steps:
        return None
    last = steps[-1]
    if syntax == SYNTAX_XPATH:
        match = XPATH_TAG_RE.match(last.lstrip("./"))
        if match and match.group(1).lower() not in _XPATH_NON_TAGS:
            return match.group(1).lower()
        return None
    match = CSS_TAG_RE.match(last)
    return match.group(1).lower() if match else None


# What kind of thing each tag is, for locators that do not say. Only tags whose
# role is unambiguous appear here: a <div> could be anything, so it stays
# unmapped and the ranker simply scores one signal fewer rather than acting on
# a guess.
_ROLE_BY_TAG = {
    "a": "link",
    "button": "button",
    "input": "textbox",
    "textarea": "textbox",
    "select": "dropdown",
    "img": "image",
    "h1": "text", "h2": "text", "h3": "text",
    "h4": "text", "h5": "text", "h6": "text",
    "p": "text", "span": "text", "td": "text", "li": "text",
}


def intent_tokens(description):
    """Tokens from a prose description, minus English function words.

    The description reaches the ranker twice: whole, as the query for the
    semantic signal and the AI prompt, and tokenised, as extra entries in the
    lexical token bag. Only the second use needs filtering, because
    ``attribute_signal`` scores containment as ``matched / total`` - every
    token in the target that no DOM element could possibly carry inflates the
    denominator and drags *every* candidate down.

    Measured on the shipped Samsung repository, the unfiltered descriptions
    were costing the top-ranked candidate around 0.13 of final score - enough,
    at an auto-apply threshold of 0.62, to turn an applied repair into an
    escalation.

    Deliberately not folded into ``_STOP_TOKENS``: that set is applied by
    ``tokenize`` to attribute values on *both* sides of the comparison, so
    adding "the" or "of" there would quietly change candidate fingerprints and
    selector mining too.
    """
    return [t for t in tokenize(description) if t not in _PROSE_STOP]


# Characters a repository key never contains. Keys are dotted identifiers
# (plp.product_card); selectors carry syntax.
_SELECTOR_CHARS = set("[]()#>+~=" + chr(34) + chr(39) + "@ " + chr(9))


def looks_like_selector(value) -> bool:
    """Is this string a selector written inline, rather than a locator key?

    Recognises XPath outright, which covers the common case for the many RPA
    teams that write XPath for everything. A bare CSS selector such as
    ``a.download-pdf`` is genuinely indistinguishable from a key like
    ``plp.heading``, so it has to be written as ``css=a.download-pdf`` to be
    used inline - guessing there would turn a typo in a key into a silent,
    very confusing heal.
    """
    if not value:
        return False
    body = str(value).strip()
    if body.lower().startswith(("xpath=", "css=", "text=")):
        return True
    if selector_syntax(body) == SYNTAX_XPATH:
        return True
    return any(ch in _SELECTOR_CHARS for ch in body)


def normalize_text(value, limit: int = 160) -> str:
    """Collapse whitespace and trim - used for every text comparison."""
    if not value:
        return ""
    return _WS.sub(" ", str(value)).strip()[:limit]


def page_scope(url) -> str:
    """Normalise a URL into a stable key for anything remembered per page.

    Query strings and fragments are dropped: the same page reached with
    different tracking parameters must not fragment what has been learned
    about it.

    Lives here rather than beside the memory file because the healing
    service keeps baselines for thin clients and has to derive exactly the
    same key the toolkit derives, or a bot would look up its own baseline
    under a name nothing ever wrote.
    """
    try:
        parsed = urlparse(str(url or ""))
        path = (parsed.path or "/").rstrip("/") or "/"
        return "%s%s" % (parsed.netloc, path)
    except Exception:
        return str(url or "") or "unknown"


KNOWN_LOCATOR_KEYS = frozenset({
    "selector", "description", "tag", "role", "expected_text",
    "scope", "fallbacks", "critical", "multiple",
})

# Keys that used to live on a locator and have moved. Worth naming explicitly:
# "unknown key" is unhelpful when the key was documented last release.
_MOVED_LOCATOR_KEYS = {
    "stale": "the repository-level 'stale' list",
}


def _warn_unknown_keys(key: str, raw: Dict[str, Any]) -> None:
    """Say something when a locator carries a key the engine will ignore.

    Warn rather than raise: a typo in a field that was going to be optional
    anyway must not take a running process down. But it must not be silent
    either - a misspelled 'description' costs real healing quality, forever,
    with nothing in the logs to say so.
    """
    for name in raw:
        if name in KNOWN_LOCATOR_KEYS:
            continue
        if name in _MOVED_LOCATOR_KEYS:
            log.warning("locator '%s': '%s' has moved to %s and is ignored "
                        "here", key, name, _MOVED_LOCATOR_KEYS[name])
            continue
        near = difflib.get_close_matches(name, KNOWN_LOCATOR_KEYS, 1, 0.7)
        log.warning("locator '%s': unknown key '%s' is ignored%s", key, name,
                    (" - did you mean '%s'?" % near[0]) if near else "")


@dataclass
class LocatorSpec:
    """A design-time locator plus the business intent behind it.

    ``description`` is not documentation - it is a first-class input to the
    healing engine.  It is what allows a repair on the very first run, before
    any successful baseline has ever been recorded.
    """

    key: str                                   # stable id, e.g. plp.product_name
    selector: str                              # CSS, xpath=..., or text=...
    description: str = ""                      # plain-English business intent
    tag: Optional[str] = None                  # expected tag, if known
    role: Optional[str] = None                 # button | link | textbox | text
    expected_text: Optional[str] = None        # regex a repair must satisfy
    scope: Optional[str] = None                # CSS container to search within
    fallbacks: List[str] = field(default_factory=list)   # tried before any AI
    critical: bool = True                      # False => step may be skipped
    multiple: bool = False
    """True when the step reads a *set* of elements (every product name on the
    page) rather than one. It changes what a good repair looks like: a
    collection locator must stay generic, so a selector pinned to one instance
    is a failed repair even though it resolves."""

    def __post_init__(self) -> None:
        # Accept bare XPath as authored and give it its prefix once, here,
        # rather than making every consumer sniff for it.
        self.selector = normalize_selector(self.selector)
        self.fallbacks = [normalize_selector(f) for f in self.fallbacks]
        # Derive the role rather than asking an author to type what the tag
        # already implies. This is the single seam both sides pass through -
        # LocatorRepository.load on the bot, spec_from_payload on the service -
        # so the two agree without the wire payload growing a field.
        if not self.role:
            self.role = _ROLE_BY_TAG.get(self.tag or selector_tag(self.selector) or "")

    @property
    def syntax(self) -> str:
        return selector_syntax(self.selector)

    def intent_sentence(self) -> str:
        """Render the spec as one sentence for the embedding / LLM stages."""
        bits = [self.description or self.key.replace(".", " ").replace("_", " ")]
        if self.role:
            bits.append("role: " + self.role)
        if self.tag:
            bits.append("tag: " + self.tag)
        if self.expected_text:
            bits.append("text matching: " + self.expected_text)
        return "; ".join(bits)

    @classmethod
    def from_dict(cls, key: str, raw: Dict[str, Any]) -> "LocatorSpec":
        _warn_unknown_keys(key, raw)
        if not raw.get("selector"):
            raise KeyError(
                "locator '%s' has no 'selector' - it is the only required "
                "field" % key)
        return cls(
            key=key,
            selector=raw["selector"],
            description=raw.get("description", ""),
            tag=raw.get("tag"),
            role=raw.get("role"),
            expected_text=raw.get("expected_text"),
            scope=raw.get("scope"),
            fallbacks=list(raw.get("fallbacks", [])),
            critical=bool(raw.get("critical", True)),
            multiple=bool(raw.get("multiple", False)),
        )


# Keys the browser-side JavaScript emits that are not dataclass field names.
# Their presence is what tells the two wire shapes apart - see
# ElementDescriptor.coerce().
_JS_ONLY_KEYS = ("ariaLabel", "dataAttrs", "accName", "cssPath", "sibIndex",
                 "neighborText")


@dataclass
class ElementDescriptor:
    """Run-time fingerprint of a single DOM element."""

    tag: str = ""
    element_id: str = ""
    classes: List[str] = field(default_factory=list)
    name: str = ""
    input_type: str = ""
    role: str = ""
    aria_label: str = ""
    placeholder: str = ""
    title: str = ""
    alt: str = ""
    href: str = ""
    value: str = ""
    data_attrs: Dict[str, str] = field(default_factory=dict)
    text: str = ""
    accessible_name: str = ""
    xpath: str = ""
    css_path: str = ""
    ancestors: List[str] = field(default_factory=list)
    depth: int = 0
    sibling_index: int = 0
    rect: Dict[str, float] = field(default_factory=dict)
    visible: bool = True
    enabled: bool = True
    interactive: bool = False
    neighbor_text: str = ""
    selectors: List[str] = field(default_factory=list)
    """Ready-made selector proposals generated in-page, most durable
    first. Because they are derived from a real element, the healer can
    only ever *choose* a selector, never invent one."""

    def attribute_tokens(self) -> List[str]:
        """Every identity-bearing token this element exposes."""
        toks = []
        toks += tokenize(self.element_id)
        for c in self.classes:
            toks += tokenize(c)
        toks += tokenize(self.name)
        toks += tokenize(self.input_type)
        toks += tokenize(self.role)
        for k, v in self.data_attrs.items():
            toks += tokenize(k)
            toks += tokenize(v)
        return toks

    def label_text(self) -> str:
        """Best available human-facing label for this element."""
        for candidate in (
            self.accessible_name, self.aria_label, self.text,
            self.placeholder, self.title, self.alt, self.value,
        ):
            candidate = normalize_text(candidate)
            if candidate:
                return candidate
        return ""

    def short_path(self, depth: int = 4) -> str:
        """Tail of the ancestor chain - enough context without the noise."""
        return "/".join(self.ancestors[-depth:] + [self.tag])

    def semantic_sentence(self) -> str:
        """Natural-language rendering, fed to the embedder and the LLM.

        Deliberately reads like a sentence a person would write, because that
        is the distribution the sentence encoder was trained on.
        """
        kind = self.role or {
            "a": "link", "button": "button", "input": "input field",
            "select": "dropdown", "textarea": "text area", "img": "image",
        }.get(self.tag, self.tag or "element")
        label = self.label_text()
        parts = [kind]
        if label:
            parts.append('labelled "' + label + '"')
        if self.element_id:
            parts.append("with id " + self.element_id)
        if self.classes:
            parts.append("styled " + " ".join(self.classes[:4]))
        if self.neighbor_text:
            parts.append("near " + normalize_text(self.neighbor_text, 60))
        return " ".join(parts)

    def compact(self, index: int) -> str:
        """One-line rendering for the LLM shortlist (token budget matters)."""
        attrs = []
        if self.element_id:
            attrs.append('id="%s"' % self.element_id)
        if self.classes:
            attrs.append('class="%s"' % " ".join(self.classes[:5]))
        if self.name:
            attrs.append('name="%s"' % self.name)
        if self.input_type:
            attrs.append('type="%s"' % self.input_type)
        if self.role:
            attrs.append('role="%s"' % self.role)
        if self.aria_label:
            attrs.append('aria-label="%s"' % self.aria_label)
        if self.placeholder:
            attrs.append('placeholder="%s"' % self.placeholder)
        for k, v in list(self.data_attrs.items())[:2]:
            attrs.append('%s="%s"' % (k, str(v)[:24]))
        text = normalize_text(self.text, 70)
        rect = self.rect or {}
        geom = "at(%d,%d) %dx%d" % (
            int(rect.get("x", 0)), int(rect.get("y", 0)),
            int(rect.get("width", 0)), int(rect.get("height", 0)),
        )
        txt = (" text=" + repr(text)) if text else ""
        return "[%d] <%s %s>%s %s path=%s" % (
            index, self.tag, " ".join(attrs), txt, geom, self.short_path(),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_js(cls, raw: Dict[str, Any]) -> "ElementDescriptor":
        """Build from the payload returned by js_harvest.HARVEST_JS."""
        return cls(
            tag=(raw.get("tag") or "").lower(),
            element_id=raw.get("id") or "",
            classes=[c for c in (raw.get("classes") or []) if c],
            name=raw.get("name") or "",
            input_type=raw.get("type") or "",
            role=raw.get("role") or "",
            aria_label=raw.get("ariaLabel") or "",
            placeholder=raw.get("placeholder") or "",
            title=raw.get("title") or "",
            alt=raw.get("alt") or "",
            href=raw.get("href") or "",
            value=raw.get("value") or "",
            data_attrs=raw.get("dataAttrs") or {},
            text=normalize_text(raw.get("text")),
            accessible_name=normalize_text(raw.get("accName")),
            xpath=raw.get("xpath") or "",
            css_path=raw.get("cssPath") or "",
            ancestors=raw.get("ancestors") or [],
            depth=int(raw.get("depth") or 0),
            sibling_index=int(raw.get("sibIndex") or 0),
            rect=raw.get("rect") or {},
            visible=bool(raw.get("visible")),
            enabled=bool(raw.get("enabled", True)),
            interactive=bool(raw.get("interactive")),
            neighbor_text=normalize_text(raw.get("neighborText"), 80),
            selectors=list(raw.get("selectors") or []),
        )

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "ElementDescriptor":
        allowed = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in raw.items() if k in allowed})

    @classmethod
    def coerce(cls, raw: Dict[str, Any]) -> "ElementDescriptor":
        """Build from either wire shape - harvester output or a stored dict.

        The full agent runs from_js() in the bot and sends the *serialised
        descriptor*; a thin client has no descriptor class and sends the
        harvester's own output, which names the same facts differently
        (``id`` for element_id, ``dataAttrs`` for data_attrs, and so on).

        Anything decoding a payload that could have come from either must
        use this. from_dict() silently drops every key it does not
        recognise, so a harvester payload decoded with it arrives with no
        id, no type, no aria-label, no data-* and no accessible name - the
        exact fields the identity and attribute signals are computed from.
        That is not a degraded score, it is a wrong one, and it is invisible:
        the request succeeds and the ranking is quietly built on a third of
        the evidence.
        """
        if not raw:
            return cls()
        return (cls.from_js(raw) if any(k in raw for k in _JS_ONLY_KEYS)
                else cls.from_dict(raw))
