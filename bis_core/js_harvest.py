"""Browser-side DOM harvesting.

The healing engine never guesses selectors in the abstract - it looks at what
is *actually on the page right now*.  This module holds the JavaScript that
runs inside the page and returns a structured fingerprint for every plausible
element, together with a ranked list of ready-made selectors for each one.

Generating the selectors in the browser (rather than asking an LLM to invent a
string) is a deliberate design choice: every selector we later apply provably
came from a real element, so the model can only *choose*, never *hallucinate*.
"""


# One description of an element, spliced into both entry points below.
#
# It used to be implemented twice - here for candidates, and again inside
# DESCRIBE_ONE_JS for baselines - and the two drifted. Measured over 738
# interactive elements on four live sites, they disagreed about `visible` 40%
# of the time and about `neighborText` 46%, and the baseline copy derived no
# accessible name from a <label>, which on a login form is every field.
#
# Worse than the disagreement was where it landed. Both the service store and
# the agent refuse a baseline that is not `visible`, and the baseline copy
# defined that as "has a box" - which is what the other half of the same guard
# already checked. The visibility test invariant 12 describes was not running
# on that path at all.
#
# A candidate and a baseline are compared field against field. They have to be
# produced by the same code, so there is one copy of it.
_DESCRIBE_JS = r"""
  const INTERACTIVE = new Set(['A','BUTTON','INPUT','SELECT','TEXTAREA',
                               'LABEL','SUMMARY','OPTION']);

  const clean = (s) => (s || '').toString().replace(/\s+/g, ' ').trim();

  // --- CSS.escape is not universally available on older engines ------------
  const esc = (s) => (window.CSS && CSS.escape)
      ? CSS.escape(s) : String(s).replace(/([^\w-])/g, '\$1');

  // A class is "stable enough" to build a selector from if it is not one of
  // the framework state classes that flip on every interaction.
  const VOLATILE = /^(is-|has-|ng-|v-|active$|open$|show$|hide$|hidden$|selected$|focus|hover|animate|swiper-|slick-)/i;
  const stableClasses = (el) => Array.from(el.classList || [])
      .filter(c => c && !VOLATILE.test(c) && !/^[a-z]{1,2}\d{3,}$/i.test(c));

  const absXPath = (el) => {
    const parts = [];
    let node = el;
    while (node && node.nodeType === 1 && parts.length < 14) {
      let i = 1, sib = node.previousElementSibling;
      while (sib) { if (sib.tagName === node.tagName) i++; sib = sib.previousElementSibling; }
      parts.unshift(node.tagName.toLowerCase() + '[' + i + ']');
      node = node.parentElement;
    }
    return '/' + parts.join('/');
  };

  const cssPath = (el) => {
    const parts = [];
    let node = el;
    while (node && node.nodeType === 1 && parts.length < 7) {
      if (node.id) { parts.unshift('#' + esc(node.id)); break; }
      let i = 1, sib = node.previousElementSibling;
      while (sib) { if (sib.tagName === node.tagName) i++; sib = sib.previousElementSibling; }
      parts.unshift(node.tagName.toLowerCase() + ':nth-of-type(' + i + ')');
      node = node.parentElement;
    }
    return parts.join(' > ');
  };

  const accName = (el) => {
    if (el.getAttribute('aria-label')) return el.getAttribute('aria-label');
    const lb = el.getAttribute('aria-labelledby');
    if (lb) {
      const t = lb.split(/\s+/).map(id => {
        const n = document.getElementById(id); return n ? n.innerText : '';
      }).join(' ');
      if (clean(t)) return clean(t);
    }
    if (el.tagName === 'INPUT' && el.labels && el.labels.length)
      return clean(el.labels[0].innerText);
    return clean(el.getAttribute('alt') || el.getAttribute('title')
                 || el.getAttribute('placeholder') || '');
  };

  // Text that surrounds the element - a strong disambiguator when class names
  // have been rewritten wholesale but the page copy has not.
  const neighborText = (el) => {
    // Climb until the surroundings actually say something. Stopping at the
    // immediate parent leaves this empty for 77% of the interactive elements
    // on real pages - a form control usually sits alone in a layout div and
    // its label is a level or two up. Three hops takes that to 25%.
    //
    // Climbing further is deliberately not done. The higher you go the more
    // generic the text becomes, until every control in a section shares it,
    // and text shared by fifty candidates discriminates between none of them.
    const own = clean(el.innerText);
    let p = el.parentElement, hops = 0;
    while (p && hops++ < 3) {
      let t = clean(p.innerText);
      if (own && t.startsWith(own)) t = t.slice(own.length);
      t = clean(t);
      if (t.length >= 2) return t.slice(0, 90);
      p = p.parentElement;
    }
    return '';
  };

  const describe = (el) => {
    const r = el.getBoundingClientRect();
    const cs = window.getComputedStyle(el);
    const visible = r.width > 0 && r.height > 0 &&
                    cs.visibility !== 'hidden' && cs.display !== 'none' &&
                    parseFloat(cs.opacity || '1') > 0.05;
    const data = {};
    for (const a of el.attributes || []) {
      if (a.name.startsWith('data-') && a.value && a.value.length < 60)
        data[a.name] = a.value;
    }
    const ancestors = [];
    let p = el.parentElement, guard = 0;
    while (p && guard++ < 8) {
      const c = stableClasses(p)[0];
      ancestors.unshift(p.tagName.toLowerCase() + (c ? '.' + c : ''));
      p = p.parentElement;
    }
    let idx = 1, sib = el.previousElementSibling;
    while (sib) { if (sib.tagName === el.tagName) idx++; sib = sib.previousElementSibling; }
    return {
      tag: el.tagName.toLowerCase(),
      id: el.id || '',
      classes: Array.from(el.classList || []),
      name: el.getAttribute('name') || '',
      type: el.getAttribute('type') || '',
      role: el.getAttribute('role') || '',
      ariaLabel: el.getAttribute('aria-label') || '',
      placeholder: el.getAttribute('placeholder') || '',
      title: el.getAttribute('title') || '',
      alt: el.getAttribute('alt') || '',
      href: (el.getAttribute('href') || '').slice(0, 120),
      value: (el.value || '').toString().slice(0, 60),
      dataAttrs: data,
      text: clean(el.innerText).slice(0, 160),
      accName: accName(el),
      xpath: absXPath(el),
      cssPath: cssPath(el),
      ancestors: ancestors,
      depth: ancestors.length,
      sibIndex: idx,
      rect: {x: r.x, y: r.y, width: r.width, height: r.height},
      // getBoundingClientRect is viewport-relative, so a rect alone cannot be
      // compared across a scroll or across two screen sizes. The scroll offset
      // makes the position absolute; the viewport makes the distance between
      // two positions mean the same thing on a laptop and on a 4K panel.
      scrollX: Math.round(window.scrollX || window.pageXOffset || 0),
      scrollY: Math.round(window.scrollY || window.pageYOffset || 0),
      viewportW: window.innerWidth || 0,
      viewportH: window.innerHeight || 0,
      visible: visible,
      enabled: !el.disabled && el.getAttribute('aria-disabled') !== 'true',
      // reuse the style object captured above - a second getComputedStyle
      // call here forces a fresh style recalc on every harvested element,
      // which on a page this size costs more than the rest of the harvest
      interactive: INTERACTIVE.has(el.tagName) || !!el.onclick ||
                   ['button','link','textbox','searchbox','tab','menuitem','checkbox']
                     .indexOf(el.getAttribute('role')) >= 0 ||
                   cs.cursor === 'pointer',
      neighborText: neighborText(el)
    };
  };
"""


HARVEST_JS = r"""
(opts) => {
  const MAX = opts.maxCandidates || 900;
  const SKIP = new Set(['SCRIPT','STYLE','LINK','META','HEAD','NOSCRIPT',
                        'TEMPLATE','BR','HR','PATH','G','DEFS','SVG','USE',
                        'SOURCE','TITLE','CIRCLE','RECT','POLYGON','LINE']);
  const root = opts.scope ? document.querySelector(opts.scope) : document.body;
  if (!root) return {error: 'scope-not-found', candidates: []};
""" + _DESCRIBE_JS + r"""
  // Ranked selector proposals, most durable first.  The healer validates each
  // against the live DOM and keeps the first that resolves to this element.
  //
  // Each durability tier is offered in both selector languages, with the
  // language the step was *authored* in first. A team standardised on XPath
  // gets XPath repairs back, which is what they will review and maintain -
  // but never at the cost of durability, because tier always wins over
  // language.
  const xpLit = (s) => {
    if (s.indexOf("'") < 0) return "'" + s + "'";
    if (s.indexOf('"') < 0) return '"' + s + '"';
    return null;                       // both quote styles present: skip it
  };

  const proposeSelectors = (el, prefer) => {
    const tiers = [];
    const tag = el.tagName.toLowerCase();
    const cls = stableClasses(el);
    const pair = (css, xp) => tiers.push({css: css, xp: xp});

    if (el.id) {
      const lit = xpLit(el.id);
      pair('#' + esc(el.id), lit ? '//' + tag + '[@id=' + lit + ']' : null);
    }
    for (const a of ['data-testid','data-test','data-qa','data-automation-id','data-omni-type']) {
      const v = el.getAttribute(a);
      if (!v) continue;
      const lit = xpLit(v);
      pair(tag + '[' + a + '="' + v.replace(/"/g, '\\"') + '"]',
           lit ? '//' + tag + '[@' + a + '=' + lit + ']' : null);
    }
    // Component classes before aria-label / name on purpose: an aria-label
    // often carries an *instance* value ("Galaxy Z Flip8"), which would pin a
    // collection locator to a single row; a component class describes the
    // role and stays generic.
    // An XPath class predicate has to AND one contains() per class to be as
    // specific as the CSS form. Offering only the first class would make the
    // XPath side of this tier much looser, and a single-element locator would
    // fail validation and drop to a weaker tier.
    const xpClasses = (t, list) => {
      const parts = [];
      for (const c of list) {
        const lit = xpLit(c);
        if (!lit) return null;
        parts.push('contains(@class,' + lit + ')');
      }
      return parts.length ? '//' + t + '[' + parts.join(' and ') + ']' : null;
    };
    if (cls.length) {
      pair(tag + '.' + cls.slice(0, 3).map(esc).join('.'),
           xpClasses(tag, cls.slice(0, 3)));
      pair(tag + '.' + esc(cls[0]), xpClasses(tag, cls.slice(0, 1)));
      pair('.' + cls.slice(0, 2).map(esc).join('.'),
           xpClasses('*', cls.slice(0, 2)));
    }
    for (const a of ['aria-label', 'name', 'placeholder']) {
      const v = el.getAttribute(a);
      if (!v) continue;
      const lit = xpLit(v);
      pair(tag + '[' + a + '="' + v.replace(/"/g, '\\"') + '"]',
           lit ? '//' + tag + '[@' + a + '=' + lit + ']' : null);
    }
    // Text is a first-class locator in XPath and awkward in CSS, so this tier
    // is XPath-only. Two guards, both learned the hard way:
    //   - keep it short; long copy is edited far more often than a label;
    //   - only bind to text that reads like a *label*, never a *value*. A
    //     price cell healed to //div[normalize-space()='Rs 65,150'] resolves
    //     perfectly today and breaks the next time the price moves, which for
    //     a price monitor is the one thing guaranteed to happen.
    // Rejecting every label with a digit in it was too blunt. It threw away
    // "Add to cart (3)" and "Step 2 of 4" along with "Rs 65,150", so any
    // control carrying a count or a badge had no text tier at all. Strip the
    // number and keep the words instead - after Playwright, which offers the
    // stripped form and prefers it. A label that is *only* a value still has
    // nothing left once the digits go, so it drops out on its own.
    const stripNumbers = (t) => clean(t
        .replace(/^[\s\u20b9$\u20ac\u00a3\u00a5]*[\d.,]+\s*/, '')
        .replace(/\s*[\u20b9$\u20ac\u00a3\u00a5]?[\d.,]+\s*$/, ''));
    const label = clean(el.innerText);
    for (const t of [label, stripNumbers(label)]) {
      // Still no binding to a value: a price cell healed to
      // //div[normalize-space()='Rs 65,150'] resolves today and breaks the
      // next time the price moves, which for a price monitor is guaranteed.
      if (!t || t.length > 40 || /[\d\u20b9$\u20ac\u00a3\u00a5]/.test(t)) continue;
      const lit = xpLit(t);
      if (lit) {
        pair(null, '//' + tag + '[normalize-space()=' + lit + ']');
        break;
      }
    }
    pair(cssPath(el), absXPath(el));

    const asXp = (v) => v ? 'xpath=' + v : null;
    const out = [];
    for (const t of tiers) {
      const ordered = prefer === 'xpath'
          ? [asXp(t.xp), t.css]
          : [t.css, asXp(t.xp)];
      for (const s of ordered) if (s) out.push(s);
    }
    return out.filter((v, i, arr) => v && arr.indexOf(v) === i);
  };

  // ---- collect ------------------------------------------------------------
  // Cheap prefilter first: anything the target could plausibly be.  The
  // expensive similarity maths happens in Python on the survivors only.
  const wantTag = (opts.tagHint || '').toUpperCase();
  const needText = !!opts.needText;
  const all = root.querySelectorAll('*');
  const pool = [];
  for (const el of all) {
    if (SKIP.has(el.tagName)) continue;
    if (el.childElementCount > 40) continue;          // layout shells
    const r = el.getBoundingClientRect();
    const isInteractive = INTERACTIVE.has(el.tagName) ||
                          !!el.getAttribute('role') || !!el.onclick;
    const hasText = clean(el.innerText).length > 0;
    const onScreen = r.width > 0 && r.height > 0;
    if (!onScreen && !isInteractive) continue;
    if (needText && !hasText && !isInteractive) continue;
    // priority drives which elements survive the MAX cut
    let pri = 0;
    if (onScreen) pri += 4;
    if (isInteractive) pri += 3;
    if (hasText) pri += 1;
    if (wantTag && el.tagName === wantTag) pri += 3;
    if (el.id) pri += 1;
    pool.push([pri, el]);
  }
  pool.sort((a, b) => b[0] - a[0]);

  const candidates = [];
  for (const [, el] of pool.slice(0, MAX)) {
    try {
      const d = describe(el);
      d.selectors = proposeSelectors(el, opts.preferSyntax || 'css');
      candidates.push(d);
    } catch (e) { /* one bad node must never abort the harvest */ }
  }
  return {
    url: location.href,
    title: document.title,
    scanned: all.length,
    pooled: pool.length,
    candidates: candidates
  };
}
"""


PROBE_JS = r"""
(selector) => {
  // Resolve a selector the way the healer intends to use it and report back
  // enough for the validation gate to make a decision.
  let nodes = [];
  try {
    if (selector.startsWith('xpath=')) {
      const it = document.evaluate(selector.slice(6), document, null,
                                   XPathResult.ORDERED_NODE_SNAPSHOT_TYPE, null);
      for (let i = 0; i < it.snapshotLength; i++) nodes.push(it.snapshotItem(i));
    } else {
      nodes = Array.from(document.querySelectorAll(selector));
    }
  } catch (e) {
    return {ok: false, error: 'invalid-selector: ' + e.message, count: 0};
  }
  if (!nodes.length) return {ok: false, error: 'no-match', count: 0};
  const visibleNodes = nodes.filter(n => {
    const r = n.getBoundingClientRect();
    const cs = window.getComputedStyle(n);
    return r.width > 0 && r.height > 0 && cs.visibility !== 'hidden' &&
           cs.display !== 'none' && parseFloat(cs.opacity || '1') > 0.05;
  });
  const el = visibleNodes[0] || nodes[0];
  const r = el.getBoundingClientRect();
  return {
    ok: true,
    count: nodes.length,
    visibleCount: visibleNodes.length,
    firstVisibleIndex: visibleNodes.length ? nodes.indexOf(visibleNodes[0]) : -1,
    tag: el.tagName.toLowerCase(),
    text: (el.innerText || '').replace(/\s+/g, ' ').trim().slice(0, 200),
    enabled: !el.disabled,
    rect: {x: r.x, y: r.y, width: r.width, height: r.height}
  };
}
"""


# The baseline fingerprinter: resolve one element, then hand it to the same
# describe() every candidate goes through.
DESCRIBE_ONE_JS = r"""
(opts) => {
  const selector = opts.selector;
  const wantIndex = opts.index || 0;
  // Compact fingerprint of a single element, captured right after a step
  // succeeds.  This becomes the *baseline* in healing memory: the structural
  // and geometric evidence that turns a future cold heal into a warm one.
  let el = null;
  let nodes = [];
  try {
    if (selector.startsWith('xpath=')) {
      const it = document.evaluate(selector.slice(6), document, null,
                                   XPathResult.ORDERED_NODE_SNAPSHOT_TYPE, null);
      for (let i = 0; i < it.snapshotLength; i++) nodes.push(it.snapshotItem(i));
    } else {
      nodes = Array.from(document.querySelectorAll(selector));
    }
    // Take the node the step actually used. Re-resolving to "the first
    // visible match" can fingerprint a different element entirely - a
    // selector that matches an off-screen duplicate would store that
    // duplicate as the baseline and poison the structure and geometry
    // signals on every later heal.
    el = nodes[wantIndex] || nodes.find(n => {
      const b = n.getBoundingClientRect();
      return b.width > 0 && b.height > 0;
    }) || nodes[0];
  } catch (e) { return null; }
  if (!el) return null;
""" + _DESCRIBE_JS + r"""
  return describe(el);
}
"""


PRESENCE_JS = r"""
(opts) => {
  // Is anything of this element's component family on the page at all?
  //
  // This separates the two cases a healer cannot otherwise tell apart: a
  // selector that broke because the site renamed something, and a selector
  // that failed because the element simply is not here this session. They
  // look identical - both match zero nodes - but they want opposite
  // responses. A renamed element should be repaired; an absent optional one
  // should be skipped without spending a harvest and an adjudication on it.
  //
  // The test is deliberately loose. Identifiers are split on digit runs and
  // the longest surviving fragment is matched as a substring, so
  // 'pd19-product-card__item' still recognises 'pd21-product-card__item' as
  // family - the version bump is exactly the redesign this system exists to
  // survive. It is a pre-filter, not a score: nothing here ranks anything.
  //
  // When in doubt it answers true. A false 'present' costs one wasted heal;
  // a false 'absent' silently stops repairing a step forever, which is much
  // worse and much harder to notice.
  const sel = (opts && opts.selector) || '';
  if (!sel) return {present: true, reason: 'no-selector', fragments: []};

  const MIN = 4;                       // shorter than this matches everything
  const GENERIC = new Set(['button','input','link','item','text','icon','list',
                           'title','name','value','label','wrap','wrapper',
                           'container','content','active','hidden','open',
                           'true','false','main','body','page','row','col']);

  // --- pull identifiers out of either selector language -------------------
  const ids = [];
  const push = (v) => { if (v) ids.push(String(v)); };

  // #id and .class in CSS
  let m;
  const cssIdent = /[#.]([A-Za-z_][\w-]*)/g;
  while ((m = cssIdent.exec(sel)) !== null) push(m[1]);

  // [attr='value'], [attr="value"], [attr*='value']
  const attrVal = /\[[^\]=]+[*^$~|]?=\s*['"]?([^'"\]]+)['"]?\]/g;
  while ((m = attrVal.exec(sel)) !== null) push(m[1]);

  // XPath: @attr='value' and any quoted literal
  const xpAttr = /@[\w-]+\s*=\s*['"]([^'"]+)['"]/g;
  while ((m = xpAttr.exec(sel)) !== null) push(m[1]);

  // --- reduce each identifier to its most durable fragment ----------------
  const frags = [];
  for (const raw of ids) {
    for (const piece of raw.split(/\s+/)) {
      // Split on digit runs and keep the longest piece: the digits are the
      // part a redesign changes, the rest is the part that survives it.
      let best = '';
      for (const part of piece.split(/\d+/)) {
        if (part.length > best.length) best = part;
      }
      best = best.replace(/^[-_]+|[-_]+$/g, '');
      if (best.length < MIN) continue;
      if (GENERIC.has(best.toLowerCase())) continue;
      if (frags.indexOf(best) < 0) frags.push(best);
    }
  }

  if (!frags.length) {
    // Nothing distinctive enough to test - do not gate on a guess.
    return {present: true, reason: 'no-usable-fragment', fragments: []};
  }

  // --- does anything on the page carry one of them? -----------------------
  const esc = (s) => s.replace(/["\\]/g, '\\$&');
  let matched = 0;
  const hits = [];
  for (const f of frags) {
    const q = ['class', 'id', 'name', 'aria-label', 'data-testid', 'href']
        .map((a) => '[' + a + '*="' + esc(f) + '"]').join(',');
    let n = 0;
    try {
      n = document.querySelectorAll(q).length;
    } catch (e) {
      return {present: true, reason: 'query-failed', fragments: frags};
    }
    if (n > 0) { matched += n; hits.push(f); }
  }

  return {present: matched > 0, reason: matched > 0 ? 'family-found' : 'absent',
          fragments: frags, hits: hits, matches: matched};
}
"""
