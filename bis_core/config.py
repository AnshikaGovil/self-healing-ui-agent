"""Runtime configuration for the healing engine.

Every knob a delivery team is likely to tune in production lives here so the
engine itself stays free of magic numbers.  Values can be overridden from a
JSON file (``config.json``) or per-field in code.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict, fields
from typing import Any, Dict


@dataclass
class ScoringWeights:
    """Relative weight of each similarity signal in the candidate ranker.

    The weights are normalised at scoring time, so they express *relative*
    importance rather than absolute magnitudes.
    """

    identity: float = 0.24     # survival of the exact normalised id/class name
    tag: float = 0.08          # tag / ARIA-role compatibility
    attributes: float = 0.18   # id, name, class, data-* token overlap
    text: float = 0.18         # visible + accessible text similarity
    structure: float = 0.10    # DOM path and ancestor-chain similarity
    geometry: float = 0.08     # on-screen position and size (needs a baseline)
    neighbor: float = 0.10     # surrounding page copy (needs a baseline)
    semantic: float = 0.14     # embedding cosine vs the business intent


@dataclass
class HealingConfig:
    """Top-level configuration object."""

    # --- Locator resolution -------------------------------------------------
    action_timeout_ms: int = 6000
    """How long the *original* locator is given before it is declared broken."""

    post_heal_timeout_ms: int = 8000
    """Timeout applied to the healed locator when the action is retried."""

    # --- Healing budget (this is what makes it 'real time') -----------------
    heal_budget_s: float = 25.0
    """Hard wall-clock ceiling for one healing attempt, LLM call included."""

    max_candidates: int = 900

    # How long a harvested candidate set stays reusable for the same page.
    # A release breaks locators in groups, so the same page is harvested once
    # per broken locator within seconds of itself. Zero disables the reuse.
    harvest_cache_s: float = 20.0

    # Reuse one *widened* harvest for every locator on the page rather than
    # one per set of harvest options.
    #
    # Off by default, and that default is a measurement rather than caution.
    # Keyed on the exact options the cache is correct but almost never hits,
    # because `needText` and `tagHint` differ per locator. Widening it makes
    # it hit - and on the largest page in the benchmark, correctness moved
    # from 86.9% to 84.5%. Two locators, on a news site whose front page had
    # also rotated between the runs, so the cause was never established.
    #
    # Unresolved is not the same as harmless. Turn it on per site, with the
    # benchmark, once you can show it costs nothing there.
    harvest_cache_shared: bool = False

    # Learned host-wide renames. A repair proven on one locator is tried on
    # its siblings before anything expensive runs. The probe is short because
    # a miss must stay cheap - there may be several to try - and the
    # confidence is deliberately below auto_apply's neighbours: the
    # transformation is proven, this particular application of it is not, so
    # it still has to clear validation like anything else.
    use_rename_map: bool = True
    rename_probe_ms: int = 1200
    rename_confidence: float = 0.80
    """Upper bound on elements harvested from the live DOM per attempt."""

    shortlist_size: int = 8
    """Number of top-ranked candidates handed to the LLM adjudicator."""

    semantic_pool: int = 45
    """Embeddings are computed only for the top N lexically-ranked candidates.

    Encoding 900 sentences costs seconds; encoding 45 costs milliseconds, and
    an element invisible to all five lexical signals yet correct on semantics
    alone is rare enough not to justify that latency on every heal."""

    # --- Decision policy ----------------------------------------------------
    auto_apply_threshold: float = 0.62
    """Confidence at/above which a repair is applied autonomously."""

    review_threshold: float = 0.42
    """Confidence at/above which a repair is offered to a human reviewer.
    Below this the engine refuses to guess and the step fails loudly."""

    require_unique_match: bool = False
    """If True, a healed selector matching >1 element is rejected outright.
    If False the engine keeps it but scores it down and pins it to index 0."""

    # ------------------------------------------------------------------ policy
    APPLY = "apply"
    REVIEW = "review"
    REFUSE = "refuse"

    # An optional locator carries the opposite risk to a required one.
    # If a required element is refused the bot stops, loudly, and someone
    # looks at it. If an optional one is repaired wrongly the bot clicks
    # something nobody asked it to click, and no exception is ever raised
    # to say so. Absence is also the ordinary case for these - a consent
    # banner is missing far more often than it is renamed - so the bar to
    # act on one is deliberately higher than the auto-apply band.
    optional_min_confidence: float = 0.85

    def decide_optional(self, confidence: float) -> str:
        """The band for a locator the developer marked as optional.

        Layered on top of ``decide``, never around it: a repair must clear
        the normal apply band *and* this stricter floor. Skipping an
        optional step costs a pop-up that stays open; clicking the wrong
        element can cost a great deal more.
        """
        if (self.decide(confidence) == self.APPLY
                and confidence >= self.optional_min_confidence):
            return self.APPLY
        return self.REFUSE

    def decide(self, confidence: float) -> str:
        """Which band a repair falls into.

        Defined once and used by both the engine (which reports the band) and
        the agent (which enforces it), so the two can never disagree about what
        the thresholds mean.
        """
        if confidence >= self.auto_apply_threshold:
            return self.APPLY
        if confidence >= self.review_threshold:
            return self.REVIEW
        return self.REFUSE

    # --- Signals ------------------------------------------------------------
    weights: ScoringWeights = field(default_factory=ScoringWeights)

    use_embeddings: bool = True
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- AI adjudication ----------------------------------------------------
    use_ai: bool = True

    ai_weight: float = 0.35
    """How much the AI verdict is allowed to move the final confidence.
    The deterministic scorer keeps the remaining (1 - ai_weight) share, so a
    hallucinating model can never single-handedly authorise a bad repair."""

    model_path: str = "models/healing-adjudicator.gguf"
    """Open-weight model file, shipped inside the project. Only ever loaded by
    whichever process actually adjudicates - the service, or a bot running
    standalone."""

    gpu_layers: int = -1
    """Layers to offload to the accelerator. -1 offloads the whole model;
    0 forces CPU. Ignored when the installed runtime has no GPU support."""

    ai_context: int = 8192
    ai_max_tokens: int = 160

    decision_cache_s: float = 300.0
    """How long an adjudication may be reused for an identical request.

    A sharded job puts every worker on the same broken locator within seconds
    of each other. The candidate list they harvest is identical, so the verdict
    is identical - and at CPU latency, paying for it once instead of once per
    worker is the difference between a fleet stalling for 25s and stalling for
    25s divided by the number of workers. Set to 0 to disable."""

    decision_cache_size: int = 256

    # --- Remote healing service ---------------------------------------------
    service_url: str = ""
    """Base URL of the healing service, e.g. http://gpu-host:8077.

    Set it and this process becomes a thin client: it harvests the DOM, does
    the model-free lexical ranking, and sends only the surviving shortlist for
    scoring. Leave it empty to run the models in-process."""

    service_api_key: str = ""
    service_timeout_s: float = 90.0

    # --- Human in the loop --------------------------------------------------
    interactive_review: bool = True
    """Show a modal approval dialog for repairs in the review band."""

    show_message_boxes: bool = True
    """Show the operator-facing message boxes (used for the client demo)."""

    # --- Persistence --------------------------------------------------------
    memory_path: str = "memory/healing_memory.json"
    artifacts_dir: str = "artifacts"

    # --- Misc ---------------------------------------------------------------
    verbose: bool = True

    # ------------------------------------------------------------------ util
    @classmethod
    def load(cls, path: str | None = None) -> "HealingConfig":
        """Load config from JSON, falling back to defaults for missing keys."""
        if not path or not os.path.exists(path):
            return cls()
        with open(path, "r", encoding="utf-8") as fh:
            raw: Dict[str, Any] = json.load(fh)
        weights = ScoringWeights(**raw.pop("weights", {}) or {})
        known = {f.name for f in fields(cls)} - {"weights"}
        clean = {k: v for k, v in raw.items() if k in known}
        return cls(weights=weights, **clean)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
