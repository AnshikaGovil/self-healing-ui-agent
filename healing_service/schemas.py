"""Wire schemas for the healing service.

Kept deliberately permissive on the element fingerprints: a bot running an
older agent build must not start failing because the service learned about a
new descriptor field. Everything the service actually needs is required;
everything else is passed through.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class LocatorPayload(BaseModel):
    key: str = "locator"
    selector: str = ""
    description: str = ""
    tag: Optional[str] = None
    role: Optional[str] = None
    expected_text: Optional[str] = None
    multiple: bool = False


class PagePayload(BaseModel):
    url: str = ""
    title: str = ""


class AdjudicateRequest(BaseModel):
    locator: LocatorPayload
    action: str = "interact"
    page: PagePayload = Field(default_factory=PagePayload)
    candidates: List[Dict[str, Any]] = Field(default_factory=list)
    baseline: Optional[Dict[str, Any]] = None
    use_stored_baseline: bool = False
    """May the service supply a baseline this client did not send?

    Only true for a client with no memory of its own. The full agent keeps
    its own and a `baseline` of None from it means "I have none for this
    locator", which is a fact about its memory, not an invitation to
    substitute the fleet's. The benchmark depends on exactly that: it wipes
    its memory per suite so every row is a cold heal, and a service that
    quietly warmed those requests would make the measurement a lie without
    changing a single line of it."""

    optional: bool = False
    shortlist_size: int = 8
    budget_s: float = 25.0
    protocol: str = "1"


class BaselineRequest(BaseModel):
    """A fingerprint of an element a step has just used successfully.

    ``descriptor`` is deliberately untyped. It is whatever the client had:
    the harvester's own output from a thin client, a serialised
    ElementDescriptor from the full agent. The service coerces one shape
    into the other rather than making every client learn the difference.
    """

    key: str = "locator"
    selector: str = ""
    url: str = ""
    descriptor: Dict[str, Any] = Field(default_factory=dict)
    protocol: str = "1"


class BaselineResponse(BaseModel):
    stored: bool = False
    reason: str = ""
    entries: int = 0


class ShortlistItem(BaseModel):
    index: int
    score: float
    signals: Dict[str, Optional[float]] = Field(default_factory=dict)
    summary: str = ""
    chosen: bool = False


class AdjudicateResponse(BaseModel):
    chosen: int = -1
    confidence: float = 0.0
    # Which band the confidence falls in, per HealingConfig.decide(). The
    # full agent computes this itself from the same policy; a thin client
    # has no config of its own, so the service states it. Without it such a
    # client applies every repair it is handed, including refused ones.
    band: str = "refuse"
    strategy: str = ""
    rationale: str = ""
    shortlist: List[ShortlistItem] = Field(default_factory=list)
    ai_consulted: bool = False
    ai_choice: int = -1
    ai_confidence: float = 0.0
    ai_reason: str = ""
    ai_latency_s: float = 0.0
    ai_device: str = ""
    duration_s: float = 0.0
    error: str = ""


class JsBundleResponse(BaseModel):
    """The in-page JavaScript a thin client executes on the bot.

    Served rather than embedded so that improving the harvester does not mean
    redeploying every bot in the fleet. ``version`` is a content hash: a client
    caches the bundle against it and re-fetches only when it changes.
    """

    protocol: str = "1"
    version: str = ""
    harvest: str = ""
    probe: str = ""
    describe_one: str = ""
    presence: str = ""


class HealthResponse(BaseModel):
    ready: bool
    service: str
    version: str
    protocol: str
    encoder_ready: bool = False
    ai: Dict[str, Any] = Field(default_factory=dict)
    uptime_s: float = 0.0
    requests_served: int = 0
    baselines: int = 0
    """How many locator baselines are held for thin clients. Zero on a fresh
    service is expected; zero after a fleet has been running is the symptom
    of clients that are not reporting them."""
