"""Where the accelerator-backed parts plug in, without the core knowing them.

The sentence encoder and the GGUF model are the two pieces that need real
hardware. They belong to the healing service, not to the shared core, and
certainly not on an RPA machine: a bot should never install torch to click a
button.

So the core declares the two slots and the service fills them. Importing
``healing_service`` registers both; nothing else has to happen, and no module
here ever names the service. The dependency runs one way - service depends on
core - which is what lets the core stay pure standard library.

Nothing registered is a normal state, not an error. It means this process has
no accelerator half, which is exactly true of every bot in the fleet: they
call the service over HTTP instead. The adjudicator then falls back to the
deterministic signals, which is the same degradation as an unreachable
service or a missing model.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

log = logging.getLogger("bis.runtime")

_encoder_factory: Optional[Callable[..., Any]] = None
_model_factory: Optional[Callable[..., Any]] = None


def register_encoder(factory: Callable[..., Any]) -> None:
    """Provide the sentence encoder. Called by the healing service."""
    global _encoder_factory
    _encoder_factory = factory


def register_model(factory: Callable[..., Any]) -> None:
    """Provide the GGUF adjudicator model. Called by the healing service."""
    global _model_factory
    _model_factory = factory


def has_local_runtime() -> bool:
    """Can this process run the models itself?"""
    return _encoder_factory is not None or _model_factory is not None


def make_encoder(model_name: str):
    """The encoder, or None when this process has no accelerator half."""
    if _encoder_factory is None:
        log.info("no local sentence encoder in this process - the semantic "
                 "signal is unavailable, the deterministic ones still apply")
        return None
    return _encoder_factory(model_name)


def make_model(**kwargs):
    """The adjudicator model, or None when this process has no local half."""
    if _model_factory is None:
        log.info("no local adjudicator model in this process - set "
                 "service_url to reach one, or install the healing service "
                 "here to run it in-process")
        return None
    return _model_factory(**kwargs)
