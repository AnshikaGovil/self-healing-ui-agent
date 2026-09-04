"""The healing service - the GPU-bound half of the agent, behind an API.

Run it on one accelerated host and point any number of bots at it:

    python -m healing_service --host 0.0.0.0 --port 8077
"""


from bis_core import runtime as _runtime

from .embeddings import default_encoder as _default_encoder
from .inference import AdjudicatorModel as _AdjudicatorModel

# Importing this package is what gives a process its accelerator half. The
# core declares the slots; filling them here keeps the dependency running
# one way, so bis-core never has to know this package exists.
_runtime.register_encoder(_default_encoder)
_runtime.register_model(_AdjudicatorModel)

from .app import create_app

__all__ = ["create_app"]
