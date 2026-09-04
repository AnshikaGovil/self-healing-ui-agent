"""Local inference runtime for the healing adjudicator.

Loads a GGUF model straight off disk with llama.cpp - no daemon, no model
registry, no external service. The weights live in ``models/`` inside the
project, so a deployment is a copy of this directory plus a pip install.

GPU is the intended deployment target: ``gpu_layers = -1`` offloads every
layer, and the service reports the device it *actually* got rather than the
one it hoped for. That honesty matters, because the whole point of putting
this behind an API is that one GPU host serves many bots - if it quietly fell
back to CPU, the fleet would slow down with nothing in the logs to say why.

Thread safety: a llama.cpp context is single-threaded. All generation goes
through one lock, so the service serialises requests rather than corrupting
state under concurrent load.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

log = logging.getLogger("bis.inference")

DEFAULT_MODEL_PATH = os.path.join("models", "healing-adjudicator.gguf")

# Constrained decoding: the sampler is restricted to tokens that keep the
# output valid against this schema, so the model physically cannot emit prose,
# a reasoning preamble, or a malformed object. It is what makes a small
# open-weight model dependable in an automated decision path.
DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "choice": {"type": "integer"},
        "confidence": {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": ["choice", "confidence", "reason"],
}

SYSTEM_PROMPT = (
    "You are the locator adjudicator inside an RPA self-healing agent. "
    "A UI automation step failed because its selector no longer matches the "
    "page. You are given the business intent of the step, the selector that "
    "broke, and a numbered shortlist of real elements harvested from the live "
    "DOM. Choose the single element that a human RPA developer would rebind "
    "the step to.\n"
    "Rules: prefer visible, interactive elements for click and type steps; "
    "prefer elements whose text or aria-label matches the intent; treat class "
    "name version bumps (pd19 -> pd21, v3 -> v4) as the same component; ignore "
    "elements that merely contain the target as a child when a tighter match "
    "exists. If none of the candidates plausibly serves the intent, answer -1. "
    "Reply with JSON only."
)


@dataclass
class Verdict:
    """What the AI adjudicator concluded about a shortlist."""

    choice: int = -1
    confidence: float = 0.0
    reason: str = ""
    latency_s: float = 0.0
    ok: bool = False
    error: str = ""
    device: str = ""

    @property
    def abstained(self) -> bool:
        return not self.ok or self.choice < 0


class AdjudicatorModel:
    """A loaded GGUF model, ready to adjudicate locator shortlists."""

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH,
                 gpu_layers: int = -1, context: int = 8192,
                 threads: Optional[int] = None, batch: int = 512,
                 max_tokens: int = 160):
        self.model_path = model_path
        self.gpu_layers = gpu_layers
        self.context = context
        self.threads = threads or max(2, (os.cpu_count() or 8) // 2)
        self.batch = batch
        self.max_tokens = max_tokens
        self._llm = None
        self._lock = threading.Lock()
        self._load_error = ""
        self._device = "unloaded"
        self._loaded_at = 0.0

    # ------------------------------------------------------------- loading
    @property
    def available(self) -> bool:
        return self._llm is not None

    @property
    def device(self) -> str:
        return self._device

    def describe(self) -> Dict[str, Any]:
        """Everything an operator needs to know about this worker."""
        return {
            "loaded": self.available,
            "device": self._device,
            "gpu_offload_supported": _gpu_offload_supported(),
            "gpu_layers_requested": self.gpu_layers,
            "context": self.context,
            "threads": self.threads,
            "model_file": os.path.basename(self.model_path),
            "model_size_gb": (round(os.path.getsize(self.model_path) / 1e9, 2)
                              if os.path.exists(self.model_path) else None),
            "load_error": self._load_error,
            "loaded_at": self._loaded_at,
        }

    def load(self) -> bool:
        """Load the weights. Safe to call repeatedly; only the first works."""
        if self._llm is not None:
            return True
        with self._lock:
            if self._llm is not None:
                return True
            if not os.path.exists(self.model_path):
                self._load_error = "model file not found: %s" % self.model_path
                self._device = "unavailable"
                log.error(self._load_error)
                return False
            try:
                from llama_cpp import Llama
            except ImportError as exc:
                self._load_error = "llama-cpp-python not installed (%s)" % exc
                self._device = "unavailable"
                log.error(self._load_error)
                return False

            started = time.time()
            supported = _gpu_offload_supported()
            layers = self.gpu_layers if supported else 0
            if self.gpu_layers != 0 and not supported:
                # Say it once, loudly. A GPU deployment that silently serves
                # from CPU is the failure mode this service exists to avoid.
                log.warning(
                    "this llama-cpp-python build has no GPU offload compiled "
                    "in - serving on CPU. Install a CUDA/Metal/ROCm build on "
                    "the inference host to use the accelerator.")
            try:
                self._llm = Llama(
                    model_path=self.model_path,
                    n_ctx=self.context,
                    n_gpu_layers=layers,
                    n_threads=self.threads,
                    n_batch=self.batch,
                    verbose=False,
                )
            except Exception as exc:
                self._load_error = "model load failed: %s" % exc
                self._device = "unavailable"
                log.error(self._load_error)
                return False
            self._device = "gpu" if (supported and layers != 0) else "cpu"
            self._loaded_at = time.time()
            log.info("adjudicator model loaded on %s in %.1fs (%s)",
                     self._device, time.time() - started,
                     os.path.basename(self.model_path))
            return True

    # ---------------------------------------------------------- inference
    @staticmethod
    def build_prompt(intent: str, broken_selector: str, action: str,
                     page_title: str, shortlist: List[str],
                     heuristic_pick: int = 0,
                     required_text: Optional[str] = None) -> str:
        lines = [
            "BUSINESS INTENT: " + intent,
            "STEP ACTION: " + action,
            "BROKEN SELECTOR: " + broken_selector,
            "PAGE: " + (page_title or "-"),
        ]
        if required_text:
            # A hard constraint, stated as one: the validation gate on the bot
            # will reject any pick that fails it, so spending prompt on it pays.
            lines.append("HARD CONSTRAINT: the element text must match the "
                         "regular expression /%s/" % required_text)
        lines += ["", "CANDIDATES (harvested from the live DOM):"]
        lines += shortlist
        lines += [
            "",
            "The heuristic ranker's current best guess is index %d." % heuristic_pick,
            "Confirm it or choose better.",
            'Answer JSON: {"choice": <index or -1>, '
            '"confidence": <0.0-1.0>, "reason": "<max 25 words>"}',
        ]
        return "\n".join(lines)

    def adjudicate(self, intent: str, broken_selector: str, action: str,
                   page_title: str, shortlist: List[str],
                   heuristic_pick: int = 0,
                   required_text: Optional[str] = None) -> Verdict:
        """Pick an index from the shortlist. Never raises.

        Any failure is reported as an abstention so the caller can fall back to
        the deterministic ranking rather than losing the step.
        """
        verdict = Verdict(device=self._device)
        if not self.load():
            verdict.error = self._load_error or "model unavailable"
            return verdict
        # Read the device *after* loading. Before this, the first heal in every
        # process reported "unloaded" - which is exactly the run an operator
        # reads to find out whether the GPU is being used.
        verdict.device = self._device

        prompt = self.build_prompt(intent, broken_selector, action, page_title,
                                   shortlist, heuristic_pick, required_text)
        started = time.time()
        try:
            with self._lock:
                out = self._llm.create_chat_completion(
                    messages=[{"role": "system", "content": SYSTEM_PROMPT},
                              {"role": "user", "content": prompt}],
                    max_tokens=self.max_tokens,
                    temperature=0.0,      # same page, same verdict
                    top_p=0.9,
                    response_format={"type": "json_object",
                                     "schema": DECISION_SCHEMA},
                )
        except Exception as exc:
            verdict.latency_s = time.time() - started
            verdict.error = "inference failed: %s" % exc
            log.warning("adjudication failed after %.1fs: %s",
                        verdict.latency_s, exc)
            return verdict
        verdict.latency_s = time.time() - started

        raw = ((out.get("choices") or [{}])[0].get("message") or {}).get("content", "")
        parsed = _extract_json(raw or "")
        if parsed is None:
            verdict.error = "unparseable response: %s" % (raw or "")[:120]
            return verdict
        try:
            verdict.choice = int(parsed.get("choice", -1))
            verdict.confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0.0))))
            verdict.reason = str(parsed.get("reason", ""))[:220]
            verdict.ok = True
        except (TypeError, ValueError) as exc:
            verdict.error = "bad fields: %s" % exc
        return verdict

    def warmup(self) -> bool:
        """Load the weights and run one token, so the first real request does
        not pay the load cost and misreport steady-state latency."""
        if not self.load():
            return False
        try:
            with self._lock:
                self._llm.create_completion("ok", max_tokens=1, temperature=0.0)
            return True
        except Exception as exc:
            log.warning("warmup generation failed: %s", exc)
            return False


_backends_registered = False


def _register_backends() -> None:
    """Register ggml's dynamic backends so the GPU probe can see them.

    Current llama-cpp-python builds ship every ggml backend as its own
    shared library and register them lazily, inside ``Llama.__init__``.
    Until that happens ``llama_supports_gpu_offload()`` answers False on a
    perfectly good CUDA build - so this service announced CPU, passed
    ``n_gpu_layers=0``, and offloaded nothing, which is exactly the silent
    fallback the split exists to prevent. Registering first is cheap and
    idempotent; builds that link their backends statically have no such
    symbol and need nothing done.
    """
    global _backends_registered
    if _backends_registered:
        return
    try:
        import ctypes
        import pathlib

        import llama_cpp
        from llama_cpp._ggml import ggml_backend_load_all_from_path

        lib = pathlib.Path(llama_cpp.__file__).resolve().parent / "lib"
        ggml_backend_load_all_from_path(
            ctypes.c_char_p(str(lib).encode("utf-8")))
    except Exception:
        # Older or statically linked builds. The probe below is still
        # authoritative either way, so this is not worth a warning.
        pass
    _backends_registered = True


def _gpu_offload_supported() -> bool:
    try:
        from llama_cpp import llama_supports_gpu_offload
        _register_backends()
        return bool(llama_supports_gpu_offload())
    except Exception:
        return False


def _extract_json(raw: str) -> Optional[dict]:
    """Parse JSON out of a response, tolerating fences and stray prose."""
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    start, depth = -1, 0
    for i, ch in enumerate(raw):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    return json.loads(raw[start:i + 1])
                except json.JSONDecodeError:
                    start = -1
    return None
