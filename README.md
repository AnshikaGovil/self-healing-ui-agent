# Self-Healing UI Agent

**Locator repair for UI automation, as a service.** When an automated process
stops because the front end it drives has been redesigned, this decides which
element the step *meant* and hands back an answer the bot can act on — inline,
while the process is still running.

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-42%20offline-brightgreen)
![Runtime](https://img.shields.io/badge/inference-llama.cpp%20%7C%20GGUF-orange)

---

## The problem

Every UI automation is built on selectors, and every selector is a bet that a
front-end team will not rename anything. They do. A component library goes
from `pd19-product-card` to `pd21-product-card`, a `data-testid` is dropped,
a wrapper `div` appears — and a process that ran for two years stops at 2 a.m.

Nothing is wrong with the logic. The button is still on the screen, in the
same place, with the same label. Only its address changed.

This service answers the question a human would answer in four seconds:
**given what this step was trying to do, which element on the page is it now?**

## How it decides

Repair is not pattern matching, and it is not "ask a model". It is evidence,
weighted, with a language model as one voice among several:

| signal | weight | what it knows |
| --- | --- | --- |
| identity | 0.24 | survival of the normalised id / class name |
| attributes | 0.18 | id, name, class and `data-*` token overlap |
| text | 0.18 | visible and accessible text similarity |
| semantic | 0.14 | sentence-embedding distance from the step's stated intent |
| structure | 0.10 | DOM path and ancestor chain, compared step by step |
| neighbour | 0.10 | the page copy *around* the element, word overlap weighted by word length |
| tag | 0.08 | tag and ARIA-role compatibility |
| geometry | 0.08 | position and size, in document coordinates, normalised by the real viewport |

A local GGUF model then adjudicates the shortlist and its verdict is fused in
at a weight of 0.35. It is deliberately not the decision-maker — see
[Guardrails](#guardrails).

**Digit runs are collapsed before comparison**, so `pd19` and `pd21` are the
same token and `v3` and `v4` are the same component. That single rule is the
highest-yield signal against real redesigns, because versioned class names are
how most front ends actually break.

**The neighbour signal is the one a rename cannot touch.** A design-system
bump rewrites every class on the page and leaves the words beside the button
exactly where they were. Word overlap is weighted by word *length*, so sharing
"Delivery" counts for more than sharing "the".

## Architecture

The pipeline is split at the point where the hardware requirements diverge.
Everything that needs a **browser** runs on the bot; everything that needs an
**accelerator** runs here. This repository is the second half.

```mermaid
flowchart LR
    subgraph BOT["Bot - many, no GPU"]
        A["step fails"] --> B["harvest the live DOM<br/>~900 elements"]
        B --> C["lexical pre-rank<br/>7 signals, no model"]
        F["bind + validate<br/>the selector"] --> G["apply / escalate / refuse"]
    end

    subgraph SVC["This service - one, has the accelerator"]
        D["semantic ranking<br/>sentence encoder"] --> E["AI adjudication<br/>GGUF, schema-constrained"]
    end

    C -- "POST fingerprints" --> D
    E -- "an INDEX, never a selector" --> F
```

One accelerated host serves an entire fleet. A bot needs no GPU, no weights
and no model runtime — it posts element fingerprints and gets back a number.

**What crosses the wire** is tag, attributes, label text, geometry and the
ancestor chain, for the shortlist the bot already believes in. Not the page,
not the DOM, not a screenshot, and never the values being typed. A repair
request is tens of kilobytes.

## Guardrails

The design choices that make this safe to put in front of a production robot,
each of which exists because the alternative fails badly:

**The model never invents a selector.** It returns an *index* into a list of
elements harvested from the live DOM. A hallucinated selector is not a
possible output.

**The service never returns a selector at all.** Only the machine holding the
browser can prove a selector resolves, so binding stays there. The response
carries an index, a confidence and a rationale.

**Absent signals score `None`, never zero.** A signal with no evidence is
dropped from the weighted mean rather than counted as disagreement —
otherwise a first-time repair is punished for having no history.

**Confidence is banded, and the bands are defined once.** Above
`auto_apply_threshold` a repair is applied; between that and
`review_threshold` a human is asked; below it the agent refuses and the step
fails loudly. Refusing to guess is a feature.

**A candidate and a baseline are described by the same code.** Every signal
compares one fingerprint against another field by field, so there is exactly
one `describe()` and both the harvester and the baseline fingerprinter splice
it in. This was two implementations that had quietly drifted: measured over
738 interactive elements on four live sites they disagreed about `visible` 40%
of the time and about `neighborText` 46%, and the baseline copy derived no
accessible name from a `<label>` — which on a login form is every field.
The worst of it was not the weaker score but a guard that had stopped running:
a baseline that is not visible is supposed to be refused, and the baseline
copy defined "visible" as "has a box", which the other half of the same check
already covered.

**Everything fails open.** An unreachable service, a missing model or
malformed output degrades to "no opinion" and the deterministic signals stand.
An automation must never stop because a healing service is down.

## Quick start

```bash
git clone https://github.com/AnshikaGovil/self-healing-ui-agent.git
cd self-healing-ui-agent

python -m venv .venv
.venv\Scripts\activate                # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

### Getting the weights

**The model is not in this repository.** A few gigabytes of binary does not
belong in git, so a fresh clone has no weights and the service will not start
until it does. One command fixes that.

Any instruction-tuned GGUF that llama.cpp can load will work. A 1.5-8B model
at 4-bit is the right band — the adjudicator picks an index out of a shortlist
of eight, which is not a task that needs a large model.

```bash
# see a few that are known to work
python scripts/provision_model.py --suggest

# download one (about 1.1 GB, fits a 4 GB card with room to spare)
python scripts/provision_model.py --hf \
    Qwen/Qwen2.5-1.5B-Instruct-GGUF/qwen2.5-1.5b-instruct-q4_k_m.gguf
```

If the machine already has GGUF models — an Ollama or LM Studio install, say
— it can source one locally instead of downloading anything:

```bash
python scripts/provision_model.py --list      # what is already on this machine
python scripts/provision_model.py --prefer 3b
python scripts/provision_model.py --source path/to/model.gguf
```

Any of those routes leaves the weights at `models/healing-adjudicator.gguf`,
which is what the service loads and what `.gitignore` keeps out of commits.
The download is written to a `.part` file and renamed only once it is whole,
so an interrupted transfer leaves nothing rather than a truncated file that
loads far enough to fail confusingly later. `--url` takes any direct link if
you host your own weights.

Run it. `--gpu-layers` is the only thing that differs between the two
deployments:

```bash
# CPU - works anywhere, nothing further to install
python -m healing_service --host 0.0.0.0 --port 8077 --gpu-layers 0 --context 4096

# GPU - offload every layer that fits
python -m healing_service --host 0.0.0.0 --port 8077 --gpu-layers -1 --context 4096

curl http://localhost:8077/v1/health
```

The service reports the device it *actually* got, rather than the one it hoped
for — a fleet that has silently dropped to CPU should be visible, not a
mystery:

```json
{ "ready": true,
  "ai": { "loaded": true, "device": "gpu", "gpu_offload_supported": true } }
```

### Running on CPU

CPU is a supported deployment, not a degraded one. `pip install -r
requirements.txt` already installs a CPU-only llama.cpp build, so the steps
above are complete as they stand — no driver to match, no CUDA toolkit, no
further downloads:

```bash
python -m healing_service --port 8077 --gpu-layers 0 --context 4096
```

`--gpu-layers 0` forces it explicitly, which is worth doing even on a machine
that has a GPU when you want a like-for-like comparison. Health then reads:

```json
{ "ready": true,
  "ai": { "loaded": true, "device": "cpu",
          "gpu_offload_supported": true, "gpu_layers_requested": 0 } }
```

Those last two fields answer different questions, and the distinction is the
point of reporting both. `gpu_offload_supported` describes the **build** -
whether this llama.cpp could offload if asked. `device` describes this
**run**. Seeing `true` and `cpu` together means a capable build was told to
stay on the CPU, which is what `--gpu-layers 0` does. Seeing `false` means the
installed wheel has no GPU support compiled in at all, which is the case a
fleet needs to notice.

Measured with a 4B model at 4-bit: **about 8 s per repair on CPU against
about 6 s on a 4 GB GPU.** A repair only happens when a selector has already
broken, so that difference rarely decides anything — and the deterministic
signals, which do most of the work, cost the same either way.

Two settings matter more on CPU than they look:

* **`--context 4096`, or lower.** The adjudication prompt is a few hundred
  tokens. A larger context buys nothing and costs memory and
  prompt-processing time on every request.
* **`--no-ai` is a supported mode**, not a broken one. It serves the six
  deterministic signals and the semantic ranking with no language model at
  all, which is the fastest configuration available. Confidence falls, so
  more repairs land in the review band and fewer are applied autonomously —
  that is the honest trade, and the bands are what keep it safe.

To check what a given environment is actually capable of before starting the
service:

```bash
python scripts/provision_gpu.py --check
```

### Getting onto a GPU

`pip install llama-cpp-python` gives a CPU-only build however good your driver
is, so a fresh environment always starts on CPU. One command fixes that, into
whichever environment runs it:

```bash
python scripts/provision_gpu.py --check     # report, change nothing
python scripts/provision_gpu.py             # install a CUDA build + cuBLAS
```

It installs a CUDA-capable wheel *and* the cuBLAS runtime that wheel links
against but does not bundle. That second part is the one that costs people an
afternoon: without it the backend fails to load with an error naming
`ggml-cuda.dll` rather than the library it could not find, and the offload
probe simply reports `False`.

## API

| endpoint | purpose |
| --- | --- |
| `POST /v1/adjudicate` | rank a candidate pool and choose one — returns an index |
| `GET /v1/js` | the in-page JavaScript a thin client executes, plus a content hash |
| `POST /v1/baseline` | what an element looked like while its locator worked, kept for clients with no memory of their own |
| `GET /v1/health` | readiness, device, model, uptime, requests served, baselines held |
| `GET /docs` | generated OpenAPI reference |

Set `BIS_API_KEY` (or `--api-key`) to require an `X-API-Key` header. Unset
means open, which is right on localhost and wrong on a network — the service
logs a warning at startup either way.

`/v1/js` is what keeps a client thin. The DOM-harvesting JavaScript is the
component most likely to need work as real sites change, so it is **served
rather than embedded**: improving it is a service restart instead of a
redeploy to every bot in the fleet. Clients cache it against the content hash.

`/v1/baseline` is what keeps a thin client from being permanently cold.

Seven of the eight ranking signals compare a candidate against the element the
step used *while the locator still worked*. Structure, geometry and neighbour
have nothing to compare against without such a record; tag and text fall back to
whatever can be read out of the selector string, which for a bare `#firstName`
is nothing at all. Signals with no evidence are dropped from the weighted mean
— correctly, since scoring them zero would punish a repair for being the first
— so a cold heal is judged on a fraction of the total weight, and correct
repairs land in the review band and get refused.

A client with its own memory keeps baselines itself and sends one inline. A
client without one posts what worked to `/v1/baseline` and the service keys it
by locator and page. Three properties are worth stating:

- **It is shared, and that is the point.** Keys are per locator and per page,
  not per bot, so the first process to walk a page warms every process that
  walks it afterwards, including ones on other hosts.
- **The policy lives in one place.** The store refuses a fingerprint that is
  not on screen. No baseline beats a false one: seeding structure and geometry
  from a hidden or zero-sized element makes every later repair worse, and
  nothing downstream can tell.
- **Reading one back is opt-in** (`use_stored_baseline` on the adjudicate
  request). A null baseline from a client that manages its own memory means
  "my memory has none for this locator", not "I have no memory", and quietly
  substituting the fleet's would overrule it — starting with the benchmark,
  which wipes its memory per suite so that every row it grades is a cold heal.

Measured on one repair, cold then warm: a heuristic score of 0.281 against
0.918 for the same correct element. That is the difference between a repair
refused into the review band and a repair applied. Against a live retail
listing the same repair scored 94% cold and 100% warm, and after a service
restart the first repair of the next run was already 100%.

Baselines are element fingerprints — tag, attributes, label, ancestor chain,
bounding box. No page, no DOM, no typed values. They live in
`run-memory/baselines.json`; `--baselines PATH` moves them, and several
service processes should share one file or each will learn its own half of
what the fleet knows.

## Measured

Graded on a benchmark built from live sites — BBC, The Guardian, Flipkart and
Wikipedia, in both CSS and XPath — where real selectors are perturbed the way
a release leaves them and a repair is scored by resolving it and the answer key
in the same page and **comparing DOM nodes by identity**. "Healed" and "healed
to the right element" are different questions, and only the second one matters:

| | |
| --- | --- |
| correct element | **86.9%** of 84 locators |
| wrong element | 9.6% |
| latency per repair, GPU | ~6 s |
| latency per repair, CPU | ~8 s |
| deterministic ranking, 900 candidates | 131 ms cold, 261 ms warm |

**That percentage is only comparable to a run from the same day.** The suites
are built from live sites, and live sites move. Re-running them months later
reads 79.8%, entirely because three suites now yield 9-11 gradable locators
where they once yielded 12 — a control run with the change under test removed
reproduces the lower figure exactly. Compare against a control you ran
yourself, never against a number in a file.

The number that moved most recently is the one that matters most for safety.
Across a change set that unified the fingerprint, added the neighbour signal
and taught the ranker to keep the shape the developer wrote, **no suite got
worse**, correct held flat, and **wrong-element fell from 11.9% to 9.6%**.
Read that as a direction rather than a precision: two repairs out of eighty-odd
on sites that visibly drift between runs is not a large enough sample to call
a 2.3-point improvement exact. What it establishes is that the change set went
in without costing correctness anywhere.

Three findings from that data worth stating, because none is obvious:

**The language model has never overridden the ranker.** Across every recorded
repair: 61 agreements, zero overrides. The model is not selecting the element —
the deterministic signals are. What it contributes is *confidence*, and the
scores are systematically under-confident without it.

**Which means the obvious optimisation is wrong.** Dropping the model to save
the round trip would leave correct repairs scoring below the apply threshold,
so they would stop being applied and start interrupting a human — a worse
outcome than the latency it saves.

**But "under-confident" had two causes, and only one was calibration.** The
other was that the deterministic layer was being handed less evidence than it
should have had: no baseline, and — for clients posting the harvester's own
field names — attributes silently dropped in transit by a strict decoder. With
the AI disabled, so the deterministic layer stands alone, that bug had the
ranker putting the *wrong* element first while the model carried the result.
Both are fixed. The remaining gap is smaller and differently shaped than the
one that motivated the calibration work, which is an argument for measuring it
again before fitting a curve to it.

## Tests

```bash
python tests/test_similarity.py    # 35 - the ranking signals, CSS and XPath
python tests/test_service.py       # 21 - API contract, decision policy, wire schema, baselines
```

Both are **offline by design**: no browser, no service, no model, no network.
The FastAPI app runs in-process with the adjudicator disabled. That is
deliberate — the whole point of a deterministic layer is that it holds up when
none of the rest is available, and a suite that quietly needed a GPU would stop
proving it.

One of them enforces that `bis_core` imports nothing outside the standard
library. Every bot in a fleet installs that package, and the moment it reaches
for torch, that cost lands on every automation host for no reason.

## Layout

```
bis_core/            the shared core - pure standard library, no dependencies
  config.py          every tunable, and the confidence-band policy
  descriptors.py     LocatorSpec / ElementDescriptor, tokenising, digit collapsing
  similarity.py      the seven deterministic signals, selector mining
  adjudicator.py     the wire contract, and local / remote implementations
  js_harvest.py      in-page JS: fingerprinting, selector proposal, probes
  runtime.py         the slots the accelerator half plugs into

healing_service/     the API - this is what runs on the accelerated host
  app.py             FastAPI application and CLI
  schemas.py         request and response models
  inference.py       GGUF runtime, schema-constrained decoding
  embeddings.py      sentence encoder - lazy, offline, fail-open

scripts/             provision_model.py, provision_gpu.py
tests/               two offline suites
models/              created by provision_model.py - gitignored, never committed
```

`bis_core` declares two slots — an encoder and a model — and `healing_service`
fills them on import. The dependency runs one way only, which is what lets the
core stay dependency-free and installable anywhere.

## Licence

MIT.
