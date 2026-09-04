"""Put the adjudicator weights into models/.

The service loads a plain GGUF file from disk. This script finds one and
copies it into place, so provisioning a new inference host is one command
rather than a paragraph of instructions.

    python scripts/provision_model.py                 # search the usual places
    python scripts/provision_model.py --source path/to/model.gguf
    python scripts/provision_model.py --list          # show what was found

It deliberately *copies* rather than symlinks: the deployment owns its weights,
and must keep working if whatever it was sourced from is uninstalled.
"""

from __future__ import annotations

import argparse
import urllib.request
import json
import os
import shutil
import struct
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET = os.path.join(ROOT, "models", "healing-adjudicator.gguf")

# GGUF metadata keys are typed; we only need enough of the header to prove the
# file is a real GGUF and read its architecture.
GGUF_MAGIC = b"GGUF"


def is_gguf(path: str) -> bool:
    try:
        with open(path, "rb") as fh:
            return fh.read(4) == GGUF_MAGIC
    except OSError:
        return False


def gguf_header(path: str):
    """Return (version, tensor_count, kv_count) or None."""
    try:
        with open(path, "rb") as fh:
            if fh.read(4) != GGUF_MAGIC:
                return None
            version = struct.unpack("<I", fh.read(4))[0]
            tensors = struct.unpack("<Q", fh.read(8))[0]
            kv = struct.unpack("<Q", fh.read(8))[0]
            return version, tensors, kv
    except (OSError, struct.error):
        return None


def registry_blobs(prefer: str = ""):
    """Find GGUF model blobs in a local model registry cache.

    Registries keep weights as content-addressed blobs with no file extension,
    and a manifest maps a readable name onto one. Reading the manifest is how
    we pick the *model* layer rather than a template or licence layer.

    Ranked newest-first, then largest, so the model most recently pulled onto
    this machine wins without the script having to hardcode a family name.
    """
    home = os.path.expanduser("~")
    root = os.environ.get("OLLAMA_MODELS", os.path.join(home, ".ollama", "models"))
    manifests = os.path.join(root, "manifests")
    blobs = os.path.join(root, "blobs")
    found = []
    if not os.path.isdir(manifests):
        return found
    for dirpath, _dirs, files in os.walk(manifests):
        for name in files:
            manifest = os.path.join(dirpath, name)
            try:
                with open(manifest, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
            except Exception:
                continue
            label = os.path.relpath(manifest, manifests).replace(os.sep, "/")
            for layer in data.get("layers", []):
                if layer.get("mediaType") != "application/vnd.ollama.image.model":
                    continue
                blob = os.path.join(blobs, layer["digest"].replace(":", "-"))
                if os.path.exists(blob):
                    found.append({
                        "label": label,
                        "path": blob,
                        "size_gb": round(layer.get("size", 0) / 1e9, 2),
                    })
    for f in found:
        try:
            f["mtime"] = os.path.getmtime(f["path"])
        except OSError:
            f["mtime"] = 0.0
    found.sort(key=lambda f: (bool(prefer) and prefer not in f["label"],
                              -f["mtime"], -f["size_gb"]))
    return found


def discover(prefer: str):
    """Every plausible source on this machine, best first."""
    candidates = list(registry_blobs(prefer))
    for extra in (os.path.join(ROOT, "models"),
                  os.path.expanduser("~/models"),
                  os.path.expanduser("~/Downloads")):
        if not os.path.isdir(extra):
            continue
        for name in sorted(os.listdir(extra)):
            path = os.path.join(extra, name)
            if name.endswith(".gguf") and os.path.isfile(path) and path != TARGET:
                candidates.append({
                    "label": os.path.join(os.path.basename(extra), name),
                    "path": path,
                    "size_gb": round(os.path.getsize(path) / 1e9, 2),
                })
    return candidates


def copy_with_progress(src: str, dst: str) -> None:
    total = os.path.getsize(src)
    done = 0
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".partial"
    with open(src, "rb") as fin, open(tmp, "wb") as fout:
        while True:
            chunk = fin.read(8 << 20)
            if not chunk:
                break
            fout.write(chunk)
            done += len(chunk)
            pct = 100.0 * done / max(total, 1)
            sys.stdout.write("\r  copying ... %5.1f%%  (%.2f / %.2f GB)"
                             % (pct, done / 1e9, total / 1e9))
            sys.stdout.flush()
    sys.stdout.write("\n")
    # Only take the target's name once the bytes are all there, so an
    # interrupted copy can never leave a half-model the service would load.
    os.replace(tmp, dst)


# --------------------------------------------------------------- downloading
#
# Discovery only helps a machine that already has a GGUF on it - an Ollama or
# LM Studio install, usually. A freshly cloned repository on a clean host has
# nothing to find, and that is the common case for a new inference node, so
# the script has to be able to fetch one too.

HF_TEMPLATE = "https://huggingface.co/%s/resolve/main/%s"

# Sizes are the 4-bit quantisations. The adjudicator picks an index out of a
# shortlist of eight, which does not need a large model; anything in this band
# is a reasonable starting point and the first one fits comfortably on a
# 4 GB card.
SUGGESTED = [
    ("Qwen/Qwen2.5-1.5B-Instruct-GGUF", "qwen2.5-1.5b-instruct-q4_k_m.gguf"),
    ("bartowski/Qwen2.5-3B-Instruct-GGUF", "Qwen2.5-3B-Instruct-Q4_K_M.gguf"),
    ("bartowski/Llama-3.2-3B-Instruct-GGUF",
     "Llama-3.2-3B-Instruct-Q4_K_M.gguf"),
]


def download(url: str, dst: str) -> bool:
    """Stream a GGUF to disk, then move it into place only once it is whole.

    Downloaded beside the target and renamed at the end, so an interrupted
    transfer leaves no file rather than a truncated one that loads far enough
    to fail confusingly later.
    """
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    tmp = dst + ".part"
    last = [0.0]
    print("downloading %s" % url)
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            total = int(resp.headers.get("Content-Length") or 0)
            done = 0
            with open(tmp, "wb") as out:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
                    done += len(chunk)
                    if not total:
                        continue
                    pct = 100.0 * done / total
                    # A carriage return is a progress bar on a terminal
                    # and thirty kilobytes of noise in a build log, so
                    # only one of those two audiences gets it.
                    if sys.stdout.isatty():
                        sys.stdout.write("\r  %5.1f%%  %6.2f / %5.2f GB"
                                         % (pct, done / 1e9, total / 1e9))
                        sys.stdout.flush()
                    elif pct - last[0] >= 10:
                        last[0] = pct
                        print("  %3.0f%%  %.2f / %.2f GB"
                              % (pct, done / 1e9, total / 1e9))
        if sys.stdout.isatty():
            print("")
    except Exception as exc:
        print("download failed: %s" % exc)
        if os.path.exists(tmp):
            os.unlink(tmp)
        return False

    if not is_gguf(tmp):
        print("what arrived is not a GGUF file - discarding it")
        os.unlink(tmp)
        return False
    os.replace(tmp, dst)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", help="explicit path to a .gguf file")
    ap.add_argument("--target", default=TARGET)
    ap.add_argument("--prefer", default="",
                    help="substring of a model name to rank first; by default "
                         "the most recently pulled model wins")
    ap.add_argument("--list", action="store_true", help="show sources and exit")
    ap.add_argument("--force", action="store_true", help="overwrite the target")
    ap.add_argument("--url", default="",
                    help="download a .gguf from this URL instead of "
                         "searching this machine")
    ap.add_argument("--hf", default="",
                    help="download from Hugging Face, as 'repo/file.gguf' - "
                         "e.g. Qwen/Qwen2.5-1.5B-Instruct-GGUF/"
                         "qwen2.5-1.5b-instruct-q4_k_m.gguf")
    ap.add_argument("--suggest", action="store_true",
                    help="print some models known to work, and exit")
    args = ap.parse_args()

    if args.suggest:
        print("Any instruction-tuned GGUF that llama.cpp can load will work.")
        print("These are reasonable starting points:\n")
        for repo, name in SUGGESTED:
            print("  python scripts/provision_model.py --hf %s/%s" % (repo, name))
        return 0

    if args.url or args.hf:
        if os.path.exists(args.target) and not args.force:
            print("%s already exists - pass --force to replace it" % args.target)
            return 1
        url = args.url
        if not url:
            repo, _, name = args.hf.rpartition("/")
            if not repo or not name.endswith(".gguf"):
                print("--hf wants 'repo/file.gguf', for example:\n"
                      "  Qwen/Qwen2.5-1.5B-Instruct-GGUF/"
                      "qwen2.5-1.5b-instruct-q4_k_m.gguf")
                return 2
            url = HF_TEMPLATE % (repo, name)
        if not download(url, args.target):
            return 1
        header = gguf_header(args.target)
        print("ready: %s" % args.target)
        print("  %.2f GB, GGUF v%s, %d tensors"
              % (os.path.getsize(args.target) / 1e9, header[0], header[1]))
        return 0

    if args.source:
        source = args.source
        if not is_gguf(source):
            print("not a GGUF file: %s" % source)
            return 2
    else:
        found = discover(args.prefer)
        if args.list:
            if not found:
                print("no GGUF models found on this machine")
                return 1
            for f in found:
                print("  %-46s %5.2f GB  %s" % (f["label"], f["size_gb"], f["path"]))
            return 0
        if not found:
            print("no GGUF model found on this machine.\n\n"
                  "Download one:\n"
                  "  python scripts/provision_model.py --suggest\n\n"
                  "or point at a file you already have:\n"
                  "  python scripts/provision_model.py --source path/to/model.gguf")
            return 1
        if len(found) > 1 and not args.prefer:
            # Several models on this machine and nothing to choose
            # between them. Guessing would silently install the wrong
            # weights, and the mistake would surface much later as poor
            # healing decisions rather than as an error here.
            print("found %d models - pick one:" % len(found))
            for f in found:
                print("  %-46s %5.2f GB" % (f["label"], f["size_gb"]))
            print("")
            print("  python scripts/provision_model.py --prefer <substring>")
            print("  python scripts/provision_model.py --source <path>")
            return 1
        source = found[0]["path"]
        print("source: %s (%s, %.2f GB)"
              % (found[0]["label"], source, found[0]["size_gb"]))

    if os.path.exists(args.target) and not args.force:
        print("target already exists: %s\n  (use --force to replace it)"
              % args.target)
        return 0

    copy_with_progress(source, args.target)
    header = gguf_header(args.target)
    if not header:
        print("copied file is not a readable GGUF - removing")
        os.remove(args.target)
        return 2
    print("ready: %s\n  GGUF v%d, %d tensors, %d metadata keys, %.2f GB"
          % (args.target, header[0], header[1], header[2],
             os.path.getsize(args.target) / 1e9))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
