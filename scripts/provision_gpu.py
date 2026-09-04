"""Put a GPU-capable llama.cpp into the interpreter that runs this script.

`pip install llama-cpp-python` gives you the CPU-only wheel however good your
driver is, so a fresh virtual environment always starts on CPU. This script
does the two things that fix that, into whichever environment invokes it:

1. installs a llama-cpp-python built with CUDA support, and
2. puts the cuBLAS runtime it links against next to the backend DLL.

Step 2 is the one that catches people out. A CUDA wheel does not bundle
cuBLAS, so ``ggml-cuda.dll`` fails to load with an error naming *itself*
rather than the library it could not find, and the offload probe quietly
reports False on a perfectly good card.

    cd healing-service
    .venv\\Scripts\\activate
    python scripts/provision_gpu.py            # then restart the service

`fix_nvidia_driver.md` is the long-form version, including what to do when
the driver itself is too old. Nothing here touches the driver.
"""

from __future__ import annotations

import argparse
import ctypes
import glob
import json
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import urllib.request
import zipfile

# A maintained fork that publishes Windows CUDA wheels. The official index
# stopped at 0.2.67 (April 2024), which predates most current model
# architectures and will refuse a modern GGUF with "unknown model
# architecture" - see fix_nvidia_driver.md, Option A.
WHEEL_RELEASES = ("https://api.github.com/repos/JamePeng/llama-cpp-python"
                  "/releases?per_page=12")

# NVIDIA's own redistributable index. Only libcublas is needed; the full
# toolkit is roughly six times the size and none of the rest is used.
CUDA_REDIST = ("https://developer.download.nvidia.com/compute/cuda/redist"
               "/redistrib_%s.json")
CUDA_BASE = "https://developer.download.nvidia.com/compute/cuda/redist/"

NEEDED_DLLS = ("cublas64_13.dll", "cublasLt64_13.dll")


def _log(msg: str) -> None:
    print("  %s" % msg, flush=True)


def driver_cuda_version() -> str:
    """What CUDA the installed driver can support, via nvidia-smi."""
    try:
        out = subprocess.run(["nvidia-smi"], capture_output=True, text=True,
                             timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return ""
    for token in out.replace(":", " ").split():
        if token.count(".") == 1 and token.replace(".", "").isdigit():
            major = token.split(".")[0]
            if major.isdigit() and 11 <= int(major) <= 20:
                return token
    return ""


def lib_dir() -> str:
    """Where this interpreter's llama_cpp keeps its shared libraries."""
    import llama_cpp
    return os.path.join(os.path.dirname(os.path.abspath(llama_cpp.__file__)),
                        "lib")


# Two things make an in-process check wrong here, and both bit during
# development:
#
#   * backends are registered lazily, inside Llama.__init__, so a bare
#     llama_supports_gpu_offload() answers False on a working CUDA build, and
#   * this script *replaces* llama_cpp while running, so any module it has
#     already imported is the old one. Asking again in the same interpreter
#     reports the state before the install, which looks exactly like failure.
#
# So the probe runs in a fresh subprocess, every time.
_PROBE = """
import ctypes, pathlib
import llama_cpp
try:
    from llama_cpp._ggml import ggml_backend_load_all_from_path as load
    lib = pathlib.Path(llama_cpp.__file__).resolve().parent / 'lib'
    load(ctypes.c_char_p(str(lib).encode('utf-8')))
except Exception:
    pass
print('OFFLOAD=%s' % llama_cpp.llama_supports_gpu_offload())
"""


def gpu_offload_ready() -> bool:
    """Can this environment offload to a GPU? Asked in a clean interpreter."""
    try:
        out = subprocess.run([sys.executable, "-c", _PROBE],
                             capture_output=True, text=True, timeout=180)
    except (OSError, subprocess.SubprocessError):
        return False
    return "OFFLOAD=True" in (out.stdout or "")


def pick_wheel(cuda: str) -> str:
    """Newest published wheel matching this Python, OS and CUDA major."""
    tag = "cp%d%d" % sys.version_info[:2]
    plat = "win_amd64" if os.name == "nt" else "linux_x86_64"
    major = (cuda or "13").split(".")[0]
    with urllib.request.urlopen(WHEEL_RELEASES, timeout=60) as fh:
        releases = json.load(fh)
    fallback = ""
    for rel in releases:
        for asset in rel.get("assets", []):
            name = asset["name"]
            if tag not in name or plat not in name or not name.endswith(".whl"):
                continue
            if ("cu%s" % major) in name:
                return asset["browser_download_url"]
            fallback = fallback or asset["browser_download_url"]
    return fallback


def download(url: str, dest: str) -> str:
    _log("downloading %s" % os.path.basename(url.split("?")[0]))
    with urllib.request.urlopen(url, timeout=1800) as src, \
            open(dest, "wb") as out:
        shutil.copyfileobj(src, out, 1 << 20)
    _log("  %.0f MB" % (os.path.getsize(dest) / 1e6))
    return dest


def install_wheel(path: str) -> None:
    # --no-deps so a pinned numpy in the wheel metadata cannot downgrade the
    # one the rest of the service is already using.
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-deps",
                           "--force-reinstall", "--disable-pip-version-check",
                           path])


def install_cublas(cuda: str, work: str) -> bool:
    """Fetch libcublas from NVIDIA and drop the DLLs beside ggml-cuda.dll."""
    target = lib_dir()
    if all(os.path.exists(os.path.join(target, d)) for d in NEEDED_DLLS):
        _log("cuBLAS already present")
        return True

    for version in (cuda, "13.1.0", "13.0.0"):
        if not version:
            continue
        if version.count(".") == 1:
            version += ".0"
        try:
            with urllib.request.urlopen(CUDA_REDIST % version,
                                        timeout=60) as fh:
                manifest = json.load(fh)
        except Exception:
            continue
        entry = (manifest.get("libcublas") or {}).get("windows-x86_64")
        if not entry:
            continue
        archive = download(CUDA_BASE + entry["relative_path"],
                           os.path.join(work, "cublas.zip"))
        with zipfile.ZipFile(archive) as z:
            for name in z.namelist():
                base = os.path.basename(name)
                if base in NEEDED_DLLS:
                    with z.open(name) as s, \
                            open(os.path.join(target, base), "wb") as d:
                        shutil.copyfileobj(s, d, 1 << 20)
                    _log("installed %s" % base)
        return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wheel", default="",
                    help="a local .whl to use instead of downloading one")
    ap.add_argument("--check", action="store_true",
                    help="report the state and change nothing")
    args = ap.parse_args()

    in_venv = sys.prefix != sys.base_prefix
    print("interpreter : %s" % sys.executable)
    print("virtualenv  : %s" % ("yes" if in_venv else
                                "NO - this will modify the system install"))
    cuda = driver_cuda_version()
    print("driver CUDA : %s" % (cuda or "nvidia-smi not found"))
    print("gpu offload : %s" % gpu_offload_ready())

    if args.check:
        return 0
    if not cuda:
        print("\nNo NVIDIA driver visible. Nothing to do here - see "
              "fix_nvidia_driver.md steps 1-3.")
        return 1
    if gpu_offload_ready():
        print("\nAlready GPU-capable. Nothing to do.")
        return 0

    print()
    with tempfile.TemporaryDirectory() as work:
        wheel = args.wheel or pick_wheel(cuda)
        if not wheel:
            print("no CUDA wheel published for Python %d.%d on this platform "
                  "- build from source, fix_nvidia_driver.md Option B"
                  % sys.version_info[:2])
            return 1
        if not os.path.exists(wheel):
            wheel = download(wheel, os.path.join(work, "llama_cpp.whl"))
        install_wheel(wheel)
        if not install_cublas(cuda, work):
            print("could not fetch cuBLAS automatically - "
                  "fix_nvidia_driver.md step 4b does it by hand")
            return 1

    ok = gpu_offload_ready()
    print("\ngpu offload : %s" % ok)
    if ok:
        print("Restart the service; it caches Python at import.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
