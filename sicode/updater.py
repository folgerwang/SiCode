"""Self-update helpers: check GitHub Releases for newer SiCode wheels and install them."""
import json
import os
import platform
import re
import subprocess
import sys
import urllib.error
import urllib.request

from . import __version__

GITHUB_API = "https://api.github.com/repos/folgerwang/SiCode/releases/latest"


def _parse_ver(s):
    nums = [int(x) for x in re.findall(r"\d+", str(s or ""))]
    return tuple(nums[:4]) if nums else (0,)


def _platform_match(name):
    low = name.lower()
    if not low.endswith(".whl"):
        return False
    if sys.platform == "win32":
        return "win_amd64" in low
    if sys.platform == "darwin":
        arch = platform.machine().lower()
        return ("arm64" in low or "universal2" in low) if "arm" in arch or "aarch64" in arch else ("macosx" in low)
    return "manylinux" in low or "linux_x86_64" in low


def check_update(current_version=None, timeout=6):
    """Fetch latest release metadata from GitHub and compare against current_version."""
    cur = current_version or __version__
    req = urllib.request.Request(
        GITHUB_API,
        headers={"Accept": "application/vnd.github+json", "User-Agent": f"SiCode/{cur}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
    except (urllib.error.URLError, TimeoutError, ValueError, OSError) as e:
        return {"current": cur, "latest": cur, "has_update": False, "error": str(e)}

    tag = str(data.get("tag_name") or "")
    latest_nums = _parse_ver(tag)
    cur_nums = _parse_ver(cur)
    latest_str = ".".join(str(x) for x in latest_nums) if latest_nums != (0,) else tag
    has_update = latest_nums > cur_nums

    assets = data.get("assets") or []
    wheel = next((a for a in assets if _platform_match(str(a.get("name") or ""))), None)
    return {
        "current": cur,
        "latest": latest_str,
        "has_update": has_update,
        "tag": tag,
        "url": str(data.get("html_url") or "https://github.com/folgerwang/SiCode/releases"),
        "wheel_url": str(wheel.get("browser_download_url")) if wheel else "",
        "wheel_name": str(wheel.get("name")) if wheel else "",
        "notes": str(data.get("body") or "")[:600],
        "error": None,
    }


def apply_update(wheel_url=None):
    """Download and install the latest SiCode wheel for this platform via pip."""
    info = check_update() if not wheel_url else {"wheel_url": wheel_url, "has_update": True, "error": None}
    if info.get("error"):
        return {"ok": False, "output": info["error"]}
    url = info.get("wheel_url")
    if not url:
        return {"ok": False, "output": "No matching wheel asset found for this platform on GitHub Releases."}
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade", url]
    if os.name != "nt" and not os.environ.get("VIRTUAL_ENV"):
        cmd.insert(4, "--user")
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    out = ((p.stdout or "") + "\n" + (p.stderr or "")).strip()
    return {"ok": p.returncode == 0, "latest": info.get("latest", ""), "output": out[-1200:]}
