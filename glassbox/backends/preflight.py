"""Checks that run before anything touches a local GPU."""
from __future__ import annotations

import os
import socket

from .base import PreflightError

#: Port used by a local llama-server chat stack. If it is listening, the GPUs are full.
BUSY_PORT = int(os.environ.get("GLASSBOX_BUSY_PORT", "8080"))


def port_in_use(port: int = BUSY_PORT, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex((host, port)) == 0


def gpu_preflight() -> dict:
    """Refuse to start a GPU run while another local model server holds the VRAM.

    Glassbox never stops that server itself; it tells the user to.
    """
    if os.environ.get("GLASSBOX_SKIP_PREFLIGHT") == "1":
        return {"port_checked": BUSY_PORT, "skipped": True}
    if port_in_use():
        raise PreflightError(
            f"Port {BUSY_PORT} is in use: a local model server (for example llama-server) is running "
            "and is probably holding most of the VRAM on your GPUs. Please close that server "
            "yourself, then run this again. Glassbox will not stop it for you."
        )
    return {"port_checked": BUSY_PORT, "in_use": False}


def gpu_report() -> list[dict]:
    """Memory per GPU as the driver sees it across all processes.

    nvidia-smi is asked first: on Windows, CUDA's own free-memory query does not reliably count what other
    processes hold, and glassbox needs the machine-wide picture.
    """
    import shutil
    import subprocess

    if shutil.which("nvidia-smi"):
        try:
            rows = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used", "--format=csv,noheader,nounits"],
                                  capture_output=True, text=True, timeout=15).stdout.strip().splitlines()
            out = []
            for row in rows:
                i, name, total, used = [x.strip() for x in row.split(",")]
                out.append({"index": int(i), "name": name, "total_gb": round(float(total) / 1024, 1),
                            "free_gb": round((float(total) - float(used)) / 1024, 1)})
            return out
        except Exception:  # noqa: BLE001
            pass
    import torch

    out = []
    if not torch.cuda.is_available():
        return out
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        out.append({"index": i, "name": torch.cuda.get_device_name(i), "total_gb": round(total / 2**30, 1),
                    "free_gb": round(free / 2**30, 1)})
    return out
