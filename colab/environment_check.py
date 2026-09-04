"""Environment / GPU detection for Colab training center."""
from __future__ import annotations
import platform
import sys
from dataclasses import asdict, dataclass, field
from typing import Any

@dataclass
class EnvReport:
    python_version: str = ""
    platform: str = ""
    cpu_count: int = 0
    ram_gb: float = 0.0
    gpu_available: bool = False
    gpu_name: str = ""
    gpu_vram_gb: float = 0.0
    cuda_available: bool = False
    libraries: dict = field(default_factory=dict)
    status: str = "UNKNOWN"
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

def _ram_gb() -> float:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / 1e6
    except Exception:
        pass
    return 0.0

def _lib_versions() -> dict[str, str]:
    out = {}
    for name in ("numpy", "pandas", "polars", "sklearn", "lightgbm", "httpx"):
        try:
            mod = __import__(name if name != "sklearn" else "sklearn")
            out[name] = getattr(mod, "__version__", "?")
        except Exception:
            out[name] = "missing"
    return out

def detect_gpu() -> tuple:
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            vram = getattr(props, "total_memory", 0) / (1024 ** 3)
            return True, name, float(vram), True
    except Exception:
        pass
    try:
        import subprocess
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0 and r.stdout.strip():
            parts = r.stdout.strip().split("\n")[0].split(",")
            name = parts[0].strip()
            vram = float(parts[1].strip()) / 1024.0 if len(parts) > 1 else 0.0
            return True, name, vram, True
    except Exception:
        pass
    return False, "", 0.0, False

def check_environment() -> EnvReport:
    import os
    gpu_ok, gpu_name, vram, cuda = detect_gpu()
    rep = EnvReport(
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        cpu_count=os.cpu_count() or 0,
        ram_gb=round(_ram_gb(), 2),
        gpu_available=gpu_ok,
        gpu_name=gpu_name,
        gpu_vram_gb=round(vram, 2),
        cuda_available=cuda,
        libraries=_lib_versions(),
    )
    missing = [k for k, v in rep.libraries.items() if v == "missing"]
    rep.status = "DEGRADED" if missing else "PASS"
    if missing:
        rep.notes.append(f"missing_libs:{','.join(missing)}")
    rep.notes.append(f"GPU_AVAILABLE=true; {gpu_name}" if gpu_ok else "GPU_AVAILABLE=false; CPU fallback active")
    return rep
