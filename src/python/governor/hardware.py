"""Hardware capability detection for training allocation — CPU + optional GPU/T4."""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
from typing import Any

@dataclass
class HardwareProfile:
    cpu_cores: int = 2
    ram_gb: float = 4.0
    gpu_available: bool = False
    gpu_name: str = ""
    gpu_count: int = 0
    vram_gb: float = 0.0
    cuda_available: bool = False
    is_tesla_t4: bool = False
    framework_gpu: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def training_capacity_score(self) -> float:
        score = 0.2
        score += min(0.3, self.cpu_cores / 16.0)
        score += min(0.2, self.ram_gb / 64.0)
        if self.gpu_available:
            score += 0.2
            score += min(0.3, self.vram_gb / 24.0)
        return max(0.05, min(1.0, score))

def detect_hardware() -> HardwareProfile:
    import os, platform
    cores = os.cpu_count() or 2
    ram = 4.0
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    ram = int(line.split()[1]) / 1e6
                    break
    except Exception:
        pass
    gpu_ok, name, vram, cuda, count = False, "", 0.0, False, 0
    fw: dict = {}
    try:
        import torch
        fw["torch"] = True
        if torch.cuda.is_available():
            gpu_ok = True
            cuda = True
            count = int(torch.cuda.device_count())
            name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            vram = float(getattr(props, "total_memory", 0) / (1024 ** 3))
    except Exception:
        fw["torch"] = False
    if not gpu_ok:
        try:
            import subprocess
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            if r.returncode == 0 and r.stdout.strip():
                lines = [ln for ln in r.stdout.strip().split("\n") if ln.strip()]
                count = len(lines)
                parts = lines[0].split(",")
                name = parts[0].strip()
                vram = float(parts[1].strip()) / 1024.0 if len(parts) > 1 else 0.0
                gpu_ok, cuda = True, True
        except Exception:
            pass
    for lib in ("lightgbm", "xgboost", "catboost"):
        try:
            __import__(lib)
            fw[lib] = True
        except Exception:
            fw[lib] = False
    is_t4 = "t4" in name.lower() or "tesla t4" in name.lower()
    notes = []
    notes.append("GPU_AVAILABLE=false; CPU fallback" if not gpu_ok else f"GPU={name}; VRAM_GB={vram:.1f}")
    notes.append(f"platform={platform.platform()}")
    return HardwareProfile(
        cpu_cores=cores, ram_gb=round(ram, 2), gpu_available=gpu_ok, gpu_name=name,
        gpu_count=count, vram_gb=round(vram, 2), cuda_available=cuda, is_tesla_t4=is_t4,
        framework_gpu=fw, notes=notes,
    )

def mock_tesla_t4() -> HardwareProfile:
    return HardwareProfile(
        cpu_cores=4, ram_gb=25.0, gpu_available=True, gpu_name="Tesla T4", gpu_count=1,
        vram_gb=16.0, cuda_available=True, is_tesla_t4=True,
        framework_gpu={"torch": True, "lightgbm": True, "xgboost": True, "catboost": False},
        notes=["mock_tesla_t4"],
    )
