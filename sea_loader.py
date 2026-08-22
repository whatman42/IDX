# sea_loader.py
"""
Dynamic Module Loader for S.E.A.
Tidak ada hardcode. Semua kode modul (risk, features, dll.) di-generate oleh S.E.A. saat runtime.
"""
import os
import sys
import importlib.util
from typing import Any, Dict, Optional

_SEA_AGENT = None
_MODULE_CACHE = {}

def set_agent(agent):
    global _SEA_AGENT
    _SEA_AGENT = agent

def _get_module_code_from_sea(module_name: str) -> str:
    """Minta S.E.A. untuk menghasilkan kode modul (JIT)."""
    if _SEA_AGENT is None:
        raise RuntimeError("S.E.A. Agent not initialized!")
    
    # 1. Cek memory S.E.A. (apakah sudah pernah dibuat sebelumnya?)
    code = _SEA_AGENT.memory.get(f"module_code_{module_name}")
    if code:
        return code
    
    # 2. Jika belum ada, suruh S.E.A. menulis dari nol (Zero-shot generation)
    print(f"🧠 [SEA_LOADER] Generating module '{module_name}' from scratch using AI...")
    
    # Panggil metode AI di sea.py untuk menulis kode
    code = _SEA_AGENT.generate_module_code(module_name)
    
    # Simpan di memory agar tidak perlu generate ulang tiap restart
    _SEA_AGENT.memory.set(f"module_code_{module_name}", code)
    return code

def load_module_dynamically(module_name: str):
    """Load modul secara dinamis dari kode yang di-generate S.E.A."""
    if module_name in _MODULE_CACHE:
        return _MODULE_CACHE[module_name]
    
    code = _get_module_code_from_sea(module_name)
    
    # Buat module object dari string kode
    spec = importlib.util.spec_from_loader(module_name, loader=None)
    module = importlib.util.module_from_spec(spec)
    exec(code, module.__dict__)
    
    # Simpan di cache runtime
    _MODULE_CACHE[module_name] = module
    
    # Inject ke sys.modules agar import biasa bekerja
    sys.modules[module_name] = module
    return module

# Fungsi proxy untuk menggantikan 'from risk import UnifiedRiskEngine'
def __getattr__(name):
    # Ini akan dipanggil saat ada yang import * dari sea_loader
    # Misal: from sea_loader import UnifiedRiskEngine
    # Kita cari di semua module yang sudah di-load
    for mod_name, mod in _MODULE_CACHE.items():
        if hasattr(mod, name):
            return getattr(mod, name)
    raise AttributeError(f"Module/Attribute '{name}' not found in S.E.A. generated modules")