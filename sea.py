#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
S.E.A. v8.0 "Transactional Evolution"
Self-Evolving Agent — Full Authority, Safe Mutation, Contract Validation

KEY IMPROVEMENTS:
- CandidateWorkspace: all mutations isolated, atomic commit/rollback
- Two-phase module generation: generate → validate → store (only if valid)
- Contract testing for loaded modules
- Global deadline (GOD_TIMEOUT) for entire run
- Pipeline state machine in main.py
- Cleanup never touches production files without explicit dry-run
- Data contract normalization (fix rows=0 issue)
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import logging
import os
import platform
import re
import shutil
import sqlite3
import subprocess
import sys
import time
import traceback
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# -----------------------------------------------------------------------------
# Dependency checks
# -----------------------------------------------------------------------------
try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    from ddgs import DDGS
    HAS_DDGS = True
except ImportError:
    HAS_DDGS = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import astor
    HAS_ASTOR = True
except ImportError:
    HAS_ASTOR = False

try:
    import resource
    HAS_RESOURCE = True
except ImportError:
    HAS_RESOURCE = False

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
VERSION = "8.0-Transactional"
AGENT_NAME = "S.E.A."
CORE_FILENAME = "sea.py"
GENERATION_PREFIX = "S"
DEFAULT_TIMEOUT = 300
DEFAULT_MAX_ITERATIONS = 5
DEFAULT_SUBPROCESS_TIMEOUT = 60
MEMORY_DB = ".sea_memory.db"
MEMORY_JSON = ".sea_memory.json"
CANDIDATE_DIR = ".sea_candidates"          # legacy, replaced by workspace
WORKSPACE_ROOT = "candidate_workspace"      # new isolated workspace
LOG_DIR = ".sea_logs"
GIT_MODE_DEFAULT = "disabled"
GENOMIC_VAULT_FILE = ".sea_genomic_vault.jsonl"

PRIMARY_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
FALLBACK_MODEL = os.environ.get("GEMINI_FALLBACK_MODEL", "gemini-3.5-flash-lite")
MODEL_DISCOVERY = os.environ.get("MODEL_DISCOVERY", "true").lower() == "true"

PROTECTED_FILES = {"sea.py", "main.py", "sea_loader.py", "logger.py", "data.py"}

# -----------------------------------------------------------------------------
# Enums & dataclasses
# -----------------------------------------------------------------------------
class Event(str, Enum):
    ENVIRONMENT_DISCOVERED = "ENVIRONMENT_DISCOVERED"
    CAPABILITY_DISCOVERED = "CAPABILITY_DISCOVERED"
    REPOSITORY_DISCOVERED = "REPOSITORY_DISCOVERED"
    RESEARCH_STARTED = "RESEARCH_STARTED"
    RESEARCH_COMPLETED = "RESEARCH_COMPLETED"
    GOAL_RECEIVED = "GOAL_RECEIVED"
    PLAN_CREATED = "PLAN_CREATED"
    ACTION_STARTED = "ACTION_STARTED"
    ACTION_COMPLETED = "ACTION_COMPLETED"
    ACTION_FAILED = "ACTION_FAILED"
    EXPERIMENT_STARTED = "EXPERIMENT_STARTED"
    EXPERIMENT_COMPLETED = "EXPERIMENT_COMPLETED"
    EVOLUTION_STARTED = "EVOLUTION_STARTED"
    CANDIDATE_CREATED = "CANDIDATE_CREATED"
    VALIDATION_STARTED = "VALIDATION_STARTED"
    VALIDATION_FAILED = "VALIDATION_FAILED"
    VALIDATION_SUCCESS = "VALIDATION_SUCCESS"
    GENERATION_PROMOTED = "GENERATION_PROMOTED"
    GENERATION_ROLLED_BACK = "GENERATION_ROLLED_BACK"
    RECOVERY_STARTED = "RECOVERY_STARTED"
    RECOVERY_COMPLETED = "RECOVERY_COMPLETED"
    BOOT_COMPLETE = "BOOT_COMPLETE"
    LOOP_ITERATION = "LOOP_ITERATION"
    MEMORY_INITIALIZED = "MEMORY_INITIALIZED"
    SECURITY_AUDIT_COMPLETED = "SECURITY_AUDIT_COMPLETED"
    CHANGE_IMPACT_ANALYSIS = "CHANGE_IMPACT_ANALYSIS"
    BENCHMARK_COMPLETED = "BENCHMARK_COMPLETED"
    FAILURE_RECORDED = "FAILURE_RECORDED"
    CHECKPOINT_CREATED = "CHECKPOINT_CREATED"
    LINEAGE_UPDATED = "LINEAGE_UPDATED"
    SELF_HEALING_STARTED = "SELF_HEALING_STARTED"
    SELF_HEALING_COMPLETED = "SELF_HEALING_COMPLETED"
    PROBLEM_SOLVED = "PROBLEM_SOLVED"
    BRUTAL_REJECT = "BRUTAL_REJECT"
    BRUTAL_PROMOTE = "BRUTAL_PROMOTE"
    FITNESS_EVALUATION = "FITNESS_EVALUATION"
    MODULE_GENERATED = "MODULE_GENERATED"
    MODULE_EVOLVED = "MODULE_EVOLVED"
    FILE_DELETED = "FILE_DELETED"
    FILE_CREATED = "FILE_CREATED"
    WORKFLOW_REFACTORED = "WORKFLOW_REFACTORED"
    CLEANUP_COMPLETED = "CLEANUP_COMPLETED"
    WORKSPACE_CREATED = "WORKSPACE_CREATED"
    WORKSPACE_COMMITTED = "WORKSPACE_COMMITTED"
    WORKSPACE_ROLLED_BACK = "WORKSPACE_ROLLED_BACK"
    MODULE_REJECTED = "MODULE_REJECTED"
    DEADLINE_EXCEEDED = "DEADLINE_EXCEEDED"

class GenerationStatus(str, Enum):
    ACTIVE = "ACTIVE"
    CANDIDATE = "CANDIDATE"
    FAILED = "FAILED"
    PROMOTED = "PROMOTED"
    ROLLED_BACK = "ROLLED_BACK"
    SUPERSEDED = "SUPERSEDED"

@dataclass
class Generation:
    id: str
    parent: Optional[str]
    timestamp: str
    objective: str
    source_hash: str
    changes: List[str] = field(default_factory=list)
    test_results: Dict[str, Any] = field(default_factory=dict)
    evaluation: Dict[str, Any] = field(default_factory=dict)
    status: GenerationStatus = GenerationStatus.ACTIVE
    fitness_score: Optional[float] = None

    @staticmethod
    def next_id(existing: List[str]) -> str:
        nums = []
        for e in existing:
            m = re.match(rf"{GENERATION_PREFIX}(\d+)", e)
            if m:
                nums.append(int(m.group(1)))
        n = max(nums) + 1 if nums else 1
        return f"{GENERATION_PREFIX}{n:06d}"

# -----------------------------------------------------------------------------
# Logger
# -----------------------------------------------------------------------------
class StructuredLogger:
    def __init__(self, name: str = AGENT_NAME, log_dir: Optional[Path] = None):
        self.name = name
        self.log_dir = Path(log_dir or LOG_DIR)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        if not self.logger.handlers:
            fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s",
                                    datefmt="%Y-%m-%dT%H:%M:%SZ")
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(logging.INFO)
            ch.setFormatter(fmt)
            self.logger.addHandler(ch)
            fh = logging.FileHandler(self.log_dir / "agent.log", encoding="utf-8")
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(fmt)
            self.logger.addHandler(fh)

    def event(self, event: Event, **payload: Any) -> None:
        safe = {k: v for k, v in payload.items() if not self._is_sensitive(k, v)}
        self.logger.info(f"[{event.value}] {json.dumps(safe, default=str, ensure_ascii=False)}")

    def info(self, msg: str, **kwargs) -> None: self.logger.info(msg, **kwargs)
    def warning(self, msg: str, **kwargs) -> None: self.logger.warning(msg, **kwargs)
    def error(self, msg: str, **kwargs) -> None: self.logger.error(msg, **kwargs)
    def debug(self, msg: str, **kwargs) -> None: self.logger.debug(msg, **kwargs)

    @staticmethod
    def _is_sensitive(key: str, value: Any) -> bool:
        key_l = key.lower()
        sensitive = ("key", "token", "secret", "password", "credential", "cookie",
                     "auth", "private", "api_key", "access_token", "telegram", "chat_id")
        if any(s in key_l for s in sensitive):
            return True
        if isinstance(value, str) and len(value) > 20 and re.search(r"(sk-|AIza|ghp_|gho_|xoxb-|Bearer\s)", value):
            return True
        return False

log = StructuredLogger()

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def safe_run(cmd: List[str], cwd: Optional[Path] = None, timeout: int = DEFAULT_SUBPROCESS_TIMEOUT,
             env: Optional[Dict[str, str]] = None, input_text: Optional[str] = None) -> Tuple[int, str, str]:
    try:
        result = subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True,
                                timeout=timeout, env=env, input=input_text)
        return result.returncode, result.stdout or "", result.stderr or ""
    except subprocess.TimeoutExpired:
        return -1, "", "TIMEOUT"
    except FileNotFoundError:
        return -2, "", "EXECUTABLE_NOT_FOUND"
    except Exception as e:
        return -3, "", str(e)

def is_secret_env(key: str) -> bool:
    key_l = key.lower()
    return any(s in key_l for s in ("key", "token", "secret", "password", "credential", "cookie", "auth", "private", "api_key", "access_token", "ssh", "bank", "wallet"))

# -----------------------------------------------------------------------------
# Memory (extended with module_code table)
# -----------------------------------------------------------------------------
class Memory:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.db_path = base_dir / MEMORY_DB
        self.json_path = base_dir / MEMORY_JSON
        self.conn: Optional[sqlite3.Connection] = None
        self.backend = "none"
        self._init()

    def _init(self) -> None:
        try:
            self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            self._create_tables()
            self.backend = "sqlite"
            log.event(Event.MEMORY_INITIALIZED, backend="sqlite", path=str(self.db_path))
        except Exception as e:
            log.warning(f"SQLite unavailable ({e}), falling back to JSON")
            self.backend = "json"
            if not self.json_path.exists():
                self._write_json({})
            log.event(Event.MEMORY_INITIALIZED, backend="json", path=str(self.json_path))

    def _create_tables(self) -> None:
        assert self.conn
        cur = self.conn.cursor()
        cur.executescript("""
            CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT, updated_at TEXT);
            CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY AUTOINCREMENT, event TEXT, payload TEXT, created_at TEXT);
            CREATE TABLE IF NOT EXISTS generations (id TEXT PRIMARY KEY, parent TEXT, timestamp TEXT, objective TEXT, source_hash TEXT, changes TEXT, test_results TEXT, evaluation TEXT, status TEXT, fitness_score REAL);
            CREATE TABLE IF NOT EXISTS experiments (id TEXT PRIMARY KEY, hypothesis TEXT, change_desc TEXT, result TEXT, metrics TEXT, conclusion TEXT, created_at TEXT);
            CREATE TABLE IF NOT EXISTS observations (id INTEGER PRIMARY KEY AUTOINCREMENT, category TEXT, content TEXT, created_at TEXT);
            CREATE TABLE IF NOT EXISTS lineage (id TEXT PRIMARY KEY, parent TEXT, child TEXT, evidence TEXT, experiment TEXT, score REAL, decision TEXT, rollback_reason TEXT, created_at TEXT);
            CREATE TABLE IF NOT EXISTS evidence (id TEXT PRIMARY KEY, query TEXT, source TEXT, url TEXT, title TEXT, snippet TEXT, content_hash TEXT, evidence_type TEXT, confidence REAL, retrieved_at TEXT, contradiction TEXT);
            CREATE TABLE IF NOT EXISTS failures (id INTEGER PRIMARY KEY AUTOINCREMENT, category TEXT, context TEXT, error_type TEXT, stacktrace TEXT, created_at TEXT);
            CREATE TABLE IF NOT EXISTS checkpoints (id TEXT PRIMARY KEY, stage TEXT, generation_id TEXT, timestamp TEXT, state TEXT, active INTEGER DEFAULT 1);
            CREATE TABLE IF NOT EXISTS blacklist (pattern_hash TEXT PRIMARY KEY, reason TEXT, created_at TEXT);
            CREATE TABLE IF NOT EXISTS module_code (module_name TEXT PRIMARY KEY, code TEXT, hash TEXT, created_at TEXT, updated_at TEXT, valid INTEGER DEFAULT 0);
        """)
        self.conn.commit()

    def set(self, key: str, value: Any) -> None:
        payload = json.dumps(value, default=str)
        if self.backend == "sqlite" and self.conn:
            self.conn.execute("INSERT OR REPLACE INTO kv (key, value, updated_at) VALUES (?, ?, ?)", (key, payload, utc_now()))
            self.conn.commit()
        else:
            data = self._read_json()
            data[key] = {"value": value, "updated_at": utc_now()}
            self._write_json(data)

    def get(self, key: str, default: Any = None) -> Any:
        if self.backend == "sqlite" and self.conn:
            row = self.conn.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
            if row:
                return json.loads(row["value"])
            return default
        data = self._read_json()
        entry = data.get(key)
        return entry.get("value", default) if entry else default

    def log_event(self, event: str, payload: Dict[str, Any]) -> None:
        if self.backend == "sqlite" and self.conn:
            self.conn.execute("INSERT INTO events (event, payload, created_at) VALUES (?, ?, ?)",
                              (event, json.dumps(payload, default=str), utc_now()))
            self.conn.commit()

    def save_generation(self, gen: Generation) -> None:
        if self.backend == "sqlite" and self.conn:
            self.conn.execute("""INSERT OR REPLACE INTO generations
                (id, parent, timestamp, objective, source_hash, changes, test_results, evaluation, status, fitness_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (gen.id, gen.parent, gen.timestamp, gen.objective, gen.source_hash,
                 json.dumps(gen.changes, default=str), json.dumps(gen.test_results, default=str),
                 json.dumps(gen.evaluation, default=str), gen.status.value, gen.fitness_score))
            self.conn.commit()
        else:
            data = self._read_json()
            gens = data.get("generations", {})
            gen_dict = asdict(gen)
            gen_dict["status"] = gen.status.value
            gens[gen.id] = gen_dict
            data["generations"] = gens
            self._write_json(data)

    def list_generations(self) -> List[Dict[str, Any]]:
        if self.backend == "sqlite" and self.conn:
            rows = self.conn.execute("SELECT * FROM generations ORDER BY timestamp DESC").fetchall()
            return [dict(r) for r in rows]
        data = self._read_json()
        return list(data.get("generations", {}).values())

    def get_active_generation(self) -> Optional[Dict[str, Any]]:
        for g in self.list_generations():
            if g.get("status") == GenerationStatus.ACTIVE.value:
                return g
        return None

    def add_observation(self, category: str, content: Any) -> None:
        if self.backend == "sqlite" and self.conn:
            self.conn.execute("INSERT INTO observations (category, content, created_at) VALUES (?, ?, ?)",
                              (category, json.dumps(content, default=str), utc_now()))
            self.conn.commit()

    def save_lineage(self, lineage: Dict[str, Any]) -> None:
        if self.backend == "sqlite" and self.conn:
            self.conn.execute("""INSERT OR REPLACE INTO lineage
                (id, parent, child, evidence, experiment, score, decision, rollback_reason, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (lineage.get("id", uuid.uuid4().hex), lineage.get("parent"), lineage.get("child"),
                 json.dumps(lineage.get("evidence", [])), lineage.get("experiment"),
                 lineage.get("score", 0.0), lineage.get("decision", "unknown"),
                 lineage.get("rollback_reason", ""), utc_now()))
            self.conn.commit()

    def save_evidence(self, evidence: Dict[str, Any]) -> None:
        if self.backend == "sqlite" and self.conn:
            self.conn.execute("""INSERT OR REPLACE INTO evidence
                (id, query, source, url, title, snippet, content_hash, evidence_type, confidence, retrieved_at, contradiction)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (evidence.get("id", uuid.uuid4().hex), evidence.get("query", ""), evidence.get("source", ""),
                 evidence.get("url", ""), evidence.get("title", ""), evidence.get("snippet", ""),
                 evidence.get("content_hash", ""), evidence.get("evidence_type", "UNKNOWN"),
                 evidence.get("confidence", 0.0), evidence.get("retrieved_at", utc_now()),
                 evidence.get("contradiction", "")))
            self.conn.commit()

    def save_failure(self, failure: Dict[str, Any]) -> None:
        if self.backend == "sqlite" and self.conn:
            self.conn.execute("INSERT INTO failures (category, context, error_type, stacktrace, created_at) VALUES (?, ?, ?, ?, ?)",
                              (failure.get("category", "UNKNOWN"), failure.get("context", ""),
                               failure.get("error_type", ""), failure.get("stacktrace", ""), utc_now()))
            self.conn.commit()

    def save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if self.backend == "sqlite" and self.conn:
            self.conn.execute("INSERT OR REPLACE INTO checkpoints (id, stage, generation_id, timestamp, state, active) VALUES (?, ?, ?, ?, ?, 1)",
                              (checkpoint.get("id", uuid.uuid4().hex), checkpoint.get("stage", "unknown"),
                               checkpoint.get("generation_id"), utc_now(), json.dumps(checkpoint.get("state", {}))))
            self.conn.commit()

    def get_failures(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        if self.backend == "sqlite" and self.conn:
            if category:
                rows = self.conn.execute("SELECT * FROM failures WHERE category = ? ORDER BY id DESC LIMIT 100", (category,)).fetchall()
            else:
                rows = self.conn.execute("SELECT * FROM failures ORDER BY id DESC LIMIT 100").fetchall()
            return [dict(r) for r in rows]
        return []

    def add_blacklist(self, pattern_hash: str, reason: str) -> None:
        if self.backend == "sqlite" and self.conn:
            self.conn.execute("INSERT OR REPLACE INTO blacklist (pattern_hash, reason, created_at) VALUES (?, ?, ?)",
                              (pattern_hash, reason, utc_now()))
            self.conn.commit()
        else:
            data = self._read_json()
            bl = data.get("blacklist", [])
            bl.append({"pattern_hash": pattern_hash, "reason": reason, "created_at": utc_now()})
            data["blacklist"] = bl
            self._write_json(data)

    def get_blacklist(self) -> Set[str]:
        if self.backend == "sqlite" and self.conn:
            rows = self.conn.execute("SELECT pattern_hash FROM blacklist").fetchall()
            return {r["pattern_hash"] for r in rows}
        data = self._read_json()
        return {b["pattern_hash"] for b in data.get("blacklist", [])}

    def set_module_code(self, module_name: str, code: Optional[str], valid: bool = True) -> None:
        if code is None:
            if self.backend == "sqlite" and self.conn:
                self.conn.execute("DELETE FROM module_code WHERE module_name = ?", (module_name,))
                self.conn.commit()
            else:
                data = self._read_json()
                if "module_code" in data and module_name in data["module_code"]:
                    del data["module_code"][module_name]
                    self._write_json(data)
            return
        h = sha256_text(code)
        valid_flag = 1 if valid else 0
        if self.backend == "sqlite" and self.conn:
            self.conn.execute("""INSERT OR REPLACE INTO module_code
                (module_name, code, hash, created_at, updated_at, valid)
                VALUES (?, ?, ?, COALESCE((SELECT created_at FROM module_code WHERE module_name=?), ?), ?, ?)""",
                (module_name, code, h, module_name, utc_now(), utc_now(), valid_flag))
            self.conn.commit()
        else:
            data = self._read_json()
            mc = data.get("module_code", {})
            old = mc.get(module_name, {})
            mc[module_name] = {"code": code, "hash": h, "created_at": old.get("created_at", utc_now()),
                               "updated_at": utc_now(), "valid": valid_flag}
            data["module_code"] = mc
            self._write_json(data)

    def get_module_code(self, module_name: str) -> Optional[str]:
        if self.backend == "sqlite" and self.conn:
            row = self.conn.execute("SELECT code, valid FROM module_code WHERE module_name = ?", (module_name,)).fetchone()
            if row and row["valid"]:
                return row["code"]
            return None
        data = self._read_json()
        entry = data.get("module_code", {}).get(module_name)
        if entry and entry.get("valid", False):
            return entry.get("code")
        return None

    def is_module_valid(self, module_name: str) -> bool:
        if self.backend == "sqlite" and self.conn:
            row = self.conn.execute("SELECT valid FROM module_code WHERE module_name = ?", (module_name,)).fetchone()
            return bool(row and row["valid"])
        data = self._read_json()
        entry = data.get("module_code", {}).get(module_name)
        return bool(entry and entry.get("valid", False))

    def _read_json(self) -> Dict:
        if self.json_path.exists():
            try:
                return json.loads(self.json_path.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _write_json(self, data: Dict) -> None:
        self.json_path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

    def close(self) -> None:
        if self.conn:
            self.conn.close()

# -----------------------------------------------------------------------------
# Scanners (stub)
# -----------------------------------------------------------------------------
class EnvironmentScanner:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
    def scan(self) -> Dict[str, Any]:
        return {"os": platform.system(), "cwd": str(self.base_dir), "timestamp": utc_now()}

class RepositoryScanner:
    def __init__(self, root: Path):
        self.root = root.resolve()
    def scan(self) -> Dict[str, Any]:
        files = []
        for p in self.root.rglob("*"):
            if p.is_file() and not any(x in p.parts for x in [".git", "__pycache__", ".sea_candidates", ".sea_logs", WORKSPACE_ROOT]):
                files.append(str(p))
        return {"file_count": len(files), "files": files[:200]}

class CapabilityRegistry:
    def __init__(self, env: Dict, repo: Dict):
        self.env = env; self.repo = repo
    def discover(self) -> Dict[str, Any]:
        return {"network": True, "git": False, "self_modify": True}

# -----------------------------------------------------------------------------
# Model Providers (Gemini with fallback)
# -----------------------------------------------------------------------------
class ModelProvider(ABC):
    @abstractmethod
    def name(self) -> str: ...
    @abstractmethod
    def available(self) -> bool: ...
    @abstractmethod
    def generate(self, prompt: str, system: Optional[str] = None, max_tokens: int = 2048) -> str: ...

class NullProvider(ModelProvider):
    def name(self) -> str: return "null"
    def available(self) -> bool: return True
    def generate(self, prompt, system=None, max_tokens=2048) -> str:
        return "[NULL_PROVIDER] No LLM."

class GeneticMutator(ModelProvider):
    def name(self) -> str: return "genetic_mutator"
    def available(self) -> bool: return True
    def generate(self, prompt, system=None, max_tokens=2048) -> str:
        return "[GENETIC] No LLM."
    def mutate_ast(self, source: str) -> str:
        try:
            tree = ast.parse(source)
            import random
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                    if random.random() < 0.2:
                        node.value += random.uniform(-2.0, 2.0) if isinstance(node.value, float) else random.randint(-2, 2)
                        node.value = max(0, node.value)
            if HAS_ASTOR:
                return astor.to_source(tree)
            return source + "\n# MUTATED_BY_GENETIC\n"
        except Exception:
            return source

class GeminiProvider(ModelProvider):
    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.primary = os.environ.get("GEMINI_MODEL", PRIMARY_MODEL)
        self.fallback = os.environ.get("GEMINI_FALLBACK_MODEL", FALLBACK_MODEL)
        self.model_discovery = os.environ.get("MODEL_DISCOVERY", "true").lower() == "true"
        self.available_models: List[str] = []
        self.active_model: Optional[str] = None

    def name(self) -> str: return "gemini"
    def available(self) -> bool: return bool(self.api_key) and (HAS_HTTPX or HAS_REQUESTS)

    def _discover_models(self) -> None:
        if not self.model_discovery or not self.available():
            return
        url = f"https://generativelanguage.googleapis.com/v1beta/models?key={self.api_key}"
        try:
            r = httpx.get(url, timeout=10) if HAS_HTTPX else requests.get(url, timeout=10)
            if r.status_code == 200:
                data = r.json()
                self.available_models = [m.get("name", "").split("/")[-1] for m in data.get("models", [])]
                log.info(f"🧠 [GEMINI] Discovered {len(self.available_models)} models")
        except Exception as e:
            log.warning(f"⚠️ Model discovery failed: {e}")

    def _generate_with_model(self, model: str, prompt: str, system: Optional[str], max_tokens: int) -> Tuple[Optional[str], Dict[str, Any]]:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.api_key}"
        body = {
            "contents": [{"parts": [{"text": (system + "\n\n" + prompt) if system else prompt}]}],
            "generationConfig": {"maxOutputTokens": max_tokens}
        }
        start = time.time()
        try:
            r = httpx.post(url, json=body, timeout=60) if HAS_HTTPX else requests.post(url, json=body, timeout=60)
            elapsed = time.time() - start
            if r.status_code != 200:
                return None, {"error": f"HTTP {r.status_code}", "latency_ms": int(elapsed*1000)}
            data = r.json()
            candidates = data.get("candidates", [])
            if not candidates:
                return None, {"error": "EMPTY_CANDIDATES", "latency_ms": int(elapsed*1000)}
            parts = candidates[0].get("content", {}).get("parts", [])
            if not parts:
                return None, {"error": "EMPTY_PARTS", "latency_ms": int(elapsed*1000)}
            return parts[0].get("text", ""), {"latency_ms": int(elapsed*1000), "model": model}
        except Exception as e:
            elapsed = time.time() - start
            return None, {"error": type(e).__name__, "latency_ms": int(elapsed*1000)}

    def generate(self, prompt: str, system: Optional[str] = None, max_tokens: int = 2048) -> str:
        if not self.available():
            return NullProvider().generate(prompt, system, max_tokens)
        if self.model_discovery and not self.available_models:
            self._discover_models()
        text, audit = self._generate_with_model(self.primary, prompt, system, max_tokens)
        if text is not None:
            self.active_model = self.primary
            return text
        log.warning(f"⚠️ [GEMINI] Primary failed, falling back to {self.fallback}")
        text2, audit2 = self._generate_with_model(self.fallback, prompt, system, max_tokens)
        if text2 is not None:
            self.active_model = self.fallback
            return text2
        log.error(f"❌ [GEMINI] Both models failed.")
        return f"[GEMINI_ERROR] All models failed."

class OpenAIProvider(ModelProvider):
    def __init__(self):
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self.base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    def name(self) -> str: return "openai"
    def available(self) -> bool: return bool(self.api_key) and (HAS_HTTPX or HAS_REQUESTS)
    def generate(self, prompt, system=None, max_tokens=2048) -> str:
        if not self.available():
            return NullProvider().generate(prompt, system, max_tokens)
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        messages = [{"role": "system", "content": system}] if system else []
        messages.append({"role": "user", "content": prompt})
        body = {"model": self.model, "messages": messages, "max_tokens": max_tokens}
        try:
            r = requests.post(f"{self.base}/chat/completions", json=body, headers=headers, timeout=60) if HAS_REQUESTS else httpx.post(f"{self.base}/chat/completions", json=body, headers=headers, timeout=60)
            if r.status_code != 200:
                return f"[OPENAI_ERROR] {r.status_code}"
            return r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"[OPENAI_EXCEPTION] {e}"

class GrokProvider(ModelProvider):
    def __init__(self):
        self.api_key = os.environ.get("XAI_API_KEY") or os.environ.get("GROK_API_KEY")
        self.model = os.environ.get("GROK_MODEL", "grok-3")
        self.base = "https://api.x.ai/v1"
    def name(self) -> str: return "grok"
    def available(self) -> bool: return bool(self.api_key) and (HAS_HTTPX or HAS_REQUESTS)
    def generate(self, prompt, system=None, max_tokens=2048) -> str:
        if not self.available():
            return NullProvider().generate(prompt, system, max_tokens)
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        messages = [{"role": "system", "content": system}] if system else []
        messages.append({"role": "user", "content": prompt})
        body = {"model": self.model, "messages": messages, "max_tokens": max_tokens}
        try:
            r = requests.post(f"{self.base}/chat/completions", json=body, headers=headers, timeout=60) if HAS_REQUESTS else httpx.post(f"{self.base}/chat/completions", json=body, headers=headers, timeout=60)
            if r.status_code != 200:
                return f"[GROK_ERROR] {r.status_code}"
            return r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"[GROK_EXCEPTION] {e}"

def select_model_provider() -> ModelProvider:
    gemini = GeminiProvider()
    if gemini.available():
        log.info(f"🧠 Selected model provider: {gemini.name()} (primary={gemini.primary}, fallback={gemini.fallback})")
        return gemini
    for cls in (OpenAIProvider, GrokProvider):
        p = cls()
        if p.available():
            log.info(f"🧠 Selected model provider: {p.name()}")
            return p
    log.info("⚙️ No LLM available. Using GeneticMutator (offline).")
    return GeneticMutator()

# -----------------------------------------------------------------------------
# Research Engine (stub)
# -----------------------------------------------------------------------------
class AdaptiveResearchEngine:
    def __init__(self, network: bool, memory: Memory, model: Optional[ModelProvider] = None):
        self.network = network
        self.memory = memory
        self.model = model
        self.functional = False

    def research(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        log.event(Event.RESEARCH_STARTED, query=query)
        if not self.network:
            return []
        results = []
        if HAS_DDGS:
            try:
                ddgs = DDGS()
                for r in ddgs.text(query, max_results=max_results):
                    results.append({"title": r.get("title"), "url": r.get("href"), "snippet": r.get("body"), "source": "ddgs"})
            except Exception: pass
        if not results:
            try:
                url = "https://en.wikipedia.org/w/api.php"
                params = {"action": "query", "list": "search", "srsearch": query, "format": "json", "srlimit": max_results}
                r = requests.get(url, params=params, timeout=5) if HAS_REQUESTS else httpx.get(url, params=params, timeout=5)
                data = r.json()
                for item in data.get("query", {}).get("search", []):
                    results.append({"title": item["title"], "url": f"https://en.wikipedia.org/wiki/{item['title'].replace(' ', '_')}", "snippet": item.get("snippet"), "source": "wikipedia"})
            except Exception: pass
        self.functional = len(results) > 0
        log.event(Event.RESEARCH_COMPLETED, query=query, results=len(results))
        return results

# -----------------------------------------------------------------------------
# Validation & Benchmark
# -----------------------------------------------------------------------------
class ValidationEngine:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir

    def verify(self, path: Path) -> Dict[str, Any]:
        results = {"checks": {}}
        try:
            ast.parse(path.read_text(encoding="utf-8"))
            results["checks"]["ast"] = {"ok": True}
        except SyntaxError as e:
            results["checks"]["ast"] = {"ok": False, "error": str(e)}
            results["overall"] = False
            return results
        code, _, err = safe_run([sys.executable, "-m", "py_compile", str(path)])
        results["checks"]["compile"] = {"ok": code == 0, "stderr": err[:300]}
        code, out, err = safe_run([sys.executable, str(path), "--test-boot"], timeout=120, cwd=self.base_dir)
        results["checks"]["boot"] = {"ok": code == 0, "stdout": out[-500:], "stderr": err[-300:]}
        results["overall"] = all(c.get("ok", False) for c in results["checks"].values())
        return results

class BenchmarkEngine:
    def compute_score(self, verification: Dict[str, Any], additional: Optional[Dict] = None) -> float:
        score = 0.0
        checks = verification.get("checks", {})
        if checks.get("ast", {}).get("ok"): score += 20
        if checks.get("compile", {}).get("ok"): score += 30
        if checks.get("boot", {}).get("ok"): score += 50
        return score

# -----------------------------------------------------------------------------
# Git Manager
# -----------------------------------------------------------------------------
class GitManager:
    def __init__(self, root: Path, mode: str = GIT_MODE_DEFAULT):
        self.root = root
        self.mode = mode
        self.available = shutil.which("git") is not None

    def commit_if_enabled(self, msg: str) -> bool:
        if self.mode == "disabled" or not self.available:
            return False
        safe_run(["git", "add", "-A"], cwd=self.root)
        code, _, err = safe_run(["git", "commit", "-m", msg], cwd=self.root)
        if code != 0:
            return False
        if self.mode in ("push", "autonomous"):
            safe_run(["git", "push"], cwd=self.root, timeout=30)
        return True

# -----------------------------------------------------------------------------
# Candidate Workspace (Transactional Evolution)
# -----------------------------------------------------------------------------
class CandidateWorkspace:
    """Isolated staging area for safe mutations. All changes happen here."""
    def __init__(self, base_dir: Path, workspace_id: str):
        self.base_dir = base_dir
        self.workspace_dir = base_dir / WORKSPACE_ROOT / workspace_id
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.production_files: Dict[Path, Path] = {}  # production → workspace
        self.modules_written: Set[str] = set()
        self.log = log

    def stage_file(self, production_path: Path) -> Path:
        """Copy a production file into the workspace."""
        if not production_path.exists():
            raise FileNotFoundError(f"Production file not found: {production_path}")
        ws_path = self.workspace_dir / production_path.name
        shutil.copy2(production_path, ws_path)
        self.production_files[production_path] = ws_path
        return ws_path

    def write_module(self, module_name: str, code: str) -> Path:
        """Write generated module code to workspace (not production)."""
        target = self.workspace_dir / f"{module_name}.py"
        target.write_text(code, encoding="utf-8")
        self.modules_written.add(module_name)
        # Also record as a staged file
        self.production_files[self.base_dir / f"{module_name}.py"] = target
        return target

    def validate_workspace(self, agent) -> bool:
        """Run syntax + import + contract checks on all staged modules."""
        for prod_path, ws_path in self.production_files.items():
            # Validate Python syntax
            try:
                ast.parse(ws_path.read_text(encoding="utf-8"))
            except SyntaxError as e:
                self.log.error(f"❌ Workspace syntax error in {ws_path.name}: {e}")
                return False
            # Try compile
            try:
                compile(ws_path.read_text(encoding="utf-8"), str(ws_path), "exec")
            except Exception as e:
                self.log.error(f"❌ Workspace compile error in {ws_path.name}: {e}")
                return False
            # Contract check if it's a module we know
            module_name = ws_path.stem
            if module_name in agent.module_contracts:
                if not agent.validate_module_contract(module_name, agent.module_contracts[module_name]):
                    return False
        return True

    def commit(self) -> bool:
        """Atomic replace: copy all workspace files to production."""
        for prod_path, ws_path in self.production_files.items():
            # Backup production
            backup = prod_path.with_suffix(".py.bak")
            shutil.copy2(prod_path, backup)
            # Replace
            shutil.copy2(ws_path, prod_path)
        # Clean up workspace
        shutil.rmtree(self.workspace_dir, ignore_errors=True)
        return True

    def rollback(self) -> bool:
        """Discard workspace, keep production untouched."""
        shutil.rmtree(self.workspace_dir, ignore_errors=True)
        return True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.rollback()
        else:
            # If no exception, caller decides commit/rollback
            pass

# -----------------------------------------------------------------------------
# Evolution Engine (for sea.py itself) — uses workspace
# -----------------------------------------------------------------------------
class EvolutionEngine:
    def __init__(self, core_path: Path, memory: Memory, validator: ValidationEngine,
                 git: GitManager, base_dir: Path, model_provider: Optional[ModelProvider] = None,
                 research: Optional[AdaptiveResearchEngine] = None, benchmark: Optional[BenchmarkEngine] = None,
                 brutal_mode: bool = True):
        self.core_path = core_path
        self.memory = memory
        self.validator = validator
        self.git = git
        self.base_dir = base_dir
        self.model = model_provider or GeneticMutator()
        self.research = research
        self.benchmark = benchmark or BenchmarkEngine()
        self.brutal_mode = brutal_mode
        self.blacklist = memory.get_blacklist()

    def create_candidate_in_workspace(self, objective: str, workspace: CandidateWorkspace) -> Tuple[str, Path]:
        """Create a candidate sea.py inside the workspace."""
        log.event(Event.EVOLUTION_STARTED, objective=objective)
        existing = [g.get("id", "") for g in self.memory.list_generations()]
        new_id = Generation.next_id(existing)
        # Stage current sea.py into workspace
        ws_core = workspace.stage_file(self.core_path)
        changes = ["duplicated"]
        if isinstance(self.model, GeneticMutator):
            code = ws_core.read_text()
            mutated = self.model.mutate_ast(code)
            ws_core.write_text(mutated)
            changes.append("ast_mutated")
        else:
            # LLM-based enhancement
            prompt = f"Enhance this sea.py code for objective: {objective}. Return ONLY Python code."
            code = self.model.generate(prompt, max_tokens=2000)
            code = re.sub(r'```python|```', '', code).strip()
            if code and not code.startswith("[") and not code.startswith("```"):
                with open(ws_core, "a") as f:
                    f.write("\n# === LLM ENHANCEMENT ===\n")
                    f.write(code)
                    f.write("\n# === END ===\n")
                changes.append("llm_enhanced")
        # Record generation in memory
        gen = Generation(id=new_id, parent=(self.memory.get_active_generation() or {}).get("id"),
                         timestamp=utc_now(), objective=objective,
                         source_hash=sha256_file(ws_core), changes=changes,
                         status=GenerationStatus.CANDIDATE)
        self.memory.save_generation(gen)
        log.event(Event.CANDIDATE_CREATED, generation=new_id, path=str(ws_core))
        return new_id, ws_core

    def evaluate_fitness(self, path: Path) -> Dict[str, Any]:
        verification = self.validator.verify(path)
        if not verification.get("overall"):
            return {"fitness": -1.0, "verification": verification, "reason": "VERIFICATION_FAILED"}
        score = self.benchmark.compute_score(verification)
        return {"fitness": score, "verification": verification}

    def promote(self, gen_id: str, cand_path: Path, fitness_data: Dict[str, Any]) -> bool:
        # For sea.py evolution, promotion is atomic replace
        active = self.memory.get_active_generation()
        active_fitness = active.get("fitness_score", 0.0) if active else 0.0
        cand_fitness = fitness_data.get("fitness", -1.0)
        if cand_fitness < 0:
            return False
        import random
        improved = cand_fitness > active_fitness * 1.01
        if not improved and random.random() < 0.1:
            if cand_fitness > active_fitness * 0.95:
                improved = True
        if self.brutal_mode and not improved:
            return False
        backup = self.core_path.with_suffix(".py.bak")
        shutil.copy2(self.core_path, backup)
        try:
            shutil.copy2(cand_path, self.core_path)
            for g in self.memory.list_generations():
                if g.get("status") == GenerationStatus.ACTIVE.value and g.get("id") != gen_id:
                    old = Generation(**{**g, "status": GenerationStatus.SUPERSEDED})
                    self.memory.save_generation(old)
            gen_obj = Generation(id=gen_id, parent=active.get("id") if active else None,
                                 timestamp=utc_now(), objective="promoted",
                                 source_hash=sha256_file(self.core_path), changes=["promoted"],
                                 test_results=fitness_data.get("verification", {}),
                                 evaluation={"backup": str(backup), "fitness": cand_fitness},
                                 status=GenerationStatus.ACTIVE, fitness_score=cand_fitness)
            self.memory.save_generation(gen_obj)
            self.memory.save_lineage({"parent": active.get("id") if active else None, "child": gen_id, "score": cand_fitness, "decision": "PROMOTE"})
            self.git.commit_if_enabled(f"S.E.A.: promote {gen_id} (fit={cand_fitness:.2f})")
            return True
        except Exception as e:
            if backup.exists():
                shutil.copy2(backup, self.core_path)
            return False

    def rollback(self) -> bool:
        backup = self.core_path.with_suffix(".py.bak")
        if backup.exists():
            shutil.copy2(backup, self.core_path)
            log.event(Event.GENERATION_ROLLED_BACK, generation="last")
            return True
        return False

# -----------------------------------------------------------------------------
# GOD ENTITY — Final (Transactional, Deadline, Contract)
# -----------------------------------------------------------------------------
class GodEntity:
    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = Path(base_dir or Path.cwd()).resolve()
        self.core_path = Path(__file__).resolve() if "__file__" in globals() else self.base_dir / CORE_FILENAME
        self.memory = Memory(self.base_dir)
        self.provider = select_model_provider()
        self.research = AdaptiveResearchEngine(self.memory.get("network_available", False), self.memory, self.provider)
        self.validator = ValidationEngine(self.base_dir)
        self.benchmark = BenchmarkEngine()
        self.git = GitManager(self.base_dir, os.environ.get("GIT_MODE", GIT_MODE_DEFAULT))
        self.evolution = EvolutionEngine(
            self.core_path, self.memory, self.validator, self.git, self.base_dir,
            self.provider, self.research, self.benchmark, brutal_mode=True
        )
        self.goal = None
        self._booted = False
        self.memory.set("network_available", bool(self.research.network))
        # Deadline management
        self.deadline = None  # absolute monotonic timestamp
        # Module contracts: mapping module_name → list of required symbols
        self.module_contracts = {
            "risk": ["UnifiedRiskEngine", "evaluate_market_risk"],
            "features": ["UnifiedFeatureEngine", "transform", "build"],
            "portfolio": ["UnifiedPortfolioEngine", "process_trading_signals"],
            "prediction": ["UnifiedPredictionEngine", "run_prediction_pipeline"],
            "signal_idx": ["UnifiedSignalEngine", "execute_pipeline", "generate_signals"],
            "storage": ["UnifiedStorageEngine", "persist_signals"],
            "reporting": ["UnifiedReportingEngine", "send_telegram_broadcast"],
            "monitoring": ["UnifiedMonitoringEngine", "execute_full_audit"],
        }

    # -------------------------------------------------------------------------
    # Deadline management
    # -------------------------------------------------------------------------
    def set_deadline(self, timeout_seconds: int) -> None:
        self.deadline = time.monotonic() + timeout_seconds
        log.info(f"⏰ Deadline set to {timeout_seconds}s from now.")

    def remaining_time(self) -> float:
        if self.deadline is None:
            return float('inf')
        return max(0, self.deadline - time.monotonic())

    def check_deadline(self, operation: str) -> None:
        if self.deadline is not None and time.monotonic() > self.deadline:
            log.event(Event.DEADLINE_EXCEEDED, operation=operation)
            raise TimeoutError(f"Global deadline exceeded during {operation}")

    # -------------------------------------------------------------------------
    # Module generation with validation (two-phase)
    # -------------------------------------------------------------------------
    def _generate_code_from_llm(self, module_name: str) -> str:
        goal = self.memory.get("current_goal") or "IDX Scalping Optimizer"
        market_context = self.memory.get("last_market_observation") or "Bullish moderate volatility"
        prompt = f"""
You are S.E.A. (Self-Evolving Agent). Write Python code for the IDX trading module: '{module_name}'.

Context:
- Goal: {goal}
- Market: {market_context}

Rules:
1. Do NOT hardcode numeric parameters.
2. Read parameters from memory (sea_agent.memory.get()).
3. Use Polars/NumPy.
4. Main class: `Unified{module_name.capitalize()}Engine`.
5. __init__ must accept `sea_agent` (optional).
6. Implement core methods.

Return ONLY Python code.
"""
        response = self.provider.generate(prompt, max_tokens=3000)
        code = re.sub(r'```python|```', '', response).strip()
        return code

    def generate_module_code(self, module_name: str) -> Tuple[bool, str, Optional[str]]:
        """Generate module with full validation. Returns (success, code, error_message)."""
        # Check existing valid code
        existing = self.memory.get_module_code(module_name)
        if existing:
            # Re-validate in case memory got corrupted
            try:
                ast.parse(existing)
                compile(existing, f"<{module_name}>", "exec")
                return True, existing, None
            except Exception as e:
                log.warning(f"⚠️ Stored module {module_name} invalid, regenerating. Error: {e}")
                self.memory.set_module_code(module_name, None)

        # Generate from LLM
        code = self._generate_code_from_llm(module_name)
        if not code or code.startswith("[") and code.endswith("]"):
            return False, code, "LLM response empty or error marker"

        # Phase 1: Syntax validation
        try:
            ast.parse(code)
        except SyntaxError as e:
            log.event(Event.MODULE_REJECTED, module=module_name, reason=f"syntax: {e}")
            return False, code, str(e)

        # Phase 2: Compile validation
        try:
            compile(code, f"<{module_name}>", "exec")
        except Exception as e:
            log.event(Event.MODULE_REJECTED, module=module_name, reason=f"compile: {e}")
            return False, code, str(e)

        # Phase 3: Contract validation (if module has known contract)
        contract = self.module_contracts.get(module_name, [])
        if contract:
            try:
                namespace = {}
                exec(code, namespace)
                for sym in contract:
                    if sym not in namespace:
                        raise NameError(f"Missing symbol: {sym}")
            except Exception as e:
                log.event(Event.MODULE_REJECTED, module=module_name, reason=f"contract: {e}")
                return False, code, str(e)

        # All valid → store
        self.memory.set_module_code(module_name, code, valid=True)
        log.event(Event.MODULE_GENERATED, module=module_name, length=len(code))
        return True, code, None

    def validate_module_contract(self, module_name: str, required_symbols: List[str]) -> bool:
        code = self.memory.get_module_code(module_name)
        if not code:
            return False
        try:
            namespace = {}
            exec(code, namespace)
            for sym in required_symbols:
                if sym not in namespace:
                    log.error(f"❌ [CONTRACT] {module_name} missing symbol: {sym}")
                    return False
            return True
        except Exception as e:
            log.error(f"❌ [CONTRACT] {module_name} validation error: {e}")
            return False

    # -------------------------------------------------------------------------
    # Transactional Workspace
    # -------------------------------------------------------------------------
    def start_candidate_session(self, objective: str) -> CandidateWorkspace:
        session_id = f"session_{uuid.uuid4().hex[:8]}"
        ws = CandidateWorkspace(self.base_dir, session_id)
        # Stage critical files
        ws.stage_file(self.core_path)
        main_path = self.base_dir / "main.py"
        if main_path.exists():
            ws.stage_file(main_path)
        # Also stage any module that might be modified
        for mod in ["risk", "features", "portfolio", "prediction", "signal_idx", "storage", "reporting", "monitoring"]:
            mod_path = self.base_dir / f"{mod}.py"
            if mod_path.exists():
                ws.stage_file(mod_path)
        log.event(Event.WORKSPACE_CREATED, workspace=str(ws.workspace_dir))
        return ws

    def promote_workspace(self, workspace: CandidateWorkspace) -> bool:
        if workspace.validate_workspace(self):
            if workspace.commit():
                log.event(Event.WORKSPACE_COMMITTED, workspace=str(workspace.workspace_dir))
                return True
        else:
            workspace.rollback()
            log.event(Event.WORKSPACE_ROLLED_BACK, workspace=str(workspace.workspace_dir))
        return False

    # -------------------------------------------------------------------------
    # Main evolution loop with global deadline
    # -------------------------------------------------------------------------
    def boot(self) -> None:
        log.info(f"🚀 S.E.A. v{VERSION} Booting...")
        active = self.memory.get_active_generation()
        if not active:
            init = Generation(id="S000001", parent=None, timestamp=utc_now(), objective="initial_boot",
                              source_hash=sha256_file(self.core_path) if self.core_path.exists() else "unknown",
                              status=GenerationStatus.ACTIVE, fitness_score=0.0)
            self.memory.save_generation(init)
        self._booted = True
        log.event(Event.BOOT_COMPLETE, provider=self.provider.name())

    def run_loop(self, max_iterations: int = 5) -> Dict[str, Any]:
        if not self._booted:
            self.boot()
        self.goal = self.memory.get("current_goal") or "Optimize IDX trading system"
        # Set global deadline
        timeout = int(os.environ.get("GOD_TIMEOUT", "300"))
        self.set_deadline(timeout)

        history = []
        for i in range(max_iterations):
            try:
                self.check_deadline(f"iteration {i}")
            except TimeoutError:
                log.warning(f"⏰ Global deadline exceeded at iteration {i}")
                break
            log.event(Event.LOOP_ITERATION, iteration=i)

            # Use workspace for evolution
            with self.start_candidate_session(self.goal) as ws:
                try:
                    # Create candidate in workspace
                    gen_id, cand_path = self.evolution.create_candidate_in_workspace(self.goal, ws)
                    # Evaluate fitness (using workspace file)
                    fitness = self.evolution.evaluate_fitness(cand_path)
                    # If fitness > threshold, promote workspace
                    if fitness.get("fitness", -1) > 0:
                        if self.promote_workspace(ws):
                            # Also promote generation status
                            self.evolution.promote(gen_id, cand_path, fitness)
                            history.append({"iteration": i, "promoted": True, "fitness": fitness.get("fitness")})
                        else:
                            history.append({"iteration": i, "promoted": False, "reason": "workspace validation failed"})
                    else:
                        history.append({"iteration": i, "promoted": False, "reason": "fitness failed"})
                except Exception as e:
                    ws.rollback()
                    log.error(f"❌ Iteration {i} failed: {e}")
                    history.append({"iteration": i, "promoted": False, "error": str(e)})

            # Self-heal check on core
            if not self.validator.verify(self.core_path).get("overall"):
                self.evolution.rollback()

            active = self.memory.get_active_generation()
            if active and active.get("fitness_score", 0) > 90:
                break

        return {"iterations": len(history), "history": history}

    # -------------------------------------------------------------------------
    # Other methods
    # -------------------------------------------------------------------------
    def propose_trading_config(self) -> Dict[str, Any]:
        cached = self.memory.get("trading_config")
        if cached and isinstance(cached, dict):
            return cached
        defaults = {
            "min_adtv_idr": 5_000_000_000.0,
            "min_confidence": 0.72,
            "min_rrr": 1.20,
            "risk_scale": 0.20,
            "configured_by": "SEA_DEFAULT"
        }
        if self.provider.available() and not isinstance(self.provider, GeneticMutator):
            try:
                prompt = ("Return ONLY JSON: min_adtv_idr, min_confidence, min_rrr, risk_scale. "
                          "Base on conservative IDX scalping.")
                response = self.provider.generate(prompt, max_tokens=300)
                start, end = response.find("{"), response.rfind("}")
                if start >= 0 and end > start:
                    raw = json.loads(response[start:end+1])
                    raw["configured_by"] = f"SEA_LLM:{self.provider.name()}"
                    self.memory.set("trading_config", raw)
                    return raw
            except Exception as e:
                log.warning(f"LLM config generation failed: {e}")
        self.memory.set("trading_config", defaults)
        return defaults

    def narrate_market(self, portfolio_state: Dict[str, Any], top_signals: Optional[List[Dict[str, Any]]] = None) -> str:
        if self.provider.available() and not isinstance(self.provider, GeneticMutator):
            try:
                context = json.dumps({"portfolio": portfolio_state, "top_signals": (top_signals or [])[:5]}, default=str)
                response = self.provider.generate(f"Narrate IDX market based on: {context[:1500]}", max_tokens=300)
                # Validate narrative format (simple)
                if response and not response.startswith("[") and len(response) > 10:
                    return response
            except Exception:
                pass
        return f"IDX simulation complete. Signals: {len(top_signals or [])}."

    def analyze_candidates_deep(self, candidates_report: str) -> Any:
        if self.provider.available() and not isinstance(self.provider, GeneticMutator):
            try:
                response = self.provider.generate(f"Analyze IDX candidates:\n{candidates_report[:2000]}", max_tokens=800)
                if response and not response.startswith("[") and len(response) > 10:
                    return response
            except Exception:
                pass
        return None

    def observe(self) -> Dict[str, Any]:
        return {"provider": self.provider.name(), "active_gen": self.memory.get_active_generation()}

    def set_goal(self, goal: str) -> None:
        self.goal = goal
        self.memory.set("current_goal", goal)

    def plan(self, goal: str) -> Dict[str, Any]:
        plan = {"goal": goal, "tasks": ["Observe", "Research", "Evolve", "Validate", "Promote"], "created_at": utc_now()}
        self.memory.set("last_plan", plan)
        return plan

    def status(self) -> Dict[str, Any]:
        active = self.memory.get_active_generation()
        return {
            "version": VERSION,
            "provider": self.provider.name(),
            "active_gen": active,
            "fitness": active.get("fitness_score") if active else None,
            "model_primary": getattr(self.provider, 'primary', None),
            "model_fallback": getattr(self.provider, 'fallback', None)
        }

    def test_boot(self) -> Dict[str, Any]:
        return self.validator.verify(self.core_path)

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(description="S.E.A. v8.0 Transactional Evolution")
    p.add_argument("--goal", type=str)
    p.add_argument("--run", action="store_true")
    p.add_argument("--status", action="store_true")
    p.add_argument("--test-boot", action="store_true")
    p.add_argument("--generate-module", type=str)
    p.add_argument("--max-iterations", type=int, default=5)
    args = p.parse_args(argv)

    agent = GodEntity()
    agent.boot()

    if args.test_boot:
        res = agent.test_boot()
        print(json.dumps(res, indent=2))
        return 0 if res.get("overall") else 1
    if args.goal:
        agent.set_goal(args.goal)
    if args.generate_module:
        success, code, err = agent.generate_module_code(args.generate_module)
        if success:
            print(f"✅ Generated module '{args.generate_module}' (len={len(code)})")
        else:
            print(f"❌ Failed: {err}")
            return 1
        return 0
    if args.run:
        report = agent.run_loop(args.max_iterations)
        print(json.dumps(report, indent=2))
        return 0
    print(json.dumps(agent.status(), indent=2))
    return 0

if __name__ == "__main__":
    sys.exit(main())
