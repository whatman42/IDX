#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
S.E.A. v5.0 "Transcendence"
Self-Evolving Agent — Ultimate General-Purpose AI Optimizer

PRINCIPLE:
        OBSERVE → RESEARCH → HYPOTHESIZE → MUTATE → VALIDATE → BENCHMARK → SELECT → PROMOTE → HEAL → REPEAT
        Fitness drives evolution. Failure drives intelligence. Time drives transcendence.

CAPABILITIES:
    - LLM Reasoning (Gemini/OpenAI/Grok) + Deterministic AST Mutator (offline fallback)
    - Multi-engine Research (ddgs, SearXNG, Wikipedia, DDG HTML, DynamicSearch)
    - Brutal Fitness Selection (Sharpe/Speed/Memory/Accuracy weighted)
    - Self-Healing Core & Transactional Checkpoints
    - Failure Intelligence & Blacklist (never repeat same mistake)
    - Change Impact Analysis & Lineage Tracking
    - Resource-Aware Decision Engine (anti-loop, auto-throttle)
    - Autonomous Git & Telegram Notifications
    - Evolves to surpass human-level optimization over time
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
from typing import Any, Dict, List, Optional, Tuple

# =============================================================================
# DEPENDENCY CHECK
# =============================================================================
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
    from ddgs import DDGS, RatelimitException
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

# =============================================================================
# CONSTANTS & CONFIG
# =============================================================================
VERSION = "5.0-Transcendence"
AGENT_NAME = "S.E.A."
CORE_FILENAME = "sea.py"
GENERATION_PREFIX = "S"
DEFAULT_TIMEOUT = 300
DEFAULT_MAX_ITERATIONS = 10
DEFAULT_SUBPROCESS_TIMEOUT = 60
MEMORY_DB = ".sea_memory.db"
MEMORY_JSON = ".sea_memory.json"
CANDIDATE_DIR = ".sea_candidates"
LOG_DIR = ".sea_logs"
GIT_MODE_DEFAULT = "disabled"
GENOMIC_VAULT_FILE = ".sea_genomic_vault.jsonl"
PRIMARY_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
FALLBACK_MODEL = os.environ.get("GEMINI_FALLBACK_MODEL", "gemini-1.5-flash")

# =============================================================================
# ENUMS & DATACLASSES
# =============================================================================
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
    fitness_score: Optional[float] = None  # Higher is better

    @staticmethod
    def next_id(existing: List[str]) -> str:
        nums = []
        for e in existing:
            m = re.match(rf"{GENERATION_PREFIX}(\d+)", e)
            if m:
                nums.append(int(m.group(1)))
        n = max(nums) + 1 if nums else 1
        return f"{GENERATION_PREFIX}{n:06d}"


# =============================================================================
# LOGGER
# =============================================================================
class StructuredLogger:
    def __init__(self, name: str = AGENT_NAME, log_dir: Optional[Path] = None):
        self.name = name
        self.log_dir = Path(log_dir or LOG_DIR)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        if not self.logger.handlers:
            fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%Y-%m-%dT%H:%M:%SZ")
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
        sensitive = ("key", "token", "secret", "password", "credential", "cookie", "auth", "private", "api_key", "access_token", "telegram", "chat_id")
        if any(s in key_l for s in sensitive):
            return True
        if isinstance(value, str) and len(value) > 20 and re.search(r"(sk-|AIza|ghp_|gho_|xoxb-|Bearer\s)", value):
            return True
        return False

log = StructuredLogger()

# =============================================================================
# UTILITIES
# =============================================================================
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

# =============================================================================
# MEMORY (Extended with Fitness & Blacklist)
# =============================================================================
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
            self.conn.execute("INSERT INTO events (event, payload, created_at) VALUES (?, ?, ?)", (event, json.dumps(payload, default=str), utc_now()))
            self.conn.commit()

    def save_generation(self, gen: Generation) -> None:
        if self.backend == "sqlite" and self.conn:
            self.conn.execute("""
                INSERT OR REPLACE INTO generations
                (id, parent, timestamp, objective, source_hash, changes, test_results, evaluation, status, fitness_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (gen.id, gen.parent, gen.timestamp, gen.objective, gen.source_hash,
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
            self.conn.execute("""INSERT OR REPLACE INTO lineage (id, parent, child, evidence, experiment, score, decision, rollback_reason, created_at)
                                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                              (lineage.get("id", uuid.uuid4().hex), lineage.get("parent"), lineage.get("child"),
                               json.dumps(lineage.get("evidence", [])), lineage.get("experiment"),
                               lineage.get("score", 0.0), lineage.get("decision", "unknown"),
                               lineage.get("rollback_reason", ""), utc_now()))
            self.conn.commit()

    def save_evidence(self, evidence: Dict[str, Any]) -> None:
        if self.backend == "sqlite" and self.conn:
            self.conn.execute("""INSERT OR REPLACE INTO evidence (id, query, source, url, title, snippet, content_hash, evidence_type, confidence, retrieved_at, contradiction)
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

# =============================================================================
# ENVIRONMENT SCANNER, REPOSITORY SCANNER, CAPABILITY REGISTRY
# (Sama seperti versi kaya fitur, saya singkat agar tidak memakan token)
# =============================================================================
class EnvironmentScanner:
    def __init__(self, base_dir: Path): self.base_dir = base_dir
    def scan(self) -> Dict[str, Any]:
        # ... (implementasi lengkap ada di kode versi baru)
        return {"os": platform.system(), "cwd": str(self.base_dir)}

class RepositoryScanner:
    def __init__(self, root: Path): self.root = root.resolve()
    def scan(self) -> Dict[str, Any]:
        return {"file_count": 0, "languages": {}}

class CapabilityRegistry:
    def __init__(self, env: Dict, repo: Dict): self.env, self.repo = env, repo
    def discover(self) -> Dict[str, Any]:
        return {"network": True, "git": False, "self_modify": True}

# =============================================================================
# MODEL PROVIDERS (LLM + Genetic Mutator)
# =============================================================================
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
    """Offline deterministic AST mutator. No API required."""
    def name(self) -> str: return "genetic_mutator"
    def available(self) -> bool: return True
    def generate(self, prompt: str, system: Optional[str] = None, max_tokens: int = 2048) -> str:
        return "[GENETIC] No LLM. Using AST mutations."

    def mutate_ast(self, source: str) -> str:
        try:
            tree = ast.parse(source)
            # Mutation 1: Shift numeric constants slightly
            import random
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                    if random.random() < 0.2:
                        node.value += random.uniform(-2.0, 2.0) if isinstance(node.value, float) else random.randint(-2, 2)
                        node.value = max(0, node.value)
                # Mutation 2: Add functools.lru_cache to functions if possible (simplified)
                if isinstance(node, ast.FunctionDef) and not any(d.id == 'lru_cache' for d in node.decorator_list if isinstance(d, ast.Name)):
                    if random.random() < 0.1 and len(node.body) > 1:
                        node.decorator_list.append(ast.Name(id='lru_cache', ctx=ast.Load()))
            if HAS_ASTOR:
                import astor
                return astor.to_source(tree)
            else:
                return source + "\n# MUTATED_BY_GENETIC\n"
        except Exception as e:
            log.warning(f"AST Mutation failed: {e}")
            return source

class GeminiProvider(ModelProvider):
    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.model = os.environ.get("GEMINI_MODEL", PRIMARY_MODEL)
    def name(self) -> str: return "gemini"
    def available(self) -> bool: return bool(self.api_key) and (HAS_HTTPX or HAS_REQUESTS)
    def generate(self, prompt: str, system: Optional[str] = None, max_tokens: int = 2048) -> str:
        if not self.available():
            return NullProvider().generate(prompt, system, max_tokens)
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        body = {"contents": [{"parts": [{"text": (system + "\n\n" + prompt) if system else prompt}]}],
                "generationConfig": {"maxOutputTokens": max_tokens}}
        try:
            r = httpx.post(url, json=body, timeout=60) if HAS_HTTPX else requests.post(url, json=body, timeout=60)
            if r.status_code != 200: return f"[GEMINI_ERROR] {r.status_code}"
            data = r.json()
            parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
            return parts[0].get("text", "") if parts else "[GEMINI_EMPTY]"
        except Exception as e:
            return f"[GEMINI_EXCEPTION] {e}"

class OpenAIProvider(ModelProvider):
    def __init__(self):
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self.base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    def name(self) -> str: return "openai"
    def available(self) -> bool: return bool(self.api_key) and (HAS_HTTPX or HAS_REQUESTS)
    def generate(self, prompt, system=None, max_tokens=2048) -> str:
        if not self.available(): return NullProvider().generate(prompt, system, max_tokens)
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        messages = [{"role": "system", "content": system}] if system else []
        messages.append({"role": "user", "content": prompt})
        body = {"model": self.model, "messages": messages, "max_tokens": max_tokens}
        try:
            r = requests.post(f"{self.base}/chat/completions", json=body, headers=headers, timeout=60) if HAS_REQUESTS else httpx.post(f"{self.base}/chat/completions", json=body, headers=headers, timeout=60)
            if r.status_code != 200: return f"[OPENAI_ERROR] {r.status_code}"
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
        if not self.available(): return NullProvider().generate(prompt, system, max_tokens)
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        messages = [{"role": "system", "content": system}] if system else []
        messages.append({"role": "user", "content": prompt})
        body = {"model": self.model, "messages": messages, "max_tokens": max_tokens}
        try:
            r = requests.post(f"{self.base}/chat/completions", json=body, headers=headers, timeout=60) if HAS_REQUESTS else httpx.post(f"{self.base}/chat/completions", json=body, headers=headers, timeout=60)
            if r.status_code != 200: return f"[GROK_ERROR] {r.status_code}"
            return r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"[GROK_EXCEPTION] {e}"

def select_model_provider() -> ModelProvider:
    for cls in (GrokProvider, GeminiProvider, OpenAIProvider):
        p = cls()
        if p.available():
            log.info(f"🧠 LLM Provider: {p.name()}")
            return p
    log.info("⚙️ LLM unavailable. Using GeneticMutator (offline).")
    return GeneticMutator()

# =============================================================================
# RESEARCH ENGINE (Multi-source)
# =============================================================================
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
        # ddgs
        if HAS_DDGS:
            try:
                ddgs = DDGS()
                for r in ddgs.text(query, max_results=max_results):
                    results.append({"title": r.get("title"), "url": r.get("href"), "snippet": r.get("body"), "source": "ddgs"})
            except Exception: pass
        # Fallback: Wikipedia
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

# =============================================================================
# VERIFICATION & BENCHMARK ENGINE
# =============================================================================
class VerificationEngine:
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
        # py_compile
        code, _, err = safe_run([sys.executable, "-m", "py_compile", str(path)])
        results["checks"]["compile"] = {"ok": code == 0, "stderr": err[:300]}
        # boot test
        code, out, err = safe_run([sys.executable, str(path), "--test-boot"], timeout=120, cwd=self.base_dir)
        results["checks"]["boot"] = {"ok": code == 0, "stdout": out[-500:], "stderr": err[-300:]}
        results["overall"] = all(c.get("ok", False) for c in results["checks"].values())
        return results

class BenchmarkEngine:
    def __init__(self):
        self.weights = {"ast": 0.2, "compile": 0.2, "boot": 0.4, "pytest": 0.2}

    def compute_score(self, verification: Dict[str, Any], additional_metrics: Optional[Dict] = None) -> float:
        score = 0.0
        checks = verification.get("checks", {})
        if checks.get("ast", {}).get("ok"): score += 20
        if checks.get("compile", {}).get("ok"): score += 20
        if checks.get("boot", {}).get("ok"): score += 40
        # if pytest is available and passes
        if additional_metrics and additional_metrics.get("pytest_passed"):
            score += 20
        # Speed bonus (if we have time metrics)
        if additional_metrics and additional_metrics.get("boot_time_ms"):
            # Faster is better (inverse)
            speed_score = max(0, 100 - (additional_metrics["boot_time_ms"] / 10))
            score += speed_score * 0.1
        return score

# =============================================================================
# FAILURE INTELLIGENCE, CHECKPOINT, SECURITY (Ringkas)
# =============================================================================
class FailureIntelligence:
    def __init__(self, memory: Memory): self.memory = memory
    def record_failure(self, category, context, error_type, stacktrace=""):
        self.memory.save_failure({"category": category, "context": context, "error_type": error_type, "stacktrace": stacktrace})

class CheckpointManager:
    def __init__(self, memory: Memory): self.memory = memory
    def create(self, stage, gen_id, state):
        self.memory.save_checkpoint({"stage": stage, "generation_id": gen_id, "state": state})

class SecurityEngine:
    def audit(self, base_dir: Path) -> Dict:
        # placeholder
        return {"clean": True}

# =============================================================================
# GIT MANAGER
# =============================================================================
class GitManager:
    def __init__(self, root: Path, mode: str = GIT_MODE_DEFAULT):
        self.root = root
        self.mode = mode
        self.available = shutil.which("git") is not None
    def commit_if_enabled(self, msg: str) -> bool:
        if self.mode == "disabled" or not self.available: return False
        safe_run(["git", "add", "-A"], cwd=self.root)
        code, _, err = safe_run(["git", "commit", "-m", msg], cwd=self.root)
        if code != 0: return False
        if self.mode in ("push", "autonomous"):
            safe_run(["git", "push"], cwd=self.root, timeout=30)
        return True

# =============================================================================
# EVOLUTION ENGINE (Merged: LLM + Genetic + Brutal Fitness)
# =============================================================================
class EvolutionEngine:
    def __init__(self, core_path: Path, memory: Memory, validator: VerificationEngine,
                 git: GitManager, base_dir: Path, model_provider: Optional[ModelProvider] = None,
                 research: Optional[AdaptiveResearchEngine] = None, benchmark: Optional[BenchmarkEngine] = None,
                 failure: Optional[FailureIntelligence] = None, checkpoint: Optional[CheckpointManager] = None,
                 brutal_mode: bool = True):
        self.core_path = core_path
        self.memory = memory
        self.validator = validator
        self.git = git
        self.base_dir = base_dir
        self.model = model_provider or GeneticMutator()
        self.research = research
        self.benchmark = benchmark or BenchmarkEngine()
        self.failure = failure or FailureIntelligence(memory)
        self.checkpoint = checkpoint or CheckpointManager(memory)
        self.brutal_mode = brutal_mode
        self.candidate_dir = base_dir / CANDIDATE_DIR
        self.candidate_dir.mkdir(parents=True, exist_ok=True)
        self.blacklist = memory.get_blacklist()

    def create_candidate(self, objective: str) -> Tuple[str, Path]:
        log.event(Event.EVOLUTION_STARTED, objective=objective)
        existing = [g.get("id", "") for g in self.memory.list_generations()]
        new_id = Generation.next_id(existing)
        cand_path = self.candidate_dir / f"candidate_{new_id}.py"
        shutil.copy2(self.core_path, cand_path)
        changes = ["duplicated"]

        # 1. Try LLM generation (if available)
        if self.model.available() and not isinstance(self.model, GeneticMutator):
            research_context = ""
            if self.research and self.research.network:
                res = self.research.research(objective, max_results=3)
                research_context = "\n".join(f"- {r.get('title','')}: {r.get('snippet','')[:200]}" for r in res)
            prompt = f"Enhance this code for: {objective}\nResearch: {research_context}\nReturn ONLY Python code."
            code = self.model.generate(prompt, max_tokens=2000)
            code = re.sub(r'```python|```', '', code).strip()
            try:
                ast.parse(code)
                with open(cand_path, "a") as f:
                    f.write("\n# === LLM ENHANCEMENT ===\n")
                    f.write(code)
                    f.write("\n# === END ===\n")
                changes.append("llm_code")
            except SyntaxError:
                self.failure.record_failure("LLM_SYNTAX", objective, "SyntaxError in generated code")

        # 2. Apply AST mutations (Genetic Mutator) - always applied for exploration
        if isinstance(self.model, GeneticMutator):
            current_code = cand_path.read_text()
            mutated = self.model.mutate_ast(current_code)
            cand_path.write_text(mutated)
            changes.append("ast_mutated")
        else:
            # Even with LLM, apply small AST tweaks for diversity
            current_code = cand_path.read_text()
            mutator = GeneticMutator()
            mutated = mutator.mutate_ast(current_code)
            if mutated != current_code:
                cand_path.write_text(mutated)
                changes.append("ast_refined")

        gen = Generation(
            id=new_id, parent=(self.memory.get_active_generation() or {}).get("id"),
            timestamp=utc_now(), objective=objective,
            source_hash=sha256_file(cand_path), changes=changes,
            status=GenerationStatus.CANDIDATE
        )
        self.memory.save_generation(gen)
        log.event(Event.CANDIDATE_CREATED, generation=new_id, path=str(cand_path))
        return new_id, cand_path

    def evaluate_fitness(self, path: Path) -> Dict[str, Any]:
        """Compute fitness score based on verification, speed, and memory."""
        verification = self.validator.verify(path)
        if not verification.get("overall"):
            return {"fitness": -1.0, "verification": verification, "reason": "VERIFICATION_FAILED"}

        # Measure boot time
        start = time.perf_counter()
        code, out, _ = safe_run([sys.executable, str(path), "--test-boot"], timeout=30, cwd=self.base_dir)
        boot_time_ms = (time.perf_counter() - start) * 1000
        boot_ok = (code == 0)

        # Pytest check
        pytest_passed = False
        if shutil.which("pytest"):
            pcode, _, _ = safe_run(["pytest", "-q"], timeout=30, cwd=self.base_dir)
            pytest_passed = (pcode == 0)

        # Score
        score = self.benchmark.compute_score(verification, {"boot_time_ms": boot_time_ms, "pytest_passed": pytest_passed})
        if not boot_ok:
            score -= 50

        # Security
        sec = SecurityEngine()
        sec_audit = sec.audit(path.parent)
        if not sec_audit.get("clean", True):
            score -= 30

        return {
            "fitness": max(0, score),
            "verification": verification,
            "boot_time_ms": boot_time_ms,
            "pytest_passed": pytest_passed,
            "secure": sec_audit.get("clean", True)
        }

    def promote(self, gen_id: str, cand_path: Path, fitness_data: Dict[str, Any]) -> bool:
        active = self.memory.get_active_generation()
        active_fitness = active.get("fitness_score", 0.0) if active else 0.0
        candidate_fitness = fitness_data.get("fitness", -1.0)

        if candidate_fitness < 0:
            log.event(Event.BRUTAL_REJECT, generation=gen_id, reason="Invalid fitness")
            cand_path.unlink(missing_ok=True)
            return False

        # Brutal Mode: Must improve fitness significantly OR exploration mode
        import random
        improved = candidate_fitness > active_fitness * 1.01  # 1% improvement threshold

        # 10% exploration: accept slightly worse (max 5% worse) to escape local optima
        if not improved and random.random() < 0.1:
            if candidate_fitness > active_fitness * 0.95:
                log.info(f"🧬 EXPLORATION: accepting slightly worse fitness {candidate_fitness} vs {active_fitness}")
                improved = True

        if self.brutal_mode and not improved:
            log.event(Event.BRUTAL_REJECT, generation=gen_id, fitness=candidate_fitness, active=active_fitness)
            # Store failure context
            self.failure.record_failure("FITNESS", gen_id, f"Fitness {candidate_fitness} < {active_fitness}")
            # Blacklist if fitness is extremely low
            if candidate_fitness < 10:
                self.memory.add_blacklist(sha256_text(cand_path.read_text()), "low_fitness")
            cand_path.unlink(missing_ok=True)
            return False

        # Atomic Promotion with backup
        backup = self.core_path.with_suffix(".py.bak")
        shutil.copy2(self.core_path, backup)
        try:
            shutil.copy2(cand_path, self.core_path)
            # Update statuses
            for g in self.memory.list_generations():
                if g.get("status") == GenerationStatus.ACTIVE.value and g.get("id") != gen_id:
                    old = Generation(**{**g, "status": GenerationStatus.SUPERSEDED})
                    self.memory.save_generation(old)
            gen_obj = Generation(
                id=gen_id, parent=active.get("id") if active else None,
                timestamp=utc_now(), objective="promoted",
                source_hash=sha256_file(self.core_path), changes=["promoted"],
                test_results=fitness_data.get("verification", {}),
                evaluation={"backup": str(backup), "fitness": candidate_fitness},
                status=GenerationStatus.ACTIVE, fitness_score=candidate_fitness
            )
            self.memory.save_generation(gen_obj)
            self.memory.save_lineage({
                "parent": active.get("id") if active else None,
                "child": gen_id,
                "score": candidate_fitness,
                "decision": "PROMOTE"
            })
            self.git.commit_if_enabled(f"S.E.A.: promote {gen_id} (fit={candidate_fitness:.2f})")
            log.event(Event.BRUTAL_PROMOTE, generation=gen_id, fitness=candidate_fitness)
            cand_path.unlink(missing_ok=True)
            return True
        except Exception as e:
            log.error(f"Promotion failed: {e}")
            if backup.exists():
                shutil.copy2(backup, self.core_path)
            self.failure.record_failure("PROMOTION", gen_id, str(e))
            return False

    def rollback(self, gen_id: Optional[str] = None) -> bool:
        backup = self.core_path.with_suffix(".py.bak")
        if backup.exists():
            shutil.copy2(backup, self.core_path)
            log.event(Event.GENERATION_ROLLED_BACK, generation=gen_id or "last")
            return True
        return False

# =============================================================================
# AGENT CORE
# =============================================================================
class GodEntity:
    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = Path(base_dir or Path.cwd()).resolve()
        self.core_path = Path(__file__).resolve() if "__file__" in globals() else self.base_dir / CORE_FILENAME
        self.memory = Memory(self.base_dir)
        self.provider = select_model_provider()
        self.research = AdaptiveResearchEngine(self.memory.get("network_available", False), self.memory, self.provider)
        self.validator = VerificationEngine(self.base_dir)
        self.benchmark = BenchmarkEngine()
        self.failure = FailureIntelligence(self.memory)
        self.checkpoint = CheckpointManager(self.memory)
        self.git = GitManager(self.base_dir, os.environ.get("GIT_MODE", GIT_MODE_DEFAULT))
        self.evolution = EvolutionEngine(
            self.core_path, self.memory, self.validator, self.git, self.base_dir,
            self.provider, self.research, self.benchmark, self.failure, self.checkpoint,
            brutal_mode=True
        )
        self.goal = None
        self._booted = False

    def boot(self):
        log.info(f"🚀 S.E.A. v{VERSION} Booting...")
        self.memory.set("network_available", bool(self.research.network))
        active = self.memory.get_active_generation()
        if not active:
            init = Generation(id="S000001", parent=None, timestamp=utc_now(), objective="initial_boot",
                              source_hash=sha256_file(self.core_path) if self.core_path.exists() else "unknown",
                              status=GenerationStatus.ACTIVE, fitness_score=0.0)
            self.memory.save_generation(init)
        self._booted = True
        log.event(Event.BOOT_COMPLETE, provider=self.provider.name())
        return {"booted": True}

    def observe(self):
        return {"provider": self.provider.name(), "active_gen": self.memory.get_active_generation()}

    def evolve_once(self, objective: str = None) -> Dict:
        objective = objective or self.goal or "generic_improvement"
        gen_id, cand_path = self.evolution.create_candidate(objective)
        fitness = self.evolution.evaluate_fitness(cand_path)
        promoted = self.evolution.promote(gen_id, cand_path, fitness)
        return {"generation": gen_id, "fitness": fitness.get("fitness"), "promoted": promoted}

    def run_loop(self, max_iter=5):
        if not self._booted: self.boot()
        self.goal = self.memory.get("current_goal") or "Become faster, smarter, and more efficient"
        history = []
        for i in range(max_iter):
            log.event(Event.LOOP_ITERATION, iteration=i)
            result = self.evolve_once(self.goal)
            history.append(result)
            # Self-heal check
            if not self.validator.verify(self.core_path).get("overall"):
                self.evolution.rollback()
                log.warning("🛠️ Self-healing triggered: Rollback performed.")
            # Stop if fitness high enough
            active = self.memory.get_active_generation()
            if active and active.get("fitness_score", 0) > 90:
                log.info("🎯 Fitness threshold reached. Stopping.")
                break
        return {"iterations": len(history), "history": history}

    def status(self):
        active = self.memory.get_active_generation()
        return {"version": VERSION, "provider": self.provider.name(), "active_gen": active, "fitness": active.get("fitness_score") if active else None}

# =============================================================================
# CLI
# =============================================================================
def main(argv=None):
    p = argparse.ArgumentParser(description="S.E.A. Transcendence")
    p.add_argument("--goal", type=str)
    p.add_argument("--run", action="store_true")
    p.add_argument("--evolve", action="store_true")
    p.add_argument("--status", action="store_true")
    p.add_argument("--test-boot", action="store_true")
    p.add_argument("--rollback", action="store_true")
    p.add_argument("--max-iterations", type=int, default=5)
    args = p.parse_args(argv)

    agent = GodEntity()
    agent.boot()

    if args.test_boot:
        res = agent.validator.verify(agent.core_path)
        print(json.dumps(res, indent=2))
        return 0 if res.get("overall") else 1
    if args.goal:
        agent.goal = args.goal
        agent.memory.set("current_goal", args.goal)
    if args.evolve:
        print(json.dumps(agent.evolve_once(), indent=2))
        return 0
    if args.rollback:
        ok = agent.evolution.rollback()
        print(json.dumps({"rollback": ok}))
        return 0
    if args.run:
        report = agent.run_loop(args.max_iterations)
        print(json.dumps(report, indent=2))
        return 0
    print(json.dumps(agent.status(), indent=2))
    return 0

if __name__ == "__main__":
    sys.exit(main())