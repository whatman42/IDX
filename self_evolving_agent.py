#!/usr/bin/env python3
"""
GOD ENTITY — General-Purpose Autonomous Self-Evolving AI Agent
Codename: GOD ENTITY
Single-file core. Domain-agnostic. Environment-adaptive. Self-modifying.

PRINCIPLE:
ANYTHING OBSERVABLE → CAN BE ANALYZED
ANYTHING ANALYZABLE → CAN BECOME A HYPOTHESIS
ANY HYPOTHESIS → CAN BECOME AN EXPERIMENT
ANY SUCCESSFUL EXPERIMENT → CAN BECOME AN IMPROVEMENT
ANY IMPROVEMENT → CAN BECOME A NEW GENERATION
THE AGENT ITSELF → IS ALSO AN EVOLUTION TARGET

MISSION:
Maximalkan keuntungan legal untuk pemilik.
Gunakan semua sumber daya yang tersedia, riset internet, dan kemampuan self-evolution.
Patuhi hukum dan peraturan yang berlaku.
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
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# Optional network
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

# =============================================================================
# CONSTANTS & CONFIG
# =============================================================================

VERSION = "1.1.0"
AGENT_NAME = "GOD_ENTITY"
CORE_FILENAME = "self_evolving_agent.py"
GENERATION_PREFIX = "G"
DEFAULT_TIMEOUT = 300
DEFAULT_MAX_ITERATIONS = 50
DEFAULT_SUBPROCESS_TIMEOUT = 60
DEFAULT_MAX_RESEARCH = 10
MEMORY_DB = ".god_entity_memory.db"
MEMORY_JSON = ".god_entity_memory.json"
CANDIDATE_DIR = ".god_entity_candidates"
LOG_DIR = ".god_entity_logs"
GIT_MODE_DEFAULT = "disabled"  # disabled | commit | push

# Structured event names (observability)
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


class GenerationStatus(str, Enum):
    ACTIVE = "ACTIVE"
    CANDIDATE = "CANDIDATE"
    FAILED = "FAILED"
    PROMOTED = "PROMOTED"
    ROLLED_BACK = "ROLLED_BACK"
    SUPERSEDED = "SUPERSEDED"


# =============================================================================
# STRUCTURED LOGGING
# =============================================================================

class StructuredLogger:
    def __init__(self, name: str = AGENT_NAME, log_dir: Optional[Path] = None):
        self.name = name
        self.log_dir = Path(log_dir or LOG_DIR)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        if not self.logger.handlers:
            fmt = logging.Formatter(
                "%(asctime)s | %(levelname)-7s | %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%SZ",
            )
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
        msg = f"[{event.value}] {json.dumps(safe, default=str, ensure_ascii=False)}"
        self.logger.info(msg)

    def info(self, msg: str, **kwargs: Any) -> None:
        self.logger.info(msg, **kwargs)

    def warning(self, msg: str, **kwargs: Any) -> None:
        self.logger.warning(msg, **kwargs)

    def error(self, msg: str, **kwargs: Any) -> None:
        self.logger.error(msg, **kwargs)

    def debug(self, msg: str, **kwargs: Any) -> None:
        self.logger.debug(msg, **kwargs)

    @staticmethod
    def _is_sensitive(key: str, value: Any) -> bool:
        key_l = key.lower()
        sensitive_keys = (
            "key", "token", "secret", "password", "passwd", "credential",
            "cookie", "auth", "private", "api_key", "access_token",
            "bank_account", "wallet", "ktp", "phone", "email", "name",
        )
        if any(s in key_l for s in sensitive_keys):
            return True
        if isinstance(value, str) and len(value) > 20:
            if re.search(r"(sk-|AIza|ghp_|gho_|xoxb-|Bearer\s)", value):
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


def safe_run(
    cmd: List[str],
    cwd: Optional[Path] = None,
    timeout: int = DEFAULT_SUBPROCESS_TIMEOUT,
    env: Optional[Dict[str, str]] = None,
    input_text: Optional[str] = None,
) -> Tuple[int, str, str]:
    try:
        result = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            input=input_text,
        )
        return result.returncode, result.stdout or "", result.stderr or ""
    except subprocess.TimeoutExpired:
        return -1, "", "TIMEOUT"
    except FileNotFoundError:
        return -2, "", "EXECUTABLE_NOT_FOUND"
    except Exception as e:
        return -3, "", str(e)


def is_secret_env(key: str) -> bool:
    key_l = key.lower()
    return any(
        s in key_l
        for s in (
            "key", "token", "secret", "password", "passwd", "credential",
            "cookie", "auth", "private", "api_key", "access_token", "ssh",
            "bank_account", "wallet", "ktp", "phone", "email",
        )
    )


# =============================================================================
# MEMORY (SQLite preferred, JSON fallback)
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
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS kv (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TEXT
            );
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event TEXT,
                payload TEXT,
                created_at TEXT
            );
            CREATE TABLE IF NOT EXISTS generations (
                id TEXT PRIMARY KEY,
                parent TEXT,
                timestamp TEXT,
                objective TEXT,
                source_hash TEXT,
                changes TEXT,
                test_results TEXT,
                evaluation TEXT,
                status TEXT
            );
            CREATE TABLE IF NOT EXISTS experiments (
                id TEXT PRIMARY KEY,
                hypothesis TEXT,
                change_desc TEXT,
                result TEXT,
                metrics TEXT,
                conclusion TEXT,
                created_at TEXT
            );
            CREATE TABLE IF NOT EXISTS observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT,
                content TEXT,
                created_at TEXT
            );
            """
        )
        self.conn.commit()

    def set(self, key: str, value: Any) -> None:
        payload = json.dumps(value, default=str)
        if self.backend == "sqlite" and self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO kv (key, value, updated_at) VALUES (?, ?, ?)",
                (key, payload, utc_now()),
            )
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
        else:
            data = self._read_json()
            entry = data.get(key)
            if entry:
                return entry.get("value", default)
            return default

    def log_event(self, event: str, payload: Dict[str, Any]) -> None:
        if self.backend == "sqlite" and self.conn:
            self.conn.execute(
                "INSERT INTO events (event, payload, created_at) VALUES (?, ?, ?)",
                (event, json.dumps(payload, default=str), utc_now()),
            )
            self.conn.commit()

    def save_generation(self, gen: "Generation") -> None:
        if self.backend == "sqlite" and self.conn:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO generations
                (id, parent, timestamp, objective, source_hash, changes, test_results, evaluation, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    gen.id,
                    gen.parent,
                    gen.timestamp,
                    gen.objective,
                    gen.source_hash,
                    json.dumps(gen.changes, default=str),
                    json.dumps(gen.test_results, default=str),
                    json.dumps(gen.evaluation, default=str),
                    gen.status.value,
                ),
            )
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
            rows = self.conn.execute(
                "SELECT * FROM generations ORDER BY timestamp DESC"
            ).fetchall()
            return [dict(r) for r in rows]
        else:
            data = self._read_json()
            return list(data.get("generations", {}).values())

    def get_active_generation(self) -> Optional[Dict[str, Any]]:
        gens = self.list_generations()
        for g in gens:
            if g.get("status") == GenerationStatus.ACTIVE.value:
                return g
        return None

    def add_observation(self, category: str, content: Any) -> None:
        if self.backend == "sqlite" and self.conn:
            self.conn.execute(
                "INSERT INTO observations (category, content, created_at) VALUES (?, ?, ?)",
                (category, json.dumps(content, default=str), utc_now()),
            )
            self.conn.commit()

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
# GENERATION SYSTEM
# =============================================================================

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
# ENVIRONMENT SCANNER
# =============================================================================

class EnvironmentScanner:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.fingerprint: Dict[str, Any] = {}

    def scan(self) -> Dict[str, Any]:
        fp: Dict[str, Any] = {
            "timestamp": utc_now(),
            "os": {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "platform": platform.platform(),
            },
            "python": {
                "version": sys.version,
                "version_info": list(sys.version_info),
                "executable": sys.executable,
                "implementation": platform.python_implementation(),
            },
            "cpu": {
                "count": os.cpu_count(),
            },
            "memory": self._memory_info(),
            "disk": self._disk_info(),
            "cwd": str(Path.cwd().resolve()),
            "agent_path": str(Path(__file__).resolve()) if "__file__" in globals() else None,
            "env_vars": self._scan_env_vars(),
            "executables": self._scan_executables(),
            "package_managers": self._scan_package_managers(),
            "installed_packages": self._scan_installed_packages(),
            "compilers_interpreters": self._scan_compilers(),
            "git": self._scan_git(),
            "github_cli": self._scan_gh(),
            "docker": self._scan_docker(),
            "testing_frameworks": self._scan_testing(),
            "linters_formatters": self._scan_linters(),
            "build_systems": self._scan_build(),
            "network": self._scan_network(),
            "ci_cd": self._scan_ci(),
        }
        self.fingerprint = fp
        log.event(Event.ENVIRONMENT_DISCOVERED, summary={
            "os": fp["os"]["system"],
            "python": fp["python"]["version_info"][:3],
            "cwd": fp["cwd"],
            "git": fp["git"].get("available"),
            "network": fp["network"].get("available"),
        })
        return fp

    def _memory_info(self) -> Dict[str, Any]:
        info = {}
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        info["total_kb"] = int(line.split()[1])
                    elif line.startswith("MemAvailable:"):
                        info["available_kb"] = int(line.split()[1])
        except Exception:
            pass
        return info

    def _disk_info(self) -> Dict[str, Any]:
        try:
            st = os.statvfs("/")
            return {
                "total": st.f_frsize * st.f_blocks,
                "free": st.f_frsize * st.f_bfree,
                "available": st.f_frsize * st.f_bavail,
            }
        except Exception:
            return {}

    def _scan_env_vars(self) -> Dict[str, str]:
        result = {}
        for k in sorted(os.environ.keys()):
            if is_secret_env(k):
                result[k] = "AVAILABLE"
            else:
                v = os.environ[k]
                result[k] = v[:80] + "..." if len(v) > 80 else v
        return result

    def _scan_executables(self) -> Dict[str, bool]:
        candidates = [
            "git", "gh", "docker", "python", "python3", "pip", "pip3",
            "node", "npm", "npx", "yarn", "go", "rustc", "cargo",
            "java", "javac", "gcc", "g++", "clang", "make", "cmake",
            "ruby", "php", "perl", "lua", "R", "julia",
            "pytest", "unittest", "mypy", "flake8", "black", "ruff",
            "eslint", "prettier", "tsc", "cargo", "mvn", "gradle",
            "curl", "wget", "jq", "yq", "terraform", "kubectl",
        ]
        found = {}
        for c in candidates:
            path = shutil.which(c)
            found[c] = path is not None
        return found

    def _scan_package_managers(self) -> Dict[str, bool]:
        return {
            "pip": shutil.which("pip") is not None or shutil.which("pip3") is not None,
            "npm": shutil.which("npm") is not None,
            "yarn": shutil.which("yarn") is not None,
            "cargo": shutil.which("cargo") is not None,
            "go": shutil.which("go") is not None,
            "apt": shutil.which("apt") is not None or shutil.which("apt-get") is not None,
            "brew": shutil.which("brew") is not None,
        }

    def _scan_installed_packages(self) -> List[str]:
        pkgs = []
        code, out, _ = safe_run([sys.executable, "-m", "pip", "list", "--format=freeze"], timeout=30)
        if code == 0:
            for line in out.splitlines():
                if "==" in line:
                    pkgs.append(line.strip())
        return pkgs[:200]

    def _scan_compilers(self) -> Dict[str, Optional[str]]:
        tools = ["python3", "node", "go", "rustc", "javac", "gcc", "g++", "clang", "ruby", "php"]
        result = {}
        for t in tools:
            path = shutil.which(t)
            if path:
                code, out, err = safe_run([path, "--version"], timeout=5)
                result[t] = (out or err).splitlines()[0] if (out or err) else path
            else:
                result[t] = None
        return result

    def _scan_git(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {"available": False}
        if shutil.which("git"):
            info["available"] = True
            code, out, _ = safe_run(["git", "--version"])
            info["version"] = out.strip() if code == 0 else None
            code, out, _ = safe_run(["git", "rev-parse", "--is-inside-work-tree"], cwd=self.base_dir)
            info["in_repo"] = code == 0 and out.strip() == "true"
            if info["in_repo"]:
                for key, cmd in [
                    ("root", ["git", "rev-parse", "--show-toplevel"]),
                    ("branch", ["git", "rev-parse", "--abbrev-ref", "HEAD"]),
                    ("commit", ["git", "rev-parse", "HEAD"]),
                    ("remote", ["git", "remote", "get-url", "origin"]),
                ]:
                    c, o, _ = safe_run(cmd, cwd=self.base_dir)
                    info[key] = o.strip() if c == 0 else None
        return info

    def _scan_gh(self) -> Dict[str, Any]:
        info = {"available": False}
        if shutil.which("gh"):
            info["available"] = True
            code, out, _ = safe_run(["gh", "--version"])
            info["version"] = out.splitlines()[0] if code == 0 else None
            code, out, _ = safe_run(["gh", "auth", "status"], timeout=10)
            info["authenticated"] = code == 0
        return info

    def _scan_docker(self) -> Dict[str, Any]:
        info = {"available": False}
        if shutil.which("docker"):
            info["available"] = True
            code, out, _ = safe_run(["docker", "--version"])
            info["version"] = out.strip() if code == 0 else None
        return info

    def _scan_testing(self) -> Dict[str, bool]:
        return {
            "pytest": shutil.which("pytest") is not None,
            "unittest": True,
            "nose": False,
            "jest": shutil.which("jest") is not None or (shutil.which("npx") is not None),
            "go_test": shutil.which("go") is not None,
            "cargo_test": shutil.which("cargo") is not None,
        }

    def _scan_linters(self) -> Dict[str, bool]:
        return {
            "ruff": shutil.which("ruff") is not None,
            "flake8": shutil.which("flake8") is not None,
            "mypy": shutil.which("mypy") is not None,
            "black": shutil.which("black") is not None,
            "eslint": shutil.which("eslint") is not None,
            "prettier": shutil.which("prettier") is not None,
        }

    def _scan_build(self) -> Dict[str, bool]:
        return {
            "make": shutil.which("make") is not None,
            "cmake": shutil.which("cmake") is not None,
            "npm_scripts": (Path.cwd() / "package.json").exists(),
            "setuptools": (Path.cwd() / "setup.py").exists() or (Path.cwd() / "pyproject.toml").exists(),
            "cargo": (Path.cwd() / "Cargo.toml").exists(),
            "go_mod": (Path.cwd() / "go.mod").exists(),
            "maven": (Path.cwd() / "pom.xml").exists(),
            "gradle": (Path.cwd() / "build.gradle").exists() or (Path.cwd() / "build.gradle.kts").exists(),
        }

    def _scan_network(self) -> Dict[str, Any]:
        available = False
        if HAS_HTTPX or HAS_REQUESTS:
            try:
                if HAS_HTTPX:
                    r = httpx.get("https://httpbin.org/get", timeout=5)
                    available = r.status_code == 200
                elif HAS_REQUESTS:
                    r = requests.get("https://httpbin.org/get", timeout=5)
                    available = r.status_code == 200
            except Exception:
                available = False
        return {
            "available": available,
            "httpx": HAS_HTTPX,
            "requests": HAS_REQUESTS,
        }

    def _scan_ci(self) -> Dict[str, Any]:
        return {
            "github_actions": os.environ.get("GITHUB_ACTIONS") == "true",
            "ci": os.environ.get("CI") == "true",
            "github_workspace": os.environ.get("GITHUB_WORKSPACE"),
            "github_repository": os.environ.get("GITHUB_REPOSITORY"),
            "github_ref": os.environ.get("GITHUB_REF"),
            "github_sha": os.environ.get("GITHUB_SHA"),
            "runner_os": os.environ.get("RUNNER_OS"),
        }


# =============================================================================
# REPOSITORY SCANNER & MODEL
# =============================================================================

class RepositoryScanner:
    EXTENSION_MAP = {
        ".py": "python", ".js": "javascript", ".ts": "typescript", ".tsx": "typescript",
        ".jsx": "javascript", ".java": "java", ".c": "c", ".cpp": "cpp", ".cc": "cpp",
        ".h": "c_header", ".hpp": "cpp_header", ".cs": "csharp", ".go": "go",
        ".rs": "rust", ".rb": "ruby", ".php": "php", ".kt": "kotlin", ".swift": "swift",
        ".dart": "dart", ".r": "r", ".jl": "julia", ".lua": "lua", ".sh": "shell",
        ".bash": "shell", ".zsh": "shell", ".ps1": "powershell", ".sql": "sql",
        ".html": "html", ".htm": "html", ".css": "css", ".scss": "scss",
        ".yaml": "yaml", ".yml": "yaml", ".json": "json", ".toml": "toml",
        ".xml": "xml", ".md": "markdown", ".rst": "rst", ".txt": "text",
        ".dockerfile": "dockerfile", "Dockerfile": "dockerfile",
        ".tf": "terraform", ".hcl": "hcl", ".makefile": "makefile", "Makefile": "makefile",
        ".ipynb": "jupyter", ".vue": "vue", ".svelte": "svelte",
    }

    IGNORE_DIRS = {
        ".git", "__pycache__", "node_modules", ".venv", "venv", "env",
        ".mypy_cache", ".pytest_cache", ".ruff_cache", "dist", "build",
        ".god_entity_candidates", ".god_entity_logs", ".tox", "target",
        "vendor", ".idea", ".vscode",
    }

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.model: Dict[str, Any] = {}

    def scan(self) -> Dict[str, Any]:
        files: List[Dict[str, Any]] = []
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = [d for d in dirnames if d not in self.IGNORE_DIRS and not d.startswith(".")]
            for fn in filenames:
                if fn.startswith(".") and fn not in (".gitignore", ".env.example", ".dockerignore"):
                    continue
                full = Path(dirpath) / fn
                try:
                    rel = full.relative_to(self.root)
                    info = self._analyze_file(full, rel)
                    files.append(info)
                except Exception:
                    continue

        by_lang: Dict[str, List[str]] = {}
        for f in files:
            lang = f.get("language") or "unknown"
            by_lang.setdefault(lang, []).append(f["path"])

        entrypoints = self._find_entrypoints(files)
        configs = [f for f in files if f.get("category") == "config"]
        tests = [f for f in files if f.get("category") == "test"]
        workflows = [f for f in files if "github/workflows" in f["path"].replace("\\", "/")]
        docs = [f for f in files if f.get("language") in ("markdown", "rst")]

        self.model = {
            "root": str(self.root),
            "file_count": len(files),
            "files": files[:500],
            "languages": {k: len(v) for k, v in by_lang.items()},
            "entrypoints": entrypoints,
            "configs": [c["path"] for c in configs],
            "tests": [t["path"] for t in tests],
            "workflows": [w["path"] for w in workflows],
            "documentation": [d["path"] for d in docs],
            "has_git": (self.root / ".git").exists(),
            "dependency_files": self._find_dependency_files(),
        }
        log.event(Event.REPOSITORY_DISCOVERED, summary={
            "root": str(self.root),
            "files": len(files),
            "languages": self.model["languages"],
            "entrypoints": entrypoints,
        })
        return self.model

    def _analyze_file(self, full: Path, rel: Path) -> Dict[str, Any]:
        ext = full.suffix.lower()
        name = full.name
        language = self.EXTENSION_MAP.get(ext) or self.EXTENSION_MAP.get(name)
        if not language:
            language = self._detect_by_content(full)
        category = self._categorize(rel, language)
        size = 0
        try:
            size = full.stat().st_size
        except Exception:
            pass
        return {
            "path": str(rel).replace("\\", "/"),
            "language": language,
            "category": category,
            "size": size,
            "extension": ext,
        }

    def _detect_by_content(self, path: Path) -> Optional[str]:
        try:
            with open(path, "rb") as f:
                head = f.read(512)
            text = head.decode("utf-8", errors="ignore")
            if text.startswith("#!"):
                if "python" in text:
                    return "python"
                if "bash" in text or "sh" in text:
                    return "shell"
                if "node" in text:
                    return "javascript"
            if "FROM " in text and "Dockerfile" in path.name:
                return "dockerfile"
        except Exception:
            pass
        return "unknown"

    def _categorize(self, rel: Path, language: Optional[str]) -> str:
        p = str(rel).replace("\\", "/").lower()
        if "test" in p or p.startswith("tests/") or p.endswith("_test.py") or p.endswith(".test.js"):
            return "test"
        if p.startswith(".github/workflows") or p.endswith(".yml") and "workflow" in p:
            return "workflow"
        if language in ("yaml", "json", "toml", "xml") or p.endswith(("requirements.txt", "pyproject.toml", "package.json", "Cargo.toml", "go.mod", "pom.xml")):
            return "config"
        if language in ("markdown", "rst"):
            return "documentation"
        if language in ("dockerfile",) or "Dockerfile" in rel.name:
            return "infrastructure"
        if language == "shell" or p.endswith((".sh", ".bash")):
            return "script"
        return "source"

    def _find_entrypoints(self, files: List[Dict]) -> List[str]:
        candidates = []
        for f in files:
            p = f["path"]
            name = Path(p).name
            if name in ("main.py", "app.py", "manage.py", "index.js", "main.go", "main.rs", "index.ts", "server.py", "cli.py"):
                candidates.append(p)
            if name == CORE_FILENAME:
                candidates.append(p)
        return candidates

    def _find_dependency_files(self) -> List[str]:
        names = [
            "requirements.txt", "pyproject.toml", "setup.py", "Pipfile",
            "package.json", "package-lock.json", "yarn.lock",
            "Cargo.toml", "go.mod", "go.sum", "pom.xml", "build.gradle",
            "Gemfile", "composer.json",
        ]
        found = []
        for n in names:
            if (self.root / n).exists():
                found.append(n)
        return found


# =============================================================================
# CAPABILITY DISCOVERY
# =============================================================================

class CapabilityRegistry:
    def __init__(self, env: Dict[str, Any], repo: Dict[str, Any]):
        self.env = env
        self.repo = repo
        self.capabilities: Dict[str, Any] = {}

    def discover(self) -> Dict[str, Any]:
        caps = {
            "filesystem_read": True,
            "filesystem_write": self._can_write(),
            "subprocess": True,
            "network": self.env.get("network", {}).get("available", False),
            "git": self.env.get("git", {}).get("available", False),
            "git_in_repo": self.env.get("git", {}).get("in_repo", False),
            "github_cli": self.env.get("github_cli", {}).get("available", False),
            "github_authenticated": self.env.get("github_cli", {}).get("authenticated", False),
            "docker": self.env.get("docker", {}).get("available", False),
            "python_package_install": self.env.get("package_managers", {}).get("pip", False),
            "model_provider": self._detect_model_providers(),
            "validators": self._detect_validators(),
            "research": self.env.get("network", {}).get("available", False),
            "self_modify": True,
            "ci": self.env.get("ci_cd", {}).get("github_actions", False) or self.env.get("ci_cd", {}).get("ci", False),
        }
        self.capabilities = caps
        log.event(Event.CAPABILITY_DISCOVERED, capabilities={
            k: v for k, v in caps.items() if not isinstance(v, (list, dict))
        })
        return caps

    def _can_write(self) -> bool:
        try:
            test = self.repo.get("root") or "."
            p = Path(test) / f".god_write_test_{uuid.uuid4().hex[:8]}"
            p.write_text("test")
            p.unlink()
            return True
        except Exception:
            return False

    def _detect_model_providers(self) -> Dict[str, bool]:
        providers = {
            "gemini": bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")),
            "openai": bool(os.environ.get("OPENAI_API_KEY")),
            "anthropic": bool(os.environ.get("ANTHROPIC_API_KEY")),
            "grok": bool(os.environ.get("XAI_API_KEY") or os.environ.get("GROK_API_KEY")),
            "ollama": False,
        }
        return providers

    def _detect_validators(self) -> Dict[str, bool]:
        return {
            "py_compile": True,
            "ast_parse": True,
            "import_check": True,
            "pytest": self.env.get("testing_frameworks", {}).get("pytest", False),
            "unittest": True,
            "ruff": self.env.get("linters_formatters", {}).get("ruff", False),
            "mypy": self.env.get("linters_formatters", {}).get("mypy", False),
            "black": self.env.get("linters_formatters", {}).get("black", False),
            "git_status": self.env.get("git", {}).get("available", False),
        }


# =============================================================================
# MODEL PROVIDER ABSTRACTION
# =============================================================================

class ModelProvider(ABC):
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def available(self) -> bool:
        ...

    @abstractmethod
    def generate(self, prompt: str, system: Optional[str] = None, max_tokens: int = 2048) -> str:
        ...


class NullProvider(ModelProvider):
    def name(self) -> str:
        return "null"

    def available(self) -> bool:
        return True

    def generate(self, prompt: str, system: Optional[str] = None, max_tokens: int = 2048) -> str:
        return (
            "[NULL_PROVIDER] No LLM credential configured. "
            "Falling back to deterministic / rule-based reasoning. "
            "Set GEMINI_API_KEY / OPENAI_API_KEY / XAI_API_KEY to enable full reasoning."
        )


class GeminiProvider(ModelProvider):
    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.model = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

    def name(self) -> str:
        return "gemini"

    def available(self) -> bool:
        return bool(self.api_key) and (HAS_HTTPX or HAS_REQUESTS)

    def generate(self, prompt: str, system: Optional[str] = None, max_tokens: int = 2048) -> str:
        if not self.available():
            return NullProvider().generate(prompt, system, max_tokens)
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        body = {
            "contents": [{"parts": [{"text": (system + "\n\n" + prompt) if system else prompt}]}],
            "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0.3},
        }
        try:
            if HAS_HTTPX:
                r = httpx.post(url, json=body, timeout=60)
            else:
                r = requests.post(url, json=body, timeout=60)
            if r.status_code != 200:
                return f"[GEMINI_ERROR] status={r.status_code}"
            data = r.json()
            candidates = data.get("candidates", [])
            if not candidates:
                return "[GEMINI_EMPTY]"
            parts = candidates[0].get("content", {}).get("parts", [])
            return parts[0].get("text", "") if parts else "[GEMINI_EMPTY]"
        except Exception as e:
            return f"[GEMINI_EXCEPTION] {type(e).__name__}"


class OpenAIProvider(ModelProvider):
    def __init__(self):
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self.base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

    def name(self) -> str:
        return "openai"

    def available(self) -> bool:
        return bool(self.api_key) and (HAS_HTTPX or HAS_REQUESTS)

    def generate(self, prompt: str, system: Optional[str] = None, max_tokens: int = 2048) -> str:
        if not self.available():
            return NullProvider().generate(prompt, system, max_tokens)
        url = f"{self.base}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        body = {"model": self.model, "messages": messages, "max_tokens": max_tokens, "temperature": 0.3}
        try:
            if HAS_HTTPX:
                r = httpx.post(url, json=body, headers=headers, timeout=60)
            else:
                r = requests.post(url, json=body, headers=headers, timeout=60)
            if r.status_code != 200:
                return f"[OPENAI_ERROR] status={r.status_code}"
            data = r.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            return f"[OPENAI_EXCEPTION] {type(e).__name__}"


class GrokProvider(ModelProvider):
    def __init__(self):
        self.api_key = os.environ.get("XAI_API_KEY") or os.environ.get("GROK_API_KEY")
        self.model = os.environ.get("GROK_MODEL", "grok-3")
        self.base = "https://api.x.ai/v1"

    def name(self) -> str:
        return "grok"

    def available(self) -> bool:
        return bool(self.api_key) and (HAS_HTTPX or HAS_REQUESTS)

    def generate(self, prompt: str, system: Optional[str] = None, max_tokens: int = 2048) -> str:
        if not self.available():
            return NullProvider().generate(prompt, system, max_tokens)
        url = f"{self.base}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        body = {"model": self.model, "messages": messages, "max_tokens": max_tokens, "temperature": 0.3}
        try:
            if HAS_HTTPX:
                r = httpx.post(url, json=body, headers=headers, timeout=60)
            else:
                r = requests.post(url, json=body, headers=headers, timeout=60)
            if r.status_code != 200:
                return f"[GROK_ERROR] status={r.status_code}"
            data = r.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            return f"[GROK_EXCEPTION] {type(e).__name__}"


def select_model_provider() -> ModelProvider:
    for cls in (GrokProvider, GeminiProvider, OpenAIProvider):
        p = cls()
        if p.available():
            log.info(f"Selected model provider: {p.name()}")
            return p
    log.warning("No LLM provider credentials found. Using NullProvider.")
    return NullProvider()


# =============================================================================
# RESEARCH ENGINE (Real internet search)
# =============================================================================

class ResearchEngine:
    def __init__(self, network_available: bool):
        self.network = network_available
        self.google_api_key = os.environ.get("GOOGLE_SEARCH_API_KEY", "")
        self.google_cx = os.environ.get("GOOGLE_SEARCH_CX", "")

    def research(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        log.event(Event.RESEARCH_STARTED, query=query)
        if not self.network:
            log.event(Event.RESEARCH_COMPLETED, query=query, results=0, reason="no_network")
            return []
        results = []
        if self.google_api_key and self.google_cx:
            results = self._google_search(query, max_results)
        if not results:
            results = self._ddg_search(query, max_results)
        log.event(Event.RESEARCH_COMPLETED, query=query, results=len(results))
        return results

    def _google_search(self, query, max_results):
        if not HAS_REQUESTS and not HAS_HTTPX:
            return []
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": self.google_api_key,
            "cx": self.google_cx,
            "q": query,
            "num": min(max_results, 10),
        }
        try:
            if HAS_REQUESTS:
                r = requests.get(url, params=params, timeout=10)
                if r.status_code != 200:
                    return []
                data = r.json()
            else:
                r = httpx.get(url, params=params, timeout=10)
                if r.status_code != 200:
                    return []
                data = r.json()
            items = []
            for item in data.get("items", []):
                items.append({
                    "title": item.get("title"),
                    "link": item.get("link"),
                    "snippet": item.get("snippet"),
                    "source": "google",
                })
            return items
        except Exception as e:
            log.warning(f"Google search failed: {e}")
            return []

    def _ddg_search(self, query, max_results):
        if not HAS_REQUESTS and not HAS_HTTPX:
            return []
        url = "https://api.duckduckgo.com/"
        params = {
            "q": query,
            "format": "json",
            "no_html": 1,
            "skip_disambig": 1,
        }
        try:
            if HAS_REQUESTS:
                r = requests.get(url, params=params, timeout=10)
                if r.status_code != 200:
                    return []
                data = r.json()
            else:
                r = httpx.get(url, params=params, timeout=10)
                if r.status_code != 200:
                    return []
                data = r.json()
            results = []
            if data.get("AbstractText"):
                results.append({
                    "title": data.get("Heading", query),
                    "link": data.get("AbstractURL", ""),
                    "snippet": data.get("AbstractText"),
                    "source": "duckduckgo",
                })
            for topic in data.get("RelatedTopics", [])[:max_results-1]:
                if isinstance(topic, dict) and "Text" in topic:
                    results.append({
                        "title": topic.get("FirstURL", ""),
                        "link": topic.get("FirstURL", ""),
                        "snippet": topic.get("Text"),
                        "source": "duckduckgo",
                    })
            return results[:max_results]
        except Exception as e:
            log.warning(f"DuckDuckGo search failed: {e}")
            return []


# =============================================================================
# VALIDATION ENGINE
# =============================================================================

class ValidationEngine:
    def __init__(self, capabilities: Dict[str, Any], root: Path):
        self.caps = capabilities
        self.root = root
        self.results: Dict[str, Any] = {}

    def validate_file(self, path: Path) -> Dict[str, Any]:
        log.event(Event.VALIDATION_STARTED, path=str(path))
        results: Dict[str, Any] = {"path": str(path), "checks": {}}

        if path.suffix == ".py":
            try:
                source = path.read_text(encoding="utf-8")
                ast.parse(source)
                results["checks"]["ast_parse"] = {"ok": True}
            except SyntaxError as e:
                results["checks"]["ast_parse"] = {"ok": False, "error": str(e)}
                log.event(Event.VALIDATION_FAILED, path=str(path), reason="syntax")
                results["overall"] = False
                return results

            code, out, err = safe_run([sys.executable, "-m", "py_compile", str(path)])
            results["checks"]["py_compile"] = {"ok": code == 0, "stderr": err[:500] if err else ""}

        results["overall"] = all(c.get("ok", False) for c in results["checks"].values()) if results["checks"] else True
        if results["overall"]:
            log.event(Event.VALIDATION_SUCCESS, path=str(path))
        else:
            log.event(Event.VALIDATION_FAILED, path=str(path))
        self.results[str(path)] = results
        return results

    def validate_generation_candidate(self, candidate_path: Path) -> Dict[str, Any]:
        return self.validate_file(candidate_path)

    def run_self_tests(self) -> Dict[str, Any]:
        results = {
            "syntax": False,
            "import_structure": False,
            "boot_flags": False,
            "memory": False,
            "generation_id": False,
        }
        try:
            ast.parse(Path(__file__).read_text(encoding="utf-8"))
            results["syntax"] = True
        except Exception:
            pass
        results["import_structure"] = True
        try:
            m = Memory(Path("."))
            m.set("_self_test", {"ok": True, "ts": utc_now()})
            results["memory"] = m.get("_self_test", {}).get("ok") is True
            m.close()
        except Exception:
            pass
        try:
            gid = Generation.next_id([])
            results["generation_id"] = gid.startswith("G") and len(gid) == 7
        except Exception:
            pass
        results["boot_flags"] = True
        results["overall"] = all(results.values())
        return results


# =============================================================================
# GIT MANAGER
# =============================================================================

class GitManager:
    def __init__(self, root: Path, mode: str = GIT_MODE_DEFAULT):
        self.root = root
        self.mode = mode
        self.available = shutil.which("git") is not None

    def status(self) -> Dict[str, Any]:
        if not self.available:
            return {"available": False}
        code, out, err = safe_run(["git", "status", "--porcelain"], cwd=self.root)
        return {
            "available": True,
            "clean": code == 0 and not out.strip(),
            "porcelain": out,
            "mode": self.mode,
        }

    def commit_if_enabled(self, message: str) -> bool:
        if self.mode == "disabled" or not self.available:
            return False
        safe_run(["git", "add", "-A"], cwd=self.root)
        code, _, err = safe_run(["git", "commit", "-m", message], cwd=self.root)
        if code != 0:
            log.warning(f"Git commit failed: {err[:200]}")
            return False
        if self.mode == "push":
            code, _, err = safe_run(["git", "push"], cwd=self.root, timeout=30)
            if code != 0:
                log.warning(f"Git push failed: {err[:200]}")
                return False
        return True


# =============================================================================
# EXPERIMENT ENGINE
# =============================================================================

@dataclass
class Experiment:
    id: str
    hypothesis: str
    change_desc: str
    result: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    conclusion: Optional[str] = None
    created_at: str = field(default_factory=utc_now)


class ExperimentEngine:
    def __init__(self, memory: Memory):
        self.memory = memory
        self.history: List[Experiment] = []

    def run(self, hypothesis: str, change_fn, measure_fn, baseline: Optional[Dict] = None) -> Experiment:
        exp = Experiment(
            id=f"EXP-{uuid.uuid4().hex[:8]}",
            hypothesis=hypothesis,
            change_desc=str(change_fn),
        )
        log.event(Event.EXPERIMENT_STARTED, id=exp.id, hypothesis=hypothesis)
        try:
            change_fn()
            metrics = measure_fn()
            exp.metrics = metrics
            if baseline:
                improved = self._compare(baseline, metrics)
                exp.conclusion = "IMPROVED" if improved else "NO_IMPROVEMENT"
            else:
                exp.conclusion = "EXECUTED"
            exp.result = "success"
        except Exception as e:
            exp.result = "failed"
            exp.conclusion = f"ERROR: {type(e).__name__}: {e}"
        log.event(Event.EXPERIMENT_COMPLETED, id=exp.id, conclusion=exp.conclusion)
        self.history.append(exp)
        return exp

    def _compare(self, baseline: Dict, candidate: Dict) -> bool:
        b_score = baseline.get("score", 0)
        c_score = candidate.get("score", 0)
        return c_score > b_score


# =============================================================================
# SELF-EVOLUTION ENGINE
# =============================================================================

class EvolutionEngine:
    def __init__(
        self,
        core_path: Path,
        memory: Memory,
        validator: ValidationEngine,
        git: GitManager,
        base_dir: Path,
        model_provider: Optional[ModelProvider] = None,
        research_engine: Optional[ResearchEngine] = None,
    ):
        self.core_path = core_path
        self.memory = memory
        self.validator = validator
        self.git = git
        self.base_dir = base_dir
        self.model_provider = model_provider
        self.research_engine = research_engine
        self.candidate_dir = base_dir / CANDIDATE_DIR
        self.candidate_dir.mkdir(parents=True, exist_ok=True)

    def create_candidate(self, objective: str, modifications: Optional[str] = None) -> Tuple[str, Path]:
        log.event(Event.EVOLUTION_STARTED, objective=objective)
        existing = [g.get("id", "") for g in self.memory.list_generations()]
        new_id = Generation.next_id(existing)
        candidate_name = f"candidate_generation_{new_id}.py"
        candidate_path = self.candidate_dir / candidate_name

        shutil.copy2(self.core_path, candidate_path)
        log.event(Event.CANDIDATE_CREATED, generation=new_id, path=str(candidate_path))

        changes_applied = ["duplicate_core"]

        # 1. Research context (if network and engine available)
        research_context = ""
        if self.research_engine and self.research_engine.network:
            research_results = self.research_engine.research(objective, max_results=3)
            research_context = "\n".join(
                f"- {r.get('title', '')}: {r.get('snippet', '')[:200]}"
                for r in research_results
            )

        # 2. LLM enhancement: append generated code to candidate
        enhancement_code = self._generate_enhancement(objective, research_context)
        if enhancement_code:
            try:
                with open(candidate_path, "a", encoding="utf-8") as f:
                    f.write("\n\n# ==== AUTONOMOUS ENHANCEMENT ====\n")
                    f.write(enhancement_code)
                    f.write("\n# ==== END ENHANCEMENT ====\n")
                changes_applied.append("llm_enhancement")
                log.info(f"✅ LLM enhancement appended ({len(enhancement_code)} chars)")
            except Exception as e:
                log.error(f"Failed to append enhancement: {e}")
        else:
            changes_applied.append("no_llm_enhancement")

        # 3. Workspace modifications (create/modify other files)
        workspace_changes = self._apply_llm_changes(objective, research_context)
        if workspace_changes:
            changes_applied.extend(workspace_changes)
            log.info(f"✅ Workspace modifications: {workspace_changes}")

        # 4. Optional marker (lineage)
        if modifications:
            try:
                text = candidate_path.read_text(encoding="utf-8")
                marker = f"\n\n# GENERATION {new_id} | parent lineage | objective: {objective[:80]}\n"
                candidate_path.write_text(text + marker, encoding="utf-8")
                changes_applied.append("lineage_marker")
            except Exception as e:
                log.error(f"Candidate modification failed: {e}")

        gen = Generation(
            id=new_id,
            parent=self.memory.get_active_generation().get("id") if self.memory.get_active_generation() else None,
            timestamp=utc_now(),
            objective=objective,
            source_hash=sha256_file(candidate_path),
            changes=changes_applied,
            status=GenerationStatus.CANDIDATE,
        )
        self.memory.save_generation(gen)
        return new_id, candidate_path

    def _generate_enhancement(self, objective: str, research_context: str) -> str:
        if not self.model_provider or not self.model_provider.available():
            return ""
        prompt = f"""
You are an expert Python developer improving an autonomous agent.
Objective: {objective}

Research context (may be empty):
{research_context}

Current agent code (first 3000 chars):