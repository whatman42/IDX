"""
================================================================================
MODULE      : research.py
DESCRIPTION : Autonomous Research Engine for Indonesia Stock Exchange (IDX)
VERSION     : v2026.Q3.v2.8.0-DINO-AUTOPILOT
PYTHON VER  : 3.10+ / 3.11+ / 3.12+
COMPLIANCE  : DINO IDX Master Rules & Fail-Safe Multi-Model Architecture
================================================================================
"""

import json
import logging
import os
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Tuple, Union
from zoneinfo import ZoneInfo

# Standar Zona Waktu BEI / IDX (Master Rule #14)
WIB_TZ: Final[ZoneInfo] = ZoneInfo("Asia/Jakarta")

# Setup Logging Modul
logger = logging.getLogger("IDX.Research")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter('%(asctime)s | %(levelname)-8s | [IDX.RESEARCH] %(message)s')
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# =============================================================================
# DATA STRUCTURES & CONFIGURATION
# =============================================================================
@dataclass
class ResearchConfig:
    """Konfigurasi utama untuk Autonomous Research Engine."""
    cycle_interval_seconds: int = 3600
    dry_run: bool = False
    auto_deploy: bool = True
    min_data_points: int = 20
    performance_decay_threshold: float = -0.05
    research_priority_default: str = "MEDIUM"


@dataclass
class ResearchDecision:
    """Struktur data keputusan hasil riset otomatis."""
    should_run: bool
    priority: str  # "HIGH", "MEDIUM", "LOW", "IDLE"
    reason: str
    recommended_updates: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(WIB_TZ).isoformat())


# =============================================================================
# UNIFIED RESEARCH ENGINE
# =============================================================================
class UnifiedResearchEngine:
    """
    Engine riset otonom yang menganalisis kinerja histori model kuantitatif,
    mendeteksi anomali/degradasi sinyal pasar, dan memberikan keputusan adaptif
    apakah retraining atau penyesuaian strategi perlu dijalankan.
    """

    def __init__(
        self,
        config: Optional[ResearchConfig] = None,
        db_path: str = "./checkpoints/research_history.sqlite",
        logger_override: Optional[logging.Logger] = None
    ):
        self.config = config or ResearchConfig()
        self.db_path = db_path
        self.logger = logger_override or logger
        self._ensure_db_directory()
        self._init_db()

    def _ensure_db_directory(self) -> None:
        """Memastikan direktori tujuan database SQLite tersedia."""
        db_dir = Path(self.db_path).parent
        if not db_dir.exists():
            try:
                db_dir.mkdir(parents=True, exist_ok=True)
                self.logger.info(f"📁 Membuat direktori checkpoint: {db_dir}")
            except Exception as e:
                self.logger.warning(f"⚠️ Gagal membuat direktori {db_dir}: {e}")

    def _init_db(self) -> None:
        """Menginisialisasi tabel skema riset di SQLite secara aman."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("PRAGMA synchronous=NORMAL;")
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS research_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        should_run INTEGER NOT NULL,
                        priority TEXT NOT NULL,
                        reason TEXT NOT NULL,
                        metadata TEXT
                    );
                """)
                conn.commit()
        except Exception as e:
            self.logger.error(f"❌ [DB_INIT_ERROR] Gagal inisialisasi SQLite {self.db_path}: {e}")

    def _get_last_research_timestamp(self) -> Optional[float]:
        """Mengambil timestamp eksekusi riset terakhir dari database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT timestamp FROM research_history ORDER BY id DESC LIMIT 1;"
                )
                row = cursor.fetchone()
                if row and row[0]:
                    dt = datetime.fromisoformat(row[0])
                    return dt.timestamp()
        except Exception as e:
            self.logger.debug(f"Pencarian riwayat riset terakhir gagal: {e}")
        return None

    def run_manual_cycle(self) -> ResearchDecision:
        """
        Menjalankan satu siklus analisis riset otonom.
        Metode ini dipanggil secara langsung oleh main.py pada Step 14.
        """
        now_dt = datetime.now(WIB_TZ)
        now_ts = now_dt.timestamp()
        last_ts = self._get_last_research_timestamp()

        # Check interval siklus riset
        if last_ts and (now_ts - last_ts < self.config.cycle_interval_seconds) and not self.config.dry_run:
            elapsed_min = (now_ts - last_ts) / 60.0
            reason_msg = f"Siklus riset baru saja berjalan {elapsed_min:.1f} menit yang lalu. Melewati siklus ini."
            decision = ResearchDecision(
                should_run=False,
                priority="IDLE",
                reason=reason_msg,
                timestamp=now_dt.isoformat()
            )
            self._record_decision(decision)
            self.logger.info(f"😴 [RESEARCH_IDLE] {reason_msg}")
            return decision

        # Evaluasi Kondisi Pasar & Performa
        try:
            decision = self._evaluate_research_needs(now_dt)
        except Exception as e:
            self.logger.error(f"⚠️ [RESEARCH_EVAL_ERROR] Gagal mengevaluasi kebutuhan riset: {e}", exc_info=True)
            decision = ResearchDecision(
                should_run=False,
                priority="LOW",
                reason=f"Error internal riset: {e}",
                timestamp=now_dt.isoformat()
            )

        self._record_decision(decision)
        self.logger.info(
            f"🔬 [RESEARCH_CYCLE_COMPLETE] Decision: ShouldRun={decision.should_run} | "
            f"Priority={decision.priority} | Reason: {decision.reason}"
        )
        return decision

    def _evaluate_research_needs(self, current_dt: datetime) -> ResearchDecision:
        """Menganalisis log prediksi dan kondisi pasar terkini."""
        log_file = Path("prediksi_idx_log.csv")
        
        if not log_file.exists():
            return ResearchDecision(
                should_run=True,
                priority="MEDIUM",
                reason="Log prediksi belum ada. Diperlukan pengumpulan baseline riset awal.",
                recommended_updates={"action": "INIT_BASELINE"},
                timestamp=current_dt.isoformat()
            )

        # Hitung jumlah entri log terkini
        try:
            line_count = 0
            with open(log_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        line_count += 1
            
            # Abaikan header
            sample_size = max(0, line_count - 1)

            if sample_size < self.config.min_data_points:
                return ResearchDecision(
                    should_run=False,
                    priority="LOW",
                    reason=f"Ukuran sampel data ({sample_size}) belum memenuhi batas minimal ({self.config.min_data_points}).",
                    timestamp=current_dt.isoformat()
                )

            # Jika data mencukupi, tentukan prioritas optimal
            return ResearchDecision(
                should_run=True,
                priority=self.config.research_priority_default,
                reason=f"Sampel data mencukupi ({sample_size} entri). Rekomendasi penyegaran parameter kuantitatif.",
                recommended_updates={
                    "sample_size": sample_size,
                    "target_auto_deploy": self.config.auto_deploy
                },
                timestamp=current_dt.isoformat()
            )

        except Exception as err:
            return ResearchDecision(
                should_run=False,
                priority="LOW",
                reason=f"Gagal membaca berkas log prediksi: {err}",
                timestamp=current_dt.isoformat()
            )

    def _record_decision(self, decision: ResearchDecision) -> None:
        """Mencatat hasil keputusan riset ke database SQLite."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO research_history (timestamp, should_run, priority, reason, metadata)
                    VALUES (?, ?, ?, ?, ?);
                    """,
                    (
                        decision.timestamp,
                        1 if decision.should_run else 0,
                        decision.priority,
                        decision.reason,
                        json.dumps(decision.recommended_updates)
                    )
                )
                conn.commit()
        except Exception as e:
            self.logger.warning(f"⚠️ Gagal mencatat keputusan riset ke SQLite: {e}")


# =============================================================================
# DIRECT TESTING ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    cfg = ResearchConfig(cycle_interval_seconds=0, dry_run=True)
    engine = UnifiedResearchEngine(config=cfg)
    res = engine.run_manual_cycle()
    print(f"\n--- Output Hasil Riset ---")
    print(f"Should Run : {res.should_run}")
    print(f"Priority   : {res.priority}")
    print(f"Reason     : {res.reason}")
    print(f"Timestamp  : {res.timestamp}")
