"""
================================================================================
MODULE      : main.py
DESCRIPTION : Single-Run Production Orchestrator for Indonesia Stock Exchange (IDX)
VERSION     : 2026.Q3.v2.5.0-PRODUCTION-FINAL (Aligned with Portfolio, Risk, & Reporting)
PYTHON VER  : 3.11+ / 3.12+
ARCHITECTURE: Clean Architecture, Deterministic Pipeline Engine for Actions/Cron
================================================================================
"""

import gc
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

import numpy as np
import polars as pl
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

# =============================================================================
# 1. FAIL-FAST STATIC MODULE IMPORTS
# =============================================================================
try:
    import data
    import features
    import machine_learning
    import prediction

    try:
        import signal_idx
    except ImportError:
        import signal_crypto as signal_idx

    import validation
    import risk
    from portfolio import UnifiedPortfolioEngine, normalize_idx_symbol
    import simulation
    import evaluation
    import self_learning

    try:
        import autonomous_engine_idx
    except ImportError:
        import autonomous_engine_crypto as autonomous_engine_idx

    import research
    import reporting
    import monitoring
    import storage
except ImportError as err:
    sys.stderr.write(f"🛑 [CRITICAL_BOOTSTRAP_ERROR] Failed to load dependency module: {err}\n")
    sys.exit(1)


# =============================================================================
# DEFAULT TICKER UNIVERSE & BEI CONSTANTS
# =============================================================================
DEFAULT_IDX_TICKERS: List[str] = [
    "AALI.JK", "ABBA.JK", "ABMM.JK", "ACES.JK", "ACST.JK", "ADHI.JK", "ADMF.JK", "ADMG.JK", "ADRO.JK", "AGII.JK",
    "AGRO.JK", "AKRA.JK", "AKSI.JK", "ALDO.JK", "ALKA.JK", "ALMI.JK", "AMAG.JK", "AMRT.JK", "ANDI.JK", "ANJT.JK",
    "ANTM.JK", "APIC.JK", "APLN.JK", "ARNA.JK", "ARTA.JK", "ASII.JK", "ASJT.JK", "ASRI.JK", "ASSA.JK", "ATIC.JK",
    "AUTO.JK", "BABP.JK", "BACA.JK", "BAEK.JK", "BALI.JK", "BAPA.JK", "BAPI.JK", "BATA.JK", "BBCA.JK", "BBHI.JK",
    "BBKP.JK", "BBNI.JK", "BBRI.JK", "BBTN.JK", "BINA.JK", "BIPP.JK", "BISI.JK", "BJBR.JK", "BJTM.JK", "BKDP.JK",
    "BKSL.JK", "BKSW.JK", "BLTA.JK", "BLTZ.JK", "BLUE.JK", "BMAS.JK", "BMRI.JK", "BMSR.JK", "BMTR.JK", "BNBA.JK",
    "BNGA.JK", "BNII.JK", "BNLI.JK", "BOBA.JK", "BOGA.JK", "BOLT.JK", "BOSS.JK", "BPFI.JK", "BPII.JK", "BPTR.JK",
    "BRAM.JK", "BRIS.JK", "BRMS.JK", "BRPT.JK", "BSDE.JK", "BSSR.JK", "BTEK.JK", "BTEL.JK", "BTON.JK", "BTPN.JK",
    "BTPS.JK", "BUDI.JK", "BUVA.JK", "BVSN.JK", "BYAN.JK", "CAKK.JK", "CAMP.JK", "CANI.JK", "CARS.JK", "CASA.JK",
    "CASH.JK", "CBMF.JK", "CEKA.JK", "CENT.JK", "CFIN.JK", "CINT.JK", "CITA.JK", "CITY.JK", "CLAY.JK", "CLEO.JK",
    "CLPI.JK", "CMNP.JK", "CMRY.JK", "CMPP.JK", "CNKO.JK", "CNTX.JK", "COCO.JK", "COWL.JK", "CPIN.JK", "CPRO.JK",
    "CSAP.JK", "CSIS.JK", "CTRA.JK", "CTTH.JK", "DEAL.JK", "DEFI.JK", "DEPO.JK", "DGIK.JK", "DIGI.JK", "DILD.JK",
    "DIVA.JK", "DKFT.JK", "DLTA.JK", "DMAS.JK", "DNAR.JK", "DOID.JK", "DSSA.JK", "DUCK.JK", "DUTI.JK", "DVLA.JK",
    "DYAN.JK", "EAST.JK", "ECII.JK", "EDGE.JK", "EKAD.JK", "ELSA.JK", "EMDE.JK", "EMTK.JK", "ENRG.JK", "ENZO.JK",
    "EPAC.JK", "ERA.JK", "ERAA.JK", "ESSA.JK", "ESTA.JK", "FAST.JK", "FASW.JK", "FILM.JK", "FISH.JK", "FITT.JK",
    "FLMC.JK", "FMII.JK", "FOOD.JK", "FORU.JK", "FPNI.JK", "GAMA.JK", "GEMS.JK", "GGRM.JK", "GJTL.JK", "GLVA.JK",
    "GOOD.JK", "GPRA.JK", "GSMF.JK", "GZCO.JK", "HDTX.JK", "HERO.JK", "HEXA.JK", "HITS.JK", "HKMU.JK", "HMSP.JK",
    "HOKI.JK", "HRUM.JK", "ICBP.JK", "IDPR.JK", "IFII.JK", "INAF.JK", "INAI.JK", "INCF.JK", "INCI.JK", "INCO.JK",
    "INDF.JK", "INDY.JK", "INKP.JK", "INPP.JK", "INTA.JK", "INTD.JK", "INTP.JK", "IPCC.JK", "IPCM.JK", "IPOL.JK",
    "IPTV.JK", "IRRA.JK", "ISAT.JK", "ITMG.JK", "JAST.JK", "JAWA.JK", "JGLE.JK", "JKON.JK", "JPFA.JK", "JSMR.JK",
    "KAEF.JK", "KARW.JK", "KBLI.JK", "KBLM.JK", "KDSI.JK", "KIAS.JK", "KIJA.JK", "KINO.JK", "KLBF.JK", "KMTR.JK",
    "LEAD.JK", "LIFE.JK", "LINK.JK", "LPKR.JK", "LPPF.JK", "LUCK.JK", "MAIN.JK", "MAPB.JK", "MAPA.JK", "MASA.JK",
    "MCAS.JK", "MDKA.JK", "MEDC.JK", "MFIN.JK", "MIDI.JK", "MIRA.JK", "MITI.JK", "MKNT.JK", "MLPL.JK", "MLPT.JK",
    "MNCN.JK", "MPPA.JK", "MPRO.JK", "MTDL.JK", "MYOR.JK", "NATO.JK", "NELY.JK", "NFCX.JK", "NISP.JK", "NRCA.JK",
    "OKAS.JK", "OMRE.JK", "PANI.JK", "PBID.JK", "PCAR.JK", "PDES.JK", "PEHA.JK", "PGAS.JK", "PJAA.JK", "PMJS.JK",
    "PNBN.JK", "PNLF.JK", "POLA.JK", "POOL.JK", "PPGL.JK", "PPRO.JK", "PSSI.JK", "PTBA.JK", "PTIS.JK", "PWON.JK",
    "RAJA.JK", "RDTX.JK", "REAL.JK", "RICY.JK", "RIGS.JK", "ROTI.JK", "SAME.JK", "SAPX.JK", "SCCO.JK", "SCMA.JK",
    "SIDO.JK", "SILO.JK", "SIMP.JK", "SIPD.JK", "SMBR.JK", "SMCB.JK", "SMDR.JK", "SMGR.JK", "SMKL.JK", "SMRA.JK",
    "SMSM.JK", "SOCI.JK", "SQMI.JK", "SRAJ.JK", "SRTG.JK", "STAA.JK", "STTP.JK", "TALF.JK", "TARA.JK", "TBIG.JK",
    "TCID.JK", "TINS.JK", "TKIM.JK", "TLKM.JK", "TOTO.JK", "TPIA.JK", "TRIM.JK", "TURI.JK", "ULTJ.JK", "UNIC.JK",
    "UNTR.JK", "UNVR.JK", "WIKA.JK", "WSBP.JK", "WSKT.JK", "YPAS.JK", "ZINC.JK"
]

DEFAULT_SIGNAL_HORIZONS: List[int] = [1, 3, 5, 10, 20, 60]
DEFAULT_DRYRUN_CAPITAL: float = 10_000_000.0  # Rp 10.000.000 IDR Modal Virtual Simulasi


# =============================================================================
# 2. HTTP SESSION UTILITIES
# =============================================================================
def create_http_session(
    total_retries: int = 5,
    backoff_factor: float = 2.0,
    status_forcelist: Tuple[int, ...] = (408, 429, 500, 502, 503, 504)
) -> requests.Session:
    """Membangun HTTP Session terpusat untuk penarikan data & Telegram Webhooks."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
        "Accept": "application/json, text/html, */*",
    })

    proxy_url = os.getenv("IDX_PROXY_URL", os.getenv("HTTPS_PROXY", os.getenv("HTTP_PROXY", ""))).strip()
    if proxy_url:
        session.proxies = {"http": proxy_url, "https": proxy_url}

    retries = Retry(
        total=total_retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        raise_on_status=False
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=20)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


# =============================================================================
# 3. MODE ISOLATION & CONFIGURATION (SIMULATION & SIGNAL ONLY)
# =============================================================================
def _determine_execution_mode() -> str:
    """
    Menentukan mode eksekusi sederhana:
    - 'reset-dryrun' : Reset saldo simulasi ke modal awal.
    - 'live-signal'  : Analisis data terkini & kirim notifikasi sinyal ke Telegram.
    - 'dry-run'      : Full Simulasi / Paper Trading internal.
    """
    if "--reset-dryrun" in sys.argv:
        return "reset-dryrun"
    if "--live" in sys.argv or "--live-signal" in sys.argv:
        return "live-signal"

    env_mode = os.getenv("EXECUTION_MODE", os.getenv("TRADING_MODE", "dry-run")).lower().strip()
    return "live-signal" if env_mode in ["live", "live-signal"] else "dry-run"


def _get_mode_suffix() -> str:
    mode = _determine_execution_mode()
    return "live" if mode == "live-signal" else "simulation"


def _get_mode_file_path(default_prefix: str, ext: str) -> str:
    suffix = _get_mode_suffix()
    return f"{default_prefix}_{suffix}.{ext}"


@dataclass
class OrchestratorConfig:
    EXECUTION_MODE: str = field(default_factory=_determine_execution_mode)
    STATE_SUFFIX: str = field(default_factory=_get_mode_suffix)

    LOCK_FILE: str = field(default_factory=lambda: os.getenv("BOT_LOCK_FILE", _get_mode_file_path("idx_signal_bot", "lock")))
    LOG_FILE: str = field(default_factory=lambda: os.getenv("BOT_LOG_FILE", _get_mode_file_path("idx_orchestrator", "log")))
    CHECKPOINT_FILE: str = field(default_factory=lambda: os.getenv("BOT_CHECKPOINT_FILE", _get_mode_file_path("checkpoint", "json")))

    IDX_TICKERS: List[str] = field(
        default_factory=lambda: [
            normalize_idx_symbol(t) for t in os.getenv("IDX_TICKERS", ",".join(DEFAULT_IDX_TICKERS)).split(",") if t.strip()
        ]
    )
    SIGNAL_HORIZONS: List[int] = field(default_factory=lambda: DEFAULT_SIGNAL_HORIZONS)
    BENCHMARK_TICKER: str = "^JKSE"

    CIRCUIT_FAILURE_THRESHOLD: int = 5
    CIRCUIT_RECOVERY_TIME_SEC: int = 60


# =============================================================================
# 4. DRY-RUN BALANCE & PORTFOLIO STATE RESET HANDLER
# =============================================================================
def reset_dryrun_balance(logger: logging.Logger) -> None:
    logger.info(f"🔄 [RESET] Memulai pemulihan modal virtual Dry-Run IDX ke Rp {DEFAULT_DRYRUN_CAPITAL:,.0f} IDR...")

    portfolio_state = {
        "cash_balance": DEFAULT_DRYRUN_CAPITAL,
        "total_equity": DEFAULT_DRYRUN_CAPITAL,
        "equity": DEFAULT_DRYRUN_CAPITAL,
        "cash": DEFAULT_DRYRUN_CAPITAL,
        "exposure_pct": 0.0,
        "return_pct": 0.0,
        "realized_pnl_accumulated": 0.0,
        "positions_count": 0,
        "active_positions_count": 0,
        "positions": {},
        "active_positions": [],
        "top_pick": "-",
        "transaction_history": [],
        "reset_event": True,
        "last_updated_utc": datetime.now(timezone.utc).isoformat()
    }

    state_file = "portfolio_simulation_state.json"
    try:
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(portfolio_state, f, indent=2)
        logger.info(f"✔ Berhasil menulis ulang {state_file} dengan modal Rp {DEFAULT_DRYRUN_CAPITAL:,.0f} IDR.")
    except Exception as e:
        logger.error(f"✖ Gagal memperbarui {state_file}: {e}", exc_info=True)

    files_to_clear = [
        "positions_dryrun_state.csv",
        "signal_history_dryrun.csv",
        "checkpoint_dryrun.json"
    ]
    for file_path in files_to_clear:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                logger.warning(f"⚠️ Tidak dapat menghapus berkas {file_path}: {e}")

    try:
        reporting_engine = reporting.UnifiedReportingEngine(mode="reset-dryrun")
        reporting_engine.send_telegram_broadcast(portfolio_data=portfolio_state)
        logger.info("✔ Berhasil mempublikasikan notifikasi Reset Portfolio ke Telegram.")
    except Exception as e:
        logger.warning(f"⚠️ Gagal mempublikasikan reset state ke Telegram: {e}", exc_info=True)


# =============================================================================
# 5. CIRCUIT BREAKER & INFRASTRUCTURE HELPERS
# =============================================================================
class CircuitBreaker:
    """Pattern CircuitBreaker aktif untuk perlindungan runtime eksternal."""
    def __init__(self, failure_threshold: int = 5, recovery_time_sec: int = 60, logger: Optional[logging.Logger] = None):
        self.failure_threshold = failure_threshold
        self.recovery_time_sec = recovery_time_sec
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.state = "CLOSED"
        self.logger = logger or logging.getLogger("CircuitBreaker")

    def record_success(self) -> None:
        if self.state != "CLOSED":
            self.logger.info("⚡ [CIRCUIT_BREAKER] Pemulihan berhasil! Mengubah status menjadi CLOSED.")
        self.failure_count = 0
        self.state = "CLOSED"

    def record_failure(self) -> None:
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            self.logger.error(f"🛑 [CIRCUIT_BREAKER] Ambang batas kegagalan ({self.failure_threshold}) tercapai! Status menjadi OPEN.")

    def can_execute(self) -> bool:
        if self.state == "CLOSED":
            return True
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_time_sec:
                self.state = "HALF-OPEN"
                self.logger.info("⚠️ [CIRCUIT_BREAKER] Waktu pemulihan berakhir. Status beralih ke HALF-OPEN.")
                return True
            return False
        return True


class CheckpointManager:
    """Mengelola status checkpoint eksekusi pipeline secara tertulis."""
    def __init__(self, filepath: str, logger: Optional[logging.Logger] = None):
        self.filepath = filepath
        self.logger = logger or logging.getLogger("CheckpointManager")

    def save_checkpoint(self, step_number: int, step_name: str, execution_id: str, extra_data: Optional[Dict[str, Any]] = None) -> None:
        try:
            payload = {
                "execution_id": execution_id,
                "completed_step": step_number,
                "step_name": step_name,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            if extra_data:
                payload["metadata"] = extra_data

            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            self.logger.error(f"⚠️ Gagal menyimpan checkpoint pada Step {step_number}: {e}", exc_info=True)

    def load_checkpoint(self) -> Optional[Dict[str, Any]]:
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"⚠️ Gagal membaca berkas checkpoint: {e}", exc_info=True)
                return None
        return None

    def clear(self) -> None:
        if os.path.exists(self.filepath):
            try:
                os.remove(self.filepath)
            except Exception as e:
                self.logger.warning(f"⚠️ Gagal membersihkan checkpoint: {e}")


class ProcessLocker:
    def __init__(self, lock_file: str):
        self.lock_file = lock_file
        self.fp: Optional[Any] = None

    def acquire(self) -> None:
        try:
            self.fp = open(self.lock_file, 'w')
            if os.name == 'posix':
                import fcntl
                fcntl.flock(self.fp, fcntl.LOCK_EX | fcntl.LOCK_NB)
            elif os.name == 'nt':
                import msvcrt
                msvcrt.locking(self.fp.fileno(), msvcrt.LK_NBLCK, 1)
        except (IOError, OSError) as e:
            raise PermissionError(f"Process locked! ({self.lock_file}): {e}")

    def release(self) -> None:
        if self.fp:
            try:
                if os.name == 'posix':
                    import fcntl
                    fcntl.flock(self.fp, fcntl.LOCK_UN)
                elif os.name == 'nt':
                    import msvcrt
                    msvcrt.locking(self.fp.fileno(), msvcrt.LK_UNLCK, 1)
            except Exception:
                pass
            finally:
                self.fp.close()
                if os.path.exists(self.lock_file):
                    try:
                        os.remove(self.lock_file)
                    except Exception:
                        pass


class StepContext:
    def __init__(self, step_number: int, step_name: str, logger: logging.Logger, critical: bool = True):
        self.step_number = step_number
        self.step_name = step_name
        self.logger = logger
        self.critical = critical
        self.start_time = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        self.logger.info(f"▶ [START STEP {self.step_number}] {self.step_name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.perf_counter() - self.start_time
        if exc_type is None:
            self.logger.info(f"✔ [SUCCESS STEP {self.step_number}] {self.step_name} completed in {elapsed:.4f}s.")
            return False
        
        self.logger.error(
            f"✖ [FAILED STEP {self.step_number}] {self.step_name} failed: {exc_val}",
            exc_info=(exc_type, exc_val, exc_tb)
        )
        return not self.critical


# =============================================================================
# 6. PRODUCTION ORCHESTRATOR ENGINE (SINGLE PASS 17-STEP PIPELINE)
# =============================================================================
class ProductionOrchestrator:
    def __init__(self):
        self.config = OrchestratorConfig()
        os.environ["EXECUTION_MODE"] = self.config.EXECUTION_MODE
        os.environ["STATE_SUFFIX"] = self.config.STATE_SUFFIX

        self.state: Dict[str, Any] = {}
        self.logger = self._setup_logging()
        self.http_session = create_http_session()
        self.locker = ProcessLocker(self.config.LOCK_FILE)
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=self.config.CIRCUIT_FAILURE_THRESHOLD,
            recovery_time_sec=self.config.CIRCUIT_RECOVERY_TIME_SEC,
            logger=self.logger
        )
        self.checkpoint_mgr = CheckpointManager(self.config.CHECKPOINT_FILE, logger=self.logger)

    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger("ProductionOrchestrator")
        if logger.handlers:
            return logger
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s')
        
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        if self.config.LOG_FILE:
            try:
                fh = logging.FileHandler(self.config.LOG_FILE)
                fh.setLevel(logging.DEBUG)
                fh.setFormatter(formatter)
                logger.addHandler(fh)
            except Exception as e:
                sys.stderr.write(f"⚠️ Gagal membuat handler log berkas: {e}\n")

        return logger

    def run(self) -> None:
        """Eksekusi tunggal dari Step 1 hingga Step 17."""
        if self.config.EXECUTION_MODE == "reset-dryrun":
            try:
                self.locker.acquire()
                reset_dryrun_balance(self.logger)
            finally:
                self.locker.release()
            return

        try:
            # Step 1: Bootstrap
            with StepContext(1, "Bootstrap & Ticker Validation", self.logger, critical=True):
                self._step_1_bootstrap()
                self.checkpoint_mgr.save_checkpoint(1, "Bootstrap", self.state['execution_id'])

            # Step 2: Data Ingestion Layer
            with StepContext(2, "Ingest Market Data via Integrated data.py Engine", self.logger, critical=True):
                self._step_2_data_layer()
                self.checkpoint_mgr.save_checkpoint(2, "Data Layer Ingestion", self.state['execution_id'])

            # Step 3: Feature Engineering
            with StepContext(3, "Feature Engineering", self.logger, critical=True):
                self._step_3_feature_engineering()
                self.checkpoint_mgr.save_checkpoint(3, "Feature Engineering", self.state['execution_id'])

            # Step 4: Machine Learning Inference
            with StepContext(4, "Machine Learning Inference", self.logger, critical=True):
                self._step_4_machine_learning()
                self.checkpoint_mgr.save_checkpoint(4, "Machine Learning", self.state['execution_id'])

            # Step 5: Multi-Horizon Prediction
            with StepContext(5, "Quant Multi-Horizon Prediction", self.logger, critical=True):
                self._step_5_prediction()
                self.checkpoint_mgr.save_checkpoint(5, "Prediction Pipeline", self.state['execution_id'])

            # Step 6: Signal Generation Engine
            with StepContext(6, "IDX Signal Engine Processing", self.logger, critical=True):
                self._step_6_signal_engine()
                self.checkpoint_mgr.save_checkpoint(6, "Signal Engine", self.state['execution_id'])

            # Step 7: Validation Engine
            with StepContext(7, "Validation Engine (Real Market Historical Returns)", self.logger, critical=False):
                self._step_7_validation()
                self.checkpoint_mgr.save_checkpoint(7, "Validation Engine", self.state['execution_id'])

            # Step 8: Risk Management
            with StepContext(8, "Risk Management Audit", self.logger, critical=True):
                self._step_8_risk_management()
                self.checkpoint_mgr.save_checkpoint(8, "Risk Management Audit", self.state['execution_id'])

            # Step 9: Portfolio Execution & Rebalancing (Paper Simulation)
            with StepContext(9, "Portfolio Allocation & Restricted Top 1 Execution (Simulated)", self.logger, critical=True):
                self._step_9_portfolio_execution()
                self.checkpoint_mgr.save_checkpoint(9, "Portfolio Allocation", self.state['execution_id'])

            # Step 10: Simulation Engine
            with StepContext(10, "Execution Layer Simulation", self.logger, critical=True):
                self._step_10_execution_layer()
                self.checkpoint_mgr.save_checkpoint(10, "Execution Simulation", self.state['execution_id'])

            # Step 11: Evaluation Engine
            with StepContext(11, "Quantitative Evaluation Engine", self.logger, critical=False):
                self._step_11_evaluation()
                self.checkpoint_mgr.save_checkpoint(11, "Evaluation Engine", self.state['execution_id'])

            # Step 12: Self-Learning Engine
            with StepContext(12, "Self-Learning Loop", self.logger, critical=False):
                self._step_12_self_learning()
                self.checkpoint_mgr.save_checkpoint(12, "Self-Learning Engine", self.state['execution_id'])

            # Step 13: Autonomous Feedback Loop
            with StepContext(13, "Autonomous Feedback Loop & Risk Control Gate", self.logger, critical=False):
                self._step_13_autonomous_engine()
                self.checkpoint_mgr.save_checkpoint(13, "Autonomous Feedback Engine", self.state['execution_id'])

            # Step 14: Autonomous Research Engine
            with StepContext(14, "Autonomous Research Engine", self.logger, critical=False):
                self._step_14_research()
                self.checkpoint_mgr.save_checkpoint(14, "Autonomous Research Engine", self.state['execution_id'])

            # Step 15: Telemetry & Monitoring Audit
            with StepContext(15, "System Telemetry & Hardware Diagnostics", self.logger, critical=False):
                self._step_15_monitoring()
                self.checkpoint_mgr.save_checkpoint(15, "Telemetry Monitoring Engine", self.state['execution_id'])

            # Step 16: Reporting & Telegram Notification Broadcast
            with StepContext(16, "Reporting & Telegram Notification Broadcast", self.logger, critical=False):
                self._step_16_reporting()
                self.checkpoint_mgr.save_checkpoint(16, "Reporting & Telegram Broadcast", self.state['execution_id'])

            # Step 17: Storage & Memory Cleanup
            with StepContext(17, "Storage Persistence & Memory Cleanup", self.logger, critical=True):
                self._step_17_storage()
                self.checkpoint_mgr.save_checkpoint(17, "Storage Persistence", self.state['execution_id'])

            self.logger.info("🎉 [PIPELINE_COMPLETE] Seluruh 17 Langkah Pipeline Eksekusi Berhasil Dijalankan.")

        finally:
            self.checkpoint_mgr.clear()
            try:
                self.locker.release()
            except Exception as e:
                self.logger.warning(f"⚠️ Gagal melepas lock file: {e}")

    # -------------------------------------------------------------------------
    # PIPELINE STEPS IMPLEMENTATION
    # -------------------------------------------------------------------------
    def _step_1_bootstrap(self) -> None:
        self.locker.acquire()
        self.state['execution_id'] = f"EXEC-IDX-{int(time.time())}"
        self.state['timestamp'] = datetime.now(timezone.utc).isoformat()
        self.config.IDX_TICKERS = [normalize_idx_symbol(t) for t in self.config.IDX_TICKERS if t]

    def _step_2_data_layer(self) -> None:
        if not self.circuit_breaker.can_execute():
            raise RuntimeError("🛑 [CIRCUIT_OPEN] Terlalu banyak kegagalan eksternal berturut-turut. Menghentikan penarikan data.")

        try:
            data_engine = data.UnifiedDataEngine(
                universe_file="universe.json",
                cache_dir=".cache",
                enable_cache=True
            )
            
            market_data = data_engine.load_and_prepare_market_data(
                symbols=self.config.IDX_TICKERS,
                use_cache=True
            )

            if market_data is None or market_data.height == 0:
                raise ValueError("Gagal memperoleh data pasar: DataFrame yang dihasilkan oleh UnifiedDataEngine kosong.")

            if "ticker" in market_data.columns and "asset" not in market_data.columns:
                market_data = market_data.rename({"ticker": "asset"})

            self.circuit_breaker.record_success()
            self.state['market_df'] = market_data
            
            latest_prices = {}
            latest_df = market_data.sort("date").group_by("asset").last()
            for row in latest_df.to_dicts():
                latest_prices[str(row["asset"])] = float(row.get("close", 0.0))
            self.state['latest_prices_dict'] = latest_prices

            self.logger.info(
                f"✔ [DATA_INGESTION_SUCCESS] Berhasil memuat {market_data.height} baris data pasar "
                f"untuk {market_data['asset'].n_unique()} saham."
            )

        except Exception as e:
            self.circuit_breaker.record_failure()
            self.logger.error(f"✖ [DATA_INGESTION_FAILED] Gagal mengeksekusi data layer: {e}")
            raise e

    def _step_3_feature_engineering(self) -> None:
        feature_engine = features.UnifiedFeatureEngine()
        self.state['features_df'] = feature_engine.compute_full_feature_grid(self.state['market_df'], run_selection=True)

    def _step_4_machine_learning(self) -> None:
        ml_engine = machine_learning.UnifiedModelEngine()
        self.state['ml_output_df'] = ml_engine.predict_and_calibrate(self.state['features_df'])

    def _step_5_prediction(self) -> None:
        pred_engine = prediction.UnifiedPredictionEngine()
        ml_df = self.state['ml_output_df']
        features_df = self.state.get('features_df')
        market_df = self.state.get('market_df')

        if "current_price" not in ml_df.columns:
            price_col = next((c for c in ["close", "last_price", "price"] if c in ml_df.columns), None)
            if price_col:
                ml_df = ml_df.with_columns(pl.col(price_col).alias("current_price"))
            else:
                source_df = market_df if isinstance(market_df, pl.DataFrame) and market_df.height > 0 else features_df
                if isinstance(source_df, pl.DataFrame) and source_df.height > 0:
                    asset_col = next((c for c in ["asset", "ticker", "symbol"] if c in ml_df.columns and c in source_df.columns), None)
                    time_col = next((c for c in ["date", "timestamp"] if c in ml_df.columns and c in source_df.columns), None)
                    source_price_col = next((c for c in ["close", "last_price", "price", "current_price"] if c in source_df.columns), None)
                    
                    if asset_col and time_col and source_price_col:
                        ml_df = ml_df.join(
                            source_df.select([asset_col, time_col, pl.col(source_price_col).alias("current_price")]),
                            on=[asset_col, time_col],
                            how="left"
                        )
                
                if "current_price" not in ml_df.columns or ml_df["current_price"].is_null().any():
                    raise RuntimeError("❌ [CRITICAL_DATA_ERROR] Gagal mendapatkan 'current_price' riil dari data pipeline!")

        if "feature_atr" not in ml_df.columns and isinstance(features_df, pl.DataFrame) and "feature_atr" in features_df.columns:
            asset_col = next((c for c in ["asset", "ticker", "symbol"] if c in ml_df.columns and c in features_df.columns), None)
            time_col = next((c for c in ["date", "timestamp"] if c in ml_df.columns and c in features_df.columns), None)
            if asset_col and time_col:
                ml_df = ml_df.join(features_df.select([asset_col, time_col, "feature_atr"]), on=[asset_col, time_col], how="left")

        predictions_df = pred_engine.run_prediction_pipeline(ml_df)
        self.state['predictions_df'] = predictions_df

    def _step_6_signal_engine(self) -> None:
        signal_engine = signal_idx.UnifiedSignalEngine()
        self.state['signals_df'] = signal_engine.execute_pipeline(self.state['predictions_df'])
        self.state['raw_signals_df'] = self.state['signals_df']

    def _step_7_validation(self) -> None:
        val_engine = validation.UnifiedValidationEngine()
        market_df = self.state.get('market_df')

        if isinstance(market_df, pl.DataFrame) and market_df.height > 0:
            returns_df = (
                market_df.sort("date")
                .group_by("date")
                .agg(pl.col("close").mean().alias("avg_close"))
                .with_columns(
                    ((pl.col("avg_close") - pl.col("avg_close").shift(1)) / pl.col("avg_close").shift(1)).alias("daily_return")
                )
                .drop_nulls()
            )
            real_returns = returns_df["daily_return"].to_numpy()
        else:
            real_returns = np.array([])

        if len(real_returns) == 0:
            self.logger.warning("⚠️ Data return riil tidak mencukupi untuk validasi. Menggunakan array return nol.")
            real_returns = np.zeros(10)

        self.state['val_results'] = val_engine.validate_full_pipeline(df=self.state['raw_signals_df'], returns=real_returns)

    def _step_8_risk_management(self) -> None:
        """Step 8: Integrated Risk Management & Order Audit (Aligned with risk.py)."""
        risk_engine = risk.UnifiedRiskEngine(
            mode=self.config.EXECUTION_MODE,
            market_regime=self.state.get("market_regime", "neutral")
        )

        market_df = self.state.get("market_df")
        raw_signals = self.state.get("raw_signals_df", self.state.get("signals_df"))

        aggregated_risk = None
        if isinstance(market_df, pl.DataFrame) and market_df.height > 0:
            try:
                aggregated_risk = risk_engine.evaluate_market_risk(
                    df_klines=market_df,
                    pipeline_timestamp=self.state.get("timestamp"),
                    execution_id=self.state.get("execution_id")
                )
                self.state["aggregated_risk_output"] = aggregated_risk
            except Exception as e:
                self.logger.warning(f"⚠️ Gagal mengevaluasi market risk komposit: {e}")

        validated_orders = risk_engine.audit_portfolio_orders(
            orders=raw_signals,
            aggregated_risk=aggregated_risk
        )

        self.state['validated_orders'] = validated_orders
        self.logger.info("✔ [RISK_AUDIT_SUCCESS] Audit risiko kuantitatif & penyaringan order selesai.")

    def _step_9_portfolio_execution(self) -> None:
        """Step 9: Portfolio Allocation & Restricted Top 1 Execution (Aligned with portfolio.py)."""
        mode_suffix = _get_mode_suffix()
        state_file = f"portfolio_{mode_suffix}_state.json"
        
        portfolio_engine = UnifiedPortfolioEngine(state_file=state_file)

        signals_df = self.state.get("validated_orders", self.state.get("signals_df"))
        latest_prices = self.state.get("latest_prices_dict", {})

        processed_signals = portfolio_engine.process_trading_signals(signals_df, latest_prices)
        
        portfolio_state = {}
        if hasattr(portfolio_engine, "get_state_dict"):
            portfolio_state = portfolio_engine.get_state_dict()
        elif hasattr(portfolio_engine, "execution_engine"):
            portfolio_state = portfolio_engine.execution_engine.get_state_summary()

        total_eq = getattr(portfolio_engine, "total_equity", portfolio_state.get("total_equity", DEFAULT_DRYRUN_CAPITAL))
        avail_cash = getattr(portfolio_engine, "available_cash", portfolio_state.get("cash_balance", DEFAULT_DRYRUN_CAPITAL))
        exposure_pct = ((total_eq - avail_cash) / total_eq * 100.0) if total_eq > 0 else 0.0
        return_pct = ((total_eq - DEFAULT_DRYRUN_CAPITAL) / DEFAULT_DRYRUN_CAPITAL * 100.0) if DEFAULT_DRYRUN_CAPITAL > 0 else 0.0

        portfolio_state.update({
            "total_equity": total_eq,
            "equity": total_eq,
            "available_cash": avail_cash,
            "cash_balance": avail_cash,
            "cash": avail_cash,
            "exposure_pct": exposure_pct,
            "return_pct": return_pct,
            "overall_return_pct": return_pct,
            "last_updated": datetime.now(timezone.utc).isoformat()
        })

        self.state["signals_df"] = processed_signals if (isinstance(processed_signals, pl.DataFrame) and processed_signals.height > 0) else signals_df
        self.state["portfolio_engine"] = portfolio_engine
        self.state["portfolio_state"] = portfolio_state
        self.state["portfolio_summary"] = {
            "total_equity": total_eq,
            "available_cash": avail_cash,
            "exposure_pct": exposure_pct,
            "overall_return_pct": return_pct
        }

    def _step_10_execution_layer(self) -> None:
        signals_input = self.state.get('signals_df')
        portfolio_engine: Optional[UnifiedPortfolioEngine] = self.state.get("portfolio_engine")

        sim_engine = simulation.UnifiedSimulationEngine()
        sim_engine.run_full_execution_simulation(market_df=self.state['market_df'], signals_df=signals_input)

        portfolio_state = self.state.get("portfolio_state", {})
        
        if portfolio_engine:
            if hasattr(portfolio_engine, "get_state_dict"):
                engine_state = portfolio_engine.get_state_dict()
                if engine_state:
                    portfolio_state.update(engine_state)
            elif hasattr(portfolio_engine, "execution_engine"):
                engine_state = portfolio_engine.execution_engine.get_state_summary()
                if engine_state:
                    portfolio_state.update(engine_state)

            total_eq = getattr(portfolio_engine, "total_equity", portfolio_state.get("total_equity", DEFAULT_DRYRUN_CAPITAL))
            avail_cash = getattr(portfolio_engine, "available_cash", portfolio_state.get("cash_balance", DEFAULT_DRYRUN_CAPITAL))
            
            positions_dict = getattr(portfolio_engine, "positions", portfolio_state.get("positions", {}))
            active_pos_list = getattr(portfolio_engine, "active_positions", portfolio_state.get("active_positions", []))
            
            if isinstance(positions_dict, dict) and not active_pos_list:
                active_pos_list = list(positions_dict.values())
            elif isinstance(active_pos_list, list) and not positions_dict:
                positions_dict = {
                    (p.get("ticker") or p.get("symbol") or f"POS_{i}"): p 
                    for i, p in enumerate(active_pos_list) if isinstance(p, dict)
                }

            active_pos_count = len(positions_dict) if isinstance(positions_dict, dict) else len(active_pos_list)

            top_pick = portfolio_state.get("top_pick", "-")
            if (not top_pick or top_pick == "-") and positions_dict:
                first_key = list(positions_dict.keys())[0]
                top_pick = normalize_idx_symbol(first_key)
            
            if (not top_pick or top_pick == "-") and isinstance(signals_input, pl.DataFrame) and signals_input.height > 0:
                try:
                    ticker_col = next((c for c in ["ticker", "asset", "asset_id", "symbol"] if c in signals_input.columns), None)
                    rank_col = next((c for c in ["ranking_score", "score", "rank", "probability", "confidence"] if c in signals_input.columns), None)
                    if ticker_col:
                        sorted_df = signals_input.sort(rank_col, descending=True) if rank_col else signals_input
                        top_pick = normalize_idx_symbol(str(sorted_df[ticker_col][0]))
                except Exception as e:
                    self.logger.warning(f"⚠️ Gagal mengekstrak Top Pick dari signals_input: {e}")
                    top_pick = "-"

            exposure_pct = ((total_eq - avail_cash) / total_eq * 100.0) if total_eq > 0 else 0.0
            return_pct = ((total_eq - DEFAULT_DRYRUN_CAPITAL) / DEFAULT_DRYRUN_CAPITAL * 100.0) if DEFAULT_DRYRUN_CAPITAL > 0 else 0.0

            portfolio_state.update({
                "equity": total_eq,
                "cash": avail_cash,
                "cash_balance": avail_cash,
                "total_equity": total_eq,
                "exposure_pct": exposure_pct,
                "return_pct": return_pct,
                "overall_return_pct": return_pct,
                "active_positions_count": active_pos_count,
                "positions_count": active_pos_count,
                "positions": positions_dict,
                "active_positions": active_pos_list,
                "top_pick": top_pick,
                "last_updated": datetime.now(timezone.utc).isoformat()
            })
            
            self.state['portfolio_state'] = portfolio_state

    def _step_11_evaluation(self) -> None:
        """Step 11: Quantitative Performance Evaluation Engine."""
        try:
            try:
                eval_facade = evaluation.UnifiedEvaluationEngine(config={"mode": self.config.EXECUTION_MODE})
            except TypeError:
                try:
                    eval_facade = evaluation.UnifiedEvaluationEngine(mode=self.config.EXECUTION_MODE)
                except TypeError:
                    eval_facade = evaluation.UnifiedEvaluationEngine()

            eval_metrics = {}
            executed = False
            target_methods = [
                "execute_pipeline_evaluation",
                "run_pipeline_evaluation",
                "evaluate_pipeline",
                "evaluate",
                "execute_evaluation",
                "run"
            ]

            for method_name in target_methods:
                if hasattr(eval_facade, method_name):
                    method = getattr(eval_facade, method_name)
                    try:
                        eval_metrics = method(
                            market_df=self.state.get("market_df"),
                            signals_df=self.state.get("signals_df")
                        )
                    except TypeError:
                        try:
                            eval_metrics = method(self.state.get("signals_df"))
                        except TypeError:
                            eval_metrics = method()
                    executed = True
                    break

            if not executed:
                self.logger.warning("⚠️ [EVALUATION_WARNING] Tidak ditemukan metode evaluasi yang valid pada UnifiedEvaluationEngine.")

            self.state["evaluation_metrics"] = eval_metrics
            self.logger.info("✔ [EVALUATION_SUCCESS] Audit metrik performa kuantitatif berhasil dikompilasi.")

        except Exception as e:
            self.logger.error(f"⚠️ [EVALUATION_BYPASS] Step 11 dilewati karena error non-kritis: {e}", exc_info=True)

    def _step_12_self_learning(self) -> None:
        """Step 12: Continuous Model Optimization Engine."""
        try:
            try:
                sl_facade = self_learning.UnifiedSelfLearningEngine(config={"mode": self.config.EXECUTION_MODE})
            except TypeError:
                try:
                    sl_facade = self_learning.UnifiedSelfLearningEngine(mode=self.config.EXECUTION_MODE)
                except TypeError:
                    sl_facade = self_learning.UnifiedSelfLearningEngine()

            predictions_df = self.state.get("predictions_df")
            if predictions_df is None:
                predictions_df = self.state.get("signals_df")
            
            actuals_df = self.state.get("market_df")
            features_df = self.state.get("features_df")
            
            if actuals_df is not None and isinstance(actuals_df, pl.DataFrame):
                if "realized_return" not in actuals_df.columns:
                    asset_col = next((c for c in ["asset", "ticker", "symbol"] if c in actuals_df.columns), None)
                    time_col = next((c for c in ["date", "timestamp"] if c in actuals_df.columns), None)
                    if asset_col and time_col and "close" in actuals_df.columns:
                        actuals_df = actuals_df.sort([asset_col, time_col]).with_columns(
                            (pl.col("close") / pl.col("close").shift(1).over(asset_col) - 1.0)
                            .fill_null(0.0)
                            .alias("realized_return")
                        )
            
            sl_results = {}
            executed = False
            
            if hasattr(sl_facade, "run_full_feedback_cycle"):
                feature_cols = []
                if isinstance(features_df, pl.DataFrame):
                    feature_cols = [c for c in features_df.columns if c not in ["date", "timestamp", "asset", "ticker", "symbol"]]

                try:
                    sl_results = sl_facade.run_full_feedback_cycle(
                        predictions_df=predictions_df,
                        actuals_df=actuals_df,
                        baseline_features_df=features_df,
                        current_features_df=features_df,
                        feature_cols=feature_cols,
                        processed_prediction_ids=set(),
                        active_model_id="champion_idx_v1",
                        model_instance=None,
                        auto_persist_model=True
                    )
                except TypeError:
                    sl_results = sl_facade.run_full_feedback_cycle(
                        predictions_df=predictions_df,
                        actuals_df=actuals_df
                    )
                executed = True
            else:
                target_methods = [
                    "execute_self_learning_cycle", "run_self_learning_cycle", 
                    "execute_cycle", "run_cycle", "run_self_learning", "run", "optimize"
                ]
                for method_name in target_methods:
                    if hasattr(sl_facade, method_name):
                        method = getattr(sl_facade, method_name)
                        try:
                            sl_results = method(
                                market_df=actuals_df,
                                signals_df=predictions_df
                            )
                        except TypeError:
                            try:
                                sl_results = method(actuals_df)
                            except TypeError:
                                sl_results = method()
                        executed = True
                        break

            if not executed:
                self.logger.warning("⚠️ [SELF_LEARNING_WARNING] Tidak ditemukan metode eksekusi yang valid pada UnifiedSelfLearningEngine.")
            else:
                self.state["self_learning_results"] = sl_results
                self.logger.info("✔ [SELF_LEARNING_SUCCESS] Siklus self-learning & update parameter model selesai.")

        except Exception as e:
            self.logger.error(f"⚠️ [SELF_LEARNING_BYPASS] Step 12 dilewati karena error non-kritis: {e}", exc_info=True)

    def _step_13_autonomous_engine(self) -> None:
        """Step 13: Autonomous Feedback Loop & Risk Control Gate."""
        try:
            auto_facade = autonomous_engine_idx.UnifiedAutonomousEngine(
                log_path="prediksi_idx_log.csv",
                db_path="autonomous_engine.db"
            )

            candles_dict = {}
            market_df = self.state.get("market_df")
            if isinstance(market_df, pl.DataFrame) and market_df.height > 0:
                asset_col = next((c for c in ["asset", "ticker", "symbol"] if c in market_df.columns), None)
                if asset_col:
                    for symbol, group in market_df.group_by(asset_col):
                        sym_str = str(symbol[0]) if isinstance(symbol, tuple) else str(symbol)
                        candles_dict[sym_str] = group

            current_drawdown = self.state.get("portfolio_state", {}).get("current_drawdown_pct", 0.0)

            auto_results = auto_facade.execute_evaluation(
                candles_dict=candles_dict,
                current_drawdown_pct=current_drawdown,
                evaluation_days=5
            )

            self.state["autonomous_results"] = auto_results

            if "risk_config" in auto_results:
                self.state["adaptive_risk_config"] = auto_results["risk_config"]

        except Exception as e:
            self.logger.error(f"⚠️ [AUTONOMOUS_ENGINE_BYPASS] Step 13 dilewati karena error non-kritis: {e}", exc_info=True)

    def _step_14_research(self) -> None:
        """Step 14: Autonomous Research Engine & Meta-Learning."""
        try:
            research_config = research.ResearchConfig(
                cycle_interval_seconds=3600,
                max_consecutive_failures=3,
                dry_run=False,
                auto_deploy=True,
                random_seed=42
            )

            research_engine = research.UnifiedResearchEngine(
                config=research_config,
                db_path="./checkpoints/research_history.sqlite"
            )

            decision = research_engine.run_manual_cycle()
            self.state["research_decision"] = {
                "should_run": decision.should_run,
                "reason": decision.reason,
                "priority": decision.priority
            }
            self.logger.info(f"✔ [RESEARCH_CYCLE_COMPLETE] Status: {decision.should_run} | Priority: {decision.priority}")
        except Exception as e:
            self.logger.error(f"⚠️ [RESEARCH_ENGINE_BYPASS] Step 14 dilewati karena error non-kritis: {e}", exc_info=True)

    def _step_15_monitoring(self) -> None:
        """Step 15: System Telemetry & Hardware Health Diagnostics Audit."""
        try:
            monitoring_engine = monitoring.UnifiedMonitoringEngine({
                "health_check": {
                    "memory_threshold_pct": 85.0,
                    "disk_threshold_pct": 90.0,
                    "db_path": "./autonomous_engine.db"
                }
            })

            features_df = self.state.get("features_df")
            target_cols = ["close", "volume"]

            audit_results = monitoring_engine.execute_full_audit(
                df=features_df,
                target_columns=target_cols,
                timestamp_col="date" if (isinstance(features_df, pl.DataFrame) and "date" in features_df.columns) else None,
                model_id="UnifiedModelEngine_v2026",
                task_type="regression"
            )

            self.state["monitoring_audit"] = audit_results
            global_status = audit_results.get("global_status", "UNKNOWN")
            self.logger.info(f"✔ [TELEMETRY_AUDIT_PASSED] Global System Health Status: {global_status}")
        except Exception as e:
            self.logger.error(f"⚠️ [MONITORING_BYPASS] Step 15 dilewati karena error non-kritis: {e}", exc_info=True)

    def _step_16_reporting(self) -> None:
        """Step 16: Centralized Reporting & Telegram Notification Broadcast (Aligned with reporting.py)."""
        try:
            reporting_engine = reporting.UnifiedReportingEngine(
                config={"INITIAL_CAPITAL_IDR": DEFAULT_DRYRUN_CAPITAL},
                mode=self.config.EXECUTION_MODE
            )
            
            signals_input = self.state.get('signals_df')
            signals_dicts = []
            if isinstance(signals_input, pl.DataFrame) and signals_input.height > 0:
                signals_dicts = signals_input.to_dicts()

            success = reporting_engine.send_telegram_broadcast(
                orders=signals_dicts,
                portfolio_data=self.state.get('portfolio_state')
            )
            if success:
                self.logger.info("✔ [REPORTING_BROADCAST_SUCCESS] Telegram notification broadcast berhasil dikirim.")
            else:
                self.logger.warning("⚠️ [REPORTING_WARNING] Telegram broadcast mengembalikan status false (periksa kembali token/chat ID).")
        except Exception as e:
            self.logger.error(f"⚠️ [REPORTING_BYPASS] Step 16 dilewati karena error non-kritis: {e}", exc_info=True)

    def _step_17_storage(self) -> None:
        """Step 17: Persistent Storage & Memory Cleanup."""
        try:
            storage_engine = storage.UnifiedStorageEngine()
            if hasattr(storage_engine, "persist_signals"):
                storage_engine.persist_signals(self.state.get('validated_orders'))
        except Exception as e:
            self.logger.error(f"⚠️ Gagal menyimpan sinyal ke storage persistence: {e}", exc_info=True)
        finally:
            keys_to_clean = ["market_df", "features_df", "ml_output_df", "predictions_df", "raw_signals_df"]
            for key in keys_to_clean:
                if key in self.state:
                    del self.state[key]
            
            gc.collect()
            self.logger.info("🧹 Pembersihan memori DataFrame selesai dijalankan (Garbage Collection).")


# =============================================================================
# 8. DIRECT ENTRY POINT (FOR GITHUB ACTIONS / CRON RUNNER)
# =============================================================================
if __name__ == "__main__":
    # Handle decoupled bootstrap entry point for CI/CD & initial setups
    if "--bootstrap-universe" in sys.argv:
        sys.stdout.write("⚙️ [BOOTSTRAP] Memulai pembuatan/generasi universe.json secara mandiri...\n")
        try:
            # 1. Inisialisasi Data Engine untuk penarikan data & filter semesta
            data_engine = data.UnifiedDataEngine(
                universe_file="universe.json",
                cache_dir=".cache",
                enable_cache=True
            )
            
            universe_tickers = []
            if hasattr(data_engine, "get_universe"):
                universe_tickers = data_engine.get_universe(force_refresh=True)
            elif hasattr(data_engine, "load_and_prepare_market_data"):
                market_df = data_engine.load_and_prepare_market_data(symbols=DEFAULT_IDX_TICKERS, use_cache=True)
                if market_df is not None and "asset" in market_df.columns:
                    universe_tickers = market_df["asset"].unique().to_list()

            # Failsafe: Jika data engine tidak mengembalikan ticker, gunakan daftar default BEI
            if not universe_tickers:
                universe_tickers = DEFAULT_IDX_TICKERS

            # 2. EXPORT EKSPLISIT ke berkas universe.json jika belum ada di disk
            if not os.path.exists("universe.json"):
                universe_payload = {
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "total_tickers": len(universe_tickers),
                    "symbols": [normalize_idx_symbol(t) for t in universe_tickers]
                }
                with open("universe.json", "w", encoding="utf-8") as f:
                    json.dump(universe_payload, f, indent=2)
                sys.stdout.write(f"✔ [BOOTSTRAP_WRITE] Berhasil menulis {len(universe_tickers)} ticker ke universe.json\n")

            if os.path.exists("universe.json"):
                sys.stdout.write("✔ [BOOTSTRAP_SUCCESS] universe.json berhasil diverifikasi di disk.\n")
                sys.exit(0)
            else:
                sys.stderr.write("❌ [BOOTSTRAP_ERROR] universe.json gagal dibuat ke disk!\n")
                sys.exit(1)

        except Exception as bootstrap_err:
            sys.stderr.write(f"🛑 [BOOTSTRAP_CRITICAL_ERROR] {bootstrap_err}\n")
            traceback.print_exc(file=sys.stderr)
            sys.exit(1)

    orchestrator = ProductionOrchestrator()
    orchestrator.run()
