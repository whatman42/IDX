# ==============================================================================
# GitHub Actions Workflow: IDX Stock Analysis Engine & Scalping Pipeline
# File: .github/workflows/idx_analysis_pipeline.yml
# Version: v2026.Q3.v27.0 (Unified Auto-Routing Schedule: Scalping, ML, Health)
# Compliance: Indonesia Stock Exchange (IDX) Quantitative Pipeline (v2026)
# ==============================================================================
name: IDX Stock Analysis Engine

on:
  # 1. Cron Job: Jadwal Terpadu Sesuai Jam Kerja Bursa Efek Indonesia (IDX)
  schedule:
    # 🩺 [HEALTH CHECK] Pre-market: 08:00 WIB (01:00 UTC) - Memastikan API/Network siap
    - cron: '0 1 * * 1-5'
    
    # ⚡ [SCALPING] Sesi I BEI: 09:00 - 12:00 WIB (02:00 - 05:00 UTC) - Setiap 15 Menit
    - cron: '*/15 2-4 * * 1-5'
    
    # ⚡ [SCALPING] Sesi II BEI: 13:30 - 15:50 WIB (06:30 - 08:50 UTC) - Setiap 15 Menit
    - cron: '30/15 6-8 * * 1-5'
    
    # 🧠 [SELF LEARNING] Post-market: 16:15 WIB (09:15 UTC) - Retraining Data Hari Ini
    - cron: '15 9 * * 1-5'

  # 2. Eksekusi Manual (Workflow Dispatch)
  workflow_dispatch:
    inputs:
      run_mode:
        description: "Pilih Mode Eksekusi Pipeline"
        required: true
        default: 'dry_run'
        type: choice
        options:
          - dry_run
          - self_learning
          - health_check
          - reset

# Concurrency Group Tunggal Global untuk Mencegah Race Condition DB & Git State
concurrency:
  group: idx-production-global-lock
  cancel-in-progress: false

permissions:
  contents: write

env:
  TZ: "Asia/Jakarta"
  PYTHONUNBUFFERED: "1"
  CI: "true"
  IDX_MAX_STALENESS_SEC: "86400.0" # 24 Jam Toleransi Staleness untuk Intraday Check

jobs:
  run-idx-pipeline:
    runs-on: ubuntu-latest
    timeout-minutes: 15 # Menurunkan timeout ke 15 menit agar responsif pada skenario Intraday

    steps:
      # ------------------------------------------------------------------------
      # 1. Checkout Repository
      # ------------------------------------------------------------------------
      - name: Checkout Repository
        uses: actions/checkout@v4
        with:
          fetch-depth: 0
          token: ${{ secrets.GITHUB_TOKEN }}

      # ------------------------------------------------------------------------
      # 2. Setup Python Environment
      # ------------------------------------------------------------------------
      - name: Set up Python 3.11
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: "pip"

      # ------------------------------------------------------------------------
      # 3. Persistent Cache Local Market Data & Parquet Cache
      # ------------------------------------------------------------------------
      - name: Restore Persistent Scalping Market Data Cache
        uses: actions/cache@v4
        with:
          path: .cache/
          key: idx-scalping-cache-${{ runner.os }}-py3.11-${{ github.run_id }}
          restore-keys: |
            idx-scalping-cache-${{ runner.os }}-py3.11-
            idx-scalping-cache-${{ runner.os }}-

      # ------------------------------------------------------------------------
      # 4. Install Dependencies & System Tools
      # ------------------------------------------------------------------------
      - name: Install Dependencies & Tools
        run: |
          set -Eeuo pipefail
          python -m pip install --upgrade pip
          if [ -f requirements.txt ]; then
            pip install -r requirements.txt
          fi
          # Install sqlite3 CLI tool jika tidak ada di runner OS
          if ! command -v sqlite3 >/dev/null 2>&1; then
            echo "📦 Installing sqlite3 CLI utility..."
            sudo apt-get update -y && sudo apt-get install -y sqlite3
          else
            echo "✔ sqlite3 CLI already available on runner."
          fi

      # ------------------------------------------------------------------------
      # 5. Pre-flight Syntax & Compilation Gate (P1 PROTECTION)
      # ------------------------------------------------------------------------
      - name: Python Syntax & Compilation Gate
        run: |
          set -Eeuo pipefail
          echo "🔍 Auditing Python syntax across repository..."
          python -m compileall -q .
          echo "✔ All Python scripts compiled successfully with zero syntax errors."

      # ------------------------------------------------------------------------
      # 6. Prepare Output Directories & Placeholders
      # ------------------------------------------------------------------------
      - name: Prepare Directories & Placeholders
        run: |
          set -Eeuo pipefail
          mkdir -p .cache/corporate_actions reports logs data storage models
          touch reports/.gitkeep logs/.gitkeep data/.gitkeep storage/.gitkeep models/.gitkeep .cache/.gitkeep

          if [ ! -f "prediksi_idx_log.csv" ]; then
            echo "timestamp,ticker,prediction_probability,prediction_confidence,signal_status" > prediksi_idx_log.csv
            echo "✅ File 'prediksi_idx_log.csv' placeholder berhasil diinisialisasi."
          fi

      # ------------------------------------------------------------------------
      # 7. Decoupled Idempotent Universe Bootstrap
      # ------------------------------------------------------------------------
      - name: Bootstrap Universe State & Validate Dependencies
        env:
          IDX_PROXY_URL: ${{ secrets.IDX_PROXY_URL }}
        run: |
          set -Eeuo pipefail

          if [ ! -f "universe.json" ]; then
            echo "⚠️ universe.json tidak ditemukan. Memicu bootstrap via entry point publik aplikasi..."
            python main.py --bootstrap-universe
          fi

          if [ ! -f "universe.json" ]; then
            echo "❌ CRITICAL ERROR: universe.json gagal dibuat oleh aplikasi!"
            exit 1
          fi

          echo "✔ Dependency Check Passed: universe.json valid & tersedia."

      # ------------------------------------------------------------------------
      # 8. Run IDX Scalping Analysis Pipeline (AUTO-ROUTING ENGINE)
      # ------------------------------------------------------------------------
      - name: Run IDX Stock Scalping Engine
        env:
          GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
          TELEGRAM_BOT_TOKEN: ${{ secrets.TELEGRAM_BOT_TOKEN }}
          TELEGRAM_CHAT_ID: ${{ secrets.TELEGRAM_CHAT_ID }}
          IDX_PROXY_URL: ${{ secrets.IDX_PROXY_URL }}
          IDX_FETCH_TIMEOUT_SEC: "15.0"
        run: |
          set -Eeuo pipefail

          MODE="${{ github.event.inputs.run_mode }}"

          # ---------------------------------------------------
          # 🔄 DYNAMIC CRON ROUTING LOGIC
          # ---------------------------------------------------
          if [ "${{ github.event_name }}" = "schedule" ]; then
            TRIGGER_CRON="${{ github.event.schedule }}"
            echo "🕒 Pipeline dipicu oleh Auto-Schedule (Cron): $TRIGGER_CRON"
            
            if [ "$TRIGGER_CRON" = "0 1 * * 1-5" ]; then
              MODE="health_check"
            elif [ "$TRIGGER_CRON" = "15 9 * * 1-5" ]; then
              MODE="self_learning"
            else
              MODE="dry_run"
            fi
          fi

          # Fallback default jika eksekusi tidak membawa argumen
          if [ -z "$MODE" ]; then
            MODE="dry_run"
          fi

          echo "🚀 Executing Pipeline in Mode: $MODE"

          if [ "$MODE" = "reset" ]; then
            echo "🔄 [MODE: RESET] Mengosongkan Simulasi Modal & Portofolio..."
            python main.py --reset-dryrun
          elif [ "$MODE" = "self_learning" ]; then
            echo "🧠 [MODE: SELF LEARNING] Melatih ulang Model ML dengan data hari ini..."
            python main.py --self-learning
          elif [ "$MODE" = "health_check" ]; then
            echo "🩺 [MODE: HEALTH CHECK] Memeriksa konektivitas Egress & SQLite Status..."
            python monitoring.py
          else
            echo "⚡ [MODE: SCALPING] Menghasilkan Sinyal Dry-Run Intraday..."
            python main.py --dry-run
          fi

      # ------------------------------------------------------------------------
      # 9. SQLite Checkpoint & Integrity Audit
      # ------------------------------------------------------------------------
      - name: Checkpoint & Validate SQLite Databases
        run: |
          set -Eeuo pipefail
          echo "🔍 Checking SQLite Database Integrity & Flushing WAL..."
          
          find . \( -name "*.db" -o -name "*.sqlite" \) | while read -r db_file; do
            echo "Processing DB: $db_file"
            sqlite3 "$db_file" "PRAGMA wal_checkpoint(TRUNCATE);" || true
            
            CHECK_RESULT=$(sqlite3 "$db_file" "PRAGMA integrity_check;")
            if [ "$CHECK_RESULT" != "ok" ]; then
              echo "❌ CRITICAL: Database $db_file CORRUPTED! Integrity Check: $CHECK_RESULT"
              exit 1
            fi
            echo "✔ Database $db_file OK."
          done

          find . \( -name "*.db-wal" -o -name "*.db-shm" \) -size 0 -delete || true

      # ------------------------------------------------------------------------
      # 10. Atomic Git Commit & Self-Healing Push
      # ------------------------------------------------------------------------
      - name: Commit & Push Updated Files
        shell: bash
        run: |
          set -Eeuo pipefail

          git config user.name "github-actions[bot]"
          git config user.email "41898282+github-actions[bot]@users.noreply.github.com"

          git fetch origin main
          git rebase --abort 2>/dev/null || true
          git pull --rebase origin main || {
            echo "⚠️ Rebase conflict. Stashing untracked/local changes..."
            git rebase --abort || true
            git stash -u
            git pull --rebase origin main
            git stash pop || true
          }

          # Whitelist Staging - Hanya Stage State/DB/Model/Report Wajib
          git add -f universe.json 2>/dev/null || true
          git add portfolio_*.json checkpoint*.json 2>/dev/null || true
          git add reports/*.md reports/*.csv 2>/dev/null || true
          git add models/*.joblib storage/*.json 2>/dev/null || true
          git add prediksi_idx_log.csv 2>/dev/null || true

          # Stage SQLite DB Files
          find . \( -name "*.db" -o -name "*.sqlite" \) -print0 | xargs -0 -r git add -f

          if git diff --cached --quiet; then
            echo "✔ Tidak ada perubahan state/database untuk di-commit."
            exit 0
          fi

          git commit -m "chrono(bot): auto-update scalping signals, models, and portfolio state [skip ci]"

          MAX_PUSH_ATTEMPTS=3
          PUSH_ATTEMPT=1
          
          until git push origin HEAD; do
            if [ $PUSH_ATTEMPT -ge $MAX_PUSH_ATTEMPTS ]; then
              echo "❌ Failed to push commits after $MAX_PUSH_ATTEMPTS attempts."
              exit 1
            fi
              echo "⚠️ Push rejected. Re-syncing with remote..."
            sleep 3
            git rebase --abort 2>/dev/null || true
            git pull --rebase origin main
            PUSH_ATTEMPT=$((PUSH_ATTEMPT + 1))
          done

          echo "🎉 Commit & Push Berhasil Tersimpan di Remote Repository!"

      # ------------------------------------------------------------------------
      # 11. Upload Telemetry Logs on Failure
      # ------------------------------------------------------------------------
      - name: Upload Telemetry Logs on Failure
        if: failure()
        uses: actions/upload-artifact@v4
        with:
          name: idx-scalping-execution-failure-logs
          retention-days: 14
          path: |
            logs/*.log
            reports/*.md
            reports/*.csv
