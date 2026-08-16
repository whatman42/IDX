# ==============================================================================
# GitHub Actions Workflow: IDX Stock Analysis Engine & Scalping Pipeline
# File: .github/workflows/idx.yml
# Version: v2026.Q3.v27.1
# Scope: Signal + Multi-Horizon + Rule-Based Fallback + Simulated Portfolio
# Execution: SIMULATION ONLY
# Live Broker: DISABLED
#
# Security:
# - All third-party GitHub Actions pinned to full commit SHA.
# - No broker credentials used by this workflow.
# - Secrets supplied only through GitHub Actions secrets.
# ==============================================================================

name: IDX Stock Analysis Engine

on:
  # ============================================================================
  # AUTOMATIC SCHEDULE
  # ============================================================================
  schedule:

    # --------------------------------------------------------------------------
    # HEALTH CHECK
    # 08:00 WIB = 01:00 UTC
    # --------------------------------------------------------------------------
    - cron: '0 1 * * 1-5'

    # --------------------------------------------------------------------------
    # SESSION I
    # 09:00–12:00 WIB
    # 02:00–05:00 UTC
    #
    # Every 15 minutes
    # --------------------------------------------------------------------------
    - cron: '*/15 2-4 * * 1-5'

    # --------------------------------------------------------------------------
    # SESSION II
    # 13:30–15:50 WIB
    # 06:30–08:50 UTC
    #
    # Every 15 minutes
    # --------------------------------------------------------------------------
    - cron: '30/15 6-8 * * 1-5'

    # --------------------------------------------------------------------------
    # POST-MARKET / SELF LEARNING
    # 16:15 WIB = 09:15 UTC
    # --------------------------------------------------------------------------
    - cron: '15 9 * * 1-5'

  # ============================================================================
  # MANUAL EXECUTION
  # ============================================================================
  workflow_dispatch:
    inputs:

      run_mode:
        description: "Pilih mode eksekusi pipeline"
        required: true
        default: "dry_run"
        type: choice
        options:
          - dry_run
          - self_learning
          - health_check
          - reset

# ==============================================================================
# GLOBAL CONCURRENCY LOCK
#
# Mencegah dua workflow memodifikasi portfolio/state/database secara bersamaan.
# ==============================================================================
concurrency:
  group: idx-production-global-lock
  cancel-in-progress: false

# ==============================================================================
# MINIMUM REQUIRED PERMISSIONS
# ==============================================================================
permissions:
  contents: write

# ==============================================================================
# GLOBAL ENVIRONMENT
# ==============================================================================
env:
  TZ: "Asia/Jakarta"
  PYTHONUNBUFFERED: "1"
  CI: "true"

  # Maximum allowed market-data staleness.
  IDX_MAX_STALENESS_SEC: "86400.0"

  # Explicitly simulation-only.
  IDX_SIMULATION_ONLY: "true"
  LIVE_BROKER_ENABLED: "false"

# ==============================================================================
# JOB
# ==============================================================================
jobs:

  run-idx-pipeline:

    name: IDX Pipeline

    runs-on: ubuntu-latest

    timeout-minutes: 15

    steps:

      # =========================================================================
      # 1. CHECKOUT
      #
      # actions/checkout@v4.2.2
      # Full SHA pinned for repository policy.
      # =========================================================================
      - name: Checkout Repository
        uses: actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683
        with:
          fetch-depth: 0
          token: ${{ secrets.GITHUB_TOKEN }}

      # =========================================================================
      # 2. PYTHON
      #
      # actions/setup-python@v5.6.0
      # =========================================================================
      - name: Set up Python 3.11
        uses: actions/setup-python@a26af69be951a213d495a4c3e4e4022e16d87065
        with:
          python-version: "3.11"
          cache: "pip"

      # =========================================================================
      # 3. RESTORE MARKET-DATA CACHE
      #
      # actions/cache@v4.2.4
      # =========================================================================
      - name: Restore Persistent Market Data Cache
        uses: actions/cache@0400d5f644dc74513175e3cd8d07132dd4860809
        with:
          path: .cache/
          key: idx-market-cache-${{ runner.os }}-py3.11-${{ github.run_id }}
          restore-keys: |
            idx-market-cache-${{ runner.os }}-py3.11-
            idx-market-cache-${{ runner.os }}-

      # =========================================================================
      # 4. INSTALL DEPENDENCIES
      # =========================================================================
      - name: Install Dependencies & Tools
        shell: bash
        run: |
          set -Eeuo pipefail

          python -m pip install --upgrade pip

          if [ -f requirements.txt ]; then
            pip install -r requirements.txt
          else
            echo "No requirements.txt found."
          fi

          if ! command -v sqlite3 >/dev/null 2>&1; then
            echo "Installing sqlite3..."
            sudo apt-get update -y
            sudo apt-get install -y sqlite3
          else
            echo "sqlite3 already available."
          fi

      # =========================================================================
      # 5. SECURITY / ACTION PINNING AUDIT
      #
      # Fail if workflow source contains unpinned action references.
      # =========================================================================
      - name: Verify GitHub Actions SHA Pinning
        shell: bash
        run: |
          set -Eeuo pipefail

          echo "Checking GitHub Actions references..."

          if grep -RInE 'uses:[[:space:]]+[^@]+@(v[0-9]+|main|master|latest)' \
            .github/workflows; then

            echo "ERROR: Unpinned GitHub Action detected."
            exit 1
          fi

          echo "All GitHub Actions references are pinned."

      # =========================================================================
      # 6. PYTHON COMPILE GATE
      # =========================================================================
      - name: Python Syntax & Compilation Gate
        shell: bash
        run: |
          set -Eeuo pipefail

          echo "Compiling Python repository..."

          python -m compileall -q .

          echo "Python compilation PASS."

      # =========================================================================
      # 7. BASIC REPOSITORY INTEGRITY
      # =========================================================================
      - name: Repository Integrity Gate
        shell: bash
        run: |
          set -Eeuo pipefail

          echo "Checking repository integrity..."

          test -f main.py

          if grep -RInE \
            'PLACEHOLDER_TOO_LARGE|PLACEHOLDER_READ_FROM_DISK|READ_FROM_DISK' \
            --include='*.py' .; then

            echo "ERROR: Placeholder/truncated source detected."
            exit 1
          fi

          echo "Repository integrity PASS."

      # =========================================================================
      # 8. PREPARE DIRECTORIES
      # =========================================================================
      - name: Prepare Runtime Directories
        shell: bash
        run: |
          set -Eeuo pipefail

          mkdir -p \
            .cache \
            .cache/corporate_actions \
            reports \
            logs \
            data \
            storage \
            models

          touch \
            .cache/.gitkeep \
            reports/.gitkeep \
            logs/.gitkeep \
            data/.gitkeep \
            storage/.gitkeep \
            models/.gitkeep

          if [ ! -f prediksi_idx_log.csv ]; then
            echo "timestamp,ticker,prediction_probability,prediction_confidence,signal_status" \
              > prediksi_idx_log.csv
          fi

      # =========================================================================
      # 9. UNIVERSE BOOTSTRAP
      # =========================================================================
      - name: Bootstrap Universe
        env:
          IDX_PROXY_URL: ${{ secrets.IDX_PROXY_URL }}
        shell: bash
        run: |
          set -Eeuo pipefail

          if [ ! -f universe.json ]; then
            echo "universe.json not found."

            if python main.py --help | grep -q -- "--bootstrap-universe"; then
              echo "Bootstrap command available."
              python main.py --bootstrap-universe
            else
              echo "Bootstrap CLI unavailable."
              echo "Pipeline will continue only if universe can be loaded internally."
            fi
          fi

          if [ -f universe.json ]; then
            echo "Universe available."
          else
            echo "WARNING: universe.json unavailable after bootstrap."
          fi

      # =========================================================================
      # 10. DETERMINE EXECUTION MODE
      # =========================================================================
      - name: Determine Execution Mode
        id: mode
        shell: bash
        run: |
          set -Eeuo pipefail

          MODE="${{ github.event.inputs.run_mode }}"

          if [ "${{ github.event_name }}" = "schedule" ]; then

            TRIGGER_CRON="${{ github.event.schedule }}"

            echo "Scheduled cron: ${TRIGGER_CRON}"

            if [ "$TRIGGER_CRON" = "0 1 * * 1-5" ]; then
              MODE="health_check"

            elif [ "$TRIGGER_CRON" = "15 9 * * 1-5" ]; then
              MODE="self_learning"

            else
              MODE="dry_run"
            fi
          fi

          if [ -z "$MODE" ]; then
            MODE="dry_run"
          fi

          case "$MODE" in
            dry_run|self_learning|health_check|reset)
              ;;
            *)
              echo "ERROR: Invalid mode: $MODE"
              exit 1
              ;;
          esac

          echo "Selected mode: $MODE"

          echo "mode=$MODE" >> "$GITHUB_OUTPUT"

      # =========================================================================
      # 11. RUN IDX PIPELINE
      # =========================================================================
      - name: Run IDX Pipeline
        env:
          GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
          TELEGRAM_BOT_TOKEN: ${{ secrets.TELEGRAM_BOT_TOKEN }}
          TELEGRAM_CHAT_ID: ${{ secrets.TELEGRAM_CHAT_ID }}
          IDX_PROXY_URL: ${{ secrets.IDX_PROXY_URL }}
          IDX_FETCH_TIMEOUT_SEC: "15.0"

          # Explicit safety controls.
          IDX_SIMULATION_ONLY: "true"
          LIVE_BROKER_ENABLED: "false"

        shell: bash
        run: |
          set -Eeuo pipefail

          MODE="${{ steps.mode.outputs.mode }}"

          echo "=============================================="
          echo "IDX PIPELINE"
          echo "MODE: $MODE"
          echo "SIMULATION ONLY: $IDX_SIMULATION_ONLY"
          echo "LIVE BROKER: $LIVE_BROKER_ENABLED"
          echo "=============================================="

          if [ "$MODE" = "reset" ]; then

            echo "[RESET]"
            python main.py --reset-dryrun

          elif [ "$MODE" = "self_learning" ]; then

            echo "[SELF LEARNING]"
            python main.py --self-learning

          elif [ "$MODE" = "health_check" ]; then

            echo "[HEALTH CHECK]"

            if python main.py --help | grep -q -- "--health-check"; then
              python main.py --health-check
            elif [ -f monitoring.py ]; then
              python monitoring.py
            else
              echo "Health-check entry point unavailable."
              exit 1
            fi

          else

            echo "[DRY RUN / SIGNAL ENGINE]"

            python main.py --dry-run

          fi

      # =========================================================================
      # 12. POST-RUN SIGNAL / PORTFOLIO SAFETY AUDIT
      #
      # This does not fabricate a signal. It only verifies source/runtime
      # safety properties.
      # =========================================================================
      - name: Simulation Safety Gate
        shell: bash
        run: |
          set -Eeuo pipefail

          echo "Checking simulation-only controls..."

          python - <<'PY'
          import os

          simulation = os.environ.get("IDX_SIMULATION_ONLY", "").lower()
          broker = os.environ.get("LIVE_BROKER_ENABLED", "").lower()

          if simulation != "true":
              raise SystemExit(
                  "FAIL: IDX_SIMULATION_ONLY is not true."
              )

          if broker != "false":
              raise SystemExit(
                  "FAIL: LIVE_BROKER_ENABLED is not false."
              )

          print("Simulation safety PASS.")
          print("LIVE_BROKER_ENABLED =", broker)
          PY

      # =========================================================================
      # 13. SQLITE INTEGRITY
      # =========================================================================
      - name: SQLite Checkpoint & Integrity Audit
        shell: bash
        run: |
          set -Eeuo pipefail

          echo "Scanning SQLite databases..."

          DB_FOUND=0

          while IFS= read -r -d '' db_file; do

            DB_FOUND=1

            echo "Processing: $db_file"

            sqlite3 "$db_file" \
              "PRAGMA wal_checkpoint(TRUNCATE);" || true

            CHECK_RESULT="$(sqlite3 "$db_file" \
              "PRAGMA integrity_check;")"

            if [ "$CHECK_RESULT" != "ok" ]; then
              echo "ERROR: SQLite integrity failure."
              echo "Database: $db_file"
              echo "Result: $CHECK_RESULT"
              exit 1
            fi

            echo "SQLite PASS: $db_file"

          done < <(
            find . \
              -type f \
              \( -name "*.db" -o -name "*.sqlite" \) \
              -print0
          )

          if [ "$DB_FOUND" -eq 0 ]; then
            echo "No SQLite database generated."
          fi

          find . \
            \( -name "*.db-wal" -o -name "*.db-shm" \) \
            -size 0 \
            -delete || true

      # =========================================================================
      # 14. STATE / OUTPUT SUMMARY
      # =========================================================================
      - name: Runtime State Summary
        if: always()
        shell: bash
        run: |
          set -Eeuo pipefail

          echo "=============================================="
          echo "IDX RUNTIME OUTPUT SUMMARY"
          echo "=============================================="

          echo
          echo "Portfolio state:"
          find . -maxdepth 3 \
            -type f \
            \( -name "portfolio_*.json" -o -name "portfolio_state.json" \) \
            -print || true

          echo
          echo "Reports:"
          find reports \
            -maxdepth 2 \
            -type f \
            -print 2>/dev/null || true

          echo
          echo "Logs:"
          find logs \
            -maxdepth 2 \
            -type f \
            -print 2>/dev/null || true

          echo
          echo "Models:"
          find models \
            -maxdepth 2 \
            -type f \
            -print 2>/dev/null || true

      # =========================================================================
      # 15. COMMIT / PUSH STATE
      #
      # Only runtime state/output is whitelisted.
      # Production source files are deliberately excluded.
      # =========================================================================
      - name: Commit Runtime State
        if: success()
        shell: bash
        run: |
          set -Eeuo pipefail

          git config user.name "github-actions[bot]"
          git config user.email \
            "41898282+github-actions[bot]@users.noreply.github.com"

          git fetch origin main

          # --------------------------------------------------------------------
          # Only runtime state is allowed.
          # --------------------------------------------------------------------
          git add -f universe.json 2>/dev/null || true

          git add \
            portfolio_*.json \
            checkpoint*.json \
            prediksi_idx_log.csv \
            2>/dev/null || true

          git add \
            reports/*.md \
            reports/*.csv \
            storage/*.json \
            models/*.joblib \
            2>/dev/null || true

          # SQLite runtime state.
          while IFS= read -r -d '' db_file; do
            git add -f "$db_file"
          done < <(
            find . \
              -type f \
              \( -name "*.db" -o -name "*.sqlite" \) \
              -print0
          )

          if git diff --cached --quiet; then
            echo "No runtime state changes."
            exit 0
          fi

          git commit \
            -m "chrono(bot): update simulation state and analysis outputs [skip ci]"

          # --------------------------------------------------------------------
          # Re-sync before push.
          # --------------------------------------------------------------------
          MAX_PUSH_ATTEMPTS=3
          PUSH_ATTEMPT=1

          while true; do

            if git push origin HEAD:main; then
              echo "Runtime state pushed successfully."
              break
            fi

            if [ "$PUSH_ATTEMPT" -ge "$MAX_PUSH_ATTEMPTS" ]; then
              echo "ERROR: Push failed after $MAX_PUSH_ATTEMPTS attempts."
              exit 1
            fi

            echo "Push rejected. Re-syncing..."

            git fetch origin main

            git rebase origin/main || {
              echo "Rebase failed."
              git rebase --abort || true
              exit 1
            }

            PUSH_ATTEMPT=$((PUSH_ATTEMPT + 1))

            sleep 3

          done

      # =========================================================================
      # 16. FAILURE TELEMETRY
      #
      # actions/upload-artifact@v4.6.2
      # Full SHA pinned.
      # =========================================================================
      - name: Upload Telemetry Logs on Failure
        if: failure()
        uses: actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02
        with:
          name: idx-pipeline-failure-${{ github.run_id }}
          retention-days: 14
          if-no-files-found: ignore
          path: |
            logs/**/*.log
            logs/*.log
            reports/**/*.md
            reports/*.md
            reports/**/*.csv
            prediksi_idx_log.csv
            portfolio_*.json
            portfolio_state.json
