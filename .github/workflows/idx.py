# ==============================================================================
# GitHub Actions Workflow: IDX Stock Analysis Engine
# File: .github/workflows/idx_analysis_pipeline.yml
# Version: v2026.Q3.v28.0
# Scope: Analysis + RULE_BASED/ML signal + simulated portfolio
# LIVE BROKER: DISABLED
# ==============================================================================

name: IDX Stock Analysis Engine

on:
  # --------------------------------------------------------------------------
  # AUTOMATIC SCHEDULE
  # --------------------------------------------------------------------------
  schedule:
    # HEALTH CHECK — 08:00 WIB / 01:00 UTC
    - cron: "0 1 * * 1-5"

    # SESSION I — 09:00–12:00 WIB
    - cron: "*/15 2-4 * * 1-5"

    # SESSION II — 13:30–15:45 WIB
    - cron: "30,45 6 * * 1-5"
    - cron: "*/15 7 * * 1-5"
    - cron: "*/15 8 * * 1-5"

    # POST MARKET — 16:15 WIB / 09:15 UTC
    - cron: "15 9 * * 1-5"

  # --------------------------------------------------------------------------
  # MANUAL RUN
  # --------------------------------------------------------------------------
  workflow_dispatch:
    inputs:
      run_mode:
        description: "Pilih mode eksekusi"
        required: true
        default: "dry_run"
        type: choice
        options:
          - dry_run
          - health_check
          - self_learning
          - release_audit
          - reset

      confirm_simulation:
        description: "Konfirmasi bahwa eksekusi tetap simulation-only"
        required: true
        default: true
        type: boolean

# Prevent concurrent mutation of portfolio/state/database.
concurrency:
  group: idx-production-global-lock
  cancel-in-progress: false

permissions:
  contents: write

env:
  TZ: "Asia/Jakarta"
  PYTHONUNBUFFERED: "1"
  CI: "true"
  IDX_MAX_STALENESS_SEC: "86400.0"

jobs:

  # ============================================================================
  # MAIN PIPELINE
  # ============================================================================
  run-idx-pipeline:

    runs-on: ubuntu-latest
    timeout-minutes: 15

    steps:

      # ------------------------------------------------------------------------
      # 1. CHECKOUT
      # ------------------------------------------------------------------------
      - name: Checkout Repository
        uses: actions/checkout@v4
        with:
          fetch-depth: 0
          token: ${{ secrets.GITHUB_TOKEN }}

      # ------------------------------------------------------------------------
      # 2. PYTHON
      # ------------------------------------------------------------------------
      - name: Set up Python 3.11
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: "pip"

      # ------------------------------------------------------------------------
      # 3. CACHE
      # ------------------------------------------------------------------------
      - name: Restore Market Data Cache
        uses: actions/cache@v4
        with:
          path: .cache/
          key: idx-cache-${{ runner.os }}-py311-${{ github.run_id }}
          restore-keys: |
            idx-cache-${{ runner.os }}-py311-
            idx-cache-${{ runner.os }}-

      # ------------------------------------------------------------------------
      # 4. DEPENDENCIES
      # ------------------------------------------------------------------------
      - name: Install Dependencies
        run: |
          set -Eeuo pipefail

          python -m pip install --upgrade pip

          if [ -f requirements.txt ]; then
            python -m pip install -r requirements.txt
          fi

          if ! command -v sqlite3 >/dev/null 2>&1; then
            sudo apt-get update -y
            sudo apt-get install -y sqlite3
          fi

      # ------------------------------------------------------------------------
      # 5. REPOSITORY INTEGRITY
      # ------------------------------------------------------------------------
      - name: Repository Integrity Gate
        run: |
          set -Eeuo pipefail

          echo "=== Repository Integrity ==="

          if [ ! -d ".github/workflows" ]; then
            echo "❌ .github/workflows directory missing."
            exit 1
          fi

          if find . -type f \( \
            -name "*PLACEHOLDER*" \
            -o -name "*READ_FROM_DISK*" \
            \) | grep -q .; then
            echo "❌ Placeholder artifact detected."
            find . -type f \( \
              -name "*PLACEHOLDER*" \
              -o -name "*READ_FROM_DISK*" \
            \)
            exit 1
          fi

          echo "✔ Repository integrity gate passed."

      # ------------------------------------------------------------------------
      # 6. COMPILE
      # ------------------------------------------------------------------------
      - name: Python Syntax & Compilation Gate
        run: |
          set -Eeuo pipefail

          python -m compileall -q .

          echo "✔ Python compilation passed."

      # ------------------------------------------------------------------------
      # 7. PREPARE STATE
      # ------------------------------------------------------------------------
      - name: Prepare Runtime Directories
        run: |
          set -Eeuo pipefail

          mkdir -p \
            .cache/corporate_actions \
            reports \
            logs \
            data \
            storage \
            models

          touch \
            reports/.gitkeep \
            logs/.gitkeep \
            data/.gitkeep \
            storage/.gitkeep \
            models/.gitkeep

          if [ ! -f "prediksi_idx_log.csv" ]; then
            echo "timestamp,ticker,prediction_probability,prediction_confidence,signal_status" \
              > prediksi_idx_log.csv
          fi

      # ------------------------------------------------------------------------
      # 8. UNIVERSE
      # ------------------------------------------------------------------------
      - name: Bootstrap Universe
        env:
          IDX_PROXY_URL: ${{ secrets.IDX_PROXY_URL }}
        run: |
          set -Eeuo pipefail

          if [ ! -f "universe.json" ]; then
            echo "Universe missing. Running bootstrap..."
            python main.py --bootstrap-universe
          fi

          if [ ! -f "universe.json" ]; then
            echo "❌ universe.json was not created."
            exit 1
          fi

          echo "✔ Universe available."

      # ------------------------------------------------------------------------
      # 9. DETERMINE EXECUTION MODE
      # ------------------------------------------------------------------------
      - name: Determine Execution Mode
        id: mode
        shell: bash
        run: |
          set -Eeuo pipefail

          MODE=""

          # Manual execution
          if [ "${{ github.event_name }}" = "workflow_dispatch" ]; then
            MODE="${{ github.event.inputs.run_mode }}"

            if [ "${{ github.event.inputs.confirm_simulation }}" != "true" ]; then
              echo "❌ Simulation-only confirmation was not provided."
              exit 1
            fi

          # Scheduled execution
          elif [ "${{ github.event_name }}" = "schedule" ]; then
            TRIGGER="${{ github.event.schedule }}"

            echo "Scheduled trigger: $TRIGGER"

            case "$TRIGGER" in

              "0 1 * * 1-5")
                MODE="health_check"
                ;;

              "15 9 * * 1-5")
                MODE="self_learning"
                ;;

              *)
                MODE="dry_run"
                ;;

            esac

          else
            MODE="dry_run"
          fi

          case "$MODE" in
            dry_run|health_check|self_learning|release_audit|reset)
              ;;
            *)
              echo "❌ Invalid execution mode: $MODE"
              exit 1
              ;;
          esac

          echo "MODE=$MODE"
          echo "mode=$MODE" >> "$GITHUB_OUTPUT"

      # ------------------------------------------------------------------------
      # 10. EXECUTE
      # ------------------------------------------------------------------------
      - name: Execute IDX Pipeline
        env:
          GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
          TELEGRAM_BOT_TOKEN: ${{ secrets.TELEGRAM_BOT_TOKEN }}
          TELEGRAM_CHAT_ID: ${{ secrets.TELEGRAM_CHAT_ID }}
          IDX_PROXY_URL: ${{ secrets.IDX_PROXY_URL }}
          IDX_FETCH_TIMEOUT_SEC: "15.0"
        shell: bash
        run: |
          set -Eeuo pipefail

          MODE="${{ steps.mode.outputs.mode }}"

          echo "=========================================="
          echo "IDX PIPELINE"
          echo "MODE: $MODE"
          echo "LIVE BROKER: DISABLED"
          echo "=========================================="

          case "$MODE" in

            dry_run)
              echo "Running simulation-only signal pipeline..."
              python main.py --dry-run
              ;;

            health_check)
              echo "Running health check..."
              python monitoring.py
              ;;

            self_learning)
              echo "Running self-learning..."
              python main.py --self-learning
              ;;

            release_audit)
              echo "Running final release audit..."

              python -m compileall -q .

              if grep -R \
                -n \
                -E \
                "PLACEHOLDER_TOO_LARGE|PLACEHOLDER_READ_FROM_DISK" \
                --include="*.py" \
                .; then
                echo "❌ Placeholder detected."
                exit 1
              fi

              echo "✔ Release audit basic gates passed."
              ;;

            reset)
              echo "Resetting simulation state..."
              python main.py --reset-dryrun
              ;;

          esac

      # ------------------------------------------------------------------------
      # 11. SQLITE INTEGRITY
      # ------------------------------------------------------------------------
      - name: SQLite Integrity Check
        run: |
          set -Eeuo pipefail

          shopt -s nullglob

          DB_FILES=(
            $(find . -type f \( -name "*.db" -o -name "*.sqlite" \))
          )

          if [ "${#DB_FILES[@]}" -eq 0 ]; then
            echo "No SQLite database found."
            exit 0
          fi

          for db_file in "${DB_FILES[@]}"; do

            echo "Checking: $db_file"

            sqlite3 "$db_file" \
              "PRAGMA wal_checkpoint(TRUNCATE);" \
              || true

            RESULT="$(sqlite3 "$db_file" "PRAGMA integrity_check;")"

            if [ "$RESULT" != "ok" ]; then
              echo "❌ SQLite integrity failure: $db_file"
              echo "$RESULT"
              exit 1
            fi

            echo "✔ SQLite OK: $db_file"

          done

      # ------------------------------------------------------------------------
      # 12. STATE COMMIT
      # ------------------------------------------------------------------------
      - name: Commit Simulation State
        if: |
          success() &&
          (
            github.event_name == 'schedule' ||
            github.event_name == 'workflow_dispatch'
          ) &&
          (
            steps.mode.outputs.mode == 'dry_run' ||
            steps.mode.outputs.mode == 'self_learning' ||
            steps.mode.outputs.mode == 'reset'
          )
        shell: bash
        run: |
          set -Eeuo pipefail

          git config user.name "github-actions[bot]"
          git config user.email "41898282+github-actions[bot]@users.noreply.github.com"

          git fetch origin main

          # Stage ONLY runtime state.
          git add -f universe.json 2>/dev/null || true
          git add portfolio_*.json 2>/dev/null || true
          git add checkpoint*.json 2>/dev/null || true
          git add reports/*.md 2>/dev/null || true
          git add reports/*.csv 2>/dev/null || true
          git add storage/*.json 2>/dev/null || true
          git add prediksi_idx_log.csv 2>/dev/null || true

          # Do NOT automatically commit arbitrary source/model/database files.
          # This prevents runtime jobs from accidentally modifying production code.

          if git diff --cached --quiet; then
            echo "✔ No runtime state changes."
            exit 0
          fi

          git commit \
            -m "chrono(bot): update simulation state [skip ci]"

          git push origin HEAD:main

          echo "✔ Simulation state pushed to main."

      # ------------------------------------------------------------------------
      # 13. FAILURE TELEMETRY
      # ------------------------------------------------------------------------
      - name: Upload Failure Telemetry
        if: failure()
        uses: actions/upload-artifact@v4
        with:
          name: idx-pipeline-failure-${{ github.run_id }}
          retention-days: 14
          path: |
            logs/
            reports/
            prediksi_idx_log.csv