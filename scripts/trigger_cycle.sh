#!/usr/bin/env bash
# External cron helper – call this from Cron-job.org or similar
# Requires GH_PAT with repo scope

set -euo pipefail

REPO="whatman42/idx"
EVENT_TYPE="run_trading_cycle"

if [[ -z "${GH_PAT:-}" ]]; then
  echo "GH_PAT environment variable required"
  exit 1
fi

curl -sS -X POST \
  -H "Accept: application/vnd.github+json" \
  -H "Authorization: Bearer ${GH_PAT}" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  "https://api.github.com/repos/${REPO}/dispatches" \
  -d "{\"event_type\":\"${EVENT_TYPE}\"}"

echo "Dispatched ${EVENT_TYPE} to ${REPO}"
