#!/usr/bin/env sh


set -euo pipefail

JOBS_FILE="${1:-jobs.sh}"   # Pfad als erstes Argument, default: jobs.sh

TOTAL=$(wc -l < "${JOBS_FILE}")
echo "$TOTAL"
DONE=0

while IFS= read -r CMD; do
  DONE=$(( DONE + 1 ))
  echo "[${DONE}/${TOTAL}] ${CMD}"
  eval "$CMD"
done < "${JOBS_FILE}"
