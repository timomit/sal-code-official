#!/usr/bin/env bash
# slurm_submit.sh

# snapshot a copy of jobs.sh
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SNAPSHOT="jobs_${TIMESTAMP}.sh"

cp jobs.sh "${SNAPSHOT}" 
echo "Copy a snapshot of jobs.sh to ${SNAPSHOT}"
TOTAL=$(wc -l < "${SNAPSHOT}")

sbatch --array=1-${TOTAL} --export="ALL,JOBS_FILE=${SNAPSHOT}" slurm.sh
echo "Submitted ${TOTAL} jobs."
