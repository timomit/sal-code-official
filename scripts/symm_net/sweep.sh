#!/usr/bin/env bash
# sweep.sh — run all dataset × algo × seed combinations for the paper.
#
# Usage (sequential local run):
#   bash sweep.sh
#
# For SLURM: replace the `python` call in the marked block below with an
# sbatch invocation. Minimal example:
#
#   sbatch \
#     --partition=gpu --gres=gpu:1 --time=4:00:00 --mem=16G \
#     --job-name="${ALGO}_${DATASET}_s${SEED}" \
#     --wrap="cd $(pwd) && conda activate myenv && python main_salnet.py \
#             -f ${PARAM_FILE} -s ${ALGO} --dataset ${DATASET} \
#             --seed ${SEED} --run-dir ${RUN_DIR}"
#
# Re-running is safe: finished runs (metrics.json present) are skipped.

set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────────
DATASETS=("cifar10" "fmnist" "svhn")
ALGOS=("bp" "fa" "bp_w_fa" "akrout" "scfa" "sal" "rdd")
N_SEEDS=5
SWEEP_NAME="sweep"
BASE_DIR="../../results/symm_net"
PARAM_FILE="exp_settings.yaml"
# ─────────────────────────────────────────────────────────────────────────────

DATASETS=("fmnist")
ALGOS=("bp")
N_SEEDS=2
SWEEP_NAME="sh_sweep_1"
BASE_DIR="/tmp/symmnet_test"
PARAM_FILE="fast_exp.yaml"

SWEEP_DIR="${BASE_DIR}/${SWEEP_NAME}"
TOTAL=$(( ${#DATASETS[@]} * ${#ALGOS[@]} * N_SEEDS ))
DONE=0

echo "Sweep: ${TOTAL} runs → ${SWEEP_DIR}"
echo "  datasets: ${DATASETS[*]}"
echo "  algos: ${ALGOS[*]}"
echo "  seeds: 0 … $(( N_SEEDS - 1 ))"
echo ""

for DATASET in "${DATASETS[@]}"; do
  for ALGO in "${ALGOS[@]}"; do
    for (( SEED=0; SEED<N_SEEDS; SEED++ )); do
      RUN_DIR="${SWEEP_DIR}/${DATASET}/${ALGO}/seed_${SEED}"
      DONE=$(( DONE + 1 ))

      if [[ -f "${RUN_DIR}/metrics.json" ]]; then
        echo "[${DONE}/${TOTAL}] Skip  ${DATASET}/${ALGO}/seed_${SEED}"
        continue
      fi

      echo "[${DONE}/${TOTAL}] Run   ${DATASET}/${ALGO}/seed_${SEED} ..."
      mkdir -p "${RUN_DIR}"

      # ── replace this block with sbatch for SLURM ─────────────────────────
      python main_salnet.py \
        -f "${PARAM_FILE}" \
        -s "${ALGO}" \
        --dataset "${DATASET}" \
        --seed "${SEED}" \
        --run-dir "${RUN_DIR}"
      # ─────────────────────────────────────────────────────────────────────

    done
  done
done

echo ""
echo "Sweep complete. Results in ${SWEEP_DIR}"
