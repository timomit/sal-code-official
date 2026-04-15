#!/usr/bin/env python3
"""Sweep launcher for convenient reproduction of data for SymmNet paper figure

Runs dataset × algo × seed combinations.

Sequential run (default):
    python sweep.py

Parallel run with 4 workers:
    python sweep.py --n-workers 4

Subset example:
    python sweep.py --datasets cifar10 --algos bp sal --n-seeds 2

Re-running is safe: finished runs (metrics.json present) are skipped.
"""

import argparse
from pathlib import Path

ALL_DATASETS = ["cifar10", "fmnist", "svhn"]
ALL_ALGOS = ["bp", "fa", "bp_w_fa", "akrout", "scfa", "sal", "rdd"]
# PARAM_FILE = "exp_settings.yaml"
PARAM_FILE = "fast_exp.yaml"
PY_FILE = "main_salnet.py"
JOBS_FILE = "jobs.sh"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        default=ALL_DATASETS,
        choices=ALL_DATASETS,
        metavar="DS",
        help="Datasets to run (default: all three).",
    )
    p.add_argument(
        "--algos",
        nargs="+",
        default=ALL_ALGOS,
        choices=ALL_ALGOS,
        metavar="ALGO",
        help="Algorithm sections from exp_settings.yaml (default: all seven).",
    )
    p.add_argument(
        "--n-seeds",
        type=int,
        default=5,
        dest="n_seeds",
        help="Number of seeds, numbered 0 … N-1 (default: 5).",
    )
    p.add_argument(
        "--sweep-name",
        default="sweep",
        dest="sweep_name",
        help="Subdirectory name under --base-dir (default: sweep).",
    )
    p.add_argument(
        "--base-dir",
        default="../../results/symm_net",
        dest="base_dir",
        help="Root output directory (default: ../../results/symm_net).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    sweep_dir = Path(args.base_dir) / args.sweep_name
    total = len(args.datasets) * len(args.algos) * args.n_seeds
    done = 0

    print(f"Sweep: {total} runs → {sweep_dir}")
    print(f"  datasets: {args.datasets}")
    print(f"  algos: {args.algos}")
    print(f"  seeds: 0 … {args.n_seeds - 1}")
    print()

    if Path(JOBS_FILE).exists():
        print(f"{JOBS_FILE} already exists. Stop here!")
        return

    proc_calls = []
    for dataset in args.datasets:
        for algo in args.algos:
            for seed in range(args.n_seeds):
                run_dir = sweep_dir / dataset / algo / f"seed_{seed}"
                done += 1

                if (run_dir / "metrics.json").exists():
                    print(f"[{done}/{total}] Skip  {dataset}/{algo}/seed_{seed}")
                    continue

                print(f"[{done}/{total}] Run   {dataset}/{algo}/seed_{seed}")
                proc_calls.append(
                    (
                        f"python {PY_FILE} -f {PARAM_FILE} -s {algo}"
                        f" --dataset {dataset} --run-dir {run_dir}\n"
                    )
                )

    with open(JOBS_FILE, "w") as f:
        f.writelines(proc_calls)


if __name__ == "__main__":
    main()
