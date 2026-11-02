import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

METHODS = {
    "gsdb": "{python} gsdb.py --dataset {dataset}",
    "rev2": "{python} rev2.py --dataset {dataset}",
    "gsrank": "{python} gsrank.py --dataset {dataset}",
}

DATASETS = [
    "yelp",
    "Books",
    "Beauty_and_Personal_Care",
    "Pet_Supplies",
    "Baby_Products",
    "Toys_and_Games",
]


def run(method: str, dataset: str) -> bool:
    cmd = METHODS[method].format(python=sys.executable, dataset=dataset)
    print(f"\n=== [{method}] {dataset} ===\n$ {cmd}", flush=True)
    start = time.time()
    result = subprocess.run(cmd, shell=True, cwd=ROOT)
    elapsed = time.time() - start
    ok = result.returncode == 0
    status = "OK" if ok else f"ERRORE (exit {result.returncode})"
    print(f"--- [{method}] {dataset}: {status} in {elapsed:.1f}s", flush=True)
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", choices=list(METHODS), default=list(METHODS))
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args()

    results = {}
    for dataset in args.datasets:
        for method in args.methods:
            ok = run(method, dataset)
            results[(method, dataset)] = ok
            if not ok and args.stop_on_error:
                break
        else:
            continue
        break

    print("\n===== RESUME =====")
    for (method, dataset), ok in results.items():
        print(f"{'OK    ' if ok else 'FALLITO'}  {method:7s} {dataset}")
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
