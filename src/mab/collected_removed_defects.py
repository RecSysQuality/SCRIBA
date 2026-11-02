#!/usr/bin/env python3
"""
Copia i removed_defects.tsv dagli esperimenti MAB nella cartella di ablation di recbole_f.

Struttura attesa:
  <RESULTS_ROOT>/<DATASET>/<exp>_max300000_rt-slog_stop_pat1000000/<tipo>/removed_defects.tsv

Scope:
  ..._src-rev2_...  -> rev2   (idem group, gsdb, ...)
  nessun "src-"     -> all

Output:
  <OUT_DIR>/<DATASET>/<DATASET>.removed_all__<tipo>__<scope>_50K_ablation.tsv
"""
import argparse
import re
import shutil
from pathlib import Path

RESULTS_ROOT = Path("./src/mab/results")
OUT_DIR = Path("./src/recbole_f/removed_defects_ablation_new_version")
SUFFIX = "max300000_rt-slog_stop_pat11500000"
DEFAULT_DATASETS = ["Books", "yelp", "Beauty_and_Personal_Care", "Health_and_Household"]
SRC_RE = re.compile(r"(?:^|_)src-([^_]+)")


def get_scope(exp_name: str) -> str:
    m = SRC_RE.search(exp_name)
    return m.group(1) if m else "all"


def process_dataset(dataset, results_root, out_root, dry_run, overwrite):
    base = results_root / dataset
    out_dir = out_root / dataset
    print(f"\n========== {dataset} ==========")

    if not base.is_dir():
        print(f"Cartella non trovata: {base} -> salto")
        return 0, 0, 0

    exp_dirs = sorted(d for d in base.iterdir() if d.is_dir() and d.name.endswith(SUFFIX))
    if not exp_dirs:
        print(f"Nessuna cartella che finisce per {SUFFIX} in {base} -> salto")
        return 0, 0, 0

    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    seen = {}  # nome destinazione -> sorgente, per segnalare collisioni
    copied = skipped = missing = 0

    for exp in exp_dirs:
        scope = get_scope(exp.name)
        print(f"\n[{exp.name}]  scope = {scope}")
        for run in sorted(d for d in exp.iterdir() if d.is_dir()):
            src = run / "removed_defects.tsv"
            if not src.is_file():
                print(f"  - {run.name}: removed_defects.tsv mancante")
                missing += 1
                continue

            dst_name = f"{dataset}.removed_{run.name}__{scope}_50K_ablation.tsv"

            #dst_name = f"{dataset}.removed_mab_all__{scope}_50K_ablation.tsv"
            dst = out_dir / dst_name

            if dst_name in seen:
                print(f"  ! COLLISIONE: {dst_name}\n      già da {seen[dst_name]}\n      ora da {src} -> salto")
                skipped += 1
                continue
            seen[dst_name] = src

            if dst.exists() and not overwrite:
                print(f"  = {dst_name} esiste già (usa --overwrite)")
                skipped += 1
                continue

            print(f"  + {run.name} -> {dataset}/{dst_name}")
            if not dry_run:
                shutil.copy2(src, dst)
            copied += 1

    print(f"\n{dataset}: {copied} copiati, {skipped} saltati, {missing} senza removed_defects.tsv -> {out_dir}")
    return copied, skipped, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS,
                    help=f"dataset da processare (default: {' '.join(DEFAULT_DATASETS)})")
    ap.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--dry-run", action="store_true", help="mostra cosa farebbe senza copiare")
    ap.add_argument("--overwrite", action="store_true", help="sovrascrive file già esistenti")
    args = ap.parse_args()

    tot = [0, 0, 0]
    for ds in args.datasets:
        res = process_dataset(ds, args.results_root, args.out_dir, args.dry_run, args.overwrite)
        tot = [a + b for a, b in zip(tot, res)]

    tag = " (dry-run)" if args.dry_run else ""
    print(f"\nTOTALE{tag}: {tot[0]} copiati, {tot[1]} saltati, {tot[2]} senza removed_defects.tsv")


if __name__ == "__main__":
    main()