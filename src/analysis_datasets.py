"""
Tabella LaTeX con le statistiche dei dataset.

Legge, per ogni dataset, gli stessi file usati da run_my_eval.py:
  <data>/<ds>/<ds>.rec_train.csv
  <data>/<ds>/<ds>.rec_valid.csv   (oppure .rec_vali.csv)
  <data>/<ds>/<ds>.rec_test.csv
  <data>/<ds>/<ds>.rec_test_active.csv   (oppure _rec_test_active.csv)
  <data>/<ds>/<ds>.rec_test_lt.csv       (oppure _rec_test_lt.csv)

Colonne della tabella:
  #Train, #Valid, #Test  -> numero di interazioni per split
  #Users, #Items         -> utenti / item distinti su train+valid+test
  Sparsity               -> 1 - |interazioni| / (#Users * #Items)
  #Active, #Long-tail    -> utenti distinti nei file dei segmenti (con % su #Users)

Tabella 2: difetti per dataset e per sorgente (Group, GSDB, REV2), più una riga
All con tutte le sorgenti insieme, letti da
  <data>/<ds>/<ds>.<source>.csv   (TSV, una riga = un difetto, colonne users, items)
con source in --sources (default: group_f gsdb_ff rev2_f). Nella riga All utenti,
item e archi in comune tra sorgenti sono contati una volta. Colonne:
  #Defects  -> numero di difetti (righe con almeno un utente e un item)
  #Users    -> utenti distinti coinvolti in almeno un difetto   (% su #Users del dataset)
  #Items    -> item distinti coinvolti in almeno un difetto     (% su #Items del dataset)
  #Edges    -> archi distinti del training tra utenti e item dello stesso difetto
               (stessa regola di make_removed_defects.defect_edges: archi condivisi
               da più difetti contati una volta)                 (% su #Train)
Richiede i pacchetti LaTeX booktabs (cmidrule) e graphicx (resizebox).

Uso:
  python dataset_stats_table.py
  python dataset_stats_table.py --data-dir ../data/original --datasets Books yelp
  python dataset_stats_table.py --out tables/dataset_stats.tex --csv tables/dataset_stats.csv
  python dataset_stats_table.py --sources group_f gsdb_ff rev2_f
"""
import argparse
import csv
import re
import sys
from pathlib import Path

import pandas as pd

csv.field_size_limit(sys.maxsize)                       # righe con liste di utenti molto lunghe

ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = (ROOT_DIR / "./data/original/").resolve()

DATASETS = [
    "Toys_and_Games", "Pet_Supplies",
    "Baby_Products", "Beauty_and_Personal_Care", "Books",  "yelp","Health_and_Household"
]

FILES = {
    "train":  ["{d}.rec_train.csv"],
    "valid":  ["{d}.rec_valid.csv", "{d}.rec_vali.csv"],
    "test":   ["{d}.rec_test.csv"],
    "active": ["{d}.rec_test_active.csv", "{d}_rec_test_active.csv"],
    "lt":     ["{d}.rec_test_lt.csv", "{d}_rec_test_lt.csv"],
}


def find(ds_dir: Path, d: str, key: str):
    for name in FILES[key]:
        p = ds_dir / name.format(d=d)
        if p.exists():
            return p
    return None


def read_inter(path: Path) -> pd.DataFrame:
    """Stessa logica di _parse_inter: separatore da header, colonne 'user_id:token' ok."""
    with path.open(encoding="utf-8") as f:
        head = f.readline().rstrip("\n")
    if "\t" in head:
        sep = "\t"
    elif re.search(r"  +", head):
        sep = r"\s+"
    else:
        sep = ","
    df = pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False,
                     engine="python" if sep == r"\s+" else "c")
    df.columns = [c.strip().split(":")[0].lower() for c in df.columns]
    missing = {"user_id", "item_id"} - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: colonne mancanti {missing}")
    return df[["user_id", "item_id"]]


def stats_for(data_dir: Path, d: str):
    """Statistiche del dataset e DataFrame del training (serve per gli archi dei difetti)."""
    ds_dir = data_dir / d
    split = {}
    for s in ("train", "valid", "test"):
        p = find(ds_dir, d, s)
        if p is None:
            raise FileNotFoundError(f"[{d}] manca lo split '{s}' in {ds_dir}")
        split[s] = read_inter(p)

    allx = pd.concat(split.values(), ignore_index=True)
    n_users = allx["user_id"].nunique()
    n_items = allx["item_id"].nunique()
    n_inter = len(allx)

    seg = {}
    for g in ("active", "lt"):
        p = find(ds_dir, d, g)
        seg[g] = read_inter(p)["user_id"].nunique() if p else None
        if p is None:
            print(f"[{d}] attenzione: file segmento '{g}' non trovato")

    r = {
        "dataset": d,
        "train": len(split["train"]), "valid": len(split["valid"]),
        "test": len(split["test"]),
        "users": n_users, "items": n_items, "interactions": n_inter,
        "sparsity": 1.0 - n_inter / (n_users * n_items),
        "active": seg["active"], "lt": seg["lt"],
    }
    return r, split["train"]


def _tok(s: str) -> set:
    return {t.strip() for t in (s or "").split(",") if t.strip()}


SOURCE_NAMES = {"group_f": "Group","group_ff": "Group", "gsdb_f": "GSDB", "gsdb_ff": "GSDB",
                "rev2": "REV2", "rev2_f": "REV2","rev2_ff": "REV2"}


def source_name(s: str) -> str:
    return SOURCE_NAMES.get(s, s.replace("_", r"\_"))


def defect_stats(data_dir: Path, d: str, sources: list, train: pd.DataFrame) -> list:
    """
    Una riga per sorgente più una riga 'all' (tutte le sorgenti insieme): numero di difetti,
    utenti e item distinti, archi distinti del training tra utenti e item dello stesso difetto.
    Nella riga 'all' utenti, item e archi in comune tra sorgenti sono contati una volta.
    Sorgente mancante: valori None.
    """
    u2i = train.groupby("user_id")["item_id"].agg(set).to_dict()
    tot = {"defects": 0, "users": set(), "items": set(), "edges": set()}
    out = []
    for s in sources:
        p = data_dir / d / f"{d}.{s}.csv"
        if not p.exists():
            print(f"[{d}] attenzione: {p.name} non trovato")
            out.append({"dataset": d, "source": s, "defects": None, "d_users": None,
                        "d_items": None, "d_edges": None})
            continue
        n_s = n_empty = 0
        users, items, edges = set(), set(), set()
        with p.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for col in ("users", "items"):
                if col not in (reader.fieldnames or []):
                    raise ValueError(f"{p.name}: colonna '{col}' mancante")
            for row in reader:
                U, I = _tok(row["users"]), _tok(row["items"])
                if not U or not I:
                    n_empty += 1
                    continue
                n_s += 1
                users |= U
                items |= I
                for u in U:
                    its = u2i.get(u)
                    if its:
                        common = (its & I) if len(its) < len(I) else (I & its)
                        edges.update((u, i) for i in common)
        if n_empty:
            print(f"[{d}] {p.name}: {n_empty} righe senza utenti o item, escluse")
        out.append({"dataset": d, "source": s, "defects": n_s, "d_users": len(users),
                    "d_items": len(items), "d_edges": len(edges)})
        tot["defects"] += n_s
        tot["users"] |= users
        tot["items"] |= items
        tot["edges"] |= edges
    out.append({"dataset": d, "source": "all", "defects": tot["defects"],
                "d_users": len(tot["users"]), "d_items": len(tot["items"]),
                "d_edges": len(tot["edges"])})
    return out


def fmt_int(x) -> str:
    return "--" if x is None else f"{x:,}".replace(",", "{,}")


def fmt_seg(x, n_users) -> str:
    if x is None:
        return "--"
    return f"{fmt_int(x)} ({100 * x / n_users:.1f}\\%)"


def ds_name(d: str) -> str:
    return "Yelp" if d == "yelp" else d.replace("_", r"\_")


def to_latex(rows: list, caption: str, label: str) -> str:
    out = [
        r"\begin{table*}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"Dataset & \#Train & \#Valid & \#Test & \#Users & \#Items & Sparsity "
        r"& \#Active users & \#Long-tail users \\",
        r"\midrule",
    ]
    for r in rows:
        name = ds_name(r["dataset"])
        out.append(" & ".join([
            name, fmt_int(r["train"]), fmt_int(r["valid"]), fmt_int(r["test"]),
            fmt_int(r["users"]), fmt_int(r["items"]),
            f"{100 * r['sparsity']:.3f}\\%",
            fmt_seg(r["active"], r["users"]), fmt_seg(r["lt"], r["users"]),
        ]) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}"]
    return "\n".join(out)


def defects_to_latex(rows: list, drows: list, caption: str, label: str) -> str:
    """rows: statistiche dei dataset (per le percentuali); drows: una riga per
    (dataset, sorgente), più la riga 'all' di ogni dataset."""
    base = {r["dataset"]: r for r in rows}
    out = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Dataset & Source & \#Defects & \#Users & \#Items & \#Edges \\",
    ]
    prev = None
    for dr in drows:
        d = dr["dataset"]
        if d != prev:                                   # nuovo dataset: riga di separazione
            out.append(r"\midrule")
        b = base[d]
        is_all = dr["source"] == "all"
        if is_all:
            out.append(r"\cmidrule(l){2-6}")
        cells = [
            fmt_int(dr["defects"]),
            fmt_seg(dr["d_users"], b["users"]),
            fmt_seg(dr["d_items"], b["items"]),
            fmt_seg(dr["d_edges"], b["train"]),
        ]
        src = r"\textit{All}" if is_all else source_name(dr["source"])
        out.append(" & ".join([ds_name(d) if d != prev else "", src] + cells) + r" \\")
        prev = d
    out += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--out", type=Path, default=ROOT_DIR / "dataset_stats.tex",
                    help="file .tex con le due tabelle (default: nella cartella da cui lanci)")
    ap.add_argument("--csv", type=Path, default=ROOT_DIR / "dataset_stats.csv",
                    help="numeri grezzi (default: nella cartella da cui lanci; '' per non salvarlo)")
    ap.add_argument("--caption", default="Statistics of the datasets. Sparsity is computed "
                    "on train, validation and test interactions; percentages of active "
                    "and long-tail users refer to the total number of users.")
    ap.add_argument("--label", default="tab:dataset_stats")
    ap.add_argument("--sources", nargs="+", default=["group_ff","rev2_fff", "gsdb_ff"],
                    help="file dei difetti: <ds>.<source>.csv")
    ap.add_argument("--defects-caption", default="Candidate defects per dataset and source: "
                    "number of defects, distinct users and items involved, and distinct "
                    "training interactions between the users and items of the same defect. "
                    "\\textit{All} counts users, items and interactions shared across sources "
                    "once. Percentages refer to the users, items and training interactions "
                    "of the dataset.")
    ap.add_argument("--defects-label", default="tab:defect_counts")
    args = ap.parse_args()

    rows, drows = [], []
    for d in args.datasets:
        if not (args.data_dir / d).is_dir():
            print(f"[{d}] cartella non trovata in {args.data_dir}, salto")
            continue
        r, train = stats_for(args.data_dir, d)
        print(f"[{d}] users={r['users']} items={r['items']} train={r['train']} "
              f"valid={r['valid']} test={r['test']} sparsity={r['sparsity']:.5f} "
              f"active={r['active']} lt={r['lt']}")
        rows.append(r)
        for dr in defect_stats(args.data_dir, d, args.sources, train):
            print(f"[{d}] {dr['source']:>8}: difetti={dr['defects']} users={dr['d_users']} "
                  f"items={dr['d_items']} edges={dr['d_edges']}")
            drows.append(dr)

    if not rows:
        raise SystemExit(f"Nessun dataset trovato in {args.data_dir}")

    tex = (to_latex(rows, args.caption, args.label) + "\n\n"
           + defects_to_latex(rows, drows, args.defects_caption, args.defects_label))
    print(f"\n{tex}\n")
    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tex + "\n", encoding="utf-8")
    print(f"Salvato: {out}")

    if str(args.csv) not in ("", "."):
        csv_out = args.csv.resolve()
        csv_out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).merge(pd.DataFrame(drows), on="dataset").to_csv(csv_out, index=False)
        print(f"Salvato: {csv_out}")


if __name__ == "__main__":
    main()