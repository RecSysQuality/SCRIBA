"""
Defect embedding [media utenti ; media item] per ogni dataset e ogni file
di gruppi (group_f, gsdb_f, ...), con il collegamento riga -> defect.

Struttura attesa:
  {data_dir}/{dataset}/{dataset}.train.csv       interazioni (user_id, item_id, rating, ...)
  {data_dir}/{dataset}/{dataset}.{source}.csv    gruppi, TSV con colonne
                                                 n_users, n_items, users, items, ...

Chiave di un defect: (dataset, source, defect_id), dove defect_id e' la
posizione della riga nel file dei gruppi (0 = prima riga dopo l'header).
Usa la stessa chiave per le label LOO.

Output per ogni (dataset, source), accanto al file dei gruppi:
  {dataset}.{source}_emb.pt          {"V": [m, 2*out_dim], "defect_id": [...]}
  {dataset}.{source}_emb_index.csv   dataset, source, defect_id, n_users, n_items
                                     (riga r del csv = riga r di V)

Output complessivo, solo per il passive set (--train-on), in --out-dir:
  std_stats.pt        media/std per standardizzare (fit SOLO sul passive set;
                      riusale identiche nell'active phase)
  passive_index.csv   una riga per defect: dataset, source, defect_id, ...
  passive_X.pt        embedding standardizzati, righe allineate a passive_index.csv

Uso:
  python compute_defect_embeddings.py --out-dim 16
  python compute_defect_embeddings.py --retrain            # riaddestra l'encoder

Richiede graphsage_scriba.py nella stessa cartella.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from graphsage import (GraphSAGE, build_graph, defect_embeddings,
                              embed_graph, fit_standardizer, load_encoder,
                              read_interactions,
                              save_encoder, set_seed, standardize, train)


ROOT_DIR      = Path(__file__).resolve().parent
BASE_DATA_DIR = (ROOT_DIR / "../data/original/").resolve()
EMBEDDINGS_DIR = (ROOT_DIR / "../node_embeddings/").resolve()

PASSIVE = ["Baby_Products", "Toys_and_Games", "Pet_Supplies"]
ACTIVE = ["Books_injseg", "yelp_injseg", "Beauty_and_Personal_Care_injseg"]
ACTIVE = ["Health_and_Household"]
LABELED = PASSIVE

"""
Per ogni dataset salva UN pickle con gli embedding dei suoi defect.

  passivi -> {dataset}.pkl = {"X", "defect_id", "source", "impact", "n_affected_all"}
  attivi  -> {dataset}.pkl = {"X", "defect_id", "source"}

  X          [m, 2*out_dim] float32, standardizzato (stessa scala per tutti)
  defect_id  [m] riga del defect nel file dei gruppi
  source     [m] "group_f" / "gsdb_f" (da quale file viene il defect)
  impact     [m] n_affected_all * (-all_loc_d_prop_rr_best)   (solo passivi)

Riga r di X = defect (source[r], defect_id[r]).

Salva anche std_stats.pkl (media/std dei passivi): serve nell'active phase
per standardizzare gli embedding ricalcolati sul grafo aggiornato.

Uso (da qualsiasi cartella):
  python compute_embeddings.py
  python compute_embeddings.py --retrain --out-dim 16

Richiede graphsage.py (la versione nuova) nella stessa cartella.
"""

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from graphsage import (GraphSAGE, build_graph, defect_embeddings, embed_graph,
                       fit_standardizer, load_encoder, read_interactions,
                       save_encoder, set_seed, standardize, train)

# .../noise_suggestor  (questo file e' in noise_suggestor/src/embeddings/)
ROOT = Path(__file__).resolve().parents[1]
#ROOT      = Path(__file__).resolve().parent


SOURCES = ["group_f", "gsdb_f", "rev2","group_inj"]
SOURCES = ["group_ff", "gsdb_ff", "rev2_fff"]

LOO_NAME = {
    "group_f": "{d}_loo_0_15000.csv",
    "group_ff": "{d}___loo_group_ff_0_20000_tndcg.csv",
    "group_inj": "{d}_loo_group_inj_0_15000.csv",
    "gsdb_f": "{d}_loo_gsdb_0_15000.csv",
    "gsdb_ff": "{d}___loo_gsdb_ff_0_20000_tndcg.csv",
    "rev2_fff": "{d}___loo_rev2_fff_0_20000_tndcg.csv",
    "rev2": "{d}_loo_rev2_0_15000.csv",
}


# ------------------------------------------------------------------ percorsi

def train_path(data_dir, d):
    return os.path.join(data_dir, d, f"{d}.train.csv")


def group_path(data_dir, d, s):
    return os.path.join(data_dir, d, f"{d}.{s}.csv")


# ------------------------------------------------------------------ lettura

def load_defects(path):
    """File dei gruppi -> (defect_id, lista di (users, items))."""
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                     usecols=["users", "items"])
    return (np.arange(len(df)),
            list(zip(df["users"].str.split(","), df["items"].str.split(","))))


def read_loo(path):
    dt = {"users": str, "items": str}
    df = pd.read_csv(path, sep="\t", dtype=dt, keep_default_na=False)
    if "defect_id" not in df.columns:
        df = pd.read_csv(path, sep=",", dtype=dt, keep_default_na=False)
    df.columns = df.columns.str.strip()
    for c in ("defect_id", "n_affected_all","n_affected_lt","n_affected_active", "all_loc_d_prop_rr_best","all_loc_d_prop_tndcg","lt_loc_d_prop_rr_best","active_loc_d_prop_rr_best", "all_loc_d_prop_entropy"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ------------------------------------------------------------------ encoder

def get_encoder(args):
    """Carica l'encoder se esiste, altrimenti lo addestra sui passivi."""
    if os.path.exists(args.encoder) and not args.retrain:
        enc = load_encoder(args.encoder)
        print(f"[encoder] caricato {args.encoder} (out_dim={enc.out_dim})")
        return enc
    print(f"[encoder] training su {args.passive}")
    graphs = [build_graph(read_interactions(train_path(args.data_dir, d)))
              for d in args.passive]
    model = GraphSAGE(hid=args.hid, out_dim=args.out_dim)
    model = train(model, graphs, epochs=args.epochs,
                  steps_per_epoch=args.steps_per_epoch)
    save_encoder(model, args.encoder)
    print(f"[encoder] salvato {args.encoder}")
    return model.eval()


# ------------------------------------------------------------------ passi

def raw_embeddings(enc, args, d):
    """Embedding NON standardizzati dei defect di d: {source: (defect_id, V)}."""
    present = [s for s in args.sources
               if os.path.exists(group_path(args.data_dir, d, s))]
    if not present:
        print(f"[{d}] nessun file dei gruppi trovato in {args.data_dir}/{d}, salto")
        return {}
    emb = embed_graph(enc, build_graph(read_interactions(train_path(args.data_dir, d))))
    out = {}
    for s in present:
        ids, defects = load_defects(group_path(args.data_dir, d, s))
        out[s] = (ids, defect_embeddings(emb, defects))
        print(f"[{d}/{s}] {len(ids)} defect")
    return out


def to_frame(ids, V, stats):
    X = standardize(V, stats).numpy()
    df = pd.DataFrame(X, columns=[f"e{j}" for j in range(X.shape[1])])
    df.insert(0, "defect_id", ids)
    return df


def add_labels(df, args, d, s):
    """Tiene solo i defect con label LOO valida e aggiunge impact."""
    path = os.path.join(args.loo_dir, LOO_NAME[s].format(d=d))
    if not os.path.exists(path):
        print(f"[{d}/{s}] manca {path}, salto")
        return None
    loo = read_loo(path)
    n_loo = len(loo)

    # controllo: la riga LOO deve riferirsi allo stesso gruppo del file dei gruppi
    _, defects = load_defects(group_path(args.data_dir, d, s))
    ok = np.zeros(len(loo), dtype=bool)
    for r, (k, us, its) in enumerate(zip(loo["defect_id"], loo["users"], loo["items"])):
        if pd.notna(k) and 0 <= int(k) < len(defects):
            gu, gi = defects[int(k)]
            ok[r] = set(us.split(",")) <= set(gu) and set(its.split(",")) <= set(gi)
    loo = loo[ok]

    loo = loo.assign(n_affected_all=loo["n_affected_all"])
    loo = loo.assign(n_affected_active=loo["n_affected_active"])
    loo = loo.assign(n_affected_lt=loo["n_affected_lt"])
    #loo = loo.assign(impact=loo["n_affected_all"] * (-loo["all_loc_d_prop_rr_best"]))
    loo = loo.assign(impact=loo["n_affected_all"] * (-loo["all_loc_d_prop_tndcg"]))
    loo = loo.assign(impact_lt=loo["n_affected_lt"] * (-loo["lt_loc_d_prop_rr_best"]))
    loo = loo.assign(impact_active=loo["n_affected_active"] * (-loo["active_loc_d_prop_rr_best"]))
    loo = loo.assign(impact_h=loo["n_affected_all"] * (loo["all_loc_d_prop_entropy"]))
    loo = loo.assign(entropy=loo["all_loc_d_prop_entropy"])
    loo = loo.dropna(subset=["defect_id", "impact"])
    loo["defect_id"] = loo["defect_id"].astype(int)
    out = loo[["defect_id", "n_affected_all","n_affected_lt","n_affected_active", "impact","impact_active","impact_lt", "impact_h", "entropy"]].merge(df, on="defect_id")
    print(f"[{d}/{s}] LOO {n_loo} righe | non allineate {int((~ok).sum())} | "
          f"con label {len(out)}")
    return out



def save_pickle(path, frames, with_labels):
    out = pd.concat(frames, ignore_index=True)
    cols = [c for c in out.columns if c[0] == "e" and c[1:].isdigit()]
    blob = {
        "X": out[cols].to_numpy(dtype=np.float32),
        "defect_id": out["defect_id"].to_numpy(dtype=np.int64),
        "source": out["source"].to_numpy(dtype=str),
    }
    #if with_labels:
    blob["impact_h"] = out["impact_h"].to_numpy(dtype=np.float32)
    blob["entropy"] = out["entropy"].to_numpy(dtype=np.float32)
    blob["impact"] = out["impact"].to_numpy(dtype=np.float32)
    blob["impact_active"] = out["impact_active"].to_numpy(dtype=np.float32)
    blob["impact_lt"] = out["impact_lt"].to_numpy(dtype=np.float32)
    blob["n_affected_lt"] = out["n_affected_lt"].to_numpy(dtype=np.float32)
    blob["n_affected_active"] = out["n_affected_active"].to_numpy(dtype=np.float32)
    blob["n_affected_all"] = out["n_affected_all"].to_numpy(dtype=np.float32)
    with open(path, "wb") as f:
        pickle.dump(blob, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  -> {path}  X {blob['X'].shape}")


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "original"))
    ap.add_argument("--loo-dir", default=str(ROOT / "recbole_f" / "results" / "loo"))
    ap.add_argument("--encoder", default=str(ROOT / "node_embeddings" / "weights" / "sage_shared_50_tndcg.pt"))
    ap.add_argument("--out-dir", default=str(ROOT / "node_embeddings" / "defects_tndcg"))
    ap.add_argument("--passive", default=",".join(PASSIVE))
    ap.add_argument("--active", default=",".join(ACTIVE))
    ap.add_argument("--sources", default=",".join(SOURCES))
    ap.add_argument("--out-dim", type=int, default=16,
                    help="dimensione per blocco; X ha 2 * out_dim colonne")
    ap.add_argument("--hid", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--steps-per-epoch", type=int, default=500)
    ap.add_argument("--retrain", action="store_true")
    args = ap.parse_args()

    split = lambda s: [x.strip() for x in s.split(",") if x.strip()]
    args.passive, args.active, args.sources = (
        split(args.passive), split(args.active), split(args.sources))
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[paths] data={args.data_dir}\n        loo={args.loo_dir}\n        out={args.out_dir}")

    set_seed(42)
    enc = get_encoder(args)

    # 1) passivi: embedding grezzi -> statistiche di standardizzazione
    raw = {d: raw_embeddings(enc, args, d) for d in args.passive}
    all_V = [V for per_src in raw.values() for (_, V) in per_src.values()]

    if not all_V:
        raise SystemExit("Nessun defect nei dataset passivi: controlla --data-dir.")
    stats = fit_standardizer(torch.cat(all_V))
    with open(os.path.join(args.out_dir, "std_stats.pkl"), "wb") as f:
        pickle.dump({k: v.numpy() for k, v in stats.items()}, f)

    # 2) passivi: un pickle per dataset, con label
    for d in args.passive:
        frames = []
        for s, (ids, V) in raw[d].items():
            df = add_labels(to_frame(ids, V, stats), args, d, s)
            if df is not None:

                df.insert(0, "source", s)
                frames.append(df)
        if frames:
            save_pickle(os.path.join(args.out_dir, f"{d}.pkl"), frames, True)
    del raw

    # 3) attivi: un pickle per dataset, senza label
    for d in args.active:
        print('ATTIVO', d)
        frames = []
        for s, (ids, V) in raw_embeddings(enc, args, d).items():
            df = add_labels(to_frame(ids, V, stats), args, d, s)
            #df = to_frame(ids, V, stats)
            #df.insert(0, "source", s)
            if df is not None:
                df.insert(0, "source", s)
                frames.append(df)
            #frames.append(df)
        if frames:
            save_pickle(os.path.join(args.out_dir, f"{d}.pkl"), frames, False)


if __name__ == "__main__":
    main()