
import argparse
import csv
import itertools
import time
import warnings
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.data.interaction import Interaction
from recbole.utils import get_model, init_seed
import random
import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from recbole_f import run_my_eval as rc

#import run_my_eval as rc          # stesso modulo usato per il training
#from recbole_f import run_my_eval as rc

log = rc.log
MODEL_NAME = "LightGCN"
NEG_INF = -float("inf")

# Azzerare archi già presenti in una CSR è sicuro: il warning di scipy è solo rumore
warnings.filterwarnings("ignore", category=sp.SparseEfficiencyWarning)


# ==============================================================================
#  CONFIGURAZIONE — deve essere IDENTICA a quella usata in run_single
# ==============================================================================
def build_parameter_dict(dataset_name: str, tag: str, staging_root: Path,
                         run_saved_dir: Path) -> dict:
    """
    Stessa configurazione del ramo LightGCN di run_single.
    Se cambi qualcosa lì, cambialo anche qui (oppure sposta questa funzione
    in run_cf.py e usala in entrambi i posti).
    """
    params = {
        "data_path":      str(staging_root),
        "dataset":        tag,
        "checkpoint_dir": str(run_saved_dir),
        "save_dataset":   False,
        "show_progress":  False,
        "load_col":       {"inter": ["user_id", "item_id", "timestamp"]},
        "train_neg_sample_args": {
            "distribution": "uniform",
            "sample_num":   3,
            "dynamic":      False,
        },
        "benchmark_filename": ["train", "valid", "test"],
        "eval_args": {"group_by": "user", "order": "TO", "mode": "full"},
    }
    params.update(rc.get_model_hparams(dataset_name, MODEL_NAME))
    return params


def load_base_model(dataset_name: str, staging_root: Path):
    """Carica dataset, split e checkpoint del modello di partenza."""
    dataset_dir = rc.DATASETS[dataset_name]
    noise = rc.build_noise_registry(dataset_name)["clean"]
    _, tag = rc.stage_dataset(
        dataset_name, dataset_dir, "clean", "0.0",
        noise["levels"]["0.0"], noise["subdir"], staging_root,
    )
    run_saved_dir = rc.SAVED_DIR / tag / MODEL_NAME
    ckpt_path = rc.find_checkpoint(run_saved_dir)
    if ckpt_path is None:
        raise FileNotFoundError(f"Nessun checkpoint in {run_saved_dir}")

    dataset_cfg = str(rc.CONFIGS_DIR / "datasets" / f"{dataset_name}.yaml")
    model_cfg   = str(rc.CONFIGS_DIR / "models" / f"{MODEL_NAME}.yaml")
    cfg_files   = [f for f in (dataset_cfg, model_cfg) if Path(f).exists()]

    params = build_parameter_dict(dataset_name, tag, staging_root, run_saved_dir)
    cfg = Config(model=MODEL_NAME, dataset=tag,
                 config_file_list=cfg_files, config_dict=params)
    init_seed(cfg["seed"], cfg["reproducibility"])

    dataset = create_dataset(cfg)
    train_data, _, _ = data_preparation(cfg, dataset)
    train_ds = train_data.dataset

    ckpt = torch.load(ckpt_path, map_location=cfg["device"])
    for key in ("n_layers", "embedding_size"):
        if ckpt["config"][key] != cfg[key]:
            raise ValueError(f"{key}: checkpoint={ckpt['config'][key]} "
                             f"config attuale={cfg[key]}. Le config non coincidono.")

    model = get_model(MODEL_NAME)(cfg, train_ds).to(cfg["device"])
    model.load_state_dict(ckpt["state_dict"])
    model.restore_user_e = None
    model.restore_item_e = None
    model.eval()

    log.info(f"  Checkpoint: {ckpt_path.name}")
    log.info(f"  Utenti={dataset.user_num - 1}  item={dataset.item_num - 1}  "
             f"interazioni train={len(train_ds.inter_feat)}")
    return cfg, dataset, train_ds, model


# ==============================================================================
#  GRAFO
# ==============================================================================
class TrainGraph:
    """Grafo di training come matrice utenti x item, con rimozione degli archi."""

    def __init__(self, train_ds, device):
        R = train_ds.inter_matrix(form="csr").astype(np.float32)
        R.data[:] = 1.0                      # eventuali duplicati contano una volta
        self.R = R
        self.n_users, self.n_items = R.shape
        self.device = device

    def norm_adj(self, R: sp.csr_matrix) -> torch.Tensor:
        """
        Matrice di adiacenza normalizzata D^-1/2 A D^-1/2, come
        LightGCN.get_norm_adj_mat di RecBole (stesso +1e-7 sul grado),
        ma vettorizzata: si ricostruisce in frazioni di secondo.
        """
        A = sp.bmat([[None, R], [R.T, None]], format="coo")
        deg = np.asarray(A.sum(axis=1)).ravel() + 1e-7
        d = np.power(deg, -0.5)
        vals = d[A.row] * A.data * d[A.col]
        idx = torch.from_numpy(np.vstack([A.row, A.col]).astype(np.int64))
        adj = torch.sparse_coo_tensor(
            idx, torch.from_numpy(vals.astype(np.float32)), A.shape)
        return adj.coalesce().to(self.device)

    def without_edges(self, users: np.ndarray, items: np.ndarray):
        """Toglie gli archi utenti x item presenti nel training."""
        sub = self.R[users][:, items].tocoo()
        if sub.nnz == 0:
            return self.R, 0
        R_mod = self.R.copy()
        R_mod[users[sub.row], items[sub.col]] = 0.0
        R_mod.eliminate_zeros()
        return R_mod, int(sub.nnz)


@torch.no_grad()
def propagate(model, adj):
    """Embedding finali di utenti e item propagati sul grafo `adj`."""
    model.norm_adj_matrix = adj
    model.restore_user_e = None
    model.restore_item_e = None
    return model.forward()


# ==============================================================================
#  TOP-K E RANK
# ==============================================================================
@torch.no_grad()
def topk_and_ranks(user_e, item_e, users, K, hist, items=None, batch=512):
    """
    Top-K per gli utenti indicati, escludendo l'item di padding e gli item
    già visti in training (hist). Se `items` è dato, restituisce anche il
    rank (1 = primo) di ciascuno di quegli item nel ranking completo;
    inf se l'item è escluso perché già visto dall'utente.
    """
    device = user_e.device
    tops, ranks = [], []
    items_t = torch.as_tensor(items, device=device) if items is not None else None

    for s in range(0, len(users), batch):
        ub = users[s:s + batch]
        sc = user_e[torch.as_tensor(ub, device=device)] @ item_e.T
        sc[:, 0] = NEG_INF
        r, c = hist[ub].nonzero()
        if len(r):
            sc[torch.as_tensor(r, device=device),
               torch.as_tensor(c, device=device)] = NEG_INF
        tops.append(torch.topk(sc, K, dim=1).indices.cpu().numpy())

        if items_t is not None:
            s_it = sc[:, items_t]                                  # [b, n_def]
            rk = torch.stack(
                [(sc > s_it[:, j:j + 1]).sum(dim=1) + 1 for j in range(len(items))],
                dim=1,
            ).float()
            rk[torch.isinf(s_it)] = float("inf")
            ranks.append(rk.cpu().numpy())

    top = np.vstack(tops) if tops else np.zeros((0, K), dtype=np.int64)
    rks = np.vstack(ranks) if items is not None and ranks else None
    return top, rks


# ==============================================================================
#  METRICHE
# ==============================================================================
def gini_from_counts(counts: np.ndarray) -> float:
    """Stessa formula di compute_beyond_accuracy in run_cf.py."""
    c = np.sort(counts.astype(float))
    n = len(c)
    if n <= 1:
        return 0.0
    c /= c.sum()
    idx = np.arange(1, n + 1)
    g = 2 * np.sum(idx * c) / n - (n + 1) / n
    return float(min(max(g, 0.0), 1.0))


def list_metrics(topk: np.ndarray, ranks: np.ndarray, K: int, catalog: int) -> dict:
    """
    Metriche sulle top-K di un gruppo di utenti.
      entropy, gini, coverage: sulla distribuzione degli item raccomandati
      rr_mean: media di 1/rank sulle coppie (utente, item del difetto)
      rr_best: media sugli utenti del miglior 1/rank tra gli item del difetto
      tndcg:   media sugli utenti dell'NDCG sul ranking intero (vedi per_user_tndcg)
      hit:     quota di coppie (utente, item del difetto) con l'item in top-K
    """
    _, counts = np.unique(topk.ravel(), return_counts=True)
    p = counts / counts.sum()
    rr = np.where(np.isfinite(ranks), 1.0 / ranks, 0.0)
    return {
        "entropy":  float(-(p * np.log2(p + 1e-12)).sum()),
        "gini":     gini_from_counts(counts),
        "coverage": len(counts) / catalog,
        "rr_mean":  float(rr.mean()),
        "rr_best":  float(rr.max(axis=1).mean()),
        "tndcg":    float(per_user_tndcg(ranks).mean()),
        "hit":      float((ranks <= K).mean()),
    }


def counts_metrics(counts: np.ndarray, catalog: int) -> dict:
    """Entropia, gini e coverage da un vettore di conteggi per item."""
    nz = counts[counts > 0]
    if nz.size == 0:
        return {"entropy": 0.0, "gini": 0.0, "coverage": 0.0}
    p = nz / nz.sum()
    return {
        "entropy":  float(-(p * np.log2(p + 1e-12)).sum()),
        "gini":     gini_from_counts(nz),
        "coverage": nz.size / catalog,
    }


def per_user_rr_hit(ranks: np.ndarray, K: int):
    """Per ogni utente: miglior 1/rank tra gli item del difetto e quota in top-K."""
    rr = np.where(np.isfinite(ranks), 1.0 / ranks, 0.0)
    return rr.max(axis=1), (ranks <= K).mean(axis=1)


def per_user_tndcg(ranks: np.ndarray) -> np.ndarray:
    """
    Per ogni utente: NDCG binario sul ranking INTERO, rilevanti = item del difetto.
      ranks  (n_utenti, n_item_difetto) da topk_and_ranks: rank nel ranking completo,
             inf se l'item è già visto (escluso dal ranking, quindi non rilevante)
      DCG    = sum_{rank finiti} 1/log2(rank+1)
      IDCG   = sum_{k=1..n_rel} 1/log2(k+1),  n_rel = n. di rank finiti
    Utente senza item rilevanti -> 0.
    """
    ranks = np.atleast_2d(np.asarray(ranks, float))
    fin = np.isfinite(ranks)
    dcg = np.where(fin, 1.0 / np.log2(np.where(fin, ranks, 1.0) + 1.0), 0.0).sum(axis=1)
    n_rel = fin.sum(axis=1)
    idcg = np.cumsum(1.0 / np.log2(np.arange(2, ranks.shape[1] + 2)))
    out = np.zeros(len(ranks))
    v = n_rel > 0
    out[v] = dcg[v] / idcg[n_rel[v] - 1]
    return out


LOC_METRICS = ("entropy", "gini", "coverage", "rr_mean", "rr_best", "tndcg", "hit")
GLOB_METRICS = ("entropy", "gini", "coverage", "rr_best", "tndcg", "hit")
METRICS = LOC_METRICS
GROUPS = ("all", "active", "lt")


# ==============================================================================
#  FINE-TUNING
# ==============================================================================
class FineTuner:
    """
    Riparte sempre dai pesi del modello di partenza e allena con BPR
    (stessa loss e stesso numero di negativi del training) sul grafo dato.
    """

    def __init__(self, model, graph: TrainGraph, epochs, lr, batch_size, n_neg):
        self.model = model
        self.graph = graph
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.n_neg = n_neg
        self.base_state = {k: v.detach().clone()
                           for k, v in model.state_dict().items()}

    def _negatives(self, users, R, rng, max_tries=10):
        """Negativi uniformi, ricampionati se l'utente ha già visto l'item."""
        neg = rng.integers(1, self.graph.n_items, size=len(users))
        for _ in range(max_tries):
            bad = np.asarray(R[users, neg]).ravel() > 0
            if not bad.any():
                break
            neg[bad] = rng.integers(1, self.graph.n_items, size=int(bad.sum()))
        return neg

    def run(self, R_mod, adj_mod, seed):
        m = self.model
        m.load_state_dict(self.base_state)
        m.norm_adj_matrix = adj_mod
        m.train()
        opt = torch.optim.Adam(m.parameters(), lr=self.lr)
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        coo = R_mod.tocoo()
        u_all = np.repeat(coo.row, self.n_neg)
        p_all = np.repeat(coo.col, self.n_neg)
        device = next(m.parameters()).device

        for _ in range(self.epochs):
            perm = rng.permutation(len(u_all))
            for s in range(0, len(perm), self.batch_size):
                idx = perm[s:s + self.batch_size]
                u, p = u_all[idx], p_all[idx]
                n = self._negatives(u, R_mod, rng)
                inter = Interaction({
                    m.USER_ID:     torch.as_tensor(u, dtype=torch.long),
                    m.ITEM_ID:     torch.as_tensor(p, dtype=torch.long),
                    m.NEG_ITEM_ID: torch.as_tensor(n, dtype=torch.long),
                }).to(device)
                loss = m.calculate_loss(inter)
                opt.zero_grad()
                loss.backward()
                opt.step()

        m.eval()
        user_e, item_e = propagate(m, adj_mod)
        return user_e.detach(), item_e.detach()

    def restore(self, adj_base):
        """Rimette il modello esattamente com'era prima del fine-tuning."""
        self.model.load_state_dict(self.base_state)
        self.model.norm_adj_matrix = adj_base
        self.model.restore_user_e = None
        self.model.restore_item_e = None
        self.model.eval()


# ==============================================================================
#  LETTURA DEI DIFETTI
# ==============================================================================
def read_defects(path: Path, offset: int, limit: int,
                 shuffle: bool = False, seed: int = 42):
    f = open(path, newline="", encoding="utf-8")
    reader = csv.DictReader(f, delimiter="\t")

    rows = list(enumerate(reader))   # lista, non islice
    fieldnames = reader.fieldnames
    f.close()

    if shuffle:
        random.Random(seed).shuffle(rows)

    end = offset + limit if limit else None
    rows = iter(rows[offset:end])    # offset/limit con slicing della lista
    return f, fieldnames, rows


def load_segment(dataset_name: str, segment: str, token2id: dict) -> np.ndarray:
    """
    ID interni degli utenti di un segmento ("active" o "lt"), letti dagli
    stessi file usati in run_cf.py. Prova sia il nome con il punto sia
    quello con l'underscore.
    """
    d = rc.DATASETS[dataset_name]
    for name in (f"{dataset_name}.rec_test_{segment}.csv",
                 f"{dataset_name}_rec_test_{segment}.csv"):
        path = d / name
        if path.exists():
            tokens = {row[0] for row in rc._parse_inter(path, want_rating=False)}
            ids = np.array(sorted(token2id[t] for t in tokens if t in token2id),
                           dtype=np.int64)
            log.info(f"  Segmento {segment}: {len(ids)} utenti ({path.name})")
            return ids
    raise FileNotFoundError(f"File del segmento '{segment}' non trovato in {d}")


def to_ids(tokens: str, token2id: dict) -> np.ndarray:
    """Token esterni → ID interni di RecBole (quelli sconosciuti sono ignorati)."""
    ids = [token2id[t] for t in tokens.split(",") if t and t in token2id]
    return np.array(sorted(set(ids)), dtype=np.int64)


# ==============================================================================
#  MAIN
# ==============================================================================
def parse_args():
    p = argparse.ArgumentParser(description="LOO sui difetti candidati")
    p.add_argument("--datasets",      nargs="+", default=list(rc.DATASETS.keys()))
    p.add_argument("--defects-file", default=None,
                   help="default: <data>/<dataset>/<dataset>.group_inj.csv")
    p.add_argument("--out", default=None,
                   help="default: results/loo/<dataset>_loo.csv")
    p.add_argument("--variants", nargs="+", default=["prop", "ft"],
                   choices=["prop", "ft"])
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--ft-epochs", type=int, default=5)
    p.add_argument("--ft-lr", type=float, default=None,
                   help="default: learning_rate del training")
    p.add_argument("--ft-batch", type=int, default=None,
                   help="default: train_batch_size del training")
    p.add_argument("--no-control", action="store_true",
                   help="non calcolare il controllo del fine-tuning")
    p.add_argument("--seed", type=int, default=2020)
    p.add_argument("--offset", type=int, default=0,
                   help="primo difetto da processare")
    p.add_argument("--limit", type=int, default=0,
                   help="numero di difetti da processare (0 = tutti)")
    p.add_argument("--score-batch", type=int, default=512)
    p.add_argument("--staging-dir", default=str(rc.STAGING_DIR))
    return p.parse_args()


def main():
    args = parse_args()
    ds_name = args.datasets[0]
    staging_root = Path(args.staging_dir)
    t = 'group_ff'
    defects_path = Path(args.defects_file) if args.defects_file else \
        rc.DATASETS[ds_name] / f"{ds_name}.{t}.csv"
    # t = 'group'
    # t = 'gsdb'
    # t = 'rev2'
    #
    # if t == 'group' or t == 'gsdb':
    #     defects_path = Path(args.defects_file) if args.defects_file else \
    #         rc.DATASETS[ds_name] / f"{ds_name}.{t}_f.csv"
    # else:
    #     defects_path = Path(args.defects_file) if args.defects_file else \
    #         rc.DATASETS[ds_name] / f"{ds_name}.rev2.csv"
    out_path = Path(args.out) if args.out else \
        rc.BASE_DIR / "results" / "loo" / f"{ds_name}___loo.csv"
    if args.offset or args.limit:
        out_path = out_path.with_name(
            f"{out_path.stem}_{t}_{args.offset}_{args.offset + args.limit}_tndcg{out_path.suffix}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    K = args.topk
    do_prop = "prop" in args.variants
    do_ft = "ft" in args.variants
    do_ctrl = do_ft and not args.no_control

    # ── Modello di partenza ───────────────────────────────────────────────────
    cfg, dataset, train_ds, model = load_base_model(ds_name, staging_root)
    graph = TrainGraph(train_ds, cfg["device"])
    catalog = graph.n_items - 1                      # escluso l'item di padding
    hist = graph.R                                   # item già visti in training

    # Controllo: la nostra adiacenza deve dare gli stessi embedding di RecBole
    with torch.no_grad():
        u_rb, i_rb = model.forward()
    adj_base = graph.norm_adj(graph.R)
    user_e0, item_e0 = propagate(model, adj_base)
    diff = max((u_rb - user_e0).abs().max().item(),
               (i_rb - item_e0).abs().max().item())
    log.info(f"  Controllo adiacenza: differenza massima con RecBole = {diff:.2e}")
    if diff > 1e-3:
        raise RuntimeError("L'adiacenza ricostruita non coincide con quella di RecBole")
    del u_rb, i_rb

    # Top-K di tutti gli utenti nel modello di partenza (per trovare gli affetti)
    all_users = np.arange(1, graph.n_users)          # ID interni; indice = id - 1
    t0 = time.time()
    top0_all, _ = topk_and_ranks(user_e0, item_e0, all_users, K, hist,
                                 batch=args.score_batch)
    log.info(f"  Top-{K} di partenza per {len(all_users)} utenti "
             f"in {time.time() - t0:.1f}s")

    # ── Gruppi di utenti: tutti, active, long-tail ────────────────────────────
    uid_map = dataset.field2token_id[dataset.uid_field]
    iid_map = dataset.field2token_id[dataset.iid_field]
    group_mask = {"all": np.ones(len(all_users), dtype=bool)}
    for seg in ("active", "lt"):
        ids = load_segment(ds_name, seg, uid_map)
        m = np.zeros(len(all_users), dtype=bool)
        m[ids - 1] = True
        group_mask[seg] = m
    pop_size = {g: int(m.sum()) for g, m in group_mask.items()}
    n_it = graph.n_items

    def bc(top_rows):
        return np.bincount(top_rows.ravel(), minlength=n_it)

    counts0 = {g: bc(top0_all[m]) for g, m in group_mask.items()}

    # ── Fine-tuning: setup e controllo sul grafo intero ───────────────────────
    tuner = None
    if do_ft:
        tuner = FineTuner(
            model, graph,
            epochs=args.ft_epochs,
            lr=args.ft_lr or cfg["learning_rate"],
            batch_size=args.ft_batch or cfg["train_batch_size"],
            n_neg=cfg["train_neg_sample_args"]["sample_num"],
        )
        log.info(f"  Fine-tuning: {args.ft_epochs} epoche  lr={tuner.lr}  "
                 f"batch={tuner.batch_size}  negativi={tuner.n_neg}")
    if do_ctrl:
        t0 = time.time()
        user_ec, item_ec = tuner.run(graph.R, adj_base, args.seed)
        tuner.restore(adj_base)
        topC_all, _ = topk_and_ranks(user_ec, item_ec, all_users, K, hist,
                                     batch=args.score_batch)
        countsC = {g: bc(topC_all[m]) for g, m in group_mask.items()}
        log.info(f"  Controllo fine-tuning (grafo intero) in {time.time() - t0:.1f}s")

    # ── Colonne di output ─────────────────────────────────────────────────────
    fh, in_fields, defects = read_defects(defects_path, args.offset, args.limit, shuffle=False)
    for col in ("users", "items"):
        if col not in in_fields:
            raise ValueError(f"Colonna '{col}' mancante in {defects_path}")

    states = (["before"] + (["prop"] if do_prop else [])
              + (["ft"] if do_ft else []) + (["ftctrl"] if do_ctrl else []))
    # differenze: (nome, stato "dopo", stato di riferimento)
    deltas = []
    if do_prop:
        deltas.append(("prop", "prop", "before"))
    if do_ft:
        deltas.append(("ft", "ft", "before"))
    if do_ctrl:
        deltas.append(("ftctrl", "ft", "ftctrl"))

    def scope_metrics(scope):
        return LOC_METRICS if scope == "loc" else GLOB_METRICS

    def scope_deltas(scope):
        # In glob, con il controllo attivo, "ft" usa come base le liste del
        # controllo per gli utenti non affetti: il confronto con "before"
        # mescolerebbe l'effetto delle epoche in più su tutti gli utenti.
        if scope == "glob" and do_ctrl:
            return [d for d in deltas if d[0] != "ft"]
        return deltas

    metric_cols, delta_cols = [], []
    for g in GROUPS:
        for scope in ("loc", "glob"):
            mets = scope_metrics(scope)
            metric_cols += [f"{g}_{scope}_{st}_{m}" for st in states for m in mets]
            delta_cols += [f"{g}_{scope}_d_{name}_{m}"
                           for name, _, _ in scope_deltas(scope) for m in mets]
    fields = (["defect_id", "n_users_found", "n_items_found", "n_edges_removed",
               "n_affected_all", "n_affected_active", "n_affected_lt", "seconds"]
              + delta_cols + metric_cols
              + ["users", "items"])                      # membri del difetto, in fondo

    out = open(out_path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(out, fieldnames=fields, delimiter="\t",
                            extrasaction="ignore", restval="")
    writer.writeheader()

    # ── Loop sui difetti ──────────────────────────────────────────────────────
    n_done = 0
    for k, row in defects:
        t0 = time.time()
        members = to_ids(row["users"], uid_map)
        items = to_ids(row["items"], iid_map)
        rec = {"defect_id": k, "n_users_found": len(members),
               "n_items_found": len(items),
               "users": row["users"], "items": row["items"]}

        if len(items) == 0:
            rec.update({"n_edges_removed": 0, "n_affected_all": 0,
                        "n_affected_active": 0, "n_affected_lt": 0})
            writer.writerow(rec)
            continue

        # Utenti affetti: item del difetto in top-K, esclusi i membri del difetto
        has_item = np.isin(top0_all, items).any(axis=1)
        affected = np.setdiff1d(all_users[has_item], members, assume_unique=True)
        aidx = affected - 1                              # righe in top0_all
        R_mod, n_removed = graph.without_edges(members, items)
        rec["n_edges_removed"] = n_removed
        for g in GROUPS:
            rec[f"n_affected_{g}"] = int(group_mask[g][aidx].sum())

        if len(affected) == 0:
            rec["seconds"] = round(time.time() - t0, 2)
            writer.writerow(rec)
            continue

        # ── Liste top-K e rank degli item del difetto, per ogni stato ─────────
        lists = {}
        lists["before"] = topk_and_ranks(user_e0, item_e0, affected, K, hist,
                                         items, batch=args.score_batch)
        adj_mod = graph.norm_adj(R_mod) if n_removed else adj_base

        if do_prop:                                      # solo propagazione
            ue, ie = propagate(model, adj_mod)
            lists["prop"] = topk_and_ranks(ue, ie, affected, K, hist, items,
                                           batch=args.score_batch)
            model.norm_adj_matrix = adj_base

        if do_ft:                                        # fine-tuning
            ue, ie = tuner.run(R_mod, adj_mod, args.seed)
            lists["ft"] = topk_and_ranks(ue, ie, affected, K, hist, items,
                                         batch=args.score_batch)
            tuner.restore(adj_base)

        if do_ctrl:                                      # controllo
            lists["ftctrl"] = topk_and_ranks(user_ec, item_ec, affected, K, hist,
                                             items, batch=args.score_batch)

        # ── Metriche per gruppo e scope ───────────────────────────────────────
        vals = {}                                        # (g, scope, state) -> dict
        for g in GROUPS:
            in_g = group_mask[g][aidx]                   # affetti di questo gruppo
            rows0 = top0_all[aidx][in_g]
            rowsC = topC_all[aidx][in_g] if do_ctrl else None

            for st, (top, rk) in lists.items():
                top_g, rk_g = top[in_g], rk[in_g]

                # loc: solo gli utenti affetti del gruppo
                if len(top_g):
                    vals[(g, "loc", st)] = list_metrics(top_g, rk_g, K, catalog)

                # glob: tutta la popolazione del gruppo, affetti sostituiti
                if st == "before":
                    cnt = counts0[g]
                elif st == "prop":
                    cnt = counts0[g] - bc(rows0) + bc(top_g)
                elif st == "ftctrl":
                    cnt = countsC[g]
                else:                                    # ft
                    base = (countsC[g] - bc(rowsC)) if do_ctrl else (counts0[g] - bc(rows0))
                    cnt = base + bc(top_g)
                gm = counts_metrics(cnt, catalog)
                if len(rk_g):
                    rr_u, hit_u = per_user_rr_hit(rk_g, K)
                    gm["rr_best"] = float(rr_u.sum() / pop_size[g])
                    gm["tndcg"] = float(per_user_tndcg(rk_g).sum() / pop_size[g])
                    gm["hit"] = float(hit_u.sum() / pop_size[g])
                else:
                    gm["rr_best"] = gm["tndcg"] = gm["hit"] = 0.0
                vals[(g, "glob", st)] = gm

        for (g, scope, st), mets in vals.items():
            for m, v in mets.items():
                rec[f"{g}_{scope}_{st}_{m}"] = round(v, 6)
        for g in GROUPS:
            for scope in ("loc", "glob"):
                for name, after, ref in scope_deltas(scope):
                    a, b = vals.get((g, scope, after)), vals.get((g, scope, ref))
                    if a is None or b is None:
                        continue
                    for m in scope_metrics(scope):
                        rec[f"{g}_{scope}_d_{name}_{m}"] = round(a[m] - b[m], 6)

        rec["seconds"] = round(time.time() - t0, 2)
        writer.writerow(rec)
        out.flush()
        n_done += 1
        if n_done % 20 == 0:
            log.info(f"  [{ds_name}] difetti processati: {n_done}  "
                     f"(ultimo: id={k}, affetti={len(affected)}, {rec['seconds']}s)")

    fh.close()
    out.close()
    log.info(f"  Completato: {n_done} difetti con utenti affetti → {out_path}")


if __name__ == "__main__":
    main()