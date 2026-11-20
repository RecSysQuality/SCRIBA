
from __future__ import annotations

import fnmatch
import re
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import numpy as np
import scipy.sparse as sp
import torch
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import get_model, get_trainer, init_seed

try:
    import run_my_eval as rc
    import loo_defects as lo
except ImportError:
    from recbole_f import run_my_eval as rc
    from recbole_f import loo_defects as lo


def log(msg: str) -> None:
    rc.log.info(msg)


MODEL = "LightGCN"
SEGMENTS = ("all", "active", "lt")
SUBSETS = SEGMENTS + tuple(f"aff_{s}" for s in SEGMENTS)
SEG_LABEL = {"all": "All", "active": "Active", "lt": "Long-tail"}
NEG_INF = -float("inf")


# ============================================================================ #
#  RECBOLE: dataset, training, caricamento
# ============================================================================ #
def build_env(ds: str, tag: str, staging_root: Path, saved_dir: Path) -> SimpleNamespace:
    """Config + dataset + dataloader con la STESSA configurazione di run_single."""
    saved_dir.mkdir(parents=True, exist_ok=True)
    params = lo.build_parameter_dict(ds, tag, staging_root, saved_dir)
    cfg_files = [f for f in (str(rc.CONFIGS_DIR / "datasets" / f"{ds}.yaml"),
                             str(rc.CONFIGS_DIR / "models" / f"{MODEL}.yaml")) if Path(f).exists()]
    cfg = Config(model=MODEL, dataset=tag, config_file_list=cfg_files, config_dict=params)
    init_seed(cfg["seed"], cfg["reproducibility"])
    dataset = create_dataset(cfg)
    train_data, valid_data, test_data = data_preparation(cfg, dataset)
    return SimpleNamespace(ds=ds, tag=tag, cfg=cfg, dataset=dataset, train=train_data,
                           valid=valid_data, test=test_data, saved_dir=saved_dir)


def fit_or_load(env: SimpleNamespace, force_train: bool):
    """Allena (se richiesto o se non c'è un checkpoint) e restituisce il modello migliore."""
    cfg = env.cfg
    ckpt = None if force_train else rc.find_checkpoint(env.saved_dir)
    init_seed(cfg["seed"], cfg["reproducibility"])
    model = get_model(MODEL)(cfg, env.train.dataset).to(cfg["device"])
    trainer = get_trainer(cfg["MODEL_TYPE"], cfg["model"])(cfg, model)
    t0 = time.time()
    if ckpt is None:
        log(f"  → Training {env.tag}")
        best_valid, _ = trainer.fit(env.train, env.valid, saved=True, show_progress=cfg["show_progress"])
        ckpt = Path(trainer.saved_model_file)
        trained = True
    else:
        log(f"  → Checkpoint esistente: {ckpt.name}")
        best_valid = torch.load(ckpt, map_location=cfg["device"]).get("best_valid_score")
        trained = False
    test_result = trainer.evaluate(env.test, load_best_model=True, model_file=str(ckpt),
                                   show_progress=False)
    elapsed = time.time() - t0

    ck = torch.load(ckpt, map_location=cfg["device"])
    for key in ("n_layers", "embedding_size"):
        if ck["config"][key] != cfg[key]:
            raise ValueError(f"{key}: checkpoint={ck['config'][key]} config={cfg[key]}")
    model = get_model(MODEL)(cfg, env.train.dataset).to(cfg["device"])
    model.load_state_dict(ck["state_dict"])
    if hasattr(model, "restore_user_e"):             # embedding propagati: ricalcolati
        model.restore_user_e = None
        model.restore_item_e = None
    model.eval()
    info = {"checkpoint": str(ckpt), "trained": trained, "elapsed_s": round(elapsed, 1),
            "best_valid_score": _num(best_valid), "recbole_test": {k: _num(v) for k, v in dict(test_result).items()}}
    log(f"  ✓ {'allenato' if trained else 'caricato'} in {elapsed:.0f}s  best_valid={info['best_valid_score']}")
    return model, info


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


def remap(dataset, field: str, ref_map: Dict[str, int]) -> np.ndarray:
    """ID interni di `dataset` -> ID del modello base (via token). -1 se il token non c'è."""
    return np.array([ref_map.get(t, -1) for t in dataset.field2id_token[field]], dtype=np.int64)


@torch.no_grad()
def collect(model, env, max_k: int, u_map: np.ndarray, i_map: np.ndarray) -> SimpleNamespace:
    """
    Top-max_k e hit di tutti gli utenti di test, come run_my_eval.eval_user_subset:
    padding e item già visti in training esclusi, ground truth dal dataloader.
    Utenti e item restituiti negli ID del modello base.
    """
    device = env.cfg["device"]
    n_items = env.dataset.item_num
    uid_field = env.dataset.uid_field
    U, T, H, N = [], [], [], []
    for batched in env.test:
        if not (isinstance(batched, (tuple, list)) and len(batched) == 4):
            raise RuntimeError("Il dataloader di test non è in modalità full-sort")
        interaction, history_index, positive_u, positive_i = batched
        interaction = interaction.to(device)
        uids = interaction[uid_field]
        scores = model.full_sort_predict(interaction).view(-1, n_items)
        scores[:, 0] = NEG_INF
        if history_index is not None:
            scores[history_index] = NEG_INF
        top = torch.topk(scores, k=max_k, dim=-1).indices
        pos = torch.zeros((uids.shape[0], n_items), dtype=torch.bool, device=device)
        pos[positive_u.to(device), positive_i.to(device)] = True
        hits = pos.gather(1, top)
        npos = pos.sum(dim=1)
        keep = npos > 0
        U.append(uids[keep].cpu().numpy())
        T.append(top[keep].cpu().numpy())
        H.append(hits[keep].cpu().numpy())
        N.append(npos[keep].cpu().numpy())
    users = u_map[np.concatenate(U)]
    top = i_map[np.concatenate(T)]
    if (users < 0).any() or (top < 0).any():
        raise RuntimeError("Utenti o item del modello non presenti nel modello base")
    return SimpleNamespace(users=users, top=top, hits=np.concatenate(H), npos=np.concatenate(N))


# ============================================================================ #
#  METRICHE
# ============================================================================ #
def gini(counts: np.ndarray) -> float:
    """Stessa formula di rc.compute_beyond_accuracy."""
    c = np.sort(counts.astype(float))
    n = len(c)
    if n <= 1:
        return 0.0
    c /= c.sum()
    g = 2 * np.sum(np.arange(1, n + 1) * c) / n - (n + 1) / n
    return float(min(max(g, 0.0), 1.0))


def beyond(top: np.ndarray, pop: np.ndarray, n_train: int, catalog: int,
           tail: np.ndarray | None = None) -> dict:
    """Beyond-accuracy su liste top [n_utenti, k]. Formule di rc.compute_beyond_accuracy (+ APLT)."""
    counts = np.bincount(top.ravel(), minlength=len(pop))
    nz = counts[counts > 0]
    p = nz / nz.sum()
    p_i = np.where(pop > 0, pop, 1).astype(float) / max(n_train, 1)   # pop mancante = 1
    out = {"entropy": float(-(p * np.log2(p + 1e-12)).sum()),
           "coverage": float(nz.size / catalog),
           "novelty": float((-np.log2(p_i[top])).mean(1).mean()),
           "gini": gini(nz)}
    if tail is not None:
        out["aplt"] = float(tail[top].mean(1).mean())                  # quota media di item long-tail
    return out


def tail_items(pop: np.ndarray, head_share: float) -> np.ndarray:
    """Item long-tail: tutto il catalogo tranne il `head_share` più popolare nel training (default 20%)."""
    items = np.arange(1, len(pop))                                     # 0 = padding
    order = items[np.argsort(-pop[items], kind="stable")]
    tail = np.zeros(len(pop), dtype=bool)
    tail[order[int(np.ceil(head_share * len(items))):]] = True
    return tail


def subset_metrics(res, sel: np.ndarray, ks, topk: int, pop: np.ndarray, n_train: int,
                   catalog: int, tail: np.ndarray | None = None) -> dict:
    """
    Metriche sugli utenti di `res` selezionati da `sel` (bool, allineato a res.users).
    Accuratezza come rc._topk_accuracy, beyond-accuracy come rc.compute_beyond_accuracy.
    """
    n = int(sel.sum())
    out = {"n_users": n}
    if n == 0:
        return out
    hits = res.hits[sel].astype(float)
    npos = res.npos[sel].astype(float)
    max_k = hits.shape[1]
    disc = 1.0 / np.log2(np.arange(1, max_k + 1) + 1)
    cum = np.cumsum(disc)
    for k in ks:
        h = hits[:, :k]
        nh = h.sum(1)
        first = h.argmax(1) + 1
        idcg = cum[np.minimum(npos, k).astype(int) - 1]
        out[f"ndcg@{k}"] = float(((h * disc[:k]).sum(1) / idcg).mean())
        out[f"recall@{k}"] = float((nh / npos).mean())
        out[f"hit@{k}"] = float((nh > 0).mean())
        out[f"mrr@{k}"] = float(np.where(nh > 0, 1.0 / first, 0.0).mean())

    top_all = res.top[sel]
    out.update(beyond(top_all[:, :topk], pop, n_train, catalog, tail))          # a --topk (compatibilità)
    for k in ks:                                                                # a ogni cutoff: <metrica>@k
        for m_, v_ in beyond(top_all[:, :k], pop, n_train, catalog, tail).items():
            out[f"{m_}@{k}"] = v_
    return out


def defect_exposure(res, sel: np.ndarray, dmask: np.ndarray, topk: int) -> dict:
    """
    Quanto gli item dei difetti rimossi compaiono nelle top-k degli utenti selezionati:
      defect_exposure  quota delle posizioni in top-k occupate da item dei difetti
      defect_reach     quota di utenti con almeno un item dei difetti in top-k
    È la misura diretta dell'effetto della rimozione: dovrebbe scendere dopo il retrain.
    """
    if not sel.any():
        return {}
    hit = dmask[res.top[sel][:, :topk]]
    return {"defect_exposure": float(hit.mean()), "defect_reach": float(hit.any(1).mean())}


def overlap_with(res0, res1, sel1: np.ndarray, topk: int) -> dict:
    """
    Per gli utenti selezionati di res1: quota media della top-k in comune con il modello
    base (1 = raccomandazioni identiche). Misura quanto è cambiato il modello, utente per utente.
    """
    row0 = np.full(max(res0.users.max(), res1.users.max()) + 1, -1, dtype=np.int64)
    row0[res0.users] = np.arange(len(res0.users))
    r1 = np.flatnonzero(sel1)
    r0 = row0[res1.users[r1]]
    ok = r0 >= 0
    r1, r0 = r1[ok], r0[ok]
    if len(r1) == 0:
        return {}
    n = len(r1)
    base = np.arange(n, dtype=np.int64)[:, None] * (int(max(res0.top.max(), res1.top.max())) + 1)
    k0 = (base + res0.top[r0][:, :topk]).ravel()
    k1 = (base + res1.top[r1][:, :topk])
    common = np.isin(k1, k0).reshape(n, -1).sum(1) / topk
    return {"overlap_base": float(common.mean()),
            "unchanged_users": float((common == 1.0).mean())}


def member_ids(defects, ref_uid) -> np.ndarray:
    """ID (base) di tutti gli utenti elencati nei difetti (i 'membri')."""
    toks = {u for d in defects for u in d.users}
    return np.array(sorted(ref_uid[t] for t in toks if t in ref_uid), dtype=np.int64)


def per_user(res, ks) -> Dict[str, np.ndarray]:
    """NDCG@k, Recall@k e MRR@k per utente (stesse formule di subset_metrics), per i test appaiati."""
    hits = res.hits.astype(float)
    npos = res.npos.astype(float)
    disc = 1.0 / np.log2(np.arange(1, hits.shape[1] + 1) + 1)
    cum = np.cumsum(disc)
    out = {"users": res.users}
    for k in ks:
        h = hits[:, :k]
        out[f"ndcg@{k}"] = ((h * disc[:k]).sum(1) / cum[np.minimum(npos, k).astype(int) - 1]).astype(np.float32)
        out[f"recall@{k}"] = (h.sum(1) / npos).astype(np.float32)
        nh = h.sum(1)
        out[f"hit@{k}"] = (nh > 0).astype(np.float32)
        out[f"mrr@{k}"] = np.where(nh > 0, 1.0 / (h.argmax(1) + 1), 0.0).astype(np.float32)
    return out


def all_subset_metrics(res, masks: Dict[str, np.ndarray], names, **kw) -> dict:
    return {s: subset_metrics(res, masks[s][res.users], **kw) for s in names}


def delta(after: dict, before: dict) -> dict:
    out = {}
    for s in after:
        a, b = after[s], before.get(s, {})
        out[s] = {m: a[m] - b[m] for m in a if m != "n_users" and m in b}
    return out


# ============================================================================ #
#  UTENTI AFFETTI E RIMOZIONE
# ============================================================================ #
def affected_users(res0, defects, ref_uid, ref_iid, n_items: int, k: int,
                   include_members: bool) -> np.ndarray:
    """
    ID (base) degli utenti di test con almeno un item di un difetto rimosso nella
    top-k del modello base, esclusi i membri di quel difetto (come loo_defects.py).
    """
    n = len(res0.users)
    top = res0.top[:, :k]
    T = sp.csc_matrix((np.ones(top.size, dtype=np.int8),
                       (np.repeat(np.arange(n), k), top.ravel())), shape=(n, n_items))
    row_of = np.full(max(res0.users.max() + 1, len(ref_uid)), -1, dtype=np.int64)
    row_of[res0.users] = np.arange(n)
    aff = np.zeros(n, dtype=bool)
    for d in defects:
        items = lo.to_ids(",".join(d.items), ref_iid)
        if len(items) == 0:
            continue
        rows = np.concatenate([T.indices[T.indptr[i]:T.indptr[i + 1]] for i in items])
        if not include_members and len(rows):
            m = row_of[lo.to_ids(",".join(d.users), ref_uid)]
            rows = np.setdiff1d(rows, m[m >= 0])
        aff[rows] = True
    return res0.users[aff]


def stage_removed(ds: str, tag: str, keep_rows: list, staging_root: Path) -> None:
    """File RecBole del dataset ripulito: training filtrato, valid e test invariati."""
    d = rc.DATASETS[ds]
    stage = staging_root / tag
    stage.mkdir(parents=True, exist_ok=True)
    rc._write_inter(stage / f"{tag}.train.inter", keep_rows, include_rating=False)
    for split, names in (("valid", [f"{ds}.rec_valid.csv", f"{ds}.rec_vali.csv"]),
                         ("test", [f"{ds}.rec_test.csv"])):
        rows = rc._parse_inter(rc._find_file(d, names), want_rating=False)
        rc._write_inter(stage / f"{tag}.{split}.inter", rows, include_rating=False)
    log(f"  Staged {tag}: train={len(keep_rows)}")


# ============================================================================ #
#  NOMI DEI FILE DI RIMOZIONE: metodo, scope, budget
# ============================================================================ #
METHODS_FILTER = None          # --methods: solo questi metodi (None = tutti); lo imposta main()


def keep_method(sel: str) -> bool:
    return METHODS_FILTER is None or selection_parts(sel)[0] in METHODS_FILTER


def find_removed_files(removed_dir: Path, ds: str, pattern: str) -> List[Path]:
    """
    File <ds>.<pattern> in removed_dir e in removed_dir/<ds>/.
    Nome del dataset senza distinzione tra maiuscole e minuscole, sia per la cartella
    (books/ = Books/) sia per il prefisso del file (books.removed_... = Books.removed_...).
    Ordine alfabetico sul percorso: è l'ordine degli indici di --file-index.
    """
    if not removed_dir.is_dir():
        log(f"  ATTENZIONE: {removed_dir} non esiste")
        return []
    dirs = [removed_dir] + [d for d in removed_dir.iterdir()
                            if d.is_dir() and d.name.lower() == ds.lower()]
    prefix = f"{ds.lower()}."
    files = set()
    for d in dirs:
        for f in d.glob("*.tsv"):
            if f.name.lower().startswith(prefix) and fnmatch.fnmatch(f.name[len(prefix):], pattern):
                files.add(f)
    return sorted(f for f in files if keep_method(selection_name(f, ds)))


def selection_name(path: Path, ds: str) -> str:
    """Books.removed_random_50K.tsv -> random_50K"""
    stem = path.stem
    if stem.lower().startswith(f"{ds.lower()}."):
        stem = stem[len(ds) + 1:]
    return stem[len("removed_"):] if stem.startswith("removed_") else stem


TARGETS = ("active", "lt")      # dopo '__' indicano il segmento obiettivo del metodo, non i candidati
SCOPE_ALIASES = {"group": "group_f", "rev2": "rev2_fff", "gsdb": "gsdb_ff"}   # mab__gsdb_50K = mab__gsdb_ff_50K


def selection_parts(sel: str):

    s = re.sub(r"_\d+K?$", "", sel)
    method, _, scope = s.partition("__")
    scope = SCOPE_ALIASES.get(scope, scope)
    if scope in TARGETS:
        method, scope = f"{method}_{scope}", "all"
    return method, (scope or "all")


ACRONYMS = {"mab": "MAB"}


def selection_label(sel: str) -> str:
    """Nome del metodo per la tabella: Stratified, Regressor lt, MAB, MAB active, ..."""
    method, _ = selection_parts(sel)
    words = method.split("_")
    words = [ACRONYMS.get(w, w) for w in words]
    if words[0] not in ACRONYMS.values():
        words[0] = words[0].capitalize()
    return " ".join(words)


BASELINES = ("stratified", "heuristic", "random")


def comparison_group(sel: str):
    """Gruppo di confronto di un file: (sorgente dei candidati, budget).
    'mab_lt__rev2_50K' -> ('rev2', '50K'). I file dello stesso gruppo finiscono nella stessa
    tabella e devono essere valutati sugli STESSI utenti."""
    m = re.search(r"_(\d+K?)$", sel)
    return selection_parts(sel)[1], (m.group(1) if m else "")
