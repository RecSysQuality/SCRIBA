
from __future__ import annotations

import argparse
import csv
import fnmatch
import gc
import json
import os
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
    import make_removed_defects as mrd
except ImportError:
    from recbole_f import run_my_eval as rc
    from recbole_f import loo_defects as lo
    from recbole_f import make_removed_defects as mrd

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


METHODS_FILTER = None          # --methods: solo questi metodi (None = tutti)


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
SCOPE_ALIASES = {"group": "group_ff","rev2": "rev2_fff", "gsdb": "gsdb_f", "gsdb": "gsdb_ff"}   # mab__gsdb_50K = mab__gsdb_f_50K
SCOPE_ALIASES = {"group": "group_f","rev2": "rev2_fff", "gsdb": "gsdb_f", "gsdb": "gsdb_ff"}   # mab__gsdb_50K = mab__gsdb_f_50K


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


# ============================================================================ #
#  PIPELINE PER DATASET
# ============================================================================ #
def run_dataset(ds: str, args) -> List[dict]:
    staging = Path(args.staging_dir)
    out_ds = args.out / ds
    out_ds.mkdir(parents=True, exist_ok=True)
    files = find_removed_files(args.removed_dir, ds, args.pattern)
    if not files:
        log(f"  [{ds}] nessun file {ds}.{args.pattern} in {args.removed_dir}: salto")
        return []
    all_files = list(files)                              # per --exclude-members union
    if args.file_index is not None:                      # un solo file (job array)
        if not 0 <= args.file_index < len(files):
            log(f"  [{ds}] --file-index {args.file_index} fuori range (0..{len(files) - 1}): salto")
            return []
        files = [files[args.file_index]]
    res_dir = out_ds / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    if args.skip_done:                                   # ripresa: salta i file già valutati
        todo = [f for f in files if not (res_dir / f"{selection_name(f, ds)}.json").exists()]
        if len(todo) < len(files):
            log(f"  [{ds}] già valutati, saltati: {[f.name for f in files if f not in todo]}")
        files = todo
        if not files:
            return []
    log(f"  [{ds}] {len(files)} file di difetti rimossi:")
    for f in files:
        log(f"      {f.resolve()}")
    ks = tuple(sorted(set(args.ks)))
    max_k = max(max(ks), args.topk, args.affected_k)

    # ── 1. Modello base ──────────────────────────────────────────────────────
    dataset_dir = rc.DATASETS[ds]
    noise = rc.build_noise_registry(ds)["clean"]
    _, base_tag = rc.stage_dataset(ds, dataset_dir, "clean", "0.0",
                                   noise["levels"]["0.0"], noise["subdir"], staging)
    base = build_env(ds, base_tag, staging, rc.SAVED_DIR / base_tag / MODEL)
    uid_f, iid_f = base.dataset.uid_field, base.dataset.iid_field
    ref_uid = base.dataset.field2token_id[uid_f]
    ref_iid = base.dataset.field2token_id[iid_f]
    uid_tok = base.dataset.field2id_token[uid_f]
    n_users, n_items = base.dataset.user_num, base.dataset.item_num
    catalog = n_items - 1

    model0, info0 = fit_or_load(base, force_train=args.retrain_base)
    res0 = collect(model0, base, max_k, np.arange(n_users), np.arange(n_items))
    graph = None
    if args.mode == "prop":
        # PROPAGAZIONE: parametri del modello base FISSI; dopo la rimozione si ricalcola solo la
        # propagazione di LightGCN sul training ripulito (come l'ambiente del MAB / loo_defects).
        graph = lo.TrainGraph(base.train.dataset, base.cfg["device"])
        lo.propagate(model0, graph.norm_adj(graph.R))     # controllo: grafo intero = modello base
        res_chk = collect(model0, base, max_k, np.arange(n_users), np.arange(n_items))
        same = float((res_chk.top[:, :20] == res0.top[:, :20]).all(1).mean())
        log(f"  [{ds}] modalità PROP: controllo sul grafo intero, top-20 identiche al modello base "
            f"per il {100 * same:.2f}% degli utenti (atteso ~100%)")
        if same < 0.95:
            log("  ATTENZIONE: la propagazione sul grafo intero non riproduce il modello base: "
                "controlla che TrainGraph.norm_adj corrisponda a LightGCN.get_norm_adj_mat")
        del res_chk
    else:
        del model0
    gc.collect(); torch.cuda.empty_cache()
    base_items = base.train.dataset.inter_feat[iid_f].numpy()
    pop0 = np.bincount(base_items, minlength=n_items)
    n_train0 = len(base_items)
    log(f"  [{ds}] base: {len(res0.users)} utenti di test, {catalog} item, {n_train0} interazioni di training")

    masks = {"all": np.ones(n_users, dtype=bool)}
    masks["all"][0] = False
    for seg in ("active", "lt"):
        m = np.zeros(n_users, dtype=bool)
        try:
            m[lo.load_segment(ds, seg, ref_uid)] = True
        except FileNotFoundError as e:
            log(f"  ATTENZIONE: {e}")
        masks[seg] = m
    tail0 = tail_items(pop0, args.head_share)            # long-tail fisso (training base) per tutti i modelli
    kw = dict(ks=ks, topk=args.topk, catalog=catalog, tail=tail0)

    # ── Gruppi di confronto ─────────────────────────────────────────────────
    # Un gruppo = i file confrontati nella stessa tabella: stessa sorgente (rev2, group_f, ...)
    # e stesso budget. Esempio Books/rev2/50K: heuristic, regressor, mab_all, mab_lt, ...
    # Membri esclusi (--exclude-members union) e utenti a rischio (--eval-users risk) si
    # calcolano SOLO sui file del gruppo: stessi utenti per i metodi confrontati tra loro.
    by_group: Dict[tuple, List[Path]] = {}
    for f in all_files:
        by_group.setdefault(comparison_group(selection_name(f, ds)), []).append(f)
    group_defs = {}
    for g_, fs in by_group.items():
        dd_all = []
        for f in fs:
            try:
                dd_all.append(mrd.read_removed(f, ds))
            except TypeError:
                dd_all.append(mrd.read_removed(f))
        group_defs[g_] = dd_all
    log(f"  [{ds}] gruppi di confronto (sorgente, budget): "
        + "; ".join(f"{g_[0]}/{g_[1]}: {[selection_parts(selection_name(f, ds))[0] for f in fs]}"
                    for g_, fs in sorted(by_group.items())))

    union_excl = {}
    if args.exclude_members == "union":                  # stessi utenti valutati per i metodi del gruppo
        for g_, dd_all in group_defs.items():
            ex = np.zeros(n_users, dtype=bool)
            for dd in dd_all:
                ex[member_ids(dd, ref_uid)] = True
            union_excl[g_] = ex
            log(f"  [{ds}] {g_[0]}/{g_[1]}: esclusi i membri di {len(dd_all)} file: "
                f"{int(ex[res0.users].sum())} utenti di test su {len(res0.users)}")

    # ── Utenti a rischio (--eval-users risk) ────────────────────────────────
    # Sul modello BASE: utenti con almeno --risk-min item sospetti nella top---risk-k.
    # Item sospetti = item dei difetti dei file DEL GRUPPO (i metodi confrontati tra loro).
    # Con --risk-min 1 --risk-k 20: utenti affetti da almeno una delle liste del gruppo.
    risk = {}
    if args.eval_users == "risk":
        if args.exclude_members != "union":
            log("  ATTENZIONE: con --eval-users risk usa anche --exclude-members union "
                "(altrimenti gli utenti valutati non sono gli stessi per tutti i metodi)")
        for g_, dd_all in group_defs.items():
            susp = np.zeros(n_items, dtype=bool)
            for dd in dd_all:
                for d in dd:
                    susp[lo.to_ids(",".join(d.items), ref_iid)] = True
            cnt = susp[res0.top[:, :args.risk_k]].sum(1)      # item sospetti per utente di test
            rk = np.zeros(n_users, dtype=bool)
            rk[res0.users[cnt >= args.risk_min]] = True
            risk[g_] = rk
            ok = ~union_excl[g_] if g_ in union_excl else np.ones(n_users, dtype=bool)
            log(f"  [{ds}] {g_[0]}/{g_[1]}: UTENTI A RISCHIO (>= {args.risk_min} item sospetti in "
                f"top-{args.risk_k}, {len(dd_all)} file, {int(susp.sum())} item sospetti): "
                + ", ".join(f"{s_}={int((masks[s_] & rk & ok)[res0.users].sum())}"
                            f"/{int((masks[s_] & ok)[res0.users].sum())}" for s_ in SEGMENTS)
                + "  (a rischio / totale)")

    peru_dir = out_ds / "peruser"                        # metriche per utente (test appaiati)
    peru_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(peru_dir / "base.npz", tokens=uid_tok[res0.users], **per_user(res0, ks))

    # training in token, per trovare gli archi da togliere
    train_src = dataset_dir / noise["subdir"] / noise["levels"]["0.0"]
    train_rows = rc._parse_inter(train_src, want_rating=False)
    u2i = mrd.user_items(train_rows)

    records = []
    for path in files:
        sel = selection_name(path, ds)
        log("=" * 72)
        meth, scope = selection_parts(sel)
        log(f"  {ds} · rimozione {sel}  (metodo={meth}, scope={scope}, tabella='{selection_label(sel)}')")
        log(f"  file: {path.resolve()}")
        log("=" * 72)
        t0 = time.time()
        try:
            defects = mrd.read_removed(path, ds)
        except TypeError:                                # make_removed_defects.py vecchio: read_removed(path)
            defects = mrd.read_removed(path)

        # ── 2. Archi da togliere e utenti affetti ────────────────────────────
        removed = set()
        for d in defects:
            removed.update(mrd.defect_edges(u2i, d))
        keep_rows = [r for r in train_rows if (r[0], r[1]) not in removed]
        n_rows_removed = len(train_rows) - len(keep_rows)
        if n_rows_removed == 0:
            log("  ATTENZIONE: nessuna riga del training rimossa! I token del file non corrispondono "
                "a quelli di rec_train.csv: il retrain sarebbe identico al base.")
        # utenti esclusi dalla valutazione (membri dei difetti), prima e dopo
        if args.exclude_members == "own":
            excl = np.zeros(n_users, dtype=bool)
            excl[member_ids(defects, ref_uid)] = True
        elif args.exclude_members == "union":
            excl = union_excl[comparison_group(sel)]
        else:
            excl = np.zeros(n_users, dtype=bool)
        n_excl = int(excl[res0.users].sum())
        seg_masks = {s_: masks[s_] & ~excl for s_ in SEGMENTS}
        if risk:                                         # valutazione solo sugli utenti a rischio del gruppo
            seg_masks = {s_: m_ & risk[comparison_group(sel)] for s_, m_ in seg_masks.items()}
            log(f"  utenti a rischio del gruppo {comparison_group(sel)}: "
                + ", ".join(f"{s_}={int(seg_masks[s_][res0.users].sum())}" for s_ in SEGMENTS))
        if args.exclude_members != "none":
            log(f"  esclusi dalla valutazione ({args.exclude_members}): {n_excl} membri dei difetti · "
                + ", ".join(f"{s_}={int(seg_masks[s_][res0.users].sum())}" for s_ in SEGMENTS) + " utenti valutati")
        aff_ids = affected_users(res0, defects, ref_uid, ref_iid, n_items,
                                 args.affected_k, args.include_members)
        aff = np.zeros(n_users, dtype=bool)
        aff[aff_ids] = True
        sub = dict(seg_masks)
        sub.update({f"aff_{s}": seg_masks[s] & aff for s in SEGMENTS})
        n_aff = {s: int(sub[f"aff_{s}"][res0.users].sum()) for s in SEGMENTS}
        log(f"  {len(defects)} difetti · {len(removed)} archi ({n_rows_removed} righe di training, "
            f"{100 * n_rows_removed / max(len(train_rows), 1):.2f}% del training) · "
            f"affetti: " + ", ".join(f"{s}={v}" for s, v in n_aff.items()))
        with open(out_ds / f"affected_{sel}.txt", "w") as f:
            f.write("\n".join(uid_tok[aff_ids]))

        # ── 3. Prima ─────────────────────────────────────────────────────────
        before = all_subset_metrics(res0, sub, SUBSETS, pop=pop0, n_train=n_train0, **kw)
        # item toccati dagli archi rimossi: la loro esposizione è l'effetto diretto della rimozione
        dmask = np.zeros(n_items, dtype=bool)
        dmask[[ref_iid[i] for i in {i for _, i in removed} if i in ref_iid]] = True
        for s_ in SUBSETS:
            before[s_].update(defect_exposure(res0, sub[s_][res0.users], dmask, args.topk))

        if args.mode == "prop":
            # ── 4-5. Rimozione + PROPAGAZIONE (parametri fissi, nessun retrain) ──
            t1 = time.time()
            ru = np.array([ref_uid.get(u, -1) for u, _ in removed], dtype=np.int64)
            ri = np.array([ref_iid.get(i, -1) for _, i in removed], dtype=np.int64)
            ok_ = (ru > 0) & (ri > 0)
            coo = graph.R.tocoo()
            keys = coo.row.astype(np.int64) * n_items + coo.col
            keep = ~np.isin(keys, ru[ok_] * n_items + ri[ok_])
            R_mod = sp.csr_matrix((coo.data[keep], (coo.row[keep], coo.col[keep])), shape=graph.R.shape)
            lo.propagate(model0, graph.norm_adj(R_mod))
            res1 = collect(model0, base, max_k, np.arange(n_users), np.arange(n_items))
            info1 = {"mode": "prop", "edges_removed_from_graph": int((~keep).sum()),
                     "elapsed_s": round(time.time() - t1, 1)}
            log(f"  ✓ propagazione: {int((~keep).sum())} archi tolti dal grafo "
                f"({int((~ok_).sum())} coppie non presenti nel modello) in {time.time() - t1:.0f}s")
            if args.novelty_pop == "own":
                pop1 = np.asarray(R_mod.sum(0)).ravel().astype(np.int64)
                n_train1 = int(R_mod.nnz)
            else:
                pop1, n_train1 = pop0, n_train0
            del R_mod, coo, keys, keep
        else:
            # ── 4-5. Rimozione + retrain ─────────────────────────────────────
            tag = re.sub(r"[^A-Za-z0-9_.]+", "_", f"{ds}__rm__{sel}")
            stage_removed(ds, tag, keep_rows, staging)
            env = build_env(ds, tag, staging, rc.SAVED_DIR / tag / MODEL)
            model1, info1 = fit_or_load(env, force_train=not args.skip_existing)
            u_map = remap(env.dataset, env.dataset.uid_field, ref_uid)
            i_map = remap(env.dataset, env.dataset.iid_field, ref_iid)
            res1 = collect(model1, env, max_k, u_map, i_map)
            del model1
            gc.collect(); torch.cuda.empty_cache()
            if args.novelty_pop == "own":
                items1 = i_map[env.train.dataset.inter_feat[env.dataset.iid_field].numpy()]
                pop1, n_train1 = np.bincount(items1, minlength=n_items), len(items1)
            else:
                pop1, n_train1 = pop0, n_train0

        # ── 6. Dopo ──────────────────────────────────────────────────────────
        after = all_subset_metrics(res1, sub, SUBSETS, pop=pop1, n_train=n_train1, **kw)
        np.savez_compressed(peru_dir / f"{sel}.npz", tokens=uid_tok[res1.users], **per_user(res1, ks),
                            **{f"in_{s_}": sub[s_][res1.users] for s_ in SEGMENTS})
        for s_ in SUBSETS:
            m1 = sub[s_][res1.users]
            after[s_].update(defect_exposure(res1, m1, dmask, args.topk))
            after[s_].update(overlap_with(res0, res1, m1, args.topk))

        rec = {"dataset": ds, "selection": sel, "label": selection_label(sel),
               "method": selection_parts(sel)[0], "scope": selection_parts(sel)[1], "file": str(path),
               "n_defects": len(defects),
               "defects_per_source": {s: sum(d.source == s for d in defects)
                                      for s in sorted({d.source for d in defects})},
               "n_edges_removed": len(removed), "n_train_rows_removed": n_rows_removed,
               "n_train_rows": len(train_rows), "n_affected": n_aff,
               "removed_share": n_rows_removed / max(len(train_rows), 1),
               "n_defect_items": int(dmask.sum()),
               "affected_k": args.affected_k, "include_members": args.include_members,
               "exclude_members": args.exclude_members, "n_excluded_test_users": n_excl,
               "eval_users": args.eval_users, "risk_k": args.risk_k, "risk_min": args.risk_min,
               "topk": args.topk, "ks": list(ks), "novelty_pop": args.novelty_pop,
               "head_share": args.head_share, "n_tail_items": int(tail0.sum()),
               "base": info0, "retrained": info1, "mode": args.mode,
               "before": before, "after": after, "delta": delta(after, before),
               "elapsed_s": round(time.time() - t0, 1)}
        records.append(rec)
        tk = args.table_k
        for s in SUBSETS:
            b, a = before[s], after[s]
            if a.get("n_users", 0) and b.get("n_users", 0):
                log(f"    [{s:10s}] n={a['n_users']:6d}  ndcg@{tk} {b[f'ndcg@{tk}']:.4f}→{a[f'ndcg@{tk}']:.4f}  "
                    f"entropy {b['entropy']:.3f}→{a['entropy']:.3f}  "
                    f"coverage {b['coverage']:.4f}→{a['coverage']:.4f}  "
                    f"novelty {b['novelty']:.3f}→{a['novelty']:.3f}  "
                    f"def.exp {b.get('defect_exposure', 0):.4f}→{a.get('defect_exposure', 0):.4f}  "
                    f"overlap {a.get('overlap_base', float('nan')):.3f}")
        rb, ra = info0.get("recbole_test", {}), info1.get("recbole_test", {})
        common_k = [k_ for k_ in rb if k_ in ra]
        if common_k:
            log("    RecBole test  " + "  ".join(f"{k_} {rb[k_]:.4f}→{ra[k_]:.4f}" for k_ in common_k[:6]))
        with open(res_dir / f"{sel}.json", "w") as f:        # un file per selezione: i job non si pestano
            json.dump(rec, f)
    return records


# ============================================================================ #
#  TABELLA LATEX + CSV
# ============================================================================ #
def table_metrics(k: int):
    return [(f"ndcg@{k}", f"NDCG@{k}", 4), (f"recall@{k}", f"Recall@{k}", 4),
            (f"hit@{k}", f"Hit@{k}", 4), ("entropy", "Entropy", 3),
            ("coverage", "Coverage", 4), ("novelty", "Novelty", 3),
            ("defect_exposure", "Def.~exp.", 4), ("overlap_base", "Overlap", 3)]


def _tex(s: str) -> str:
    return (str(s).replace("\\", r"\textbackslash{}").replace("_", r"\_")
            .replace("%", r"\%").replace("&", r"\&").replace("#", r"\#"))


def _cell(m: dict, key: str, dig: int, ref: dict | None = None) -> str:
    if not m or not m.get("n_users") or key not in m:
        return "--"
    v = m[key]
    s = f"{v:.{dig}f}"
    if ref is not None and ref.get("n_users") and key in ref:
        b = ref[key]
        if abs(b) > 1e-12:
            s += rf" {{\scriptsize ({(v - b) / abs(b) * 100:+.1f}\%)}}"
        else:
            s += rf" {{\scriptsize ({v - b:+.{dig}f})}}"
    return s


def _block(lines: list, seg_label: str, rows: list, mets) -> None:
    """rows: (nome riga, metriche, riferimento per il delta o None)."""
    for j, (name, m, ref) in enumerate(rows):
        first = rf"\multirow{{{len(rows)}}}{{*}}{{{seg_label}}}" if j == 0 else ""
        n = str(m.get("n_users", 0)) if m else "0"
        cells = " & ".join(_cell(m, key, dig, ref) for key, _, dig in mets)
        lines.append(f"{first} & {name} & {n} & {cells} \\\\")
    lines.append(r"\midrule")


def latex_table(ds: str, recs: List[dict], k: int, scope: str = "all") -> str:
    mets = table_metrics(k)
    r0 = recs[0]
    cand = ("all defect files together (group\\_f, gsdb\\_f, rev2)" if scope == "all"
            else f"only {_tex(scope)} defects")
    L = [r"\begin{table*}[t]", r"\centering", r"\footnotesize"]
    sel_desc = "; ".join(f"{_tex(r['label'])}: {r['n_defects']} defects, {r['n_edges_removed']} edges"
                         for r in recs)
    L.append(rf"\caption{{{_tex(ds)}, candidates: {cand} -- LightGCN before (Base) and after removing the "
             rf"selected defects from the training edges and "
             rf"{'re-propagating with fixed parameters' if r0.get('mode') == 'prop' else 'retraining'} ({sel_desc}). Top-{r0['topk']} lists for "
             rf"entropy, coverage and novelty. In parentheses: relative change with respect to Base on "
             rf"the same users. Affected users: at least one item of a removed defect in their "
             rf"top-{r0['affected_k']} list of the base model.}}")
    L.append(rf"\label{{tab:removed_{re.sub(r'[^A-Za-z0-9]+', '_', f'{ds}_{scope}')}}}")
    L.append(r"\begin{tabular}{ll r " + "r" * len(mets) + "}")
    L.append(r"\toprule")
    L.append("Users & Model & $|U|$ & " + " & ".join(h for _, h, _ in mets) + r" \\")
    L.append(r"\midrule")
    for s in SEGMENTS:                                   # tutti gli utenti del segmento
        base = r0["before"][s]
        rows = [("Base", base, None)] + [(_tex(r["label"]), r["after"][s], r["before"][s]) for r in recs]
        _block(L, SEG_LABEL[s], rows, mets)
    for s in SEGMENTS:                                   # solo gli affetti: base diverso per selezione
        rows = []
        for r in recs:
            lab = _tex(r["label"])
            rows.append((rf"Base\textsubscript{{{lab}}}", r["before"].get(f"aff_{s}", {}), None))
            rows.append((lab, r["after"].get(f"aff_{s}", {}), r["before"].get(f"aff_{s}", {})))
        _block(L, rf"\shortstack[l]{{Affected\\{SEG_LABEL[s]}}}", rows, mets)
    L[-1] = r"\bottomrule"
    L += [r"\end{tabular}", r"\end{table*}", ""]
    return "\n".join(L)


PAPER_SEGMENTS = [("all", "All users"), ("active", "Active users"), ("lt", "Long-tail users")]


def _paper_value(m: dict, key: str):
    """Valore per la tabella del paper: gini diventa 1-gini (più alto = meglio)."""
    if key.startswith("1-gini"):
        v = m.get("gini" + key[len("1-gini"):])
        return None if v is None else 1.0 - v
    return m.get(key)


def _paper_table(recs: List[dict], cols, caption: str, label: str, rank_prefixes=None, sig=None) -> str:
    r0 = recs[0]
    competes = [not rank_prefixes or selection_parts(r["selection"])[0].startswith(tuple(rank_prefixes))
                for r in recs]

    groups = []                                           # intestazione: metrica sopra, cutoff sotto
    for _, head, _ in cols:
        name, _, k = head.partition("@")
        if groups and groups[-1][0] == name:
            groups[-1][1].append(k)
        else:
            groups.append((name, [k]))
    rules, c0 = [], 3
    for _, ks_ in groups:
        rules.append(rf"\cmidrule(lr){{{c0}-{c0 + len(ks_) - 1}}}")
        c0 += len(ks_)

    L = [r"\begin{table*}[t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
         rf"\caption{{{caption}}}", rf"\label{{{label}}}",
         r"\begin{tabular}{ll" + "r" * len(cols) + "}", r"\toprule",
         " & & " + " & ".join(rf"\multicolumn{{{len(ks_)}}}{{c}}{{{nm}}}" for nm, ks_ in groups) + r" \\",
         " ".join(rules),
         "Users & Method & " + " & ".join(f"@{k}" for _, ks_ in groups for k in ks_) + r" \\",
         r"\midrule"]

    for si, (seg, seg_name) in enumerate(PAPER_SEGMENTS):
        rows = [("Base", r0["before"].get(seg, {}), True, False, None)]
        rows += [(_tex(r["label"]), r["after"].get(seg, {}), False, ok, r["selection"])
                 for r, ok in zip(recs, competes)]
        best = {}                                         # colonna -> valori dei metodi, dal migliore
        for key, _, _ in cols:
            best[key] = sorted({round(v, 6) for _, m, _, ok, _ in rows
                                if ok and m.get("n_users") and (v := _paper_value(m, key)) is not None},
                               reverse=True)
        seg_cell = r"\shortstack[l]{" + seg_name.replace(" ", r"\\") + "}"
        for j, (name, m, is_base, ok, sel_) in enumerate(rows):
            cells = []
            for key, _, dig in cols:
                v = _paper_value(m, key)
                if v is None or not m.get("n_users"):
                    cells.append("--")
                    continue
                txt = f"{v:.{dig}f}"
                if is_base:
                    txt = rf"\textit{{{txt}}}"
                elif ok and best[key] and round(v, 6) == best[key][0]:
                    txt = rf"\textbf{{{txt}}}"
                elif ok and len(best[key]) > 1 and round(v, 6) == best[key][1]:
                    txt = rf"\underline{{{txt}}}"
                if sig and sig.get((sel_, seg, key)):
                    txt += sig[(sel_, seg, key)]
                cells.append(txt)
            first = rf"\multirow{{{len(rows)}}}{{*}}{{{seg_cell}}}" if j == 0 else ""
            L.append(f"{first} & {name} & " + " & ".join(cells) + r" \\")
            if is_base:
                L.append(rf"\cmidrule(lr){{2-{2 + len(cols)}}}")
        L.append(r"\midrule" if si < len(PAPER_SEGMENTS) - 1 else r"\bottomrule")
    L += [r"\end{tabular}", r"\end{table*}", ""]
    return "\n".join(L)


def paired_ttest_p(a: np.ndarray, b: np.ndarray) -> float:
    """p-value a due code del t-test appaiato (come scipy.stats.ttest_rel)."""
    from scipy.special import stdtr
    d = a - b
    n = len(d)
    sd = d.std(ddof=1)
    if n < 2 or sd == 0:
        return 1.0 if d.mean() == 0 else 0.0
    t = d.mean() / (sd / np.sqrt(n))
    return float(2 * stdtr(n - 1, -abs(t)))


def load_peruser(out: Path, ds: str, sel: str):
    f = out / ds / "peruser" / f"{sel}.npz"
    if not f.exists():
        return None
    z = dict(np.load(f, allow_pickle=False))
    return z if "in_all" in z else None                  # file vecchi senza gruppi: niente test


def holm(pvals: Dict[str, float]) -> Dict[str, float]:
    """p-value corretti con Holm-Bonferroni (step-down)."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, run, out = len(items), 0.0, {}
    for rank, (k, p_) in enumerate(items):
        run = max(run, min(1.0, (m - rank) * p_))
        out[k] = run
    return out


def significance(out: Path, ds: str, recs: List[dict], refs: Dict[str, str], keys, alpha: float,
                 correction: str = "holm"):
    cache = {}

    def z_of(r):
        if r["selection"] not in cache:
            cache[r["selection"]] = load_peruser(out, ds, r["selection"])
        return cache[r["selection"]]

    marks, used = {}, {}
    for seg in SEGMENTS:
        ref = next((r for r in recs if selection_parts(r["selection"])[0] == refs.get(seg)), None)
        if ref is None:
            continue
        zr = z_of(ref)
        if zr is None:
            log(f"  [{ds}] test saltato per '{seg}': per-utente mancante per {ref['selection']} "
                "(rivaluta con la versione nuova e --skip-existing)")
            continue
        used[seg] = ref
        pos_r = {u: i_ for i_, u in enumerate(zr["users"].tolist())}
        others = [r for r in recs if r is not ref]
        for key in keys:
            pv, better = {}, {}
            for r in others:
                za = z_of(r)
                if za is None or key not in za or key not in zr:
                    continue
                ia = np.array([i_ for i_, u in enumerate(za["users"].tolist()) if u in pos_r], dtype=np.int64)
                ir = np.array([pos_r[u] for u in za["users"][ia].tolist()], dtype=np.int64)
                msk = za[f"in_{seg}"][ia] & zr[f"in_{seg}"][ir]
                if msk.sum() < 2:
                    continue
                a = zr[key][ir][msk].astype(float)          # riferimento
                b = za[key][ia][msk].astype(float)          # altro metodo
                pv[r["selection"]] = paired_ttest_p(a, b)
                better[r["selection"]] = a.mean() > b.mean()
            if not pv:
                continue
            padj = holm(pv) if correction == "holm" else pv
            wins = [sel for sel in pv if better[sel] and padj[sel] < alpha]
            for sel in wins:
                marks[(sel, seg, key)] = "$^{*}$"
            if len(wins) == len(pv) == len(others):
                marks[(ref["selection"], seg, key)] = r"$^{\dagger}$"
    return marks, used


def paper_tables(ds: str, scope: str, recs: List[dict], ks, beyond_ks, rank_prefixes=None,
                 out: Path | None = None, sig_refs: Dict[str, str] | None = None, alpha: float = 0.05,
                 correction: str = "holm") -> str:
    r0 = recs[0]
    cand = "all defect files" if scope == "all" else f"{_tex(scope)} defects only"
    who = {"none": "all test users",
           "own": "test users that are not members of the removed defects",
           "union": "test users that are not members of any removed defect set (same users for every method)"
           }[r0.get("exclude_members", "none")]
    if r0.get("eval_users") == "risk":
        who += (f", restricted to at-risk users (at least {r0['risk_min']} items of the defects removed by "
                f"any of the compared methods in their top-{r0['risk_k']} before removal; "
                f"same users for every method)")
    ranked = ("the best removal method" if not rank_prefixes else
              "the best among " + ", ".join("MAB" if p_ == "mab" else _tex(p_) for p_ in rank_prefixes))
    how = ("LightGCN with fixed parameters, re-propagated on the training graph" if r0.get("mode") == "prop"
           else "LightGCN retrained")
    note = (f"{how} after removing {r0['n_edges_removed']} training edges selected by each "
            f"method (candidates: {cand}); evaluated on {who}. Base: model before removal (reference, in "
            f"italics). Higher is better; within each user group {ranked} is in bold, the second is underlined.")
    tail_note = (r"APLT: share of long-tail items in the lists (long-tail = all but the "
                 + f"{int(round(100 * r0.get('head_share', 0.2)))}" + r"\% most popular training items).")
    lab = re.sub(r"[^A-Za-z0-9]+", "_", f"{ds}_{scope}")
    acc = [(f"{m}@{k}", f"{h}@{k}", 4)
           for m, h in (("ndcg", "NDCG"), ("recall", "Recall"), ("mrr", "MRR")) for k in ks]
    bey = [(f"{m}@{k}", f"{h}@{k}", d)
           for m, h, d in (("coverage", "Coverage", 3), ("entropy", "Entropy", 2), ("novelty", "Novelty", 2),
                           ("1-gini", "1$-$Gini", 3), ("aplt", "APLT", 3)) for k in beyond_ks]
    sig, used = ({}, {})
    if sig_refs and out is not None:
        sig, used = significance(out, ds, recs, sig_refs, [c[0] for c in acc], alpha, correction)
    acc_note = note
    if used:
        refs_txt = "; ".join(f"{name}: {_tex(used[g]['label'])}" for g, name in PAPER_SEGMENTS if g in used)
        corr = " with Holm correction" if correction == "holm" else ""
        acc_note += (rf" Paired t-test on the same users against the method targeted at each user group "
                     rf"({refs_txt}), $p<{alpha}${corr}. $^{{*}}$: the targeted method is significantly "
                     r"better than this method. $^{\dagger}$: the targeted method is significantly better "
                     r"than all the other methods.")
    return (_paper_table(recs, acc, rf"{_tex(ds)} -- accuracy. {acc_note}", f"tab:paper_acc_{lab}",
                         rank_prefixes, sig)
            + "\n" +
            _paper_table(recs, bey, rf"{_tex(ds)} -- beyond-accuracy. {note} {tail_note}",
                         f"tab:paper_beyond_{lab}", rank_prefixes))


def compact_table(ds: str, scope: str, recs: List[dict], k: int = 100, rank_prefixes=None,
                  out: Path | None = None, sig_refs: Dict[str, str] | None = None, alpha: float = 0.05,
                  correction: str = "holm") -> str:
    """Una sola tabella: Hit, Recall, Entropy, Coverage, Novelty a @k (default 100)."""
    r0 = recs[0]
    cand = "all defect files" if scope == "all" else f"{_tex(scope)} defects only"
    who = {"none": "all test users",
           "own": "test users that are not members of the removed defects",
           "union": "test users that are not members of any compared defect set (same users for every method)"
           }[r0.get("exclude_members", "none")]
    if r0.get("eval_users") == "risk":
        who += (f", restricted to at-risk users (at least {r0['risk_min']} items of the defects removed by "
                f"any of the compared methods in their top-{r0['risk_k']} before removal)")
    ranked = ("the best removal method" if not rank_prefixes else
              "the best among " + ", ".join("MAB" if p_ == "mab" else _tex(p_) for p_ in rank_prefixes))
    how = ("LightGCN with fixed parameters, re-propagated on the training graph" if r0.get("mode") == "prop"
           else "LightGCN retrained")
    note = (f"{how} after removing {r0['n_edges_removed']} training edges selected by each "
            f"method (candidates: {cand}); evaluated on {who}. Base: model before removal (reference, in "
            f"italics). All metrics at @{k}, higher is better; within each user group {ranked} is in bold, "
            f"the second is underlined.")
    cols = [(f"hit@{k}", f"Hit@{k}", 4), (f"recall@{k}", f"Recall@{k}", 4),
            (f"entropy@{k}", f"Entropy@{k}", 2), (f"coverage@{k}", f"Coverage@{k}", 3),
            (f"novelty@{k}", f"Novelty@{k}", 2)]
    sig, used = ({}, {})
    if sig_refs and out is not None:
        sig, used = significance(out, ds, recs, sig_refs, [f"hit@{k}", f"recall@{k}"], alpha, correction)
    if used:
        refs_txt = "; ".join(f"{name}: {_tex(used[g]['label'])}" for g, name in PAPER_SEGMENTS if g in used)
        corr = " with Holm correction" if correction == "holm" else ""
        note += (rf" Hit and Recall: paired t-test on the same users against the method targeted at each "
                 rf"user group ({refs_txt}), $p<{alpha}${corr}. $^{{*}}$: the targeted method is "
                 r"significantly better than this method. $^{\dagger}$: the targeted method is significantly "
                 r"better than all the other methods.")
    lab = re.sub(r"[^A-Za-z0-9]+", "_", f"{ds}_{scope}")
    return _paper_table(recs, cols, rf"{_tex(ds)} -- {note}", f"tab:compact_{lab}", rank_prefixes, sig)


SCOPE_LABEL = {"group_f": r"group\_f", "gsdb_f": r"gsdb\_f", "rev2": "rev2", "all": "all sources"}


def wins_table(groups, ks, rank_prefixes, out: Path) -> str:
    SEGMENTS = tuple(s_ for s_, _ in PAPER_SEGMENTS)     # solo i gruppi di utenti scelti (--paper-segments)
    """
    Un blocco di righe per scope (candidati: group_f, gsdb_f, rev2, all).
    Per ogni metodo e gruppo di utenti: in quanti casi è il migliore tra i metodi che competono,
    con casi = dataset x {NDCG, Recall, MRR} x cutoff. Più alto = meglio.
    """
    keys = [f"{m}@{k}" for m in ("ndcg", "recall", "mrr") for k in ks]
    scopes = sorted({sc for _, sc in groups}, key=lambda sc: (sc == "all", sc))
    datasets = sorted({ds for ds, _ in groups})
    L = [r"\begin{table}[t]", r"\centering", r"\footnotesize",
         r"\caption{Number of cases (" + ", ".join(_tex(d) for d in datasets)
         + r" $\times$ NDCG/Recall/MRR $\times$ cutoffs " + ", ".join(str(k) for k in ks)
         + r") in which each method is the best for each user group, per candidate set. Higher is better.}",
         r"\label{tab:paper_wins}",
         r"\begin{tabular}{ll" + "r" * len(SEGMENTS) + "}", r"\toprule",
         "Candidates & Method & " + " & ".join(name for _, name in PAPER_SEGMENTS) + r" \\", r"\midrule"]
    any_rows = False
    for si, scope in enumerate(scopes):
        count: Dict[str, Dict[str, int]] = {}
        n_cases = {s_: 0 for s_ in SEGMENTS}
        labels = {}
        for (ds, sc), recs in groups.items():
            if sc != scope:
                continue
            cand = [r for r in recs if not rank_prefixes
                    or selection_parts(r["selection"])[0].startswith(tuple(rank_prefixes))]
            for r in cand:
                meth = selection_parts(r["selection"])[0]
                labels[meth] = r["label"]
                count.setdefault(meth, {s_: 0 for s_ in SEGMENTS})
            for seg in SEGMENTS:
                for key in keys:
                    vals = [(r["after"].get(seg, {}).get(key), selection_parts(r["selection"])[0]) for r in cand]
                    vals = [(v, m_) for v, m_ in vals if v is not None]
                    if len(vals) < 2:
                        continue
                    n_cases[seg] += 1
                    top = max(v for v, _ in vals)
                    for v, m_ in vals:
                        if round(v, 6) == round(top, 6):
                            count[m_][seg] += 1
        if not count:
            continue
        any_rows = True
        meths = sorted(count)
        for j, meth in enumerate(meths):
            cells = []
            for seg in SEGMENTS:
                best = max(count[m_][seg] for m_ in count)
                c = f"{count[meth][seg]}/{n_cases[seg]}"
                cells.append(rf"\textbf{{{c}}}" if count[meth][seg] == best and best > 0 else c)
            first = rf"\multirow{{{len(meths)}}}{{*}}{{{SCOPE_LABEL.get(scope, _tex(scope))}}}" if j == 0 else ""
            L.append(f"{first} & {_tex(labels[meth])} & " + " & ".join(cells) + r" \\")
        L.append(r"\midrule" if si < len(scopes) - 1 else r"\bottomrule")
    if not any_rows:
        return ""
    if L[-1] == r"\midrule":
        L[-1] = r"\bottomrule"
    L += [r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(L)


def write_summary(records: List[dict], out: Path, k: int, rank_prefixes=None,
                  ks=(10, 20, 50, 100), beyond_ks=(20, 100), sig_refs=None, alpha=0.05,
                  paper_scopes=None, correction="holm", compact_k=100) -> None:
    if not records:
        log("Nessun risultato: tabella non scritta")
        return
    # una tabella per (dataset, scope): euristica e regressori sugli stessi candidati
    groups: Dict[tuple, List[dict]] = {}
    for r in records:
        scope = r.get("scope") or selection_parts(r["selection"])[1]
        groups.setdefault((r["dataset"], scope), []).append(r)
    scope_rank = lambda sc: (sc == "all", sc)                    # prima le singole sorgenti, poi all
    tex = ["% Richiede \\usepackage{booktabs} e \\usepackage{multirow}", ""]
    per_ds: Dict[str, List[str]] = {}
    for (ds, scope) in sorted(groups, key=lambda g: (g[0], scope_rank(g[1]))):
        # prima le baseline (stratified / heuristic / random), poi i regressori
        recs = sorted(groups[(ds, scope)],
                      key=lambda r: (selection_parts(r["selection"])[0] not in BASELINES, r["selection"]))
        t = latex_table(ds, recs, k, scope)
        per_ds.setdefault(ds, []).append(t)
        tex.append(t)
    for ds, ts in per_ds.items():
        (out / ds).mkdir(parents=True, exist_ok=True)
        (out / ds / "summary.tex").write_text("\n".join(ts), encoding="utf-8")
    (out / "summary_removed.tex").write_text("\n".join(tex), encoding="utf-8")

    # tabelle per il paper: valori assoluti, più alto = meglio, niente delta
    paper = ["% Richiede \\usepackage{booktabs} e \\usepackage{multirow}. Valori assoluti: più alto = meglio.",
             "% Celle '--': metrica assente nel JSON (valutato con una versione precedente: "
             "rivaluta con --skip-existing).", ""]
    pgroups = {g: v for g, v in groups.items() if not paper_scopes or g[1] in paper_scopes}
    for (ds, scope) in sorted(pgroups, key=lambda g: (g[0], scope_rank(g[1]))):
        recs = sorted(pgroups[(ds, scope)],
                      key=lambda r: (selection_parts(r["selection"])[0] not in BASELINES, r["selection"]))
        paper.append(f"% ===== {ds} · candidati: {scope} · accuratezza + beyond-accuracy =====")
        paper.append(paper_tables(ds, scope, recs, ks, beyond_ks, rank_prefixes, out, sig_refs, alpha,
                                  correction))
    compact = [f"% Richiede \\usepackage{{booktabs}} e \\usepackage{{multirow}}. "
               f"Hit, Recall, Entropy, Coverage, Novelty @{compact_k}.", ""]
    for (ds, scope) in sorted(pgroups, key=lambda g: (g[0], scope_rank(g[1]))):
        recs = sorted(pgroups[(ds, scope)],
                      key=lambda r: (selection_parts(r["selection"])[0] not in BASELINES, r["selection"]))
        compact.append(f"% ===== {ds} · candidati: {scope} · tabella compatta @{compact_k} =====")
        compact.append(compact_table(ds, scope, recs, compact_k, rank_prefixes, out, sig_refs, alpha,
                                     correction))
    (out / "paper_compact.tex").write_text("\n".join(compact), encoding="utf-8")
    log(f"Tabella compatta @{compact_k} → {out / 'paper_compact.tex'}")
    paper.append("% ===== riepilogo: vittorie per candidati e gruppo di utenti =====")
    paper.append(wins_table(pgroups, ks, rank_prefixes, out))
    (out / "paper_removed.tex").write_text("\n".join(paper), encoding="utf-8")
    log(f"Tabelle paper → {out / 'paper_removed.tex'}")

    keys = sorted({m for r in records for st in ("before", "after") for s in r[st].values() for m in s})
    keys = ["n_users"] + [m for m in keys if m != "n_users"]
    with open(out / "summary_removed.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["dataset", "scope", "method", "selection", "n_defects", "n_edges_removed",
                    "users", "state"] + keys)
        for r in records:
            meth, scope = selection_parts(r["selection"])
            for s in SUBSETS:
                for st in ("before", "after", "delta"):
                    m = r[st].get(s, {})
                    w.writerow([r["dataset"], scope, meth, r["selection"], r["n_defects"],
                                r["n_edges_removed"], s, st]
                               + [m.get(key, "") for key in keys])
    log(f"Tabella LaTeX → {out / 'summary_removed.tex'}")
    log(f"CSV          → {out / 'summary_removed.csv'}")


def load_records(out: Path, datasets=None) -> List[dict]:
    """Un record per (dataset, selezione): results/<sel>.json; i vecchi results.jsonl solo se
    la stessa selezione non ha già un file nuovo."""
    recs = {}
    for p in sorted(out.glob("*/results.jsonl")):          # formato vecchio
        with open(p) as f:
            for l in f:
                if l.strip():
                    r = json.loads(l)
                    recs[(r["dataset"], r["selection"])] = r
    for p in sorted(out.glob("*/results/*.json")):         # formato nuovo: vince
        with open(p) as f:
            r = json.load(f)
        recs[(r["dataset"], r["selection"])] = r
    return [r for r in recs.values() if (not datasets or r["dataset"] in datasets)
            and keep_method(r["selection"])]


# ============================================================================ #
#  CLI
# ============================================================================ #
def parse_args():
    p = argparse.ArgumentParser(description="Metriche prima/dopo la rimozione dei difetti + retrain",
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--datasets", nargs="+", default=list(rc.DATASETS.keys()))
    p.add_argument("--removed-dir", type=Path, default=rc.BASE_DIR / "removed_defects",
                   help="cartella con i file <ds>.removed_*.tsv (anche in sottocartelle <ds>/)")
    p.add_argument("--pattern", default="*_ablation.tsv",
                   help="quali file usare, dopo '<ds>.' (es. 'removed_random_*.tsv')")
    p.add_argument("--topk", type=int, default=50, help="lunghezza delle liste per entropia/coverage/novelty")
    p.add_argument("--ks", nargs="+", type=int, default=[10, 20, 50, 100],
                   help="cutoff per NDCG/Recall/Hit/MRR e per la beyond-accuracy (<metrica>@k)")
    p.add_argument("--head-share", type=float, default=0.2,
                   help="quota di item più popolari (training base) considerati 'head'; il resto è long-tail (APLT)")
    p.add_argument("--paper-beyond-ks", nargs="+", type=int, default=[20, 100],
                   help="cutoff della tabella beyond-accuracy del paper")
    p.add_argument("--affected-k", type=int, default=20,
                   help="top-k del modello base usata per trovare gli utenti affetti (come loo_defects)")
    p.add_argument("--exclude-members", choices=["none", "own", "union"], default="none",
                   help="toglie dagli utenti valutati (tutti i gruppi, prima e dopo) chi compare nei difetti: "
                        "own = membri dei difetti di quel file; union = membri di tutti i file trovati "
                        "(stessi utenti per tutti i metodi: usare questo per confrontare i metodi). "
                        "Senza --out, i risultati vanno in results/removed_nomembers_<modo>/")
    p.add_argument("--methods", nargs="+", default=None,
                   help="valuta solo questi metodi (es. heuristic regressor stratified mab_all); i file degli "
                        "altri metodi vengono ignorati, anche per l'esclusione dei membri e le tabelle")
    p.add_argument("--paper-segments", nargs="+", choices=["all", "active", "lt"], default=["all", "active", "lt"],
                   help="gruppi di utenti nelle tabelle del paper (es. solo 'all')")
    p.add_argument("--mode", choices=["retrain", "prop"], default="retrain",
                   help="retrain = LightGCN riallenato sul training ripulito; prop = parametri del modello "
                        "base fissi, si ricalcola solo la propagazione sul grafo ripulito (nessun rumore "
                        "di training, come l'ambiente del MAB)")
    p.add_argument("--eval-users", choices=["segment", "risk"], default="segment",
                   help="segment = tutti gli utenti di ogni segmento; risk = solo gli utenti a rischio "
                        "(stessi per tutti i metodi, vedi --risk-k e --risk-min)")
    p.add_argument("--risk-k", type=int, default=10,
                   help="top-K del modello base usata per definire gli utenti a rischio")
    p.add_argument("--risk-min", type=int, default=2,
                   help="numero minimo di item sospetti nella top---risk-k per essere a rischio")
    p.add_argument("--include-members", action="store_true",
                   help="conta come affetti anche gli utenti membri dei difetti")
    p.add_argument("--novelty-pop", choices=["base", "own"], default="base",
                   help="popolarità per la novelty: training base (default) o training di ogni modello")
    p.add_argument("--paper-rank", nargs="+", default=None, metavar="PREFIX",
                   help="tabelle del paper: grassetto/sottolineato solo tra i metodi con questi prefissi "
                        "(es. --paper-rank mab); gli altri restano come riferimento. Default: tutti")
    p.add_argument("--paper-scopes", nargs="+", default=None,
                   help="per quali candidati fare le tabelle del paper (group_f gsdb_f rev2 all); default: tutti")
    p.add_argument("--sig-ref", nargs="+", default=["all=mab_all", "active=mab_active", "lt=mab_lt"],
                   metavar="GRUPPO=METODO",
                   help="riferimento del test appaiato per ogni gruppo di utenti (default: all=mab_all "
                        "active=mab_active lt=mab_lt); 'none' per disattivarlo")
    p.add_argument("--correction", choices=["holm", "none"], default="holm",
                   help="correzione per confronti multipli dentro ogni (gruppo, metrica)")
    p.add_argument("--alpha", type=float, default=0.05, help="soglia del test appaiato")
    p.add_argument("--table-k", type=int, default=20, help="k delle metriche di accuratezza in tabella")
    p.add_argument("--compact-k", type=int, default=100,
                   help="k della tabella compatta (Hit, Recall, Entropy, Coverage, Novelty): paper_compact.tex")
    p.add_argument("--retrain-base", action="store_true",
                   help="riallena anche il modello base invece di usare il checkpoint di run_my_eval")
    p.add_argument("--skip-existing", action="store_true",
                   help="se il modello riallenato ha già un checkpoint lo riusa invece di riallenare")
    p.add_argument("--table-only", action="store_true",
                   help="non allena: rifà tabella e CSV dai risultati salvati in --out")
    p.add_argument("--list-files", action="store_true",
                   help="stampa i file trovati per ogni dataset con il loro indice ed esce")
    p.add_argument("--file-index", type=int, default=None,
                   help="valuta solo il file con questo indice (vedi --list-files): per i job array. "
                        "Default: $SLURM_ARRAY_TASK_ID se definita e --array, altrimenti tutti")
    p.add_argument("--array", action="store_true",
                   help="usa $SLURM_ARRAY_TASK_ID come --file-index")
    p.add_argument("--skip-done", action="store_true",
                   help="salta i file che hanno già un risultato in --out (ripresa dopo un'interruzione)")
    p.add_argument("--staging-dir", default=str(rc.STAGING_DIR))
    p.add_argument("--out", type=Path, default=None,
                   help="default results/removed (o results/removed_nomembers_<modo> con --exclude-members)")
    return p.parse_args()


def main():
    global METHODS_FILTER, PAPER_SEGMENTS
    args = parse_args()
    if args.methods:
        METHODS_FILTER = set(args.methods)
    PAPER_SEGMENTS = [(s_, lab) for s_, lab in PAPER_SEGMENTS if s_ in args.paper_segments]
    if [x.lower() for x in args.sig_ref] == ["none"]:
        args.sig_refs = None
    else:
        args.sig_refs = dict(x.split("=", 1) for x in args.sig_ref)
    if args.out is None:                                 # risultati separati: non sovrascrivono gli altri
        name = "removed" if args.exclude_members == "none" else f"removed_nomembers_{args.exclude_members}"
        if args.eval_users == "risk":
            name += f"_risk{args.risk_k}x{args.risk_min}"
        if args.mode == "prop":
            name += "_prop"
        if args.methods:                                 # un sottoinsieme di metodi: cartella a parte
            name += "_m-" + "+".join(sorted(args.methods))
        args.out = rc.BASE_DIR / "results" / name
    args.out.mkdir(parents=True, exist_ok=True)
    if args.table_k not in args.ks or args.compact_k not in args.ks:
        args.ks = sorted(set(args.ks) | {args.table_k, args.compact_k})
    if args.array and args.file_index is None:
        args.file_index = int(os.environ["SLURM_ARRAY_TASK_ID"])
    if args.list_files:
        print("dataset\tindex\tmethod\tscope\tlabel\tpath")
        for ds in args.datasets:
            if ds not in rc.DATASETS:
                continue
            files = find_removed_files(args.removed_dir, ds, args.pattern)
            for i, f in enumerate(files):
                sel = selection_name(f, ds)
                meth, scope = selection_parts(sel)
                print(f"{ds}\t{i}\t{meth}\t{scope}\t{selection_label(sel)}\t{f.resolve()}")
            print(f"# {ds}: {len(files)} file → {len(files)} retrain (--array=0-{max(len(files) - 1, 0)})")
        return
    if not args.table_only:
        for ds in args.datasets:
            if ds not in rc.DATASETS:
                log(f"Dataset sconosciuto '{ds}', saltato.")
                continue
            try:
                run_dataset(ds, args)
            except Exception as exc:                     # un dataset fallito non ferma gli altri
                rc.log.exception(f"  ✗ {ds} fallito: {exc}")
    if args.file_index is not None and not args.table_only:
        log("Job su un solo file: tabella non scritta (a job finiti: --table-only)")
        return                                           # più job in parallelo non scrivono la stessa tabella
    write_summary(load_records(args.out), args.out, args.table_k, args.paper_rank,
                  sorted(set(args.ks)), args.paper_beyond_ks, args.sig_refs, args.alpha, args.paper_scopes,
                  args.correction, args.compact_k)


if __name__ == "__main__":
    main()