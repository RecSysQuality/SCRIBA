from __future__ import annotations

import argparse
import csv
import glob
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import scipy.sparse as sp

from mab_core import (GROUPS, IMPACTS, LOO_REWARD_COL, PAIRS, POL, Defect, FeatureBuilder, RewardScaler,
                      SegmentLinUCB, _extra, _fmt, _loo, _tok, offline_matrices, offline_raws,
                      read_defect_rows, segment_metrics, source_of_loo, structural_metrics,
                      tndcg_full)


# ============================================================================ #
#  DATI: grafo, gruppi, test, LOO offline
# ============================================================================ #
def group_masks(n_users: int, seg_ids: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    masks = {"all": np.ones(n_users, bool)}
    masks["all"][0] = False
    for g, ids in seg_ids.items():
        m = np.zeros(n_users, bool)
        m[ids] = True
        masks[g] = m
    return masks


def load_graph(ds_name: str, staging_root: Path):
    lo = _loo()
    cfg, dataset, train_ds, model = lo.load_base_model(ds_name, staging_root)
    graph = lo.TrainGraph(train_ds, cfg["device"])
    uid = dataset.field2token_id[dataset.uid_field]
    iid = dataset.field2token_id[dataset.iid_field]
    seg = {g: lo.load_segment(ds_name, g, uid) for g in ("active", "lt")}
    masks = group_masks(graph.n_users, seg)
    return cfg, graph, model, uid, iid, masks


def _read_pairs(path: Path):
    """Coppie (utente, item) da un file di interazioni (tsv/csv, header RecBole o semplice)."""
    import pandas as pd
    with open(path, encoding="utf-8") as f:
        head = f.readline()
    sep = "\t" if "\t" in head else ","
    df = pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False)
    df.columns = [c.split(":")[0].strip() for c in df.columns]
    return df["user_id"].to_numpy(), df["item_id"].to_numpy()


def load_test_matrix(cfg, ds_name, uid, iid, shape, test_file=None, log=None):
    """Test split per NDCG/recall. Ordine: --test-file, <data>/<ds>.test.csv,
    file .test.inter dello staging RecBole."""
    lo = _loo()
    cands = []
    if test_file:
        cands.append(Path(test_file))
    cands.append(lo.rc.DATASETS[ds_name] / f"{ds_name}.test.csv")
    dp, tag = Path(cfg["data_path"]), cfg["dataset"]
    cands += [dp / f"{tag}.test.inter", dp / tag / f"{tag}.test.inter"]
    for p in cands:
        if p.exists():
            us, it = _read_pairs(p)
            u = np.array([uid.get(x, -1) for x in us])
            i = np.array([iid.get(x, -1) for x in it])
            ok = (u > 0) & (i > 0)
            T = sp.csr_matrix((np.ones(ok.sum(), np.float32), (u[ok], i[ok])), shape=shape)
            T.data[:] = 1.0
            if log:
                log.info(f"  Test split: {p.name} ({ok.sum()} interazioni, {int((np.diff(T.indptr) > 0).sum())} utenti)")
            return T
    if log:
        log.info("  ATTENZIONE: test split non trovato, NDCG/recall/hit non calcolati (usa --test-file)")
    return None


def load_offline(ds_name, staging_root, loo_dir, extra_cols, reward_col=LOO_REWARD_COL["rr"]):
    """reward_col: colonna della reward nei CSV del LOO, con {g} al posto del gruppo
    (es. {g}_glob_d_prop_rr_best oppure {g}_glob_d_prop_tndcg). Reward = -colonna."""
    lo = _loo()
    cfg, graph, model, uid, iid, masks = load_graph(ds_name, staging_root)
    del model
    rows_by_key = {}
    for path in sorted(glob.glob(str(Path(loo_dir) / f"{ds_name}___loo*_tndcg.csv"))):
        s = source_of_loo(path)
        for row in read_defect_rows(Path(path)):
            rows_by_key[(s, int(row["defect_id"]))] = row
    if not rows_by_key:
        raise FileNotFoundError(f"Nessun CSV LOO per {ds_name} in {loo_dir}")
    cols = {g: reward_col.format(g=g) for g in GROUPS}
    present = set().union(*(r.keys() for r in rows_by_key.values()))
    missing = [c for c in cols.values() if c not in present]
    if missing:
        glob_cols = sorted(c for c in present if "_glob_" in c)
        raise KeyError(f"[offline {ds_name}] colonne della reward mancanti nei CSV LOO di {loo_dir}: "
                       f"{missing}. Colonne '_glob_' presenti: {glob_cols}. "
                       f"Rigenera il LOO con la misura scelta o indica la colonna con --loo-reward-col")
    # CSV vecchi (senza la colonna) mescolati a quelli nuovi: le loro righe avrebbero reward 0
    n_old = sum(any(c not in r for c in cols.values()) for r in rows_by_key.values())
    if n_old:
        raise KeyError(f"[offline {ds_name}] {n_old}/{len(rows_by_key)} difetti vengono da CSV LOO senza "
                       f"la colonna {reward_col} (file vecchi nella stessa cartella {loo_dir}?). "
                       f"Separa i CSV nuovi e indica la cartella con --loo-dir")
    src_rows = {}
    if extra_cols:
        for s in {k[0] for k in rows_by_key}:
            src_rows[s] = read_defect_rows(lo.rc.DATASETS[ds_name] / f"{ds_name}.{s}.csv")
    sizes = {g: max(int(m.sum()), 1) for g, m in masks.items()}
    defects, rew, n_aff = [], {g: [] for g in GROUPS}, {g: [] for g in GROUPS}
    for (s, k), row in sorted(rows_by_key.items()):
        users, items = lo.to_ids(row["users"], uid), lo.to_ids(row["items"], iid)
        if len(users) == 0 or len(items) == 0:
            continue
        extra = _extra(src_rows[s][k], extra_cols) if extra_cols else np.zeros(0)
        defects.append(Defect(k, users, items, extra, s, _tok(row["users"]), _tok(row["items"])))
        for g in GROUPS:
            v = row.get(cols[g], "")
            rew[g].append(-float(v) if v not in ("", None) else 0.0)
            na = row.get(f"n_affected_{g}", "")
            n_aff[g].append(float(na) if na not in ("", None) else 0.0)
    rew = {g: np.array(v) for g, v in rew.items()}
    exposure = {g: np.array(n_aff[g]) / sizes[g] for g in GROUPS}
    per = {s: sum(d.source == s for d in defects) for s in sorted({d.source for d in defects})}
    lo.log.info(f"  [offline {ds_name}] {len(defects)} difetti con reward LOO {per} "
                f"(colonna {reward_col})")
    return graph.R, masks, defects, rew, exposure


# ============================================================================ #
#  AMBIENTE: il LightGCN del dataset, stessa misura del LOO
# ============================================================================ #
def remove_edges(R, users, items):
    sub = R[users][:, items].tocoo()
    if sub.nnz == 0:
        return R, np.zeros(0, np.int64), np.zeros(0, np.int64)
    R2 = R.copy()
    ru, ri = users[sub.row], items[sub.col]
    R2[ru, ri] = 0.0
    R2.eliminate_zeros()
    return R2, ru, ri


class EnvironmentBase:
    """
    Parte comune a Environment (LightGCN reale) e FakeEnvironment (test).
    Le sottoclassi implementano _propagate(R), _topk(ue, ie, users), _rr_best(ue, ie, users, items)
    e, per --impact tndcg, _tndcg(ue, ie, users, items) oppure _scores(ue, ie, users).
    impact: "rr" (1/rank del miglior item del difetto)
          | "tndcg" (NDCG sul ranking intero, rilevanti = item del difetto)
    """

    def _setup(self, R, masks, K, uid_tok, iid_tok, test_R=None, head_frac=0.2, impact="rr"):
        import warnings
        warnings.filterwarnings("ignore", category=sp.SparseEfficiencyWarning)
        if impact not in IMPACTS:
            raise ValueError(f"impact sconosciuto: {impact} (ammessi: {IMPACTS})")
        self.impact = impact
        self.R0 = R.copy()
        self.R = R.copy()
        self.hist = R.copy().tocsr()             # item visti: training originale, come nel LOO
        self.masks, self.K = masks, K
        self.n_users, self.n_items = R.shape
        self.all_users = np.arange(1, self.n_users)
        self.sizes = {g: max(int(m.sum()), 1) for g, m in masks.items()}
        self.uid_tok, self.iid_tok = uid_tok, iid_tok
        self.test_R = test_R
        self.pop0 = np.asarray(R.sum(0)).ravel().astype(float)
        order = np.argsort(-self.pop0[1:]) + 1
        n_head = int(round(head_frac * (self.pop0[1:] > 0).sum()))
        self.tail = np.ones(self.n_items, bool)
        self.tail[order[:n_head]] = False
        self.tail[0] = False
        self.def_items = np.zeros(self.n_items, bool)
        self.removed_pairs: set = set()
        self.joint_max_items = 2000

    def set_defect_items(self, defects):
        self.def_items[:] = False
        for d in defects:
            self.def_items[d.items] = True

    def reset(self):
        self.R = self.R0.copy()
        self.removed_pairs = set()

    def begin_round(self):
        self.ue, self.ie = self._propagate(self.R)
        self.top_all = self._topk(self.ue, self.ie, self.all_users)
        u = np.repeat(self.all_users, self.K)
        self.hit = sp.csc_matrix((np.ones(len(u), bool), (u, self.top_all.ravel())),
                                 shape=(self.n_users, self.n_items))

    def group_item_hits(self, g) -> np.ndarray:
        """Per ogni item: n. di utenti del gruppo g che lo hanno in top-K (dopo begin_round).
        Un difetto con tutti i suoi item a 0 NON può toccare g (reward r_g = 0 esatta)."""
        return np.asarray(self.hit.T @ self.masks[g].astype(np.float64)).ravel()

    def exposure(self, defects) -> Dict[str, np.ndarray]:
        """Quota di utenti del gruppo (esclusi i membri) con un item del difetto in top-K."""
        out = {g: np.zeros(len(defects)) for g in GROUPS}
        for k, d in enumerate(defects):
            sub = self.hit[:, d.items]
            if sub.nnz == 0:
                continue
            exposed = np.flatnonzero(np.asarray(sub.sum(1)).ravel() > 0)
            exposed = np.setdiff1d(exposed, d.users)
            for g in GROUPS:
                out[g][k] = self.masks[g][exposed].sum() / self.sizes[g]
        return out

    def _impact_score(self, ue, ie, users, items) -> np.ndarray:
        """Punteggio per utente su cui si misura l'impatto (prima - dopo la rimozione).
        rr:    1/rank del miglior item del difetto
        tndcg: NDCG sul ranking intero, rilevanti = item del difetto"""
        if self.impact == "rr":
            return self._rr_best(ue, ie, users, items)
        return self._tndcg(ue, ie, users, items)

    def _tndcg(self, ue, ie, users, items) -> np.ndarray:
        """Default (FakeEnvironment): dai punteggi completi di _scores, versione numpy."""
        return tndcg_full(self._scores(ue, ie, users), items)

    def _measure(self, users_rm, items, R_mod):
        has = np.isin(self.top_all, items).any(axis=1)
        affected = np.setdiff1d(self.all_users[has], users_rm, assume_unique=True)
        info = {"glob": {g: 0.0 for g in GROUPS}, "loc": {g: np.nan for g in GROUPS},
                "n_aff": {g: 0 for g in GROUPS}, "n_aff_total": int(affected.size)}
        if affected.size == 0:
            return info
        s0 = self._impact_score(self.ue, self.ie, affected, items)
        ue1, ie1 = self._propagate(R_mod)
        s1 = self._impact_score(ue1, ie1, affected, items)
        diff = s0 - s1
        for g in GROUPS:
            m = self.masks[g][affected]
            info["glob"][g] = float(diff[m].sum() / self.sizes[g])
            info["loc"][g] = float(diff[m].mean()) if m.any() else np.nan
            info["n_aff"][g] = int(m.sum())
        return info

    def reward(self, d: Defect):
        """Per gruppo: glob (= reward del MAB, -glob_d_prop_<impact>),
        loc (= label del regressore, -loc_d_prop_<impact>), n_aff."""
        R_mod, _, _ = remove_edges(self.R, d.users, d.items)
        return self._measure(d.users, d.items, R_mod)

    def joint_effect(self, batch: Sequence[Defect]):
        """Stessa misura, ma rimuovendo INSIEME tutto il batch (item = unione)."""
        if not batch:
            return None
        items = np.unique(np.concatenate([d.items for d in batch]))
        if len(items) > self.joint_max_items:
            return None
        members = np.unique(np.concatenate([d.users for d in batch]))
        R_mod = self.R
        for d in batch:
            R_mod, _, _ = remove_edges(R_mod, d.users, d.items)
        return self._measure(members, items, R_mod)

    def remove(self, defects: Sequence[Defect]) -> int:
        n = 0
        for d in defects:
            self.R, ru, ri = remove_edges(self.R, d.users, d.items)
            n += len(ru)
            if self.uid_tok is not None:
                self.removed_pairs.update(f"{self.uid_tok[u]}\x1f{self.iid_tok[i]}" for u, i in zip(ru, ri))
        return n

    def affected_users(self, d: Defect) -> np.ndarray:
        """Utenti con almeno un item del difetto nella top-K attuale (esclusi i membri):
        stessa definizione di 'affetto' del LOO e di reward()."""
        has = np.isin(self.top_all, d.items).any(axis=1)
        return np.setdiff1d(self.all_users[has], d.users, assume_unique=True)

    def metrics_on(self, masks, top, ref_top=None):
        """Metriche di stato su un sottoinsieme di utenti, per una top-K data
        (attuale o quella del round 0)."""
        return segment_metrics(top, masks, self.n_items, self.pop0, self.tail,
                               self.def_items, self.test_R, ref_top)

    def state_metrics(self, ref_top=None):
        seg = segment_metrics(self.top_all, self.masks, self.n_items, self.pop0, self.tail,
                              self.def_items, self.test_R, ref_top)
        return seg, structural_metrics(self.R, self.R0, self.masks)


class Environment(EnvironmentBase):
    """Produce la reward. Il MAB non lo vede: riceve solo i numeri."""

    def __init__(self, ds_name, staging_root, K, score_batch, test_file=None, head_frac=0.2, log=None,
                 impact="rr"):
        self.lo = _loo()
        self.cfg, self.graph, self.model, uid, iid, masks = load_graph(ds_name, staging_root)
        self.uid, self.iid, self.batch = uid, iid, score_batch
        inv_u = np.empty(self.graph.n_users, dtype=object)
        inv_i = np.empty(self.graph.n_items, dtype=object)
        for t, i in uid.items():
            inv_u[i] = t
        for t, i in iid.items():
            inv_i[i] = t
        test_R = load_test_matrix(self.cfg, ds_name, uid, iid, self.graph.R.shape, test_file, log)
        self._setup(self.graph.R, masks, K, inv_u, inv_i, test_R, head_frac, impact)

    def _propagate(self, R):
        return self.lo.propagate(self.model, self.graph.norm_adj(R))

    def _topk(self, ue, ie, users):
        return np.asarray(self.lo.topk_and_ranks(ue, ie, users, self.K, self.hist, batch=self.batch)[0])

    def _rr_best(self, ue, ie, users, items):
        """1/rank del MIGLIORE item del difetto (esclusi gli item già visti).
        Identico a per_user_rr_hit(topk_and_ranks(...))[0] del LOO: il rank minimo
        tra gli item è 1 + #item con punteggio > max dei punteggi degli item.
        Un solo confronto per utente invece di uno per item."""
        import torch
        dev = ue.device
        it = torch.as_tensor(items, device=dev)
        out = []
        with torch.no_grad():
            for s in range(0, len(users), self.batch):
                ub = users[s:s + self.batch]
                sc = ue[torch.as_tensor(ub, device=dev)] @ ie.T
                sc[:, 0] = -float("inf")
                r, c = self.hist[ub].nonzero()
                if len(r):
                    sc[torch.as_tensor(r, device=dev), torch.as_tensor(c, device=dev)] = -float("inf")
                smax = sc[:, it].max(dim=1).values
                rank = (sc > smax[:, None]).sum(dim=1).float() + 1
                rr = torch.where(torch.isinf(smax), torch.zeros_like(rank), 1.0 / rank)
                out.append(rr.cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)

    def _tndcg(self, ue, ie, users, items):
        """NDCG sul ranking INTERO (non troncato), rilevanti = item del difetto.
        Stesso ranking di _rr_best (item 0 e già visti esclusi, rank = 1 + #punteggi maggiori),
        ma per OGNI item del difetto: un ordinamento dei punteggi per utente e una ricerca
        binaria per item. Identico a tndcg_full (versione numpy)."""
        import torch
        dev = ue.device
        items = np.unique(np.asarray(items, dtype=np.int64))
        it = torch.as_tensor(items, device=dev)
        idcg = torch.cumsum(1.0 / torch.log2(torch.arange(2, len(items) + 2, device=dev,
                                                          dtype=torch.float64)), 0)
        out = []
        with torch.no_grad():
            for s in range(0, len(users), self.batch):
                ub = users[s:s + self.batch]
                sc = ue[torch.as_tensor(ub, device=dev)] @ ie.T
                sc[:, 0] = -float("inf")
                r, c = self.hist[ub].nonzero()
                if len(r):
                    sc[torch.as_tensor(r, device=dev), torch.as_tensor(c, device=dev)] = -float("inf")
                s_it = sc[:, it].contiguous()                                # (b, n_def)
                srt = torch.sort(sc, dim=1).values                           # crescente
                n_gt = sc.shape[1] - torch.searchsorted(srt, s_it, right=True)  # punteggi > s_it
                valid = torch.isfinite(s_it)
                gain = 1.0 / torch.log2(n_gt.to(torch.float64) + 2.0)        # 1/log2(rank+1)
                dcg = torch.where(valid, gain, torch.zeros_like(gain)).sum(1)
                n_rel = valid.sum(1)
                nd = torch.where(n_rel > 0, dcg / idcg[(n_rel - 1).clamp(min=0)], torch.zeros_like(dcg))
                out.append(nd.cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)


# ============================================================================ #
#  HOLDOUT
# ============================================================================ #
@dataclass
class Holdout:
    """Difetti misurati UNA volta sul grafo iniziale, esclusi dalla selezione:
    stesso set di valutazione per tutti i round e tutte le politiche."""
    defects: List[Defect]
    info: List[dict]
    X_ctx: Optional[Dict[str, np.ndarray]] = None   # contesto MAB al round 0
    X_emb: Optional[np.ndarray] = None              # embedding GraphSAGE al round 0

    @property
    def keys(self):
        return {d.key for d in self.defects}

    def glob(self, g):
        return np.array([i["glob"][g] for i in self.info])

    def loc(self, g):
        return np.array([i["loc"][g] for i in self.info])


def measure_holdout(env, defects, n, seed, fb, embedder, log):
    rng = np.random.default_rng(seed + 991)
    idx = rng.choice(len(defects), min(n, len(defects)), replace=False)
    hd = [defects[i] for i in idx]
    env.reset()
    env.begin_round()
    t0 = time.time()
    info = [env.reward(d) for d in hd]
    exposure = env.exposure(hd) if fb.use_exposure else None
    X, _, _, ok, _ = fb.transform(env.R, hd, env.masks, exposure)
    hd, info = [hd[j] for j in ok], [info[j] for j in ok]
    X_emb = embedder.embed(hd, set()) if embedder is not None else None
    log.info(f"  Holdout: {len(hd)} difetti misurati sul grafo iniziale in {time.time() - t0:.0f}s")
    return Holdout(hd, info, X, X_emb)


# ============================================================================ #
#  CONTROLLO E PRIOR OFFLINE
# ============================================================================ #
def offline_check(off, fb_base, log, raws=None):
    from scipy.stats import spearmanr
    names = list(off)
    raws = raws if raws is not None else offline_raws(off, log)
    scaler = RewardScaler().fit({g: np.concatenate([off[n][3][g] for n in names]) for g in GROUPS})
    for n in names:
        rew = off[n][3]
        rs = spearmanr(rew["lt"], rew["active"]).correlation
        t20 = lambda r: set(np.argsort(-r)[:20])
        ov = len(t20(rew["lt"]) & t20(rew["active"])) / 20
        log.info(f"  [{n}] Spearman(r_lt, r_active)={rs:.2f}   top-20 in comune tra lt e active={ov:.0%}")
    configs = [("no_seg", False, False), ("seg", True, False), ("seg+exp", True, True)]
    mats_by = {tag: {n: offline_matrices(*off[n], scaler, dict(fb_base, use_seg=us, use_exposure=ue), raws[n])
                     for n in names} for tag, us, ue in configs}
    rows = []
    for held in names:
        res = {}
        for tag, use_seg, use_exp in configs:
            mats = mats_by[tag]
            b = SegmentLinUCB(mats[held][3].p)
            for n in names:
                if n != held:
                    b.update(mats[n][0], mats[n][1])
            Xh, rh, push, _ = mats[held]
            for g in GROUPS:
                res[(tag, g)] = spearmanr(b.ucb(Xh[g], g)[0], rh[g]).correlation
                res[("push", g)] = spearmanr(push[g], rh[g]).correlation
        for g in GROUPS:
            log.info(f"  test={held:20s} {g:6s}  Spearman  solo spinta={res[('push', g)]:.3f}  "
                     f"senza gruppo={res[('no_seg', g)]:.3f}  con gruppo={res[('seg', g)]:.3f}  "
                     f"con esposizione={res[('seg+exp', g)]:.3f}")
            rows.append({"held_out": held, "group": g, **{k: res[(k, g)] for k in
                                                          ("push", "no_seg", "seg", "seg+exp")}})
    return rows


def build_offline_prior(off, fb_kwargs, scaler, bandit_kwargs, warm_weight, warm_neff, log, raws=None,
                        hurdle=False):
    """LinUCB con il prior offline. Il prior equivale a aver già visto gli esempi offline:
    A = lam I + w sum x x^T, b = w sum r x. warm_neff fissa w = neff / N_offline."""
    p = FeatureBuilder(**fb_kwargs).p
    bandit = SegmentLinUCB(p, **bandit_kwargs)
    raws = raws if raws is not None else offline_raws(off, log)
    mats = {n: offline_matrices(*off[n], scaler, fb_kwargs, raws[n], touched_only=hurdle) for n in off}
    # il peso per esempio è quello del prior senza filtro: con --hurdle un gruppo toccato
    # da pochi difetti (lt) ha semplicemente meno esempi equivalenti
    n_off = sum(len(off[n][2]) for n in off) if hurdle else sum(len(m[0]["all"]) for m in mats.values())
    w = (warm_neff / max(n_off, 1)) if warm_neff is not None else warm_weight
    if w > 0:
        for n, (X, r, _, _) in mats.items():
            bandit.update(X, r, weight=w)
    log.info(f"  Prior offline: {n_off} difetti x {len(GROUPS)} gruppi, peso {w:.4g} "
             f"(= {w * n_off:.0f} esempi equivalenti per gruppo)")
    if hurdle:
        log.info("  Prior hurdle: difetti che toccano il gruppo " + ", ".join(
            f"{g}={sum(len(m[1][g]) for m in mats.values())}" for g in GROUPS))
    return bandit


# ============================================================================ #
#  REPORT
# ============================================================================ #
def _write_rounds(rows, path):
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n", restval="")
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(v, 6) for k, v in r.items()})


def _save_reg(reg_log, reg_models, reg_dir):
    """Batch online nel formato dei pickle di compute_embeddings.py (+ round, kind),
    così si possono rianalizzare con gli stessi strumenti del regressore."""
    if reg_log:
        blob = {"X": np.vstack([r["X"] for r in reg_log]).astype(np.float32),
                "defect_id": np.array([r["defect_id"] for r in reg_log], np.int64),
                "source": np.array([r["source"] for r in reg_log], dtype=str),
                "round": np.array([r["round"] for r in reg_log], np.int64),
                "kind": np.array([r["kind"] for r in reg_log], dtype=str),
                "trained": np.array([r["trained"] for r in reg_log], np.int64)}
        for g in GROUPS:
            n = np.array([r[f"n_aff_{g}"] for r in reg_log], np.float32)
            loc = np.array([r[f"loc_{g}"] for r in reg_log], np.float32)
            blob[f"n_affected_{g}"] = n
            blob["impact" if g == "all" else f"impact_{g}"] = np.nan_to_num(loc) * n
            blob[f"glob_{g}"] = np.array([r[f"glob_{g}"] for r in reg_log], np.float32)
        with open(reg_dir / "online_batches.pkl", "wb") as f:
            pickle.dump(blob, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(reg_dir / "models_per_round.pkl", "wb") as f:
        pickle.dump(reg_models, f, protocol=pickle.HIGHEST_PROTOCOL)


def _v(row, k, nd=3):
    v = row.get(k)
    return "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{nd}f}"


def _log_round(log, row, cfg, has_reg):
    g = cfg.target
    tag = cfg.policy + (f"+pool-{cfg.pool}" if cfg.pool != "none" else "")
    L = [f"── round {row['round']} [{cfg.target}/{tag}/s{cfg.seed}] archi {row['edges_before']}→"
         f"{row['edges_after']} · candidati {row['n_cands']} · scelti {row['n_chosen']} "
         f"(rimossi {row['n_removed']}) · {row['seconds']}s"
         + (f" · toccano {g}: {row['n_touch']}" if cfg.only_touch and "n_touch" in row else ""),
         f"   reward {g}: round {_v(row, f'r_{g}_sum')} · cumulata {_v(row, f'cum_{g}')} · "
         f"cumulata congiunta {_v(row, f'cum_joint_{g}')} · "
         f"percentile vs sonda {_v(row, f'pct_{g}', 2)} · collaterale "
         + " ".join(f"{h}={_v(row, f'cum_{h}')}" for h in GROUPS if h != g)]
    if f"joint_{g}" in row:
        L.append(f"   effetto congiunto {g}: {_v(row, f'joint_{g}')} vs somma singoli {_v(row, f'sumind_{g}')}")
    if "theta_delta" in row:
        L.append(f"   MAB: Δθ {_v(row, 'theta_delta', 4)} · bonus/UCB {_v(row, 'bonus_frac', 2)} · "
                 "Spearman sonda " + " ".join(f"{h}={_v(row, f'sp_probe_{h}', 2)}" for h in GROUPS)
                 + (" · holdout " + " ".join(f"{h}={_v(row, f'sp_hold_{h}', 2)}" for h in GROUPS)
                    if f"sp_hold_{g}" in row else "")
                 + (" · holdout toccati " + " ".join(f"{h}={_v(row, f'sp_hold_t_{h}', 2)}" for h in GROUPS)
                    if f"sp_hold_t_{g}" in row else ""))
        L.append(f"   MAB diagnostica: ‖θ‖ {_v(row, 'theta_norm', 3)} · σ_{g} {_v(row, f'sigma_{g}', 3)} · "
                 f"calibrazione sonda {g}: prevista {_v(row, f'calib_pred_{g}', 3)} "
                 f"vs misurata {_v(row, f'calib_obs_{g}', 3)}")
    L.append("   liste: Jaccard " + " ".join(f"{a}/{b}={_v(row, f'jac_{a}_{b}', 2)}" for a, b in PAIRS))
    L.append(f"   stato {g} dopo la rimozione: " + " ".join(f"{m} {_v(row, f'{g}_{m}', 4)} ({row.get(f'{g}_d_{m}', 0):+.4f})"
                                         for m in ("entropy", "gini", "coverage", "novelty", "aplt", "ndcg")
                                         if np.isfinite(row.get(f"{g}_{m}", np.nan))))
    if has_reg:
        gg = g if f"reg_hold_{g}_sp" in row or f"reg_probe_{g}_sp" in row else "all"
        L.append(f"   regressore {gg}: holdout sp {_v(row, f'reg_hold_{gg}_sp', 3)} "
                 f"(congelato {_v(row, f'reg_hold_{gg}_sp_frozen', 3)}) · R² {_v(row, f'reg_hold_{gg}_r2', 3)} "
                 f"· sonda sp {_v(row, f'reg_probe_{gg}_sp', 3)} · label online {row.get('reg_n_online', 0)}")
    log.info("\n".join(L))


# ============================================================================ #
#  ARGOMENTI
# ============================================================================ #
def parse_args():
    p = argparse.ArgumentParser(description="MAB segment-aware sui difetti")
    p.add_argument("--offline-datasets", nargs="+", default=["Baby_Products", "Toys_and_Games", "Pet_Supplies"])
    p.add_argument("--online-dataset", default=None)
    p.add_argument("--impact", choices=IMPACTS, default="rr",
                   help="misura dell'impatto di un difetto sugli utenti affetti (reward del MAB e label "
                        "del regressore): rr = 1/rank del miglior item del difetto; tndcg = NDCG "
                        "sul ranking intero (non troncato), rilevanti = item del difetto")
    p.add_argument("--loo-reward-col", default=None,
                   help="colonna della reward nei CSV del LOO, con {g} al posto del gruppo. "
                        "Default: {g}_glob_d_prop_rr_best (rr) o {g}_glob_d_prop_tndcg (tndcg)")
    p.add_argument("--no-bandit-update", action="store_true")
    p.add_argument("--offline-check", action="store_true")
    p.add_argument("--targets", nargs="+", choices=GROUPS, default=["lt"])
    p.add_argument("--target", choices=GROUPS, default=None, help="alias di --targets con un solo gruppo")
    p.add_argument("--policies", nargs="+", choices=POL, default=["mab"])
    p.add_argument("--policy", choices=POL, default=None)
    p.add_argument("--pools", nargs="+", choices=["none", "regressor", "random"], default=None,
                   help="candidati che arrivano al MAB a ogni round: i più impattanti secondo il regressore "
                        "(regressor, DEFAULT se c'è il regressore), casuali (random) o tutti (none)")
    p.add_argument("--pool-mult", type=float, default=3.0,
                   help="dimensione del pool in archi = pool-mult x budget del round "
                        "(1 = solo quelli che stanno nel budget: il MAB non ha scelta)")
    p.add_argument("--pool-size", type=int, default=0, help="in alternativa: N difetti fissi nel pool")
    p.add_argument("--only-touching", action="store_true",
                   help="pool, scelta e sonda solo tra i difetti che toccano il target (exposure > 0)")
    p.add_argument("--hurdle", action="store_true",
                   help="LinUCB (e prior) per il gruppo g allenato solo sui difetti che toccano g")
    p.add_argument("--random-accept", action="store_true",
                   help="a ogni round rimuove un numero casuale (da 1 a tutti) dei difetti scelti, a caso")
    p.add_argument("--max-defects", type=int, default=0,
                   help="se > 0: tiene al massimo N difetti (stratificati per n. archi) oltre a quelli "
                        "delle sorgenti --keep-all-sources; il filtro è fatto prima delle feature")
    p.add_argument("--keep-all-sources", nargs="+", default=["gsdb_ff","rev2_f"],
                   help="sorgenti di difetti da tenere sempre tutte")
    p.add_argument("--feature-cache", default=None,
                   help="cartella dove salvare/riusare le feature online del grafo iniziale")
    p.add_argument("--fit-sample", type=int, default=0,
                   help="se > 0: standardizzazione stimata su N difetti a caso (seed fisso)")
    p.add_argument("--features-only", action="store_true",
                   help="calcola (e salva in --feature-cache) le feature online, poi esce")
    p.add_argument("--init-rounds", type=int, default=3,
                   help="mab_cold: round iniziali con difetti casuali prima di passare all'UCB")
    p.add_argument("--seeds", nargs="+", type=int, default=[2020])
    p.add_argument("--alpha", type=float, default=0.7)
    p.add_argument("--lam-shared", type=float, default=1.0)
    p.add_argument("--lam-seg", type=float, default=5.0)
    p.add_argument("--warm-weight", type=float, default=0.5, help="peso di ogni esempio offline (0 = da zero)")
    p.add_argument("--warm-neff", type=float, default=None,
                   help="in alternativa: il prior offline vale N esempi per gruppo (sovrascrive --warm-weight)")
    p.add_argument("--extra-cols", nargs="*", default=[])
    p.add_argument("--defect-files", nargs="+", default=["group_ff", "gsdb_ff", "rev2_f","rev2_ff", "rev2_fff"])
    p.add_argument("--defect-type", choices=["group_ff", "rev2_fff","group_f", "gsdb_ff", "rev2_f", "group_inj"], default=None,
                   help="usa SOLO i difetti di questo tipo (sovrascrive --defect-files); "
                        "le uscite vanno in una sottocartella _src-<tipo>")
    p.add_argument("--tag", default="", help="suffisso della cartella di output (ablation)")
    p.add_argument("--budget-cut", type=float, default=0.4,
                   help="se gli archi distinti del pool (grafo iniziale, holdout escluso) sono meno del "
                        "budget totale: budget totale = archi del pool x (1 - budget-cut). 0 = disattivato")
    p.add_argument("--use-exposure", action="store_true")
    p.add_argument("--no-seg-features", action="store_true", help="ablation: senza blocco gruppo")
    # regressore
    p.add_argument("--regressor", default=None, help="final_model.pkl di impact_regressor.py")
    p.add_argument("--regressor-active", default=None,
                   help="final_model.pkl allenato con --label rr_active (usato con --yhat seg)")
    p.add_argument("--regressor-lt", default=None,
                   help="final_model.pkl allenato con --label rr_lt (usato con --yhat seg)")
    p.add_argument("--emb-dir", default=None, help="pickle GraphSAGE dei difetti (<dataset>.pkl, std_stats.pkl)")
    p.add_argument("--sage-encoder", default=None, help="encoder GraphSAGE: ricalcola gli embedding a ogni round")
    p.add_argument("--train-csv", default=None, help="default: <data>/<dataset>/<dataset>.train.csv")
    p.add_argument("--yhat", choices=["none", "common", "seg"], default="common",
                   help="come entra y_hat nel contesto (seg = un regressore per gruppo)")
    p.add_argument("--reg-update", choices=["none", "ridge", "adam"], default="ridge")
    p.add_argument("--reg-online-weight", type=float, default=5.0)
    p.add_argument("--reg-train-on", choices=["chosen", "chosen+probe"], default="chosen")
    p.add_argument("--sage-every", type=int, default=1)
    # round
    p.add_argument("--budget-per-round", type=int, default=1000)
    p.add_argument("--total-budget", type=int, default=5000)
    p.add_argument("--rounds", type=int, default=1000)
    p.add_argument("--max-per-round", type=int, default=0, help="difetti massimi per round (0 = solo budget)")
    p.add_argument("--max-overlap", type=float, default=0.5)
    p.add_argument("--probe", type=int, default=20, help="difetti casuali misurati per round (valutazione)")
    p.add_argument("--probe-update", action="store_true", help="la sonda aggiorna anche il MAB")
    p.add_argument("--holdout", type=int, default=100, help="difetti misurati una volta al round 0, esclusi")
    p.add_argument("--no-joint", action="store_true")
    p.add_argument("--no-online-scale", action="store_true")
    p.add_argument("--reward-transform", choices=["none", "slog"], default="none",
                   help="trasformazione della reward usata dal LinUCB per imparare (prior e online): "
                        "slog = sign(r)*log(1+|r|). La reward riportata resta quella originale")
    p.add_argument("--patience", type=int, default=0,
                   help="se > 0: l'episodio finisce quando la cumulata congiunta del target non migliora "
                        "per N round di fila (vale per TUTTE le politiche; motivo in stop.json)")
    p.add_argument("--patience-rel", type=float, default=0.001,
                   help="miglioramento minimo perché un round conti, relativo al miglior valore (default 0.1%%)")
    p.add_argument("--mab-stop", action="store_true",
                   help="dal round 1 il MAB non è obbligato a scegliere: se nessun candidato ha UCB > 0 "
                        "l'episodio finisce (motivo in stop.json)")
    p.add_argument("--interactive", action="store_true")
    # ambiente
    p.add_argument("--loo-dir", default=None)
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--score-batch", type=int, default=512)
    p.add_argument("--test-file", default=None)
    p.add_argument("--head-frac", type=float, default=0.2)
    p.add_argument("--staging-dir", default=None)
    p.add_argument("--out", default=None, help="default: mab/results/<dataset>")
    return p.parse_args()
