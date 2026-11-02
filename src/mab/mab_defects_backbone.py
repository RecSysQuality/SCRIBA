#!/usr/bin/env python3
"""
mab_defects.py
==============
Selezione segment-aware dei difetti (stelle e biclique) con UN SOLO LinUCB,
sotto un budget di archi per round. Si appoggia a loo_defects.py (ambiente) e,
opzionalmente, a mab_regressor.py (regressore GraphSAGE aggiornato online).

CONTESTO x_{k,g}  (quello che il MAB sa PRIMA di scegliere, ricalcolato a ogni round
                   sul grafo attuale)
  blocco comune   forma del difetto, densità, popolarità degli item, quanto il
                  difetto "gonfia" i suoi item, grado dei suoi utenti
                  (+ colonne extra del file dei difetti, + y_hat_all se --yhat common)
  blocco gruppo   spinta del difetto sugli utenti del gruppo g (push, reach, intensity,
                  relative alla mediana del gruppo) (+ esposizione in top-K, + y_hat_g)
  => per ogni difetto ci sono TRE contesti, uno per gruppo, che condividono il
     blocco comune e differiscono nel blocco gruppo.

MODELLO  punteggio(k, g) = (theta_comune + theta_g) . x_{k,g}  +  bonus di incertezza
  Un LinUCB sul vettore  phi(x, g) = [x | 0 .. x (posto di g) .. 0].
  lam_seg > lam_shared: i gruppi partono uguali e si separano solo con evidenza.
  Ogni misura aggiorna tutti e tre i gruppi (la reward è misurata per tutti).

REWARD  r_g = sum_{u affetti in g} [rr_best_prima(u) - rr_best_dopo(u)] / |g|
  (= -{g}_glob_d_prop_rr_best del LOO), misurata dal LightGCN del dataset,
  difetto per difetto, sul grafo del round PRIMA della rimozione (semi-bandit).
  L'effetto congiunto del batch rimosso viene misurato a parte (joint_{g}).

OFFLINE  i CSV di loo_defects.py (variante prop) sui dataset passivi: stesse feature,
  stessa reward -> warm start del LinUCB (prior). --warm-neff fissa quanti
  "esempi equivalenti" vale il prior (consigliato: 100-1000).

ROUND t
  1. propagazione LightGCN sul grafo G_t, top-K di tutti gli utenti
  2. metriche di stato per segmento (entropia, gini, coverage, novelty, APLT, NDCG, ...)
  3. [regressore] embedding GraphSAGE su G_t -> y_hat_g nel contesto
  4. lista raccomandata per OGNI gruppo (si esegue quella del --target)
  5. misura della reward dei difetti scelti + di una SONDA casuale (valutazione non distorta)
  6. [regressore] valutazione prequenziale, poi aggiornamento con il batch
  7. aggiornamento del LinUCB, rimozione, effetto congiunto

Uso
---
  # controllo offline (leave-one-dataset-out)
  python mab_defects.py --offline-datasets A B C --offline-check

  # esperimento completo: 3 target x 4 politiche, report per il paper
  python mab_defects.py --offline-datasets A B C --online-dataset D \\
      --targets lt active all --policies mab push random \\
      --budget-per-round 300 --total-budget 6000 --max-per-round 15 \\
      --probe 20 --holdout 100 --warm-neff 300

  # MAB senza prior offline, che parte da difetti casuali (baseline di convergenza)
  python mab_defects.py ... --policies mab mab_cold --init-rounds 3

  # DEFAULT con il regressore: a ogni round al MAB arrivano i difetti più impattanti secondo il
  # regressore, fino a 3 x il budget del round (archi). Baseline: pool casuale della stessa taglia.
  python mab_defects.py ... --policies mab --pools regressor random --pool-mult 3

  # con il regressore ricalcolato e riallenato a ogni round
  python mab_defects.py ... --policies mab regressor push random \\
      --regressor scriba_regressor_out/final_model.pkl --emb-dir node_embeddings/defects \\
      --sage-encoder node_embeddings/weights/sage_shared.pt --yhat seg --reg-update ridge

Output (results/mab/<dataset>/)
  <target>_<policy>_s<seed>/
      recommendations.tsv   lista raccomandata a ogni round, per OGNI gruppo
      measurements.tsv      reward misurate (scelti, sonda, holdout), predizioni
      rounds.tsv/.jsonl     resoconto per round (stato, reward, convergenza, regressore)
      regressor/            batch di label online (formato compute_embeddings) + modelli
  REPORT.md + fig_*.png     (mab_report.py) tabelle e grafici per il paper
"""
from __future__ import annotations

import argparse
import copy
import csv
import glob
import hashlib
import json
import os
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.stats import norm, rankdata


def rank_norm(v):
    """y_hat -> scala N(0,1) usando solo l'ORDINE dei valori.
    Così la feature y_hat ha sempre la stessa scala, qualunque sia il regressore
    (pre-allenato, riallenato online, out-of-fold offline)."""
    v = np.asarray(v, float)
    if len(v) < 2:
        return np.zeros_like(v)
    return norm.ppf((rankdata(v) - 0.5) / len(v))

GROUPS: Tuple[str, ...] = ("all", "active", "lt")
PAIRS = (("lt", "active"), ("lt", "all"), ("active", "all"))
STATE_METRICS = ("entropy", "gini", "coverage", "novelty", "arp", "aplt",
                 "ndcg", "recall", "hit", "churn", "defect_share")
POL = ["mab", "mab_cold", "random", "push", "regressor", "oracle"]

def _loo():
    import sys
    ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(ROOT))

    """Import pigro: il nucleo (feature, bandit, metriche) non richiede torch/recbole."""
    import recbole_f.loo_defects as lo
    return lo


class _Log:
    def info(self, m):
        print(m, flush=True)


# ============================================================================ #
#  DIFETTI
# ============================================================================ #
@dataclass
class Defect:
    id: int                      # riga nel file dei difetti (= defect_id del LOO)
    users: np.ndarray            # ID interni RecBole
    items: np.ndarray
    extra: np.ndarray = field(default_factory=lambda: np.zeros(0))
    source: str = "group_ff"
    u_tok: Tuple[str, ...] = ()  # token originali (servono a GraphSAGE)
    i_tok: Tuple[str, ...] = ()
    yhat: Dict[str, float] = field(default_factory=dict)

    @property
    def key(self):
        return (self.source, self.id)


def read_defect_rows(path: Path) -> List[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _extra(row: dict, cols: Sequence[str]) -> np.ndarray:
    out = []
    for c in cols:
        try:
            out.append(float(row.get(c, "") or 0.0))
        except ValueError:
            out.append(0.0)
    return np.array(out, float)


def _tok(s) -> Tuple[str, ...]:
    return tuple(t for t in (s or "").split(",") if t)


def source_of_loo(path) -> str:
    return "gsdb_ff" if "gsdb" in Path(path).name else "group_ff"


# ============================================================================ #
#  FEATURE DAL GRAFO
# ============================================================================ #
def defect_features(R: sp.csr_matrix, Rc: sp.csc_matrix, deg_u, deg_i, d: Defect,
                    masks: Dict[str, np.ndarray], sizes: Dict[str, int], chunk: int = 4096):
    """
    Feature di un difetto sul grafo attuale. None se il difetto non ha più archi,
    altrimenti (common, seg: {g: (push, reach, intensity)}, keys degli archi, costo).
    push      = spinta totale sul gruppo / |gruppo|
    reach     = utenti del gruppo con spinta > 0 / |gruppo|
    intensity = spinta media per utente raggiunto
    Spinta: percorsi u -> j (item di u) -> f (utente del difetto) -> t (item del
    difetto, arco del difetto), peso 1/(deg u * deg j * deg f); max sugli item t.
    """
    M = R[d.users][:, d.items].tocsr()
    if M.nnz == 0:
        return None
    fm = np.diff(M.indptr) > 0
    F, M = d.users[fm], M[fm]
    im = np.asarray(M.sum(0)).ravel() > 0
    I, M = d.items[im], M[:, im].tocsr()
    nu, ni, ne = len(F), len(I), int(M.nnz)
    Mc = M.tocoo()
    keys = F[Mc.row].astype(np.int64) * R.shape[1] + I[Mc.col]

    icnt = np.asarray(M.sum(0)).ravel()
    dI, dF = np.maximum(deg_i[I], 1), np.maximum(deg_u[F], 1)
    common = [float(nu == 1 or ni == 1),
              np.log1p(nu), np.log1p(ni), np.log1p(ne),
              ne / (nu * ni),
              np.log1p(dI.min()),
              float((icnt / dI).max()),
              np.log1p(dF.mean())]

    RF = R[F].tocsr()
    J = np.unique(RF.indices)
    B = (RF[:, J].T @ sp.diags(1.0 / dF) @ M).toarray().astype(np.float32)   # [j, t]
    B *= (1.0 / np.maximum(deg_i[J], 1))[:, None].astype(np.float32)
    B[np.searchsorted(J, I), np.arange(ni)] = 0.0                            # niente u->t->f->t
    XJ = Rc[:, J].tocsr()
    rows = np.flatnonzero(np.diff(XJ.indptr))
    rows = rows[~np.isin(rows, F)]
    push = np.zeros(len(rows), np.float32)
    for a in range(0, len(rows), chunk):
        r = rows[a:a + chunk]
        P = XJ[r] @ B
        push[a:a + chunk] = np.asarray(P).max(axis=1) / np.maximum(deg_u[r], 1)
    seg = {}
    for g, m in masks.items():
        pg = push[m[rows]] if rows.size else push[:0]
        tot, n = float(pg.sum()), int((pg > 0).sum())
        seg[g] = (tot / sizes[g], n / sizes[g], tot / max(n, 1))
    return common, seg, keys, ne


class FeatureBuilder:
    """
    Contesto per gruppo = [intercetta | comune | blocco del gruppo].
    yhat_mode: None | "common" (y_hat_all nel blocco comune)
                    | "seg"    (y_hat_g nel blocco del gruppo g)
    Standardizzazione (mu, sd) e mediane del gruppo stimate UNA volta per grafo
    (offline: sui suoi difetti; online: sui candidati del round 0) e poi fisse,
    così le feature riflettono i cambiamenti del grafo round dopo round.
    """
    COMMON = ["is_star", "log_n_users", "log_n_items", "log_n_edges", "density",
              "log_min_item_deg", "max_item_inflation", "log_mean_user_deg"]
    SEG = ["push_rel", "log_push", "log_reach", "log_intensity"]
    EXP = ["exposure_rel", "log_exposure"]

    def __init__(self, extra_cols: Sequence[str] = (), use_seg: bool = True,
                 use_exposure: bool = False, yhat_mode: Optional[str] = None):
        self.extra_cols, self.use_seg, self.use_exposure = list(extra_cols), use_seg, use_exposure
        self.yhat_mode = yhat_mode
        self.med: Dict[str, np.ndarray] = {}
        self.med_exp: Dict[str, float] = {}
        self.mu = self.sd = None

    @property
    def names(self) -> List[str]:
        return (["intercept"] + self.COMMON + self.extra_cols
                + (["y_hat"] if self.yhat_mode == "common" else [])
                + (self.SEG if self.use_seg else [])
                + (["y_hat_seg"] if self.yhat_mode == "seg" else [])
                + (self.EXP if self.use_exposure else []))

    @property
    def p(self) -> int:
        return len(self.names)

    @staticmethod
    def raw(R, defects, masks, log=None, tag=""):
        """Feature grezze di tutti i difetti (la parte costosa). Non dipendono dalla
        configurazione (use_seg, use_exposure, yhat_mode): si calcolano UNA volta per
        grafo e si riusano (offline: controllo e prior; online: ogni round)."""
        Rc = R.tocsc()
        deg_u = np.asarray(R.sum(1)).ravel()
        deg_i = np.asarray(R.sum(0)).ravel()
        sizes = {g: max(int(m.sum()), 1) for g, m in masks.items()}
        C, S, Y, keys, costs, ok = [], {g: [] for g in masks}, {g: [] for g in masks}, [], [], []
        t0 = t_last = time.time()
        for k, d in enumerate(defects):
            f = defect_features(R, Rc, deg_u, deg_i, d, masks, sizes)
            if log is not None and time.time() - t_last > 60:
                t_last = time.time()
                el = t_last - t0
                log.info(f"  [feature {tag}] {k + 1}/{len(defects)} difetti · {el:.0f}s "
                         f"(restano ~{el / (k + 1) * (len(defects) - k - 1):.0f}s)")
            if f is None:
                continue
            common, seg, ky, ne = f
            C.append(common + list(d.extra))
            for g in masks:
                S[g].append(seg[g])
                Y[g].append(float(d.yhat.get(g, d.yhat.get("all", 0.0))))
            keys.append(ky)
            costs.append(ne)
            ok.append(k)
        nc = len(FeatureBuilder.COMMON) + (len(defects[0].extra) if defects else 0)
        if log is not None and tag:
            log.info(f"  [feature {tag}] {len(ok)} difetti in {time.time() - t0:.0f}s")
        return {"C": np.array(C, float).reshape(len(ok), nc),
                "S": {g: np.array(v, float).reshape(len(ok), 3) for g, v in S.items()},
                "Y": {g: np.array(v, float) for g, v in Y.items()},
                "keys": keys, "costs": np.array(costs, dtype=np.int64), "ok": np.array(ok, dtype=int)}

    def _raw(self, R, defects, masks, raw=None):
        r = raw if raw is not None else self.raw(R, defects, masks)
        C = r["C"]
        if self.yhat_mode == "common":
            C = np.column_stack([C, r["Y"]["all"]])
        return C, r["S"], r["Y"], r["keys"], r["costs"], r["ok"]

    def _assemble(self, C, Sg, g, yg=None, eg=None):
        parts = [C]
        if self.use_seg:
            rel = Sg / self.med[g]
            parts.append(np.column_stack([rel[:, 0], np.log1p(rel[:, 0]),
                                          np.log1p(rel[:, 1]), np.log1p(rel[:, 2])]))
        if self.yhat_mode == "seg":
            parts.append(np.asarray(yg, float)[:, None])
        if self.use_exposure:
            rel = np.asarray(eg, float) / self.med_exp[g]
            parts.append(np.column_stack([rel, np.log1p(rel)]))
        return np.hstack(parts)

    def fit(self, R, defects, masks, exposure=None, raw=None) -> "FeatureBuilder":
        C, S, Y, _, _, ok = self._raw(R, defects, masks, raw)
        for g, Sg in S.items():
            self.med[g] = np.array([np.median(c[c > 0]) if (c > 0).any() else 1.0 for c in Sg.T])
        E = {}
        if self.use_exposure:
            if exposure is None:
                raise ValueError("use_exposure=True richiede exposure in fit()")
            for g in masks:
                e = np.asarray(exposure[g])[ok]
                self.med_exp[g] = float(np.median(e[e > 0])) if (e > 0).any() else 1.0
                E[g] = e
        X = np.vstack([self._assemble(C, S[g], g, Y[g], E.get(g)) for g in S])
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-8
        return self

    def transform(self, R, defects, masks, exposure=None, raw=None):
        """X ({g: (n, p)}), keys, costs, ok (indici in defects), push ({g: spinta grezza})."""
        C, S, Y, keys, costs, ok = self._raw(R, defects, masks, raw)
        X = {}
        for g in S:
            eg = np.asarray(exposure[g])[ok] if self.use_exposure else None
            Z = np.clip((self._assemble(C, S[g], g, Y[g], eg) - self.mu) / self.sd, -6.0, 6.0)
            X[g] = np.hstack([np.ones((len(Z), 1)), Z])
        return X, keys, costs, ok, {g: S[g][:, 0] for g in S}


# ============================================================================ #
#  LinUCB: pesi condivisi + correzione per gruppo
# ============================================================================ #
def _sm(A_inv, z):
    Az = A_inv @ z
    return A_inv - np.outer(Az, Az) / (1.0 + z @ Az)


def _edge_index(keys):
    sizes = np.array([len(k) for k in keys], dtype=np.int64)
    all_keys = np.concatenate(keys) if len(keys) else np.zeros(0, np.int64)
    owner = np.repeat(np.arange(len(keys)), sizes)
    return all_keys, owner


class _Budget:
    """Archi effettivi (non già presi in questo round) di ogni candidato, aggiornati
    in modo incrementale: un np.isin su tutti gli archi per ogni scelta."""

    def __init__(self, keys, costs):
        self.keys = keys
        self.all_keys, self.owner = _edge_index(keys)
        self.taken = np.zeros(len(self.all_keys), bool)
        self.eff = np.asarray(costs, float).copy()
        self.costs = np.asarray(costs, float)

    def take(self, j):
        newly = np.isin(self.all_keys, self.keys[j]) & ~self.taken
        self.taken |= newly
        np.subtract.at(self.eff, self.owner[newly], 1.0)

    def feasible(self, alive, spent, budget, max_overlap):
        return alive & (self.eff > 0) & (self.eff >= (1 - max_overlap) * self.costs) \
            & (self.eff <= budget - spent)


class SegmentLinUCB:
    """
    punteggio(k, g) = (w_comuni + w_g) . x_{k,g}  +  alpha * sigma_g * incertezza
    Un unico LinUCB sul vettore  phi(x, g) = [x | 0 .. x (posto di g) .. 0].

    reward_transform: trasformazione applicata alla reward SOLO per imparare
      "none" = reward così com'è
      "slog" = sign(r) * log(1 + |r|): comprime la coda (un difetto da 500 mediane pesa ~6),
               conserva segno e ordine. Media, bonus e UCB sono allora in unità slog.
      La reward misurata e riportata nei log/TSV resta quella originale.
    """

    TRANSFORMS = ("none", "slog")

    def __init__(self, p: int, groups: Sequence[str] = GROUPS, lam_shared: float = 1.0,
                 lam_seg: float = 5.0, alpha: float = 0.7, sigma0: float = 1.0, n0: float = 5.0,
                 reward_transform: str = "none"):
        if reward_transform not in self.TRANSFORMS:
            raise ValueError(f"reward_transform sconosciuta: {reward_transform}")
        self.reward_transform = reward_transform
        self.G, self.p = list(groups), p
        prec = np.r_[np.full(p, lam_shared), np.full(p * len(self.G), lam_seg)]
        self.A_inv = np.diag(1.0 / prec)
        self.b = np.zeros(len(prec))
        self.alpha, self.sigma0, self.n0 = alpha, sigma0, n0
        self.sse = {g: 0.0 for g in self.G}
        self.n = {g: 0.0 for g in self.G}

    def phi(self, X, g):
        X = np.atleast_2d(X)
        out = np.zeros((len(X), self.p * (1 + len(self.G))))
        j = self.G.index(g)
        out[:, : self.p] = X
        out[:, self.p * (1 + j): self.p * (2 + j)] = X
        return out

    @property
    def theta(self):
        return self.A_inv @ self.b

    def sigma(self, g):
        return float(np.sqrt((self.n0 * self.sigma0 ** 2 + self.sse[g]) / (self.n0 + self.n[g])))

    def ucb(self, X, g):
        F = self.phi(X, g)
        mean = F @ self.theta
        bonus = self.alpha * self.sigma(g) * np.sqrt(
            np.einsum("ij,jk,ik->i", F, self.A_inv, F).clip(min=0))
        return mean, bonus, mean + bonus

    def tr(self, r):
        """Reward nelle unità in cui il modello impara (vedi reward_transform)."""
        r = np.asarray(r, float)
        if getattr(self, "reward_transform", "none") == "slog":
            return np.sign(r) * np.log1p(np.abs(r))
        return r

    def update(self, X_by_g, r_by_g, weight: float = 1.0):
        r_by_g = {g: self.tr(np.atleast_1d(r_by_g[g])) for g in X_by_g}
        th = self.theta
        for g in X_by_g:
            F, r = self.phi(X_by_g[g], g), np.atleast_1d(r_by_g[g])
            res = np.clip(r - F @ th, -5.0, 5.0)
            self.sse[g] += weight * float((res ** 2).sum())
            self.n[g] += weight * len(r)
        # tutte le osservazioni insieme: A <- A + w F^T F, b <- b + w F^T r
        # (identico a una sequenza di Sherman-Morrison, ma in un solo passo)
        Fs = [self.phi(X_by_g[g], g) for g in X_by_g if len(np.atleast_1d(r_by_g[g]))]
        rs = [np.atleast_1d(np.asarray(r_by_g[g], float)) for g in X_by_g if len(np.atleast_1d(r_by_g[g]))]
        if not Fs:
            return
        F, r = np.vstack(Fs), np.concatenate(rs)
        A = np.linalg.inv(self.A_inv) + weight * F.T @ F
        A_inv = np.linalg.inv(A)
        self.A_inv = (A_inv + A_inv.T) / 2
        self.b += weight * F.T @ r

    def select(self, X, g, keys, costs, budget, max_overlap=0.5, max_k=0, allowed=None,
               force_one=True):
        """
        Greedy su UCB / costo effettivo. Dopo ogni scelta l'incertezza viene
        ridotta come se quel difetto fosse già stato osservato ("hallucinated
        update"): la MEDIA resta quella del posterior vero, cambia solo il bonus,
        così il batch non sceglie k difetti quasi identici.
        Ritorna (scelti, media, bonus) — media e bonus prima di qualsiasi scelta.
        force_one: se nessun candidato ha UCB > 0 sceglie comunque il meno negativo
        (comportamento storico). Con force_one=False restituisce una lista vuota:
        il MAB è "saturo". In self.last_stop resta il motivo per cui si è fermato:
        "saturo" (candidati ammissibili ma tutti con UCB <= 0), "budget", "max_k".
        """
        n = len(costs)
        F = self.phi(X, g)
        mean = F @ self.theta
        A = self.A_inv.copy()
        var = np.einsum("ij,jk,ik->i", F, A, F).clip(min=0)
        s = self.alpha * self.sigma(g)
        bonus0 = s * np.sqrt(var)
        bud = _Budget(keys, costs)
        alive = np.ones(n, bool) if allowed is None else np.asarray(allowed, bool).copy()
        chosen, spent = [], 0.0
        self.last_stop = "budget"
        while alive.any():
            if max_k and len(chosen) >= max_k:
                self.last_stop = "max_k"
                break
            alive = bud.feasible(alive, spent, budget, max_overlap)
            if not alive.any():
                self.last_stop = "budget"
                break
            score = mean + s * np.sqrt(var)

            pos = alive & (score > 0)
            if pos.any():
                ratio = np.where(pos, score / np.maximum(bud.eff, 1), -np.inf)
            elif not chosen and force_one:      # almeno un difetto per round
                ratio = np.where(alive, score, -np.inf)
            else:
                self.last_stop = "saturo"
                break
            j = int(np.argmax(ratio))
            chosen.append(j)
            spent += bud.eff[j]
            alive[j] = False
            bud.take(j)
            f = F[j]
            v = A @ f
            den = 1.0 + f @ v
            c = F @ v
            var = (var - c * c / den).clip(min=0)
            A -= np.outer(v, v) / den
        return chosen, mean, bonus0

    def explain(self, names):
        th = self.theta
        out = {"comuni": dict(zip(names, np.round(th[: self.p], 4)))}
        for j, g in enumerate(self.G):
            out[f"correzione_{g}"] = dict(zip(names, np.round(th[self.p * (1 + j): self.p * (2 + j)], 4)))
        return out


def greedy_fill(order, keys, costs, budget, max_overlap=0.5, max_k=0, allowed=None) -> List[int]:
    """Riempimento del budget seguendo un ordine dato (baseline)."""
    bud = _Budget(keys, costs)
    chosen, spent = [], 0.0
    alive = np.ones(len(costs), bool)
    for j in order:
        if max_k and len(chosen) >= max_k:
            break
        if allowed is not None and not allowed[j]:
            continue
        e = bud.eff[j]
        if e <= 0 or e < (1 - max_overlap) * bud.costs[j] or spent + e > budget:
            continue
        chosen.append(int(j))
        spent += e
        alive[j] = False
        bud.take(j)
    return chosen


class RewardScaler:
    """Divide la reward di ogni gruppo per la mediana di |r| > 0."""

    def fit(self, rewards: Dict[str, np.ndarray]) -> "RewardScaler":
        self.scale = {}
        for g, r in rewards.items():
            a = np.abs(np.asarray(r, float))
            a = a[np.isfinite(a)]
            self.scale[g] = float(np.median(a[a > 0])) if (a > 0).any() else 1.0
        self.source = "offline"
        return self

    def fit_online(self, rewards: Dict[str, np.ndarray], min_nonzero=5):
        """Riscala sul dataset nuovo (round 0): la reward glob dipende da |g|,
        quindi la scala offline non è quella del dataset nuovo."""
        for g, r in rewards.items():
            a = np.abs(np.asarray(r, float))
            a = a[np.isfinite(a) & (a > 0)]
            if len(a) >= min_nonzero:
                self.scale[g] = float(np.median(a))
        self.source = "online"

    def __call__(self, g, r):
        return np.asarray(r, float) / self.scale[g]


# ============================================================================ #
#  METRICHE DI STATO (per segmento, sulle top-K di TUTTI gli utenti del gruppo)
# ============================================================================ #
def gini_from_counts(counts) -> float:
    """Stessa formula di loo_defects.gini_from_counts (su item raccomandati)."""
    c = np.sort(np.asarray(counts, float))
    n = len(c)
    if n <= 1 or c.sum() == 0:
        return 0.0
    c /= c.sum()
    g = 2 * np.sum(np.arange(1, n + 1) * c) / n - (n + 1) / n
    return float(min(max(g, 0.0), 1.0))


def segment_metrics(top, masks, n_items, pop0, tail, def_items, test_R=None, ref_top=None,
                    chunk=65536) -> Dict[str, Dict[str, float]]:
    """
    top: (n_users-1, K), riga r = utente r+1.
      entropy, gini, coverage  distribuzione degli item raccomandati al gruppo
      novelty    media di -log2(pop0(i)/n_utenti)   (pop0 = grado nel training ORIGINALE)
      arp        popolarità media raccomandata (pop0)
      aplt       quota di slot occupati da item long-tail (fuori dalla testa)
      ndcg, recall, hit   @K sul test split (utenti del gruppo con almeno un item di test)
      churn      1 - Jaccard(top-K attuale, top-K del round 0), media sugli utenti
      defect_share  quota di slot occupati da item di difetti candidati
    """
    K = top.shape[1]
    catalog = n_items - 1
    n_u = top.shape[0]
    selfinfo = -np.log2(np.maximum(pop0, 1) / max(n_u, 1))
    disc = 1.0 / np.log2(np.arange(2, K + 2))
    idcg = np.cumsum(disc)
    out = {}
    for g, m in masks.items():
        rows = np.flatnonzero(m[1:n_u + 1])
        res = {k: np.nan for k in STATE_METRICS}
        if not rows.size:
            out[g] = res
            continue
        T = top[rows]
        cnt = np.bincount(T.ravel(), minlength=n_items)
        nz = cnt[cnt > 0]
        p = nz / nz.sum()
        res.update(entropy=float(-(p * np.log2(p)).sum()), gini=gini_from_counts(nz),
                   coverage=len(nz) / catalog, novelty=float(selfinfo[T].mean()),
                   arp=float(pop0[T].mean()), aplt=float(tail[T].mean()),
                   defect_share=float(def_items[T].mean()))
        if ref_top is not None:
            inter = 0.0
            for a in range(0, len(rows), chunk):
                A, B = T[a:a + chunk], ref_top[rows[a:a + chunk]]
                n_in = (A[:, :, None] == B[:, None, :]).any(2).sum(1)
                inter += (1 - n_in / (2 * K - n_in)).sum()
            res["churn"] = float(inter / len(rows))
        if test_R is not None:
            users = rows + 1
            ntest = np.diff(test_R.indptr)[users]
            v = ntest > 0
            if v.any():
                uu, TT, nt = users[v], T[v], ntest[v]
                nd = rc = ht = 0.0
                for a in range(0, len(uu), chunk):
                    sub = test_R[uu[a:a + chunk]]
                    Tc = TT[a:a + chunk]
                    hits = np.column_stack([np.asarray(sub[np.arange(len(Tc)), Tc[:, k]]).ravel()
                                            for k in range(K)]) > 0
                    dcg = (hits * disc).sum(1)
                    nd += (dcg / idcg[np.minimum(nt[a:a + chunk], K) - 1]).sum()
                    rc += (hits.sum(1) / nt[a:a + chunk]).sum()
                    ht += hits.any(1).sum()
                res.update(ndcg=nd / len(uu), recall=rc / len(uu), hit=ht / len(uu))
        out[g] = res
    return out


def structural_metrics(R, R0, masks) -> Dict[str, float]:
    deg_i = np.asarray(R.sum(0)).ravel()[1:]
    deg_u, deg_u0 = np.asarray(R.sum(1)).ravel(), np.asarray(R0.sum(1)).ravel()
    out = {"n_edges": int(R.nnz), "edges_frac": R.nnz / max(R0.nnz, 1),
           "item_deg_gini": gini_from_counts(deg_i[deg_i > 0])}
    p = deg_i[deg_i > 0] / deg_i.sum()
    out["item_deg_entropy"] = float(-(p * np.log2(p)).sum())
    for g, m in masks.items():
        out[f"{g}_edges_frac"] = float(deg_u[m].sum() / max(deg_u0[m].sum(), 1))
    return out


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


def load_offline(ds_name, staging_root, loo_dir, extra_cols):
    lo = _loo()
    cfg, graph, model, uid, iid, masks = load_graph(ds_name, staging_root)
    del model
    rows_by_key = {}
    for path in sorted(glob.glob(str(Path(loo_dir) / f"{ds_name}_loo*.csv"))):
        s = source_of_loo(path)
        for row in read_defect_rows(Path(path)):
            rows_by_key[(s, int(row["defect_id"]))] = row
    if not rows_by_key:
        raise FileNotFoundError(f"Nessun CSV LOO per {ds_name} in {loo_dir}")
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
            v = row.get(f"{g}_glob_d_prop_rr_best", "")
            rew[g].append(-float(v) if v not in ("", None) else 0.0)
            na = row.get(f"n_affected_{g}", "")
            n_aff[g].append(float(na) if na not in ("", None) else 0.0)
    rew = {g: np.array(v) for g, v in rew.items()}
    exposure = {g: np.array(n_aff[g]) / sizes[g] for g in GROUPS}
    per = {s: sum(d.source == s for d in defects) for s in sorted({d.source for d in defects})}
    lo.log.info(f"  [offline {ds_name}] {len(defects)} difetti con reward LOO {per}")
    return graph.R, masks, defects, rew, exposure


def subsample_defects(defects, R, max_n, keep_sources=("gsdb_f",), seed=12345, log=None):
    """Riduce i candidati PRIMA di qualunque calcolo di feature.
    - i difetti delle sorgenti in keep_sources si tengono tutti;
    - degli altri si tengono max_n, stratificati per numero di archi sul grafo iniziale
      (fasce di mezza ottava: 1, 2, 3, 4-5, 6-7, 8-11, ...). Ogni fascia riceve la stessa quota;
      le fasce con meno difetti della quota si prendono intere e il resto va alle altre.
      Così compaiono tutte le taglie, anche quelle rare.
    Il campione dipende solo da seed e file dei difetti: è lo stesso per tutti i job e i seed."""
    keep = [d for d in defects if d.source in keep_sources]
    rest = [d for d in defects if d.source not in keep_sources]
    if not max_n or len(rest) <= max_n:
        return defects
    t0 = time.time()
    cost = np.array([R[d.users][:, d.items].nnz for d in rest], dtype=np.int64)
    n_rest = len(rest)
    nz = np.flatnonzero(cost > 0)                     # senza archi: mai selezionabili
    rest, cost = [rest[k] for k in nz], cost[nz]
    rng = np.random.default_rng(seed)
    bins = np.floor(2 * np.log2(cost)).astype(np.int64)
    ub, inv, cnt = np.unique(bins, return_inverse=True, return_counts=True)
    quota, left = np.zeros(len(ub), np.int64), min(max_n, len(rest))
    open_ = quota < cnt
    while left > 0 and open_.any():
        q = left // int(open_.sum())
        if q == 0:                                     # resto: 1 in più a fasce aperte a caso
            idx = rng.permutation(np.flatnonzero(open_))[:left]
            quota[idx] += 1
            break
        add = np.where(open_, np.minimum(q, cnt - quota), 0)
        quota += add
        left -= int(add.sum())
        open_ = quota < cnt
    pick = np.concatenate([rng.choice(np.flatnonzero(inv == b), quota[b], replace=False)
                           for b in range(len(ub)) if quota[b] > 0])
    pick.sort()
    out = keep + [rest[k] for k in pick]
    if log is not None:
        lo_ = np.ceil(2.0 ** (ub / 2) - 1e-9).astype(int)          # fascia b: 2^(b/2) <= archi < 2^((b+1)/2)
        hi_ = np.ceil(2.0 ** ((ub + 1) / 2) - 1e-9).astype(int) - 1
        log.info(f"  Sottocampione difetti: {len(keep)} {'/'.join(keep_sources)} tenuti tutti + "
                 f"{len(pick)} altri su {len(nz)} con archi ({n_rest - len(nz)} senza archi scartati) "
                 f"· {time.time() - t0:.0f}s")
        log.info("  archi: " + " · ".join(f"{a}-{b}: {q}/{c}" if a != b else f"{a}: {q}/{c}"
                                          for a, b, q, c in zip(lo_, hi_, quota, cnt)))
    return out


def pool_edges(R, defects) -> int:
    """Archi DISTINTI coperti dai difetti sul grafo R. I difetti si sovrappongono,
    quindi è l'unione degli archi e non la somma dei costi."""
    keys = []
    for d in defects:
        sub = R[d.users][:, d.items].tocoo()
        if sub.nnz:
            keys.append(d.users[sub.row].astype(np.int64) * R.shape[1] + d.items[sub.col])
    return int(np.unique(np.concatenate(keys)).size) if keys else 0


def fit_online_features(env, defects, fb_kwargs, use_exposure, ds, cache_dir=None, fit_sample=0, log=None):
    """FeatureBuilder adattato al grafo iniziale del dataset online.
    La parte lenta (feature grezze + esposizione di tutti i difetti sul grafo iniziale) NON dipende
    dalla configurazione del job: con cache_dir si calcola una volta e si riusa (risultati identici).
    Con fit_sample > 0, mu/sd/mediane si stimano su un campione fisso di difetti.
    y_hat NON va in cache (dipende dal regressore): si ricalcola sempre dai difetti."""
    fit_defs = defects
    if fit_sample and len(defects) > fit_sample:
        idx = np.sort(np.random.default_rng(12345).choice(len(defects), fit_sample, replace=False))
        fit_defs = [defects[i] for i in idx]
        if log is not None:
            log.info(f"  [feature online {ds}] standardizzazione su un campione di {fit_sample} difetti")
    cache = None
    if cache_dir:
        h = hashlib.md5(np.ascontiguousarray(env.top_all).tobytes()).hexdigest()[:10]
        key = f"{ds}_n{len(defects)}_nnz{env.R.nnz}_K{env.K}_top{h}" + (f"_fs{fit_sample}" if fit_defs is not defects else "")
        cache = Path(cache_dir) / f"features_{key}.pkl"
    if cache is not None and not cache.exists():
        # stesse condizioni ma top-K iniziali con hash diverso (non determinismo della GPU su
        # punteggi quasi pari): la cache resta valida, si usa quella
        fs = f"_fs{fit_sample}" if fit_defs is not defects else ""
        alt = sorted(Path(cache_dir).glob(f"features_{ds}_n{len(defects)}_nnz{env.R.nnz}_K{env.K}_top*{fs}.pkl"))
        alt = [a for a in alt if fs or "_fs" not in a.name]
        if alt:
            if log is not None:
                log.info(f"  [feature online {ds}] top-K iniziali con hash diverso: uso {alt[0].name}")
            cache = alt[0]
    if cache is not None and cache.exists():
        t0 = time.time()
        with open(cache, "rb") as f:
            blob = pickle.load(f)
        raw0, exposure0 = blob["raw"], blob["exposure"]
        if log is not None:
            log.info(f"  [feature online {ds}] caricate dalla cache {cache.name} in {time.time() - t0:.0f}s")
    else:
        exposure0 = env.exposure(fit_defs) if (use_exposure or cache is not None) else None
        raw0 = FeatureBuilder.raw(env.R, fit_defs, env.masks, log, f"online {ds}")
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache.with_suffix(f".tmp{os.getpid()}")
            with open(tmp, "wb") as f:           # scrittura atomica: job in parallelo non si rompono
                pickle.dump({"raw": {k: v for k, v in raw0.items() if k not in ("keys", "Y")},
                             "exposure": exposure0}, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, cache)
            if log is not None:
                log.info(f"  [feature online {ds}] salvate in {cache}")
    ok = raw0["ok"]
    Y = {g: np.array([float(fit_defs[k].yhat.get(g, fit_defs[k].yhat.get("all", 0.0))) for k in ok], float)
         for g in raw0["S"]}
    raw0 = dict(raw0, Y=Y, keys=raw0.get("keys"))
    return FeatureBuilder(**fb_kwargs).fit(env.R, fit_defs, env.masks,
                                           exposure0 if use_exposure else None, raw0)


def offline_raws(off, log=None):
    """Feature grezze di ogni dataset offline, UNA volta sola."""
    return {n: FeatureBuilder.raw(off[n][0], off[n][2], off[n][1], log, f"offline {n}") for n in off}


def offline_matrices(R, masks, defects, rew, exposure, scaler, fb_kwargs, raw=None, touched_only=False):
    if raw is None:
        raw = FeatureBuilder.raw(R, defects, masks)
    fb = FeatureBuilder(**fb_kwargs).fit(R, defects, masks, exposure, raw)
    X, _, _, ok, push = fb.transform(R, defects, masks, exposure, raw)
    r = {g: scaler(g, rew[g][ok]) for g in GROUPS}
    if touched_only:                     # hurdle: per il gruppo g solo i difetti che toccano g
        for g in GROUPS:
            m = np.asarray(exposure[g])[ok] > 0
            X[g], r[g] = X[g][m], r[g][m]
    return X, r, push, fb


# ============================================================================ #
#  AMBIENTE: il LightGCN del dataset, stessa misura del LOO (prop)
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
    Le sottoclassi implementano _propagate(R), _topk(ue, ie, users), _rr_best(ue, ie, users, items).
    """

    def _setup(self, R, masks, K, uid_tok, iid_tok, test_R=None, head_frac=0.2):
        import warnings
        warnings.filterwarnings("ignore", category=sp.SparseEfficiencyWarning)
        self.R0 = R.copy()
        self.R = R.copy()
        self.hist = R.copy()                     # item visti: training originale, come nel LOO
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

    def _measure(self, users_rm, items, R_mod):
        has = np.isin(self.top_all, items).any(axis=1)
        affected = np.setdiff1d(self.all_users[has], users_rm, assume_unique=True)
        info = {"glob": {g: 0.0 for g in GROUPS}, "loc": {g: np.nan for g in GROUPS},
                "n_aff": {g: 0 for g in GROUPS}, "n_aff_total": int(affected.size)}
        if affected.size == 0:
            return info
        rr0 = self._rr_best(self.ue, self.ie, affected, items)
        ue1, ie1 = self._propagate(R_mod)
        rr1 = self._rr_best(ue1, ie1, affected, items)
        diff = rr0 - rr1
        for g in GROUPS:
            m = self.masks[g][affected]
            info["glob"][g] = float(diff[m].sum() / self.sizes[g])
            info["loc"][g] = float(diff[m].mean()) if m.any() else np.nan
            info["n_aff"][g] = int(m.sum())
        return info

    def reward(self, d: Defect):
        """Per gruppo: glob (= reward del MAB, -glob_d_prop_rr_best),
        loc (= label del regressore, -loc_d_prop_rr_best), n_aff."""
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

    def __init__(self, ds_name, staging_root, K, score_batch, test_file=None, head_frac=0.2, log=None):
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
        self._setup(self.graph.R, masks, K, inv_u, inv_i, test_R, head_frac)

    def _propagate(self, R):
        return self.lo.propagate(self.model, self.graph.norm_adj(R))

    def _topk(self, ue, ie, users):
        return self.lo.topk_and_ranks(ue, ie, users, self.K, self.hist, batch=self.batch)[0]

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


# ============================================================================ #
#  CONTROLLO OFFLINE (leave-one-dataset-out)
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


# ============================================================================ #
#  EPISODIO ONLINE
# ============================================================================ #
@dataclass
class RunConfig:
    target: str
    policy: str
    seed: int
    budget_per_round: int
    total_budget: int
    rounds: int
    max_per_round: int = 0
    max_overlap: float = 0.5
    probe: int = 20
    probe_update: bool = False
    interactive: bool = False
    joint_effect: bool = True
    online_scale: bool = True
    reg_train_on: str = "chosen"
    sage_every: int = 1
    init_rounds: int = 3
    pool: str = "none"          # none | regressor | random
    pool_size: int = 0          # se > 0: N difetti; altrimenti pool_mult x budget del round (archi)
    pool_mult: float = 3.0
    only_touch: bool = False    # candidati = solo difetti che toccano il target (exposure_target > 0)
    hurdle: bool = False        # theta_g aggiornato solo con difetti che toccano g: E[r_g | tocca g]
    random_accept: bool = False # utente simulato: rimuove un numero casuale (1..tutti) dei difetti scelti
    mab_stop: bool = False      # dal round 1: se nessun candidato ha UCB > 0 l'episodio finisce
    patience: int = 0           # > 0: stop se la cumulata congiunta del target non migliora per N round
    patience_rel: float = 0.001 # miglioramento minimo, relativo al miglior valore raggiunto (0.001 = 0.1%)
    bandit_update: bool = True


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


def _fmt(v, nd=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return ""
    if isinstance(v, (float, np.floating)):
        return f"{v:.{nd}g}" if abs(v) < 1e-3 and v != 0 else round(float(v), nd + 3)
    return v


def _jac(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else np.nan


def _sp(a, b):
    from mab_regressor import spearman
    return spearman(a, b)


def run_episode(env, defects, fb, bandit0, scaler0, cfg: RunConfig, out_dir: Path, log,
                reg0=None, embedder=None, holdout: Optional[Holdout] = None, bandit_cold=None):
    """Un esperimento (target, politica, seed) dal grafo iniziale. Ritorna le righe per round."""
    out_dir.mkdir(parents=True, exist_ok=True)
    env.reset()
    rng = np.random.default_rng(cfg.seed)
    acc_rng = np.random.default_rng(cfg.seed + 7777)   # RNG separato: non altera pool e sonda
    # mab      = LinUCB con il prior offline
    # mab_cold = LinUCB SENZA prior, che nei primi cfg.init_rounds round sceglie a caso
    #            (e impara da quei difetti), poi passa all'UCB
    bandit = copy.deepcopy(bandit0) if cfg.policy == "mab" else \
        copy.deepcopy(bandit_cold) if cfg.policy == "mab_cold" else None
    scaler = copy.deepcopy(scaler0)
    reg = copy.deepcopy(reg0) if reg0 is not None else None
    tgt = cfg.target
    hold_keys = holdout.keys if holdout else set()
    pool = [d for d in defects if d.key not in hold_keys]
    for d in pool:
        d.yhat = {}
    pool_pos = {d.key: k for k, d in enumerate(pool)}
    if cfg.only_touch:                   # item di tutti i difetti concatenati: filtro vettoriale
        it_len = np.array([len(d.items) for d in pool], dtype=np.int64)
        it_cat = np.concatenate([d.items for d in pool]) if pool else np.zeros(0, np.int64)
        it_off = np.r_[0, np.cumsum(it_len)[:-1]] if pool else np.zeros(0, np.int64)

    rec_f = open(out_dir / "recommendations.tsv", "w", newline="", encoding="utf-8")
    rec_w = csv.DictWriter(rec_f, delimiter="\t", lineterminator="\n", fieldnames=[
        "round", "group", "rank", "executed", "source", "defect_id", "cost",
        "mean", "bonus", "ucb", "y_hat", "push"])
    rec_w.writeheader()
    mea_f = open(out_dir / "measurements.tsv", "w", newline="", encoding="utf-8")
    mea_fields = (["round", "kind", "removed", "source", "defect_id", "cost", "n_aff_total"]
                  + [f"{a}_{g}" for a in ("r", "raw", "loc", "n_aff", "pred", "bonus", "y_hat", "push")
                     for g in GROUPS])
    mea_w = csv.DictWriter(mea_f, delimiter="\t", lineterminator="\n", fieldnames=mea_fields, restval="")
    mea_w.writeheader()
    jl = open(out_dir / "rounds.jsonl", "w", encoding="utf-8")
    # tutti i difetti effettivamente rimossi, round per round (scritto e salvato a ogni round)
    rm_f = open(out_dir / "removed_defects.tsv", "w", newline="", encoding="utf-8")
    rm_w = csv.DictWriter(rm_f, delimiter="\t", lineterminator="\n", restval="", fieldnames=(
        ["round", "order", "source", "defect_id", "cost", "n_users", "n_items",
         "edges_before_round", "edges_after_round",
         f"mean_{tgt}", f"bonus_{tgt}", f"ucb_{tgt}", f"y_hat_{tgt}"]
        + [f"r_{g}" for g in GROUPS] + [f"n_aff_{g}" for g in GROUPS]
        + [f"joint_round_{tgt}", f"cum_joint_{tgt}", "users", "items"]))
    rm_w.writeheader()
    reg_dir = out_dir / "regressor"
    if reg is not None:
        reg_dir.mkdir(exist_ok=True)
    reg_log, reg_models = [], []

    env.begin_round()
    ref_top = env.top_all.copy()
    state0 = None
    removed: set = set()
    edges, t = 0, 0
    cum = {g: 0.0 for g in GROUPS}
    cum_joint = {g: 0.0 for g in GROUPS}   # somma degli effetti CONGIUNTI dei batch rimossi
    joint_fallback = 0                     # round in cui il congiunto non c'era (si usa la somma dei singoli)
    stop = {"reason": "rounds", "round": None}
    best_cj, best_round, no_improve, patience_hit = -np.inf, None, 0, False
    prev_theta = bandit.theta.copy() if bandit else None
    prev_mean: Dict[tuple, float] = {}
    aff_mask = np.zeros(env.n_users, bool)      # utenti affetti dai difetti RIMOSSI finora
    emb_cache: Dict[tuple, np.ndarray] = {}
    rows = []

    def state_columns():
        """Metriche di stato sul grafo ATTUALE (su tutti gli utenti e sugli affetti)."""
        nonlocal state0
        seg_state, struct = env.state_metrics(ref_top)
        if state0 is None:
            state0 = seg_state
        row = {f"struct_{k}": v for k, v in struct.items()}
        for g in GROUPS:
            for m in STATE_METRICS:
                row[f"{g}_{m}"] = seg_state[g][m]
                row[f"{g}_d_{m}"] = seg_state[g][m] - state0[g][m]
        # stesse metriche solo sugli utenti affetti dai difetti rimossi finora: grafo attuale
        # vs grafo iniziale, sullo STESSO insieme di utenti (Δ = effetto su chi era esposto)
        aff_masks = {g: env.masks[g] & aff_mask for g in GROUPS}
        if aff_mask.any():
            cur_a = env.metrics_on(aff_masks, env.top_all, ref_top)
            ini_a = env.metrics_on(aff_masks, ref_top, ref_top)
        for g in GROUPS:
            row[f"{g}_aff_n"] = int(aff_masks[g].sum())
            for m in STATE_METRICS:
                v = cur_a[g][m] if aff_mask.any() else np.nan
                row[f"{g}_aff_{m}"] = v
                row[f"{g}_aff_d_{m}"] = v - ini_a[g][m] if aff_mask.any() else np.nan
        return row

    def hold_eval(bnd, rg):
        """Holdout fisso (grafo iniziale, mai usato per allenare): MAB e/o regressore."""
        out = {}
        if holdout is None:
            return out
        if bnd is not None and holdout.X_ctx is not None:
            for g in GROUPS:
                mu_h, gl_h = bnd.ucb(holdout.X_ctx[g], g)[0], holdout.glob(g)
                out[f"sp_hold_{g}"] = _sp(mu_h, gl_h)
                tch = np.array([i["n_aff"][g] for i in holdout.info]) > 0
                if tch.sum() >= 5:       # solo i difetti che toccano g (quelli tra cui si sceglie)
                    out[f"sp_hold_t_{g}"] = _sp(mu_h[tch], gl_h[tch])
        if rg is not None and holdout.X_emb is not None:
            for key, v in rg.evaluate(holdout.X_emb, {g: holdout.loc(g) for g in GROUPS}).items():
                out[f"reg_hold_{key}"] = v
        return out

    # Righe di rounds.tsv:
    #   round = -1  stato INIZIALE (grafo originale, nessuna rimozione)
    #   round = t   scelte e reward del round t + stato DOPO la rimozione dei difetti del round t
    while True:
        t0 = time.time()
        if t > 0:
            env.begin_round()                    # propagazione sul grafo senza i difetti rimossi
        state = state_columns()
        if t == 0:
            rows.append({"round": -1, "initial": 1, "edges_after": 0, **state, **hold_eval(bandit, reg)})
            jl.write(json.dumps(rows[-1], default=float) + "\n")
        else:
            rows[-1].update(state)               # stato dopo la rimozione del round t-1
            jl.write(json.dumps(rows[-1], default=float) + "\n")
            jl.flush()
            _log_round(log, rows[-1], cfg, reg is not None)
        if patience_hit:
            stop.update(reason="pazienza", round=t, best_round=best_round, best_cum_joint=best_cj)
            log.info(f"  Pazienza esaurita al round {t}: cumulata congiunta {tgt} ferma da "
                     f"{cfg.patience} round (migliore {best_cj:.3f} al round {best_round})")
            break
        if edges >= cfg.total_budget or t >= cfg.rounds:
            stop.update(reason="total_budget" if edges >= cfg.total_budget else "rounds", round=t)
            break
        row = {"round": t, "edges_before": edges}
        budget = min(cfg.budget_per_round, cfg.total_budget - edges)
        cands = [d for d in pool if d.key not in removed]

        # ── regressore: embedding sul grafo del round e y_hat ─────────────────
        if reg is not None and embedder is not None:
            if t % cfg.sage_every == 0 or any(d.key not in emb_cache for d in cands):
                E = embedder.embed(cands, env.removed_pairs)
                emb_cache = {d.key: e for d, e in zip(cands, E)}
            E = np.array([emb_cache[d.key] for d in cands])
            yh = {g: rank_norm(v) for g, v in reg.predict(E).items()}   # scala fissa
            for i, d in enumerate(cands):
                d.yhat = {g: float(yh[g][i]) for g in reg.groups}
                if "all" in d.yhat:
                    for g in GROUPS:
                        d.yhat.setdefault(g, d.yhat["all"])

        # ── pool: il MAB (o la baseline) sceglie solo tra N candidati ─────────
        #    regressor = i primi N per y_hat_all (il tuo regressore, sul grafo del round)
        #    random    = N candidati estratti a caso
        #    La sonda viene estratta da TUTTI i candidati rimasti, così il percentile
        #    è confrontabile tra pool diversi.
        n_all = len(cands)
        if cfg.only_touch:
            # limite superiore veloce: almeno un item del difetto nella top-K di un utente del
            # target (il controllo esatto, che esclude i membri, è l'exposure più sotto)
            hits = env.group_item_hits(tgt)
            tp = np.maximum.reduceat(hits[it_cat], it_off) > 0 if len(pool) else np.zeros(0, bool)
            elig = tp[np.array([pool_pos[d.key] for d in cands], dtype=np.int64)] if n_all else np.zeros(0, bool)
        else:
            elig = np.ones(n_all, bool)
        use_pool = cfg.pool != "none"
        pre_probe = set()
        if use_pool:
            # ordine: y_hat del regressore (decrescente) oppure casuale; si prendono i difetti
            # in quest'ordine finché la somma dei loro archi attuali arriva a pool_mult x budget
            # (o i primi pool_size difetti, se dato). Il costo si calcola SOLO per i difetti
            # visitati (prima: per tutti i candidati a ogni round, la parte lenta del round).
            if cfg.pool == "regressor":
                # con --yhat seg il pool è ordinato per l'impatto previsto sul TARGET
                order = np.argsort(-np.array([d.yhat.get(tgt, d.yhat.get("all", 0.0)) for d in cands]),
                                   kind="stable")
            else:
                order = rng.permutation(n_all)
            if cfg.only_touch:
                order = order[elig[order]]
            kk, tot = [], 0
            for j in order:
                c = env.R[cands[j].users][:, cands[j].items].nnz
                if c <= 0:
                    continue
                if cfg.pool_size > 0:
                    kk.append(int(j))
                    if len(kk) >= cfg.pool_size:
                        break
                else:
                    if kk and tot + c > cfg.pool_mult * budget:
                        break
                    kk.append(int(j))
                    tot += c
            keep = set(kk)
            if cfg.probe and cfg.only_touch:     # sonda tra i difetti che toccano il target
                ei = np.flatnonzero(elig)
                pre_probe = set(ei[rng.choice(len(ei), min(cfg.probe, len(ei)), replace=False)].tolist())
            elif cfg.probe:
                pre_probe = set(rng.choice(n_all, min(cfg.probe, n_all), replace=False).tolist())
            use = sorted(keep | pre_probe)
            in_pool = np.array([j in keep for j in use], dtype=bool)
            is_pre_probe = np.array([j in pre_probe for j in use], dtype=bool)
            cands = [cands[j] for j in use]
        elif cfg.only_touch:
            cands = [d for d, e in zip(cands, elig) if e]
        exposure = env.exposure(cands) if (fb.use_exposure or cfg.only_touch) else None
        if cfg.only_touch:                       # controllo esatto: exposure_target > 0
            k = np.flatnonzero(exposure[tgt] > 0)
            cands = [cands[j] for j in k]
            exposure = {g: np.asarray(v)[k] for g, v in exposure.items()}
            if use_pool:
                in_pool, is_pre_probe = in_pool[k], is_pre_probe[k]
        row["n_touch"] = int(elig.sum())
        X, keys, costs, ok, push = fb.transform(env.R, cands, env.masks, exposure)
        cands = [cands[j] for j in ok]
        n = len(cands)
        allowed = in_pool[ok] if use_pool else None
        row["n_cands_total"] = n_all
        row["pool_size"] = int(allowed.sum()) if use_pool else n
        if not n:
            log.info("  Nessun candidato rimasto.")
            stop.update(reason="no_candidates", round=t)
            break

        # ── una lista per OGNI gruppo; si esegue quella del target ───────────
        lists, mean, bonus = {}, {}, {}
        sel_stop = None
        perm = rng.permutation(n)
        random_phase = cfg.policy == "mab_cold" and t < cfg.init_rounds
        row["random_phase"] = int(random_phase)

        # ── ORACOLO: reward vera (LOO) di tutti i candidati del pool, su G_t ──
        if cfg.policy == "oracle":
            t_or = time.time()
            idx_o = np.flatnonzero(allowed) if allowed is not None else np.arange(n)
            true_r = {g: np.full(n, np.nan) for g in GROUPS}
            for j in idx_o:
                info = env.reward(cands[j])
                for g in GROUPS:
                    true_r[g][j] = info["glob"][g]
            row["oracle_n_measured"] = int(len(idx_o))
            row["oracle_seconds"] = round(time.time() - t_or, 1)

        for g in GROUPS:
            if bandit is not None and random_phase:
                lists[g] = greedy_fill(perm, keys, costs, budget, cfg.max_overlap, cfg.max_per_round,
                                       allowed)
                mean[g], bonus[g], _ = bandit.ucb(X[g], g)
            elif bandit is not None:
                lists[g], mean[g], bonus[g] = bandit.select(X[g], g, keys, costs, budget,
                                                            cfg.max_overlap, cfg.max_per_round, allowed,
                                                            force_one=not (cfg.mab_stop and t > 0))
                if g == tgt:
                    sel_stop = bandit.last_stop
            elif cfg.policy == "oracle":  # ← NUOVO RAMO
                sc = true_r[g] / np.maximum(costs, 1)  # reward vera / archi
                ok_o = np.isfinite(sc) & (sc > 0)  # solo difetti utili
                if allowed is not None:
                    ok_o &= allowed
                order = np.argsort(-np.nan_to_num(sc, nan=-np.inf), kind="stable")
                lists[g] = greedy_fill(order, keys, costs, budget, cfg.max_overlap,
                                       cfg.max_per_round, ok_o)
                mean[g], bonus[g] = np.nan_to_num(sc, nan=0.0), np.zeros(n)
            else:
                if cfg.policy == "push":
                    sc = push[g] / np.maximum(costs, 1)
                elif cfg.policy == "regressor":
                    sc = np.array([d.yhat.get(g, 0.0) for d in cands])
                else:
                    sc = np.zeros(n)
                order = perm if cfg.policy == "random" else np.argsort(-sc, kind="stable")
                lists[g] = greedy_fill(order, keys, costs, budget, cfg.max_overlap, cfg.max_per_round,
                                       allowed)
                mean[g], bonus[g] = sc, np.zeros(n)
        chosen = lists[tgt]
        if not chosen:
            if bandit is not None and not random_phase and sel_stop == "saturo":
                log.info(f"  MAB saturo al round {t} dopo {edges} archi rimossi: "
                         f"nessun candidato con UCB > 0 per {tgt}. Episodio finito.")
                stop.update(reason="mab_saturo", round=t)
            else:
                log.info("  Nessun difetto selezionabile con il budget del round.")
                stop.update(reason="no_feasible", round=t)
            break
        for g in GROUPS:
            for rank, j in enumerate(lists[g]):
                d = cands[j]
                rec_w.writerow({"round": t, "group": g, "rank": rank, "executed": int(g == tgt),
                                "source": d.source, "defect_id": d.id, "cost": int(costs[j]),
                                "mean": _fmt(mean[g][j]), "bonus": _fmt(bonus[g][j]),
                                "ucb": _fmt(mean[g][j] + bonus[g][j]),
                                "y_hat": _fmt(d.yhat.get(g, np.nan)), "push": _fmt(push[g][j])})
        rec_f.flush()

        if cfg.interactive:
            for i, j in enumerate(chosen):
                log.info(f"  [{i}] {cands[j].source} id={cands[j].id} archi={costs[j]} "
                         f"media={mean[tgt][j]:.3f} bonus={bonus[tgt][j]:.3f}")
            ans = input("Indici da rimuovere (es. 0,2 | 'all'): ").strip()
            pick = list(range(len(chosen))) if ans in ("all", "") else \
                [int(x) for x in ans.split(",") if x.strip()]
            to_remove = [chosen[i] for i in (pick or [0])]
        elif cfg.random_accept:
            k = int(acc_rng.integers(1, len(chosen) + 1))              # quanti: da 1 a tutti
            pick = np.sort(acc_rng.choice(len(chosen), k, replace=False))  # quali: a caso
            to_remove = [chosen[i] for i in pick]
        else:
            to_remove = list(chosen)

        # ── misure: scelti + sonda casuale (stesso grafo) ────────────────────
        if use_pool:
            probe = np.setdiff1d(np.flatnonzero(is_pre_probe[ok]), chosen)
        else:
            others = np.setdiff1d(np.arange(n), chosen)
            probe = rng.choice(others, min(cfg.probe, len(others)), replace=False) if cfg.probe else []
        meas = [("chosen", int(j), env.reward(cands[j])) for j in chosen]
        meas += [("probe", int(j), env.reward(cands[j])) for j in probe]
        if t == 0 and cfg.online_scale:
            scaler.fit_online({g: [m[2]["glob"][g] for m in meas] for g in GROUPS})
            log.info("  Scala reward online: " + ", ".join(f"{g}={s:.2e}" for g, s in scaler.scale.items()))
        R = {kind: {g: [] for g in GROUPS} for kind in ("chosen", "probe")}
        P = {kind: {g: [] for g in GROUPS} for kind in ("chosen", "probe")}
        rm_set = set(to_remove)
        for kind, j, info in meas:
            d = cands[j]
            rec = {"round": t, "kind": kind, "removed": int(j in rm_set), "source": d.source,
                   "defect_id": d.id, "cost": int(costs[j]), "n_aff_total": info["n_aff_total"]}
            for g in GROUPS:
                r = float(scaler(g, info["glob"][g]))
                R[kind][g].append(r)
                P[kind][g].append(mean[g][j])
                rec.update({f"r_{g}": _fmt(r), f"raw_{g}": f"{info['glob'][g]:.4e}",
                            f"loc_{g}": _fmt(info["loc"][g]), f"n_aff_{g}": info["n_aff"][g],
                            f"pred_{g}": _fmt(mean[g][j]), f"bonus_{g}": _fmt(bonus[g][j]),
                            f"y_hat_{g}": _fmt(d.yhat.get(g, np.nan)), f"push_{g}": _fmt(push[g][j])})
            mea_w.writerow(rec)
        mea_f.flush()
        R = {k: {g: np.array(v) for g, v in d.items()} for k, d in R.items()}
        P = {k: {g: np.array(v) for g, v in d.items()} for k, d in P.items()}

        # ── reward, percentile rispetto alla sonda ──────────────────────────
        rm_mask = np.array([j in rm_set for j in chosen])
        for g in GROUPS:
            rc = R["chosen"][g]
            cum[g] += float(rc[rm_mask].sum())
            row.update({f"r_{g}_sum": rc.sum(), f"r_{g}_mean": rc.mean(), f"cum_{g}": cum[g]})
            pr = R["probe"][g]
            if len(pr) >= 5:
                row[f"pct_{g}"] = float(np.mean([(pr < x).mean() + 0.5 * (pr == x).mean() for x in rc]))
                row[f"top20_{g}"] = float((rc > np.quantile(pr, 0.8)).mean())
                row[f"probe_mean_{g}"] = pr.mean()
        row.update(n_cands=row["pool_size"], n_chosen=len(chosen), n_removed=len(to_remove), budget=budget,
                   cost_chosen=int(costs[chosen].sum()))

        # ── MAB: capacità predittiva (prima dell'aggiornamento) ──────────────
        if bandit is not None:
            for g in GROUPS:
                row[f"sp_probe_{g}"] = _sp(P["probe"][g], R["probe"][g])
                row[f"sp_chosen_{g}"] = _sp(P["chosen"][g], R["chosen"][g])
                row[f"mae_chosen_{g}"] = float(np.abs(P["chosen"][g] - bandit.tr(R["chosen"][g])).mean())
                if len(R["probe"][g]):         # calibrazione: media prevista vs misurata sulla sonda
                    row[f"calib_pred_{g}"] = float(np.mean(P["probe"][g]))
                    row[f"calib_obs_{g}"] = float(np.mean(bandit.tr(R["probe"][g])))
                row[f"sigma_{g}"] = bandit.sigma(g)
            ucb_t = mean[tgt][chosen] + bonus[tgt][chosen]
            row["bonus_mean"] = float(bonus[tgt][chosen].mean())
            row["bonus_frac"] = float(np.mean(bonus[tgt][chosen] / np.maximum(np.abs(ucb_t), 1e-12)))
            row["unc_mean"] = float(bonus[tgt].mean())
            cur = {d.key: m for d, m in zip(cands, mean[tgt])}
            common = [k for k in cur if k in prev_mean]
            if len(common) >= 10:
                row["rank_stability"] = _sp([cur[k] for k in common], [prev_mean[k] for k in common])
            prev_mean = cur
            upd = list(chosen) + (list(probe) if cfg.probe_update else [])
            r_upd = {g: np.r_[R["chosen"][g], R["probe"][g] if cfg.probe_update else []] for g in GROUPS}
            Xu = {g: X[g][upd] for g in GROUPS}
            if cfg.hurdle:                        # theta_g impara solo dai difetti che toccano g
                na = {g: np.array([m[2]["n_aff"][g] for m in meas if m[0] == "chosen"]
                                  + ([m[2]["n_aff"][g] for m in meas if m[0] == "probe"]
                                     if cfg.probe_update else []), float) > 0 for g in GROUPS}
                Xu = {g: Xu[g][na[g]] for g in GROUPS}
                r_upd = {g: np.asarray(r_upd[g])[na[g]] for g in GROUPS}
                for g in GROUPS:
                    row[f"n_upd_{g}"] = int(na[g].sum())
            #bandit.update(Xu, r_upd)
            if cfg.bandit_update:
                bandit.update(Xu, r_upd)
            th = bandit.theta
            n_prev = np.linalg.norm(prev_theta)       # θ=0 (mab_cold al round 0): Δ relativo non definito
            row["theta_delta"] = float(np.linalg.norm(th - prev_theta) / n_prev) if n_prev > 1e-12 else np.nan
            row["theta_norm"] = float(np.linalg.norm(th))
            prev_theta = th.copy()
            row.update(hold_eval(bandit, None))              # holdout: modello a FINE round

        # ── liste per segmento: quanto sono diverse ─────────────────────────
        for a, b in PAIRS:
            row[f"jac_{a}_{b}"] = _jac(lists[a], lists[b])

        # ── regressore: test-then-train ─────────────────────────────────────
        if reg is not None:
            idx = {"chosen": [m[1] for m in meas if m[0] == "chosen"],
                   "probe": [m[1] for m in meas if m[0] == "probe"]}
            infos = {"chosen": [m[2] for m in meas if m[0] == "chosen"],
                     "probe": [m[2] for m in meas if m[0] == "probe"]}
            Xe = {k: np.array([emb_cache[cands[j].key] for j in v]).reshape(len(v), reg.dim)
                  for k, v in idx.items()}
            Yl = {k: {g: np.array([i["loc"][g] for i in infos[k]]) for g in GROUPS} for k in idx}
            for k in ("chosen", "probe"):
                for key, v in reg.evaluate(Xe[k], Yl[k]).items():
                    row[f"reg_{k}_{key}"] = v
            kinds = ["chosen"] + (["probe"] if cfg.reg_train_on == "chosen+probe" else [])
            for k in kinds:
                reg.add(Xe[k], Yl[k])
            reg.update()
            row.update(hold_eval(None, reg))                 # holdout: modello a FINE round
            row["reg_n_online"] = reg.n_online("all")
            for k in ("chosen", "probe"):
                for i, (j, info) in enumerate(zip(idx[k], infos[k])):
                    d = cands[j]
                    reg_log.append({"round": t, "kind": k, "trained": int(k in kinds),
                                    "source": d.source, "defect_id": d.id, "X": Xe[k][i],
                                    **{f"loc_{g}": info["loc"][g] for g in GROUPS},
                                    **{f"glob_{g}": info["glob"][g] for g in GROUPS},
                                    **{f"n_aff_{g}": info["n_aff"][g] for g in GROUPS}})
            reg_models.append({"round": t, "bundles": reg.snapshot()})

        # ── rimozione ed effetto congiunto ───────────────────────────────────
        batch = [cands[j] for j in to_remove]
        je = env.joint_effect(batch) if cfg.joint_effect else None
        for g in GROUPS:
            ind = float(scaler(g, sum(m[2]["glob"][g] for m in meas if m[0] == "chosen" and m[1] in rm_set)))
            row[f"sumind_{g}"] = ind
            if je is not None:
                row[f"joint_{g}"] = float(scaler(g, je["glob"][g]))
                cum_joint[g] += row[f"joint_{g}"]
            else:                                # congiunto non disponibile: ripiego sulla somma dei singoli
                cum_joint[g] += ind
            row[f"cum_joint_{g}"] = cum_joint[g]
        if je is None:
            joint_fallback += 1
        row["joint_fallback"] = joint_fallback
        for d in batch:                          # affetti misurati sul grafo PRIMA della rimozione
            aff_mask[env.affected_users(d)] = True
        n_rm = env.remove(batch)
        edges_before = edges
        edges += n_rm
        removed |= {d.key for d in batch}
        info_by_j = {m[1]: m[2] for m in meas if m[0] == "chosen"}
        for order, j in enumerate(to_remove):
            d, info = cands[j], info_by_j[j]
            u = d.u_tok if len(d.u_tok) else [str(x) for x in d.users]
            i = d.i_tok if len(d.i_tok) else [str(x) for x in d.items]
            rm_w.writerow({
                "round": t, "order": order, "source": d.source, "defect_id": d.id,
                "cost": int(costs[j]), "n_users": len(d.users), "n_items": len(d.items),
                "edges_before_round": edges_before, "edges_after_round": edges,
                f"mean_{tgt}": _fmt(mean[tgt][j]), f"bonus_{tgt}": _fmt(bonus[tgt][j]),
                f"ucb_{tgt}": _fmt(mean[tgt][j] + bonus[tgt][j]),
                f"y_hat_{tgt}": _fmt(d.yhat.get(tgt, np.nan)),
                **{f"r_{g}": _fmt(float(scaler(g, info["glob"][g]))) for g in GROUPS},
                **{f"n_aff_{g}": info["n_aff"][g] for g in GROUPS},
                f"joint_round_{tgt}": _fmt(row.get(f"joint_{tgt}", np.nan)),
                f"cum_joint_{tgt}": _fmt(cum_joint[tgt]),
                "users": ",".join(map(str, u)), "items": ",".join(map(str, i))})
        rm_f.flush()
        if cfg.patience > 0:                     # pazienza sulla cumulata CONGIUNTA del target
            cj = cum_joint[tgt]
            thr = best_cj + cfg.patience_rel * abs(best_cj) if np.isfinite(best_cj) else -np.inf
            if cj > thr:
                best_cj, best_round, no_improve = cj, t, 0
            else:
                no_improve += 1
            row["patience_count"] = no_improve
            patience_hit = no_improve >= cfg.patience
        row.update(edges_after=edges, seconds=round(time.time() - t0, 1))
        rows.append(row)                         # lo stato dopo la rimozione arriva al prossimo giro
        t += 1

    rec_f.close()
    mea_f.close()
    rm_f.close()
    jl.close()
    _write_rounds(rows, out_dir / "rounds.tsv")
    stop.update(edges_removed=int(edges), rounds_done=sum(1 for r in rows if r.get("round", -1) >= 0),
                cum={g: cum[g] for g in GROUPS}, cum_joint={g: cum_joint[g] for g in GROUPS},
                joint_fallback_rounds=joint_fallback)
    with open(out_dir / "stop.json", "w") as f:
        json.dump(stop, f, indent=1, default=float)
    log.info(f"  Fine episodio: motivo={stop['reason']} · round {stop['round']} · archi rimossi {edges} · "
             f"cumulata {tgt} {cum[tgt]:.3f} · cumulata congiunta {tgt} {cum_joint[tgt]:.3f}"
             + (f" · round senza congiunto: {joint_fallback}" if joint_fallback else ""))
    if reg is not None:
        _save_reg(reg_log, reg_models, reg_dir)
    if bandit is not None:
        with open(out_dir / "bandit_final.json", "w") as f:
            json.dump({"names": fb.names, **{k: {kk: float(vv) for kk, vv in v.items()}
                                             for k, v in bandit.explain(fb.names).items()}}, f, indent=1)
    return rows


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
#  MAIN
# ============================================================================ #
def parse_args():
    p = argparse.ArgumentParser(description="MAB segment-aware sui difetti")
    p.add_argument("--offline-datasets", nargs="+", default=["Baby_Products", "Toys_and_Games", "Pet_Supplies"])
    p.add_argument("--online-dataset", default=None)
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


def main():
    ROOT_DIR = Path(__file__).resolve().parent
    EMBEDDINGS_DIR = (ROOT_DIR / "../node_embeddings/defects/").resolve()

    args = parse_args()
    args.sage_encoder = (ROOT_DIR / "../node_embeddings/weights/sage_shared.pt").resolve()
    args.emb_dir = EMBEDDINGS_DIR
    args.feature_cache = (ROOT_DIR / "../mab/features/").resolve()
    REG_DIR = (ROOT_DIR / "../impact_regressor/scriba_regressor_out/final_model.pkl").resolve()
    REG_DIR_LT = (ROOT_DIR / "../impact_regressor/scriba_regressor_out/final_model_lt.pkl").resolve()
    REG_DIR_ACTIVE = (ROOT_DIR / "../impact_regressor/scriba_regressor_out/final_model_active.pkl").resolve()
    #args.regressor = REG_DIR
    #args.regressor_lt = REG_DIR_LT
    #args.regressor_active = REG_DIR_ACTIVE

    if args.defect_type:
        args.defect_files = [args.defect_type]
    if args.target:
        args.targets = [args.target]
    if args.policy:
        args.policies = [args.policy]
    print(f"ARGS: {args}")
    lo = _loo()
    log, rc = lo.log, lo.rc
    staging = Path(args.staging_dir or rc.STAGING_DIR)
    loo_dir = Path(args.loo_dir or rc.BASE_DIR / "results" / "loo")

    use_reg = args.regressor is not None or args.sage_encoder is not None or "regressor" in args.policies \
        or "regressor" in (args.pools or [])
    yhat_mode = args.yhat if use_reg and args.yhat != "none" else None
    reg_groups = list(GROUPS) if yhat_mode == "seg" else ["all"]
    fb_kwargs = dict(extra_cols=list(args.extra_cols), use_seg=not args.no_seg_features,
                     use_exposure=args.use_exposure, yhat_mode=yhat_mode)

    # ── Offline: difetti + reward del LOO (+ y_hat out-of-fold) ─────────────
    off = {n: load_offline(n, staging, loo_dir, args.extra_cols) for n in args.offline_datasets}
    data_off, bundle = None, None
    if use_reg:
        import mab_regressor as mr
        if not args.emb_dir:
            raise ValueError("il regressore richiede --emb-dir")
        data_off = {n: mr.load_emb_pickle(args.emb_dir, n) for n in args.offline_datasets}
        if args.regressor:
            with open(args.regressor, "rb") as f:
                bundle = pickle.load(f)
        seg_bundles = {}
        for g, path in (("active", args.regressor_active), ("lt", args.regressor_lt)):
            if path:
                with open(path, "rb") as f:
                    seg_bundles[g] = pickle.load(f)
        ln = (bundle or {}).get("label_norm", "rank")
        ng, ns = (bundle or {}).get("n_group", 500), (bundle or {}).get("n_gsdb", 500)
        if yhat_mode:
            for g in reg_groups:
                oof = mr.regressor_oof(data_off, list(data_off), mr.LABEL_OF[g], ln, ng, ns)
                for n in off:
                    vals = [oof[n].get(d.key) for d in off[n][2]]
                    ok_v = np.array([v is not None for v in vals], bool)
                    z = np.zeros(len(vals))                 # mancanti = 0 (la mediana)
                    if ok_v.any():
                        z[ok_v] = rank_norm([v for v in vals if v is not None])
                    for d, v in zip(off[n][2], z):
                        d.yhat[g] = float(v)
                    log.info(f"  [offline {n}] y_hat_{g} out-of-fold: {int(ok_v.sum())}/{len(vals)}")
            if yhat_mode == "common":
                for n in off:
                    for d in off[n][2]:
                        for g in GROUPS:
                            d.yhat.setdefault(g, d.yhat["all"])
    raws = offline_raws(off, log)             # la parte lenta, una volta per dataset
    if args.offline_check:
        rows = offline_check(off, fb_kwargs, log, raws)
        out = Path(args.out or rc.BASE_DIR / "results" / "mab") / "offline_check.tsv"
        out.parent.mkdir(parents=True, exist_ok=True)
        _write_rounds(rows, out)
    if not args.online_dataset:
        return

    scaler = RewardScaler().fit({g: np.concatenate([off[n][3][g] for n in off]) for g in GROUPS})
    log.info("  Scala reward offline: " + ", ".join(f"{g}={s:.2e}" for g, s in scaler.scale.items()))

    # ── Online: ambiente + candidati ─────────────────────────────────────────
    ds = args.online_dataset
    # una sottocartella per variante: configurazioni che differiscono solo per --yhat,
    # --reg-update o i filtri non si sovrascrivono (e il REPORT non le mescola)
    variant = (f"yhat-{args.yhat}_upd-{args.reg_update}"
               + (f"_src-{args.defect_type}" if args.defect_type else "")
               + ("_touch" if args.only_touching else "") + ("_hurdle" if args.hurdle else "")
               + ("_racc" if args.random_accept else "")
               + (f"_max{args.max_defects}" if args.max_defects else "")
               + (f"_rt-{args.reward_transform}" if args.reward_transform != "none" else "")
               + ("_stop" if args.mab_stop else "")
               + (f"_pat{args.patience}" if args.patience else "")
               + (f"_{args.tag}" if args.tag else ""))
    OUT_DIR = (ROOT_DIR / f"../mab/results/{ds}/{variant}").resolve()

    args.out = OUT_DIR
    env = Environment(ds, staging, args.topk, args.score_batch, args.test_file, args.head_frac, log)
    defects = []
    for s in args.defect_files:
        path = rc.DATASETS[ds] / f"{ds}.{s}.csv"
        if not path.exists():
            log.info(f"  ATTENZIONE: {path} non trovato, tipo {s} saltato")
            continue
        n0 = len(defects)
        for k, row in enumerate(read_defect_rows(path)):
            users, items = lo.to_ids(row["users"], env.uid), lo.to_ids(row["items"], env.iid)
            if len(users) and len(items):
                defects.append(Defect(k, users, items, _extra(row, args.extra_cols), s,
                                      _tok(row["users"]), _tok(row["items"])))
        log.info(f"  [online {ds}] {len(defects) - n0} difetti {s}")
    if args.max_defects:
        defects = subsample_defects(defects, env.R, args.max_defects, tuple(args.keep_all_sources), log=log)
    env.set_defect_items(defects)

    reg0, embedder = None, None
    if use_reg:
        import mab_regressor as mr
        reg0 = mr.OnlineRegressor(data_off, reg_groups, label_norm=ln, n_group=ng, n_gsdb=ns,
                                  mode=args.reg_update, online_weight=args.reg_online_weight,
                                  init_bundle=({"all": bundle, **seg_bundles} if seg_bundles else bundle),
                                  log=log)
        if args.sage_encoder:
            train_csv = args.train_csv or rc.DATASETS[ds] / f"{ds}.train.csv"
            embedder = mr.SageEmbedder(args.sage_encoder, Path(args.emb_dir) / "std_stats.pkl", train_csv, log)
            log.info(f"  GraphSAGE: embedding ricalcolati a ogni round (dim {embedder.dim})")
            pk = Path(args.emb_dir) / f"{ds}.pkl"
            if pk.exists():                      # controllo: round 0 == pickle
                E0 = embedder.embed(defects[:500], set())
                P0 = mr.PickleEmbedder(args.emb_dir, ds).embed(defects[:500])
                okr = np.isfinite(P0).all(1)
                if okr.any():
                    log.info(f"  Controllo embedding round 0 vs pickle: max |diff| = "
                             f"{np.abs(E0[okr] - P0[okr]).max():.2e}")
        else:
            embedder = mr.PickleEmbedder(args.emb_dir, ds)
            log.info("  Embedding statici dal pickle (senza --sage-encoder non cambiano con il grafo)")

    # FeatureBuilder adattato al grafo iniziale (y_hat del modello di partenza)
    env.begin_round()
    if reg0 is not None:
        E = embedder.embed(defects, set())
        yh = {g: rank_norm(v) for g, v in reg0.predict(E).items()}  # scala fissa
        for i, d in enumerate(defects):
            d.yhat = {g: float(yh[g][i]) for g in reg0.groups}
            for g in GROUPS:
                d.yhat.setdefault(g, d.yhat["all"])
    fb = fit_online_features(env, defects, fb_kwargs, args.use_exposure, ds,
                             args.feature_cache, args.fit_sample, log)
    if args.features_only:
        log.info("  --features-only: feature online pronte, esco.")
        return

    bandit_kwargs = dict(lam_shared=args.lam_shared, lam_seg=args.lam_seg, alpha=args.alpha,
                         reward_transform=args.reward_transform)
    log.info(f"  Reward per il LinUCB: {args.reward_transform}"
             + (" (sign(r)*log(1+|r|), solo per imparare)" if args.reward_transform == "slog" else ""))
    bandit0 = build_offline_prior(off, fb_kwargs, scaler, bandit_kwargs, args.warm_weight,
                                  args.warm_neff, log, raws, hurdle=args.hurdle)
    bandit_cold0 = SegmentLinUCB(fb.p, **bandit_kwargs)     # nessun prior
    holdout = measure_holdout(env, defects, args.holdout, args.seeds[0], fb, embedder, log) \
        if args.holdout else None

    # ── budget totale vs archi disponibili nel pool ─────────────────────────
    # pool = candidati selezionabili (holdout escluso, come in run_episode), grafo iniziale
    hold_keys = holdout.keys if holdout else set()
    args.pool_edges = pool_edges(env.R0, [d for d in defects if d.key not in hold_keys])
    src = "/".join(args.defect_files)
    if args.budget_cut > 0 and args.pool_edges < args.total_budget:
        new_budget = max(1, int(args.pool_edges * (1 - args.budget_cut)))
        log.info(f"  Pool {src}: {args.pool_edges} archi distinti < budget totale {args.total_budget}: "
                 f"budget totale ridotto a {new_budget} (-{args.budget_cut:.0%} degli archi del pool)")
        args.total_budget = new_budget
        if args.budget_per_round > args.total_budget:
            log.info(f"  ATTENZIONE: budget per round {args.budget_per_round} > budget totale "
                     f"{args.total_budget}: il primo round può esaurire tutto il budget")
    else:
        log.info(f"  Pool {src}: {args.pool_edges} archi distinti, budget totale {args.total_budget}")

    out_root = Path(args.out) if args.out else rc.BASE_DIR / "results" / "mab" / ds
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "config.json", "w") as f:
        json.dump({k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}, f, indent=1)
    if args.pools is None:
        args.pools = ["regressor"] if reg0 is not None else ["none"]
    log.info(f"  Pool di candidati: {args.pools} (pool = difetti fino a {args.pool_mult} x budget del round)"
             if args.pool_size <= 0 else f"  Pool di candidati: {args.pools} ({args.pool_size} difetti)")
    for target in args.targets:
        for pool in args.pools:
            if pool == "regressor" and reg0 is None:
                log.info("  --pools regressor richiede il regressore (--regressor/--emb-dir): saltato")
                continue
            for policy in args.policies:
                if policy == "regressor" and reg0 is None:
                    continue
                for seed in args.seeds:
                    cfg = RunConfig(target, policy, seed, args.budget_per_round, args.total_budget,
                                    args.rounds, args.max_per_round, args.max_overlap, args.probe,
                                    args.probe_update, args.interactive, not args.no_joint,
                                    not args.no_online_scale, args.reg_train_on, args.sage_every,
                                    args.init_rounds, pool, args.pool_size, args.pool_mult,
                                    only_touch=args.only_touching, hurdle=args.hurdle,
                                    random_accept=args.random_accept, mab_stop=args.mab_stop,
                                    patience=args.patience, patience_rel=args.patience_rel,bandit_update=not args.no_bandit_update)
                    name = f"{target}_{policy}" + (f"_pool-{pool}" if pool != "none" else "")
                    run_dir = out_root / f"{name}_s{seed}"
                    log.info(f"\n=== {ds}: target={target} politica={policy} pool={pool} seed={seed} → {run_dir}")
                    run_episode(env, defects, fb, bandit0, scaler, cfg, run_dir, log, reg0, embedder, holdout,
                                bandit_cold0)
                    with open(run_dir / "run.json", "w") as f:
                        json.dump({"dataset": ds, **cfg.__dict__}, f, indent=1)
    import mab_report
    mab_report.build_report(out_root)


if __name__ == "__main__":
    main()