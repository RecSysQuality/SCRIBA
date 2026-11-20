from __future__ import annotations

import csv
import hashlib
import os
import pickle
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.stats import norm, rankdata


# ============================================================================ #
#  COSTANTI E UTILITÀ
# ============================================================================ #
GROUPS: Tuple[str, ...] = ("all", "active", "lt")
PAIRS = (("lt", "active"), ("lt", "all"), ("active", "all"))
STATE_METRICS = ("entropy", "gini", "coverage", "novelty", "arp", "aplt",
                 "ndcg", "recall", "hit", "churn", "defect_share")
POL = ["mab", "mab_cold", "random", "push", "regressor", "oracle"]
IMPACTS = ("rr", "tndcg")
# colonna della reward nei CSV del LOO, per misura d'impatto ({g} = gruppo)
LOO_REWARD_COL = {"rr": "{g}_glob_d_prop_rr_best", "tndcg": "{g}_glob_d_prop_tndcg"}

# .../src  (questo file sta in src/mab/)
SRC_DIR = Path(__file__).resolve().parents[1]


def rank_norm(v):
    """y_hat -> scala N(0,1) usando solo l'ORDINE dei valori.
    Così la feature y_hat ha sempre la stessa scala, qualunque sia il regressore
    (pre-allenato, riallenato online, out-of-fold offline)."""
    v = np.asarray(v, float)
    if len(v) < 2:
        return np.zeros_like(v)
    return norm.ppf((rankdata(v) - 0.5) / len(v))


def _loo():
    """Import pigro: il nucleo (feature, bandit, metriche) non richiede torch/recbole."""
    sys.path.insert(0, str(SRC_DIR))
    import recbole_f.loo_defects as lo
    return lo


class _Log:
    def info(self, m):
        print(m, flush=True)


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
#  METRICHE
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


def tndcg_full(scores, items) -> np.ndarray:
    """
    NDCG binario sul ranking INTERO (non troncato), rilevanti = item del difetto.
    Versione numpy di riferimento (FakeEnvironment e test); Environment usa _tndcg in torch.
      scores (n, n_items) punteggi degli utenti, -inf per item 0 e item già visti
      rank(i) = 1 + #item con punteggio > punteggio di i      (stessa regola di _rr_best)
      DCG     = sum_{i rilevanti non visti} 1 / log2(rank(i) + 1)
      IDCG    = sum_{k=1..n_rel} 1 / log2(k + 1),  n_rel = item del difetto non visti
    Utente senza item rilevanti raggiungibili -> 0.
    """
    sc = np.asarray(scores, float)
    items = np.unique(np.asarray(items, dtype=np.int64))
    if sc.shape[0] == 0:
        return np.zeros(0)
    s_it = sc[:, items]
    valid = np.isfinite(s_it)
    n_gt = np.column_stack([(sc > s_it[:, [j]]).sum(1) for j in range(len(items))])
    dcg = np.where(valid, 1.0 / np.log2(n_gt + 2.0), 0.0).sum(1)
    n_rel = valid.sum(1)
    idcg = np.cumsum(1.0 / np.log2(np.arange(2, len(items) + 2)))
    out = np.zeros(len(sc))
    v = n_rel > 0
    out[v] = dcg[v] / idcg[n_rel[v] - 1]
    return out


# ============================================================================ #
#  FEATURE OFFLINE / ONLINE (con cache)
# ============================================================================ #
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
