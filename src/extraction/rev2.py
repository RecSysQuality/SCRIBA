#!/usr/bin/env python3
"""
rev2_defects.py
===============
REV2 (Kumar et al., "REV2: Fraudulent User Prediction in Rating Platforms", WSDM 2018)
sul grafo di training, e scrittura dei difetti sospetti nello STESSO formato dei file
group_f / gsdb_f (TSV con colonne `users` e `items`, token separati da virgola).

Un difetto = un utente sospetto (fairness bassa) + gli item a cui è collegato.
Rimuovere il difetto (utenti x item, come in loo_defects.py) toglie quindi i suoi archi.

FORMULAZIONE (Figura 4 del paper), score(u,p) in [-1, 1]:
  G(p)   = (sum_{u in In(p)} R(u,p) score(u,p) + b1 mu_g + b2 Pi_P(p)) / (|In(p)| + b1 + b2)
  R(u,p) = (g1 F(u) + g2 (1 - |score(u,p) - G(p)| / 2) + g3 Pi_R(u,p)) / (g1 + g2 + g3)
  F(u)   = (sum_{p in Out(u)} R(u,p) + a1 mu_f + a2 Pi_U(u)) / (|Out(u)| + a1 + a2)
Ordine degli aggiornamenti, inizializzazione (F0 = Pi_U, R0 = Pi_R, G0 = Pi_P) e
mu_f = media(F0), mu_g = media(G0) come nell'Algoritmo 1.
Unsupervised: media dei punteggi su tutte le combinazioni dei 7 parametri in {0,1,2}.

IMPLEMENTAZIONE (milioni di archi)
  - tutto vettoriale in torch (GPU se c'è): gli archi sono tensori (E,), gli
    aggregati sono index_add_; niente dizionari o loop in Python sugli archi;
  - K combinazioni di parametri girano INSIEME come colonne (tensori E x K):
    una iterazione aggiorna K esecuzioni di REV2. K si sceglie da --mem-gb;
  - combinazioni equivalenti non si ricalcolano: R dipende solo dalla DIREZIONE di
    (g1,g2,g3), quindi (2,2,2) = (1,1,1) ecc. Si calcola una volta e si pesa con la
    molteplicità: la media è identica a quella sulle 3^7 = 2187 combinazioni del paper
    (esclusa g1=g2=g3=0, che nel paper darebbe 0/0). 2187 -> 81 x 19 = 1539 esecuzioni;
  - convergenza: max |variazione| per entità < --tol, per ogni combinazione
    (il paper usa la somma degli errori con eps=1e-6, impraticabile su milioni di nodi).
    Il teorema del paper garantisce contrazione (3/4)^t: ~35 iterazioni per 1e-4.

COMPORTAMENTO (Pi_U, Pi_P) — approssimazione di BIRDNEST [7]
  Per ogni utente (e prodotto): istogramma dei valori di rating e dei tempi tra
  rating consecutivi (bin log, quantili globali). Sospetto = KL(Dir(a + c) || Dir(a)),
  con a = forza * distribuzione globale (posterior vs prior della popolazione):
  cresce con quanto l'entità devia dalla popolazione E con quante prove ci sono.
  Pi = 1 - sospetto normalizzato in [0,1]. Pi_R = 1 (il paper usa il testo, qui non c'è).
  Senza timestamp o con --no-behavior: Pi = 1 ovunque, come nel paper.

INPUT   data/original/<ds>/<ds>.train.csv  (user_id, item_id, rating[, timestamp]; header RecBole ok)
OUTPUT  data/original/<ds>/<ds>.rev2.csv   TSV: defect_id  n_users  n_items  users  items
        (users / items separati da virgola; un difetto = un utente sospetto, dal meno fair)

Uso:
  python rev2_defects.py --dataset Office_Products --top-n 2000
  python rev2_defects.py --dataset Office_Products --top-frac 0.01
  python rev2_defects.py --dataset X --edges unreliable      # solo gli archi poco affidabili
  python rev2_defects.py --dataset X --grid small            # 112 esecuzioni invece di 1539
  python rev2_defects.py --selftest                          # esempio del paper + frodi sintetiche
"""
from __future__ import annotations

import argparse
import csv
import itertools
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.special import digamma, gammaln

log = logging.getLogger("rev2")


# ==============================================================================
#  LETTURA E GRAFO
# ==============================================================================
def read_train(path: Path) -> pd.DataFrame:
    """Interazioni con id come STRINGHE (come graphsage.read_interactions / mab_defects._read_pairs)."""
    with open(path, encoding="utf-8") as f:
        head = f.readline()
    sep = "\t" if "\t" in head else ","
    df = pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False)
    df.columns = [c.split(":")[0].strip() for c in df.columns]
    for c in ("user_id", "item_id"):
        if c not in df.columns:
            raise ValueError(f"Colonna '{c}' mancante in {path} (colonne: {list(df.columns)})")
    if "rating" not in df.columns:
        raise ValueError("REV2 richiede la colonna 'rating': senza voti fairness e goodness non hanno senso")
    return df


def _timestamps(col: pd.Series) -> np.ndarray:
    t = pd.to_numeric(col, errors="coerce")
    if t.isna().mean() > 0.5:                          # date in formato testo
        d = pd.to_datetime(col, errors="coerce", utc=True)
        t = (d - pd.Timestamp("1970-01-01", tz="UTC")) / pd.Timedelta(seconds=1)
    return t.to_numpy(float)


@dataclass
class Rev2Graph:
    u_tok: np.ndarray           # token utenti (indice interno -> id esterno)
    p_tok: np.ndarray           # token item
    eu: np.ndarray              # (E,) utente di ogni arco (archi deduplicati)
    ep: np.ndarray              # (E,) item di ogni arco
    score: np.ndarray           # (E,) voto riscalato in [-1, 1]
    deg_u: np.ndarray
    deg_p: np.ndarray
    # righe grezze (con duplicati), per il comportamento
    ru: np.ndarray
    rp: np.ndarray
    rating: np.ndarray
    ts: Optional[np.ndarray]

    @property
    def n_u(self):
        return len(self.u_tok)

    @property
    def n_p(self):
        return len(self.p_tok)

    @property
    def n_e(self):
        return len(self.eu)


def build_graph(df: pd.DataFrame, rmin=None, rmax=None) -> Rev2Graph:
    rating = pd.to_numeric(df["rating"], errors="coerce").to_numpy(float)
    ok = np.isfinite(rating)
    if (~ok).any():
        log.info(f"  {int((~ok).sum())} righe senza rating valido ignorate")
    ru, u_tok = pd.factorize(df["user_id"].to_numpy()[ok])
    rp, p_tok = pd.factorize(df["item_id"].to_numpy()[ok])
    rating = rating[ok]
    ts = _timestamps(df["timestamp"])[ok] if "timestamp" in df.columns else None
    if ts is not None and not np.isfinite(ts).all():
        bad = ~np.isfinite(ts)
        log.info(f"  {int(bad.sum())} timestamp non validi: esclusi solo dal comportamento")

    lo = float(np.min(rating)) if rmin is None else float(rmin)
    hi = float(np.max(rating)) if rmax is None else float(rmax)
    if hi <= lo:
        raise ValueError(f"Rating costanti ({lo}): REV2 non può distinguere nulla")
    s = np.clip(2.0 * (rating - lo) / (hi - lo) - 1.0, -1.0, 1.0)

    # archi deduplicati: (u, p) ripetuti -> voto medio
    n_p = len(p_tok)
    key = ru.astype(np.int64) * n_p + rp
    uniq, inv = np.unique(key, return_inverse=True)
    cnt = np.bincount(inv)
    s_e = np.bincount(inv, weights=s) / cnt
    eu = (uniq // n_p).astype(np.int64)
    ep = (uniq % n_p).astype(np.int64)
    g = Rev2Graph(np.asarray(u_tok, dtype=object), np.asarray(p_tok, dtype=object), eu, ep,
                  s_e.astype(np.float32),
                  np.bincount(eu, minlength=len(u_tok)).astype(np.float32),
                  np.bincount(ep, minlength=n_p).astype(np.float32),
                  ru.astype(np.int64), rp.astype(np.int64), rating, ts)
    log.info(f"  Grafo: {g.n_u} utenti, {g.n_p} item, {g.n_e} archi "
             f"({len(rating) - g.n_e} duplicati fusi), rating [{lo:g}, {hi:g}] -> [-1, 1]")
    return g


# ==============================================================================
#  COMPORTAMENTO (approssimazione di BIRDNEST)
# ==============================================================================
def _rating_bins(r: np.ndarray, max_bins=10) -> Tuple[np.ndarray, int]:
    vals = np.unique(r)
    if len(vals) <= max_bins:
        return np.searchsorted(vals, r), len(vals)
    edges = np.unique(np.quantile(r, np.linspace(0, 1, max_bins + 1)[1:-1]))
    return np.searchsorted(edges, r, side="right"), len(edges) + 1


def _iat_bins(ent: np.ndarray, ts: np.ndarray, n_bins=20):
    """Tempi tra rating consecutivi della stessa entità, in bin log (quantili globali)."""
    ok = np.isfinite(ts)
    ent, ts = ent[ok], ts[ok]
    order = np.lexsort((ts, ent))
    e, t = ent[order], ts[order]
    same = e[1:] == e[:-1]
    x = np.log1p(np.maximum(t[1:] - t[:-1], 0.0))[same]
    who = e[1:][same]
    if len(x) == 0:
        return who, np.zeros(0, np.int64), 1
    edges = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)[1:-1]))
    return who, np.searchsorted(edges, x, side="right"), len(edges) + 1


def _hist(ent: np.ndarray, b: np.ndarray, n_ent: int, nb: int) -> np.ndarray:
    return np.bincount(ent * nb + b, minlength=n_ent * nb).reshape(n_ent, nb).astype(np.float64)


def dirichlet_surprise(C: np.ndarray, strength: Optional[float] = None, chunk=200_000) -> np.ndarray:
    """KL(Dir(a + c_i) || Dir(a)) per riga, a = forza * distribuzione globale."""
    nb = C.shape[1]
    p = C.sum(0) + 1e-3
    p /= p.sum()
    a = (strength if strength else float(nb)) * p
    a0, const = a.sum(), gammaln(a).sum() - gammaln(a.sum())
    out = np.empty(len(C))
    for i in range(0, len(C), chunk):
        c = C[i:i + chunk]
        A = a + c
        A0 = A.sum(1)
        out[i:i + chunk] = (gammaln(A0) - gammaln(A).sum(1) + const
                            + (c * (digamma(A) - digamma(A0)[:, None])).sum(1))
    return np.maximum(out, 0.0)


def _normality(score: np.ndarray, mode: str) -> np.ndarray:
    """Sospetto >= 0 -> normalità Pi in [0, 1] (1 = normale)."""
    if mode == "rank":
        r = pd.Series(score).rank(method="average").to_numpy()
        bn = (r - 1) / max(len(r) - 1, 1)
    else:                                              # "log-max": coda pesante compressa
        z = np.log1p(score)
        bn = (z - z.min()) / max(z.max() - z.min(), 1e-12)
    return (1.0 - bn).astype(np.float32)


def behavior_scores(g: Rev2Graph, norm="log-max", strength=None, iat_bins=20):
    """Pi_U (utenti) e Pi_P (item). None se non ci sono timestamp."""
    if g.ts is None:
        return None, None
    rb, nrb = _rating_bins(g.rating)
    out = []
    for ent, n in ((g.ru, g.n_u), (g.rp, g.n_p)):
        sus = dirichlet_surprise(_hist(ent, rb, n, nrb), strength)
        who, ib, nib = _iat_bins(ent, g.ts, iat_bins)
        if len(who):
            sus = sus + dirichlet_surprise(_hist(who, ib, n, nib), strength)
        out.append(_normality(sus, norm))
    log.info(f"  Comportamento: Pi_U media {out[0].mean():.3f}, Pi_P media {out[1].mean():.3f} "
             f"(rating in {nrb} bin, tempi tra rating in {nib} bin log)")
    return out[0], out[1]


# ==============================================================================
#  REV2
# ==============================================================================
def param_grid(values=(0, 1, 2)) -> Tuple[np.ndarray, np.ndarray]:
    """Combinazioni (a1,a2,b1,b2,g1,g2,g3) e pesi. (g1,g2,g3) proporzionali danno le
    stesse R: si tiene un rappresentante con peso = molteplicità."""
    gam = {}
    for gv in itertools.product(values, repeat=3):
        if sum(gv) == 0:
            continue
        d = math.gcd(*[int(x) for x in gv])
        k = tuple(int(x) // d for x in gv)
        gam[k] = gam.get(k, 0) + 1
    combos, w = [], []
    for ab in itertools.product(values, repeat=4):
        for gv, m in gam.items():
            combos.append(ab + gv)
            w.append(m)
    return np.array(combos, np.float32), np.array(w, np.float64)


class Rev2:
    def __init__(self, g: Rev2Graph, pi_u=None, pi_p=None, pi_r=None, device=None,
                 dtype=torch.float32, mu_g_mode="paper"):
        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dt = dtype
        T = lambda x, d=dtype: torch.as_tensor(np.asarray(x), dtype=d, device=self.dev)
        self.nU, self.nP, self.nE = g.n_u, g.n_p, g.n_e
        self.u, self.p = T(g.eu, torch.long), T(g.ep, torch.long)
        self.s = T(g.score)[:, None]
        self.deg_u, self.deg_p = T(g.deg_u)[:, None], T(g.deg_p)[:, None]
        self.pi_u = T(pi_u if pi_u is not None else np.ones(g.n_u))[:, None]
        self.pi_p = T(pi_p if pi_p is not None else np.ones(g.n_p))[:, None]
        self.pi_r = None if pi_r is None else T(pi_r)[:, None]       # None = 1 ovunque
        self.mu_f = float(self.pi_u.mean())                          # Alg. 1, riga 6
        self.mu_g = float(self.pi_p.mean()) if mu_g_mode == "paper" else float(g.score.mean())

    def run(self, combos: np.ndarray, tol=1e-4, max_iter=100):
        """Una esecuzione di REV2 per ogni riga di combos, tutte insieme (colonne)."""
        c = torch.as_tensor(combos, dtype=self.dt, device=self.dev)
        a1, a2, b1, b2, g1, g2, g3 = c.T                             # (K,) ciascuno
        K = len(combos)
        Fden, Gden = self.deg_u + a1 + a2, self.deg_p + b1 + b2
        Fpri = a1 * self.mu_f + a2 * self.pi_u                        # (U, K)
        Gpri = b1 * self.mu_g + b2 * self.pi_p                        # (P, K)
        Rpri = g3 * (self.pi_r if self.pi_r is not None else 1.0)     # (E, K) o (K,)
        gsum = g1 + g2 + g3
        F = self.pi_u.expand(self.nU, K).clone()
        # g2 = 0: R non dipende da G e F converge come (d/(d+a1+a2))^t, lentissimo per gli
        # utenti con molti archi. Il punto fisso qui è in forma chiusa (per utente):
        #   F (d + a1 + a2 - g1 d / (g1 + g3)) = g3 sum_p Pi_R / (g1 + g3) + a1 mu_f + a2 Pi_U
        # e si parte da lì: stesso risultato finale, convergenza in poche iterazioni.
        z = g2 == 0
        if bool(z.any()):
            gg = g1 + g3
            sum_pr = self.deg_u if self.pi_r is None else torch.zeros(
                self.nU, 1, dtype=self.dt, device=self.dev).index_add_(0, self.u, self.pi_r)
            den = Fden - g1 * self.deg_u / gg
            fix = (g3 * sum_pr / gg + Fpri) / den.clamp(min=1e-12)
            F = torch.where(z & (den > 1e-9), fix, F)
        G = self.pi_p.expand(self.nP, K).clone()
        R = (self.pi_r if self.pi_r is not None
             else torch.ones(1, 1, dtype=self.dt, device=self.dev)).expand(self.nE, K).clone()
        it = 0
        for it in range(1, max_iter + 1):
            # goodness (usa R^{t-1})
            num = torch.zeros(self.nP, K, dtype=self.dt, device=self.dev)
            num.index_add_(0, self.p, R * self.s)
            Gn = (num + Gpri) / Gden
            del num
            # reliability (usa F^{t-1} e G^t), in place per non allocare E x K in più
            Rn = F.index_select(0, self.u).mul_(g1)
            tmp = Gn.index_select(0, self.p).sub_(self.s).abs_().mul_(-0.5).add_(1.0).mul_(g2)
            Rn.add_(tmp).add_(Rpri).div_(gsum)
            # errore sugli archi prima di liberare R
            err = tmp.copy_(Rn).sub_(R).abs_().amax(0)
            del tmp, R
            # fairness (usa R^t)
            num = torch.zeros(self.nU, K, dtype=self.dt, device=self.dev)
            num.index_add_(0, self.u, Rn)
            Fn = (num + Fpri) / Fden
            del num
            err = torch.maximum(err, torch.maximum((Fn - F).abs().amax(0), (Gn - G).abs().amax(0)))
            F, G, R = Fn, Gn, Rn
            if float(err.max()) < tol:
                break
        return F, G, R, it, float(err.max())

    def batch_size(self, mem_gb: float) -> int:
        """Colonne che stanno in memoria: ~5 tensori E x K + qualche U x K / P x K."""
        per_col = (5 * self.nE + 4 * self.nU + 4 * self.nP) * torch.finfo(self.dt).bits // 8
        return max(1, int(mem_gb * 1e9 // per_col))

    def unsupervised(self, combos, weights, tol=1e-4, max_iter=100, mem_gb=4.0):
        """Media pesata (molteplicità) su tutte le combinazioni: punteggi REV2 finali."""
        K = min(self.batch_size(mem_gb), len(combos))
        W = float(weights.sum())
        Fs = torch.zeros(self.nU, dtype=torch.float64, device=self.dev)
        F2 = torch.zeros_like(Fs)
        Gs = torch.zeros(self.nP, dtype=torch.float64, device=self.dev)
        Rs = torch.zeros(self.nE, dtype=torch.float64, device=self.dev)
        log.info(f"  REV2: {len(combos)} esecuzioni (pesi totali {W:g}) su {self.dev}, "
                 f"{K} alla volta, tol {tol:g}")
        t0 = time.time()
        for i in range(0, len(combos), K):
            cb, wb = combos[i:i + K], weights[i:i + K]
            F, G, R, it, err = self.run(cb, tol, max_iter)
            w = torch.as_tensor(wb, dtype=self.dt, device=self.dev)
            Fs += (F @ w).double()
            F2 += ((F * F) @ w).double()
            Gs += (G @ w).double()
            Rs += (R @ w).double()
            del F, G, R
            done = min(i + K, len(combos))
            log.info(f"    {done}/{len(combos)}  iterazioni {it}  errore {err:.1e}  "
                     f"{time.time() - t0:.0f}s" + ("  (NON convergente)" if err >= tol else ""))
        Fm = Fs / W
        Fsd = torch.sqrt(torch.clamp(F2 / W - Fm * Fm, min=0.0))
        return (Fm.cpu().numpy(), Fsd.cpu().numpy(), (Gs / W).cpu().numpy(), (Rs / W).cpu().numpy())


# ==============================================================================
#  DIFETTI
# ==============================================================================
def select_users(F, deg_u, top_n=0, top_frac=0.0, max_fairness=None, min_ratings=1):
    cand = np.flatnonzero(deg_u >= min_ratings)
    cand = cand[np.argsort(F[cand], kind="stable")]              # meno fair per primi
    n = len(cand)
    if max_fairness is not None:
        n = min(n, int((F[cand] <= max_fairness).sum()))
    if top_frac:
        n = min(n, int(math.ceil(top_frac * len(cand))))
    if top_n:
        n = min(n, top_n)
    return cand[:n]


# def write_defects(path: Path, g: Rev2Graph, users, R, edges="all", rel_q=0.5):
#     """Un difetto per utente sospetto (dal meno fair): l'utente e gli item a cui è collegato."""
#     order = np.argsort(g.eu, kind="stable")
#     ptr = np.r_[0, np.cumsum(np.bincount(g.eu, minlength=g.n_u))]
#     thr = float(np.quantile(R, rel_q)) if edges == "unreliable" else None
#     n_edges = 0
#     with open(path, "w", newline="", encoding="utf-8") as f:
#         w = csv.writer(f, delimiter="\t", lineterminator="\n")
#         w.writerow(["defect_id", "n_users", "n_items", "users", "items"])
#         for k, u in enumerate(users):
#             e = order[ptr[u]:ptr[u + 1]]
#             if thr is not None:
#                 keep = e[R[e] <= thr]
#                 e = keep if len(keep) else e[[np.argmin(R[e])]]
#             n_edges += len(e)
#             w.writerow([k, 1, len(e), g.u_tok[u], ",".join(map(str, g.p_tok[g.ep[e]]))])
#     log.info(f"  {len(users)} difetti ({n_edges} archi) -> {path}")
def write_defects(path: Path, g: Rev2Graph, users, R, edges="all", rel_q=0.5, G=None, dis_q=0.8,
                  max_item_pop=None, min_edges=2):
    """Un difetto per utente sospetto (dal meno fair): l'utente e gli item a cui è collegato.
    max_item_pop: item con più di questo numero di utenti non entrano nei difetti.
    min_edges: un utente diventa un difetto solo se gli restano almeno min_edges archi dopo i filtri
    (niente ripiego sull'arco peggiore: un solo arco è una valutazione, non una struttura)."""
    order = np.argsort(g.eu, kind="stable")
    ptr = np.r_[0, np.cumsum(np.bincount(g.eu, minlength=g.n_u))]
    thr = float(np.quantile(R, rel_q)) if edges == "unreliable" else None
    if edges == "disagree":
        dis = np.abs(g.score - G[g.ep]) / 2.0
        dthr = float(np.quantile(dis, dis_q))
    pop_ok = None
    if max_item_pop is not None:
        pop_ok = g.deg_p <= max_item_pop                       # (P,) True = item ammesso
        log.info(f"  Item troppo popolari (> {max_item_pop:g} utenti) esclusi dai difetti: "
                 f"{int((~pop_ok).sum())} su {g.n_p}")
    n_edges = n_def = n_skip_pop = n_skip_small = 0
    items, sizes = set(), []
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["defect_id", "n_users", "n_items", "users", "items"])
        for u in users:
            e = order[ptr[u]:ptr[u + 1]]
            if pop_ok is not None:                             # prima togli gli item popolari
                e = e[pop_ok[g.ep[e]]]
                if len(e) == 0:                                # solo item popolari: niente difetto
                    n_skip_pop += 1
                    continue
            if thr is not None:
                e = e[R[e] <= thr]
            elif edges == "disagree":
                e = e[dis[e] >= dthr]
            if len(e) < min_edges:                             # troppo pochi archi sospetti: niente difetto
                n_skip_small += 1
                continue
            w.writerow([n_def, 1, len(e), g.u_tok[u], ",".join(map(str, g.p_tok[g.ep[e]]))])
            n_def += 1
            n_edges += len(e)
            items.update(g.ep[e].tolist())
            sizes.append(len(e))
    log.info(f"  {n_def} difetti ({n_edges} archi) -> {path}")
    log.info(f"  utenti sospetti saltati: {n_skip_pop} con solo item popolari, "
             f"{n_skip_small} con meno di {min_edges} archi sospetti")
    if sizes:
        sz = np.array(sizes)
        log.info(f"  archi per difetto: min {sz.min()}  mediana {np.median(sz):g}  max {sz.max()}")
    log.info(f"  copertura: {len(items)} item su {g.n_p} ({100 * len(items) / max(g.n_p, 1):.1f}%), "
             f"{n_edges} archi su {g.n_e} ({100 * n_edges / max(g.n_e, 1):.2f}%), "
             f"{n_def} utenti su {g.n_u} ({100 * n_def / max(g.n_u, 1):.2f}%)")

def write_defects_old(path: Path, g: Rev2Graph, users, R, edges="all", rel_q=0.5, G=None, dis_q=0.8):
    """Un difetto per utente sospetto (dal meno fair): l'utente e gli item a cui è collegato."""
    order = np.argsort(g.eu, kind="stable")
    ptr = np.r_[0, np.cumsum(np.bincount(g.eu, minlength=g.n_u))]
    thr = float(np.quantile(R, rel_q)) if edges == "unreliable" else None
    if edges == "disagree":
        # disaccordo tra il voto e la qualità dell'item, in [0, 1]: dipende dal singolo arco,
        # non dalla fairness dell'utente
        dis = np.abs(g.score - G[g.ep]) / 2.0
        dthr = float(np.quantile(dis, dis_q))
    n_edges = 0
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["defect_id", "n_users", "n_items", "users", "items"])
        for k, u in enumerate(users):
            e = order[ptr[u]:ptr[u + 1]]
            if thr is not None:
                keep = e[R[e] <= thr]
                e = keep if len(keep) else e[[np.argmin(R[e])]]
            elif edges == "disagree":
                keep = e[dis[e] >= dthr]
                e = keep if len(keep) else e[[np.argmax(dis[e])]]
            n_edges += len(e)
            w.writerow([k, 1, len(e), g.u_tok[u], ",".join(map(str, g.p_tok[g.ep[e]]))])
    log.info(f"  {len(users)} difetti ({n_edges} archi) -> {path}")

# ==============================================================================
#  SELF-TEST
# ==============================================================================
def selftest():
    # 1) Esempio 4.1 del paper (Figura 3): a=b=0, g1=g2=1, g3=0, tutto inizializzato a 1
    U = ["A", "B", "C", "D", "E"]
    rows = ([(u, "P1", 1.0) for u in U] + [("F", "P1", -1.0)]
            + [(u, "P2", 0.5) for u in U] + [("F", "P2", -1.0)]
            + [(u, "P3", -1.0) for u in U] + [("F", "P3", 1.0)])
    df = pd.DataFrame(rows, columns=["user_id", "item_id", "rating"]).astype(str)
    g = build_graph(df, rmin=-1, rmax=1)
    m = Rev2(g, device="cpu", dtype=torch.float64)
    F, G, R, it, _ = m.run(np.array([[0, 0, 0, 0, 1, 1, 0]]), tol=0, max_iter=9)
    Fv, Gv = F[:, 0].numpy(), G[:, 0].numpy()
    pi = {t: i for i, t in enumerate(g.p_tok)}
    ui = {t: i for i, t in enumerate(g.u_tok)}
    got = [Gv[pi["P1"]], Gv[pi["P2"]], Gv[pi["P3"]], Fv[ui["A"]], Fv[ui["F"]]]
    exp = [0.68, 0.32, -0.68, 0.86, 0.22]                      # tabella dell'Esempio 4.1
    print("Esempio 4.1 (iterazione 9):", np.round(got, 2), "atteso", exp)
    assert np.allclose(got, exp, atol=0.011), "Esempio del paper non riprodotto"

    # 2) Frodi sintetiche: 2% di utenti che votano all'opposto della qualità, a raffica
    rng = np.random.default_rng(0)
    nU, nP, nF = 3000, 800, 60
    q = rng.uniform(1, 5, nP)
    rows, t = [], []
    for u in range(nU):
        fraud = u < nF
        k = rng.integers(3, 30)
        its = rng.choice(nP, k, replace=False)
        base = rng.uniform(0, 3e7)
        for j, p in enumerate(its):
            r = (6 - q[p]) if fraud else q[p] + rng.normal(0, 0.8)
            rows.append((f"u{u}", f"i{p}", float(np.clip(np.rint(r), 1, 5))))
            t.append(base + (j * 15 if fraud else j * rng.exponential(86400 * 7)))
    df = pd.DataFrame(rows, columns=["user_id", "item_id", "rating"]).astype(str)
    df["timestamp"] = np.array(t).astype(np.int64).astype(str)
    g = build_graph(df)
    pu, pp = behavior_scores(g)
    combos, w = param_grid()
    Fm, Fsd, G, R = Rev2(g, pu, pp).unsupervised(combos, w, mem_gb=0.5)
    is_f = np.array([int(t[1:]) < nF for t in g.u_tok])
    from scipy.stats import mannwhitneyu
    auc = mannwhitneyu(-Fm[is_f], -Fm[~is_f]).statistic / (is_f.sum() * (~is_f).sum())
    top = np.argsort(Fm)[:nF]
    print(f"Frodi sintetiche: AUC {auc:.3f}, precision@{nF} {is_f[top].mean():.3f}")
    assert auc > 0.9
    print("SELFTEST OK")


# ==============================================================================
#  MAIN
# ==============================================================================
def resolve_paths(args) -> Tuple[Path, Path]:
    if not args.dataset and not args.train:
        raise SystemExit("Serve --dataset (oppure --train)")
    ds = args.dataset or Path(args.train).name.split(".")[0]
    base = Path(args.data_dir) / ds
    train = Path(args.train) if args.train else base / f"{ds}.train.csv"
    out = Path(args.out) if args.out else base / f"{ds}.rev2.csv"
    return train, out


def main():
    p = argparse.ArgumentParser(description="REV2 -> difetti (utenti sospetti + item collegati)")
    p.add_argument("--dataset", default=None)
    p.add_argument("--data-dir", default="data/original")
    p.add_argument("--train", default=None, help="file di training (altrimenti <data>/<ds>/<ds>.train.csv)")
    p.add_argument("--out", default=None, help="default: <data-dir>/<ds>/<ds>.rev2_f.csv")
    p.add_argument("--rating-min", type=float, default=None, help="default: minimo osservato")
    p.add_argument("--rating-max", type=float, default=None, help="default: massimo osservato")
    # selezione
    p.add_argument("--top-n", type=int, default=0, help="al massimo N utenti (0 = nessun limite)")
    p.add_argument("--top-frac", type=float, default=0.10, help="al massimo questa frazione degli utenti")
    p.add_argument("--max-fairness", type=float, default=None, help="solo utenti con fairness <= soglia")
    p.add_argument("--min-ratings", type=int, default=1, help="utenti con almeno N archi")
    # p.add_argument("--edges", choices=["all", "unreliable"], default="unreliable",
    #                help="all: tutti gli item dell'utente; unreliable: solo gli archi con "
    #                     "reliability <= quantile --rel-quantile (almeno uno)")
    p.add_argument("--rel-quantile", type=float, default=0.5)
    # algoritmo
    p.add_argument("--grid", choices=["paper", "small"], default="paper",
                   help="paper: parametri in {0,1,2} (1539 esecuzioni); small: in {0,1} (112)")
    p.add_argument("--tol", type=float, default=1e-4)
    p.add_argument("--max-iter", type=int, default=100)
    p.add_argument("--mem-gb", type=float, default=4.0, help="memoria per il batch di combinazioni")
    p.add_argument("--device", default=None, help="cuda / cpu (default: cuda se disponibile)")
    p.add_argument("--float64", action="store_true")
    p.add_argument("--mu-g", choices=["paper", "mean-score"], default="paper",
                   help="prior della goodness: media di G0 (Alg. 1) o voto medio")
    p.add_argument("--no-behavior", action="store_true")
    p.add_argument("--behavior-norm", choices=["log-max", "rank"], default="log-max")
    p.add_argument("--behavior-strength", type=float, default=None,
                   help="pseudo-conteggi del prior Dirichlet (default: n. di bin)")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--edges", choices=["all", "unreliable", "disagree"], default="disagree")
    p.add_argument("--dis-quantile", type=float, default=0.8,
                   help="disagree: tiene gli archi con disaccordo sopra questo quantile globale")


    p.add_argument("--min-defect-edges", type=int, default=3,
                   help="un utente sospetto diventa un difetto solo con almeno N archi sospetti "
                        "(dopo il filtro sugli item popolari e su --edges); 1 = anche difetti di un solo arco")
    p.add_argument("--max-item-pop", type=int, default=None,
                   help="esclude dai difetti gli item votati da più di N utenti")
    p.add_argument("--max-item-pop-quantile", type=float, default=0.80, # beauty 0.90
                   help="se --max-item-pop non è dato: soglia = questo quantile della popolarità "
                        "degli item (1.0 = nessuna esclusione)")


    args = p.parse_args()
    ROOT_DIR = Path(__file__).resolve().parent
    args.data_dir = (ROOT_DIR / "../data/original/").resolve()
    args.train = (ROOT_DIR / f"../data/original/{args.dataset}/{args.dataset}.train.csv").resolve()
    args.out = (ROOT_DIR / f"../data/original/{args.dataset}/{args.dataset}.rev2_fff.csv").resolve()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    if args.selftest:
        selftest()
        return

    train, out = resolve_paths(args)
    t0 = time.time()
    log.info(f"[rev2] {train}")
    g = build_graph(read_train(train), args.rating_min, args.rating_max)
    pi_u, pi_p = (None, None) if args.no_behavior else behavior_scores(
        g, args.behavior_norm, args.behavior_strength)
    if pi_u is None:
        log.info("  Comportamento non usato (niente timestamp o --no-behavior): Pi = 1")

    combos, w = param_grid((0, 1, 2) if args.grid == "paper" else (0, 1))
    model = Rev2(g, pi_u, pi_p, device=args.device,
                 dtype=torch.float64 if args.float64 else torch.float32, mu_g_mode=args.mu_g)
    F, Fsd, G, R = model.unsupervised(combos, w, args.tol, args.max_iter, args.mem_gb)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    q = np.quantile(F, [0.01, 0.05, 0.25, 0.5])
    log.info(f"  Fairness: min {F.min():.3f}  q1% {q[0]:.3f}  q5% {q[1]:.3f}  "
             f"q25% {q[2]:.3f}  mediana {q[3]:.3f}")
    users = select_users(F, g.deg_u, args.top_n, args.top_frac, args.max_fairness, args.min_ratings)
    out.parent.mkdir(parents=True, exist_ok=True)
    #write_defects(out, g, users, R, args.edges, args.rel_quantile, G=G, dis_q=args.dis_quantile)

    max_pop = args.max_item_pop
    if max_pop is None and args.max_item_pop_quantile < 1.0:
        max_pop = float(np.quantile(g.deg_p, args.max_item_pop_quantile))
    write_defects(out, g, users, R, args.edges, args.rel_quantile, G=G, dis_q=args.dis_quantile,
                  max_item_pop=max_pop, min_edges=args.min_defect_edges)

    log.info(f"[rev2] fatto in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()