"""
GSDB sui dati NON filtrati, poi proiezione dei difetti sul train k-core.

Perché: il train è k-core, quindi non contiene i single-time reviewer (SR) usati
da Savg nell'Algoritmo 2 (Eq. 21) né gli account attivi per poco tempo che gli
indicatori individuali (AD, MRO, RTI, ATR) premiano. GSDB gira quindi sul file
<dataset>_kcore1.csv, tagliato in modo da non vedere validation e test, e i
gruppi trovati vengono poi ristretti a ciò che esiste nel train.

Passi, per ogni dataset:
  1. carica il train k-core e il file non filtrato (<dataset>_kcore1.csv);
  2. anti-leakage sul non filtrato:
       - tiene solo le interazioni con data <= ultima data del train;
       - toglie le coppie (utente, item) presenti in validation/test, se i file esistono;
  3. fa girare GSDB (gsdb.py, invariato) sul non filtrato tagliato, con due opzioni:
       - --max-item-pop-quantile (default 0.80): toglie dai prodotti target gli item
         popolari, con la STESSA regola di rev2_defects.py: esclusi gli item votati da
         più utenti del quantile 0.80 del train (cioè il ~20% più popolare). Su un item
         popolare ci sono così tanti recensori che bastano coincidenze per formare un
         "gruppo" in una settimana;
       - --kde-bw-days: bandwidth della KDE in giorni (Algoritmo 3), coerente con
         ISIZE = 7 giorni. 0 = bandwidth automatica di scipy (comportamento di gsdb.py);
       - gli indicatori individuali e Savg usano tutto il non filtrato, ma i prodotti
         target assenti dal train vengono saltati: i loro gruppi verrebbero comunque
         scartati nella proiezione, quindi l'output finale non cambia;
  4. proietta ogni gruppo sul train: tiene solo i membri e gli item che hanno
     almeno un arco del train tra loro; scarta i gruppi con < --min-members membri;
     unisce i gruppi che dopo la proiezione hanno gli stessi membri;
  5. calcola le feature di output sul train (come per gli altri file *_f) e salva
     <dataset>.gsdb_f.csv nello stesso formato di prima (backup del vecchio file).

Uso (dalla root del progetto):
  python3 src/extraction/gsdb_unfiltered.py --datasets Health_and_Household
"""
from __future__ import annotations

import argparse
import os
import shutil
import time

import numpy as np
import pandas as pd

from scipy.stats import gaussian_kde

from gsdb import (GSDBFast, load_reviews_csv, compute_all_group_metrics,
                  METRIC_COLS)


DAY_NS = 86_400 * 10**9
_WORKER = None          # istanza condivisa con i processi figli (fork, nessuna copia via pickle)


def _work_chunk(args):
    products, isize_days, delta_i, delta_g, theta_days, d_days = args
    g = _WORKER
    out = {}
    for p in products:
        cand = g.candidate_group_generation(p, isize_days)
        sg = g.group_purification_and_classification(cand, delta_i, delta_g, theta_days, d_days)
        if sg:
            out[p] = sg
    return out, len(products)


class GSDBTuned(GSDBFast):
    """GSDBFast con le stesse formule di gsdb.py, più:
    - i prodotti in `exclude` non diventano mai prodotti target (Algoritmo 2);
    - la KDE dell'Algoritmo 3 può usare una bandwidth fissa di `bw_days` giorni (0 = scipy);
    - Algoritmi 3 e 4 riscritti in numpy (stessi risultati, verificati contro gsdb.py):
      indici per reviewer/prodotto precalcolati, GSS calcolato una volta per gruppo
      (prima ogni GSS rifaceva 7 volte l'estrazione pandas delle recensioni) e messo
      in cache, finestre di co-attività (Eq. 17-18) in tempo lineare invece che quadratico;
    - parallelismo con fork: i processi figli condividono i dati invece di riceverne una
      copia serializzata per ogni prodotto, come succedeva con ProcessPoolExecutor."""

    def __init__(self, reviews, exclude=(), bw_days: float = 0.0, allowed=None,
                 kde_trunc: float = 8.0, **kw):
        self.exclude = set(exclude)
        self.allowed = None if allowed is None else set(allowed)
        self.bw_days = float(bw_days)
        self.kde_trunc = float(kde_trunc)
        super().__init__(reviews, **kw)
        self._build_fast_index()

    # ------------------------------------------------------------------ indici
    def _build_fast_index(self):
        t0 = time.time()
        df = self.df                                            # ordinato per data
        rcode, runiq = pd.factorize(df["reviewer_id"])
        pcode, puniq = pd.factorize(df["product_id"])
        self._runiq = np.asarray(runiq, dtype=object)
        self._rcode_of = {u: i for i, u in enumerate(self._runiq)}
        self._pcode_of = {p: i for i, p in enumerate(np.asarray(puniq, dtype=object))}
        self._rcode = rcode.astype(np.int64)
        self._pcode = pcode.astype(np.int64)
        self._t = df["date"].values.astype("datetime64[ns]").astype(np.int64)
        self._r = df["rating"].values.astype(np.float64)
        self._eday = self._t // 10**9 // 86400                    # come astype(int64)//1e9//86400
        self._fday = np.floor_divide(self._t, DAY_NS)             # come dt.floor("D")
        # righe per reviewer e per prodotto, in ordine di data (argsort stabile)
        self._r_ord = np.argsort(rcode, kind="stable")
        self._r_ptr = np.r_[0, np.cumsum(np.bincount(rcode, minlength=len(runiq)))]
        self._p_ord = np.argsort(pcode, kind="stable")
        self._p_ptr = np.r_[0, np.cumsum(np.bincount(pcode, minlength=len(puniq)))]
        self._psum = np.bincount(pcode, weights=self._r, minlength=len(puniq))
        self._pcnt = np.bincount(pcode, minlength=len(puniq)).astype(np.float64)
        ind = self.individual
        self._iss = dict(zip(self._plain_index(ind.index), ind["ISS"].values))
        self._exr = dict(zip(self._plain_index(ind.index), ind["EXR"].values))
        self._gss_cache = {}
        self._log(f"[fast] indici numpy in {time.time() - t0:.1f}s")

    def target_product_location(self, *args, **kw):
        tp = super().target_product_location(*args, **kw)
        n0 = len(tp)
        if self.allowed is not None:
            tp = [p for p in tp if p in self.allowed]
            self._log(f"[train] tolti {n0 - len(tp):,} prodotti target assenti dal train "
                      f"(i loro gruppi sparirebbero comunque nella proiezione)")
        kept = [p for p in tp if p not in self.exclude]
        self._log(f"[head] tolti {len(tp) - len(kept):,} prodotti target head, "
                  f"restano {len(kept):,}")
        return kept

    def _kde_on_grid(self, x, f, m):
        """Densità di gaussian_kde(x, bw_method=f) sulla griglia linspace(0, 1, m), sommando per
        ogni recensione solo i punti entro kde_trunc deviazioni standard (oltre, il contributo è
        < 1e-14 del picco). Stessa larghezza di banda di scipy: f * deviazione standard con ddof=1."""
        n = len(x)
        h = f * np.std(x, ddof=1)
        step = 1.0 / (m - 1)
        R = int(np.ceil(self.kde_trunc * h / step)) + 1
        if R * 2 + 1 >= m:                                          # griglia piccola: calcolo esatto
            return gaussian_kde(x, bw_method=f)(np.linspace(0, 1, m))
        off = np.arange(-R, R + 1)[None, :]
        dens = np.zeros(m)
        B = max(1, int(2e6 // (2 * R + 1)))                         # a blocchi, memoria limitata
        for s in range(0, n, B):
            xb = x[s:s + B]
            js = np.rint(xb / step).astype(np.int64)[:, None] + off
            ok = (js >= 0) & (js < m)
            z = (js * step - xb[:, None]) / h
            dens += np.bincount(js[ok], weights=np.exp(-0.5 * z * z)[ok], minlength=m)
        return dens / (n * h * np.sqrt(2 * np.pi))

    # ------------------------------------------------------------ Algoritmo 3
    def candidate_group_generation(self, product_id, isize_days: int = 7):
        c = self._pcode_of.get(product_id)
        if c is None:
            return []
        idx = self._p_ord[self._p_ptr[c]:self._p_ptr[c + 1]]     # righe del prodotto, per data
        n = len(idx)
        if n < 3:
            return []
        t = self._t[idx]
        t0 = int(t[0])
        dur = int((t[-1] - t0) // DAY_NS)                        # Timedelta.days
        if dur == 0:
            return [frozenset(self._runiq[self._rcode[idx]])]
        x = (t - t0) / 1e9 / 86400.0 / dur
        k = max(2, dur // isize_days)
        avg_rev = n / k
        try:
            if self.bw_days > 0:
                sd = x.std()
                if sd <= 0:
                    return [frozenset(self._runiq[self._rcode[idx]])]
                m = min(dur + 1, 5000)
                grid = np.linspace(0, 1, m)
                density = self._kde_on_grid(x, (self.bw_days / dur) / sd, m)
            else:                                                # come gsdb.py
                grid = np.linspace(0, 1, 500)
                density = gaussian_kde(x)(grid)
        except np.linalg.LinAlgError:
            return [frozenset(self._runiq[self._rcode[idx]])]
        d1 = np.gradient(density, grid)
        peak_idx = np.where((d1[:-1] > 0) & (d1[1:] <= 0))[0]
        if len(peak_idx) == 0:
            return []
        # t_peak = t0 + Timedelta(days=x*dur), finestra [t_peak - 3g, t_peak + 3g] inclusa
        t_peak = np.array([t0 + pd.Timedelta(days=float(grid[i] * dur)).value for i in peak_idx],
                          dtype=np.int64)
        a = np.searchsorted(t, t_peak - 3 * DAY_NS, side="left")
        b = np.searchsorted(t, t_peak + 3 * DAY_NS, side="right")
        cnt = b - a
        ok = (cnt > 1) & (cnt > avg_rev)
        groups, seen = [], set()
        for lo, hi in zip(a[ok], b[ok]):
            group = frozenset(self._runiq[self._rcode[idx[lo:hi]]])
            if group not in seen:
                seen.add(group)
                groups.append(group)
        return groups

    # ------------------------------------------------------------ Algoritmo 4
    def _purify_group(self, group, delta_i: float) -> frozenset:
        iss = self._iss
        return frozenset(a for a in group if iss.get(a, 0.0) >= delta_i)

    def _rows_of(self, members):
        parts = [self._r_ord[self._r_ptr[c]:self._r_ptr[c + 1]]
                 for c in (self._rcode_of.get(a) for a in members) if c is not None]
        return np.concatenate(parts) if parts else np.zeros(0, np.int64)

    def GSS(self, group_members, theta_days=30, d_days=7) -> float:
        key = (frozenset(group_members), theta_days, d_days)
        hit = self._gss_cache.get(key)
        if hit is None:
            hit = self._gss_fast(key[0], theta_days, d_days)
            self._gss_cache[key] = hit
        return hit

    def _gss_fast(self, members, theta_days, d_days) -> float:
        R = self._rows_of(members)
        size = len(members)
        gs = float(1.0 / (1.0 + np.exp(-(size - 3))))                                   # Eq. 12
        ger = float(np.mean([self._exr.get(a, 0.0) for a in members])) if size else 0.0  # Eq. 16
        if len(R) == 0:
            return float(np.mean([0.0, 0.0, gs, 0.0, 0.0, ger, 0.5, 0.0]))

        # per prodotto
        pu, pinv = np.unique(self._pcode[R], return_inverse=True)
        pc = np.bincount(pinv).astype(np.float64)
        # GTW, Eq. 8-9: deviazione standard (popolazione) dei giorni per prodotto
        ed = self._eday[R].astype(np.float64)
        mu = np.bincount(pinv, weights=ed) / pc
        sd = np.sqrt(np.bincount(pinv, weights=(ed - mu[pinv]) ** 2) / pc)
        sd[pc < 2] = 0.0
        gtw = float(np.mean(np.where(sd <= theta_days, 1.0 - sd / theta_days, 0.0)))
        # GRD, Eq. 10-11: media del prodotto escludendo il gruppo
        r = self._r[R]
        s = np.bincount(pinv, weights=r)
        tot_s, tot_c = self._psum[pu], self._pcnt[pu]
        rem = tot_c - pc
        rstar = np.where(rem > 0, (tot_s - s) / np.where(rem > 0, rem, 1), tot_s / tot_c)
        grd = float(np.mean(np.bincount(pinv, weights=np.abs(r - rstar[pinv]) / (self.rmax - self.rmin)) / pc))
        # GRT, Eq. 13
        grt = len(R) / (len(set(members)) * len(pu))
        # GOR, Eq. 14-15: per reviewer, quota di giorni con più di 5 recensioni
        rc = self._rcode[R]
        key = rc * (self._fday.max() + 1) + self._fday[R]
        _, kinv, kcnt = np.unique(key, return_inverse=True, return_counts=True)
        first = np.zeros(len(kcnt), np.int64)
        first[kinv] = rc                                                   # reviewer di ogni (reviewer, giorno)
        ru, rinv = np.unique(first, return_inverse=True)
        n_days = np.bincount(rinv)
        n_busy = np.bincount(rinv, weights=(kcnt > 5).astype(np.float64))
        gor = float(np.mean(n_busy / n_days))
        # GCA e GCAR, Eq. 17-18
        gca, gcar = self._co_active(R, rc, len(set(members)), d_days)
        return float(np.mean([gtw, grd, gs, grt, gor, ger, gca, gcar]))

    def _co_active(self, R, rc, n_members, d_days):
        """Finestre [c - d, c + d] (c = ogni data distinta delle recensioni del gruppo) in cui
        compaiono TUTTI i membri; finestra scorrevole, tempo lineare."""
        o = np.argsort(self._t[R], kind="stable")
        t, who = self._t[R][o], rc[o]
        n = len(t)
        _, who = np.unique(who, return_inverse=True)
        centers = np.unique(t)
        w = d_days * DAY_NS
        A = np.searchsorted(t, centers - w, side="left")
        B = np.searchsorted(t, centers + w, side="right")
        counts = np.zeros(who.max() + 1, np.int64)
        distinct = lo = hi = 0
        good = np.zeros(len(centers), bool)
        for j in range(len(centers)):
            while hi < B[j]:
                counts[who[hi]] += 1
                distinct += counts[who[hi]] == 1
                hi += 1
            while lo < A[j]:
                counts[who[lo]] -= 1
                distinct -= counts[who[lo]] == 0
                lo += 1
            good[j] = distinct == n_members
        if not good.any():
            return float(1.0 / (1.0 + np.exp(0))), 0.0
        gc = centers[good]
        n_merged = 1 + int(np.sum((gc[1:] - w) > (gc[:-1] + w)))          # finestre unite
        diff = np.zeros(n + 1, np.int64)
        np.add.at(diff, A[good], 1)
        np.add.at(diff, B[good], -1)
        covered = int((np.cumsum(diff)[:n] > 0).sum())
        return float(1.0 / (1.0 + np.exp(-n_merged))), covered / n

    # ------------------------------------------------------------ Algoritmo 1
    def run(self, delta_tp: float = 0.1, delta_i: float = 0.43, delta_g: float = 0.54,
            k: float = 0.5, isize_days: int = 7, theta_days: int = 30, d_days: int = 7,
            n_jobs: int = 1, chunk: int = 200):
        global _WORKER
        t0 = time.time()
        targets = self.target_product_location(delta_tp, k)
        # i prodotti grandi prima, così i processi finiscono insieme
        sizes = {p: self._p_ptr[c + 1] - self._p_ptr[c] for p in targets
                 for c in [self._pcode_of.get(p)] if c is not None}
        targets = sorted(sizes, key=sizes.get, reverse=True)
        chunks = [targets[i:i + chunk] for i in range(0, len(targets), chunk)]
        jobs = [(c, isize_days, delta_i, delta_g, theta_days, d_days) for c in chunks]
        results, done, last = {}, 0, time.time()

        def progress(n):
            nonlocal done, last
            done += n
            if time.time() - last > 30 or done == len(targets):
                el = time.time() - t0
                eta = el / max(done, 1) * (len(targets) - done)
                self._log(f"   {done:,}/{len(targets):,} prodotti target, {len(results):,} con gruppi, "
                          f"{el / 60:.1f} min, mancano ~{eta / 60:.1f} min")
                last = time.time()

        if n_jobs <= 1 or len(chunks) < 2:
            for job in jobs:
                _WORKER = self
                out, n = _work_chunk(job)
                results.update(out)
                progress(n)
        else:
            import multiprocessing as mp
            _WORKER = self
            with mp.get_context("fork").Pool(n_jobs) as pool:
                for out, n in pool.imap_unordered(_work_chunk, jobs):
                    results.update(out)
                    progress(n)
        _WORKER = None
        self._log(f"[Algoritmo 1] completato in {(time.time() - t0) / 60:.1f} min: "
                  f"{sum(len(v) for v in results.values())} gruppi spammer trovati "
                  f"su {len(targets):,} prodotti target ({n_jobs} processi).")
        return results


def head_items(train: pd.DataFrame, max_pop=None, quantile: float = 0.80) -> set:
    """Item popolari esclusi, con la stessa regola di rev2_defects.py:
    popolarità = numero di utenti distinti nel train; esclusi gli item con popolarità
    > max_pop, oppure, se max_pop non è dato, > quantile `quantile` delle popolarità."""
    deg = train.groupby("product_id")["reviewer_id"].nunique()
    if max_pop is None:
        if quantile >= 1.0:
            return set()
        max_pop = float(np.quantile(deg.values, quantile))
    head = set(deg.index[deg > max_pop])
    print(f"[head] item troppo popolari (> {max_pop:g} utenti) esclusi da GSDB: {len(head):,} su "
          f"{len(deg):,} ({100 * len(head) / len(deg):.1f}%), che coprono "
          f"{100 * deg[deg > max_pop].sum() / deg.sum():.1f}% degli archi del train")
    return head


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--root", default="data/original")
    p.add_argument("--full-pattern", default="{ds}_kcore1.csv",
                   help="file non filtrato dentro <root>/<ds>/")
    p.add_argument("--train-pattern", default="{ds}.train.csv")
    p.add_argument("--eval-patterns", nargs="*",
                   default=["{ds}.valid.csv", "{ds}.validation.csv", "{ds}.test.csv"],
                   help="file di validation/test le cui coppie (utente, item) vanno escluse")
    p.add_argument("--out-pattern", default="{ds}.gsdb_ff.csv")
    # soglie GSDB (stesse che usavi; delta_i = 0.43 è quella del paper)
    p.add_argument("--delta-tp", type=float, default=0.05)
    p.add_argument("--delta-i", type=float, default=0.43)
    p.add_argument("--delta-i-quantile", type=float, default=None,
                   help="se dato, delta_i = questo quantile dell'ISS (es. 0.70) invece di --delta-i")
    p.add_argument("--delta-g", type=float, default=0.2)
    p.add_argument("--min-members", type=int, default=3,
                   help="membri minimi dopo la proiezione (GSDB richiede > 2)")
    p.add_argument("--max-item-pop", type=int, default=None,
                   help="esclude da GSDB gli item votati da più di N utenti nel train (come rev2_defects.py)")
    p.add_argument("--max-item-pop-quantile", type=float, default=0.80,
                   help="se --max-item-pop non è dato: soglia = questo quantile della popolarità "
                        "degli item nel train (1.0 = nessuna esclusione). Stesso default di rev2_defects.py")
    p.add_argument("--kde-bw-days", type=float, default=2.0,
                   help="bandwidth della KDE in giorni (0 = automatica di scipy, come gsdb.py)")
    p.add_argument("--n-jobs", type=int, default=max(1, min(16, len(os.sched_getaffinity(0)) - 1)),
                   help="processi paralleli per gli Algoritmi 3-4 (default: CPU disponibili - 1, max 16)")
    return p.parse_args()


class PairCoder:
    """Coppie (utente, item) -> interi int64, con gli stessi codici per tutti i file
    (molto più leggero di stringhe concatenate e set Python su decine di milioni di righe)."""

    def __init__(self, users, items):
        self.u = pd.Index(pd.unique(np.asarray(users, dtype=object)))
        self.i = pd.Index(pd.unique(np.asarray(items, dtype=object)))

    def keys(self, users, items) -> np.ndarray:
        cu = self.u.get_indexer(np.asarray(users, dtype=object)).astype(np.int64)
        ci = self.i.get_indexer(np.asarray(items, dtype=object)).astype(np.int64)
        k = cu * len(self.i) + ci
        k[(cu < 0) | (ci < 0)] = -1                                  # coppia con utente/item sconosciuto
        return k


def load_eval_pairs(path: str):
    df = pd.read_csv(path, usecols=["user_id", "item_id"], dtype=str)
    return df["user_id"].values, df["item_id"].values


def cut_unfiltered(full: pd.DataFrame, train: pd.DataFrame, eval_paths: list) -> pd.DataFrame:
    n0 = len(full)
    cutoff = train["date"].max()
    full = full.loc[full["date"] <= cutoff]
    print(f"[leakage] cutoff temporale {cutoff}: tolte {n0 - len(full):,} interazioni successive")

    coder = PairCoder(full["reviewer_id"].values, full["product_id"].values)
    fk = coder.keys(full["reviewer_id"].values, full["product_id"].values)
    if not eval_paths:
        print("[leakage] ATTENZIONE: nessun file di validation/test trovato. Uso solo il cutoff "
              "temporale, che basta solo se lo split è temporale globale.")
    for path in eval_paths:
        eu, ei = load_eval_pairs(path)
        ek = coder.keys(eu, ei)
        ek = ek[ek >= 0]
        drop = np.isin(fk, ek)
        full, fk = full.loc[~drop], fk[~drop]
        print(f"[leakage] {os.path.basename(path)}: tolte {int(drop.sum()):,} interazioni "
              f"({len(eu):,} coppie nel file)")

    # controllo: il train dovrebbe essere (quasi) tutto contenuto nel non filtrato tagliato
    tk = coder.keys(train["reviewer_id"].values, train["product_id"].values)
    cov = np.isin(tk, fk).mean()
    print(f"[controllo] coppie del train presenti nel non filtrato tagliato: {cov:.1%}")
    if cov < 0.99:
        print("[controllo] ATTENZIONE: sotto il 99%. Controlla che ID e timestamp dei due file "
              "siano gli stessi, e che il file di validation/test non contenga coppie del train.")
    return full.reset_index(drop=True)


def project_on_train(groups_map: dict, train: pd.DataFrame, min_members: int):
    """Restringe ogni gruppo agli archi del train tra i suoi membri e i suoi item."""
    all_users = set().union(*groups_map.keys()) if groups_map else set()
    sub = train.loc[train["reviewer_id"].isin(all_users), ["reviewer_id", "product_id"]]
    user_items = sub.groupby("reviewer_id")["product_id"].agg(set).to_dict()

    stats = {"before": len(groups_map), "no_train_edges": 0, "too_small": 0}
    small_sizes = []
    proj = {}
    for members, items in groups_map.items():
        items = set(items)
        m = frozenset(u for u in members if user_items.get(u, set()) & items)
        if not m:
            stats["no_train_edges"] += 1
            continue
        if len(m) < min_members:
            stats["too_small"] += 1
            small_sizes.append(len(m))
            continue
        it = set().union(*(user_items[u] & items for u in m))
        proj.setdefault(m, set()).update(it)

    edges = set()
    for m, it in proj.items():
        for u in m:
            edges.update((u, i) for i in user_items[u] & it)
    stats["after"] = len(proj)
    stats["merged"] = stats["before"] - stats["no_train_edges"] - stats["too_small"] - len(proj)
    stats["edges"] = len(edges)
    stats["small_sizes"] = pd.Series(small_sizes, dtype=int).value_counts().sort_index().to_dict()
    return proj, stats


def run_dataset(ds: str, a) -> None:
    print("=" * 70, f"\nDATASET {ds}")
    d = os.path.join(a.root, ds)
    train_path = os.path.join(d, a.train_pattern.format(ds=ds))
    full_path = os.path.join(d, a.full_pattern.format(ds=ds))
    out_path = os.path.join(d, a.out_pattern.format(ds=ds))
    eval_paths = [os.path.join(d, p.format(ds=ds)) for p in a.eval_patterns]
    eval_paths = [p for p in eval_paths if os.path.exists(p)]

    train = load_reviews_csv(train_path)
    full = load_reviews_csv(full_path)
    print(f"[dati] train: {len(train):,} interazioni, {train['reviewer_id'].nunique():,} utenti | "
          f"non filtrato: {len(full):,} interazioni, {full['reviewer_id'].nunique():,} utenti")

    full = cut_unfiltered(full, train, eval_paths)

    head = head_items(train, a.max_item_pop, a.max_item_pop_quantile)
    gsdb = GSDBTuned(full, exclude=head, bw_days=a.kde_bw_days,
                     allowed=set(train["product_id"].unique()))
    del full                                                  # GSDBFast ne tiene già una copia
    iss = gsdb.individual["ISS"]
    n_sr = int((gsdb.total_counts == 1).sum())
    delta_i = float(iss.quantile(a.delta_i_quantile)) if a.delta_i_quantile is not None else a.delta_i
    print(f"[gsdb] utenti con 1 sola recensione (SR): {n_sr:,} | "
          f"ISS >= {delta_i:.3f}: {(iss >= delta_i).mean():.1%} degli utenti")

    detected = gsdb.run(delta_tp=a.delta_tp, delta_i=delta_i, delta_g=a.delta_g, n_jobs=a.n_jobs)

    groups_map = {}
    for product, groups in detected.items():
        for g in groups:
            groups_map.setdefault(frozenset(g["members"]), set()).add(product)

    proj, st = project_on_train(groups_map, train, a.min_members)
    print(f"[proiezione] gruppi GSDB sul non filtrato: {st['before']:,}")
    print(f"             senza archi nel train: {st['no_train_edges']:,}")
    print(f"             sotto {a.min_members} membri dopo la proiezione: {st['too_small']:,} "
          f"(dimensioni: {st['small_sizes']})")
    print(f"             uniti perché con gli stessi membri: {st['merged']:,}")
    print(f"             gruppi finali: {st['after']:,}, archi del train coperti: {st['edges']:,} "
          f"({st['edges'] / len(train):.2%} del train)")

    if proj:
        pop = train["product_id"].value_counts()
        rank = pop.rank(ascending=False, pct=True)          # 0 = il più popolare
        found = set().union(*proj.values())
        r = rank.reindex(list(found)).dropna()
        print(f"[item] {len(found):,} item nei difetti; posizione nel ranking di popolarità del train "
              f"(0% = più popolare): mediana {100 * r.median():.1f}%, "
              f"nel top 1%: {(r <= 0.01).mean():.1%}, nel top 5%: {(r <= 0.05).mean():.1%}")

    t0 = time.time()
    rows = compute_all_group_metrics(train, proj,
                                     user_counts=train["reviewer_id"].value_counts())
    print(f"[metriche] {len(rows):,} gruppi valutati sul train in {time.time() - t0:.1f}s")

    out_df = pd.DataFrame(rows, columns=["n_users", "n_items", "users", "items"] + METRIC_COLS)
    out_df.sort_values(["n_users", "n_items"], ascending=[False, False], inplace=True)

    if os.path.exists(out_path):
        bak = out_path.replace(".csv", ".kcore_bak.csv")
        if not os.path.exists(bak):
            shutil.copy2(out_path, bak)
            print(f"[output] vecchio file salvato in {bak}")
    out_df.to_csv(out_path, sep="\t", index=False, float_format="%.4f")
    print(f"[output] salvato {out_path}: {len(out_df):,} gruppi")


if __name__ == "__main__":
    args = parse_args()
    for ds in args.datasets:
        run_dataset(ds, args)