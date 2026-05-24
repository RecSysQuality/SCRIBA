"""
Spam Injection Pipeline — versione ottimizzata ENTROPY
===============================================
Ottimizzazioni rispetto alla versione originale:
  1. used_items: list → set  (O(1) lookup invece di O(n))
  2. exclude: list → set     (O(1) lookup invece di O(n))
  3. shuffle rimosso prima di random.sample (già random di per sé)
  4. pool filtering vectorizzato con numpy set operations
  5. print raggruppati / verbose flag per ridurre I/O
  6. item_counts weights precalcolati per pool fissi
  7. sample_items non ricrea list(pool) ogni volta — usa pool cached

Input:
    CSV con colonne [user_id, item_id, rating, timestamp]

Output:
    dataset_dirty.csv   — dataset originale + righe iniettate
    injected_nodes.csv  — solo le righe iniettate con tutti i metadati
"""

import os
import random
import warnings
from datetime import timedelta
from typing import List, Tuple, Optional
from src.defect_injector.configs import *

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════
# CONTESTO DI INJECTION
# ══════════════════════════════════════════════════════════════════

class InjectionContext:
    """
    Stato condiviso durante l'injection — versione ottimizzata.

    Cambiamenti rispetto all'originale:
      - used_items: set invece di list  → O(1) lookup
      - pool arrays numpy precalcolati  → no list() ogni chiamata
      - weights per contested pools precalcolati
      - verbose flag per ridurre print I/O
    """

    def __init__(self, df: pd.DataFrame, config: InjectionConfig,
                 verbose: bool = True, overlap: float = 0.0):
        random.seed(config.seed)
        np.random.seed(config.seed)

        self._user_counter = 0
        self.used_items = set()          # ← ERA lista, O(n) lookup
        self.verbose    = verbose
        self.df         = df.copy()
        self.config     = config

        # normalizza timestamp
        self.df["timestamp"] = pd.to_datetime(self.df["timestamp"])

        self.all_users = self.df["user_id"].unique().tolist()
        self.all_items = self.df["item_id"].unique().tolist()

        # ── item counts e pool ────────────────────────────────────
        item_counts = self.df["item_id"].value_counts()
        self.item_counts = item_counts

        q25 = item_counts.quantile(0.25)
        q50 = item_counts.quantile(0.50)
        q75 = item_counts.quantile(0.75)
        q99 = item_counts.quantile(0.99)

        self.popular_items = item_counts[
            (item_counts >= q75) & (item_counts <= q99)
        ].index.tolist()
        self.tail_items = item_counts[
            (item_counts >= q25) & (item_counts <= q50)
        ].index.tolist()
        self.top_items = item_counts[
            (item_counts >= q99)
        ].index.tolist()

        # shuffle una volta sola — non ripetuto in ogni sample_items
        random.shuffle(self.popular_items)
        random.shuffle(self.tail_items)
        random.shuffle(self.top_items)

        k_p = int(len(self.tail_items) * config.contested_pool_pct)
        k_n = int(len(self.top_items)  * config.contested_pool_pct)
        self.contested_items_push = set(self.tail_items[:k_p])
        self.contested_items_nuke = set(self.top_items[:k_n])
        self.contested_items_popular = set(self.popular_items[:k_p])
        self.contested_items_top = set(self.top_items[:k_p])
        self.contested_items      = self.contested_items_push | self.contested_items_nuke | self.contested_items_popular
        self.overlap = overlap # choices: 0,25,50,75
        if self.overlap == 0:
            self.contested_items = []

        # ── numpy arrays per sampling veloce ─────────────────────
        # precalcolati una volta, usati da sample_items
        self._pool_arrays = {
            'push':    np.array(self.tail_items),
            'nuke':    np.array(self.top_items),
            'popular': np.array(self.popular_items),
            'top': np.array(self.top_items),
            'all':     np.array(self.all_items),
        }

        # weights per contested pools (normalizzati) — precalcolati
        self._contested_weights = {
            'push': self._compute_weights(list(self.contested_items_push)),
            'nuke': self._compute_weights(list(self.contested_items_nuke)),
            'popular': self._compute_weights(list(self.contested_items_popular)),
            'top': self._compute_weights(list(self.contested_items_top)),
        }
        self._contested_arrays = {
            'push':    np.array(list(self.contested_items_push)),
            'nuke':    np.array(list(self.contested_items_nuke)),
            'popular': np.array(list(self.contested_items_popular)),
            'top': np.array(list(self.contested_items_top)),
        }

        # media rating per item
        self.item_means = (
            self.df.groupby("item_id")["rating"].mean().to_dict()
        )

        # range temporale
        self.ts_min = self.df["timestamp"].min()
        self.ts_max = self.df["timestamp"].max()
        self._ts_range_s = (self.ts_max - self.ts_min).total_seconds()

        # lista righe iniettate
        self.injected: List[dict] = []

    # ── helpers interni ───────────────────────────────────────────

    def _compute_weights(self, items: list) -> np.ndarray:
        """Precalcola weights normalizzati per un pool di item."""
        if not items:
            return np.array([])
        w = np.array([self.item_counts.get(i, 1) for i in items], dtype=float)
        w /= w.sum()
        return w

    def _log(self, msg: str):
        if self.verbose:
            print(msg, flush=True)

    # ── sampling ──────────────────────────────────────────────────

    def sample_with_overlap(self, n: int, contested_ratio: float = None,
                            mode: str = 'push',
                            exclude: Optional[list] = None, exclude_used_items:  Optional[bool] = True) -> list:
        exclude_set = set(exclude) if exclude else set()  # ← O(1) lookup
        if not contested_ratio or self.overlap == 0:
            contested_ratio = self.overlap
        if self.overlap == 0:
            exclude_used_items = True

        n_c = round(n * contested_ratio)

        contested = []
        thr = 1 - self.overlap


        if n_c > 0 and random.random() > thr:
            arr = self._contested_arrays.get(mode, self._contested_arrays['push'])
            w   = self._contested_weights.get(mode, self._contested_weights['push'])

            # filtra exclude con maschera numpy — molto più veloce del list comp
            if exclude_set:
                mask = np.array([i not in exclude_set for i in arr])
                arr  = arr[mask]
                w    = w[mask]
                if w.sum() > 0:
                    w = w / w.sum()

            if len(arr) > 0 and len(w) > 0:
                k = min(n_c, len(arr))
                contested = np.random.choice(arr, size=k,
                                             replace=False).tolist()
                # if self.verbose and contested:
                #     self._log(f'overlap {len(contested)}')

        private = self.sample_items(
            n - len(contested),
            mode=mode,
            exclude=list(exclude_set | set(contested)),
            exclude_used_items=exclude_used_items
        )
        return contested + private

    def sample_items(self, n: int, mode: str = 'all',
                     exclude: Optional[list] = None,
                     exclude_used_items: bool = False) -> list:
        """
        Campiona n item dal pool corrispondente a mode.
        Ottimizzazioni:
          - usa numpy array precalcolato (no list() ogni chiamata)
          - exclude come set per O(1) lookup
          - no shuffle ridondante prima di random.sample
        """
        pool = self._pool_arrays.get(mode, self._pool_arrays['all'])

        # costruisci maschera di esclusione in un solo passaggio numpy
        exclude_set = set(exclude) if exclude else set()
        if exclude_used_items:
            exclude_set |= self.used_items  # ← set union, O(1) per elemento

        if exclude_set:
            # maschera booleana numpy — molto più veloce del list comp su grandi pool
            mask = np.array([i not in exclude_set for i in pool])
            pool = pool[mask]

        if exclude_used_items and len(pool) < n:
            # fallback: pool generale
            full = self._pool_arrays['all']
            mask = np.array([i not in self.used_items for i in full])
            pool = full[mask]

        if n > 1:
            n_sample = random.randint(n, n + 3)
        else:
            n_sample = 1
        n_sample  = min(n_sample, len(pool))

        if n_sample <= 0:
            return []

        # random.sample su lista — no shuffle preventivo
        return random.sample(pool.tolist(), n_sample)

    # ── generatori ID e timestamp ─────────────────────────────────

    def new_user(self) -> str:
        uid = f"fake_user_{self._user_counter}"
        self._user_counter += 1
        return uid

    def random_ts(self, center=None, window_hours: float = 10) -> pd.Timestamp:
        if center is None:
            return self.ts_min + timedelta(
                seconds=random.uniform(0, self._ts_range_s)  # precalcolato
            )
        return center + timedelta(
            seconds=random.uniform(0, window_hours * 3600)
        )

    def clip_rating(self, r: float) -> float:
        return float(np.clip(round(r * 2) / 2, 1.0, 5.0))

    # ── add_row ───────────────────────────────────────────────────

    def add_row(self, user_id, item_id, rating: float,
                timestamp: pd.Timestamp, family: str, casistica: str,
                group_id: str, split: str = "train",
                defect_type: str = "spam"):
        self.injected.append({
            "user_id":     user_id,
            "item_id":     item_id,
            "rating":      self.clip_rating(rating),
            "timestamp":   timestamp,
            "family":      family,
            "casistica":   casistica,
            "group_id":    group_id,
            "split":       split,
            "defect_type": defect_type,
        })

    # ── output ────────────────────────────────────────────────────

    def get_dataframes(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        injected_df = pd.DataFrame(self.injected)

        base = self.df[["user_id","item_id","rating","timestamp"]].copy()
        base["family"]      = "original"
        base["casistica"]   = "original"
        base["group_id"]    = "original"
        base["split"]       = "original"
        base["defect_type"] = "legit"

        dirty_df = pd.concat(
            [base, injected_df[[
                "user_id","item_id","rating","timestamp",
                "family","casistica","group_id","split","defect_type"
            ]]],
            ignore_index=True,
        )
        return dirty_df, injected_df


# ══════════════════════════════════════════════════════════════════
# FAMIGLIA 1 — GRUPPI COORDINATI
# ══════════════════════════════════════════════════════════════════

def inject_group_dense_cluster(ctx: InjectionContext):
    """Burst puro. Rilevato da: FraudAR, GSDB."""
    cfg = ctx.config
    for g_idx in range(cfg.f1_burst_groups):
        ctx._log(f'[GROUP Dense cluster] group {g_idx}')
        group_id   = f"group_dense_cluster_{g_idx}"
        fakeusers  = random.randint(10, 21)

        targets    = ctx.sample_with_overlap(
            n=np.random.randint(10, 20), contested_ratio=0.0, mode='push'
        )
        filler    = ctx.sample_with_overlap(
            n=np.random.randint(20, 30), contested_ratio=0.25, mode='push'
        )
        ctx.used_items.update(
            i for i in targets if i not in ctx.contested_items
        )
        burst_ts = ctx.random_ts()
        for _ in range(fakeusers):
            uid = ctx.new_user()
            for item in targets:
                ts = ctx.random_ts(burst_ts, cfg.f1_burst_window_hours)
                ctx.add_row(uid, item, 5, ts, "Group", "target", group_id)
            for item in filler:
                ts = ctx.random_ts(burst_ts, cfg.f1_burst_window_hours)
                ctx.add_row(uid, item, 5, ts, "Group", "filler", group_id)


def inject_group_camouflage(ctx: InjectionContext):
    """Gruppo con camouflage. Rilevato da: FraudAR, GSDB."""
    cfg = ctx.config
    for g_idx in range(cfg.f1_burst_groups):
        ctx._log(f'[GROUP camouflage] group {g_idx}')
        fakeusers = random.randint(10, 21)
        n_it      = random.randint(5, 8)
        group_id  = f"group_camouflage_{g_idx}"
        targets   = ctx.sample_with_overlap(n_it, contested_ratio=0.0, mode='push')
        ctx.used_items.update(
            i for i in targets if i not in ctx.contested_items
        )
        burst_ts = ctx.random_ts()
        for _ in range(fakeusers):
            uid = ctx.new_user()
            for item in targets:
                ts = ctx.random_ts(burst_ts)
                ctx.add_row(uid, item, 5, ts, "Group", "target", group_id)
            noise = ctx.sample_with_overlap(
                random.randint(3, 8), mode='popular',contested_ratio=0.25,
            )
            noise += ctx.sample_with_overlap(
                random.randint(3, 8), mode='push',contested_ratio=0.25,
            )
            noise += ctx.sample_with_overlap(
                random.randint(3, 8), mode='top',contested_ratio=0.25,
            )
            ctx.used_items.update(
                i for i in noise if i not in ctx.contested_items
            )
            for item in noise:
                ts   = ctx.random_ts()
                mean = ctx.item_means.get(item, 3.0)
                ctx.add_row(
                    uid, item,
                    int(round(mean + np.random.normal(0, 0.8))),
                    ts, "Group", "filler", group_id
                )


def inject_group_badnwagon(ctx: InjectionContext):
    """Burst ritardato. Rilevato da: GSDB."""
    """Burst puro. Rilevato da: FraudAR, GSDB."""
    cfg = ctx.config
    for g_idx in range(cfg.f1_burst_groups):
        ctx._log(f'[GROUP Bandwagon] group {g_idx}')
        group_id   = f"group_badnwagon_{g_idx}"
        fakeusers  = random.randint(10, 21)

        targets    = ctx.sample_with_overlap(
            n=np.random.randint(5, 10), contested_ratio=0.0, mode='push'
        )
        filler    = ctx.sample_with_overlap(
            n=np.random.randint(10, 20), contested_ratio=0.25, mode='popular'
        )
        filler    += ctx.sample_with_overlap(
            n=np.random.randint(5, 15), contested_ratio=0.25, mode='top'
        )
        ctx.used_items.update(
            i for i in targets if i not in ctx.contested_items
        )
        burst_ts = ctx.random_ts()
        for _ in range(fakeusers):
            uid = ctx.new_user()
            for item in targets:
                ts = ctx.random_ts(burst_ts, cfg.f1_burst_window_hours)
                ctx.add_row(uid, item, 5, ts, "Group", "target", group_id)
            for item in filler:
                ts = ctx.random_ts(burst_ts, cfg.f1_burst_window_hours)
                ctx.add_row(uid, item, 5, ts, "Group", "filler", group_id)




def inject_shilling_high_deg(ctx: InjectionContext):
    """Gruppo eterogeneo. Rilevato da: FraudAR."""
    cfg = ctx.config
    for g_idx in range(cfg.f2_high_dege):
        ctx._log(f'[SHILLING high degree] group {g_idx}')
        group_id   = f"shilling_high_deg_{g_idx}"
        fakeusers  = random.randint(50, 100)

        targets    = ctx.sample_with_overlap(
            n=1, contested_ratio=0.0, mode='push'
        )

        ctx.used_items.update(
            i for i in targets if i not in ctx.contested_items
        )
        burst_ts = ctx.random_ts()
        for _ in range(fakeusers):
            filler = ctx.sample_with_overlap(
                n=np.random.randint(1, 5), contested_ratio=0.25, mode='popular'
            )
            filler += ctx.sample_with_overlap(
                n=np.random.randint(1, 5), contested_ratio=0.25, mode='push'
            )
            filler += ctx.sample_with_overlap(
                n=np.random.randint(1, 5), contested_ratio=0.25, mode='top'
            )

            uid = ctx.new_user()
            for item in targets:
                ts = ctx.random_ts(burst_ts, cfg.f1_burst_window_hours)
                ctx.add_row(uid, item, 5, ts, "Shilling", "target", group_id)
            for item in filler:
                ts = ctx.random_ts(burst_ts, cfg.f1_burst_window_hours)
                ctx.add_row(uid, item, 5, ts, "Shilling", "filler", group_id)

def inject_shilling_bridge_injection(ctx: InjectionContext):
    """Gruppo eterogeneo. Rilevato da: FraudAR."""
    cfg = ctx.config
    for g_idx in range(cfg.f2_bridge):
        ctx._log(f'[SHILLING bridge] group {g_idx}')
        group_id   = f"shilling_bridge_{g_idx}"
        fakeusers  = 1

        targets    = ctx.sample_with_overlap(
            n=1, contested_ratio=0.0, mode='push'
        )
        filler    = ctx.sample_with_overlap(
            n=np.random.randint(2, 5), contested_ratio=0.25, mode='top'
        )

        ctx.used_items.update(
            i for i in targets if i not in ctx.contested_items
        )
        burst_ts = ctx.random_ts()
        for _ in range(fakeusers):
            uid = ctx.new_user()
            for item in targets:
                ts = ctx.random_ts(burst_ts, cfg.f1_burst_window_hours)
                ctx.add_row(uid, item, 5, ts, "Shilling", "target", group_id)
            for item in filler:
                ts = ctx.random_ts(burst_ts, cfg.f1_burst_window_hours)
                ctx.add_row(uid, item, 5, ts, "Shilling", "filler", group_id)

def inject_shilling_hijacking(ctx: InjectionContext):
    """Gruppo eterogeneo. Rilevato da: FraudAR."""
    cfg = ctx.config

    df = ctx.df
    user_counts = df.groupby("user_id")["item_id"].nunique()
    # utenti con tra 6 e 10 item
    valid_users = user_counts[(user_counts >= 6) & (user_counts <= 10)].index
    # subset del dataframe con quei user
    df_filtered = df[df["user_id"].isin(valid_users)]
    # (opzionale) lista user -> items
    user_items = df_filtered.groupby("user_id")["item_id"].apply(list)


    for g_idx in range(cfg.f2_shilling):
        ctx._log(f'[SHILLING Hijacking] group {g_idx}')
        group_id   = f"shilling_hijacking_{g_idx}"
        fakeusers  = 1

        # users to emulate

        uid = user_items.sample().index[0]
        filler = user_items.loc[uid]

        targets    = ctx.sample_with_overlap(
            n=np.random.randint(1, 2), contested_ratio=0.25, mode='push'
        )

        ctx.used_items.update(
            i for i in targets if i not in ctx.contested_items
        )
        burst_ts = ctx.random_ts()
        for _ in range(fakeusers):
            uid = ctx.new_user()
            for item in targets:
                ts = ctx.random_ts(burst_ts, cfg.f1_burst_window_hours)
                ctx.add_row(uid, item, 5, ts, "Shilling", "target", group_id)
            for item in filler:
                ts = ctx.random_ts(burst_ts, cfg.f1_burst_window_hours)
                ctx.add_row(uid, item, 5, ts, "Shilling", "filler", group_id)


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

from src.utils import *

def run_injection(
    input_csv:  str,
    output_dir: str = ".",
    config:     InjectionConfig = None,
    verbose:    bool = True, overlap: float = 0.0
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Esegue l'injection completa su un dataset di recensioni.

    Parametri
    ---------
    input_csv  : path al CSV originale [user_id, item_id, rating, timestamp]
    output_dir : cartella dove salvare i due CSV di output
    config     : InjectionConfig — usa default se None
    verbose    : stampa progress (default True, metti False per velocizzare)
    """
    if config is None:
        config = InjectionConfig()

    print("Caricamento dataset...", flush=True)
    df = pd.read_csv(input_csv)

    print("STATS ORIGINAL DATASET")
    print(f"users: {df['user_id'].nunique()}")
    print(f"items: {df['item_id'].nunique()}")
    print(f"interactions: {len(df)}")

    assert {"user_id","item_id","rating","timestamp"}.issubset(df.columns), \
        "Il CSV deve avere colonne: user_id, item_id, rating, timestamp"

    ctx = InjectionContext(df, config, overlap = overlap, verbose=verbose)

    print("F1 — Gruppi coordinati...", flush=True)
    inject_group_dense_cluster(ctx) # cluster con tutta nicchia
    inject_group_badnwagon(ctx)  # target + item popolari filler
    inject_group_camouflage(ctx) # simulano utenti reali

    print("F2 — Spammer individuali...", flush=True)
    inject_shilling_bridge_injection(ctx) # collega due cluster
    inject_shilling_high_deg(ctx) # stella
    inject_shilling_hijacking(ctx) # si inserisce in nicchia

    dirty_df, injected_df = ctx.get_dataframes()

    os.makedirs(output_dir, exist_ok=True)
    fmt = '%Y-%m-%d %H:%M:%S.%f'
    dirty_df["timestamp"]    = pd.to_datetime(dirty_df["timestamp"]).dt.strftime(fmt).str[:-3]
    injected_df["timestamp"] = pd.to_datetime(injected_df["timestamp"]).dt.strftime(fmt).str[:-3]
    if overlap == 0.0:
        over = 'no_overlap'
    elif overlap == 0.25:
        over = 'low_over'
    elif overlap == 0.50:
        over = 'mid_over'
    elif overlap == 0.75:
        over = 'high_over'
    elif overlap == 0.9:
        over = 'super_high_over'
    dirty_df.drop_duplicates(subset=['user_id','item_id']).to_csv(f"{output_dir}/dataset_dirty_new.csv",    index=False)
    injected_df.drop_duplicates(subset=['user_id','item_id']).to_csv(f"{output_dir}/injected_noise_new.csv", index=False)

    spam  = injected_df[injected_df["defect_type"] == "spam"]
    legit = injected_df[injected_df["defect_type"] == "legit"]

    print("\n── Riepilogo ────────────────────────────────────────")
    summary = (
        spam.groupby(["family","casistica"])["user_id"]
        .nunique()
        .reset_index()
        .rename(columns={"user_id": "fake_users"})
    )
    print(summary.to_string(index=False))
    print(f"\nFake users totali  : {spam['user_id'].nunique()}"
          f" ({spam['user_id'].nunique() / dirty_df['user_id'].nunique() * 100:.1f}%)")
    print(f"Items totali       : {spam['item_id'].nunique()}"
          f" ({spam['item_id'].nunique() / dirty_df['item_id'].nunique() * 100:.1f}%)")
    print(f"Righe dirty totali : {len(injected_df) / len(dirty_df) * 100:.1f}%")
    print(f"Output in          : {output_dir}/")

    return dirty_df, injected_df