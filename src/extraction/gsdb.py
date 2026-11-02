# """
# GSDB - Group Spam Detection algorithm based on review Burst
# =============================================================
# Versione OTTIMIZZATA per dataset su scala ~4M interazioni
# (es. data/original/Office_Products/Office_Products.csv,
#  colonne: user_id, item_id, rating, timestamp, helpful_vote).
#
# Riferimento paper:
# Ji et al., "A burst-based unsupervised method for detecting review
# spammer groups", Information Sciences 536 (2020) 454-469.
#
# FILOSOFIA DELL'OTTIMIZZAZIONE (rispetto alla v1)
# -------------------------------------------------
# La versione originale ricalcolava indicatori individuali/di gruppo
# filtrando l'intero DataFrame ad ogni chiamata (loop Python su
# reviewer/prodotto) -> complessità O(n^2) o peggio su 4M righe.
#
# Qui invece:
#   1) Tutti gli indicatori INDIVIDUALI (Eq. 1-7, Sez. 3.1) sono calcolati
#      UNA SOLA VOLTA per l'intero dataset con operazioni pandas vettorizzate
#      (groupby + agg + diff), invece che riga per riga.
#   2) Le somme/conteggi per prodotto (necessari per le medie "escludendo un
#      insieme di reviewer", Eq. 2 e Eq. 10-11) sono precalcolati una volta
#      come Serie indicizzate per product_id: per ottenere la media
#      "escludendo un gruppo" basta una sottrazione O(|gruppo|), non una
#      scansione di tutto il dataset.
#   3) Algoritmo 2 (Target Product Location, Eq. 20-22) è completamente
#      vettorizzato: nessun loop Python su prodotti.
#   4) Algoritmo 3 (burst detection via KDE) viene eseguito SOLO sui prodotti
#      target (di norma una piccola frazione del totale dopo il filtro
#      Eq. 22) e può essere parallelizzato su più processi.
#   5) Algoritmo 4 (purificazione/classificazione) usa lookup O(1) sugli
#      indicatori individuali precalcolati al punto (1), invece di
#      richiamare le formule reviewer per reviewer.
#   6) groupby("reviewer_id") e groupby("product_id") vengono creati una sola
#      volta e riusati per estrarre sottoinsiemi di righe di un gruppo/
#      prodotto (via .get_group), evitando filtri isin() ripetuti su 4M righe.
#
# Dipendenze: numpy, pandas, scipy
# """
#
# from __future__ import annotations
# import time
# import numpy as np
# import pandas as pd
# from scipy.stats import gaussian_kde
# from concurrent.futures import ProcessPoolExecutor, as_completed
#
#
# # =============================================================================
# # CARICAMENTO DATI
# # =============================================================================
#
# def load_reviews_csv(path: str,
#                       user_col="user_id", item_col="item_id",
#                       rating_col="rating", ts_col="timestamp") -> pd.DataFrame:
#     """
#     Carica un CSV nel formato Amazon Reviews (user_id,item_id,rating,
#     timestamp,helpful_vote) e lo normalizza nel formato interno
#     [reviewer_id, product_id, rating, date] usato dal paper.
#
#     Ottimizzazioni memoria per 4M+ righe:
#       - dtype categorico per gli ID (risparmia GB su stringhe ripetute)
#       - rating come float32 (i valori sono 1-5, Eq. 1 usa il rating scale)
#       - helpful_vote NON viene caricato: non è usato da nessuna formula
#         del paper (Sez. 3), quindi lo scartiamo per risparmiare RAM/IO.
#       - auto-detect dell'unità del timestamp (secondi vs millisecondi),
#         perché i dataset "Amazon Reviews" più recenti usano i millisecondi.
#     """
#     usecols = [user_col, item_col, rating_col, ts_col]
#     dtype = {user_col: "category", item_col: "category", rating_col: "float32"}
#     dtype = {user_col: "string", item_col: "string", rating_col: "float32"}
#
#
#     df = pd.read_csv(path, usecols=usecols, dtype=dtype)
#     try:
#         df[ts_col] = pd.to_datetime(
#             df[ts_col],
#             format="%Y-%m-%d %H:%M:%S.%f"
#         )
#     except ValueError:
#         df[ts_col] = pd.to_datetime(
#             df[ts_col].astype(float),
#             unit="s"
#         )
#
#     df[ts_col] = df[ts_col].astype("int64") // 1_000_000
#     # auto-detect secondi vs millisecondi
#     sample_ts = pd.to_numeric(df[ts_col].iloc[:1000]).median()
#     unit = "ms" if sample_ts > 10**12 else "s"
#     date = pd.to_datetime(df[ts_col], unit=unit)
#
#     out = pd.DataFrame({
#         "reviewer_id": df[user_col].values,
#         "product_id": df[item_col].values,
#         "rating": df[rating_col].astype("float32").values,
#         "date": date.values,
#     })
#     out["reviewer_id"] = out["reviewer_id"].astype("category")
#     out["product_id"] = out["product_id"].astype("category")
#
#     out["reviewer_id"] = out["reviewer_id"].astype(str)
#     out["product_id"] = out["product_id"].astype(str)
#     return out.sort_values("date").reset_index(drop=True)
#
#
# # =============================================================================
# # CLASSE PRINCIPALE
# # =============================================================================
#
# class GSDBFast:
#
#     def __init__(self, reviews: pd.DataFrame, rating_scale: int = 5, verbose: bool = True):
#         """
#         reviews: DataFrame con colonne [reviewer_id, product_id, rating, date].
#         Al momento della costruzione vengono PRECALCOLATI (una sola volta,
#         in modo vettorizzato) tutti gli indicatori individuali (Sez. 3.1)
#         e le statistiche aggregate per prodotto necessarie a quasi tutte
#         le altre formule del paper.
#         """
#         self.verbose = verbose
#         t0 = time.time()
#
#         df = reviews.copy()
#         df["date"] = pd.to_datetime(df["date"])
#         self.df = df.sort_values("date").reset_index(drop=True)
#
#         self.rmax = rating_scale     # usato in Eq. 1, 2, 10-11, 16, 20 (rating scale a 5 stelle)
#         self.rmin = 1
#
#         # t_data: durata totale del dataset in giorni -> denominatore di AD, Eq. (5)
#         self.t_data = max(1, (self.df["date"].max() - self.df["date"].min()).days)
#
#         # groupby riusabili (creati una volta, evitano filtri ripetuti su 4M righe)
#         self._reviewer_gb = self.df.groupby("reviewer_id", observed=True)
#         self._product_gb = self.df.groupby("product_id", observed=True)
#
#         self._log(f"[init] {len(self.df):,} righe caricate in {time.time()-t0:.1f}s")
#
#         self._precompute_product_stats()
#         self._precompute_individual_indicators()
#
#     def _log(self, msg):
#         if self.verbose:
#             print(msg)
#
#     # -------------------------------------------------------------------
#     # Estrazione efficiente delle righe di un gruppo di reviewer
#     # -------------------------------------------------------------------
#     def _group_reviews(self, group_members) -> pd.DataFrame:
#         """
#         Ritorna le righe di un insieme (piccolo) di reviewer usando i
#         groupby precalcolati (get_group), invece di uno scan isin() su
#         tutte le 4M righe. Costo ~ O(somma delle recensioni dei membri),
#         non O(n_totale).
#         """
#         keys = self._reviewer_gb.groups.keys()
#         parts = [self._reviewer_gb.get_group(a) for a in group_members if a in keys]
#         if not parts:
#             return self.df.iloc[0:0]
#         return pd.concat(parts, axis=0)
#
#     # =====================================================================
#     # PRECALCOLO 1: statistiche per prodotto
#     # Servono a: Eq. (2) e Eq. (10)-(11) [media "escludendo" un reviewer/
#     # gruppo, calcolata per sottrazione O(1) invece di rifiltrare il df],
#     # ed Eq. (20)-(22) [Target Product Location].
#     # =====================================================================
#     def _precompute_product_stats(self):
#         t0 = time.time()
#         g = self._product_gb["rating"]
#
#         # Σ rating per prodotto e conteggio -> usati per calcolare "media
#         # escludendo un sottoinsieme di reviewer" con una sola sottrazione:
#         #   r*_p(escludendo S) = (product_sum[p] - Σ_{a in S} r_ap) / (product_count[p] - |ratings di S su p|)
#         # Questa è l'implementazione efficiente della definizione di r*_p
#         # usata nelle Eq. (2) [singolo reviewer] e Eq. (10)-(11) [gruppo].
#         self.product_sum = g.sum()      # Σ_p r_p   -> Eq. (2), Eq. (10)
#         self.product_count = g.size()   # |Ra_p|    -> Eq. (2), Eq. (10)
#
#         # conteggio rating estremi per prodotto -> Eq. (20), Sext(p)
#         rating = self.df["rating"].values
#         prod = self.df["product_id"].values
#         self.product_n5 = self.df.loc[rating == self.rmax].groupby("product_id", observed=True).size() \
#                                 .reindex(self.product_count.index, fill_value=0)
#         self.product_n1 = self.df.loc[rating == self.rmin].groupby("product_id", observed=True).size() \
#                                 .reindex(self.product_count.index, fill_value=0)
#
#         self._log(f"[precompute] statistiche prodotto in {time.time()-t0:.1f}s "
#                    f"({len(self.product_count):,} prodotti)")
#
#     # =====================================================================
#     # PRECALCOLO 2: indicatori individuali (Sez. 3.1, Eq. 1-7)
#     # Calcolati UNA VOLTA per tutti i reviewer, in modo vettorizzato.
#     # =====================================================================
#     def _precompute_individual_indicators(self, q_days: int = 28):
#         t0 = time.time()
#         df = self.df
#
#         total_counts = self._reviewer_gb.size()  # |Ra| per ogni reviewer a
#
#         # --- EXR(a): Eq. (1) -----------------------------------------------
#         # Ratio of Extreme Rating = |{r in Ra : r in {1,5}}| / |Ra|
#         extreme_mask = df["rating"].isin([self.rmin, self.rmax])
#         extreme_counts = df.loc[extreme_mask].groupby("reviewer_id", observed=True).size()
#         EXR = (extreme_counts.reindex(total_counts.index, fill_value=0) / total_counts)
#
#         # --- RD(a): Eq. (2) --------------------------------------------------
#         # RD(a) = avg_{p in Pa} |r_ap - r*_p| / 4
#         # r*_p (media del prodotto p ESCLUDENDO la riga corrente) calcolata
#         # per sottrazione dalle somme precalcolate del prodotto (vedi sopra),
#         # invece di rifiltrare il dataset per ogni riga -> O(n) totale invece
#         # di O(n^2).
#         prod_sum_row = df["product_id"].map(self.product_sum).astype("float64")
#         prod_cnt_row = df["product_id"].map(self.product_count).astype("float64")
#         denom = (prod_cnt_row - 1)
#         r_star = (prod_sum_row - df["rating"]) / denom.replace(0, np.nan)
#         # fallback (Eq. 2, caso limite non specificato dal paper): se il
#         # reviewer è l'UNICO che ha valutato il prodotto, non esistono
#         # "altri" reviewer -> usiamo la media del prodotto incluso lui stesso.
#         r_star = r_star.fillna(prod_sum_row / prod_cnt_row)
#         dev = (df["rating"] - r_star).abs() / (self.rmax - self.rmin)
#         RD = dev.groupby(df["reviewer_id"], observed=True).mean()
#         RD = RD.reindex(total_counts.index, fill_value=0.0)
#
#         # --- MRO(a): Eq. (3) -------------------------------------------------
#         # MaxRev(a) / max_a(MaxRev(a))
#         per_reviewer_day = df.groupby(
#             ["reviewer_id", df["date"].dt.floor("D")], observed=True
#         ).size()
#         max_rev_per_reviewer = per_reviewer_day.groupby(level=0, observed=True).max()
#         max_rev_per_reviewer = max_rev_per_reviewer.reindex(total_counts.index, fill_value=0)
#         max_rev_all = float(max_rev_per_reviewer.max()) if len(max_rev_per_reviewer) else 1.0
#         MRO = max_rev_per_reviewer / max(max_rev_all, 1.0)
#
#         # --- RTI(a) e ATR(a): Eq. (4) e Eq. (6) ------------------------------
#         # Richiedono le date ordinate per reviewer -> un solo sort globale
#         # (già fatto in __init__), poi diff() vettorizzato per gruppo.
#         df_sorted = df  # self.df è già ordinato per date, ma diff() va fatto
#         # raggruppando per reviewer preservando l'ordine temporale:
#         df_by_rev = df.sort_values(["reviewer_id", "date"])
#         gap_fwd = df_by_rev.groupby("reviewer_id", observed=True)["date"].diff().dt.days
#         gap_bwd = df_by_rev.groupby("reviewer_id", observed=True)["date"].diff(-1).dt.days.abs()
#
#         # RTI: Eq. (4) - frazione di intervalli ADIACENTI (solo "avanti")
#         # con gap <= q_days, su (m-1) intervalli totali.
#         within_q_fwd = (gap_fwd <= q_days) & gap_fwd.notna()
#         rti_counts = within_q_fwd.groupby(df_by_rev["reviewer_id"], observed=True).sum()
#         rti_counts = rti_counts.reindex(total_counts.index, fill_value=0)
#         rti_denom = (total_counts - 1).clip(lower=1)
#         RTI = rti_counts / rti_denom
#         RTI[total_counts < 2] = 0.0  # con < 2 review non esiste alcun intervallo
#
#         # ATR: Eq. (6) - frazione di review che cadono in un "intervallo
#         # attivo", cioè che hanno un'altra review dello stesso reviewer
#         # entro q_days PRIMA o DOPO (usiamo sia gap_fwd che gap_bwd).
#         active = ((gap_fwd <= q_days) & gap_fwd.notna()) | ((gap_bwd <= q_days) & gap_bwd.notna())
#         atr_counts = active.groupby(df_by_rev["reviewer_id"], observed=True).sum()
#         atr_counts = atr_counts.reindex(total_counts.index, fill_value=0)
#         ATR = atr_counts / total_counts
#
#         # min_gap per reviewer, riutilizzato più avanti in S_avg (Eq. 21)
#         # per individuare i "truthful reviewer" (Feng et al. [12]).
#         self._min_gap_per_reviewer = gap_fwd.groupby(df_by_rev["reviewer_id"], observed=True).min()
#
#         # --- AD(a): Eq. (5) ---------------------------------------------------
#         date_min = self._reviewer_gb["date"].min()
#         date_max = self._reviewer_gb["date"].max()
#         span_days = (date_max - date_min).dt.days
#         AD = 1.0 - (span_days / self.t_data)
#
#         # --- ISS(a): Eq. (7) = media dei 6 indicatori sopra -------------------
#         self.individual = pd.DataFrame({
#             "EXR": EXR, "RD": RD, "MRO": MRO, "RTI": RTI, "AD": AD, "ATR": ATR,
#         }).reindex(total_counts.index).fillna(0.0)
#         self.individual["ISS"] = self.individual[["EXR", "RD", "MRO", "RTI", "AD", "ATR"]].mean(axis=1)
#
#         self.total_counts = total_counts
#
#         self._log(f"[precompute] indicatori individuali (Eq.1-7) per "
#                    f"{len(self.individual):,} reviewer in {time.time()-t0:.1f}s")
#
#     @staticmethod
#     def _plain_index(idx: pd.Index) -> pd.Index:
#         """Converte un CategoricalIndex in un Index 'semplice' per evitare il
#         bug di pandas 'Buffer dtype mismatch' quando si fa reindex/align tra
#         CategoricalIndex con numero di categorie diverso (codici int8 vs int16)."""
#         if isinstance(idx, pd.CategoricalIndex):
#             return idx.astype(idx.categories.dtype)
#         return idx
#     # =====================================================================
#     # ALGORITMO 2: TargetProductLocation (Eq. 20-22) - VETTORIZZATO
#     # =====================================================================
#     def target_product_location(self, delta_tp: float = 0.1, k: float = 0.5,
#                                  min_reviews_truthful: int = 10):
#         """
#         Restituisce l'elenco dei prodotti con STP(p) > delta_tp (Eq. 22).
#         Nessun loop Python sui prodotti: tutto calcolato con operazioni
#         pandas vettoriali su Serie indicizzate per product_id.
#         """
#         t0 = time.time()
#         df = self.df
#
#         # --- Sext(p): Eq. (20) ----------------------------------------------
#         n1 = self.product_n1.astype(float)
#         n5 = self.product_n5.astype(float)
#         mx = np.maximum(n1, n5)
#         Sext = pd.Series(np.where(mx > 0, np.minimum(n1, n5) / mx.replace(0, np.nan), 0.0),
#                           index=self.product_count.index).fillna(0.0)
#
#         # --- Savg(p): Eq. (21) -----------------------------------------------
#         # SR = single-time reviewer (|Ra| == 1)
#         # TR = truthful reviewer (|Ra| > min_reviews_truthful E MAI due
#         #      review a distanza <= 1 giorno, Feng et al. [12])
#         SR_reviewers = self.total_counts[self.total_counts == 1].index
#         TR_candidates = self.total_counts[self.total_counts > min_reviews_truthful].index
#         min_gap = self._min_gap_per_reviewer.reindex(TR_candidates)
#         # se min_gap è NaN vuol dire reviewer con 1 sola review: escluso perché
#         # non può stare in TR_candidates (count>10 implica >=2 review, min_gap definito)
#         TR_reviewers = TR_candidates[min_gap > 1]
#
#         is_sr = df["reviewer_id"].isin(SR_reviewers)
#         is_tr = df["reviewer_id"].isin(TR_reviewers)
#
#         sr_avg = df.loc[is_sr].groupby("product_id", observed=True)["rating"].mean()
#         tr_avg = df.loc[is_tr].groupby("product_id", observed=True)["rating"].mean()
#         sr_avg = sr_avg.reindex(self.product_count.index)
#         tr_avg = tr_avg.reindex(self.product_count.index)
#
#
#         # fallback documentato: se per un prodotto non esistono review SR o TR,
#         # il paper non specifica il comportamento -> trattiamo Savg come 0
#         # per quel prodotto (indicatore non applicabile), NON come prodotto
#         # automaticamente sospetto.
#         Savg = ((sr_avg - tr_avg).abs() / (self.rmax - self.rmin)).fillna(0.0)
#
#         # --- STP(p): Eq. (22) --------------------------------------------------
#         STP = k * Savg + (1 - k) * Sext
#         target_products = STP[STP > delta_tp].index.tolist()
#
#         self._log(f"[Algoritmo 2] {len(target_products):,} prodotti target "
#                    f"(su {len(STP):,}) in {time.time()-t0:.1f}s")
#         return target_products
#
#     # =====================================================================
#     # ALGORITMO 3: CandidateGroupGeneration (burst via KDE)
#     # Eseguito SOLO sui prodotti target (già filtrati da Algoritmo 2),
#     # quindi il costo totale scala con #target_products, non con 4M righe.
#     # =====================================================================
#     def candidate_group_generation(self, product_id, isize_days: int = 7):
#         rows = self._product_gb.get_group(product_id).sort_values("date")
#         if len(rows) < 3:
#             return []
#
#         t0_date = rows["date"].min()
#         dur = (rows["date"].max() - t0_date).days
#         if dur == 0:
#             return [frozenset(rows["reviewer_id"])]
#
#         x = ((rows["date"] - t0_date).dt.total_seconds().values / 86400.0 / dur)
#
#         k = max(2, dur // isize_days)      # numero di intervalli (ISIZE, Sez. 4.2)
#         avg_rev = len(rows) / k            # avg_rev, usato per scartare picchi non significativi (Algoritmo 3, riga 12)
#
#         try:
#             kde = gaussian_kde(x)
#         except np.linalg.LinAlgError:
#             return [frozenset(rows["reviewer_id"])]
#
#         grid = np.linspace(0, 1, 500)
#         density = kde(grid)                # KDE(x), Sez. 4.2 (kernel Gaussiano)
#
#         d1 = np.gradient(density, grid)
#         peak_idx = np.where((d1[:-1] > 0) & (d1[1:] <= 0))[0]  # punti di picco (derivata = 0, Algoritmo 3 riga 11)
#
#         groups = []
#         seen = set()
#         for idx in peak_idx:
#             x_peak = grid[idx]
#             t_peak = t0_date + pd.Timedelta(days=float(x_peak * dur))
#
#             lo = t_peak - pd.Timedelta(days=3)
#             hi = t_peak + pd.Timedelta(days=3)
#             window_rows = rows[(rows["date"] >= lo) & (rows["date"] <= hi)]
#
#             if len(window_rows) <= 1 or len(window_rows) <= avg_rev:
#                 continue  # scarta picchi non significativi (Algoritmo 3, righe 14-15)
#
#             group = frozenset(window_rows["reviewer_id"])
#             if group not in seen:
#                 seen.add(group)
#                 groups.append(group)
#
#         return groups
#
#     # =====================================================================
#     # ALGORITMO 4: GroupPurification&Classification (Eq. 7-19)
#     # =====================================================================
#     def _purify_group(self, group, delta_i: float) -> frozenset:
#         """Rimuove i membri con ISS(a) < delta_i (Eq. 7), via lookup O(1)."""
#         iss = self.individual["ISS"]
#         return frozenset(a for a in group if iss.get(a, 0.0) >= delta_i)
#
#     # --- indicatori di GRUPPO (Sez. 3.2, Eq. 8-19) --------------------------
#
#     def GTW(self, group_members, theta_days: int = 30) -> float:
#         """Eq. (8)-(9): Group Average Time Window."""
#         rows = self._group_reviews(group_members)
#         if rows.empty:
#             return 0.0
#         tw_values = []
#         for p, sub in rows.groupby("product_id", observed=True):
#             if len(sub) < 2:
#                 sd_p = 0.0
#             else:
#                 epoch_days = sub["date"].astype("int64") // 10**9 // 86400
#                 sd_p = float(np.std(epoch_days.values))
#             tw_values.append(1.0 - sd_p / theta_days if sd_p <= theta_days else 0.0)
#         return float(np.mean(tw_values)) if tw_values else 0.0
#
#     def GRD(self, group_members) -> float:
#         """
#         Eq. (10)-(11): Group Rating Deviation.
#         r*_p (media escludendo il gruppo) calcolata per sottrazione dalle
#         somme/conteggi di prodotto precalcolati, non rifiltrando le 4M righe.
#         """
#         rows = self._group_reviews(group_members)
#         if rows.empty:
#             return 0.0
#         agg = rows.groupby("product_id", observed=True)["rating"].agg(["sum", "count"])
#         rd_vals = []
#         for p, s, c in agg.itertuples(index=True, name=None):
#             total_sum = self.product_sum.get(p, s)
#             total_cnt = self.product_count.get(p, c)
#             remaining_cnt = total_cnt - c
#             if remaining_cnt > 0:
#                 r_star = (total_sum - s) / remaining_cnt
#             else:
#                 # fallback: il gruppo copre TUTTE le review del prodotto ->
#                 # non esistono "altri" reviewer, usiamo la media del prodotto
#                 # incluso il gruppo stesso (stesso fallback di Eq. 2).
#                 r_star = total_sum / total_cnt
#             member_ratings = rows.loc[rows["product_id"] == p, "rating"]
#             rd_vals.append(float(((member_ratings - r_star).abs() / (self.rmax - self.rmin)).mean()))
#         return float(np.mean(rd_vals)) if rd_vals else 0.0
#
#     def GS(self, group_members) -> float:
#         """Eq. (12): Group Size (funzione logistica centrata su size=3)."""
#         size = len(group_members)
#         return float(1.0 / (1.0 + np.exp(-(size - 3))))
#
#     def GRT(self, group_members) -> float:
#         """Eq. (13): Group Review Tightness = |Vg| / (|Rg| * |Pg|)."""
#         rows = self._group_reviews(group_members)
#         n_reviews = len(rows)
#         n_reviewers = len(set(group_members))
#         n_products = rows["product_id"].nunique()
#         if n_reviewers == 0 or n_products == 0:
#             return 0.0
#         return n_reviews / (n_reviewers * n_products)
#
#     def GOR(self, group_members, threshold: int = 5) -> float:
#         """Eq. (14)-(15): Group One Day Reviews."""
#         rows = self._group_reviews(group_members)
#         if rows.empty:
#             return 0.0
#         scores = []
#         for a, sub in rows.groupby("reviewer_id", observed=True):
#             per_day = sub.groupby(sub["date"].dt.floor("D")).size()
#             f_gor = (per_day > threshold).astype(int)
#             scores.append(f_gor.sum() / len(per_day))
#         return float(np.mean(scores)) if scores else 0.0
#
#     def GER(self, group_members) -> float:
#         """Eq. (16): Group Extreme Rating Ratio = media di EXR(a) sui membri (lookup precalcolato, Eq.1)."""
#         exr = self.individual["EXR"]
#         vals = [exr.get(a, 0.0) for a in group_members]
#         return float(np.mean(vals)) if vals else 0.0
#
#     def _co_active_windows(self, group_members, d_days: int = 7):
#         """Supporto per Eq. (17)-(18): individua le finestre in cui TUTTI i
#         membri del gruppo hanno postato entro d_days l'uno dall'altro."""
#         rows = self._group_reviews(group_members).sort_values("date")
#         if rows.empty:
#             return [], rows
#
#         windows = []
#         review_idx_in_window = set()
#         members_set = set(group_members)
#         for center in rows["date"].unique():
#             lo = center - pd.Timedelta(days=d_days)
#             hi = center + pd.Timedelta(days=d_days)
#             window_rows = rows[(rows["date"] >= lo) & (rows["date"] <= hi)]
#             if set(window_rows["reviewer_id"].unique()) == members_set:
#                 windows.append((lo, hi))
#                 review_idx_in_window.update(window_rows.index.tolist())
#
#         merged = []
#         for lo, hi in sorted(windows):
#             if merged and lo <= merged[-1][1]:
#                 merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
#             else:
#                 merged.append((lo, hi))
#
#         rows_in_window = rows.loc[rows.index.isin(review_idx_in_window)]
#         return merged, rows_in_window
#
#     def GCA(self, group_members, d_days: int = 7) -> float:
#         """Eq. (17): Group Co-Activeness."""
#         windows, _ = self._co_active_windows(group_members, d_days)
#         return float(1.0 / (1.0 + np.exp(-len(windows))))
#
#     def GCAR(self, group_members, d_days: int = 7) -> float:
#         """Eq. (18): Group Co-Active Review Ratio."""
#         _, rows_in_window = self._co_active_windows(group_members, d_days)
#         total_reviews = len(self._group_reviews(group_members))
#         if total_reviews == 0:
#             return 0.0
#         return len(rows_in_window) / total_reviews
#
#     def GSS(self, group_members, theta_days=30, d_days=7) -> float:
#         """Eq. (19): Group Spam Score = media degli 8 indicatori di gruppo."""
#         vals = [
#             self.GTW(group_members, theta_days),
#             self.GRD(group_members),
#             self.GS(group_members),
#             self.GRT(group_members),
#             self.GOR(group_members),
#             self.GER(group_members),
#             self.GCA(group_members, d_days),
#             self.GCAR(group_members, d_days),
#         ]
#         return float(np.mean(vals))
#
#     def group_purification_and_classification(self, candidate_groups,
#                                                 delta_i: float = 0.43,
#                                                 delta_g: float = 0.54,
#                                                 theta_days: int = 30,
#                                                 d_days: int = 7):
#         """Algoritmo 4: purifica (Eq.7 via lookup) e classifica (Eq.8-19)."""
#         spammer_groups = []
#         for group in candidate_groups:
#             purified = self._purify_group(group, delta_i)
#             if len(purified) > 2:
#                 gss = self.GSS(purified, theta_days, d_days)
#                 if gss >= delta_g:
#                     spammer_groups.append({"members": purified, "GSS": gss})
#         return spammer_groups
#
#     # =====================================================================
#     # ALGORITMO 1: pipeline completa (con parallelizzazione opzionale)
#     # =====================================================================
#     def run(self, delta_tp: float = 0.1, delta_i: float = 0.43, delta_g: float = 0.54,
#             k: float = 0.5, isize_days: int = 7, theta_days: int = 30, d_days: int = 7,
#             n_jobs: int = 1):
#         """
#         Algoritmo 1 (GSDB).
#         n_jobs > 1: parallelizza il loop sui prodotti target (Algoritmi 3+4),
#         che sono indipendenti tra loro - utile quando dopo il filtro
#         dell'Algoritmo 2 restano comunque molte migliaia di prodotti target.
#         """
#         t0 = time.time()
#         target_products = self.target_product_location(delta_tp, k)
#
#         results = {}
#         if n_jobs <= 1 or len(target_products) < 50:
#             for p in target_products:
#                 candidates = self.candidate_group_generation(p, isize_days)
#                 spammer_groups = self.group_purification_and_classification(
#                     candidates, delta_i, delta_g, theta_days, d_days
#                 )
#                 if spammer_groups:
#                     results[p] = spammer_groups
#         else:
#             # NB: ProcessPoolExecutor serializza self -> conviene solo se
#             # #target_products è grande abbastanza da coprire l'overhead di
#             # pickling del DataFrame. Per dataset molto grandi, valutare di
#             # passare ai worker solo le righe dei prodotti target invece
#             # dell'oggetto self completo.
#             with ProcessPoolExecutor(max_workers=n_jobs) as ex:
#                 futs = {
#                     ex.submit(_process_one_product, self, p, isize_days,
#                               delta_i, delta_g, theta_days, d_days): p
#                     for p in target_products
#                 }
#                 for fut in as_completed(futs):
#                     p = futs[fut]
#                     spammer_groups = fut.result()
#                     if spammer_groups:
#                         results[p] = spammer_groups
#
#         self._log(f"[Algoritmo 1] completato in {time.time()-t0:.1f}s: "
#                   f"{sum(len(v) for v in results.values())} gruppi spammer trovati "
#                   f"su {len(target_products)} prodotti target.")
#         return results
#
#
# def _process_one_product(gsdb: "GSDBFast", product_id, isize_days, delta_i, delta_g,
#                           theta_days, d_days):
#     """Funzione top-level (picklable) per l'esecuzione parallela di Algoritmo 3+4."""
#     candidates = gsdb.candidate_group_generation(product_id, isize_days)
#     return gsdb.group_purification_and_classification(
#         candidates, delta_i, delta_g, theta_days, d_days
#     )
#
#
# # =============================================================================
# # ESECUZIONE SU DATI REALI
# # =============================================================================
# if __name__ == "__main__":
#     datasets = [
#         'Baby_Products',
#         'Sports_and_Outdoors',
#         'Office_Products',
#         'Toys_and_Games',
#         'Pet_Supplies',
#         'Beauty_and_Personal_Care',
#         'Books',
#         'yelp'
#     ]
#
#     for dataset in datasets:
#         print('DATASET', dataset)
#
#
#         CSV_PATH = f"data/original/{dataset}/{dataset}.train.csv"
#         CSV_OUT_PATH = f"data/original/{dataset}/{dataset}.gsdb_new.csv"
#
#         print(f"Caricamento {CSV_PATH} ...")
#         reviews = load_reviews_csv(CSV_PATH)
#         print(f"Caricate {len(reviews):,} interazioni, "
#               f"{reviews['reviewer_id'].nunique():,} utenti, "
#               f"{reviews['product_id'].nunique():,} prodotti.")
#
#         gsdb = GSDBFast(reviews)
#         iss_values = gsdb.individual["ISS"]
#         print((iss_values < 0.43).mean())
#         # Soglie di default riprese da Sez. 5.2 del paper (dTP=0.1, dI=0.43, dG=0.54)
#         detected = gsdb.run(delta_tp=0.05, delta_i=0.43, delta_g=0.2, n_jobs=1)
#
#         # ---------------------------------------------------------------
#         # OUTPUT: un gruppo per riga, formato
#         # n_users  n_items  users  items  GTW ... IMC_coupling_avg
#         # ---------------------------------------------------------------
#         METRIC_COLS = [
#             "GTW", "GD", "GETF", "GSR", "GS", "GSUP",
#             "IRD_avg", "IRD_max", "IETF_avg", "IETF_max",
#             "IMC_avg", "IMC_max", "IMC_coupling_avg",
#         ]
#
#         # Unisco i gruppi con gli stessi membri trovati su prodotti diversi:
#         # items = insieme dei prodotti su cui il gruppo è stato rilevato
#         groups_map = {}  # frozenset(membri) -> set(prodotti)
#         for product, groups in detected.items():
#             for g in groups:
#                 groups_map.setdefault(frozenset(g["members"]), set()).add(product)
#
#         rows = []
#         for members, items in groups_map.items():
#             users_sorted = sorted(map(str, members))
#             items_sorted = sorted(map(str, items))
#             row = {
#                 "n_users": len(users_sorted),
#                 "n_items": len(items_sorted),
#                 "users": ",".join(users_sorted),
#                 "items": ",".join(items_sorted),
#             }
#             row.update({c: 0.0 for c in METRIC_COLS})
#             rows.append(row)
#
#         out_df = pd.DataFrame(rows, columns=["n_users", "n_items", "users", "items"] + METRIC_COLS)
#         out_df.sort_values(["n_users", "n_items"], ascending=[False, False], inplace=True)
#
#         # Separatore TAB: users/items contengono virgole, così non servono virgolette
#         out_df.to_csv(CSV_OUT_PATH, sep="\t", index=False, float_format="%.4f")
#
#         print(f"\nSalvato: {CSV_OUT_PATH}")
#         print(f"Gruppi salvati: {len(out_df):,}")
#
#
#         # rows = []
#         #
#         # for product, groups in detected.items():
#         #     print(f"\n  Prodotto: {product}")
#         #
#         #     for g in groups:
#         #         members = sorted(g["members"])
#         #
#         #         print(
#         #             f"    Membri ({len(members)}): {members[:10]}... "
#         #             f"| GSS = {g['GSS']:.3f}"
#         #         )
#         #
#         #         rows.append({
#         #             "product_id": product,
#         #             "group_size": len(members),
#         #             "gss": g["GSS"],
#         #             "members": "|".join(map(str, members))
#         #         })
#         #
#         # out_df = pd.DataFrame(rows)
#         # out_df.sort_values(
#         #     ["gss", "group_size"],
#         #     ascending=[False, False],
#         #     inplace=True
#         # )
#         #
#         # out_df.to_csv(CSV_OUT_PATH, index=False)
#         #
#         # print(f"\nSalvato: {CSV_OUT_PATH}")
#         # print(f"Gruppi salvati: {len(out_df):,}")


"""
GSDB - Group Spam Detection algorithm based on review Burst
=============================================================
Versione OTTIMIZZATA per dataset su scala ~4M interazioni
(es. data/original/Office_Products/Office_Products.csv,
 colonne: user_id, item_id, rating, timestamp, helpful_vote).

Riferimento paper:
Ji et al., "A burst-based unsupervised method for detecting review
spammer groups", Information Sciences 536 (2020) 454-469.

FILOSOFIA DELL'OTTIMIZZAZIONE (rispetto alla v1)
-------------------------------------------------
La versione originale ricalcolava indicatori individuali/di gruppo
filtrando l'intero DataFrame ad ogni chiamata (loop Python su
reviewer/prodotto) -> complessità O(n^2) o peggio su 4M righe.

Qui invece:
  1) Tutti gli indicatori INDIVIDUALI (Eq. 1-7, Sez. 3.1) sono calcolati
     UNA SOLA VOLTA per l'intero dataset con operazioni pandas vettorizzate
     (groupby + agg + diff), invece che riga per riga.
  2) Le somme/conteggi per prodotto (necessari per le medie "escludendo un
     insieme di reviewer", Eq. 2 e Eq. 10-11) sono precalcolati una volta
     come Serie indicizzate per product_id: per ottenere la media
     "escludendo un gruppo" basta una sottrazione O(|gruppo|), non una
     scansione di tutto il dataset.
  3) Algoritmo 2 (Target Product Location, Eq. 20-22) è completamente
     vettorizzato: nessun loop Python su prodotti.
  4) Algoritmo 3 (burst detection via KDE) viene eseguito SOLO sui prodotti
     target (di norma una piccola frazione del totale dopo il filtro
     Eq. 22) e può essere parallelizzato su più processi.
  5) Algoritmo 4 (purificazione/classificazione) usa lookup O(1) sugli
     indicatori individuali precalcolati al punto (1), invece di
     richiamare le formule reviewer per reviewer.
  6) groupby("reviewer_id") e groupby("product_id") vengono creati una sola
     volta e riusati per estrarre sottoinsiemi di righe di un gruppo/
     prodotto (via .get_group), evitando filtri isin() ripetuti su 4M righe.

OUTPUT: per ogni gruppo trovato vengono calcolate le feature di
Mukherjee, Liu, Glance (WWW 2012) con le stesse definizioni di
group_features.py (GTW, GD, GETF, GSR, GS, GSUP, IRD, IETF, IMC),
così i due file di output sono confrontabili colonna per colonna.
In più: popolarità (media, max, min) degli item e degli utenti del gruppo.

Dipendenze: numpy, pandas, scipy
"""

from __future__ import annotations
import time
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from concurrent.futures import ProcessPoolExecutor, as_completed


# =============================================================================
# CARICAMENTO DATI
# =============================================================================
def _parse_timestamp(ts: pd.Series) -> pd.Series:
    """Converte una colonna timestamp (epoch s/ms/us/ns oppure stringa data) in datetime64."""
    num = pd.to_numeric(ts, errors="coerce")
    if num.notna().mean() > 0.99:          # colonna numerica -> epoch
        med = float(num.dropna().median())
        if med > 1e17:   unit = "ns"
        elif med > 1e14: unit = "us"
        elif med > 1e11: unit = "ms"
        else:            unit = "s"
        return pd.to_datetime(num, unit=unit, errors="coerce")
    try:                                   # colonna testuale
        return pd.to_datetime(ts, format="%Y-%m-%d %H:%M:%S.%f")
    except (ValueError, TypeError):
        return pd.to_datetime(ts, errors="coerce")


def load_reviews_csv(path: str,
                      user_col="user_id", item_col="item_id",
                      rating_col="rating", ts_col="timestamp") -> pd.DataFrame:
    usecols = [user_col, item_col, rating_col, ts_col]
    dtype = {user_col: "string", item_col: "string", rating_col: "float32"}
    df = pd.read_csv(path, usecols=usecols, dtype=dtype)

    date = _parse_timestamp(df[ts_col])
    bad = date.isna()
    if bad.any():
        print(f"[load] scartate {int(bad.sum()):,} righe con timestamp non valido")
        df, date = df.loc[~bad], date.loc[~bad]

    out = pd.DataFrame({
        "reviewer_id": df[user_col].astype(str).values,
        "product_id": df[item_col].astype(str).values,
        "rating": df[rating_col].astype("float32").values,
        "date": date.values,
    })
    print(f"[load] date da {out['date'].min()} a {out['date'].max()}")
    return out.sort_values("date").reset_index(drop=True)

def load_reviews_csv_old(path: str,
                      user_col="user_id", item_col="item_id",
                      rating_col="rating", ts_col="timestamp") -> pd.DataFrame:
    """
    Carica un CSV nel formato Amazon Reviews (user_id,item_id,rating,
    timestamp,helpful_vote) e lo normalizza nel formato interno
    [reviewer_id, product_id, rating, date] usato dal paper.

    Ottimizzazioni memoria per 4M+ righe:
      - dtype categorico per gli ID (risparmia GB su stringhe ripetute)
      - rating come float32 (i valori sono 1-5, Eq. 1 usa il rating scale)
      - helpful_vote NON viene caricato: non è usato da nessuna formula
        del paper (Sez. 3), quindi lo scartiamo per risparmiare RAM/IO.
      - auto-detect dell'unità del timestamp (secondi vs millisecondi),
        perché i dataset "Amazon Reviews" più recenti usano i millisecondi.
    """
    usecols = [user_col, item_col, rating_col, ts_col]
    dtype = {user_col: "category", item_col: "category", rating_col: "float32"}
    dtype = {user_col: "string", item_col: "string", rating_col: "float32"}


    df = pd.read_csv(path, usecols=usecols, dtype=dtype)
    try:
        df[ts_col] = pd.to_datetime(
            df[ts_col],
            format="%Y-%m-%d %H:%M:%S.%f"
        )
    except ValueError:
        df[ts_col] = pd.to_datetime(
            df[ts_col].astype(float),
            unit="s"
        )

    df[ts_col] = df[ts_col].astype("int64") // 1_000_000
    # auto-detect secondi vs millisecondi
    sample_ts = pd.to_numeric(df[ts_col].iloc[:1000]).median()
    unit = "ms" if sample_ts > 10**12 else "s"
    date = pd.to_datetime(df[ts_col], unit=unit)

    out = pd.DataFrame({
        "reviewer_id": df[user_col].values,
        "product_id": df[item_col].values,
        "rating": df[rating_col].astype("float32").values,
        "date": date.values,
    })
    out["reviewer_id"] = out["reviewer_id"].astype("category")
    out["product_id"] = out["product_id"].astype("category")

    out["reviewer_id"] = out["reviewer_id"].astype(str)
    out["product_id"] = out["product_id"].astype(str)
    return out.sort_values("date").reset_index(drop=True)


# =============================================================================
# CLASSE PRINCIPALE
# =============================================================================

class GSDBFast:

    def __init__(self, reviews: pd.DataFrame, rating_scale: int = 5, verbose: bool = True):
        """
        reviews: DataFrame con colonne [reviewer_id, product_id, rating, date].
        Al momento della costruzione vengono PRECALCOLATI (una sola volta,
        in modo vettorizzato) tutti gli indicatori individuali (Sez. 3.1)
        e le statistiche aggregate per prodotto necessarie a quasi tutte
        le altre formule del paper.
        """
        self.verbose = verbose
        t0 = time.time()

        df = reviews.copy()
        df["date"] = pd.to_datetime(df["date"])
        self.df = df.sort_values("date").reset_index(drop=True)

        self.rmax = rating_scale     # usato in Eq. 1, 2, 10-11, 16, 20 (rating scale a 5 stelle)
        self.rmin = 1

        # t_data: durata totale del dataset in giorni -> denominatore di AD, Eq. (5)
        self.t_data = max(1, (self.df["date"].max() - self.df["date"].min()).days)

        # groupby riusabili (creati una volta, evitano filtri ripetuti su 4M righe)
        self._reviewer_gb = self.df.groupby("reviewer_id", observed=True)
        self._product_gb = self.df.groupby("product_id", observed=True)

        self._log(f"[init] {len(self.df):,} righe caricate in {time.time()-t0:.1f}s")

        self._precompute_product_stats()
        self._precompute_individual_indicators()

    def _log(self, msg):
        if self.verbose:
            print(msg)

    # -------------------------------------------------------------------
    # Estrazione efficiente delle righe di un gruppo di reviewer
    # -------------------------------------------------------------------
    def _group_reviews(self, group_members) -> pd.DataFrame:
        """
        Ritorna le righe di un insieme (piccolo) di reviewer usando i
        groupby precalcolati (get_group), invece di uno scan isin() su
        tutte le 4M righe. Costo ~ O(somma delle recensioni dei membri),
        non O(n_totale).
        """
        keys = self._reviewer_gb.groups.keys()
        parts = [self._reviewer_gb.get_group(a) for a in group_members if a in keys]
        if not parts:
            return self.df.iloc[0:0]
        return pd.concat(parts, axis=0)

    # =====================================================================
    # PRECALCOLO 1: statistiche per prodotto
    # Servono a: Eq. (2) e Eq. (10)-(11) [media "escludendo" un reviewer/
    # gruppo, calcolata per sottrazione O(1) invece di rifiltrare il df],
    # ed Eq. (20)-(22) [Target Product Location].
    # =====================================================================
    def _precompute_product_stats(self):
        t0 = time.time()
        g = self._product_gb["rating"]

        # Σ rating per prodotto e conteggio -> usati per calcolare "media
        # escludendo un sottoinsieme di reviewer" con una sola sottrazione:
        #   r*_p(escludendo S) = (product_sum[p] - Σ_{a in S} r_ap) / (product_count[p] - |ratings di S su p|)
        # Questa è l'implementazione efficiente della definizione di r*_p
        # usata nelle Eq. (2) [singolo reviewer] e Eq. (10)-(11) [gruppo].
        self.product_sum = g.sum()      # Σ_p r_p   -> Eq. (2), Eq. (10)
        self.product_count = g.size()   # |Ra_p|    -> Eq. (2), Eq. (10)

        # conteggio rating estremi per prodotto -> Eq. (20), Sext(p)
        rating = self.df["rating"].values
        prod = self.df["product_id"].values
        self.product_n5 = self.df.loc[rating == self.rmax].groupby("product_id", observed=True).size() \
                                .reindex(self.product_count.index, fill_value=0)
        self.product_n1 = self.df.loc[rating == self.rmin].groupby("product_id", observed=True).size() \
                                .reindex(self.product_count.index, fill_value=0)

        self._log(f"[precompute] statistiche prodotto in {time.time()-t0:.1f}s "
                   f"({len(self.product_count):,} prodotti)")

    # =====================================================================
    # PRECALCOLO 2: indicatori individuali (Sez. 3.1, Eq. 1-7)
    # Calcolati UNA VOLTA per tutti i reviewer, in modo vettorizzato.
    # =====================================================================
    def _precompute_individual_indicators(self, q_days: int = 28):
        t0 = time.time()
        df = self.df

        total_counts = self._reviewer_gb.size()  # |Ra| per ogni reviewer a

        # --- EXR(a): Eq. (1) -----------------------------------------------
        # Ratio of Extreme Rating = |{r in Ra : r in {1,5}}| / |Ra|
        extreme_mask = df["rating"].isin([self.rmin, self.rmax])
        extreme_counts = df.loc[extreme_mask].groupby("reviewer_id", observed=True).size()
        EXR = (extreme_counts.reindex(total_counts.index, fill_value=0) / total_counts)

        # --- RD(a): Eq. (2) --------------------------------------------------
        # RD(a) = avg_{p in Pa} |r_ap - r*_p| / 4
        # r*_p (media del prodotto p ESCLUDENDO la riga corrente) calcolata
        # per sottrazione dalle somme precalcolate del prodotto (vedi sopra),
        # invece di rifiltrare il dataset per ogni riga -> O(n) totale invece
        # di O(n^2).
        prod_sum_row = df["product_id"].map(self.product_sum).astype("float64")
        prod_cnt_row = df["product_id"].map(self.product_count).astype("float64")
        denom = (prod_cnt_row - 1)
        r_star = (prod_sum_row - df["rating"]) / denom.replace(0, np.nan)
        # fallback (Eq. 2, caso limite non specificato dal paper): se il
        # reviewer è l'UNICO che ha valutato il prodotto, non esistono
        # "altri" reviewer -> usiamo la media del prodotto incluso lui stesso.
        r_star = r_star.fillna(prod_sum_row / prod_cnt_row)
        dev = (df["rating"] - r_star).abs() / (self.rmax - self.rmin)
        RD = dev.groupby(df["reviewer_id"], observed=True).mean()
        RD = RD.reindex(total_counts.index, fill_value=0.0)

        # --- MRO(a): Eq. (3) -------------------------------------------------
        # MaxRev(a) / max_a(MaxRev(a))
        per_reviewer_day = df.groupby(
            ["reviewer_id", df["date"].dt.floor("D")], observed=True
        ).size()
        max_rev_per_reviewer = per_reviewer_day.groupby(level=0, observed=True).max()
        max_rev_per_reviewer = max_rev_per_reviewer.reindex(total_counts.index, fill_value=0)
        max_rev_all = float(max_rev_per_reviewer.max()) if len(max_rev_per_reviewer) else 1.0
        MRO = max_rev_per_reviewer / max(max_rev_all, 1.0)

        # --- RTI(a) e ATR(a): Eq. (4) e Eq. (6) ------------------------------
        # Richiedono le date ordinate per reviewer -> un solo sort globale
        # (già fatto in __init__), poi diff() vettorizzato per gruppo.
        df_sorted = df  # self.df è già ordinato per date, ma diff() va fatto
        # raggruppando per reviewer preservando l'ordine temporale:
        df_by_rev = df.sort_values(["reviewer_id", "date"])
        gap_fwd = df_by_rev.groupby("reviewer_id", observed=True)["date"].diff().dt.days
        gap_bwd = df_by_rev.groupby("reviewer_id", observed=True)["date"].diff(-1).dt.days.abs()

        # RTI: Eq. (4) - frazione di intervalli ADIACENTI (solo "avanti")
        # con gap <= q_days, su (m-1) intervalli totali.
        within_q_fwd = (gap_fwd <= q_days) & gap_fwd.notna()
        rti_counts = within_q_fwd.groupby(df_by_rev["reviewer_id"], observed=True).sum()
        rti_counts = rti_counts.reindex(total_counts.index, fill_value=0)
        rti_denom = (total_counts - 1).clip(lower=1)
        RTI = rti_counts / rti_denom
        RTI[total_counts < 2] = 0.0  # con < 2 review non esiste alcun intervallo

        # ATR: Eq. (6) - frazione di review che cadono in un "intervallo
        # attivo", cioè che hanno un'altra review dello stesso reviewer
        # entro q_days PRIMA o DOPO (usiamo sia gap_fwd che gap_bwd).
        active = ((gap_fwd <= q_days) & gap_fwd.notna()) | ((gap_bwd <= q_days) & gap_bwd.notna())
        atr_counts = active.groupby(df_by_rev["reviewer_id"], observed=True).sum()
        atr_counts = atr_counts.reindex(total_counts.index, fill_value=0)
        ATR = atr_counts / total_counts

        # min_gap per reviewer, riutilizzato più avanti in S_avg (Eq. 21)
        # per individuare i "truthful reviewer" (Feng et al. [12]).
        self._min_gap_per_reviewer = gap_fwd.groupby(df_by_rev["reviewer_id"], observed=True).min()

        # --- AD(a): Eq. (5) ---------------------------------------------------
        date_min = self._reviewer_gb["date"].min()
        date_max = self._reviewer_gb["date"].max()
        span_days = (date_max - date_min).dt.days
        AD = 1.0 - (span_days / self.t_data)

        # --- ISS(a): Eq. (7) = media dei 6 indicatori sopra -------------------
        self.individual = pd.DataFrame({
            "EXR": EXR, "RD": RD, "MRO": MRO, "RTI": RTI, "AD": AD, "ATR": ATR,
        }).reindex(total_counts.index).fillna(0.0)
        self.individual["ISS"] = self.individual[["EXR", "RD", "MRO", "RTI", "AD", "ATR"]].mean(axis=1)

        self.total_counts = total_counts

        self._log(f"[precompute] indicatori individuali (Eq.1-7) per "
                   f"{len(self.individual):,} reviewer in {time.time()-t0:.1f}s")

    @staticmethod
    def _plain_index(idx: pd.Index) -> pd.Index:
        """Converte un CategoricalIndex in un Index 'semplice' per evitare il
        bug di pandas 'Buffer dtype mismatch' quando si fa reindex/align tra
        CategoricalIndex con numero di categorie diverso (codici int8 vs int16)."""
        if isinstance(idx, pd.CategoricalIndex):
            return idx.astype(idx.categories.dtype)
        return idx
    # =====================================================================
    # ALGORITMO 2: TargetProductLocation (Eq. 20-22) - VETTORIZZATO
    # =====================================================================
    def target_product_location(self, delta_tp: float = 0.1, k: float = 0.5,
                                 min_reviews_truthful: int = 10):
        """
        Restituisce l'elenco dei prodotti con STP(p) > delta_tp (Eq. 22).
        Nessun loop Python sui prodotti: tutto calcolato con operazioni
        pandas vettoriali su Serie indicizzate per product_id.
        """
        t0 = time.time()
        df = self.df

        # --- Sext(p): Eq. (20) ----------------------------------------------
        n1 = self.product_n1.astype(float)
        n5 = self.product_n5.astype(float)
        mx = np.maximum(n1, n5)
        Sext = pd.Series(np.where(mx > 0, np.minimum(n1, n5) / mx.replace(0, np.nan), 0.0),
                          index=self.product_count.index).fillna(0.0)

        # --- Savg(p): Eq. (21) -----------------------------------------------
        # SR = single-time reviewer (|Ra| == 1)
        # TR = truthful reviewer (|Ra| > min_reviews_truthful E MAI due
        #      review a distanza <= 1 giorno, Feng et al. [12])
        SR_reviewers = self.total_counts[self.total_counts == 1].index
        TR_candidates = self.total_counts[self.total_counts > min_reviews_truthful].index
        min_gap = self._min_gap_per_reviewer.reindex(TR_candidates)
        # se min_gap è NaN vuol dire reviewer con 1 sola review: escluso perché
        # non può stare in TR_candidates (count>10 implica >=2 review, min_gap definito)
        TR_reviewers = TR_candidates[min_gap > 1]

        is_sr = df["reviewer_id"].isin(SR_reviewers)
        is_tr = df["reviewer_id"].isin(TR_reviewers)

        sr_avg = df.loc[is_sr].groupby("product_id", observed=True)["rating"].mean()
        tr_avg = df.loc[is_tr].groupby("product_id", observed=True)["rating"].mean()
        sr_avg = sr_avg.reindex(self.product_count.index)
        tr_avg = tr_avg.reindex(self.product_count.index)


        # fallback documentato: se per un prodotto non esistono review SR o TR,
        # il paper non specifica il comportamento -> trattiamo Savg come 0
        # per quel prodotto (indicatore non applicabile), NON come prodotto
        # automaticamente sospetto.
        Savg = ((sr_avg - tr_avg).abs() / (self.rmax - self.rmin)).fillna(0.0)

        # --- STP(p): Eq. (22) --------------------------------------------------
        STP = k * Savg + (1 - k) * Sext
        target_products = STP[STP > delta_tp].index.tolist()

        self._log(f"[Algoritmo 2] {len(target_products):,} prodotti target "
                   f"(su {len(STP):,}) in {time.time()-t0:.1f}s")
        return target_products

    # =====================================================================
    # ALGORITMO 3: CandidateGroupGeneration (burst via KDE)
    # Eseguito SOLO sui prodotti target (già filtrati da Algoritmo 2),
    # quindi il costo totale scala con #target_products, non con 4M righe.
    # =====================================================================
    def candidate_group_generation(self, product_id, isize_days: int = 7):
        rows = self._product_gb.get_group(product_id).sort_values("date")
        if len(rows) < 3:
            return []

        t0_date = rows["date"].min()
        dur = (rows["date"].max() - t0_date).days
        if dur == 0:
            return [frozenset(rows["reviewer_id"])]

        x = ((rows["date"] - t0_date).dt.total_seconds().values / 86400.0 / dur)

        k = max(2, dur // isize_days)      # numero di intervalli (ISIZE, Sez. 4.2)
        avg_rev = len(rows) / k            # avg_rev, usato per scartare picchi non significativi (Algoritmo 3, riga 12)

        try:
            kde = gaussian_kde(x)
        except np.linalg.LinAlgError:
            return [frozenset(rows["reviewer_id"])]

        grid = np.linspace(0, 1, 500)
        # try:
        #     kde = gaussian_kde(x, bw_method=(2.0 / dur) / x.std())
        # except np.linalg.LinAlgError:
        #     return [frozenset(rows["reviewer_id"])]
        #
        # grid = np.linspace(0, 1, min(dur + 1, 5000))

        density = kde(grid)                # KDE(x), Sez. 4.2 (kernel Gaussiano)

        d1 = np.gradient(density, grid)
        peak_idx = np.where((d1[:-1] > 0) & (d1[1:] <= 0))[0]  # punti di picco (derivata = 0, Algoritmo 3 riga 11)

        groups = []
        seen = set()
        for idx in peak_idx:
            x_peak = grid[idx]
            t_peak = t0_date + pd.Timedelta(days=float(x_peak * dur))

            lo = t_peak - pd.Timedelta(days=3)
            hi = t_peak + pd.Timedelta(days=3)
            window_rows = rows[(rows["date"] >= lo) & (rows["date"] <= hi)]

            if len(window_rows) <= 1 or len(window_rows) <= avg_rev:
                continue  # scarta picchi non significativi (Algoritmo 3, righe 14-15)

            group = frozenset(window_rows["reviewer_id"])
            if group not in seen:
                seen.add(group)
                groups.append(group)

        return groups

    # =====================================================================
    # ALGORITMO 4: GroupPurification&Classification (Eq. 7-19)
    # =====================================================================
    def _purify_group(self, group, delta_i: float) -> frozenset:
        """Rimuove i membri con ISS(a) < delta_i (Eq. 7), via lookup O(1)."""
        iss = self.individual["ISS"]
        return frozenset(a for a in group if iss.get(a, 0.0) >= delta_i)

    # --- indicatori di GRUPPO (Sez. 3.2, Eq. 8-19) --------------------------

    def GTW(self, group_members, theta_days: int = 30) -> float:
        """Eq. (8)-(9): Group Average Time Window."""
        rows = self._group_reviews(group_members)
        if rows.empty:
            return 0.0
        tw_values = []
        for p, sub in rows.groupby("product_id", observed=True):
            if len(sub) < 2:
                sd_p = 0.0
            else:
                epoch_days = sub["date"].astype("int64") // 10**9 // 86400
                sd_p = float(np.std(epoch_days.values))
            tw_values.append(1.0 - sd_p / theta_days if sd_p <= theta_days else 0.0)
        return float(np.mean(tw_values)) if tw_values else 0.0

    def GRD(self, group_members) -> float:
        """
        Eq. (10)-(11): Group Rating Deviation.
        r*_p (media escludendo il gruppo) calcolata per sottrazione dalle
        somme/conteggi di prodotto precalcolati, non rifiltrando le 4M righe.
        """
        rows = self._group_reviews(group_members)
        if rows.empty:
            return 0.0
        agg = rows.groupby("product_id", observed=True)["rating"].agg(["sum", "count"])
        rd_vals = []
        for p, s, c in agg.itertuples(index=True, name=None):
            total_sum = self.product_sum.get(p, s)
            total_cnt = self.product_count.get(p, c)
            remaining_cnt = total_cnt - c
            if remaining_cnt > 0:
                r_star = (total_sum - s) / remaining_cnt
            else:
                # fallback: il gruppo copre TUTTE le review del prodotto ->
                # non esistono "altri" reviewer, usiamo la media del prodotto
                # incluso il gruppo stesso (stesso fallback di Eq. 2).
                r_star = total_sum / total_cnt
            member_ratings = rows.loc[rows["product_id"] == p, "rating"]
            rd_vals.append(float(((member_ratings - r_star).abs() / (self.rmax - self.rmin)).mean()))
        return float(np.mean(rd_vals)) if rd_vals else 0.0

    def GS(self, group_members) -> float:
        """Eq. (12): Group Size (funzione logistica centrata su size=3)."""
        size = len(group_members)
        return float(1.0 / (1.0 + np.exp(-(size - 3))))

    def GRT(self, group_members) -> float:
        """Eq. (13): Group Review Tightness = |Vg| / (|Rg| * |Pg|)."""
        rows = self._group_reviews(group_members)
        n_reviews = len(rows)
        n_reviewers = len(set(group_members))
        n_products = rows["product_id"].nunique()
        if n_reviewers == 0 or n_products == 0:
            return 0.0
        return n_reviews / (n_reviewers * n_products)

    def GOR(self, group_members, threshold: int = 5) -> float:
        """Eq. (14)-(15): Group One Day Reviews."""
        rows = self._group_reviews(group_members)
        if rows.empty:
            return 0.0
        scores = []
        for a, sub in rows.groupby("reviewer_id", observed=True):
            per_day = sub.groupby(sub["date"].dt.floor("D")).size()
            f_gor = (per_day > threshold).astype(int)
            scores.append(f_gor.sum() / len(per_day))
        return float(np.mean(scores)) if scores else 0.0

    def GER(self, group_members) -> float:
        """Eq. (16): Group Extreme Rating Ratio = media di EXR(a) sui membri (lookup precalcolato, Eq.1)."""
        exr = self.individual["EXR"]
        vals = [exr.get(a, 0.0) for a in group_members]
        return float(np.mean(vals)) if vals else 0.0

    def _co_active_windows(self, group_members, d_days: int = 7):
        """Supporto per Eq. (17)-(18): individua le finestre in cui TUTTI i
        membri del gruppo hanno postato entro d_days l'uno dall'altro."""
        rows = self._group_reviews(group_members).sort_values("date")
        if rows.empty:
            return [], rows

        windows = []
        review_idx_in_window = set()
        members_set = set(group_members)
        for center in rows["date"].unique():
            lo = center - pd.Timedelta(days=d_days)
            hi = center + pd.Timedelta(days=d_days)
            window_rows = rows[(rows["date"] >= lo) & (rows["date"] <= hi)]
            if set(window_rows["reviewer_id"].unique()) == members_set:
                windows.append((lo, hi))
                review_idx_in_window.update(window_rows.index.tolist())

        merged = []
        for lo, hi in sorted(windows):
            if merged and lo <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
            else:
                merged.append((lo, hi))

        rows_in_window = rows.loc[rows.index.isin(review_idx_in_window)]
        return merged, rows_in_window

    def GCA(self, group_members, d_days: int = 7) -> float:
        """Eq. (17): Group Co-Activeness."""
        windows, _ = self._co_active_windows(group_members, d_days)
        return float(1.0 / (1.0 + np.exp(-len(windows))))

    def GCAR(self, group_members, d_days: int = 7) -> float:
        """Eq. (18): Group Co-Active Review Ratio."""
        _, rows_in_window = self._co_active_windows(group_members, d_days)
        total_reviews = len(self._group_reviews(group_members))
        if total_reviews == 0:
            return 0.0
        return len(rows_in_window) / total_reviews

    def GSS(self, group_members, theta_days=30, d_days=7) -> float:
        """Eq. (19): Group Spam Score = media degli 8 indicatori di gruppo."""
        vals = [
            self.GTW(group_members, theta_days),
            self.GRD(group_members),
            self.GS(group_members),
            self.GRT(group_members),
            self.GOR(group_members),
            self.GER(group_members),
            self.GCA(group_members, d_days),
            self.GCAR(group_members, d_days),
        ]
        return float(np.mean(vals))

    def group_purification_and_classification(self, candidate_groups,
                                                delta_i: float = 0.43,
                                                delta_g: float = 0.54,
                                                theta_days: int = 30,
                                                d_days: int = 7):
        """Algoritmo 4: purifica (Eq.7 via lookup) e classifica (Eq.8-19)."""
        spammer_groups = []
        for group in candidate_groups:
            purified = self._purify_group(group, delta_i)
            if len(purified) > 2:
                gss = self.GSS(purified, theta_days, d_days)
                if gss >= delta_g:
                    spammer_groups.append({"members": purified, "GSS": gss})
        return spammer_groups

    # =====================================================================
    # ALGORITMO 1: pipeline completa (con parallelizzazione opzionale)
    # =====================================================================
    def run(self, delta_tp: float = 0.1, delta_i: float = 0.43, delta_g: float = 0.54,
            k: float = 0.5, isize_days: int = 7, theta_days: int = 30, d_days: int = 7,
            n_jobs: int = 1):
        """
        Algoritmo 1 (GSDB).
        n_jobs > 1: parallelizza il loop sui prodotti target (Algoritmi 3+4),
        che sono indipendenti tra loro - utile quando dopo il filtro
        dell'Algoritmo 2 restano comunque molte migliaia di prodotti target.
        """
        t0 = time.time()
        target_products = self.target_product_location(delta_tp, k)

        results = {}
        if n_jobs <= 1 or len(target_products) < 50:
            for p in target_products:
                candidates = self.candidate_group_generation(p, isize_days)
                spammer_groups = self.group_purification_and_classification(
                    candidates, delta_i, delta_g, theta_days, d_days
                )
                if spammer_groups:
                    results[p] = spammer_groups
        else:
            # NB: ProcessPoolExecutor serializza self -> conviene solo se
            # #target_products è grande abbastanza da coprire l'overhead di
            # pickling del DataFrame. Per dataset molto grandi, valutare di
            # passare ai worker solo le righe dei prodotti target invece
            # dell'oggetto self completo.
            with ProcessPoolExecutor(max_workers=n_jobs) as ex:
                futs = {
                    ex.submit(_process_one_product, self, p, isize_days,
                              delta_i, delta_g, theta_days, d_days): p
                    for p in target_products
                }
                for fut in as_completed(futs):
                    p = futs[fut]
                    spammer_groups = fut.result()
                    if spammer_groups:
                        results[p] = spammer_groups

        self._log(f"[Algoritmo 1] completato in {time.time()-t0:.1f}s: "
                  f"{sum(len(v) for v in results.values())} gruppi spammer trovati "
                  f"su {len(target_products)} prodotti target.")
        return results


def _process_one_product(gsdb: "GSDBFast", product_id, isize_days, delta_i, delta_g,
                          theta_days, d_days):
    """Funzione top-level (picklable) per l'esecuzione parallela di Algoritmo 3+4."""
    candidates = gsdb.candidate_group_generation(product_id, isize_days)
    return gsdb.group_purification_and_classification(
        candidates, delta_i, delta_g, theta_days, d_days
    )


# =============================================================================
# FEATURE DI OUTPUT (Mukherjee, Liu, Glance - WWW 2012)
# Stesse definizioni di group_features.py, così gli output dei due metodi
# sono confrontabili. Tutte in [0,1], 1 = sospetto (tranne IMC_avg/IMC_max,
# dove 0 = membri sincronizzati; IMC_coupling_avg = 1 - IMC_avg).
# =============================================================================

# stessi default di group_features.py (Sez. 9.1 del paper)
MLG_TAU_DAYS = 2.87 * 30.44    # finestra GTW
MLG_BETA_DAYS = 8.86 * 30.44   # soglia "early" per GETF / IETF
MLG_MAX_RATING_DEV = 4.0       # scala 1-5

METRIC_COLS = [
    "GTW", "GD", "GETF", "GSR", "GS", "GSUP",
    "IRD_avg", "IRD_max", "IETF_avg", "IETF_max",
    "IMC_avg", "IMC_max", "IMC_coupling_avg",
    # popolarità: n. di recensioni nel dataset (valori grezzi, non in [0,1])
    "ITEM_POP_avg", "ITEM_POP_max", "ITEM_POP_min",
    "USER_POP_avg", "USER_POP_max", "USER_POP_min",
]


def build_item_index(df: pd.DataFrame, items):
    """
    Precalcola, SOLO per i prodotti che compaiono nei gruppi, le stesse
    strutture di ReviewData in group_features.py:
      item_rev[p]   = {user: (rating, giorno)}  (ultima recensione se duplicata)
      item_first[p] = giorno della prima recensione del prodotto (proxy del lancio)
      item_sum[p], item_cnt[p] = somma e numero di rating del prodotto
    Le date sono troncate al giorno di calendario, come fa parse_day().
    """
    sub = df.loc[df["product_id"].isin(set(items)),
                 ["reviewer_id", "product_id", "rating", "date"]].copy()
    sub["day"] = (sub["date"] - pd.Timestamp("1970-01-01")).dt.days.astype(float)
    sub["rating"] = sub["rating"].astype("float64")

    stats = sub.groupby("product_id").agg(first=("day", "min"),
                                          s=("rating", "sum"),
                                          n=("rating", "size"))
    item_first = stats["first"].to_dict()
    item_sum = stats["s"].to_dict()
    item_cnt = stats["n"].to_dict()

    # df è ordinato per data: keep="last" = recensione più recente
    last = sub.drop_duplicates(["product_id", "reviewer_id"], keep="last")
    item_rev = {}
    for p, u, r, d in zip(last["product_id"].values, last["reviewer_id"].values,
                          last["rating"].values, last["day"].values):
        item_rev.setdefault(p, {})[u] = (float(r), float(d))
    return item_rev, item_first, item_sum, item_cnt


def group_metrics(users, items, index, user_cnt=None,
                  tau=MLG_TAU_DAYS, beta=MLG_BETA_DAYS):
    """
    Feature di un gruppo (users, items). GS e GSUP restano grezze
    (_GS_raw, _GSUP_raw) e vanno normalizzate sul max di tutti i gruppi.
    user_cnt: {utente: n. recensioni nel dataset}, per USER_POP_*.
    I gruppi GSDB hanno recensito tutti i loro item per costruzione, ma per
    sicurezza su ogni prodotto si considerano solo i membri che l'hanno
    effettivamente recensito.
    """
    item_rev, item_first, item_sum, item_cnt = index
    users = list(users)

    gtw = gd = getf = 0.0
    gsr_vals = []
    ird = dict.fromkeys(users, 0.0)
    ietf = dict.fromkeys(users, 0.0)
    imc_vals = {u: [] for u in users}

    for p in items:
        rev = item_rev.get(p)
        if not rev:
            continue
        present = [u for u in users if u in rev]
        if not present:
            continue
        n_g = len(present)
        ratings = {u: rev[u][0] for u in present}
        days = {u: rev[u][1] for u in present}
        n_all, s_all = item_cnt[p], item_sum[p]
        F, L = min(days.values()), max(days.values())
        span = L - F

        # GTW: finestra temporale del gruppo sul prodotto (max sui prodotti)
        gtw = max(gtw, 0.0 if span > tau else 1.0 - span / tau)

        # GD: |media gruppo - media non membri| / 4 (max sui prodotti)
        g_sum = sum(ratings.values())
        if n_all - n_g > 0:
            oth = (s_all - g_sum) / (n_all - n_g)
            gd = max(gd, abs(g_sum / n_g - oth) / MLG_MAX_RATING_DEV)

        # GETF: ultima recensione del gruppo vs lancio del prodotto
        late = L - item_first[p]
        getf = max(getf, 0.0 if late > beta else 1.0 - late / beta)

        # GSR: quota di recensori del prodotto che sono nel gruppo (media)
        gsr_vals.append(n_g / n_all)

        tot_off = sum(d - F for d in days.values())
        for m in present:
            # IRD: rating del membro vs media di TUTTI gli altri recensori
            if n_all - 1 > 0:
                oth_m = (s_all - ratings[m]) / (n_all - 1)
                ird[m] = max(ird[m], abs(ratings[m] - oth_m) / MLG_MAX_RATING_DEV)
            # IETF: recensione del membro vs lancio del prodotto
            late_m = days[m] - item_first[p]
            ietf[m] = max(ietf[m], 0.0 if late_m > beta else 1.0 - late_m / beta)
            # IMC: distanza temporale del membro dalla media degli altri membri
            if span <= 0 or n_g < 2:
                imc_vals[m].append(0.0)
            else:
                off_m = days[m] - F
                others_avg = (tot_off - off_m) / (n_g - 1)
                imc_vals[m].append(abs(off_m - others_avg) / span)

    def agg(vals):
        vals = list(vals)
        return (sum(vals) / len(vals), max(vals)) if vals else (0.0, 0.0)

    imc = {m: (sum(v) / len(v) if v else 0.0) for m, v in imc_vals.items()}
    ird_avg, ird_max = agg(ird.values())
    ietf_avg, ietf_max = agg(ietf.values())
    imc_avg, imc_max = agg(imc.values())

    # popolarità = n. di recensioni nel dataset
    #   item: quante recensioni ha ricevuto ciascun prodotto del gruppo
    #   user: quante recensioni ha scritto ciascun membro del gruppo
    user_cnt = user_cnt or {}
    item_pop = [float(item_cnt.get(p, 0)) for p in items]
    user_pop = [float(user_cnt.get(u, 0)) for u in users]

    def agg3(vals):
        return (sum(vals) / len(vals), max(vals), min(vals)) if vals else (0.0, 0.0, 0.0)

    ip_avg, ip_max, ip_min = agg3(item_pop)
    up_avg, up_max, up_min = agg3(user_pop)

    return {
        "GTW": gtw,
        "GD": gd,
        "GETF": getf,
        "GSR": sum(gsr_vals) / len(gsr_vals) if gsr_vals else 0.0,
        "_GS_raw": float(len(users)),
        "_GSUP_raw": float(len(items)),
        "IRD_avg": ird_avg, "IRD_max": ird_max,
        "IETF_avg": ietf_avg, "IETF_max": ietf_max,
        "IMC_avg": imc_avg, "IMC_max": imc_max,
        "IMC_coupling_avg": 1.0 - imc_avg,
        "ITEM_POP_avg": ip_avg, "ITEM_POP_max": ip_max, "ITEM_POP_min": ip_min,
        "USER_POP_avg": up_avg, "USER_POP_max": up_max, "USER_POP_min": up_min,
    }


def compute_all_group_metrics(df: pd.DataFrame, groups_map: dict,
                              user_counts=None,
                              tau=MLG_TAU_DAYS, beta=MLG_BETA_DAYS):
    """
    groups_map: {frozenset(membri): set(prodotti)}.
    user_counts: Serie/dict {utente: n. recensioni} (es. gsdb.total_counts);
    se None viene ricalcolata da df.
    Restituisce una riga (dict) per gruppo con n_users, n_items, users,
    items e tutte le METRIC_COLS. GS e GSUP normalizzate sul max fra i gruppi
    (eq. 7-8 del paper), come in group_features.py.
    """
    all_items = set().union(*groups_map.values()) if groups_map else set()
    index = build_item_index(df, all_items)

    all_users = set().union(*groups_map.keys()) if groups_map else set()
    if user_counts is None:
        user_counts = df.loc[df["reviewer_id"].isin(all_users), "reviewer_id"].value_counts()
    user_cnt = {u: int(user_counts.get(u, 0)) for u in all_users}

    rows = []
    for members, items in groups_map.items():
        users_sorted = sorted(map(str, members))
        items_sorted = sorted(map(str, items))
        row = {
            "n_users": len(users_sorted),
            "n_items": len(items_sorted),
            "users": ",".join(users_sorted),
            "items": ",".join(items_sorted),
        }
        row.update(group_metrics(users_sorted, items_sorted, index, user_cnt, tau, beta))
        rows.append(row)

    for raw, name in (("_GS_raw", "GS"), ("_GSUP_raw", "GSUP")):
        mx = max((r[raw] for r in rows), default=1.0) or 1.0
        for r in rows:
            r[name] = r.pop(raw) / mx
    return rows


# =============================================================================
# ESECUZIONE SU DATI REALI
# =============================================================================
# if __name__ == "__main__":
#     datasets = [
#         'Baby_Products',
#         'Sports_and_Outdoors',
#         'Office_Products',
#         'Toys_and_Games',
#         'Pet_Supplies',
#         'Beauty_and_Personal_Care',
#         'Books',
#         'yelp'
#     ]
#     datasets = ["Health_and_Household"]
#
#     for dataset in datasets:
#         print('DATASET', dataset)
#
#
#         CSV_PATH = f"data/original/{dataset}/{dataset}.train.csv"
#         CSV_OUT_PATH = f"data/original/{dataset}/{dataset}.gsdb_f.csv"
#
#         print(f"Caricamento {CSV_PATH} ...")
#         reviews = load_reviews_csv(CSV_PATH)
#         print(f"Caricate {len(reviews):,} interazioni, "
#               f"{reviews['reviewer_id'].nunique():,} utenti, "
#               f"{reviews['product_id'].nunique():,} prodotti.")
#
#         gsdb = GSDBFast(reviews)
#         iss_values = gsdb.individual["ISS"]
#         print((iss_values < 0.43).mean())
#         # Soglie di default riprese da Sez. 5.2 del paper (dTP=0.1, dI=0.43, dG=0.54)
#         #detected = gsdb.run(delta_tp=0.05, delta_i=0.43, delta_g=0.2, n_jobs=1)
#         detected = gsdb.run(delta_tp=0.05, delta_i=float(iss_values.quantile(0.70)), delta_g=0.2, n_jobs=1)
#         # ---------------------------------------------------------------
#         # OUTPUT: un gruppo per riga, formato
#         # n_users  n_items  users  items  GTW ... IMC_coupling_avg
#         # ---------------------------------------------------------------
#
#         # Unisco i gruppi con gli stessi membri trovati su prodotti diversi:
#         # items = insieme dei prodotti su cui il gruppo è stato rilevato
#         groups_map = {}  # frozenset(membri) -> set(prodotti)
#         for product, groups in detected.items():
#             for g in groups:
#                 groups_map.setdefault(frozenset(g["members"]), set()).add(product)
#
#         t0 = time.time()
#         rows = compute_all_group_metrics(gsdb.df, groups_map,
#                                          user_counts=gsdb.total_counts)
#         print(f"[metriche] {len(rows):,} gruppi valutati in {time.time()-t0:.1f}s")
#
#         out_df = pd.DataFrame(rows, columns=["n_users", "n_items", "users", "items"] + METRIC_COLS)
#         out_df.sort_values(["n_users", "n_items"], ascending=[False, False], inplace=True)
#
#         # Separatore TAB: users/items contengono virgole, così non servono virgolette
#         out_df.to_csv(CSV_OUT_PATH, sep="\t", index=False, float_format="%.4f")
#
#         print(f"\nSalvato: {CSV_OUT_PATH}")
#         print(f"Gruppi salvati: {len(out_df):,}")


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
       - --exclude-head-pct: toglie dai prodotti target gli item head del train, con la
         stessa definizione di paper_figures (item più popolari che coprono quella
         percentuale degli archi di training). Su un item head ci sono così tanti
         recensori che bastano coincidenze per formare un "gruppo" in una settimana;
       - --kde-bw-days: bandwidth della KDE in giorni (Algoritmo 3), coerente con
         ISIZE = 7 giorni. 0 = bandwidth automatica di scipy (comportamento di gsdb.py);
  4. proietta ogni gruppo sul train: tiene solo i membri e gli item che hanno
     almeno un arco del train tra loro; scarta i gruppi con < --min-members membri;
     unisce i gruppi che dopo la proiezione hanno gli stessi membri;
  5. calcola le feature di output sul train (come per gli altri file *_f) e salva
     <dataset>.gsdb_f.csv nello stesso formato di prima (backup del vecchio file).

Uso (dalla root del progetto):
  python3 src/extraction/gsdb_unfiltered.py --datasets Health_and_Household
"""

import argparse
import os
import shutil
import time

import pandas as pd

from scipy.stats import gaussian_kde

# from gsdb import (GSDBFast, load_reviews_csv, compute_all_group_metrics,
#                   METRIC_COLS)


class GSDBTuned(GSDBFast):
    """GSDBFast con due differenze, tutto il resto è identico a gsdb.py:
    - i prodotti in `exclude` non diventano mai prodotti target (Algoritmo 2);
    - la KDE dell'Algoritmo 3 usa una bandwidth fissa di `bw_days` giorni (0 = scipy)."""

    def __init__(self, reviews, exclude=(), bw_days: float = 0.0, **kw):
        self.exclude = set(exclude)
        self.bw_days = float(bw_days)
        super().__init__(reviews, **kw)

    def target_product_location(self, *args, **kw):
        tp = super().target_product_location(*args, **kw)
        kept = [p for p in tp if p not in self.exclude]
        self._log(f"[head] tolti {len(tp) - len(kept):,} prodotti target head, "
                  f"restano {len(kept):,}")
        return kept

    def candidate_group_generation(self, product_id, isize_days: int = 7):
        if self.bw_days <= 0:
            return super().candidate_group_generation(product_id, isize_days)
        # stessa logica di gsdb.py, cambia solo la bandwidth (e la griglia, a ~1 giorno)
        rows = self._product_gb.get_group(product_id).sort_values("date")
        if len(rows) < 3:
            return []
        t0_date = rows["date"].min()
        dur = (rows["date"].max() - t0_date).days
        if dur == 0:
            return [frozenset(rows["reviewer_id"])]
        x = ((rows["date"] - t0_date).dt.total_seconds().values / 86400.0 / dur)
        k = max(2, dur // isize_days)
        avg_rev = len(rows) / k
        sd = x.std()
        if sd <= 0:
            return [frozenset(rows["reviewer_id"])]
        try:
            kde = gaussian_kde(x, bw_method=(self.bw_days / dur) / sd)
        except np.linalg.LinAlgError:
            return [frozenset(rows["reviewer_id"])]
        grid = np.linspace(0, 1, min(dur + 1, 5000))
        density = kde(grid)
        d1 = np.gradient(density, grid)
        peak_idx = np.where((d1[:-1] > 0) & (d1[1:] <= 0))[0]

        groups, seen = [], set()
        for idx in peak_idx:
            t_peak = t0_date + pd.Timedelta(days=float(grid[idx] * dur))
            lo, hi = t_peak - pd.Timedelta(days=3), t_peak + pd.Timedelta(days=3)
            window_rows = rows[(rows["date"] >= lo) & (rows["date"] <= hi)]
            if len(window_rows) <= 1 or len(window_rows) <= avg_rev:
                continue
            group = frozenset(window_rows["reviewer_id"])
            if group not in seen:
                seen.add(group)
                groups.append(group)
        return groups


def head_items(train: pd.DataFrame, pct: float) -> set:
    """Item head come in paper_figures: i più popolari nel train finché coprono pct% degli archi
    (l'item che supera la soglia è incluso)."""
    if pct <= 0:
        return set()
    pop = train["product_id"].value_counts()          # già in ordine decrescente
    cum = pop.cumsum() / pop.sum()
    n = int(np.searchsorted(cum.values, pct / 100.0, side="left")) + 1
    head = set(pop.index[:n])
    print(f"[head] {len(head):,} item head ({100 * len(head) / len(pop):.2f}% del catalogo del train) "
          f"coprono {100 * pop.iloc[:n].sum() / pop.sum():.1f}% degli archi del train")
    return head


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--root", default="data/original")
    p.add_argument("--full-pattern", default="{ds}.csv",
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
    p.add_argument("--exclude-head-pct", type=float, default=20.0,
                   help="esclude gli item head che coprono questa %% degli archi del train "
                        "(stessa definizione di paper_figures; 0 = nessuna esclusione)")
    p.add_argument("--kde-bw-days", type=float, default=2.0,
                   help="bandwidth della KDE in giorni (0 = automatica di scipy, come gsdb.py)")
    p.add_argument("--n-jobs", type=int, default=1)
    return p.parse_args()


def pair_keys(users, items) -> pd.Series:
    return pd.Series(users, dtype=str).str.cat(pd.Series(items, dtype=str).values, sep="\t")


def load_eval_pairs(path: str) -> set:
    df = pd.read_csv(path, usecols=["user_id", "item_id"], dtype=str)
    return set(pair_keys(df["user_id"].values, df["item_id"].values))


def cut_unfiltered(full: pd.DataFrame, train: pd.DataFrame, eval_paths: list) -> pd.DataFrame:
    n0 = len(full)
    cutoff = train["date"].max()
    full = full.loc[full["date"] <= cutoff]
    print(f"[leakage] cutoff temporale {cutoff}: tolte {n0 - len(full):,} interazioni successive")

    if not eval_paths:
        print("[leakage] ATTENZIONE: nessun file di validation/test trovato. Uso solo il cutoff "
              "temporale, che basta solo se lo split è temporale globale.")
    for path in eval_paths:
        pairs = load_eval_pairs(path)
        keys = pair_keys(full["reviewer_id"].values, full["product_id"].values)
        drop = keys.isin(pairs).values
        full = full.loc[~drop]
        print(f"[leakage] {os.path.basename(path)}: tolte {int(drop.sum()):,} interazioni "
              f"({len(pairs):,} coppie nel file)")

    # controllo: il train dovrebbe essere (quasi) tutto contenuto nel non filtrato tagliato
    tk = set(pair_keys(full["reviewer_id"].values, full["product_id"].values))
    trk = pair_keys(train["reviewer_id"].values, train["product_id"].values)
    cov = trk.isin(tk).mean()
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

    head = head_items(train, a.exclude_head_pct)
    gsdb = GSDBTuned(full, exclude=head, bw_days=a.kde_bw_days)
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