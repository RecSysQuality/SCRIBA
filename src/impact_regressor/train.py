#!/usr/bin/env python3
"""
SCRIBA - regressore lineare dell'impatto dei difetti, SOLO da embedding GraphSAGE.

DOMANDA A CUI RISPONDE
    Allenato su alcuni dataset, il modello sa ordinare i difetti di un dataset MAI visto?

PROTOCOLLO (leave-one-dataset-out)
    A turno, ogni dataset fa da "dataset nuovo":
      test  = tutti i suoi difetti (group_f, gsdb_f e rev2)
      train = dagli ALTRI dataset, per ciascuno: --n-group difetti group_f + --n-gsdb difetti gsdb_f
              + --n-rev2 difetti rev2
      val   = 20% del train, serve solo a scegliere l'epoca migliore (il test non viene mai usato per scegliere)
    Il modello e' lineare (impatto = X·w + b), allenato epoca per epoca (mini-batch + Adam + L2).
    Come riferimento si calcola anche la soluzione esatta (ridge) sugli stessi dati:
    se la curva ci arriva, l'allenamento a epoche ha fatto il suo lavoro.

METRICA PRINCIPALE
    Spearman tra impatto predetto e impatto vero sul dataset nuovo (1 = ordine perfetto, 0 = a caso).
    Nella pipeline il regressore ordina i difetti di UN grafo: conta l'ordine, non la scala.

LABEL (--label, oppure --labels per allenarne piu' di una in un colpo solo)
    rr        = impact / n_affected_all         (calo medio di rr_best per utente affetto)   [default]
    rr_active = impact_active / n_affected_active (stessa cosa, solo sugli utenti active affetti)
    rr_lt     = impact_lt / n_affected_lt         (stessa cosa, solo sugli utenti lt affetti)
    entropy   = variazione di entropia sugli utenti affetti
    <chiave>  = qualsiasi altra chiave numerica del pickle
    Un difetto che non tocca nessun utente del segmento (n_affected_g = 0) non ha label per quel
    segmento e viene escluso da quel modello.
    La label viene normalizzata dentro ogni dataset (--label-norm), cosi' il modello impara
    "quali difetti sono peggio della media del loro grafo" invece della scala del singolo dataset.

USO
    python impact_regressor.py                                   # 800 group_f + 500 gsdb_f + 500 rev2 per dataset
    python impact_regressor.py --n-group 500 --n-gsdb 0 --n-rev2 0
    python impact_regressor.py --grid 600:0:0 300:300:0 300:0:300 200:200:200   # confronta composizioni
    python impact_regressor.py --label entropy
    python impact_regressor.py --labels rr rr_active rr_lt       # i 3 modelli: all, active, lt
    (-1 = usa tutti i difetti disponibili di quel tipo; in --grid "G:S" senza R usa --n-rev2)
    Se il pickle di un dataset non ha difetti rev2, per quel dataset se ne usano 0 e le colonne rev2 sono vuote.

ANALISI (tutte nel RECAP.md)
    1. Generalizzazione: Spearman e R2 sul dataset nuovo, confrontati con i dataset gia' visti
    2. Curve epoca per epoca (MSE, Spearman, R2)
    3. Composizione del training group_f / gsdb_f / rev2 (--grid)
    4. Correlazione tra ogni dimensione dell'embedding e la label (per dataset e per tipo di difetto)
    5. Learning curve: quanto migliora aumentando il numero di difetti di training
    6. Calibrazione: predetto vs reale sul dataset nuovo (pendenza, intercetta, R2 prima/dopo)
    7. Scala del target nei dataset (prima e dopo la normalizzazione)
    (--skip-analyses salta la 4 e la 5; la 6 e la 7 girano sempre perche' costano pochissimo)

OUTPUT (in --out)
    RECAP.md                 risultati + guida alla lettura
    curves_g{N}_s{M}_r{R}.png  curve epoca per epoca per ogni dataset nuovo
    composition.png          confronto tra composizioni (se --grid)
    feature_correlations.png / .csv
    learning_curve.png / .csv
    calibration.png / .csv   scatter predetto vs reale + retta di calibrazione
    target_stats.png / .csv  distribuzione del target per dataset
    results.csv              tutti i numeri
    final_model.pkl          modello allenato su TUTTI i dataset (per la fase attiva)
  Con --labels (piu' label): tutto quanto sopra va in una sottocartella per label
  (--out/rr, --out/rr_active, --out/rr_lt) e i modelli finali vengono copiati in --out come
    final_model.pkl (rr) · final_model_active.pkl (rr_active) · final_model_lt.pkl (rr_lt)
  pronti per mab_defects.py (--regressor, --regressor-active, --regressor-lt).
"""
import argparse
import copy
import pickle
import shutil
import warnings
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata, spearmanr
from sklearn.linear_model import RidgeCV

warnings.filterwarnings("ignore")

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_EMB_DIR = ROOT_DIR / "node_embeddings" / "defects"
SOURCES = ("group_f", "gsdb_f", "rev2")
COLORS = {"group_f": "tab:blue", "gsdb_f": "tab:orange", "rev2": "tab:green"}

# label per segmento: (chiave dell'impatto, chiave del numero di utenti affetti) nel pickle
SEG_LABELS = {"rr": ("impact", "n_affected_all"),
              "rr_active": ("impact_active", "n_affected_active"),
              "rr_lt": ("impact_lt", "n_affected_lt")}
# nome del modello finale quando si allenano piu' label insieme (quelli attesi da mab_defects.py)
MODEL_NAME = {"rr": "final_model.pkl", "rr_active": "final_model_active.pkl", "rr_lt": "final_model_lt.pkl"}


def comp_name(n_group, n_gsdb, n_rev2):
    return f"{n_group} group_f + {n_gsdb} gsdb_f + {n_rev2} rev2"


# =============================================================================== dati
def norm_graph(X):
    """Standardizzazione delle feature dentro UN grafo (per ora non usata: vedi load_data)."""
    return (X - X.mean(0)) / (X.std(0) + 1e-8)

def align_manual(p, p_m, name=""):
    """X di manual riordinata sulle righe di p tramite defect_id (NaN se mancante)."""
    ids = np.asarray(p["defect_id"])
    ids_m = np.asarray(p_m["defect_id"])
    if len(np.unique(ids_m)) != len(ids_m):
        raise ValueError(f"{name}: defect_id duplicati in manual")
    Xm = np.asarray(p_m["X"], dtype=np.float64).reshape(len(ids_m), -1)
    pos = {k: i for i, k in enumerate(ids_m)}
    idx = np.array([pos.get(k, -1) for k in ids])
    found = idx >= 0
    out = np.full((len(ids), Xm.shape[1]), np.nan)
    out[found] = Xm[idx[found]]
    if (~found).any():
        print(f"  [{name}] {(~found).sum()}/{len(ids)} defect_id assenti in manual")
    return out

def label_values(p, label, name=""):
    """Label grezza per ogni difetto del pickle. rr / rr_active / rr_lt = impatto medio per utente
    affetto del segmento (NaN se il difetto non tocca nessun utente del segmento)."""
    if label in SEG_LABELS:
        k_imp, k_n = SEG_LABELS[label]
        missing = [k for k in (k_imp, k_n) if k not in p]
        if missing:
            raise KeyError(f"label '{label}': mancano {missing} nel pickle di {name}. Chiavi: {list(p.keys())}")
        n_g = np.asarray(p[k_n], dtype=np.float64)
        return np.asarray(p[k_imp], dtype=np.float64) / np.where(n_g > 0, n_g, np.nan)
    if label in p:
        return np.asarray(p[label], dtype=np.float64)
    raise KeyError(f"'{label}' non e' nel pickle di {name}. Chiavi: {list(p.keys())}")


def load_data(emb_dir, datasets, label):
    """Un dizionario dataset -> (X, y, source). Tiene solo i difetti con label valida."""
    data = {}
    print(f"[dati] cartella: {emb_dir} · label: {label}")
    for d in datasets:
        with open(emb_dir / f"{d}.pkl", "rb") as f:
            p = pickle.load(f)

        X = np.asarray(p["X"], dtype=np.float64)
        y = label_values(p, label, d)
        src = np.asarray(p["source"]).astype(str)
        ok = np.isfinite(y) & np.isfinite(X).all(1)

        # per standardizzare le feature dentro ogni grafo, sostituisci X[ok] con norm_graph(X[ok])
        data[d] = SimpleNamespace(X=(X[ok]), y=y[ok], src=src[ok])
        counts = {s: int((src[ok] == s).sum()) for s in SOURCES}
        print(f"[dati] {d}: {ok.sum()} difetti con label ({counts}), scartati {int((~ok).sum())}"
              + (f" (non toccano il segmento o label non valida)" if label in SEG_LABELS and label != "rr" else ""))
        if counts["rev2"] == 0:
            print(f"[dati] {d}: nessun difetto rev2 nel pickle (serve il LOO su rev2 e compute_embeddings con rev2)")
    return data


def normalize_label(y, how):
    """Normalizzazione dentro UN dataset."""
    if how == "none":
        return y.copy()
    if how == "rank":  # ranghi -> punteggi normali: robusto agli outlier
        return norm.ppf((rankdata(y) - 0.5) / len(y))
    sd = y.std()
    return (y - y.mean()) / (sd if sd > 0 else 1.0)


def sample_train(data, names, n_group, n_gsdb, n_rev2, rng, label_norm):
    """Campiona n_group + n_gsdb + n_rev2 difetti da ciascun dataset di training."""
    X, y, strata, used = [], [], [], {}
    for d in names:
        D = data[d]
        parts = []
        for s, n in (("group_f", n_group), ("gsdb_f", n_gsdb), ("rev2", n_rev2)):
            pool = np.where(D.src == s)[0]
            k = len(pool) if n < 0 else min(n, len(pool))
            used[(d, s)] = k
            if k > 0:
                parts.append(rng.choice(pool, k, replace=False))
        if not parts:
            continue
        idx = np.concatenate(parts)
        X.append(D.X[idx])
        y.append(normalize_label(D.y[idx], label_norm))
        strata.append(np.char.add(f"{d}|", D.src[idx]))
    return np.vstack(X), np.concatenate(y), np.concatenate(strata), used


def split_val(strata, frac, rng):
    """20% per la validazione, stratificato per dataset x tipo di difetto."""
    val = np.zeros(len(strata), bool)
    for s in np.unique(strata):
        idx = rng.permutation(np.where(strata == s)[0])
        k = int(round(frac * len(idx))) if len(idx) >= 5 else 0
        val[idx[:k]] = True
    return ~val, val


# =============================================================================== modello
class LinearRegressorAdam:
    """y = X·w + b, allenato a epoche con mini-batch, Adam e penalita' L2."""

    def __init__(self, d, lr=3e-3, weight_decay=1e-3):
        self.w, self.b = np.zeros(d), 0.0
        self.lr, self.wd = lr, weight_decay
        self.m, self.v, self.t = np.zeros(d + 1), np.zeros(d + 1), 0

    def predict(self, X):
        return X @ self.w + self.b

    def step(self, X, y):
        r = self.predict(X) - y
        g = np.append(2 * X.T @ r / len(y) + 2 * self.wd * self.w, 2 * r.mean())
        self.t += 1
        self.m = 0.9 * self.m + 0.1 * g
        self.v = 0.999 * self.v + 0.001 * g * g
        upd = self.lr * (self.m / (1 - 0.9 ** self.t)) / (np.sqrt(self.v / (1 - 0.999 ** self.t)) + 1e-8)
        self.w -= upd[:-1]
        self.b -= upd[-1]

    def fit_epoch(self, X, y, batch, rng):
        order = rng.permutation(len(y))
        for i in range(0, len(y), batch):
            j = order[i:i + batch]
            self.step(X[j], y[j])

    def get(self):
        return self.w.copy(), self.b

    def set(self, wb):
        self.w, self.b = wb[0].copy(), wb[1]


def spearman(y, p):
    return spearmanr(y, p).correlation if np.std(p) > 0 and np.std(y) > 0 else np.nan


def r2(y, p):
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def train_with_history(Xtr, ytr, Xval, yval, args, rng, test=None):
    """Allena epoca per epoca; ritorna modello all'epoca migliore (su val) e la storia."""
    model = LinearRegressorAdam(Xtr.shape[1], args.lr, args.weight_decay)
    hist, best, best_wb, row_best = [], -np.inf, model.get(), 1
    for ep in range(1, args.epochs + 1):
        model.fit_epoch(Xtr, ytr, args.batch, rng)
        row = {"epoch": ep,
               "loss_train": np.mean((model.predict(Xtr) - ytr) ** 2),
               "loss_val": np.mean((model.predict(Xval) - yval) ** 2),
               "spearman_val": spearman(yval, model.predict(Xval)),
               "r2_val": r2(yval, model.predict(Xval))}
        if test is not None:
            p = model.predict(test.X)
            row["spearman_test_all"] = spearman(test.y, p)
            row["r2_test_all"] = r2(test.y, p)
            for s in SOURCES:
                m = test.src == s
                row[f"spearman_test_{s}"] = spearman(test.y[m], p[m]) if m.sum() >= 10 else np.nan
                row[f"r2_test_{s}"] = r2(test.y[m], p[m]) if m.sum() >= 10 else np.nan
        hist.append(row)
        if row["spearman_val"] > best:
            best, best_wb, row_best = row["spearman_val"], model.get(), ep
    model.set(best_wb)
    return model, pd.DataFrame(hist), row_best


# =============================================================================== esperimento
def run_config(data, n_group, n_gsdb, n_rev2, args):
    """Leave-one-dataset-out per una composizione (n_group, n_gsdb, n_rev2)."""
    names = list(data)
    rows, hists, preds = [], [], []
    for new in names:
        train_names = [d for d in names if d != new]
        D = data[new]
        test = SimpleNamespace(X=D.X, y=normalize_label(D.y, args.label_norm), src=D.src)
        for seed in range(args.seeds):
            rng = np.random.default_rng(1000 * seed + 7)
            X, y, strata, used = sample_train(data, train_names, n_group, n_gsdb, n_rev2, rng, args.label_norm)
            tr, val = split_val(strata, 0.2, rng)
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
            Z = lambda A: (A - mu) / sd
            test_z = SimpleNamespace(X=Z(test.X), y=test.y, src=test.src)
            model, h, best_ep = train_with_history(Z(X[tr]), y[tr], Z(X[val]), y[val], args, rng, test_z)
            h["new_dataset"], h["seed"] = new, seed
            hists.append(h)

            ridge = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(Z(X[tr]), y[tr])
            pr = ridge.predict(test_z.X)
            p = model.predict(test_z.X)
            from sklearn.ensemble import RandomForestRegressor

            rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=20, n_jobs=-1,
                                       random_state=seed).fit(Z(X[tr]), y[tr])


            row = {"n_group": n_group, "n_gsdb": n_gsdb, "n_rev2": n_rev2, "new_dataset": new, "seed": seed,
                   "n_train": int(tr.sum()), "best_epoch": best_ep,
                   "spearman_all": spearman(test.y, p), "r2_all": r2(test.y, p),
                   "spearman_in_domain": h.spearman_val.iloc[best_ep - 1],
                   "r2_in_domain": h.r2_val.iloc[best_ep - 1],
                   "ridge_spearman_all": spearman(test.y, pr),
                   "ridge_r2_all": r2(test.y, pr),
                   "used": "; ".join(f"{d}/{s}={k}" for (d, s), k in used.items())}
            row["rf_spearman_all"] = spearman(test.y, rf.predict(test_z.X))
            row["rf_spearman_in_domain"] = spearman(y[val], rf.predict(Z(X[val])))
            for s in SOURCES:
                m = test.src == s
                row[f"spearman_{s}"] = spearman(test.y[m], p[m]) if m.sum() >= 10 else np.nan
                row[f"n_test_{s}"] = int(m.sum())
            rho = [spearman(y[tr], Z(X[tr])[:, j]) for j in range(X.shape[1])]
            j = int(np.nanargmax(np.abs(rho)))
            row["best_dim"] = j
            row["best_dim_spearman_all"] = spearman(test.y, np.sign(rho[j]) * test_z.X[:, j])
            f = lambda A: np.sign(A) * np.log1p(np.abs(A))  # i conteggi sono asimmetrici
            B_tr, B_te = f(X[tr][:, -2:]), f(D.X[:, -2:])
            mb, sb = B_tr.mean(0), B_tr.std(0) + 1e-8
            B_tr, B_te = (B_tr - mb) / sb, (B_te - mb) / sb
            pb = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(B_tr, y[tr]).predict(B_te)
            row["due_feat_spearman_all"] = spearman(test.y, pb)  # 1. lista 2 feature vs impatto vero
            row["regressore_vs_due_feat"] = spearman(pr, pb)  # 2. quanto le due liste si somigliano
            for k, nome in ((-2, "items"), (-1, "users")):  # ciascuna feature da sola
                segno = np.sign(spearman(y[tr], B_tr[:, k]))  # segno scelto solo sul training
                row[f"solo_{nome}_spearman_all"] = spearman(test.y, segno * B_te[:, k])

            rows.append(row)

            # per l'analisi di calibrazione (sezione 6)
            pv = model.predict(Z(X[val]))
            preds.append(SimpleNamespace(
                new=new, seed=seed, y=test.y, p=p, src=test.src,
                r_val=np.corrcoef(y[val], pv)[0, 1] if np.std(pv) > 0 else 0.0,
                shift_mean=np.abs(test_z.X.mean(0)).mean(),  # spostamento medio delle feature (0 = come il train)
                shift_std=test_z.X.std(0).mean()))           # ampiezza media delle feature (1 = come il train)
        r = pd.DataFrame(rows)
        r = r[r.new_dataset == new]
        print(f"   dataset nuovo = {new:16s} regressore {r.ridge_spearman_all.mean():.3f} "
              f"| solo 2 feat {r.due_feat_spearman_all.mean():.3f} "
              f"(items {r.solo_items_spearman_all.mean():.3f}, users {r.solo_users_spearman_all.mean():.3f}) "
              f"| somiglianza liste {r.regressore_vs_due_feat.mean():.3f}")
    return pd.DataFrame(rows), pd.concat(hists, ignore_index=True), preds


def plot_curves(hist, res, n_group, n_gsdb, n_rev2, out):
    names = list(dict.fromkeys(hist.new_dataset))
    fig, axes = plt.subplots(3, len(names), figsize=(5.2 * len(names), 11), squeeze=False)
    colors = {"val": "grey", "all": "k", **COLORS}
    for j, new in enumerate(names):
        h = hist[hist.new_dataset == new].groupby("epoch").mean(numeric_only=True)
        r = res[res.new_dataset == new]
        best = r.best_epoch.median()

        ax = axes[0, j]
        ax.plot(h.index, h.loss_train, label="train")
        ax.plot(h.index, h.loss_val, label="validazione")
        ax.set_title(f"dataset nuovo: {new}\nerrore (MSE) per epoca")
        ax.set_xlabel("epoca"); ax.legend()

        for row, metric, title, ylim in [(1, "spearman", "Spearman per epoca", (-0.3, 1)),
                                         (2, "r2", "R² per epoca (sotto 0 = peggio della media)", (-1, 1))]:
            ax = axes[row, j]
            ax.plot(h.index, h[f"{metric}_val"], color=colors["val"], label="validazione (dataset di train)")
            ax.plot(h.index, h[f"{metric}_test_all"], color=colors["all"], lw=2, label="dataset nuovo: tutti")
            for s in SOURCES:
                if h[f"{metric}_test_{s}"].notna().any():
                    ax.plot(h.index, h[f"{metric}_test_{s}"], color=colors[s], label=f"dataset nuovo: {s}")
            ax.axhline(r[f"ridge_{metric}_all"].mean(), color="k", ls="--", lw=1, label="ridge (soluzione esatta)")
            ax.axvline(best, color="red", ls=":", lw=1, label="epoca scelta")
            ax.axhline(0, color="k", lw=.5)
            ax.set_ylim(*ylim); ax.set_xlabel("epoca"); ax.set_title(title)
            final = h[f"{metric}_test_all"].iloc[-1]
            if final < ylim[0]:  # es. R² = -3: scrivo il valore invece di schiacciare il grafico
                ax.text(0.02, 0.04, f"dataset nuovo a fine training: {final:.2f} (fuori scala)",
                        transform=ax.transAxes, fontsize=8)
            if j == 0:
                ax.legend(fontsize=7, loc="lower right")

    fig.suptitle(f"Training: {comp_name(n_group, n_gsdb, n_rev2)} per dataset "
                 f"(media su {hist.seed.nunique()} ripetizioni)")
    fig.tight_layout()
    path = out / f"curves_g{n_group}_s{n_gsdb}_r{n_rev2}.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path.name


def plot_composition(summary, out):
    fig, ax = plt.subplots(figsize=(1.8 + 2.0 * len(summary), 4.5))
    x = np.arange(len(summary))
    cols = [("spearman_all", "tutti")] + [(f"spearman_{s}", s) for s in SOURCES]
    w = 0.8 / len(cols)
    for k, (col, lab) in enumerate(cols):
        ax.bar(x + (k - (len(cols) - 1) / 2) * w, summary[col], w, yerr=summary[col + "_std"], capsize=3,
               label=lab)
    ax.set_xticks(x, [f"{g} group\n{s} gsdb\n{r} rev2"
                      for g, s, r in zip(summary.n_group, summary.n_gsdb, summary.n_rev2)])
    ax.set_ylabel("Spearman sul dataset nuovo\n(media sui dataset)"); ax.axhline(0, color="k", lw=.5)
    ax.legend(title="difetti valutati"); ax.set_title("Effetto della composizione del training")
    fig.tight_layout()
    fig.savefig(out / "composition.png", dpi=140)
    plt.close(fig)


# =============================================================================== analisi aggiuntive
def feature_correlations(data, out):
    """Spearman tra ogni dimensione dell'embedding e la label, dentro ogni dataset e ogni tipo di difetto."""
    names = list(data)
    d = next(iter(data.values())).X.shape[1]
    rows = []
    for j in range(d):
        row = {"dim": j}
        for n, D in data.items():
            row[n] = spearman(D.y, D.X[:, j])
        for s in SOURCES:  # media sui dataset della correlazione dentro quel tipo
            vals = [spearman(D.y[D.src == s], D.X[D.src == s, j]) for D in data.values() if (D.src == s).sum() >= 30]
            row[s] = np.nanmean(vals) if vals else np.nan
        rows.append(row)
    F = pd.DataFrame(rows)
    per = F[names].values
    F["media"] = per.mean(1)
    F["stesso_segno"] = (np.sign(per) == np.sign(per[:, :1])).all(1)
    F["min_|rho|"] = np.abs(per).min(1)
    F = F.reindex(F["media"].abs().sort_values(ascending=False).index).reset_index(drop=True)
    F.to_csv(out / "feature_correlations.csv", index=False)

    top = F.head(15)
    cols = names + list(SOURCES) + ["media"]
    fig, ax = plt.subplots(figsize=(1.3 * len(cols) + 2, 6))
    im = ax.imshow(top[cols].values, cmap="RdBu_r", vmin=-.6, vmax=.6, aspect="auto")
    ax.set_xticks(range(len(cols)), cols, rotation=30, ha="right")
    ax.set_yticks(range(len(top)), [f"dim {int(i)}" for i in top.dim])
    for i in range(len(top)):
        for k, c in enumerate(cols):
            ax.text(k, i, f"{top[c].iloc[i]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title("Spearman(dimensione, label) - 15 dimensioni piu' correlate")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out / "feature_correlations.png", dpi=140)
    plt.close(fig)
    robust = int((F.stesso_segno & (F["min_|rho|"] >= 0.1)).sum())
    print(f"[correlazioni] {robust}/{d} dimensioni con |rho| >= 0.1 e stesso segno in tutti i dataset; "
          f"massimo |rho medio| = {F.media.abs().max():.3f}")
    return F, robust


def learning_curve(data, n_group, n_gsdb, n_rev2, args):
    """Spearman sul dataset nuovo (e sui dataset gia' visti) al crescere dei difetti di training.
    Le quote dei tre tipi restano quelle della composizione (con -1 = disponibili, mediana sui dataset)."""
    avail = lambda s: np.median([(D.src == s).sum() for D in data.values()])
    req = {"group_f": n_group, "gsdb_f": n_gsdb, "rev2": n_rev2}
    cnt = {s: (n if n >= 0 else avail(s)) for s, n in req.items()}
    tot = sum(cnt.values())
    share = {s: (c / tot if tot > 0 else (1.0 if s == "group_f" else 0.0)) for s, c in cnt.items()}
    share_txt = " / ".join(f"{s} {share[s]:.0%}" for s in SOURCES)
    names, rows = list(data), []
    for new in names:
        D = data[new]
        y_test = normalize_label(D.y, args.label_norm)
        for size in [16, 32, 64, 128, 256, 512, 1024, 2048]:
            ng = int(round(size * share["group_f"]))
            ns = int(round(size * share["gsdb_f"]))
            nr = max(size - ng - ns, 0) if share["rev2"] > 0 else 0
            for seed in range(args.seeds):
                rng = np.random.default_rng(1000 * seed + size)
                X, y, strata, _ = sample_train(data, [d for d in names if d != new], ng, ns, nr, rng,
                                               args.label_norm)
                tr, val = split_val(strata, 0.2, rng)
                if tr.sum() < 10 or val.sum() < 5:
                    continue
                mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
                m = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit((X[tr] - mu) / sd, y[tr])
                rows.append({"new_dataset": new, "per_dataset": size, "n_train": int(tr.sum()), "seed": seed,
                             "spearman_new": spearman(y_test, m.predict((D.X - mu) / sd)),
                             "spearman_in_domain": spearman(y[val], m.predict((X[val] - mu) / sd))})
    LC = pd.DataFrame(rows)
    LC.to_csv(args.out / "learning_curve.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for new, gdf in LC.groupby("new_dataset", sort=False):
        m = gdf.groupby("per_dataset").spearman_new.agg(["mean", "std"])
        ax.errorbar(m.index, m["mean"], m["std"], marker="o", capsize=3, label=f"dataset nuovo: {new}")
    m = LC.groupby("per_dataset").spearman_in_domain.mean()
    ax.plot(m.index, m.values, "k--", marker="s", label="dataset gia' visti (media)")
    ax.set_xscale("log", base=2); ax.set_xlabel("difetti di training per dataset")
    ax.set_ylabel("Spearman"); ax.axhline(0, color="k", lw=.5); ax.legend(fontsize=8)
    ax.set_title(f"Learning curve ({share_txt})")
    fig.tight_layout()
    fig.savefig(args.out / "learning_curve.png", dpi=140)
    plt.close(fig)
    piv = LC.groupby(["per_dataset", "new_dataset"]).spearman_new.mean().unstack()
    piv["gia' visti"] = LC.groupby("per_dataset").spearman_in_domain.mean()
    piv["n_train medio"] = LC.groupby("per_dataset").n_train.mean().round().astype(int)
    print("[learning curve]\n", piv.round(3).to_string())
    return piv, share_txt


def calibration_analysis(preds, out, max_points=4000):
    """Sezione 6: predetto vs reale sul dataset nuovo. La pendenza e' sbagliata? Basta calibrare?"""
    rows = []
    for q in preds:
        b, a = np.polyfit(q.p, q.y, 1)                                   # y ≈ a + b·p
        p_graph = q.r_val * (q.p - q.p.mean()) / (q.p.std() + 1e-8)      # calibrazione SENZA label del grafo nuovo
        rows.append({"new_dataset": q.new, "seed": q.seed,
                     "pendenza": b, "intercetta": a,
                     "media_pred": q.p.mean(), "std_pred": q.p.std(), "std_vera": q.y.std(),
                     "shift_media_feat": q.shift_mean, "shift_std_feat": q.shift_std,
                     "R2_grezzo": r2(q.y, q.p),
                     "R2_calib_grafo": r2(q.y, p_graph),
                     "R2_calib_oracolo": r2(q.y, a + b * q.p)})
    C = pd.DataFrame(rows)
    C.to_csv(out / "calibration.csv", index=False)
    tab = C.drop(columns="seed").groupby("new_dataset", sort=False).mean().reset_index()

    names = list(tab.new_dataset)
    fig, axes = plt.subplots(1, len(names), figsize=(5.2 * len(names), 5), squeeze=False)
    rng = np.random.default_rng(0)
    for j, new in enumerate(names):
        q = next(q for q in preds if q.new == new)  # prima ripetizione
        idx = rng.choice(len(q.y), min(max_points, len(q.y)), replace=False)
        ax = axes[0, j]
        for s in SOURCES:
            k = idx[q.src[idx] == s]
            if len(k):
                ax.scatter(q.p[k], q.y[k], s=6, alpha=.3, color=COLORS[s], label=s)
        b, a = np.polyfit(q.p, q.y, 1)
        xs = np.percentile(q.p, [0.5, 99.5])
        ax.plot(xs, xs, "k--", lw=1, label="perfetto: y = p")
        ax.plot(xs, a + b * xs, "r", lw=2, label=f"calibrazione: y = {a:.2f} + {b:.2f}·p")
        ax.set_xlim(*xs); ax.set_ylim(*np.percentile(q.y, [0.5, 99.5]))
        ax.set_xlabel("impatto predetto"); ax.set_ylabel("impatto vero (normalizzato nel grafo)")
        t = tab[tab.new_dataset == new].iloc[0]
        ax.set_title(f"dataset nuovo: {new}\nR² grezzo {t.R2_grezzo:.2f} · calibrato nel grafo "
                     f"{t.R2_calib_grafo:.2f} · tetto {t.R2_calib_oracolo:.2f}")
        ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "calibration.png", dpi=140)
    plt.close(fig)

    print("[calibrazione]\n", tab.round(3).to_string(index=False))
    L = ["\n## 6. Calibrazione: predetto vs reale sul dataset nuovo\n",
         "Retta di calibrazione y = a + b·p, grafico in `calibration.png`. "
         "**Pendenza ≈ 1**: la scala delle predizioni è giusta. **Pendenza < 1**: predizioni troppo sparse "
         "(troppo sicure), che danno R² negativo anche con un buon ordine. **Pendenza > 1**: predizioni troppo "
         "compresse. **Intercetta ≠ 0**: predizioni spostate. `std_pred` va confrontata con `std_vera`.\n",
         "`shift_media_feat` e `shift_std_feat` dicono quanto le feature del grafo nuovo sono spostate "
         "(0 = per niente) e stirate (1 = come il training), in unità del training. Valori lontani da 0 e da 1 "
         "sono la causa tipica di predizioni fuori scala.\n",
         "Tre R²: `R2_grezzo` = senza correzione; `R2_calib_grafo` = predizioni standardizzate dentro il grafo "
         "nuovo e moltiplicate per la correlazione di validazione, **senza usare le label del grafo nuovo** "
         "(è quello ottenibile davvero nella pipeline); `R2_calib_oracolo` = retta stimata con le label del "
         "grafo nuovo: è il tetto massimo (= Pearson²), non un risultato da riportare.\n",
         md(tab)]
    return "\n".join(L)


def target_stats(data, label, label_norm, out):
    """Sezione 7: scala del target nei dataset, prima e dopo la normalizzazione."""
    rows = []
    for d, D in data.items():
        yn_all = normalize_label(D.y, label_norm)
        for s in ("tutti",) + SOURCES:
            m = np.ones(len(D.y), bool) if s == "tutti" else D.src == s
            if m.sum() < 2:
                continue
            y, yn = D.y[m], yn_all[m]
            rows.append({"dataset": d, "tipo": s, "n": int(m.sum()),
                         "media": y.mean(), "std": y.std(), "mediana": np.median(y),
                         "p5": np.percentile(y, 5), "p95": np.percentile(y, 95),
                         "quota_zero": np.mean(y == 0),
                         "media_norm": yn.mean(), "std_norm": yn.std()})
    T = pd.DataFrame(rows)
    T.to_csv(out / "target_stats.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    lo, hi = np.percentile(np.concatenate([D.y for D in data.values()]), [1, 99])
    for d, D in data.items():
        ax.hist(np.clip(D.y, lo, hi), bins=60, range=(lo, hi), density=True, histtype="step", lw=2,
                label=f"{d} (media {D.y.mean():.3g}, std {D.y.std():.3g})")
    ax.set_xlabel(f"label '{label}' grezza (tagliata al 1°–99° percentile)"); ax.set_ylabel("densità")
    ax.set_title("Distribuzione del target per dataset, prima della normalizzazione"); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "target_stats.png", dpi=140)
    plt.close(fig)

    print("[target]\n", T.round(4).to_string(index=False))
    L = ["\n## 7. Scala del target nei dataset\n",
         f"Colonne da `media` a `quota_zero`: label **grezza**. `media_norm` e `std_norm`: dopo la normalizzazione "
         f"`{label_norm}`, cioè ciò che il modello vede davvero in training e in test. Grafico in `target_stats.png`.\n",
         "Se le colonne grezze differiscono tra dataset ma quelle normalizzate no, la scala del target **non** "
         "spiega un R² negativo: la causa va cercata nelle predizioni (sezione 6, `std_pred` e `shift_*_feat`). "
         "Con `--label-norm none` invece la scala grezza entra direttamente nell'R².\n",
         md(T, digits=4)]
    return "\n".join(L)


# =============================================================================== modello finale
def train_final_old(data, n_group, n_gsdb, n_rev2, args):
    rng = np.random.default_rng(12345)
    X, y, strata, used = sample_train(data, list(data), n_group, n_gsdb, n_rev2, rng, args.label_norm)
    tr, val = split_val(strata, 0.2, rng)
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
    model, hist, best_ep = train_with_history((X[tr] - mu) / sd, y[tr], (X[val] - mu) / sd, y[val], args, rng)
    bundle = {"w": model.w, "b": model.b, "x_mean": mu, "x_std": sd, "label": args.label,
              "label_norm": args.label_norm, "n_group": n_group, "n_gsdb": n_gsdb, "n_rev2": n_rev2,
              "datasets": list(data), "best_epoch": best_ep, "lr": args.lr, "weight_decay": args.weight_decay}
    with open(args.out / "final_model.pkl", "wb") as f:
        pickle.dump(bundle, f)
    return bundle, used, hist.spearman_val.iloc[best_ep - 1]

def train_final(data, n_group, n_gsdb, n_rev2, args):
    """Modello finale: ridge (soluzione esatta) su TUTTI i difetti campionati dei dataset."""
    rng = np.random.default_rng(12345)
    X, y, _, used = sample_train(data, list(data), n_group, n_gsdb, n_rev2, rng, args.label_norm)
    mu, sd = X.mean(0), X.std(0) + 1e-8
    ridge = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit((X - mu) / sd, y)
    bundle = {"w": ridge.coef_.copy(), "b": float(ridge.intercept_), "x_mean": mu, "x_std": sd,
              "alpha": float(ridge.alpha_), "label": args.label, "label_norm": args.label_norm,
              "n_group": n_group, "n_gsdb": n_gsdb, "n_rev2": n_rev2, "datasets": list(data), "n_train": len(y),
              # servono a online_update, che continua a usare Adam partendo da questi pesi
              "lr": args.lr, "weight_decay": args.weight_decay}
    with open(args.out / "final_model.pkl", "wb") as f:
        pickle.dump(bundle, f)
    return bundle, used, None


def predict(bundle, X):
    """Punteggio di impatto per ogni difetto (piu' alto = piu' dannoso). Serve per ORDINARE i difetti."""
    return ((np.asarray(X) - bundle["x_mean"]) / bundle["x_std"]) @ bundle["w"] + bundle["b"]


def online_update(bundle, X_new, y_new, epochs=5, batch=32, seed=0):
    """Aggiornamento nella fase attiva (A5): pochi passi di Adam sulle nuove label, normalizzate nel grafo."""
    m = LinearRegressorAdam(len(bundle["w"]), bundle["lr"], bundle["weight_decay"])
    m.set((bundle["w"], bundle["b"]))
    Z = (np.asarray(X_new) - bundle["x_mean"]) / bundle["x_std"]
    y = normalize_label(np.asarray(y_new, float), bundle["label_norm"])
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        m.fit_epoch(Z, y, batch, rng)
    bundle = dict(bundle, w=m.w, b=m.b)
    return bundle


# =============================================================================== recap
def md(df, digits=3):
    def f(v):
        if isinstance(v, (float, np.floating)):
            return "—" if np.isnan(v) else f"{v:.{digits}f}"
        return str(v)
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "|".join(["---"] * df.shape[1]) + "|"]
    lines += ["| " + " | ".join(f(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


def quality(v):
    if np.isnan(v):
        return "n/d"
    return "a caso" if v < 0.1 else "debole" if v < 0.3 else "discreto" if v < 0.5 else "buono"


def label_desc(label):
    """Descrizione della label per il RECAP."""
    if label not in SEG_LABELS:
        return ""
    k_imp, k_n = SEG_LABELS[label]
    who = "per utente affetto" if label == "rr" else f"per utente {label[3:]} affetto"
    return f" (= {k_imp} / {k_n}: calo medio di rr_best {who})"


COMP = ["n_group", "n_gsdb", "n_rev2"]


def write_recap(args, data, results, summary, curve_files, final, analyses=None, extra=None):
    bundle, used_final, val_sp = final
    L = ["# SCRIBA – regressore lineare su embedding GraphSAGE\n",
         f"- Label: **{args.label}**"
         + label_desc(args.label)
         + f", normalizzata dentro ogni dataset con `{args.label_norm}`",
         f"- Dataset: {', '.join(data)} · modello lineare allenato {args.epochs} epoche "
         f"(Adam, lr {args.lr}, L2 {args.weight_decay}, batch {args.batch}), "
         f"{args.seeds} ripetizioni per configurazione\n",
         "## Come leggere",
         "- **Spearman** misura se il modello **ordina** i difetti come l'impatto vero: 1 = ordine perfetto, "
         "0 = come a caso, negativo = al contrario. Indicativamente: < 0.1 a caso, 0.1–0.3 debole, "
         "0.3–0.5 discreto, > 0.5 buono. È la metrica che conta, perché la pipeline ordina i difetti di un grafo.",
         "- **Dataset nuovo** = il dataset escluso dal training: è la stima di come andrà su Books, yelp, Beauty.",
         "- **group_f / gsdb_f / rev2** = Spearman calcolato solo sui difetti di quel tipo del dataset nuovo "
         "(— se il dataset nuovo ne ha meno di 10).",
         "- **R²** = quanto il valore predetto si avvicina al valore vero, dopo aver normalizzato l'impatto "
         "del dataset nuovo (si confronta la forma, non la scala). Sotto 0 = peggio che predire la media. "
         "Il tetto realistico è circa il quadrato della correlazione.",
         "- **Ridge** = la soluzione esatta del modello lineare sugli stessi dati. Se lo Spearman del modello "
         "a epoche è vicino, l'allenamento ha funzionato; se è molto sotto, servono più epoche o un lr diverso.",
         "- **Epoca migliore** = scelta guardando la validazione (20% dei dati di training), mai il test.\n"]

    L.append("## 1. Generalizzazione a un dataset nuovo\n")
    L.append("`gia_visti` = Spearman sulla validazione, cioè su difetti nuovi ma dei dataset di training. "
             "La differenza con `spearman_tutti` è quanto si perde passando a un dataset mai visto.\n")
    per_ds = (results.groupby(COMP + ["new_dataset"], sort=False)
              .agg(gia_visti=("spearman_in_domain", "mean"),
                   spearman_tutti=("spearman_all", "mean"), group_f=("spearman_group_f", "mean"),
                   gsdb_f=("spearman_gsdb_f", "mean"), rev2=("spearman_rev2", "mean"), R2=("r2_all", "mean"),
                   ridge=("ridge_spearman_all", "mean"), R2_ridge=("ridge_r2_all", "mean"),
                   epoca=("best_epoch", "median"), n_train=("n_train", "mean"))
              .reset_index())
    per_ds[["epoca", "n_train"]] = per_ds[["epoca", "n_train"]].round().astype(int)
    L.append(md(per_ds))

    L.append("\n## 2–3. Media sui dataset nuovi, per composizione del training\n")
    show = summary[COMP + ["spearman_in_domain", "spearman_all", "spearman_group_f",
                           "spearman_gsdb_f", "spearman_rev2", "ridge_spearman_all",
                           "r2_in_domain", "r2_all"]].rename(columns={
                        "spearman_in_domain": "Spearman già visti", "r2_in_domain": "R² già visti",
                        "spearman_all": "Spearman tutti", "spearman_group_f": "Spearman group_f",
                        "spearman_gsdb_f": "Spearman gsdb_f", "spearman_rev2": "Spearman rev2",
                        "ridge_spearman_all": "ridge", "r2_all": "R² dataset nuovo"})
    L.append(md(show))
    L.append("\nCurve epoca per epoca (MSE, Spearman, R²): " + ", ".join(f"`{c}`" for c in curve_files)
             + (" · confronto composizioni: `composition.png`" if len(summary) > 1 else ""))

    if analyses is not None:
        F, robust, piv, share_txt = analyses
        names = list(data)
        L.append("\n## 4. Correlazione tra dimensioni dell'embedding e label\n")
        L.append("Spearman tra ogni dimensione e la label, calcolato dentro ogni dataset (colonne dataset) e dentro "
                 "ogni tipo di difetto (colonne group_f / gsdb_f / rev2, media sui dataset). Una dimensione è "
                 "**robusta** se ha lo stesso segno in tutti i dataset e |rho| ≥ 0.1 in ognuno. "
                 "Le correlazioni della singola dimensione sono tipicamente basse: il modello le combina tutte. "
                 "Tabella completa in `feature_correlations.csv`, grafico in `feature_correlations.png`.\n")
        L.append(f"- Dimensioni robuste: **{robust}/{len(F)}**; |rho medio| massimo = {F.media.abs().max():.3f}\n")
        L.append(md(F.head(10)[["dim"] + names + list(SOURCES) + ["media", "stesso_segno"]]
                    .astype({"dim": int}).assign(stesso_segno=lambda t: t.stesso_segno.map({True: "sì", False: "no"}))))
        L.append(f"\n## 5. Learning curve ({share_txt})\n")
        L.append("Spearman sul dataset nuovo al crescere dei difetti di training per dataset (modello ridge, "
                 "equivalente al modello a epoche). Grafico in `learning_curve.png`.\n")
        L.append(md(piv.reset_index().rename(columns={"per_dataset": "difetti per dataset"})
                    .astype({"difetti per dataset": int})))

    if extra:
        L.extend(extra)

    # sintesi automatica
    L.append("\n## In sintesi\n")
    summary = summary.astype({c: int for c in COMP})
    best = summary.sort_values("spearman_all", ascending=False).iloc[0]
    L.append(f"- Migliore composizione complessiva: **{comp_name(*(int(best[c]) for c in COMP))}** "
             f"→ Spearman {best.spearman_all:.3f} ({quality(best.spearman_all)}) su dataset mai visti.")
    for s in SOURCES:
        if summary[f"spearman_{s}"].isna().all():
            L.append(f"- Difetti **{s}**: nessun dataset nuovo ne ha abbastanza (almeno 10) per lo Spearman.")
            continue
        b = summary.sort_values(f"spearman_{s}", ascending=False).iloc[0]
        L.append(f"- Per i difetti **{s}** conviene {comp_name(*(int(b[c]) for c in COMP))} "
                 f"(Spearman {b[f'spearman_{s}']:.3f}, {quality(b[f'spearman_{s}'])}).")
    loss = best.spearman_in_domain - best.spearman_all
    L.append(f"- Passando da dataset già visti a un dataset nuovo lo Spearman passa da "
             f"{best.spearman_in_domain:.3f} a {best.spearman_all:.3f} ({-loss:+.3f}).")
    if analyses is not None:
        newcols = [c for c in piv.columns if c in data]
        curve = piv[newcols].mean(1)
        plateau = next((int(k) for k, v in curve.items() if v >= 0.95 * curve.max()), None)
        L.append(f"- Learning curve: sul dataset nuovo lo Spearman medio va da {curve.iloc[0]:.3f} "
                 f"({int(curve.index[0])} difetti per dataset) a {curve.max():.3f}; arriva al 95% del massimo con "
                 f"~{plateau} difetti per dataset" + (" → oltre, più difetti aiutano poco." if plateau else "."))
        L.append(f"- Correlazioni: {robust}/{len(F)} dimensioni robuste su tutti i dataset.")
    gap = (per_ds.ridge - per_ds.spearman_tutti).max()  # per singolo dataset, non sulla media
    L.append(f"- Allenamento a epoche vs soluzione esatta: differenza massima {gap:+.3f} di Spearman "
             "(sul dataset peggiore) → "
             + ("converge bene." if gap < 0.03 else
                "c'è una differenza: il modello a epoche non è arrivato alla soluzione esatta. "
                "Prova più epoche, un batch più piccolo (più passi per epoca) o un lr diverso."))
    in_best = per_ds[np.logical_and.reduce([per_ds[c] == best[c] for c in COMP])]
    worst = in_best.loc[in_best.spearman_tutti.idxmin()]
    L.append(f"- Con la composizione migliore, il dataset nuovo più difficile è **{worst.new_dataset}** "
             f"(Spearman {worst.spearman_tutti:.3f}).")
    L.append("- Attenzione: 'Spearman tutti' è dominato dal tipo di difetto più numeroso (di solito group_f). "
             "Per capire l'effetto di gsdb_f e rev2 guarda le loro colonne.")
    L.append("- Nota: se un dataset ha meno difetti del richiesto, si usano tutti quelli disponibili "
             "(Baby_Products ha solo 47 gsdb_f). I numeri effettivi sono nella colonna `used` di `results.csv`.")
    if args.label in SEG_LABELS and args.label != "rr":
        L.append(f"- Label di segmento: entrano solo i difetti che toccano almeno un utente "
                 f"{args.label[3:]} (`{SEG_LABELS[args.label][1]}` > 0); gli altri non hanno label.")
    L.append(f"\n## Modello finale\n\nRidge allenato su tutti i dataset ({', '.join(bundle['datasets'])}) con "
             f"{comp_name(bundle['n_group'], bundle['n_gsdb'], bundle['n_rev2'])} per dataset "
             f"({bundle['n_train']} difetti, alpha scelto {bundle['alpha']:.3g}). Salvato in `final_model.pkl`. "
             "Nella fase attiva: `predict(bundle, X)` per ordinare i difetti, "
             "`online_update(bundle, X_nuovi, y_nuovi)` per aggiornarlo con le nuove label (A5).")
    text = "\n".join(L)
    (args.out / "RECAP.md").write_text(text, encoding="utf-8")
    print("\n" + "=" * 90 + "\n" + text + "\n" + "=" * 90)


# =============================================================================== main
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emb-dir", type=Path, default=DEFAULT_EMB_DIR)
    ap.add_argument("--datasets", nargs="+", default=["Baby_Products", "Toys_and_Games", "Pet_Supplies"])
    ap.add_argument("--label", default="rr", help="rr | rr_active | rr_lt | entropy | altra chiave del pickle")
    ap.add_argument("--labels", nargs="+", default=None,
                    help="piu' label in un colpo solo, es. rr rr_active rr_lt: un modello per label, "
                         "ognuno nella sua sottocartella di --out, modelli finali copiati in --out")
    ap.add_argument("--label-norm", choices=["zscore", "rank", "none"], default="rank")
    ap.add_argument("--n-group", type=int, default=800, help="group_f per dataset di training (-1 = tutti)")
    ap.add_argument("--n-gsdb", type=int, default=500, help="gsdb_f per dataset di training (-1 = tutti)")
    ap.add_argument("--n-rev2", type=int, default=500, help="rev2 per dataset di training (-1 = tutti)")
    ap.add_argument("--grid", nargs="+", default=None, metavar="G:S[:R]",
                    help="composizioni da confrontare, es. 600:0:0 300:300:0 200:200:200 "
                         "(senza R si usa --n-rev2; la prima e' usata per il modello finale)")
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--seeds", type=int, default=5, help="ripetizioni con campioni diversi")
    ap.add_argument("--out", type=Path, default=ROOT_DIR / "scriba_regressor_out")
    ap.add_argument("--skip-analyses", action="store_true", help="salta correlazioni e learning curve")
    return ap.parse_args()


def parse_grid(grid, n_rev2_default):
    configs = []
    for g in grid:
        v = [int(x) for x in g.split(":")]
        if len(v) == 2:
            v.append(n_rev2_default)
        if len(v) != 3:
            raise SystemExit(f"--grid: '{g}' non valido, usa G:S oppure G:S:R")
        configs.append(tuple(v))
    return configs


def run_label(args):
    """Tutto il protocollo per UNA label: analisi, RECAP e modello finale in args.out."""
    args.out.mkdir(parents=True, exist_ok=True)
    data = load_data(args.emb_dir, args.datasets, args.label)
    configs = (parse_grid(args.grid, args.n_rev2) if args.grid
               else [(args.n_group, args.n_gsdb, args.n_rev2)])

    all_res, curve_files, first_preds = [], [], None
    for n_group, n_gsdb, n_rev2 in configs:
        print(f"\n[composizione] {comp_name(n_group, n_gsdb, n_rev2)} per dataset di training")
        res, hist, preds = run_config(data, n_group, n_gsdb, n_rev2, args)
        if first_preds is None:  # la calibrazione si fa sulla prima composizione (quella del modello finale)
            first_preds = preds
        all_res.append(res)
        curve_files.append(plot_curves(hist, res, n_group, n_gsdb, n_rev2, args.out))
    results = pd.concat(all_res, ignore_index=True)
    results.to_csv(args.out / "results.csv", index=False)

    # media prima sulle ripetizioni, poi sui dataset nuovi
    cols = ["spearman_in_domain", "spearman_all", "spearman_group_f", "spearman_gsdb_f", "spearman_rev2",
            "ridge_spearman_all", "r2_in_domain", "r2_all"]
    per_ds = results.groupby(COMP + ["new_dataset"], sort=False)[cols].mean().reset_index()
    summary = per_ds.groupby(COMP, sort=False)[cols].agg(["mean", "std"])
    summary.columns = [c if s == "mean" else f"{c}_std" for c, s in summary.columns]
    summary = summary.reset_index()
    if len(summary) > 1:
        plot_composition(summary, args.out)

    analyses = None
    if not args.skip_analyses:
        F, robust = feature_correlations(data, args.out)
        piv, share_txt = learning_curve(data, *configs[0], args)
        analyses = (F, robust, piv, share_txt)
    extra = [calibration_analysis(first_preds, args.out),
             target_stats(data, args.label, args.label_norm, args.out)]
    final = train_final(data, *configs[0], args)
    write_recap(args, data, results, summary, curve_files, final, analyses, extra)
    print(f"\n[fatto] risultati in {args.out}")
    return summary


def main():
    args = parse_args()
    pd.set_option("display.max_columns", None)
    labels = args.labels or [args.label]
    if len(labels) == 1:                       # come prima: tutto direttamente in --out
        args.label = labels[0]
        run_label(args)
        return

    root = args.out
    root.mkdir(parents=True, exist_ok=True)
    recap = []
    for lab in labels:
        a = copy.copy(args)
        a.label, a.out = lab, root / lab
        print("\n" + "#" * 90 + f"\n# LABEL {lab}\n" + "#" * 90)
        summary = run_label(a)
        name = MODEL_NAME.get(lab, f"final_model_{lab}.pkl")
        shutil.copyfile(a.out / "final_model.pkl", root / name)
        s = summary.iloc[0]
        recap.append({"label": lab, "modello": name,
                      "Spearman già visti": s.spearman_in_domain, "Spearman dataset nuovo": s.spearman_all,
                      "ridge": s.ridge_spearman_all, "R² dataset nuovo": s.r2_all})

    R = pd.DataFrame(recap)
    text = ("# Modelli per segmento\n\n"
            "Un regressore per label, allenato con lo stesso protocollo. Dettagli nel `RECAP.md` di ogni "
            "sottocartella. Numeri = media sui dataset nuovi (prima composizione).\n\n" + md(R) +
            "\n\nPer mab_defects.py: `--regressor final_model.pkl --regressor-active final_model_active.pkl "
            "--regressor-lt final_model_lt.pkl` (con `--yhat seg`).\n")
    (root / "RECAP_segmenti.md").write_text(text, encoding="utf-8")
    print("\n" + text)
    print(f"[fatto] modelli finali in {root}: " + ", ".join(r["modello"] for r in recap))


if __name__ == "__main__":
    main()