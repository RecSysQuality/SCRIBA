#!/usr/bin/env python3
"""
mab_report.py
=============
Legge i run di mab_defects.py (results/mab/<dataset>/<target>_<policy>_s<seed>/)
e scrive REPORT.md + grafici, pronti per il paper.

    python mab_report.py results/mab/Books

Sezioni
  1. Reward per target: cumulata, AUC, percentile rispetto alla sonda casuale (+ test)
  2. Matrice di specializzazione: reward cumulata su ogni segmento per ogni target
  3. Liste diverse per segmento: Jaccard tra liste e tra insiemi di difetti rimossi
  4. Effetto sullo stato per segmento (Δ finale di entropia, gini, coverage, novelty, APLT, NDCG, ...)
  5. Convergenza del MAB: Δθ, bonus, Spearman sonda/holdout, round di convergenza
  6. Regressore online: Spearman e R² round per round, online vs congelato
  7. Additività: effetto congiunto del batch vs somma degli effetti singoli
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
_trapz = getattr(np, "trapezoid", None) or np.trapz

GROUPS = ("all", "active", "lt")
PAIRS = (("lt", "active"), ("lt", "all"), ("active", "all"))
PAPER_METRICS = ("entropy", "gini", "coverage", "novelty", "aplt", "ndcg", "recall", "churn", "defect_share")
MABS = ("mab", "mab_cold")
COLORS = {"mab": "tab:red", "mab_cold": "tab:orange", "regressor": "tab:purple", "push": "tab:blue", "random": "tab:gray"}


def md(df, digits=3):
    def f(v):
        if isinstance(v, (float, np.floating)):
            return "—" if not np.isfinite(v) else f"{v:.{digits}f}"
        return str(v)
    cols = list(df.columns)
    L = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    L += ["| " + " | ".join(f(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(L)


def load_runs(root: Path):
    runs = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        if not (d / "run.json").exists() or not (d / "rounds.tsv").exists():
            continue
        meta = json.loads((d / "run.json").read_text())
        # etichetta = politica (+ pool di candidati); "base" = politica senza pool
        meta["base"] = meta["policy"]
        meta["pool"] = meta.get("pool", "none")
        if meta["pool"] != "none":
            meta["policy"] = f"{meta['base']} (pool {meta['pool']})"
        R = pd.read_csv(d / "rounds.tsv", sep="\t")
        R = R.assign(target=meta["target"], policy=meta["policy"], seed=meta["seed"], run=d.name)
        rec = pd.read_csv(d / "recommendations.tsv", sep="\t") if (d / "recommendations.tsv").exists() else None
        mea = pd.read_csv(d / "measurements.tsv", sep="\t") if (d / "measurements.tsv").exists() else None
        runs.append({"meta": meta, "rounds": R, "rec": rec, "mea": mea, "dir": d})
    return runs


def _steps(R):
    """Solo i round (senza la riga iniziale round=-1)."""
    return R[R["round"] >= 0] if "round" in R else R


POOL_LS = {"none": "-", "regressor": "--", "random": ":"}


def _sty(runs, label):
    m = next(r["meta"] for r in runs if r["meta"]["policy"] == label)
    return dict(color=COLORS.get(m["base"]), ls=POOL_LS.get(m["pool"], "-"))


def _col(R, c):
    return R[c] if c in R else pd.Series(np.nan, index=R.index)


def _mean_curve(dfs, x, y):
    """Media (e std) tra seed, allineando per indice di round."""
    ys = [d[y].to_numpy(float) for d in dfs if y in d]
    xs = [d[x].to_numpy(float) for d in dfs if y in d]
    if not ys:
        return None
    n = min(len(v) for v in ys)
    Y = np.vstack([v[:n] for v in ys])
    X = np.vstack([v[:n] for v in xs]).mean(0)
    return X, np.nanmean(Y, 0), np.nanstd(Y, 0)


def convergence_round(theta_delta, tol=0.01, k=3):
    v = np.asarray(theta_delta, float)
    for i in range(len(v) - k + 1):
        if np.all(v[i:i + k] < tol):
            return i
    return np.nan


# ---------------------------------------------------------------------------- figure
def fig_reward(runs, target, out):
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
    for pol in sorted({r["meta"]["policy"] for r in runs}):
        S = [_steps(r["rounds"]) for r in runs if r["meta"]["policy"] == pol]
        st = _sty(runs, pol)
        c = st["color"]
        for k, (x, y, lab) in enumerate([("edges_after", f"cum_{target}", "reward cumulata"),
                                         ("round", f"pct_{target}", "percentile vs sonda"),
                                         ("round", f"r_{target}_mean", "reward media dei scelti")]):
            mc = _mean_curve(S, x, y)
            if mc is None:
                continue
            X, M, SD = mc
            if k == 1 and len(M) >= 3:
                M = pd.Series(M).rolling(3, min_periods=1).mean().to_numpy()
            ax[k].plot(X, M, color=c, ls=st["ls"], label=pol, lw=2)
            ax[k].fill_between(X, M - SD, M + SD, color=c, alpha=.15)
    ax[0].set_xlabel("archi rimossi"); ax[0].set_ylabel(f"Σ r_{target} (scalata)")
    ax[0].set_title(f"Reward cumulata sul target {target}")
    ax[1].axhline(0.5, color="k", ls="--", lw=1)
    ax[1].set_ylim(0, 1); ax[1].set_xlabel("round"); ax[1].set_title("Percentile dei scelti nella sonda (media mobile 3)")
    ax[2].set_xlabel("round"); ax[2].set_title("Reward media per difetto scelto")
    for a in ax:
        a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)


def fig_state(runs, target, out):
    mets = [m for m in PAPER_METRICS if any(f"lt_{m}" in r["rounds"] and r["rounds"][f"lt_{m}"].notna().any()
                                            for r in runs)]
    if not mets:
        return False
    fig, axes = plt.subplots(len(mets), 3, figsize=(14, 2.6 * len(mets)), squeeze=False)
    for pol in sorted({r["meta"]["policy"] for r in runs}):
        S = [r["rounds"] for r in runs if r["meta"]["policy"] == pol]
        for i, m in enumerate(mets):
            for j, g in enumerate(GROUPS):
                mc = _mean_curve(S, "edges_after", f"{g}_d_{m}")
                if mc is None:
                    continue
                X, M, SD = mc
                st = _sty(runs, pol)
                axes[i, j].plot(X, M, color=st["color"], ls=st["ls"], label=pol, lw=2 if g == target else 1.2)
                axes[i, j].fill_between(X, M - SD, M + SD, color=st["color"], alpha=.12)
                axes[i, j].axhline(0, color="k", lw=.5)
                if i == 0:
                    axes[i, j].set_title(f"segmento {g}" + ("  (TARGET)" if g == target else ""))
                if j == 0:
                    axes[i, j].set_ylabel(f"Δ {m}")
    for a in axes[-1]:
        a.set_xlabel("archi rimossi")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"Target {target}: variazione delle metriche di stato rispetto al grafo iniziale")
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    return True


def fig_convergence(runs, target, out):
    labels = sorted({r["meta"]["policy"] for r in runs if r["meta"]["base"] in MABS})
    per = {p: [r["rounds"] for r in runs if r["meta"]["policy"] == p] for p in labels}
    if not per:
        return False
    fig, ax = plt.subplots(2, 3, figsize=(16, 8))
    panels = [(ax[0, 0], ["theta_delta"], "Δθ relativo (log)", True),
              (ax[0, 1], ["bonus_frac"], "bonus / UCB dei scelti (esplorazione)", False),
              (ax[0, 2], [f"sigma_{g}" for g in GROUPS], "σ del rumore stimato", False),
              (ax[1, 0], [f"sp_probe_{g}" for g in GROUPS], "Spearman(predetto, vero) sulla sonda", False),
              (ax[1, 1], [f"sp_hold_{g}" for g in GROUPS], "Spearman sull'holdout a fine round (-1 = iniziale)", False),
              (ax[1, 2], ["rank_stability"], "stabilità dell'ordinamento (Spearman t vs t-1)", False)]
    for a, cols, title, logy in panels:
        for pol, S in per.items():
            for k, c in enumerate(cols):
                mc = _mean_curve(S, "round", c)
                if mc is None:
                    continue
                X, M, SD = mc
                if len(M) >= 3 and "sp_" in c:
                    M = pd.Series(M).rolling(3, min_periods=1).mean().to_numpy()
                ls = ["-", "--", ":", "-."][labels.index(pol) % 4]
                a.plot(X, M, lw=2 if target in c else 1.2, color=f"C{k}", ls=ls, label=f"{pol}: {c}")
        a.set_title(title); a.set_xlabel("round")
        if logy:
            a.set_yscale("log"); a.axhline(0.01, color="k", ls=":", lw=1)
        a.legend(fontsize=7)
    fig.suptitle(f"Convergenza del MAB — target {target} (stile di linea = variante)")
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    return True


def fig_regressor(runs, target, out):
    S = [r["rounds"] for r in runs if any(c.startswith("reg_") for c in r["rounds"].columns)]
    if not S:
        return False
    g = target if any(f"reg_hold_{target}_sp" in s for s in S) else "all"
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
    for a, split in zip(ax, ("hold", "probe", "chosen")):
        for suf, ls in (("", "-"), ("_frozen", "--")):
            for met, col in (("sp", "tab:red"), ("r2", "tab:blue")):
                mc = _mean_curve(S, "round", f"reg_{split}_{g}_{met}{suf}")
                if mc is None:
                    continue
                X, M, _ = mc
                if split != "hold" and len(M) >= 3:
                    M = pd.Series(M).rolling(3, min_periods=1).mean().to_numpy()
                a.plot(X, M, ls=ls, color=col, label=f"{'Spearman' if met == 'sp' else 'R²'} "
                                                     f"{'congelato' if suf else 'online'}")
        a.axhline(0, color="k", lw=.5)
        a.set_title({"hold": "holdout fisso, a fine round (-1 = iniziale)", "probe": "sonda casuale (prequenziale)",
                     "chosen": "difetti scelti (prequenziale, distorto)"}[split])
        a.set_xlabel("round"); a.legend(fontsize=7)
    fig.suptitle(f"Regressore online (label {g}) — linee continue: aggiornato, tratteggiate: modello di partenza")
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    return True


def fig_segments(runs, out):
    S = {t: [_steps(r["rounds"]) for r in runs
             if r["meta"]["base"] == "mab" and r["meta"]["pool"] == "none" and r["meta"]["target"] == t]
         for t in GROUPS}
    S = {t: v for t, v in S.items() if v}
    if not S:
        return False
    fig, ax = plt.subplots(figsize=(7, 4))
    t = next(iter(S))
    for a, b in PAIRS:
        mc = _mean_curve(S[t], "round", f"jac_{a}_{b}")
        if mc is not None:
            ax.plot(mc[0], mc[1], lw=2, label=f"{a} vs {b}")
    ax.set_ylim(0, 1); ax.set_xlabel("round"); ax.set_ylabel("Jaccard tra liste raccomandate")
    ax.set_title("Quanto differiscono le liste per segmento (MAB)"); ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)
    return True


# ---------------------------------------------------------------------------- tabelle
def table_reward(runs, target):
    from scipy.stats import wilcoxon
    rows = []
    for pol in sorted({r["meta"]["policy"] for r in runs}):
        per = []
        for r in (r for r in runs if r["meta"]["policy"] == pol):
            S = _steps(r["rounds"])
            if not len(S):
                continue
            cum = _col(S, f"cum_{target}").to_numpy(float)
            e = _col(S, "edges_after").to_numpy(float)
            auc = _trapz(cum, e) / (e[-1] ** 2 / 2) if len(e) > 1 and e[-1] > 0 else np.nan
            pct = _col(S, f"pct_{target}").dropna().to_numpy(float)
            p = wilcoxon(pct - 0.5, alternative="greater").pvalue if len(pct) >= 6 and np.any(pct != 0.5) else np.nan
            per.append({"cum": cum[-1], "auc": auc, "pct": pct.mean() if len(pct) else np.nan,
                        "top20": _col(S, f"top20_{target}").mean(), "p": p, "rounds": len(S),
                        "edges": e[-1], "n_rem": _col(S, "n_removed").sum()})
        if per:
            P = pd.DataFrame(per)
            rows.append({"politica": pol, "seed": len(P), "round": int(P.rounds.mean()),
                         "archi rimossi": int(P.edges.mean()), "difetti rimossi": int(P.n_rem.mean()),
                         f"Σ r_{target}": P.cum.mean(), "± std": P.cum.std() if len(P) > 1 else np.nan,
                         "AUC norm.": P.auc.mean(), "percentile vs sonda": P.pct.mean(),
                         "quota sopra 80° pct": P.top20.mean(), "p Wilcoxon (pct>0.5)": P.p.mean()})
    return pd.DataFrame(rows)


def table_specialization(runs, policy="mab"):
    rows = []
    for t in GROUPS:
        rs = [_steps(r["rounds"]) for r in runs if r["meta"]["policy"] == policy and r["meta"]["target"] == t]
        rs = [s for s in rs if len(s)]
        if not rs:
            continue
        rows.append({"target": t, **{f"Σ r_{g}": np.mean([s[f"cum_{g}"].iloc[-1] for s in rs]) for g in GROUPS}})
    return pd.DataFrame(rows)


def table_removed_overlap(runs, policy="mab"):
    sets = {}
    for r in runs:
        if r["meta"]["policy"] != policy or r["mea"] is None:
            continue
        m = r["mea"]
        m = m[(m.kind == "chosen") & (m.removed == 1)]
        sets.setdefault(r["meta"]["target"], set()).update(zip(m.source, m.defect_id))
    rows = []
    for a, b in PAIRS:
        if a in sets and b in sets:
            A, B = sets[a], sets[b]
            rows.append({"target A": a, "target B": b, "|A|": len(A), "|B|": len(B),
                         "in comune": len(A & B), "Jaccard": len(A & B) / max(len(A | B), 1)})
    return pd.DataFrame(rows)


def table_state(runs, target):
    rows = []
    for pol in sorted({r["meta"]["policy"] for r in runs}):
        fin = [r["rounds"].iloc[-1] for r in runs if r["meta"]["policy"] == pol]
        if not fin:
            continue
        for g in GROUPS:
            row = {"politica": pol, "segmento": g + (" ★" if g == target else "")}
            for m in PAPER_METRICS:
                v = [f.get(f"{g}_d_{m}", np.nan) for f in fin]
                row[f"Δ{m}"] = float(np.nanmean(v)) if np.isfinite(v).any() else np.nan
            rows.append(row)
    T = pd.DataFrame(rows)
    return T.dropna(axis=1, how="all")


def table_state_initial(runs):
    r0 = runs[0]["rounds"].iloc[0]
    rows = [{"segmento": g, **{m: r0.get(f"{g}_{m}", np.nan) for m in PAPER_METRICS if m not in ("churn",)}}
            for g in GROUPS]
    return pd.DataFrame(rows).dropna(axis=1, how="all")


def catch_up_round(S, level, col, w=3):
    """Primo round da cui la media mobile di col resta >= level fino alla fine."""
    v = _col(S, col).rolling(w, min_periods=1).mean().to_numpy(float)
    ok = v >= level
    for i in range(len(ok)):
        if ok[i:].all():
            return i
    return np.nan


def table_convergence(runs, target):
    rows = []
    def warm_level(pool):
        w = [_steps(r["rounds"]) for r in runs if r["meta"]["base"] == "mab" and r["meta"]["pool"] == pool]
        return np.nanmean([_col(S, f"pct_{target}").mean() for S in w]) if w else np.nan
    for r in runs:
        if r["meta"]["base"] not in MABS:
            continue
        warm_pct = warm_level(r["meta"]["pool"])
        S = _steps(r["rounds"])
        if len(S) < 2:
            continue
        k = max(1, len(S) // 3)
        row = {"politica": r["meta"]["policy"], "seed": r["meta"]["seed"], "round": len(S),
               "round casuali iniziali": int(_col(S, "random_phase").fillna(0).sum()),
               "round per raggiungere il percentile di mab":
                   catch_up_round(S, warm_pct, f"pct_{target}")
                   if np.isfinite(warm_pct) and r["meta"]["base"] == "mab_cold" else np.nan,
               "round di convergenza (Δθ<1% x3)": convergence_round(_col(S, "theta_delta")),
               "Δθ finale": _col(S, "theta_delta").iloc[-1],
               "bonus/UCB inizio": _col(S, "bonus_frac").iloc[:k].mean(),
               "bonus/UCB fine": _col(S, "bonus_frac").iloc[-k:].mean()}
        for g in GROUPS:
            row[f"sp sonda {g} inizio"] = _col(S, f"sp_probe_{g}").iloc[:k].mean()
            row[f"sp sonda {g} fine"] = _col(S, f"sp_probe_{g}").iloc[-k:].mean()
        row[f"sp holdout {target} inizio"] = _col(r["rounds"], f"sp_hold_{target}").iloc[0]
        row[f"sp holdout {target} fine"] = _col(S, f"sp_hold_{target}").iloc[-1]
        rows.append(row)
    return pd.DataFrame(rows)


def table_regressor(runs, target):
    rows = []
    for r in runs:
        S = _steps(r["rounds"])
        g = target if f"reg_hold_{target}_sp" in S else "all"
        if f"reg_hold_{g}_sp" not in S and f"reg_probe_{g}_sp" not in S:
            continue
        k = max(1, len(S) // 3)
        row = {"politica": r["meta"]["policy"], "seed": r["meta"]["seed"], "label": g}
        for split in ("hold", "probe"):
            for met in ("sp", "r2"):
                c = _col(S, f"reg_{split}_{g}_{met}")
                cf = _col(S, f"reg_{split}_{g}_{met}_frozen")
                if split == "hold":
                    row[f"{met} holdout inizio"] = _col(r["rounds"], f"reg_{split}_{g}_{met}").iloc[0]
                    row[f"{met} holdout fine"] = c.iloc[-1]
                    row[f"{met} holdout congelato"] = cf.iloc[-1]
                else:
                    row[f"{met} sonda primi 1/3"] = c.iloc[:k].mean()
                    row[f"{met} sonda ultimi 1/3"] = c.iloc[-k:].mean()
                    row[f"{met} sonda congelato ultimi 1/3"] = cf.iloc[-k:].mean()
        row["label online"] = _col(S, "reg_n_online").iloc[-1]
        rows.append(row)
    return pd.DataFrame(rows).dropna(axis=1, how="all")


def table_pool(runs, target):
    """Per ogni variante: reward media della sonda (tutti i candidati) vs dei scelti."""
    rows = []
    for pol in sorted({r["meta"]["policy"] for r in runs}):
        S = pd.concat([_steps(r["rounds"]) for r in runs if r["meta"]["policy"] == pol])
        m = next(r["meta"] for r in runs if r["meta"]["policy"] == pol)
        rows.append({"politica": pol, "pool": m["pool"], "candidati nel pool (media)": _col(S, "pool_size").mean(),
                     "candidati totali (media)": _col(S, "n_cands_total").mean(),
                     f"r_{target} media scelti": _col(S, f"r_{target}_mean").mean(),
                     f"r_{target} media sonda": _col(S, f"probe_mean_{target}").mean(),
                     "percentile vs sonda": _col(S, f"pct_{target}").mean()})
    return pd.DataFrame(rows)


def table_joint(runs, target):
    rows = []
    for pol in sorted({r["meta"]["policy"] for r in runs}):
        S = pd.concat([_steps(r["rounds"]) for r in runs if r["meta"]["policy"] == pol])
        if f"joint_{target}" not in S:
            continue
        j, s = S[f"joint_{target}"].to_numpy(float), S[f"sumind_{target}"].to_numpy(float)
        ok = np.isfinite(j) & np.isfinite(s) & (np.abs(s) > 0)
        if ok.any():
            rows.append({"politica": pol, "round": int(ok.sum()), "Σ congiunto": j[ok].sum(),
                         "Σ somma singoli": s[ok].sum(), "rapporto medio": np.median(j[ok] / s[ok])})
    return pd.DataFrame(rows)


def table_summary(runs):
    """UNA riga per (target, politica, pool): media sui seed. Salvata anche in summary.tsv."""
    from scipy.stats import wilcoxon
    rows = []
    keys = sorted({(r["meta"]["target"], r["meta"]["policy"]) for r in runs},
                  key=lambda k: (GROUPS.index(k[0]), k[1]))
    for t, lab in keys:
        rs = [r for r in runs if r["meta"]["target"] == t and r["meta"]["policy"] == lab]
        m = rs[0]["meta"]
        per = []
        for r in rs:
            S, F = _steps(r["rounds"]), r["rounds"].iloc[-1]
            if not len(S):
                continue
            pct = _col(S, f"pct_{t}").dropna().to_numpy(float)
            k = max(1, len(S) // 3)
            d = {"Σ r target": _col(S, f"cum_{t}").iloc[-1],
                 "percentile vs sonda": pct.mean() if len(pct) else np.nan,
                 "p (pct>0.5)": wilcoxon(pct - 0.5, alternative="greater").pvalue
                 if len(pct) >= 6 and np.any(pct != 0.5) else np.nan,
                 "round": len(S), "archi rimossi": _col(S, "edges_after").iloc[-1],
                 "round conv. MAB": convergence_round(_col(S, "theta_delta")) if m["base"] in MABS else np.nan,
                 "sp MAB sonda (fine)": _col(S, f"sp_probe_{t}").iloc[-k:].mean(),
                 "sp regr. holdout inizio": _col(r["rounds"], "reg_hold_all_sp").iloc[0],
                 "sp regr. holdout fine": _col(S, "reg_hold_all_sp").iloc[-1],
                 "R² regr. holdout fine": _col(S, "reg_hold_all_r2").iloc[-1],
                 "sp regr. congelato": _col(S, "reg_hold_all_sp_frozen").iloc[-1]}
            for met in ("entropy", "gini", "coverage", "novelty", "aplt"):
                d[f"Δ{met} (target)"] = F.get(f"{t}_d_{met}", np.nan)
            for g in GROUPS:                     # accuratezza su tutti i segmenti: effetti collaterali
                d[f"Δndcg {g}"] = F.get(f"{g}_d_ndcg", np.nan)
            # stesse metriche solo sugli utenti affetti dai difetti rimossi (insieme cumulato a fine run)
            d["n affetti (target)"] = F.get(f"{t}_aff_n", np.nan)
            for met in ("entropy", "gini", "coverage", "novelty", "aplt"):
                d[f"Δ{met} (target, affetti)"] = F.get(f"{t}_aff_d_{met}", np.nan)
            for g in GROUPS:
                d[f"Δndcg {g} (affetti)"] = F.get(f"{g}_aff_d_ndcg", np.nan)
            per.append(d)
        if per:
            P = pd.DataFrame(per)
            row = {"target": t, "politica": m["base"], "pool": m["pool"], "seed": len(P)}
            row.update(P.mean(numeric_only=True).to_dict())
            rows.append(row)
    return pd.DataFrame(rows).dropna(axis=1, how="all")


# ---------------------------------------------------------------------------- report
GUIDE = """## Come leggere

- **Reward** r_g = calo di rr_best (visibilità degli item del difetto) sommato sugli utenti del segmento g
  che li avevano in top-K, diviso per |g|, misurato difetto per difetto sul grafo del round
  (= −glob_d_prop_rr_best del LOO), scalato per la mediana di |r| del round 0.
- **Percentile vs sonda**: a ogni round si misurano anche difetti candidati estratti a caso (sonda).
  Il percentile della reward dei difetti scelti dentro la distribuzione della sonda vale 0.5 per una
  politica casuale, 1 per un oracolo. È confrontabile tra politiche anche se i grafi divergono,
  perché ogni round è confrontato con i candidati *dello stesso grafo*. Il p-value (Wilcoxon, una coda)
  testa percentile > 0.5 sui round.
- **Matrice di specializzazione**: righe = segmento ottimizzato, colonne = reward ottenuta su ogni
  segmento. Se il MAB è davvero segment-aware la diagonale domina la propria colonna.
- **Tempi**: ogni riga (e ogni valore finale in tabella) si riferisce allo stato **dopo** la rimozione
  dei difetti di quel round, ripropagato con LightGCN; la riga round = -1 è il grafo iniziale.
- **Stato per segmento** (sulle top-K di *tutti* gli utenti del segmento):
  entropy/gini/coverage sulla distribuzione degli item raccomandati; novelty = −log2(pop/n_utenti);
  APLT = quota di slot a item long-tail (fuori dal top {head}% per popolarità iniziale);
  NDCG/recall@K sul test split; churn = 1 − Jaccard con le liste iniziali;
  defect_share = quota di slot occupati da item dei difetti candidati.
- **Convergenza**: Δθ = ‖θ_t − θ_(t−1)‖/‖θ_(t−1)‖; round di convergenza = primo round con Δθ < 1% per 3 round
  consecutivi. Spearman sulla sonda = qualità predittiva *non distorta* del modello prima dell'aggiornamento;
  holdout = stessi contesti ogni round (grafo iniziale), quindi misura solo l'apprendimento di θ.
- **mab_cold**: stesso LinUCB ma senza prior offline; nei primi round sceglie difetti a caso e impara
  solo da quelli. Il confronto con `mab` misura quanto vale il prior. "Round per raggiungere il
  percentile di mab" = primo round da cui la media mobile (3) del percentile resta sopra il
  percentile medio del MAB con prior.
- **Holdout** (MAB e regressore): valutato a FINE round, dopo l'aggiornamento; il valore "inizio" è
  quello prima di qualsiasi aggiornamento. **Sonda e scelti**: valutati prima dell'aggiornamento del
  round (sono i dati con cui poi il modello viene aggiornato), quindi misurano la capacità predittiva
  del modello "come arriva" a quel round.
- **Pool** (`(pool regressor)`, `(pool random)`): il MAB sceglie solo tra N candidati, i primi N per
  y_hat del regressore (su all) oppure N casuali. La sonda è estratta da tutti i candidati, quindi il
  percentile dice quanto sono buoni i difetti scelti rispetto a tutto il dataset. Linea tratteggiata
  nei grafici = pool dal regressore, puntinata = pool casuale.
- **Regressore**: valutazione prequenziale (il modello del round t−1 predice il batch del round t prima
  di vederlo). "Congelato" = modello di partenza: la differenza è quanto il regressore ha imparato
  dal dataset nuovo. R² calcolato sulla label normalizzata (rank) come in impact_regressor.py.
- **Additività**: effetto congiunto del batch rimosso vs somma degli effetti singoli. Rapporto ≈ 1:
  la reward difetto-per-difetto (semi-bandit) è una buona approssimazione; ≪ 1: i difetti del batch
  si sovrappongono (stessi utenti/item) e la reward singola sovrastima il guadagno.
"""


def build_report(root):
    root = Path(root)
    runs = load_runs(root)
    if not runs:
        print(f"[report] nessun run in {root}")
        return
    cfg = json.loads((root / "config.json").read_text()) if (root / "config.json").exists() else {}
    ds = runs[0]["meta"].get("dataset", root.name)
    L = [f"# MAB segment-aware — {ds}\n",
         f"- Run: {len(runs)} ({', '.join(sorted({r['dir'].name for r in runs}))})",
         f"- Budget: {cfg.get('budget_per_round')} archi/round, totale {cfg.get('total_budget')}, "
         f"max {cfg.get('max_per_round') or '∞'} difetti/round · sonda {cfg.get('probe')} · "
         f"holdout {cfg.get('holdout')} · prior offline: "
         + (f"{cfg.get('warm_neff')} esempi equivalenti" if cfg.get("warm_neff") is not None
            else f"peso {cfg.get('warm_weight')}") + "\n",
         GUIDE.format(head=int(100 * (cfg.get("head_frac") or 0.2)))]
    T = table_summary(runs)
    T.to_csv(root / "summary.tsv", sep="\t", index=False)
    L.append("## Tabella riassuntiva (una riga per target × politica × pool, media sui seed; anche in summary.tsv)\n")
    L.append("Colonne Δ = variazione finale rispetto al grafo iniziale; `(target)` = sul segmento target della riga, tutti i suoi utenti. `(affetti)` = solo gli utenti del segmento che avevano nella top-K almeno un item di un difetto rimosso durante il run (grafo finale vs iniziale, stessi utenti); `n affetti` = quanti sono. `sp regr.` = Spearman del regressore "
             "(label all) sull'holdout; `congelato` = regressore mai aggiornato. `round conv. MAB` = primo round "
             "con Δθ < 1% per 3 round.\n")
    L.append(md(T))
    L.append("\n## 0. Stato iniziale per segmento\n")
    L.append(md(table_state_initial(runs), 4))
    if (root / "offline_check.tsv").exists():
        L.append("\n## 0b. Controllo offline (leave-one-dataset-out, Spearman)\n")
        L.append(md(pd.read_csv(root / "offline_check.tsv", sep="\t")))

    targets = [t for t in GROUPS if any(r["meta"]["target"] == t for r in runs)]
    for t in targets:
        rs = [r for r in runs if r["meta"]["target"] == t]
        L.append(f"\n---\n# Target: {t}\n")
        L.append("## 1. Reward sul target\n")
        L.append(md(table_reward(rs, t)))
        fig_reward(rs, t, root / f"fig_{t}_reward.png")
        L.append(f"\n![reward](fig_{t}_reward.png)\n")
        if any(r["meta"]["pool"] != "none" for r in rs):
            L.append("## 1b. Pool di candidati (sonda estratta da TUTTI i candidati)\n")
            L.append(md(table_pool(rs, t)) + "\n")
        L.append("## 4. Effetto sullo stato per segmento (Δ finale rispetto al grafo iniziale, ★ = target)\n")
        L.append(md(table_state(rs, t), 4))
        if fig_state(rs, t, root / f"fig_{t}_state.png"):
            L.append(f"\n![stato](fig_{t}_state.png)\n")
        C = table_convergence(rs, t)
        if len(C):
            L.append("## 5. Convergenza del MAB\n")
            L.append(md(C))
            fig_convergence(rs, t, root / f"fig_{t}_convergence.png")
            L.append(f"\n![convergenza](fig_{t}_convergence.png)\n")
        G = table_regressor(rs, t)
        if len(G):
            L.append("## 6. Regressore online\n")
            L.append(md(G))
            fig_regressor(rs, t, root / f"fig_{t}_regressor.png")
            L.append(f"\n![regressore](fig_{t}_regressor.png)\n")
        J = table_joint(rs, t)
        if len(J):
            L.append("## 7. Additività (effetto congiunto vs somma dei singoli)\n")
            L.append(md(J))

    L.append("\n---\n# Confronto tra target\n")
    for pol in sorted({r["meta"]["policy"] for r in runs}):
        S = table_specialization(runs, pol)
        if len(S) > 1:
            L.append(f"## 2. Matrice di specializzazione ({pol})\n")
            L.append(md(S))
    for lab in sorted({r["meta"]["policy"] for r in runs if r["meta"]["base"] in MABS}):
        O = table_removed_overlap(runs, lab)
        if len(O):
            L.append(f"\n## 3. Difetti rimossi in comune tra target diversi ({lab})\n")
            L.append(md(O))
    if fig_segments(runs, root / "fig_segment_lists.png"):
        L.append("\nJaccard tra le liste raccomandate ai tre segmenti nello stesso round:\n")
        L.append("![liste](fig_segment_lists.png)\n")
    (root / "REPORT.md").write_text("\n".join(L), encoding="utf-8")
    print(f"[report] {root / 'REPORT.md'}")


if __name__ == "__main__":
    build_report(sys.argv[1] if len(sys.argv) > 1 else ".")
