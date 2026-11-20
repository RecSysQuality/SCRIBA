# Valutazione per il paper: figure e tabelle LaTeX.

from __future__ import annotations

import csv

import numpy as np

import eval_core as core
from eval_core import (
    GRID, INK, INK2, METHOD_LABEL, POOL_TXT, SCOPE_LABEL, SCOPE_ORDER, SEGMENTS, SEG_COLOR, SEG_LABEL,
    USET_SUFFIX, USET_TXT, is_baseline, log, method_key, method_label, parse_sel, per_user, scope_key, style
)


SOURCE_ORDER = ("group_f", "gsdb_f", "rev2")


def fig_sources(rows, outdir, vlabel="", name="fig_sources"):
    """Istogramma: per ogni dataset (un pannello), quanti difetti di GSRank, GSDB e Rev2 sceglie ogni metodo
    nello scope All. Numero sopra le barre = difetti scelti; tra parentesi = quota degli archi rimossi."""
    if not rows:
        return
    plt = _plt()
    dss = list(dict.fromkeys(r["dataset"] for r in rows))
    for budget in sorted({r["budget"] for r in rows}):
        fig, axes = plt.subplots(1, len(dss), figsize=(3.6 * len(dss), 3.3), squeeze=False)
        for ax, ds in zip(axes[0], dss):
            R = [r for r in rows if (r["dataset"], r["budget"]) == (ds, budget)]
            if not R:
                ax.set_visible(False)
                continue
            srcs = sorted({r["source"] for r in R}, key=lambda x: (SOURCE_ORDER.index(x) if x in SOURCE_ORDER else 9, x))
            meths = sorted({r["method"] for r in R}, key=method_key)
            w = 0.8 / len(meths)
            x = np.arange(len(srcs))
            top = max(r["n_defects"] for r in R)
            for j, me in enumerate(meths):
                col, _ = style(me)
                v = [next((r["n_defects"] for r in R if r["method"] == me and r["source"] == sc), 0) for sc in srcs]
                e = [next((r["share_edges"] for r in R if r["method"] == me and r["source"] == sc), 0) for sc in srcs]
                xx = x + (j - (len(meths) - 1) / 2) * w
                ax.bar(xx, v, w * 0.92, color=col, label=method_label(me, "all") +
                       (" (baseline)" if is_baseline(me) else ""), hatch=None if me == "mab_all" else "////",
                       edgecolor="white", lw=0.8)
                for xi, vi, ei in zip(xx, v, e):
                    ax.text(xi, vi + top * 0.02, f"{vi}\n({100 * ei:.0f}%)", ha="center", va="bottom",
                            fontsize=6.5, color=INK)
            ax.set_xticks(x, [SCOPE_LABEL.get(sc, sc) for sc in srcs])
            ax.set_ylim(0, top * 1.28)
            ax.set_title(ds, fontsize=10, color=INK)
            ax.grid(axis="x", visible=False)
        axes[0, 0].set_ylabel("defects selected in scope All")
        _legend(fig, axes, 4)
        fig.suptitle(f"Which source the defects selected in scope All come from — budget {budget}"
                     + (f", {vlabel}" if vlabel else "") + "\nnumber of defects; in parentheses, share of the "
                     "removed edges", fontsize=8, color=INK2, y=1.04)
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        _save(fig, outdir / f"{name}_{budget}")


# ── figure ───────────────────────────────────────────────────────────────────────────────────────
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                         "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                         "lines.linewidth": 2, "pdf.fonttype": 42})
    return plt


def _save(fig, path):
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)


def _scopes(rows, budget):
    return sorted({r["scope"] for r in rows if r["budget"] == budget}, key=scope_key)


def _legend(fig, axes, n):
    h, l = [], []
    for ax in np.ravel(axes):
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh); l.append(ll)
    fig.legend(h, l, loc="lower center", ncol=min(len(l), n), bbox_to_anchor=(0.5, -0.02))


def _tradeoff_panel(ax, R, k, box_below=38):
    """Un pannello del piano del trade-off: Base nell'origine, una freccia e un punto (IC 95%) per metodo.
    Il riquadro dei t-test SCRIBA vs metodo sta FUORI dal pannello, box_below punti sotto l'asse x
    (abbastanza da stare sotto i numeri e l'etichetta dell'asse): non copre mai frecce e punti."""
    mg = R[0]["tost_margin"]
    ax.axvspan(-mg, mg, color="#eeeeec", zorder=0, lw=0)
    xs = [r["d_ndcg"] for r in R] + [r["d_ndcg_lo"] for r in R] + [-mg, mg]
    ys = [r["tndcg_removed_hi"] for r in R] + [0]
    xmin, xmax = min(min(xs), 0) * 1.25, max(max(xs), mg) * 1.25 + 1e-12
    ymax = max(ys) * 1.15 + 1e-12
    ax.axvline(0, color=INK2, lw=0.8)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.plot(0, 0, marker="*", ms=10, color=INK, ls="none", zorder=4, label="Base")
    for r in R:
        col, mk = style(r["method"])
        ax.annotate("", xy=(r["d_ndcg"], r["tndcg_removed"]), xytext=(0, 0), zorder=3,
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=1.4, shrinkA=6, shrinkB=7))
        ax.errorbar(r["d_ndcg"], r["tndcg_removed"],
                    xerr=[[r["d_ndcg"] - r["d_ndcg_lo"]], [r["d_ndcg_hi"] - r["d_ndcg"]]],
                    yerr=[[r["tndcg_removed"] - r["tndcg_removed_lo"]],
                          [r["tndcg_removed_hi"] - r["tndcg_removed"]]],
                    fmt=mk, color=col, ms=8 if r["method"] == "mab_all" else 6.5, mec="white", mew=1.2,
                    elinewidth=1, capsize=0, zorder=5, label=method_label(r["method"], generic=True))
        right = r["method"] == "mab_all"
        ax.annotate(method_label(r["method"], r["scope"]), (r["d_ndcg"], r["tndcg_removed"]),
                    xytext=(8, 4) if right else (-8, 4), textcoords="offset points",
                    ha="left" if right else "right", va="bottom", fontsize=7.5, color=col, weight="bold", zorder=7)
        if r.get("ndcg_equivalent"):                       # TOST: NDCG equivalente al Base
            ax.annotate("≈", (r["d_ndcg"], r["tndcg_removed"]), xytext=(6, 4), textcoords="offset points",
                        fontsize=10, color=col, weight="bold")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(min(0, min(r["tndcg_removed_lo"] for r in R)) * 1.2 - 1e-12, ymax)
    ax.ticklabel_format(axis="both", style="sci", scilimits=(-2, 3))
    # riquadro dei t-test appaiati SCRIBA vs ogni altro metodo, scritto per esteso (verso + p-value)
    ref = next((r for r in R if r["method"] == "mab_all"), None)
    others = [r for r in R if "p_rem_vs_mab" in r]
    if ref is not None and others:
        name = METHOD_LABEL["mab_all"]
        lines = []
        for r in others:
            other = method_label(r["method"], r["scope"])
            if len(others) > 1:
                lines.append(f"vs {other}:")
            p = r["p_rem_vs_mab"]
            if _sig_p(p):
                more = ref["tndcg_removed"] > r["tndcg_removed"]
                lines.append(f"Exposure removed: {name} {'higher' if more else 'lower'} ({_fmt_p(p)})")
            else:
                lines.append(f"Exposure removed: no significant difference ({_fmt_p(p)})")
            p = r.get("p_ndcg_vs_mab")
            if _sig_p(p):
                less = ref["d_ndcg"] > r["d_ndcg"]
                lines.append(f"Δ NDCG: {name} loses {'less' if less else 'more'} ({_fmt_p(p)})")
            else:
                lines.append(f"Δ NDCG: no significant difference ({_fmt_p(p)})")
        head = (f"{name} vs {method_label(others[0]['method'], others[0]['scope'])}" if len(others) == 1
                else f"{name} vs each baseline") + " — paired t-test, same users" + (", Holm" if len(others) > 1 else "")
        ax.annotate(head + "\n" + "\n".join(lines), xy=(0, 0), xycoords="axes fraction",
                    xytext=(0, -box_below), textcoords="offset points", ha="left", va="top", fontsize=6.5,
                    color=INK, linespacing=1.35, bbox=dict(fc="white", ec=GRID, lw=0.6, pad=3),
                    annotation_clip=False, zorder=6)


def _sig_p(p, alpha=0.05):
    return p is not None and np.isfinite(p) and p < alpha


def _fmt_p(p):
    if p is None or not np.isfinite(p):
        return "p not available"
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}" if p < 0.01 else f"p = {p:.2f}"


def _stars(p):
    if p is None or not np.isfinite(p):
        return "--"
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


TRADEOFF_NOTE = ("each arrow goes from Base (★, before removal) to a method after removal: higher = more exposure to "
                 "defective items removed, further right = less NDCG lost\nbars: 95% bootstrap CI;  grey band: NDCG "
                 "change within the TOST equivalence margin;  ≈ next to a point: NDCG equivalent to Base (TOST)")


def fig_tradeoff(T, outdir, k, vlabel, uset="aff"):
    """Un'immagine per dataset: una colonna per scope. uset: aff (affetti) o seg (tutti gli utenti)."""
    plt = _plt()
    for budget in sorted({r["budget"] for r in T}):
        scopes = _scopes(T, budget)
        fig, axes = plt.subplots(1, len(scopes), figsize=(3.3 * len(scopes), 3.6), squeeze=False)
        for ax, sc in zip(axes[0], scopes):
            R = [r for r in T if (r["budget"], r["scope"], r["segment"]) == (budget, sc, "all")]
            if not R:
                ax.set_visible(False)
                continue
            _tradeoff_panel(ax, R, k, box_below=38)
            ax.set_title(f"{SCOPE_LABEL.get(sc, sc)} (n={R[0]['n_users']})", fontsize=9, color=INK)
            ax.set_xlabel(f"Δ NDCG@{k}  (after − before)")
        axes[0, 0].set_ylabel(f"T-NDCG@{k} removed  (before − after)")
        _legend(fig, axes, 6)
        fig.suptitle(f"Accuracy cost vs. defective exposure removed — {USET_TXT[uset]}, budget {budget}, {vlabel}\n"
                     + TRADEOFF_NOTE, fontsize=8, color=INK2, y=1.04)
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        _save(fig, outdir / f"fig_tradeoff{USET_SUFFIX[uset]}_{budget}_k{k}")


def _kl_panel(ax, R, k):
    """x = KL della distribuzione di NDCG (dopo || prima), y = KL della distribuzione di T-NDCG.
    Base = (0, 0). Banda grigia verticale: KL di NDCG sotto la soglia di rumore (distribuzione invariata);
    banda orizzontale: lo stesso per T-NDCG."""
    xn, yn = R[0]["kl_ndcg_noise"], R[0]["kl_tndcg_noise"]
    xm = max([r["kl_ndcg"] for r in R] + [xn]) * 1.25 + 1e-12
    ym = max([r["kl_tndcg"] for r in R] + [yn]) * 1.15 + 1e-12
    ax.axvspan(0, xn, color="#eeeeec", lw=0, zorder=0)
    ax.axhspan(0, yn, color="#eeeeec", lw=0, zorder=0)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.plot(0, 0, marker="*", ms=10, color=INK, ls="none", zorder=4, label="Base")
    for r in R:
        col, mk = style(r["method"])
        ax.annotate("", xy=(r["kl_ndcg"], r["kl_tndcg"]), xytext=(0, 0), zorder=3,
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=1.4, shrinkA=6, shrinkB=7))
        ax.plot(r["kl_ndcg"], r["kl_tndcg"], marker=mk, color=col, ms=8 if r["method"] == "mab_all" else 6.5,
                mec="white", mew=1.2, ls="none", zorder=5, label=method_label(r["method"], generic=True))
        right = r["method"] == "mab_all"
        ax.annotate(method_label(r["method"], r["scope"]), (r["kl_ndcg"], r["kl_tndcg"]),
                    xytext=(8, 4) if right else (-8, 4), textcoords="offset points",
                    ha="left" if right else "right", va="bottom", fontsize=7.5, color=col, weight="bold", zorder=7)
    ax.set_xlim(-xm * 0.03, xm)
    ax.set_ylim(-ym * 0.03, ym)


KL_NOTE = ("KL(after ‖ before) between the per-user distributions of NDCG (x) and of T-NDCG (y), affected users; "
           "shaded: below the sampling-noise level (no change)\nhigh y and low x = the distribution of defective "
           "exposure changes, the distribution of accuracy does not")


def fig_kl(T, outdir, k, vlabel, grid=False):
    plt = _plt()
    dss = list(dict.fromkeys(r["dataset"] for r in T)) if grid else [None]
    for budget in sorted({r["budget"] for r in T}):
        scopes = _scopes(T, budget)
        fig, axes = plt.subplots(len(dss), len(scopes), figsize=(3.0 * len(scopes), 2.7 * len(dss) + 0.7),
                                 squeeze=False)
        for i, ds in enumerate(dss):
            for j, sc in enumerate(scopes):
                ax = axes[i, j]
                R = [r for r in T if (r["budget"], r["scope"], r["segment"]) == (budget, sc, "all")
                     and (ds is None or r["dataset"] == ds)]
                if not R:
                    ax.set_visible(False)
                    continue
                _kl_panel(ax, R, k)
                if i == 0:
                    ax.set_title(SCOPE_LABEL.get(sc, sc), fontsize=10, color=INK)
                if i == len(dss) - 1:
                    ax.set_xlabel(f"KL  NDCG@{k}")
                if j == 0:
                    ax.set_ylabel((f"{ds}\n" if ds else "") + f"KL  T-NDCG@{k}", fontsize=9)
        _legend(fig, axes, 6)
        fig.suptitle(f"How much the per-user distributions change — budget {budget}, {vlabel}\n" + KL_NOTE,
                     fontsize=8, color=INK2, y=1.03)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        _save(fig, outdir / f"fig_kl{'_grid' if grid else ''}_{budget}_k{k}")


def latex_shares(T, k, vlabel):
    """Tabella per il paper: per dataset e scope, sorgente (baseline dello scope) e SCRIBA affiancati:
    quota di utenti affetti a cui SCENDE l'esposizione difettosa (T-NDCG@k) e a cui SCENDE l'NDCG@k."""
    def who(r):
        if r["method"] == "mab_all":
            return "scriba"
        base = "stratified" if r["scope"] == "all" else "heuristic"
        return "source" if r["method"] == base else None

    out = [f"% generated by paper_figures.py — @{k}, {vlabel}; requires booktabs, multirow"]
    for budget in sorted({r["budget"] for r in T}):
        idx = {}
        for r in T:
            if r["budget"] == budget and r["segment"] == "all" and who(r):
                idx[(r["dataset"], r["scope"], who(r))] = r
        if not idx:
            continue
        pct = lambda r, c: "--" if r is None else f"{100 * r[c]:.1f}"
        out += ["", "\\begin{table}[t]\\centering\\small",
                f"\\caption{{Share of affected users (\\%) whose exposure to defective items decreases "
                f"(T-NDCG@{k}) and whose accuracy decreases (NDCG@{k}) after removal, budget {budget} ({vlabel}), "
                "for the defects ranked by the source itself (Source) and by SCRIBA. Under retraining part of "
                "the NDCG changes is training noise, which affects both methods alike: the comparison between "
                "the two columns is what matters.}",
                "\\begin{tabular}{llcccc}", "\\toprule",
                " & & \\multicolumn{2}{c}{Exposure $\\downarrow$ (\\%)} & "
                "\\multicolumn{2}{c}{NDCG $\\downarrow$ (\\%)} \\\\",
                "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}",
                "Dataset & Scope & Source & SCRIBA & Source & SCRIBA \\\\", "\\midrule"]
        dss = list(dict.fromkeys(key[0] for key in idx))
        for i, ds in enumerate(dss):
            scopes = sorted({key[1] for key in idx if key[0] == ds}, key=scope_key)
            ds_tex = ds.replace("_", "\\_")
            for j, sc in enumerate(scopes):
                src, scr = idx.get((ds, sc, "source")), idx.get((ds, sc, "scriba"))
                cells = [pct(src, "share_users_exposure_down"), pct(scr, "share_users_exposure_down"),
                         pct(src, "share_users_ndcg_down"), pct(scr, "share_users_ndcg_down")]
                first = "\\multirow{%d}{*}{%s}" % (len(scopes), ds_tex) if j == 0 else ""
                out.append(f"{first} & {SCOPE_LABEL.get(sc, sc)} & " + " & ".join(cells) + " \\\\")
            if i < len(dss) - 1:
                out.append("\\midrule")
        out += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(out) + "\n"


def latex_kl_compact(T, k, vlabel):
    """Tabella per il paper: solo la KL della distribuzione per utente di NDCG@k (dopo || prima), una riga per
    (dataset, scope), una colonna per metodo (sorgente, SCRIBA) e il livello di rumore. Valori vicini al rumore =
    distribuzione dell'accuratezza invariata."""
    out = [f"% generated by paper_figures.py — @{k}, {vlabel}; requires booktabs, multirow"]
    for budget in sorted({r["budget"] for r in T}):
        R = [r for r in T if r["budget"] == budget and r["segment"] == "all"]
        if not R:
            continue
        has_base = any(is_baseline(r["method"]) for r in R)
        cols = (["__baseline__"] if has_base else []) + sorted(
            {r["method"] for r in R if not is_baseline(r["method"])}, key=method_key)
        head = [("Source" if c == "__baseline__" else method_label(c)) for c in cols]
        out += ["", "\\begin{table}[t]\\centering\\small",
                f"\\caption{{KL divergence (bits) between the per-user distribution of NDCG@{k} after and before "
                f"removal, affected users, budget {budget} ({vlabel}). Source: the baseline of each scope (the "
                "defects ranked by the source itself). Noise: KL between the distribution before removal and a "
                "bootstrap resample of it, i.e. the value expected when nothing changes. Values close to the noise "
                "level mean that removal leaves the distribution of accuracy across users unchanged.}",
                "\\begin{tabular}{ll" + "c" * (len(cols) + 1) + "}", "\\toprule",
                "Dataset & Scope & " + " & ".join(head) + " & Noise \\\\", "\\midrule"]
        dss = list(dict.fromkeys(r["dataset"] for r in R))
        for di, ds in enumerate(dss):
            scs = sorted({r["scope"] for r in R if r["dataset"] == ds}, key=scope_key)
            ds_tex = ds.replace("_", "\\_")
            for si, sc in enumerate(scs):
                rr = [r for r in R if (r["dataset"], r["scope"]) == (ds, sc)]
                cells = []
                for c in cols:
                    x = next((r for r in rr if (is_baseline(r["method"]) if c == "__baseline__"
                                                else r["method"] == c)), None)
                    cells.append("--" if x is None else f"{x['kl_ndcg']:.5f}")
                first = ("\\multirow{%d}{*}{%s}" % (len(scs), ds_tex)) if si == 0 else ""
                out.append(f"{first} & {SCOPE_LABEL.get(sc, sc)} & " + " & ".join(cells)
                           + f" & {rr[0]['kl_ndcg_noise']:.5f} \\\\")
            if di < len(dss) - 1:
                out.append("\\midrule")
        out += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(out) + "\n"


def latex_kl(T, k, vlabel):
    out = [f"% generated by paper_figures.py — @{k}, {vlabel}; requires booktabs"]
    for budget in sorted({r["budget"] for r in T}):
        R = sorted([r for r in T if r["budget"] == budget and r["segment"] == "all"],
                   key=lambda r: (r["dataset"], scope_key(r["scope"]), method_key(r["method"])))
        out += ["", "\\begin{table}[t]\\centering\\scriptsize",
                f"\\caption{{Change in the per-user distributions of accuracy and of defective exposure, affected "
                f"users, budget {budget}, @{k} ({vlabel}): KL divergence (bits) between the distribution after and "
                "before removal; in parentheses the sampling-noise level (KL between the distribution before removal "
                "and a bootstrap resample of it): values at or below it mean no change. A high value for T-NDCG and a low value for NDCG mean that "
                "removal reshapes exposure to defective items while leaving the accuracy distribution unchanged.}",
                "\\begin{tabular}{lllcc}", "\\toprule",
                "Dataset & Scope & Method & KL NDCG & KL T-NDCG \\\\", "\\midrule"]
        for r in R:
            out.append(f"{r['dataset']} & {SCOPE_LABEL.get(r['scope'], r['scope'])} & {method_label(r['method'], r['scope'])} & "
                       f"{r['kl_ndcg']:.5f} ({r['kl_ndcg_noise']:.5f}) & "
                       f"{r['kl_tndcg']:.4f} ({r['kl_tndcg_noise']:.4f}) \\\\")
        out += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(out) + "\n"


SCOPE_STYLE = {"all": ("#52514e", "o"), "group_f": ("#2a78d6", "s"), "gsdb_f": ("#1baf7a", "D"),
               "rev2": ("#eb6834", "^")}


def fig_tradeoff_pairs(T, outdir, k, vlabel, uset="aff"):
    """Versione COMPATTA (solo con --one-baseline): un pannello per dataset. Per ogni scope due punti dello
    stesso colore: vuoto = baseline della sorgente, pieno = SCRIBA; freccia baseline -> SCRIBA.
    Freccia verso l'alto = SCRIBA toglie più esposizione difettosa; verso destra = perde meno NDCG."""
    plt = _plt()
    from matplotlib.lines import Line2D
    dss = list(dict.fromkeys(r["dataset"] for r in T))
    for budget in sorted({r["budget"] for r in T}):
        fig, axes = plt.subplots(1, len(dss), figsize=(3.6 * len(dss), 3.5), squeeze=False)
        for ax, ds in zip(axes[0], dss):
            R = [r for r in T if (r["dataset"], r["budget"], r["segment"]) == (ds, budget, "all")]
            ax.axvline(0, color=INK2, lw=0.8)
            ax.axhline(0, color=INK2, lw=0.8)
            ax.plot(0, 0, marker="*", ms=10, color=INK, ls="none", zorder=4)
            for sc in sorted({r["scope"] for r in R}, key=scope_key):
                col, mk = SCOPE_STYLE.get(sc, ("#7a7974", "h"))
                b = next((r for r in R if r["scope"] == sc and is_baseline(r["method"])), None)
                m = next((r for r in R if r["scope"] == sc and r["method"] == "mab_all"), None)
                for r, filled in ((b, False), (m, True)):
                    if r is None:
                        continue
                    ax.errorbar(r["d_ndcg"], r["tndcg_removed"],
                                xerr=[[r["d_ndcg"] - r["d_ndcg_lo"]], [r["d_ndcg_hi"] - r["d_ndcg"]]],
                                yerr=[[r["tndcg_removed"] - r["tndcg_removed_lo"]],
                                      [r["tndcg_removed_hi"] - r["tndcg_removed"]]],
                                fmt=mk, color=col, mfc=col if filled else "white", mec=col, mew=1.5, ms=8,
                                elinewidth=0.8, capsize=0, zorder=5)
                if b is not None and m is not None:
                    ax.annotate("", xy=(m["d_ndcg"], m["tndcg_removed"]), xytext=(b["d_ndcg"], b["tndcg_removed"]),
                                arrowprops=dict(arrowstyle="-|>", color=col, lw=1.4, shrinkA=6, shrinkB=6), zorder=3)
                    pr = b.get("p_rem_vs_mab")
                    ax.annotate(f"{SCOPE_LABEL.get(sc, sc)}" + (f" ({_fmt_p(pr)})" if _sig_p(pr) else ""),
                                (m["d_ndcg"], m["tndcg_removed"]), xytext=(8, 4), textcoords="offset points",
                                ha="left", va="bottom", fontsize=7.5, color=col, weight="bold")
            ax.set_title(ds, fontsize=10, color=INK)
            ax.set_xlabel(f"Δ NDCG@{k}  (after − before)")
            ax.ticklabel_format(axis="both", style="sci", scilimits=(-2, 3))
        axes[0, 0].set_ylabel(f"T-NDCG@{k} removed  (before − after)")
        hs = [Line2D([], [], marker="*", color=INK, ls="none", ms=10, label="Base"),
              Line2D([], [], marker="o", color=INK2, mfc="white", ls="none", ms=7, mew=1.5,
                     label="source baseline (hollow)"),
              Line2D([], [], marker="o", color=INK2, ls="none", ms=7, label="SCRIBA (filled)")]
        hs += [Line2D([], [], marker=SCOPE_STYLE[s_][1], color=SCOPE_STYLE[s_][0], ls="none", ms=7,
                      label=SCOPE_LABEL[s_]) for s_ in SCOPE_ORDER if any(r["scope"] == s_ for r in T)]
        fig.legend(handles=hs, loc="lower center", ncol=len(hs), bbox_to_anchor=(0.5, -0.03))
        fig.suptitle(f"Source baseline → SCRIBA, per scope — {USET_TXT[uset]}, budget {budget}, {vlabel}\n"
                     "arrow up = SCRIBA removes more defective exposure; arrow right = it loses less NDCG; "
                     "in parentheses: p-value of the paired t-test on exposure removed, SCRIBA vs source (shown if < 0.05); bars: 95% bootstrap CI",
                     fontsize=8, color=INK2, y=1.04)
        fig.tight_layout(rect=(0, 0.07, 1, 1))
        _save(fig, outdir / f"fig_tradeoff_pairs{USET_SUFFIX[uset]}_{budget}_k{k}")


def fig_tradeoff_grid(T, outdir, k, vlabel, uset="aff"):
    """Una sola immagine per il paper: righe = dataset, colonne = scope. Assi indipendenti per pannello
    (le scale di Rev2 e degli altri scope sono diverse)."""
    plt = _plt()
    dss = list(dict.fromkeys(r["dataset"] for r in T))
    for budget in sorted({r["budget"] for r in T}):
        scopes = _scopes(T, budget)
        fig, axes = plt.subplots(len(dss), len(scopes), figsize=(3.0 * len(scopes), 3.1 * len(dss) + 0.6),
                                 squeeze=False)
        for i, ds in enumerate(dss):
            last = i == len(dss) - 1
            for j, sc in enumerate(scopes):
                ax = axes[i, j]
                R = [r for r in T if (r["dataset"], r["budget"], r["scope"], r["segment"]) == (ds, budget, sc, "all")]
                if not R:
                    ax.set_visible(False)
                    continue
                _tradeoff_panel(ax, R, k, box_below=38 if last else 22)   # senza etichetta x il riquadro sale
                ax.text(0.97, 0.97, f"n={R[0]['n_users']}", transform=ax.transAxes, ha="right", va="top",
                        fontsize=7, color=INK2)
                if i == 0:
                    ax.set_title(SCOPE_LABEL.get(sc, sc), fontsize=10, color=INK)
                if last:
                    ax.set_xlabel(f"Δ NDCG@{k}")
                if j == 0:
                    ax.set_ylabel(f"{ds}\nT-NDCG@{k} removed", fontsize=9)
        _legend(fig, axes, 6)
        fig.suptitle(f"Accuracy cost vs. defective exposure removed — {USET_TXT[uset]}, budget {budget}, {vlabel}\n"
                     + TRADEOFF_NOTE, fontsize=8, color=INK2, y=1.02)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        _save(fig, outdir / f"fig_tradeoff_grid{USET_SUFFIX[uset]}_{budget}_k{k}")


def fig_decomp(T, LC, outdir, k, vlabel):
    plt = _plt()
    for budget in sorted({r["budget"] for r in T}):
        scopes = _scopes(T, budget)
        fig, axes = plt.subplots(1, len(scopes) + 1, figsize=(2.9 * (len(scopes) + 1), 2.9), squeeze=False,
                                 gridspec_kw=dict(width_ratios=[1] * len(scopes) + [0.9]))
        comps = (("d_ndcg_flagged", "flagged items", "#e34948", None),
                 ("d_ndcg_rest", "other items", "#a3a29c", "////"))
        lim = 0
        for ax, sc in zip(axes[0], scopes):
            R = sorted([r for r in T if (r["budget"], r["scope"], r["segment"]) == (budget, sc, "all")],
                       key=lambda r: method_key(r["method"]))
            y = np.arange(len(R))[::-1]
            for yy, r in zip(y, R):
                neg = pos = 0.0
                for c, lab, col, hatch in comps:
                    v = r[c]
                    start = neg if v < 0 else pos
                    ax.barh(yy, v, left=start, height=0.6, color=col, hatch=hatch, edgecolor="white", lw=0.8,
                            label=lab)
                    if v < 0:
                        neg += v
                    else:
                        pos += v
                    lim = max(lim, abs(neg), pos)
                ax.plot(r["d_ndcg"], yy, marker="|", ms=12, mew=2, color=INK, ls="none",
                        label=f"total Δ NDCG@{k}")
            ax.set_yticks(y, [method_label(r["method"], r["scope"]) for r in R])
            ax.axvline(0, color=INK2, lw=0.8)
            ax.set_title(SCOPE_LABEL.get(sc, sc), fontsize=9, color=INK)
            ax.set_xlabel(f"Δ NDCG@{k}")
            ax.grid(axis="y", visible=False)
        for ax in axes[0, :len(scopes)]:
            ax.set_xlim(-1.15 * lim, 1.15 * lim)
            ax.ticklabel_format(axis="x", style="sci", scilimits=(-2, 3))
        ax = axes[0, -1]
        L = {r["scope"]: r for r in LC if r["budget"] == budget and r["segment"] == "all"}
        x = np.arange(len(scopes))
        ax.bar(x - 0.19, [L[s]["prec_flagged"] for s in scopes], 0.36, color="#e34948", label="flagged items")
        ax.bar(x + 0.19, [L[s]["prec_other"] for s in scopes], 0.36, color="#a3a29c", hatch="////",
               edgecolor="white", label="other items")
        ax.set_xticks(x, [SCOPE_LABEL.get(s, s) for s in scopes])
        ax.set_ylabel("share of slots that are test hits")
        ax.set_title("Base lists: slot precision", fontsize=9, color=INK)
        ax.grid(axis="x", visible=False)
        _legend(fig, axes, 5)
        fig.suptitle(f"Where the NDCG loss comes from (relevance lost on flagged items vs. rest of the list) and how relevant the "
                     f"flagged items are — affected users, budget {budget}, {vlabel}",
                     fontsize=8, color=INK2, y=1.02)
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        _save(fig, outdir / f"fig_decomp_{budget}_k{k}")


def _bin(nf, k):
    edges = [0, 1, 2, 3, 4, 5]
    labs = ["0", "1", "2", "3", "4", "≥5"]
    b = np.minimum(nf, 5).astype(int)
    return b, labs


def fig_users(U, outdir, k, vlabel):
    plt = _plt()
    for budget in sorted({r["budget"] for r in U}):
        scopes = _scopes(U, budget)
        fig, axes = plt.subplots(2, len(scopes), figsize=(3.2 * len(scopes), 4.8), squeeze=False, sharex="col")
        for j, sc in enumerate(scopes):
            R = sorted([r for r in U if (r["budget"], r["scope"], r["segment"]) == (budget, sc, "all")],
                       key=lambda r: method_key(r["method"]))
            if not R:
                continue
            b, labs = _bin(R[0]["nflag"], k)
            present = [i for i in range(6) if (b == i).sum() >= 20]
            for r in R:
                col, mk = style(r["method"])
                for i, (key, ax) in enumerate((("rem", axes[0, j]), ("dn", axes[1, j]))):
                    v = r[key]
                    mu = [v[b == q].mean() for q in present]
                    se = [1.96 * v[b == q].std(ddof=1) / np.sqrt((b == q).sum()) for q in present]
                    ax.errorbar(range(len(present)), mu, yerr=se, color=col, marker=mk, ms=6, mec="white",
                                lw=2 if r["method"] == "mab_all" else 1.3, capsize=0, elinewidth=0.8,
                                label=method_label(r["method"], generic=True))
            axes[0, j].set_title(SCOPE_LABEL.get(sc, sc), fontsize=9, color=INK)
            axes[1, j].axhline(0, color=INK2, lw=0.8)
            axes[1, j].set_xticks(range(len(present)),
                                  [f"{labs[q]}\n{100 * (b == q).mean():.0f}%" for q in present])
            axes[1, j].set_xlabel(f"defective items in the user's Base top-{k}\n(share of affected users)")
            for ax in axes[:, j]:
                ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3))
        axes[0, 0].set_ylabel(f"T-NDCG@{k} removed")
        axes[1, 0].set_ylabel(f"Δ NDCG@{k}")
        _legend(fig, axes, 6)
        fig.suptitle(f"Gain and cost by user exposure — mean ± 95% CI, budget {budget}, {vlabel}",
                     fontsize=8, color=INK2, y=1.01)
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        _save(fig, outdir / f"fig_users_{budget}_k{k}")


def fig_dist(U, outdir, k, vlabel):
    plt = _plt()
    for budget in sorted({r["budget"] for r in U}):
        scopes = _scopes(U, budget)
        fig, axes = plt.subplots(2, len(scopes), figsize=(3.2 * len(scopes), 4.6), squeeze=False)
        for j, sc in enumerate(scopes):
            R = sorted([r for r in U if (r["budget"], r["scope"], r["segment"]) == (budget, sc, "all")],
                       key=lambda r: method_key(r["method"]))
            lim = max((np.percentile(np.abs(r["dn"]), 99) for r in R), default=1e-3) or 1e-3
            for r in R:
                col, _ = style(r["method"])
                lw = 2 if r["method"] == "mab_all" else 1.3
                for ax, v in ((axes[0, j], r["dn"]), (axes[1, j], r["js"])):
                    x = np.sort(v)
                    ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", color=col, lw=lw,
                            label=method_label(r["method"], generic=True))
            axes[0, j].axvline(0, color=INK2, lw=0.8)
            axes[0, j].set_xlim(-lim, lim)
            axes[0, j].ticklabel_format(axis="x", style="sci", scilimits=(-2, 3))
            axes[0, j].set_xlabel(f"per-user Δ NDCG@{k}")
            axes[1, j].set_xlabel(f"per-user JS(list before ‖ list after) @{k}")
            axes[0, j].set_title(SCOPE_LABEL.get(sc, sc), fontsize=9, color=INK)
            axes[1, j].set_xlim(0, None)
        for i in range(2):
            axes[i, 0].set_ylabel("share of affected users (ECDF)")
        _legend(fig, axes, 6)
        fig.suptitle(f"Per-user distributions, not means (jump at 0 = users with unchanged NDCG) — budget {budget}, {vlabel}", fontsize=8, color=INK2,
                     y=1.01)
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        _save(fig, outdir / f"fig_dist_{budget}_k{k}")


def fig_fairness(F, outdir, k, vlabel):
    """Active vs Long-tail. Riga in alto: riduzione relativa di T-NDCG@k (beneficio); riga in basso: variazione
    relativa di NDCG@k (costo). Barre affiancate per segmento, IC bootstrap 95% sugli utenti affetti."""
    plt = _plt()
    for budget in sorted({r["budget"] for r in F}):
        scopes = _scopes(F, budget)
        fig, axes = plt.subplots(2, len(scopes), figsize=(3.2 * len(scopes), 4.8), squeeze=False)
        for j, sc in enumerate(scopes):
            R = sorted([r for r in F if (r["budget"], r["scope"]) == (budget, sc)],
                       key=lambda r: method_key(r["method"]))
            x = np.arange(len(R))
            for i, (pre, lab) in enumerate((("tndcg_red", f"T-NDCG@{k} reduction (%)"),
                                            ("rel_ndcg", f"Δ NDCG@{k} / NDCG@{k} before (%)"))):
                ax = axes[i, j]
                for o, g, hatch in ((-0.19, "active", None), (0.19, "lt", "////")):
                    v = np.array([r[f"{pre}_{g}"] for r in R]) * 100
                    lo = np.array([r[f"{pre}_{g}_lo"] for r in R]) * 100
                    hi = np.array([r[f"{pre}_{g}_hi"] for r in R]) * 100
                    ax.bar(x + o, v, 0.36, color=SEG_COLOR[g], hatch=hatch, edgecolor="white", lw=0.8,
                           label=SEG_LABEL[g])
                    ax.errorbar(x + o, v, yerr=[v - lo, hi - v], fmt="none", ecolor=INK, elinewidth=0.8)
                ax.axhline(0, color=INK2, lw=0.8)
                ax.set_xticks(x, [method_label(r["method"], r["scope"]) for r in R], rotation=25, ha="right")
                ax.grid(axis="x", visible=False)
                if j == 0:
                    ax.set_ylabel(lab)
            axes[0, j].set_title(SCOPE_LABEL.get(sc, sc), fontsize=9, color=INK)
        _legend(fig, axes, 2)
        fig.suptitle(f"Active vs. long-tail affected users — top: T-NDCG@{k} reduction (benefit), bottom: relative "
                     f"NDCG@{k} change (cost); 95% bootstrap CI, budget {budget}, {vlabel}",
                     fontsize=8, color=INK2, y=1.01)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        _save(fig, outdir / f"fig_fairness_{budget}_k{k}")


CURVE_DIR = "curves_lists"


def budget_curves(D, a, k, uset="aff"):
    """Curve dose-risposta, variante propagazione: per ogni file di rimozione e ogni prefisso salvato dal dump
    (0, emax/N, ... archi), T-NDCG@k e T-HR@k sul POOL dello scope (D.groups: detected + nohead, come nel resto
    dello script) e NDCG@k, media sugli utenti `uset` (aff = affetti dello scope, seg = segmento intero).
    Il riferimento è il punto 0 della curva (nessun arco tolto, stessa propagazione): rimossa e ΔNDCG sono
    differenze appaiate per utente rispetto a quel punto. IC 95% normale sulla media delle differenze."""
    cdir = D.lists_dir / "prop" / CURVE_DIR
    if not cdir.is_dir():
        log(f"  {cdir} non trovato: niente curve sul pool (rilancia il dump con --curves N)")
        return []
    keep = {(s.budget, s.scope, s.method) for s in D.sels}       # stessi metodi del resto (--one-baseline, ...)
    mask = (lambda gr, g: gr.aff[g]) if uset == "aff" else (lambda gr, g: D.seg[g])
    z95 = 1.959963984540054

    def mci(x):
        mu = float(x.mean())
        h = z95 * float(x.std(ddof=1)) / np.sqrt(len(x)) if len(x) > 1 else np.nan
        return mu, mu - h, mu + h

    rows = []
    for f in sorted(cdir.glob("*.npz")):
        if f.name.endswith(".tmp.npz"):
            continue
        z = np.load(f)
        meth, scope, budget, _ = parse_sel(str(z["sel"]))
        if (budget, scope, meth) not in keep:
            continue
        if "top" not in z.files:
            log(f"  {f.name}: curva senza liste (dump vecchio): salto")
            continue
        TOP, HIT = z["top"], z["hits"]
        if TOP.shape[2] < k:
            log(f"  {f.name}: liste salvate fino a {TOP.shape[2]} < k={k}: salto (rilancia il dump con --curve-k {k})")
            continue
        zu = z["users"]
        pos = np.full(int(max(zu.max(), D.users.max())) + 1, -1, np.int64)
        pos[zu] = np.arange(len(zu))
        cols = pos[D.users]
        if (cols < 0).any():
            log(f"  {f.name}: utenti diversi da quelli dell'analisi: salto")
            continue
        gr = D.groups[(budget, scope)]
        P = [per_user(TOP[j][cols], HIT[j][cols], D.base.npos, gr.t_union, k) for j in range(len(z["edges"]))]
        for g in SEGMENTS:
            m = mask(gr, g)
            n = int(m.sum())
            if n < 3:
                continue
            t0, n0 = P[0]["tndcg"][m], P[0]["ndcg"][m]
            for j, u in enumerate(P):
                rem, lo_r, hi_r = mci(t0 - u["tndcg"][m])
                dn, lo_n, hi_n = mci(u["ndcg"][m] - n0)
                rows.append(dict(dataset=D.ds, variant="prop", userset=uset, k=k, budget=budget, scope=scope,
                                 method=meth, segment=g, n_users=n, point=j, edges=int(z["edges"][j]),
                                 n_defects=int(z["n_defects"][j]),
                                 tndcg=float(u["tndcg"][m].mean()), thr=float((u["nflag"][m] > 0).mean()),
                                 ndcg=float(u["ndcg"][m].mean()),
                                 tndcg_removed=rem, tndcg_removed_lo=lo_r, tndcg_removed_hi=hi_r,
                                 rel_tndcg_reduction=rem / float(t0.mean()) if t0.mean() else np.nan,
                                 d_ndcg=dn, d_ndcg_lo=lo_n, d_ndcg_hi=hi_n))
        log(f"  curve @{k} {uset} {budget} {scope:8s} {meth:12s}: {len(P)} punti fino a {int(z['edges'][-1])} archi")
    return rows


def fig_budget_curves(C, outdir, k, uset="aff"):
    """Riga in alto: T-NDCG@k (valore assoluto, pool dello scope) in funzione degli archi rimossi; la banda è
    l'IC 95% della variazione appaiata rispetto al punto 0. Riga in basso: ΔNDCG@k rispetto al punto 0.
    Una colonna per scope, segmento All. L'ultimo punto di ogni metodo è il suo budget completo."""
    if not C:
        return
    plt = _plt()
    for budget in sorted({r["budget"] for r in C}):
        R0 = [r for r in C if r["budget"] == budget and r["segment"] == "all"]
        scopes = sorted({r["scope"] for r in R0}, key=scope_key)
        if not scopes:
            continue
        fig, axes = plt.subplots(2, len(scopes), figsize=(3.3 * len(scopes), 5.4), squeeze=False, sharex="col")
        for j, sc in enumerate(scopes):
            R = [r for r in R0 if r["scope"] == sc]
            for me in sorted({r["method"] for r in R}, key=method_key):
                P = sorted((r for r in R if r["method"] == me), key=lambda r: r["point"])
                x = np.array([r["edges"] for r in P]) / 1000
                col, mk = style(me)
                lw = 2 if me == "mab_all" else 1.3
                t0 = P[0]["tndcg"]
                y = np.array([r["tndcg"] for r in P])
                lo = t0 - np.array([r["tndcg_removed_hi"] for r in P])
                hi = t0 - np.array([r["tndcg_removed_lo"] for r in P])
                ax = axes[0, j]
                ax.fill_between(x, lo, hi, color=col, alpha=0.15, lw=0)
                ax.plot(x, y, color=col, marker=mk, ms=4.5, mec="white", lw=lw,
                        label=method_label(me, generic=True))
                ax = axes[1, j]
                d = np.array([r["d_ndcg"] for r in P])
                ax.fill_between(x, [r["d_ndcg_lo"] for r in P], [r["d_ndcg_hi"] for r in P], color=col,
                                alpha=0.15, lw=0)
                ax.plot(x, d, color=col, marker=mk, ms=4.5, mec="white", lw=lw)
            axes[0, j].set_title(f"{SCOPE_LABEL.get(sc, sc)} (n={R[0]['n_users']})", fontsize=9, color=INK)
            axes[1, j].axhline(0, color=INK2, lw=0.8)
            axes[1, j].set_xlabel("training edges removed (thousands)")
            for ax in axes[:, j]:
                ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3))
        axes[0, 0].set_ylabel(f"T-NDCG@{k}  (lower = better)")
        axes[1, 0].set_ylabel(f"Δ NDCG@{k}  (vs. 0 edges)")
        _legend(fig, axes, 6)
        fig.suptitle(f"Defective exposure and accuracy as the edge budget grows — {USET_TXT[uset]}, budget {budget}, "
                     f"propagation\nT-NDCG on {POOL_TXT[core.TARGET_POOL]}; each marker = one prefix of the removal list; "
                     "bands: 95% CI of the paired change from 0 edges", fontsize=8, color=INK2, y=1.03)
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        _save(fig, outdir / f"fig_budget_curves{USET_SUFFIX[uset]}_{budget}_k{k}")


def fig_pareto(curves_csv, outdir, k):
    if not curves_csv.exists():
        log(f"  {curves_csv} non trovato: niente fig_pareto (serve dump --curves N e analyze)")
        return
    with open(curves_csv) as f:
        rows = [r for r in csv.DictReader(f) if r["segment"] == "all" and int(r["k"]) == k]
    if not rows:
        return
    plt = _plt()
    for budget in sorted({r["budget"] for r in rows}):
        scopes = sorted({r["scope"] for r in rows if r["budget"] == budget}, key=scope_key)
        fig, axes = plt.subplots(1, len(scopes), figsize=(3.3 * len(scopes), 3.2), squeeze=False)
        for ax, sc in zip(axes[0], scopes):
            R = [r for r in rows if (r["budget"], r["scope"]) == (budget, sc)]
            for me in sorted({r["method"] for r in R}, key=method_key):
                P = sorted((int(r["point"]), float(r["ndcg"]), float(r["t_ndcg_u"]), int(r["edges"]))
                           for r in R if r["method"] == me)
                x = np.array([p[1] - P[0][1] for p in P])
                y = np.array([P[0][2] - p[2] for p in P])
                col, mk = style(me)
                ax.plot(x, y, color=col, marker=mk, ms=5, mec="white", lw=2 if me == "mab_all" else 1.3,
                        label=method_label(me, generic=True))
            ax.axvline(0, color=INK2, lw=0.8)
            ax.axhline(0, color=INK2, lw=0.8)
            ax.set_title(SCOPE_LABEL.get(sc, sc), fontsize=9, color=INK)
            ax.set_xlabel(f"Δ NDCG@{k}")
            ax.ticklabel_format(axis="both", style="sci", scilimits=(-2, 3))
        axes[0, 0].set_ylabel(f"T-NDCG@{k} removed")
        _legend(fig, axes, 6)
        fig.suptitle(f"Trade-off as the edge budget grows (each marker = one prefix of the removal list) — "
                     f"propagation, budget {budget}", fontsize=8, color=INK2, y=1.03)
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        _save(fig, outdir / f"fig_pareto_{budget}_k{k}")


# ── tabelle ──────────────────────────────────────────────────────────────────────────────────────
def _e(x, d=4):
    return "--" if x is None or not np.isfinite(x) else f"{x:.{d}f}"


def latex_tradeoff(T, k, vlabel):
    out = [f"% generated by paper_figures.py — @{k}, {vlabel}; requires booktabs, multirow"]
    for budget in sorted({r["budget"] for r in T}):
        R = sorted([r for r in T if r["budget"] == budget and r["segment"] == "all"],
                   key=lambda r: (scope_key(r["scope"]), method_key(r["method"])))
        out += ["", "\\begin{table}[t]\\centering\\scriptsize",
                f"\\caption{{Accuracy cost and defective exposure removed, affected users, budget {budget}, @{k} "
                f"({vlabel}). $\\Delta$NDCG = after $-$ before, split into relevance lost on flagged items and the "
                f"rest of the list; T-NDCG removed = before $-$ after on {POOL_TXT[core.TARGET_POOL]}; "
                "95\\% bootstrap CI. Users$\\downarrow$: share of affected users whose NDCG decreases.}",
                "\\begin{tabular}{llcccccc}", "\\toprule",
                "Scope & Method & $\\Delta$NDCG & flagged & rest & T-NDCG removed & Ratio & Users$\\downarrow$ \\\\",
                "\\midrule"]
        last = None
        for r in R:
            sc = SCOPE_LABEL.get(r["scope"], r["scope"]) if r["scope"] != last else ""
            last = r["scope"]
            eff = "--" if not np.isfinite(r["efficiency"]) else f"{r['efficiency']:.1f}"
            out.append(f"{sc} & {method_label(r['method'], r['scope'])} & {_e(r['d_ndcg'])} [{_e(r['d_ndcg_lo'])}, "
                       f"{_e(r['d_ndcg_hi'])}] & {_e(r['d_ndcg_flagged'])} & {_e(r['d_ndcg_rest'])} & "
                       f"{_e(r['tndcg_removed'])} [{_e(r['tndcg_removed_lo'])}, {_e(r['tndcg_removed_hi'])}] "
                       f"({100 * r['rel_tndcg_reduction']:.1f}\\%) & {eff} & "
                       f"{100 * r['share_users_ndcg_down']:.1f}\\% \\\\")
        out += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(out) + "\n"
