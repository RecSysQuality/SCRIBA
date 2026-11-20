# Valutazione per il paper (trade-off accuratezza / esposizione dei difetti rimossi).
# Uso: python3 eval.py --datasets ...

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import eval_core as core
from eval_core import (
    DEFAULT_KEEP_ALSO, DEFAULT_METHODS, METHOD_LABEL, POOL_TXT, SCOPE_LABEL, SEGMENTS, USET_TXT, boot, ci,
    dist_divergence, exposure, js, kl, load, log, method_label, per_user, user_js
)
from eval_report import (
    SOURCE_ORDER, budget_curves, fig_budget_curves, fig_decomp, fig_dist, fig_fairness, fig_kl, fig_pareto,
    fig_sources, fig_tradeoff, fig_tradeoff_grid, fig_tradeoff_pairs, fig_users, latex_kl, latex_kl_compact,
    latex_shares, latex_tradeoff
)


def analyze(D, a, k, uset="aff"):
    """uset = 'aff' (utenti affetti dello scope, tutte le analisi) oppure 'seg' (segmento intero, membri dei
    difetti esclusi): con 'seg' solo quello che serve al trade-off (niente KL né divergenze)."""
    rng = np.random.default_rng(a.seed)
    light = uset != "aff"
    mask = (lambda gr, g: gr.aff[g]) if uset == "aff" else (lambda gr, g: D.seg[g])
    base_cache = {}
    T, F, V, LC, U = [], [], [], [], []
    for s in D.sels:
        gr = D.groups[(s.budget, s.scope)]
        key = (s.budget, s.scope)
        if key not in base_cache:
            base_cache[key] = per_user(D.base.top, D.base.hits, D.base.npos, gr.t_union, k)
            u0 = base_cache[key]
            for g in SEGMENTS:
                m = mask(gr, g)
                if m.sum() == 0:
                    continue
                nf = u0["nflag"][m].sum()
                LC.append(dict(dataset=D.ds, k=k, budget=s.budget, scope=s.scope, segment=g, n_users=int(m.sum()),
                               flagged_slots=int(nf), prec_flagged=float(u0["hflag"][m].sum() / max(nf, 1)),
                               prec_other=float(u0["hother"][m].sum() / max(k * m.sum() - nf, 1)),
                               users_with_flagged_hit=float((u0["hflag"][m] > 0).mean())))
        u0 = base_cache[key]
        u1 = per_user(s.top, s.hits, s.npos, gr.t_union, k)
        jsu = user_js(D.base.top, s.top, k, D.n_items)
        meta = dict(dataset=D.ds, variant=a.variant, userset=uset, k=k, budget=s.budget, scope=s.scope,
                    method=s.method, n_edges=s.n_edges)
        rr = {}
        for g in SEGMENTS:
            m = mask(gr, g)
            n = int(m.sum())
            if n < 3:
                continue
            dn = (u1["ndcg"] - u0["ndcg"])[m]
            dnf = (u1["ndcg_flag"] - u0["ndcg_flag"])[m]
            rem = (u0["tndcg"] - u1["tndcg"])[m]
            t0b = u0["tndcg"][m]
            # colonne del bootstrap: 0 ΔNDCG, 1 T-NDCG tolto, 2 NDCG prima, 3 T-NDCG prima
            bs = boot(np.stack([dn, rem, u0["ndcg"][m], t0b], 1), a.boot, rng)
            nd0 = float(u0["ndcg"][m].mean())
            r = dict(meta, segment=g, n_users=n, ndcg_before=nd0, ndcg_after=float(u1["ndcg"][m].mean()),
                     d_ndcg=float(dn.mean()), d_ndcg_lo=ci(bs[:, 0])[0], d_ndcg_hi=ci(bs[:, 0])[1],
                     d_ndcg_flagged=float(dnf.mean()), d_ndcg_rest=float((dn - dnf).mean()),
                     rel_d_ndcg=float(dn.mean() / nd0) if nd0 else np.nan,
                     tost_margin=a.tost_margin * nd0,
                     tndcg_before=float(u0["tndcg"][m].mean()), tndcg_after=float(u1["tndcg"][m].mean()),
                     tndcg_removed=float(rem.mean()), tndcg_removed_lo=ci(bs[:, 1])[0],
                     tndcg_removed_hi=ci(bs[:, 1])[1],
                     rel_tndcg_reduction=float(rem.mean() / u0["tndcg"][m].mean()) if u0["tndcg"][m].mean() else np.nan,
                     efficiency=float(rem.mean() / -dn.mean()) if dn.mean() < 0 else np.inf,
                     share_users_ndcg_down=float((dn < -1e-12).mean()), share_users_ndcg_up=float((dn > 1e-12).mean()),
                     share_users_exposure_down=float((rem > 1e-12).mean()),
                     user_js_mean=float(jsu[m].mean()), user_js_median=float(np.median(jsu[m])))
            if not light:
                # KL tra le distribuzioni per utente (non tra le medie): NDCG prima/dopo e T-NDCG prima/dopo
                n0, n1, t0, t1 = u0["ndcg"][m], u1["ndcg"][m], u0["tndcg"][m], u1["tndcg"][m]
                r["kl_ndcg"], r["js_ndcg"] = dist_divergence(n0, n1)
                r["kl_tndcg"], r["js_tndcg"] = dist_divergence(t0, t1)
                # soglia di rumore: KL tra la distribuzione PRIMA e un suo ricampionamento bootstrap (mediana).
                # È la KL che si osserva quando la distribuzione non cambia: sotto questa soglia = nessun cambiamento
                nk = (u0["_noise_kl"].get((s.budget, s.scope, g)) if "_noise_kl" in u0 else None)
                if nk is None:
                    bk = np.array([[dist_divergence(n0, n0[ix])[0], dist_divergence(t0, t0[ix])[0]]
                                   for ix in (rng.integers(0, n, n) for _ in range(max(a.boot // 5, 50)))])
                    nk = (float(np.median(bk[:, 0])), float(np.median(bk[:, 1])))
                    u0.setdefault("_noise_kl", {})[(s.budget, s.scope, g)] = nk
                r["kl_ndcg_noise"], r["kl_tndcg_noise"] = nk
            T.append(r)
            rr[g] = (rem, t0b, dn, u0["ndcg"][m], u1["ndcg"][m], bs)
            U.append(dict(meta, segment=g, nflag=u0["nflag"][m], rem=rem, dn=dn, js=jsu[m]))
            if light:
                continue
            # divergenze: sulle distribuzioni, non sulle medie
            from scipy.stats import ks_2samp, wasserstein_distance
            e0, e1 = exposure(D.base.top[m], k, D.n_items), exposure(s.top[m], k, D.n_items)
            keepm = ~gr.t_union
            keepm[0] = False
            e0n, e1n = e0 * keepm, e1 * keepm
            bins = np.linspace(0, 1, 51)
            h0 = np.histogram(u0["ndcg"][m], bins)[0].astype(float) + 1e-9
            h1 = np.histogram(u1["ndcg"][m], bins)[0].astype(float) + 1e-9
            ks = ks_2samp(u0["ndcg"][m], u1["ndcg"][m])
            V.append(dict(meta, segment=g, n_users=n,
                          exp_kl_after_before=kl(e1, e0), exp_js=js(e1, e0),
                          exp_kl_nonflagged=kl(e1n, e0n) if e1n.sum() and e0n.sum() else np.nan,
                          exp_js_nonflagged=js(e1n, e0n) if e1n.sum() and e0n.sum() else np.nan,
                          ndcg_hist_js=js(h1, h0), ndcg_ks=float(ks.statistic), ndcg_ks_p=float(ks.pvalue),
                          ndcg_wasserstein=float(wasserstein_distance(u0["ndcg"][m], u1["ndcg"][m])),
                          user_js_mean=float(jsu[m].mean()), user_js_median=float(np.median(jsu[m])),
                          user_js_p90=float(np.percentile(jsu[m], 90)),
                          share_users_same_list=float((jsu[m] < 1e-12).mean())))
        if "active" in rr and "lt" in rr:
            # fairness tra segmenti: riduzione relativa di T-NDCG = Σ (T-NDCG prima - dopo) / Σ T-NDCG prima
            # (rapporto delle medie, ricalcolato su ogni ricampionamento bootstrap); NDCG relativo; gap UGF
            (rma, ta, dna, n0a, n1a, ba), (rml, tl, dnl, n0l, n1l, bl) = rr["active"], rr["lt"]
            rta, rtl = ba[:, 1] / ba[:, 3], bl[:, 1] / bl[:, 3]
            rna, rnl = ba[:, 0] / ba[:, 2], bl[:, 0] / bl[:, 2]
            gap_b, gap_a = n0a.mean() - n0l.mean(), n1a.mean() - n1l.mean()
            dgap = ba[:, 0] - bl[:, 0]
            F.append(dict(meta, n_active=len(rma), n_lt=len(rml),
                          tndcg_red_active=float(rma.sum() / ta.sum()) if ta.sum() else np.nan,
                          tndcg_red_active_lo=ci(rta)[0], tndcg_red_active_hi=ci(rta)[1],
                          tndcg_red_lt=float(rml.sum() / tl.sum()) if tl.sum() else np.nan,
                          tndcg_red_lt_lo=ci(rtl)[0], tndcg_red_lt_hi=ci(rtl)[1],
                          rel_ndcg_active=float(dna.mean() / n0a.mean()), rel_ndcg_active_lo=ci(rna)[0],
                          rel_ndcg_active_hi=ci(rna)[1],
                          rel_ndcg_lt=float(dnl.mean() / n0l.mean()), rel_ndcg_lt_lo=ci(rnl)[0],
                          rel_ndcg_lt_hi=ci(rnl)[1],
                          ugf_gap_before=float(gap_b), ugf_gap_after=float(gap_a), ugf_delta_gap=float(gap_a - gap_b),
                          ugf_delta_gap_lo=ci(dgap)[0], ugf_delta_gap_hi=ci(dgap)[1]))
        log(f"  @{k} {s.budget} {s.scope:8s} {s.method:12s} fatto")
    add_tests(T, U, a)
    return T, F, V, LC, U


def holm(ps):
    ps = np.asarray(ps, float)
    out = np.full(len(ps), np.nan)
    ok = np.flatnonzero(np.isfinite(ps))
    run = 0.0
    for r, j in enumerate(ok[np.argsort(ps[ok])]):
        run = max(run, min(1.0, (len(ok) - r) * ps[j]))
        out[j] = run
    return out


def add_tests(T, U, a):
    """Test appaiati sugli stessi utenti affetti (come targeting_eval), aggiunti alle righe di T:
      p_ndcg_vs_base, p_rem_vs_base  t-test: ΔNDCG ≠ 0, T-NDCG tolto ≠ 0
      tost_p, ndcg_equivalent        TOST: ΔNDCG entro ±tost_margin·NDCG prima (accuratezza invariata)
      p_rem_vs_mab, p_ndcg_vs_mab    t-test appaiato MAB-All vs metodo (T-NDCG tolto, ΔNDCG), Holm sui metodi
                                     dello stesso scope."""
    from scipy.stats import t as tdist, ttest_1samp, ttest_rel
    by = {}
    for t_, u in zip(T, U):
        by.setdefault((t_["k"], t_["budget"], t_["scope"], t_["segment"]), []).append((t_, u))
    for grp in by.values():
        for t_, u in grp:
            dn, rem = u["dn"], u["rem"]
            t_["p_ndcg_vs_base"] = float(ttest_1samp(dn, 0).pvalue) if dn.std() > 0 else np.nan
            t_["p_rem_vs_base"] = float(ttest_1samp(rem, 0).pvalue) if rem.std() > 0 else np.nan
            n, mu, se = len(dn), dn.mean(), dn.std(ddof=1) / np.sqrt(len(dn))
            mg = t_["tost_margin"]
            if se > 0:
                p = max(1 - tdist.cdf((mu + mg) / se, n - 1), tdist.cdf((mu - mg) / se, n - 1))
            else:
                p = 0.0 if abs(mu) < mg else 1.0
            t_["tost_p"], t_["ndcg_equivalent"] = float(p), bool(p < a.alpha)
        ref = next((x for x in grp if x[0]["method"] == "mab_all"), None)
        others = [x for x in grp if x[0]["method"] != "mab_all"]
        if ref is None or not others:
            continue
        for key, col in (("rem", "p_rem_vs_mab"), ("dn", "p_ndcg_vs_mab")):
            ps = [float(ttest_rel(ref[1][key], o[1][key]).pvalue) for o in others]
            for (t_, _), ph in zip(others, holm(ps)):
                t_[col] = float(ph)


# ── tabelle dei valori assoluti (tables_abs, funzioni di targeting_eval) ─────────────────────────
def _te():
    """Importa le funzioni di dump_for_analysis (metriche, tabelle LaTeX) dalla cartella di questo script."""
    import importlib
    import sys
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    return importlib.import_module("targeting_core"), importlib.import_module("targeting_report")


def abs_rows(D, a, ks, te):
    """Righe nel formato di targeting_eval (metrics_long, ttests, tests) per le sole colonne di tables_abs:
    NDCG, HR, Coverage, Novelty, Gini, T-NDCG, T-HR, T-Rank (le T- sul pool degli item difettosi dello scope),
    utenti 'aff' (affetti dello scope) e 'seg' (segmento intero, membri esclusi). Stesse formule e stessi test
    di targeting_eval.analyze_variant / ttests: t-test appaiati, Holm per (tipo, budget, scope, utenti,
    segmento, metrica) sui metodi presenti; TOST su NDCG (solo aff)."""
    rows_long, rows_tt, rows_test = [], [], []
    pop = D.pop
    n_train, catalog = int(pop.sum()), D.n_items - 1
    tail = te.tail_items(pop, 0.2)                        # serve solo ad APLT, che la tabella non mostra
    acc_names = lambda k: (f"ndcg@{k}", f"hit@{k}", f"novelty_pu@{k}")
    t_names = lambda k: (f"t_ndcg_u@{k}", f"t_hr_u@{k}", f"t_rank_u@{k}")
    acc0 = {}
    for k in ks:
        acc0.update(te.accuracy_pu(D.base.hits, D.base.npos, k))
        acc0.update(te.novelty_pu(D.base.top, pop, n_train, k))
    tg0_cache, after = {}, {}

    def usermask(gr, uset, g):
        return gr.aff[g] if uset == "aff" else D.seg[g]

    for s in D.sels:
        gr = D.groups[(s.budget, s.scope)]
        tu, nt = gr.t_union, int(gr.t_union.sum())
        if (s.budget, s.scope) not in tg0_cache:
            tg0 = {}
            for k in ks:
                tg0.update(te.target_pu(D.base.top, tu, k, nt, "_u"))
                tg0[f"t_rank_u@{k}"] = te.target_rank_pu(D.base.top, D.base.top, tu, k)
            tg0_cache[(s.budget, s.scope)] = tg0
        tg0 = tg0_cache[(s.budget, s.scope)]
        acc1, tg1 = {}, {}
        for k in ks:
            acc1.update(te.accuracy_pu(s.hits, s.npos, k))
            acc1.update(te.novelty_pu(s.top, pop, n_train, k))
            tg1.update(te.target_pu(s.top, tu, k, nt, "_u"))
            tg1[f"t_rank_u@{k}"] = te.target_rank_pu(D.base.top, s.top, tu, k)
        after[s.sel] = (acc1, tg1)
        meta = dict(dataset=D.ds, variant=a.variant, budget=s.budget, scope=s.scope, method=s.method, sel=s.sel,
                    n_edges=s.n_edges)
        for uset in a.abs_usersets:
            for g in SEGMENTS:
                m = usermask(gr, uset, g)
                n = int(m.sum())
                if n == 0:
                    continue
                kw = dict(meta, userset=uset, segment=g, n_users=n)
                for k in ks:
                    for name in (f"ndcg@{k}", f"hit@{k}"):
                        rows_long.append(dict(kw, metric=name, before=float(acc0[name][m].mean()),
                                              after=float(acc1[name][m].mean())))
                    b0 = te.beyond(D.base.top[m], pop, n_train, catalog, tail, k)
                    b1 = te.beyond(s.top[m], pop, n_train, catalog, tail, k)
                    for name in (f"coverage@{k}", f"novelty@{k}", f"gini@{k}"):
                        rows_long.append(dict(kw, metric=name, before=b0[name], after=b1[name]))
                    for name in t_names(k):
                        rows_long.append(dict(kw, metric=name, before=te._nmean(tg0[name][m]),
                                              after=te._nmean(tg1[name][m])))
                    if n >= 3:
                        for name in acc_names(k) + t_names(k):
                            b_, a_ = (acc0[name], acc1[name]) if name in acc0 else (tg0[name], tg1[name])
                            rows_tt.append(dict(kind="before_after", budget=s.budget, scope=s.scope, userset=uset,
                                                segment=g, metric=name, a=s.method, b="before",
                                                **te._ttest_row(a_[m], b_[m])))
                    if uset == "aff":
                        name = f"ndcg@{k}"
                        b_, a_ = acc0[name][m], acc1[name][m]
                        md, se, p = te.tost(b_, a_, a.tost_margin * b_.mean())
                        rows_test.append(dict(meta, segment=g, n_users=n, metric=name, test="TOST",
                                              before=float(b_.mean()), after=float(a_.mean()), mean_diff=md, se=se,
                                              margin=a.tost_margin * b_.mean(), p=p,
                                              equivalent=bool(np.isfinite(p) and p < a.alpha)))
    # metodo di riferimento del segmento contro gli altri, entrambi dopo la rimozione
    groups = {}
    for s in D.sels:
        groups.setdefault((s.budget, s.scope), []).append(s)
    for (budget, scope), sels in groups.items():
        meths = {s.method: s for s in sels}
        gr = D.groups[(budget, scope)]
        for g in SEGMENTS:
            ref = meths.get(te.REF_OF_SEGMENT[g])
            if ref is None:
                continue
            for uset in a.abs_usersets:
                m = usermask(gr, uset, g)
                if m.sum() < 3:
                    continue
                for o in sels:
                    if o is ref:
                        continue
                    for k in ks:
                        for name in acc_names(k) + t_names(k):
                            j = 0 if name in acc0 else 1
                            rows_tt.append(dict(kind="ref_vs_other", budget=budget, scope=scope, userset=uset,
                                                segment=g, metric=name, a=ref.method, b=o.method,
                                                **te._ttest_row(after[ref.sel][j][name][m], after[o.sel][j][name][m])))
    fam = {}
    for i, r in enumerate(rows_tt):
        fam.setdefault((r["kind"], r["budget"], r["scope"], r["userset"], r["segment"], r["metric"]), []).append(i)
    for idx in fam.values():
        for i, ph in zip(idx, te.holm_adjust([rows_tt[i]["p"] for i in idx])):
            rows_tt[i]["p_holm"] = float(ph)
            rows_tt[i]["significant"] = bool(np.isfinite(ph) and ph < a.alpha)
    for r in rows_long:
        r["delta"] = r["after"] - r["before"]
    for r in rows_tt:
        r["dataset"] = D.ds
    return rows_long, rows_tt, rows_test


def nohead_note(a):
    """Nota per le didascalie delle tabelle quando il nohead è attivo."""
    if a.exclude_head_pct:
        note = f"; the {a.exclude_head_pct:g}\\% most popular items are not counted as defective"
    elif a.head_max_pop is not None:
        note = (f"; items with more than {a.head_max_pop:g} training interactions are not counted as defective")
    elif a.head_quantile is not None:
        note = (f"; items whose training popularity exceeds the {100 * a.head_quantile:g}th percentile are not "
                "counted as defective (same rule used by the GSDB and Rev2 detectors)")
    else:
        return ""
    if a.nohead_sources:
        note += " when flagged only by " + ", ".join(SCOPE_LABEL.get(x, x) for x in a.nohead_sources)
    return note


def write_abs_tables(D, a, ks, outdir, vlabel):
    """tables_abs_k<k>_<us>.tex + CSV, con le funzioni di targeting_eval e le etichette di questo script."""
    te, tr = _te()
    te.set_segment_mabs(a.segment_mabs)
    rows_long, rows_tt, rows_test = abs_rows(D, a, ks, te)
    write_csv(outdir / "abs_metrics.csv", rows_long)
    write_csv(outdir / "abs_ttests.csv", rows_tt)
    write_csv(outdir / "abs_tost.csv", rows_test)
    old = tr.method_label
    tr.method_label = lambda m: method_label(m)              # SCRIBA; con --one-baseline la baseline è "Source"
    old_sl = dict(te.SCOPE_LABEL)
    te.SCOPE_LABEL.update(SCOPE_LABEL)                       # nomi degli scope di questo script (GSRank, ...)
    try:
        for k in ks:
            for us in a.abs_usersets:
                txt = tr.latex_abs_tables(rows_long, rows_tt, rows_test, k, us, vlabel + nohead_note(a), a.alpha, D.K)
                txt = txt.replace("MAB-All", METHOD_LABEL["mab_all"])
                if core.TARGET_POOL == "detected":                # didascalia: pool di tutti i difetti rilevati
                    txt = txt.replace("the union of the defective items removed in the scope "
                                      "(same item set for every method)", POOL_TXT["detected"])
                if core.SOURCE_BASELINE:
                    txt = txt.replace("Base = before removal;", "Base = before removal; Source = the defects ranked "
                                      "by the source itself;")
                if a.exclude_members == "candidates":
                    txt = txt.replace("defect members excluded", "users of any candidate defect excluded")
                (outdir / f"tables_abs_k{k}_{us}.tex").write_text(txt)
    finally:
        tr.method_label = old
        te.SCOPE_LABEL.clear()
        te.SCOPE_LABEL.update(old_sl)
    log(f"  tables_abs: k={list(ks)}, utenti={list(a.abs_usersets)} -> {outdir}")


def source_rows(D):
    """Per lo scope 'all': quanti difetti (e archi) di ciascuna sorgente ha scelto ogni metodo. Legge la
    sorgente dei difetti scelti da lists/candidates.npz (indici 'chosen' salvati dal dump)."""
    cp = D.lists_dir / "candidates.npz"
    if not cp.exists():
        log(f"  {cp} non trovato: niente composizione per sorgente")
        return []
    z = np.load(cp)
    src, ne = z["src"].astype(str), z["n_edges"]
    pool = {sc: int((src == sc).sum()) for sc in np.unique(src)}
    rows = []
    for s in D.sels:
        if s.scope != "all" or s.chosen is None or len(s.chosen) == 0:
            continue
        cs, ce = src[s.chosen], ne[s.chosen]
        tot_d, tot_e = len(cs), int(ce.sum())
        for sc in sorted(set(cs) | set(SOURCE_ORDER) & set(pool), key=lambda x: (SOURCE_ORDER.index(x)
                                                                                if x in SOURCE_ORDER else 9, x)):
            m = cs == sc
            rows.append(dict(dataset=D.ds, budget=s.budget, method=s.method, source=sc,
                             n_defects=int(m.sum()), share_defects=float(m.sum() / max(tot_d, 1)),
                             n_edges=int(ce[m].sum()), share_edges=float(ce[m].sum() / max(tot_e, 1)),
                             n_candidates=pool.get(sc, 0)))
    return rows


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0], epilog=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--out", type=Path, default=Path("results/targeting"))
    p.add_argument("--variant", choices=["retrain", "prop"], default="retrain", help="liste 'dopo' da usare")
    p.add_argument("--load-variants", nargs="+", default=["retrain", "prop"],
                   help="varianti caricate per definire gli utenti comuni: come --variants di targeting_eval "
                        "analyze (default entrambe), così gli utenti affetti coincidono")
    p.add_argument("--k", nargs="+", type=int, default=[20])
    p.add_argument("--affected-k", type=int, default=20)
    p.add_argument("--target-pool", choices=["detected", "selected"], default="detected",
                   help="item difettosi di T-NDCG/T-HR/T-Rank e degli utenti affetti: detected = tutti i difetti "
                        "rilevati da ogni detector (lists/candidates.npz, anche quelli che nessun metodo ha scelto; "
                        "default), selected = unione dei difetti scelti dai metodi caricati (comportamento vecchio)")
    p.add_argument("--exclude-head-pct", type=float, default=None,
                   help="toglie dagli item difettosi il P%% di item più popolari nel training (es. 5)")
    p.add_argument("--head-quantile", type=float, default=None, metavar="Q",
                   help="toglie dagli item difettosi gli item con popolarità di training > quantile Q delle "
                        "popolarità (es. 0.80, stessa regola di GSDB e rev2_defects.py)")
    p.add_argument("--head-max-pop", type=float, default=None, metavar="N",
                   help="toglie dagli item difettosi gli item con più di N interazioni di training")
    p.add_argument("--nohead-sources", nargs="+", default=None, metavar="SORGENTE",
                   help="con una regola head: toglie solo gli item head che vengono da difetti di queste sorgenti "
                        "(es. rev2). Nello scope all un item head segnalato anche da un'altra sorgente resta")
    p.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    p.add_argument("--segment-mabs", action="store_true")
    p.add_argument("--keep-also", nargs="*", default=list(DEFAULT_KEEP_ALSO), metavar="METODO",
                   help="con --one-baseline: metodi tenuti oltre a SCRIBA e alla baseline dello scope "
                        "(default: exposure_pe random; '--keep-also' senza valori = nessuno)")
    p.add_argument("--exclude-members", choices=["candidates", "removed"], default="candidates",
                   help="utenti esclusi da tutte le valutazioni: candidates = utenti di TUTTI i difetti candidati "
                        "(lists/candidates.npz; default, stessi utenti per ogni metodo e scope), removed = solo "
                        "gli utenti dei difetti rimossi da qualche metodo (comportamento vecchio, cartelle senza "
                        "suffisso _candmem)")
    p.add_argument("--one-baseline", action="store_true",
                   help="figure e tabelle solo con MAB-All e UNA baseline per scope (come targeting_eval: "
                        "all=stratified, gli altri=heuristic; cambiala con --baseline). Gli utenti affetti e il "
                        "pool degli item difettosi non cambiano")
    p.add_argument("--baseline", nargs="*", default=[], metavar="SCOPE=METODO",
                   help="baseline per scope con --one-baseline, es. rev2=stratified")
    p.add_argument("--no-abs-tables", dest="abs_tables", action="store_false",
                   help="non scrivere tables_abs_k<k>_<us>.tex")
    p.add_argument("--abs-usersets", nargs="+", choices=["aff", "seg"], default=["aff", "seg"],
                   help="utenti delle tabelle tables_abs: aff = affetti dello scope, seg = segmento intero")
    p.add_argument("--tradeoff-usersets", nargs="+", choices=["aff", "seg"], default=["aff", "seg"],
                   help="utenti delle figure di trade-off (fig_tradeoff, _grid, _pairs): aff = affetti dello scope, "
                        "seg = tutti gli utenti (membri dei difetti esclusi; file con suffisso _seg)")
    p.add_argument("--tost-margin", type=float, default=0.01)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--boot", type=int, default=500)
    p.add_argument("--seed", type=int, default=2020)
    p.add_argument("--tag", default=None)
    p.add_argument("--sel-tag", default=None, metavar="TAG",
                   help="quando dello stesso metodo (stesso scope e budget) ci sono più versioni, usa quella con "
                        "questo suffisso dopo il budget (es. new_version per heuristic__gsdb_ff_50K_new_version; "
                        "'' = quella senza suffisso). I metodi con una sola versione si usano comunque. Default: "
                        "nessuna preferenza, ci si ferma se ci sono doppioni")
    a = p.parse_args()
    a.baseline = dict(x.split("=", 1) for x in a.baseline)
    core.SOURCE_BASELINE = a.one_baseline
    core.TARGET_POOL = a.target_pool
    if a.variant not in a.load_variants:
        a.load_variants.append(a.variant)
    n_rules = sum(x is not None for x in (a.exclude_head_pct, a.head_quantile, a.head_max_pop))
    if n_rules > 1:
        p.error("usa una sola regola tra --exclude-head-pct, --head-quantile e --head-max-pop")
    if a.head_quantile is not None and not 0 < a.head_quantile < 1:
        p.error("--head-quantile deve stare tra 0 e 1 (es. 0.80)")
    if a.nohead_sources and not n_rules:
        p.error("--nohead-sources richiede --exclude-head-pct, --head-quantile o --head-max-pop")
    if a.tag is None:
        a.tag = f"affk{a.affected_k}"
        srcs = "_" + "-".join(a.nohead_sources) if a.nohead_sources else ""
        if a.exclude_head_pct:
            a.tag += f"_nohead{a.exclude_head_pct:g}" + srcs
        elif a.head_quantile is not None:
            a.tag += f"_headq{100 * a.head_quantile:g}" + srcs
        elif a.head_max_pop is not None:
            a.tag += f"_headmax{a.head_max_pop:g}" + srcs
    a.out_suffix = f"_{a.sel_tag}" if a.sel_tag else ""   # versioni diverse dei file -> cartelle diverse
    if a.target_pool == "selected":
        a.out_suffix += "_selpool"                        # il vecchio pool non sovrascrive il nuovo
    if a.exclude_members == "candidates":
        a.out_suffix += "_candmem"                        # utenti diversi: non sovrascrive i risultati vecchi
        USET_TXT["seg"] = "all users (users of any candidate defect excluded)"
    vlabel = {"retrain": "retraining", "prop": "propagation"}[a.variant]
    gridT = {k: [] for k in a.k}
    gridTseg = {k: [] for k in a.k}             # trade-off su tutti gli utenti (solo figure di trade-off)
    gridS = []
    for ds in a.datasets:
        D = load(ds, a)
        outdir = a.out / ds / f"final_paper_{a.tag}{a.out_suffix}_{a.variant}"
        outdir.mkdir(parents=True, exist_ok=True)
        allT, allF, allV, allL, allC = [], [], [], [], []
        for k in a.k:
            T, F, V, LC, U = analyze(D, a, k)
            allT += T; allF += F; allV += V; allL += LC
            gridT[k] += T
            if "seg" in a.tradeoff_usersets:            # stesso piano, tutti gli utenti (membri esclusi)
                Ts = analyze(D, a, k, "seg")[0]
                allT += Ts
                gridTseg[k] += Ts
                try:
                    fig_tradeoff(Ts, outdir, k, vlabel, "seg")
                except Exception as e:
                    log(f"  fig_tradeoff (seg) non generata: {e!r}")
            for fn, args in ((fig_tradeoff, (T, outdir, k, vlabel)) if "aff" in a.tradeoff_usersets else
                             (lambda *x: None, ()), (fig_kl, (T, outdir, k, vlabel)),
                             (fig_decomp, (T, LC, outdir, k, vlabel)),
                             (fig_users, (U, outdir, k, vlabel)), (fig_dist, (U, outdir, k, vlabel)),
                             (fig_fairness, (F, outdir, k, vlabel)),
                             (fig_pareto, (a.out / ds / f"analysis_{a.tag}" / "curves.csv", outdir, k))):
                try:
                    fn(*args)
                except Exception as e:
                    log(f"  {fn.__name__} non generata: {e!r}")
            (outdir / f"tables_tradeoff_k{k}.tex").write_text(latex_tradeoff(T, k, vlabel))
            (outdir / f"tables_kl_k{k}.tex").write_text(latex_kl(T, k, vlabel))
            (outdir / f"tables_shares_k{k}.tex").write_text(latex_shares(T, k, vlabel))
            (outdir / f"tables_kl_ndcg_k{k}.tex").write_text(latex_kl_compact(T, k, vlabel))
            for us in a.tradeoff_usersets:                 # curve: T-NDCG al crescere del budget (prop)
                C = budget_curves(D, a, k, us)
                allC += C
                try:
                    fig_budget_curves(C, outdir, k, us)
                except Exception as e:
                    log(f"  fig_budget_curves ({us}) non generata: {e!r}")
        if a.abs_tables:
            write_abs_tables(D, a, [k for k in a.k if k <= D.K], outdir, vlabel)
        write_csv(outdir / "tradeoff.csv", allT)
        write_csv(outdir / "fairness.csv", allF)
        write_csv(outdir / "divergence.csv", allV)
        write_csv(outdir / "label_check.csv", allL)
        write_csv(outdir / "budget_curves.csv", allC)
        S = source_rows(D)
        write_csv(outdir / "sources_all.csv", S)
        gridS += S
        try:
            fig_sources(S, outdir)
        except Exception as e:
            log(f"  fig_sources non generata: {e!r}")
        log(f"[{ds}] risultati in {outdir}")
    if len(a.datasets) > 1:                      # figura unica per il paper: righe = dataset, colonne = scope
        gdir = a.out / f"paper_{a.tag}{a.out_suffix}_{a.variant}"
        gdir.mkdir(parents=True, exist_ok=True)
        try:
            fig_sources(gridS, gdir, name="fig_sources_grid")
            write_csv(gdir / "sources_all.csv", gridS)
        except Exception as e:
            log(f"  fig_sources (griglia) non generata: {e!r}")
        for k, T in gridT.items():
            figs = [(fig_kl, (T, gdir, k, vlabel, True))]
            for us, TT in (("aff", T), ("seg", gridTseg[k])):
                if us not in a.tradeoff_usersets or not TT:
                    continue
                figs.append((fig_tradeoff_grid, (TT, gdir, k, vlabel, us)))
                if a.one_baseline:
                    figs.append((fig_tradeoff_pairs, (TT, gdir, k, vlabel, us)))
            for fn, args in figs:
                try:
                    fn(*args)
                except Exception as e:
                    log(f"  {fn.__name__} (griglia) non generata: {e!r}")
            (gdir / f"tables_kl_k{k}.tex").write_text(latex_kl(T, k, vlabel))
            (gdir / f"tables_shares_k{k}.tex").write_text(latex_shares(T, k, vlabel))
            (gdir / f"tables_kl_ndcg_k{k}.tex").write_text(latex_kl_compact(T, k, vlabel))
        log(f"figura a griglia (tutti i dataset) in {gdir}")


if __name__ == "__main__":
    main()
