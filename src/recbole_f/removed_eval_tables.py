from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Dict, List

import numpy as np

try:
    from removed_eval_core import (BASELINES, SEG_LABEL, SEGMENTS, SUBSETS, keep_method, log,
                                   selection_parts)
except ImportError:
    from recbole_f.removed_eval_core import (BASELINES, SEG_LABEL, SEGMENTS, SUBSETS, keep_method, log,
                                             selection_parts)


# ============================================================================ #
#  TABELLA LATEX DETTAGLIATA
# ============================================================================ #
def table_metrics(k: int):
    return [(f"ndcg@{k}", f"NDCG@{k}", 4), (f"recall@{k}", f"Recall@{k}", 4),
            (f"hit@{k}", f"Hit@{k}", 4), ("entropy", "Entropy", 3),
            ("coverage", "Coverage", 4), ("novelty", "Novelty", 3),
            ("defect_exposure", "Def.~exp.", 4), ("overlap_base", "Overlap", 3)]


def _tex(s: str) -> str:
    return (str(s).replace("\\", r"\textbackslash{}").replace("_", r"\_")
            .replace("%", r"\%").replace("&", r"\&").replace("#", r"\#"))


def _cell(m: dict, key: str, dig: int, ref: dict | None = None) -> str:
    if not m or not m.get("n_users") or key not in m:
        return "--"
    v = m[key]
    s = f"{v:.{dig}f}"
    if ref is not None and ref.get("n_users") and key in ref:
        b = ref[key]
        if abs(b) > 1e-12:
            s += rf" {{\scriptsize ({(v - b) / abs(b) * 100:+.1f}\%)}}"
        else:
            s += rf" {{\scriptsize ({v - b:+.{dig}f})}}"
    return s


def _block(lines: list, seg_label: str, rows: list, mets) -> None:
    """rows: (nome riga, metriche, riferimento per il delta o None)."""
    for j, (name, m, ref) in enumerate(rows):
        first = rf"\multirow{{{len(rows)}}}{{*}}{{{seg_label}}}" if j == 0 else ""
        n = str(m.get("n_users", 0)) if m else "0"
        cells = " & ".join(_cell(m, key, dig, ref) for key, _, dig in mets)
        lines.append(f"{first} & {name} & {n} & {cells} \\\\")
    lines.append(r"\midrule")


def latex_table(ds: str, recs: List[dict], k: int, scope: str = "all") -> str:
    mets = table_metrics(k)
    r0 = recs[0]
    cand = ("all defect files together (group\\_f, gsdb\\_f, rev2)" if scope == "all"
            else f"only {_tex(scope)} defects")
    L = [r"\begin{table*}[t]", r"\centering", r"\footnotesize"]
    sel_desc = "; ".join(f"{_tex(r['label'])}: {r['n_defects']} defects, {r['n_edges_removed']} edges"
                         for r in recs)
    L.append(rf"\caption{{{_tex(ds)}, candidates: {cand} -- LightGCN before (Base) and after removing the "
             rf"selected defects from the training edges and "
             rf"{'re-propagating with fixed parameters' if r0.get('mode') == 'prop' else 'retraining'} ({sel_desc}). Top-{r0['topk']} lists for "
             rf"entropy, coverage and novelty. In parentheses: relative change with respect to Base on "
             rf"the same users. Affected users: at least one item of a removed defect in their "
             rf"top-{r0['affected_k']} list of the base model.}}")
    L.append(rf"\label{{tab:removed_{re.sub(r'[^A-Za-z0-9]+', '_', f'{ds}_{scope}')}}}")
    L.append(r"\begin{tabular}{ll r " + "r" * len(mets) + "}")
    L.append(r"\toprule")
    L.append("Users & Model & $|U|$ & " + " & ".join(h for _, h, _ in mets) + r" \\")
    L.append(r"\midrule")
    for s in SEGMENTS:                                   # tutti gli utenti del segmento
        base = r0["before"][s]
        rows = [("Base", base, None)] + [(_tex(r["label"]), r["after"][s], r["before"][s]) for r in recs]
        _block(L, SEG_LABEL[s], rows, mets)
    for s in SEGMENTS:                                   # solo gli affetti: base diverso per selezione
        rows = []
        for r in recs:
            lab = _tex(r["label"])
            rows.append((rf"Base\textsubscript{{{lab}}}", r["before"].get(f"aff_{s}", {}), None))
            rows.append((lab, r["after"].get(f"aff_{s}", {}), r["before"].get(f"aff_{s}", {})))
        _block(L, rf"\shortstack[l]{{Affected\\{SEG_LABEL[s]}}}", rows, mets)
    L[-1] = r"\bottomrule"
    L += [r"\end{tabular}", r"\end{table*}", ""]
    return "\n".join(L)


# ============================================================================ #
#  TABELLE PER IL PAPER
# ============================================================================ #
PAPER_SEGMENTS = [("all", "All users"), ("active", "Active users"), ("lt", "Long-tail users")]   # lo filtra main()


def _paper_value(m: dict, key: str):
    """Valore per la tabella del paper: gini diventa 1-gini (più alto = meglio)."""
    if key.startswith("1-gini"):
        v = m.get("gini" + key[len("1-gini"):])
        return None if v is None else 1.0 - v
    return m.get(key)


def _paper_table(recs: List[dict], cols, caption: str, label: str, rank_prefixes=None, sig=None) -> str:
    r0 = recs[0]
    competes = [not rank_prefixes or selection_parts(r["selection"])[0].startswith(tuple(rank_prefixes))
                for r in recs]

    groups = []                                           # intestazione: metrica sopra, cutoff sotto
    for _, head, _ in cols:
        name, _, k = head.partition("@")
        if groups and groups[-1][0] == name:
            groups[-1][1].append(k)
        else:
            groups.append((name, [k]))
    rules, c0 = [], 3
    for _, ks_ in groups:
        rules.append(rf"\cmidrule(lr){{{c0}-{c0 + len(ks_) - 1}}}")
        c0 += len(ks_)

    L = [r"\begin{table*}[t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
         rf"\caption{{{caption}}}", rf"\label{{{label}}}",
         r"\begin{tabular}{ll" + "r" * len(cols) + "}", r"\toprule",
         " & & " + " & ".join(rf"\multicolumn{{{len(ks_)}}}{{c}}{{{nm}}}" for nm, ks_ in groups) + r" \\",
         " ".join(rules),
         "Users & Method & " + " & ".join(f"@{k}" for _, ks_ in groups for k in ks_) + r" \\",
         r"\midrule"]

    for si, (seg, seg_name) in enumerate(PAPER_SEGMENTS):
        rows = [("Base", r0["before"].get(seg, {}), True, False, None)]
        rows += [(_tex(r["label"]), r["after"].get(seg, {}), False, ok, r["selection"])
                 for r, ok in zip(recs, competes)]
        best = {}                                         # colonna -> valori dei metodi, dal migliore
        for key, _, _ in cols:
            best[key] = sorted({round(v, 6) for _, m, _, ok, _ in rows
                                if ok and m.get("n_users") and (v := _paper_value(m, key)) is not None},
                               reverse=True)
        seg_cell = r"\shortstack[l]{" + seg_name.replace(" ", r"\\") + "}"
        for j, (name, m, is_base, ok, sel_) in enumerate(rows):
            cells = []
            for key, _, dig in cols:
                v = _paper_value(m, key)
                if v is None or not m.get("n_users"):
                    cells.append("--")
                    continue
                txt = f"{v:.{dig}f}"
                if is_base:
                    txt = rf"\textit{{{txt}}}"
                elif ok and best[key] and round(v, 6) == best[key][0]:
                    txt = rf"\textbf{{{txt}}}"
                elif ok and len(best[key]) > 1 and round(v, 6) == best[key][1]:
                    txt = rf"\underline{{{txt}}}"
                if sig and sig.get((sel_, seg, key)):
                    txt += sig[(sel_, seg, key)]
                cells.append(txt)
            first = rf"\multirow{{{len(rows)}}}{{*}}{{{seg_cell}}}" if j == 0 else ""
            L.append(f"{first} & {name} & " + " & ".join(cells) + r" \\")
            if is_base:
                L.append(rf"\cmidrule(lr){{2-{2 + len(cols)}}}")
        L.append(r"\midrule" if si < len(PAPER_SEGMENTS) - 1 else r"\bottomrule")
    L += [r"\end{tabular}", r"\end{table*}", ""]
    return "\n".join(L)


# ============================================================================ #
#  SIGNIFICATIVITÀ (t-test appaiato, correzione di Holm)
# ============================================================================ #
def paired_ttest_p(a: np.ndarray, b: np.ndarray) -> float:
    """p-value a due code del t-test appaiato (come scipy.stats.ttest_rel)."""
    from scipy.special import stdtr
    d = a - b
    n = len(d)
    sd = d.std(ddof=1)
    if n < 2 or sd == 0:
        return 1.0 if d.mean() == 0 else 0.0
    t = d.mean() / (sd / np.sqrt(n))
    return float(2 * stdtr(n - 1, -abs(t)))


def load_peruser(out: Path, ds: str, sel: str):
    f = out / ds / "peruser" / f"{sel}.npz"
    if not f.exists():
        return None
    z = dict(np.load(f, allow_pickle=False))
    return z if "in_all" in z else None                  # file vecchi senza gruppi: niente test


def holm(pvals: Dict[str, float]) -> Dict[str, float]:
    """p-value corretti con Holm-Bonferroni (step-down)."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, run, out = len(items), 0.0, {}
    for rank, (k, p_) in enumerate(items):
        run = max(run, min(1.0, (m - rank) * p_))
        out[k] = run
    return out


def significance(out: Path, ds: str, recs: List[dict], refs: Dict[str, str], keys, alpha: float,
                 correction: str = "holm"):
    cache = {}

    def z_of(r):
        if r["selection"] not in cache:
            cache[r["selection"]] = load_peruser(out, ds, r["selection"])
        return cache[r["selection"]]

    marks, used = {}, {}
    for seg in SEGMENTS:
        ref = next((r for r in recs if selection_parts(r["selection"])[0] == refs.get(seg)), None)
        if ref is None:
            continue
        zr = z_of(ref)
        if zr is None:
            log(f"  [{ds}] test saltato per '{seg}': per-utente mancante per {ref['selection']} "
                "(rivaluta con la versione nuova e --skip-existing)")
            continue
        used[seg] = ref
        pos_r = {u: i_ for i_, u in enumerate(zr["users"].tolist())}
        others = [r for r in recs if r is not ref]
        for key in keys:
            pv, better = {}, {}
            for r in others:
                za = z_of(r)
                if za is None or key not in za or key not in zr:
                    continue
                ia = np.array([i_ for i_, u in enumerate(za["users"].tolist()) if u in pos_r], dtype=np.int64)
                ir = np.array([pos_r[u] for u in za["users"][ia].tolist()], dtype=np.int64)
                msk = za[f"in_{seg}"][ia] & zr[f"in_{seg}"][ir]
                if msk.sum() < 2:
                    continue
                a = zr[key][ir][msk].astype(float)          # riferimento
                b = za[key][ia][msk].astype(float)          # altro metodo
                pv[r["selection"]] = paired_ttest_p(a, b)
                better[r["selection"]] = a.mean() > b.mean()
            if not pv:
                continue
            padj = holm(pv) if correction == "holm" else pv
            wins = [sel for sel in pv if better[sel] and padj[sel] < alpha]
            for sel in wins:
                marks[(sel, seg, key)] = "$^{*}$"
            if len(wins) == len(pv) == len(others):
                marks[(ref["selection"], seg, key)] = r"$^{\dagger}$"
    return marks, used


# ============================================================================ #
#  TABELLE COMPLETE: paper, compatta, vittorie, riepilogo
# ============================================================================ #
def paper_tables(ds: str, scope: str, recs: List[dict], ks, beyond_ks, rank_prefixes=None,
                 out: Path | None = None, sig_refs: Dict[str, str] | None = None, alpha: float = 0.05,
                 correction: str = "holm") -> str:
    r0 = recs[0]
    cand = "all defect files" if scope == "all" else f"{_tex(scope)} defects only"
    who = {"none": "all test users",
           "own": "test users that are not members of the removed defects",
           "union": "test users that are not members of any removed defect set (same users for every method)"
           }[r0.get("exclude_members", "none")]
    if r0.get("eval_users") == "risk":
        who += (f", restricted to at-risk users (at least {r0['risk_min']} items of the defects removed by "
                f"any of the compared methods in their top-{r0['risk_k']} before removal; "
                f"same users for every method)")
    ranked = ("the best removal method" if not rank_prefixes else
              "the best among " + ", ".join("MAB" if p_ == "mab" else _tex(p_) for p_ in rank_prefixes))
    how = ("LightGCN with fixed parameters, re-propagated on the training graph" if r0.get("mode") == "prop"
           else "LightGCN retrained")
    note = (f"{how} after removing {r0['n_edges_removed']} training edges selected by each "
            f"method (candidates: {cand}); evaluated on {who}. Base: model before removal (reference, in "
            f"italics). Higher is better; within each user group {ranked} is in bold, the second is underlined.")
    tail_note = (r"APLT: share of long-tail items in the lists (long-tail = all but the "
                 + f"{int(round(100 * r0.get('head_share', 0.2)))}" + r"\% most popular training items).")
    lab = re.sub(r"[^A-Za-z0-9]+", "_", f"{ds}_{scope}")
    acc = [(f"{m}@{k}", f"{h}@{k}", 4)
           for m, h in (("ndcg", "NDCG"), ("recall", "Recall"), ("mrr", "MRR")) for k in ks]
    bey = [(f"{m}@{k}", f"{h}@{k}", d)
           for m, h, d in (("coverage", "Coverage", 3), ("entropy", "Entropy", 2), ("novelty", "Novelty", 2),
                           ("1-gini", "1$-$Gini", 3), ("aplt", "APLT", 3)) for k in beyond_ks]
    sig, used = ({}, {})
    if sig_refs and out is not None:
        sig, used = significance(out, ds, recs, sig_refs, [c[0] for c in acc], alpha, correction)
    acc_note = note
    if used:
        refs_txt = "; ".join(f"{name}: {_tex(used[g]['label'])}" for g, name in PAPER_SEGMENTS if g in used)
        corr = " with Holm correction" if correction == "holm" else ""
        acc_note += (rf" Paired t-test on the same users against the method targeted at each user group "
                     rf"({refs_txt}), $p<{alpha}${corr}. $^{{*}}$: the targeted method is significantly "
                     r"better than this method. $^{\dagger}$: the targeted method is significantly better "
                     r"than all the other methods.")
    return (_paper_table(recs, acc, rf"{_tex(ds)} -- accuracy. {acc_note}", f"tab:paper_acc_{lab}",
                         rank_prefixes, sig)
            + "\n" +
            _paper_table(recs, bey, rf"{_tex(ds)} -- beyond-accuracy. {note} {tail_note}",
                         f"tab:paper_beyond_{lab}", rank_prefixes))


def compact_table(ds: str, scope: str, recs: List[dict], k: int = 100, rank_prefixes=None,
                  out: Path | None = None, sig_refs: Dict[str, str] | None = None, alpha: float = 0.05,
                  correction: str = "holm") -> str:
    """Una sola tabella: Hit, Recall, Entropy, Coverage, Novelty a @k (default 100)."""
    r0 = recs[0]
    cand = "all defect files" if scope == "all" else f"{_tex(scope)} defects only"
    who = {"none": "all test users",
           "own": "test users that are not members of the removed defects",
           "union": "test users that are not members of any compared defect set (same users for every method)"
           }[r0.get("exclude_members", "none")]
    if r0.get("eval_users") == "risk":
        who += (f", restricted to at-risk users (at least {r0['risk_min']} items of the defects removed by "
                f"any of the compared methods in their top-{r0['risk_k']} before removal)")
    ranked = ("the best removal method" if not rank_prefixes else
              "the best among " + ", ".join("MAB" if p_ == "mab" else _tex(p_) for p_ in rank_prefixes))
    how = ("LightGCN with fixed parameters, re-propagated on the training graph" if r0.get("mode") == "prop"
           else "LightGCN retrained")
    note = (f"{how} after removing {r0['n_edges_removed']} training edges selected by each "
            f"method (candidates: {cand}); evaluated on {who}. Base: model before removal (reference, in "
            f"italics). All metrics at @{k}, higher is better; within each user group {ranked} is in bold, "
            f"the second is underlined.")
    cols = [(f"hit@{k}", f"Hit@{k}", 4), (f"recall@{k}", f"Recall@{k}", 4),
            (f"entropy@{k}", f"Entropy@{k}", 2), (f"coverage@{k}", f"Coverage@{k}", 3),
            (f"novelty@{k}", f"Novelty@{k}", 2)]
    sig, used = ({}, {})
    if sig_refs and out is not None:
        sig, used = significance(out, ds, recs, sig_refs, [f"hit@{k}", f"recall@{k}"], alpha, correction)
    if used:
        refs_txt = "; ".join(f"{name}: {_tex(used[g]['label'])}" for g, name in PAPER_SEGMENTS if g in used)
        corr = " with Holm correction" if correction == "holm" else ""
        note += (rf" Hit and Recall: paired t-test on the same users against the method targeted at each "
                 rf"user group ({refs_txt}), $p<{alpha}${corr}. $^{{*}}$: the targeted method is "
                 r"significantly better than this method. $^{\dagger}$: the targeted method is significantly "
                 r"better than all the other methods.")
    lab = re.sub(r"[^A-Za-z0-9]+", "_", f"{ds}_{scope}")
    return _paper_table(recs, cols, rf"{_tex(ds)} -- {note}", f"tab:compact_{lab}", rank_prefixes, sig)


SCOPE_LABEL = {"group_f": r"group\_f", "gsdb_f": r"gsdb\_f", "rev2": "rev2", "all": "all sources"}


def wins_table(groups, ks, rank_prefixes, out: Path) -> str:
    """
    Un blocco di righe per scope (candidati: group_f, gsdb_f, rev2, all).
    Per ogni metodo e gruppo di utenti: in quanti casi è il migliore tra i metodi che competono,
    con casi = dataset x {NDCG, Recall, MRR} x cutoff. Più alto = meglio.
    """
    SEGMENTS = tuple(s_ for s_, _ in PAPER_SEGMENTS)     # solo i gruppi di utenti scelti (--paper-segments)
    keys = [f"{m}@{k}" for m in ("ndcg", "recall", "mrr") for k in ks]
    scopes = sorted({sc for _, sc in groups}, key=lambda sc: (sc == "all", sc))
    datasets = sorted({ds for ds, _ in groups})
    L = [r"\begin{table}[t]", r"\centering", r"\footnotesize",
         r"\caption{Number of cases (" + ", ".join(_tex(d) for d in datasets)
         + r" $\times$ NDCG/Recall/MRR $\times$ cutoffs " + ", ".join(str(k) for k in ks)
         + r") in which each method is the best for each user group, per candidate set. Higher is better.}",
         r"\label{tab:paper_wins}",
         r"\begin{tabular}{ll" + "r" * len(SEGMENTS) + "}", r"\toprule",
         "Candidates & Method & " + " & ".join(name for _, name in PAPER_SEGMENTS) + r" \\", r"\midrule"]
    any_rows = False
    for si, scope in enumerate(scopes):
        count: Dict[str, Dict[str, int]] = {}
        n_cases = {s_: 0 for s_ in SEGMENTS}
        labels = {}
        for (ds, sc), recs in groups.items():
            if sc != scope:
                continue
            cand = [r for r in recs if not rank_prefixes
                    or selection_parts(r["selection"])[0].startswith(tuple(rank_prefixes))]
            for r in cand:
                meth = selection_parts(r["selection"])[0]
                labels[meth] = r["label"]
                count.setdefault(meth, {s_: 0 for s_ in SEGMENTS})
            for seg in SEGMENTS:
                for key in keys:
                    vals = [(r["after"].get(seg, {}).get(key), selection_parts(r["selection"])[0]) for r in cand]
                    vals = [(v, m_) for v, m_ in vals if v is not None]
                    if len(vals) < 2:
                        continue
                    n_cases[seg] += 1
                    top = max(v for v, _ in vals)
                    for v, m_ in vals:
                        if round(v, 6) == round(top, 6):
                            count[m_][seg] += 1
        if not count:
            continue
        any_rows = True
        meths = sorted(count)
        for j, meth in enumerate(meths):
            cells = []
            for seg in SEGMENTS:
                best = max(count[m_][seg] for m_ in count)
                c = f"{count[meth][seg]}/{n_cases[seg]}"
                cells.append(rf"\textbf{{{c}}}" if count[meth][seg] == best and best > 0 else c)
            first = rf"\multirow{{{len(meths)}}}{{*}}{{{SCOPE_LABEL.get(scope, _tex(scope))}}}" if j == 0 else ""
            L.append(f"{first} & {_tex(labels[meth])} & " + " & ".join(cells) + r" \\")
        L.append(r"\midrule" if si < len(scopes) - 1 else r"\bottomrule")
    if not any_rows:
        return ""
    if L[-1] == r"\midrule":
        L[-1] = r"\bottomrule"
    L += [r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(L)


def write_summary(records: List[dict], out: Path, k: int, rank_prefixes=None,
                  ks=(10, 20, 50, 100), beyond_ks=(20, 100), sig_refs=None, alpha=0.05,
                  paper_scopes=None, correction="holm", compact_k=100) -> None:
    if not records:
        log("Nessun risultato: tabella non scritta")
        return
    # una tabella per (dataset, scope): euristica e regressori sugli stessi candidati
    groups: Dict[tuple, List[dict]] = {}
    for r in records:
        scope = r.get("scope") or selection_parts(r["selection"])[1]
        groups.setdefault((r["dataset"], scope), []).append(r)
    scope_rank = lambda sc: (sc == "all", sc)                    # prima le singole sorgenti, poi all
    tex = ["% Richiede \\usepackage{booktabs} e \\usepackage{multirow}", ""]
    per_ds: Dict[str, List[str]] = {}
    for (ds, scope) in sorted(groups, key=lambda g: (g[0], scope_rank(g[1]))):
        # prima le baseline (stratified / heuristic / random), poi i regressori
        recs = sorted(groups[(ds, scope)],
                      key=lambda r: (selection_parts(r["selection"])[0] not in BASELINES, r["selection"]))
        t = latex_table(ds, recs, k, scope)
        per_ds.setdefault(ds, []).append(t)
        tex.append(t)
    for ds, ts in per_ds.items():
        (out / ds).mkdir(parents=True, exist_ok=True)
        (out / ds / "summary.tex").write_text("\n".join(ts), encoding="utf-8")
    (out / "summary_removed.tex").write_text("\n".join(tex), encoding="utf-8")

    # tabelle per il paper: valori assoluti, più alto = meglio, niente delta
    paper = ["% Richiede \\usepackage{booktabs} e \\usepackage{multirow}. Valori assoluti: più alto = meglio.",
             "% Celle '--': metrica assente nel JSON (valutato con una versione precedente: "
             "rivaluta con --skip-existing).", ""]
    pgroups = {g: v for g, v in groups.items() if not paper_scopes or g[1] in paper_scopes}
    for (ds, scope) in sorted(pgroups, key=lambda g: (g[0], scope_rank(g[1]))):
        recs = sorted(pgroups[(ds, scope)],
                      key=lambda r: (selection_parts(r["selection"])[0] not in BASELINES, r["selection"]))
        paper.append(f"% ===== {ds} · candidati: {scope} · accuratezza + beyond-accuracy =====")
        paper.append(paper_tables(ds, scope, recs, ks, beyond_ks, rank_prefixes, out, sig_refs, alpha,
                                  correction))
    compact = [f"% Richiede \\usepackage{{booktabs}} e \\usepackage{{multirow}}. "
               f"Hit, Recall, Entropy, Coverage, Novelty @{compact_k}.", ""]
    for (ds, scope) in sorted(pgroups, key=lambda g: (g[0], scope_rank(g[1]))):
        recs = sorted(pgroups[(ds, scope)],
                      key=lambda r: (selection_parts(r["selection"])[0] not in BASELINES, r["selection"]))
        compact.append(f"% ===== {ds} · candidati: {scope} · tabella compatta @{compact_k} =====")
        compact.append(compact_table(ds, scope, recs, compact_k, rank_prefixes, out, sig_refs, alpha,
                                     correction))
    (out / "paper_compact.tex").write_text("\n".join(compact), encoding="utf-8")
    log(f"Tabella compatta @{compact_k} → {out / 'paper_compact.tex'}")
    paper.append("% ===== riepilogo: vittorie per candidati e gruppo di utenti =====")
    paper.append(wins_table(pgroups, ks, rank_prefixes, out))
    (out / "paper_removed.tex").write_text("\n".join(paper), encoding="utf-8")
    log(f"Tabelle paper → {out / 'paper_removed.tex'}")

    keys = sorted({m for r in records for st in ("before", "after") for s in r[st].values() for m in s})
    keys = ["n_users"] + [m for m in keys if m != "n_users"]
    with open(out / "summary_removed.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["dataset", "scope", "method", "selection", "n_defects", "n_edges_removed",
                    "users", "state"] + keys)
        for r in records:
            meth, scope = selection_parts(r["selection"])
            for s in SUBSETS:
                for st in ("before", "after", "delta"):
                    m = r[st].get(s, {})
                    w.writerow([r["dataset"], scope, meth, r["selection"], r["n_defects"],
                                r["n_edges_removed"], s, st]
                               + [m.get(key, "") for key in keys])
    log(f"Tabella LaTeX → {out / 'summary_removed.tex'}")
    log(f"CSV          → {out / 'summary_removed.csv'}")


def load_records(out: Path, datasets=None) -> List[dict]:
    """Un record per (dataset, selezione): results/<sel>.json; i vecchi results.jsonl solo se
    la stessa selezione non ha già un file nuovo."""
    recs = {}
    for p in sorted(out.glob("*/results.jsonl")):          # formato vecchio
        with open(p) as f:
            for l in f:
                if l.strip():
                    r = json.loads(l)
                    recs[(r["dataset"], r["selection"])] = r
    for p in sorted(out.glob("*/results/*.json")):         # formato nuovo: vince
        with open(p) as f:
            r = json.load(f)
        recs[(r["dataset"], r["selection"])] = r
    return [r for r in recs.values() if (not datasets or r["dataset"] in datasets)
            and keep_method(r["selection"])]
