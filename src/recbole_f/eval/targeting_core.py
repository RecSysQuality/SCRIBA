# Dump per l'analisi: costanti, metriche per utente, test statistici e analisi (solo numpy/scipy).

from __future__ import annotations

import csv
import fnmatch
import re
from types import SimpleNamespace

import numpy as np


SEGMENTS = ("all", "active", "lt")
SEG_LABEL = {"all": "All", "active": "Active", "lt": "Long-tail"}
SCOPE_LABEL = {"group_f": "GSRank","group_ff": "GSRank", "rev2_f": "Rev2","rev2_fff": "Rev2", "all": "All", "gsdb_ff": "GSDB"}
METHOD_ORDER = ("stratified", "heuristic", "random", "regressor", "mab_all", "mab_active", "mab_lt")
DEFAULT_METHODS = ("stratified", "heuristic", "mab*", "exposure_pe","exposure", "random")
DEFAULT_SOURCES = ("group_ff", "gsdb_ff", "rev2_fff")
VARIANT_LABEL = {"retrain": "retraining", "prop": "propagation"}
METHOD_LABEL = {"stratified": "Stratified", "heuristic": "Heuristic", "exposure": "Exposure","exposure_pe": "Exposure PE", "random": "Random",
                "regressor": "Regressor", "mab_all": "MAB-All", "mab_active": "MAB-Active", "mab_lt": "MAB-LT"}
SEGMENT_MABS = ("mab_active", "mab_lt")      # MAB ottimizzati su un segmento: inclusi solo con --segment-mabs


def is_segment_mab(m):
    return any(m == p or m.startswith(p + "_") for p in SEGMENT_MABS)


def want_method(meth, args):
    """Metodo da includere: pattern di --methods e, senza --segment-mabs, niente MAB-Active / MAB-LT."""
    if not getattr(args, "segment_mabs", False) and is_segment_mab(meth):
        return False
    return any(fnmatch.fnmatch(meth, p) for p in args.methods)


def set_segment_mabs(on):
    """Riferimento per segmento ("MAB del segmento"): con i MAB di segmento accesi active -> mab_active,
    lt -> mab_lt; spenti, MAB-All è il riferimento per tutti i segmenti."""
    REF_OF_SEGMENT.update({"all": "mab_all", "active": "mab_active" if on else "mab_all",
                           "lt": "mab_lt" if on else "mab_all"})


def _ref_txt():
    """Come si chiama il riferimento nelle didascalie."""
    return "the segment's MAB" if REF_OF_SEGMENT["lt"] != REF_OF_SEGMENT["all"] else "MAB-All"



def log(msg):
    print(msg, flush=True)


def method_key(m):
    for i, p in enumerate(METHOD_ORDER):
        if m == p or m.startswith(p + "_") and p == "regressor":
            return (i, m)
    return (len(METHOD_ORDER), m)


def method_label(m):
    """Nome per tabelle e figure: Heuristic, Regressor, Regressor (LT), MAB-All, MAB-LT, ..."""
    if m in METHOD_LABEL:
        return METHOD_LABEL[m]
    w = m.split("_")
    seg = lambda x: x.upper() if x == "lt" else x.capitalize()
    if w[0] == "mab":
        return "MAB-" + "-".join(seg(x) for x in w[1:])
    return w[0].capitalize() + "".join(f" ({seg(x)})" for x in w[1:])


def budget_of(sel):
    m = re.search(r"_(\d+K?)$", sel)
    return m.group(1) if m else "-"


REF_OF_SEGMENT = {"all": "mab_all", "active": "mab_active", "lt": "mab_lt"}


# ============================================================================ #
#  FASE 2: METRICHE (solo numpy/scipy)
# ============================================================================ #
def disc(K):
    return 1.0 / np.log2(np.arange(2, K + 2))


def accuracy_pu(hits, npos, k):
    """Per utente: NDCG@k, Recall@k, Hit@k (stesse formule di run_removed_eval)."""
    d = disc(hits.shape[1]); cum = np.cumsum(d)
    h = hits[:, :k].astype(float)
    nh = h.sum(1)
    idcg = cum[np.minimum(npos, k).astype(int) - 1]
    return {f"ndcg@{k}": (h * d[:k]).sum(1) / idcg, f"recall@{k}": nh / npos, f"hit@{k}": (nh > 0).astype(float)}


def gini(counts):
    c = np.sort(counts.astype(float)); n = len(c)
    if n <= 1 or c.sum() == 0:
        return 0.0
    c /= c.sum()
    return float(min(max(2 * np.sum(np.arange(1, n + 1) * c) / n - (n + 1) / n, 0.0), 1.0))


def beyond(top, pop, n_train, catalog, tail, k):
    """Beyond-accuracy su liste [n, K] tagliate a k (formule di run_removed_eval.beyond)."""
    t = top[:, :k]
    if len(t) == 0:
        return {}
    counts = np.bincount(t.ravel(), minlength=len(pop))
    nz = counts[counts > 0]
    p = nz / nz.sum()
    p_i = np.where(pop > 0, pop, 1).astype(float) / max(n_train, 1)
    return {f"entropy@{k}": float(-(p * np.log2(p + 1e-12)).sum()),
            f"coverage@{k}": float(nz.size / catalog),
            f"novelty@{k}": float((-np.log2(p_i[t])).mean(1).mean()),
            f"gini@{k}": gini(nz),
            f"aplt@{k}": float(tail[t].mean(1).mean())}


def novelty_pu(top, pop, n_train, k):
    """Novelty@k per utente (stessa formula di beyond(); la media sugli utenti coincide)."""
    p_i = np.where(pop > 0, pop, 1).astype(float) / max(n_train, 1)
    return {f"novelty_pu@{k}": (-np.log2(p_i[top[:, :k]])).mean(1)}


SPLIT_SUFFIXES = ("_niche", "_head", "_u_niche", "_u_head")


def target_splits(s, tail):
    """Maschere degli item target ristrette a nicchia (long-tail) e head, proprie e dell'unione."""
    return {"_niche": s.tmask & tail, "_head": s.tmask & ~tail,
            "_u_niche": s.t_union & tail, "_u_head": s.t_union & ~tail}


def target_pu(top, tmask, k, n_targets, suffix=""):
    """Per utente: T-HR@k, T-NDCG@k, T-Prec@k e T-DCG@k (non normalizzato)."""
    T = tmask[top[:, :k]]
    d = disc(k)
    dcg = (T * d).sum(1)
    idcg = np.cumsum(d)[max(min(k, n_targets), 1) - 1]
    return {f"t_hr{suffix}@{k}": T.any(1).astype(float), f"t_ndcg{suffix}@{k}": dcg / idcg,
            f"t_prec{suffix}@{k}": T.mean(1), f"t_dcg{suffix}@{k}": dcg}


def target_rank_pu(top0, top_eval, tmask, k):
    """Per utente: rank medio, nella lista top_eval, degli item target che stavano nella top-k del
    modello base (top0). top_eval = top0 -> rank prima; top_eval = lista dopo -> rank dopo.
    Se un item non è nella top-K salvata di top_eval vale K+1 (censura: limite inferiore).
    NaN per gli utenti senza item target nella top-k del base."""
    n, K = top_eval.shape
    out = np.full(n, np.nan)
    T0 = tmask[top0[:, :k]]
    rows = np.flatnonzero(T0.any(1))
    if len(rows) == 0:
        return out
    n_items = np.int64(len(tmask))
    r = np.arange(len(rows), dtype=np.int64)[:, None]
    key1 = (r * n_items + top_eval[rows].astype(np.int64)).ravel()
    order = np.argsort(key1, kind="stable")
    sk = key1[order]
    q = (r * n_items + top0[rows, :k].astype(np.int64)).ravel()
    p = np.minimum(np.searchsorted(sk, q), len(sk) - 1)
    pos = np.where(sk[p] == q, order[p] % K + 1, K + 1).reshape(len(rows), k)   # rank (1..K+1)
    T = T0[rows]
    out[rows] = np.where(T, pos, 0).sum(1) / T.sum(1)
    return out


def _nmean(x):
    """Media sui valori finiti (NaN se non ce ne sono): T-Rank non è definito per tutti gli utenti."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(x.mean()) if len(x) else np.nan


def foe(top, k, gmask, merit, n_items, tag):
    """Fairness of Exposure (Singh & Joachims 2018) per il gruppo di item gmask vs il resto."""
    n = len(top)
    if n == 0:
        return {}
    t = top[:, :k]
    exp_i = np.bincount(t.ravel(), weights=np.tile(disc(k), n), minlength=n_items) / n
    items = np.zeros(n_items, bool); items[1:] = True                     # 0 = padding
    G, R = gmask & items, ~gmask & items
    eg, er = exp_i[G].mean(), exp_i[R].mean()
    ug, ur = merit[G].mean(), merit[R].mean()
    out = {f"foe_{tag}_share@{k}": float(exp_i[G].sum() / exp_i.sum()),
           f"foe_{tag}_dp@{k}": float(eg / er) if er > 0 else np.nan}
    out[f"foe_{tag}_dtr@{k}"] = float((eg / ug) / (er / ur)) if min(ug, ur, er) > 0 else np.nan
    return out


def tail_items(pop, head_share):
    items = np.arange(1, len(pop))
    order = items[np.argsort(-pop[items], kind="stable")]
    tail = np.zeros(len(pop), bool)
    tail[order[int(np.ceil(head_share * len(items))):]] = True
    return tail


def tost(before, after, margin):
    """TOST di equivalenza sulla differenza appaiata (after - before), margine assoluto."""
    from scipy.stats import t as tdist
    d = after - before
    n = len(d)
    if n < 3:
        return np.nan, np.nan, np.nan
    m, se = d.mean(), d.std(ddof=1) / np.sqrt(n)
    if se == 0:
        return float(m), 0.0, 0.0 if abs(m) < margin else 1.0
    p1 = 1 - tdist.cdf((m + margin) / se, n - 1)
    p2 = tdist.cdf((m - margin) / se, n - 1)
    return float(m), float(se), float(max(p1, p2))


def wilcoxon_p(a, b):
    from scipy.stats import wilcoxon
    d = a - b
    if len(d) < 3 or np.all(d == 0):
        return np.nan
    return float(wilcoxon(a, b, zero_method="zsplit").pvalue)


# ---------------------------------------------------------------------------- #
def load_dataset(ds, args):
    L = args.out / ds / "lists"
    B = np.load(L / "base.npz")
    variants = {}
    for variant in args.variants:
        vdir = L if variant == "retrain" else L / variant
        sels = []
        for f in sorted(vdir.glob("*.npz")):
            if f.name in ("base.npz", "candidates.npz") or f.name.endswith(".tmp.npz"):
                continue
            z = np.load(f)
            meth = str(z["method"])
            if not want_method(meth, args):
                continue
            sels.append(SimpleNamespace(sel=str(z["sel"]), method=meth, scope=str(z["scope"]), variant=variant,
                                        budget=budget_of(str(z["sel"])), users=z["users"], top=z["top"],
                                        hits=z["hits"], npos=z["npos"], targets=z["targets"],
                                        members=z["members"], n_edges=int(z["n_edges"]),
                                        chosen=z["chosen"] if "chosen" in z else None))
        if sels:
            variants[variant] = sels
        else:
            log(f"[{ds}] nessuna lista per la variante '{variant}' in {vdir}: salto")
    if not variants:
        raise FileNotFoundError(f"nessuna selezione in {L}")
    if len(variants) > 1:
        names = {v: {s.sel for s in ss} for v, ss in variants.items()}
        common = set.intersection(*names.values())
        for v, ss in variants.items():
            if names[v] - common:
                log(f"[{ds}] ATTENZIONE: selezioni solo in '{v}': {sorted(names[v] - common)}")
    sels = [s for ss in variants.values() for s in ss]      # tutte: stessi utenti per tutte le varianti
    n_users, n_items = int(B["n_users"]), int(B["n_items"])

    # stessi utenti ovunque: utenti di test presenti nel base e in OGNI lista "dopo" (tutte le varianti)
    users = B["users"]
    keep = np.ones(len(users), bool)
    for s in sels:
        pos = np.full(n_users, -1, np.int64); pos[s.users] = np.arange(len(s.users))
        s.rows = pos[users]
        keep &= s.rows >= 0
    users = users[keep]
    for s in sels:
        r = s.rows[keep]
        s.top, s.hits, s.npos = s.top[r], s.hits[r], s.npos[r]
        s.tmask = np.zeros(n_items, bool); s.tmask[s.targets] = True
    base = SimpleNamespace(top=B["top"][keep], hits=B["hits"][keep], npos=B["npos"][keep])

    # membri: unione GLOBALE (tutti i file) -> il segmento intero è lo stesso per ogni scope
    members = np.zeros(n_users, bool)
    t_all = np.zeros(n_items, bool)
    for s in sels:
        members[s.members] = True
        t_all |= s.tmask
    nonmem = ~members[users]
    seg = {"all": nonmem.copy()}
    for g in ("active", "lt"):
        m = np.zeros(n_users, bool); m[B[f"seg_{g}"]] = True
        seg[g] = m[users] & nonmem
    usersets = {("seg", g): seg[g] for g in SEGMENTS}

    # affetti PER (budget, scope): almeno un item rimosso da un metodo di quello scope e budget
    # nella top-affected_k del modello base. Stessi utenti per tutti i metodi (e le varianti)
    # dello stesso scope e budget.
    t_group = {}
    for s in sels:
        key = (s.budget, s.scope)
        t_group.setdefault(key, np.zeros(n_items, bool))
        t_group[key] |= s.tmask
    groups = {}
    for key, tu in sorted(t_group.items()):
        aff = nonmem & tu[base.top[:, :args.affected_k]].any(1)
        us = dict(usersets)
        us.update({("aff", g): seg[g] & aff for g in SEGMENTS})
        us.update({("unaff", g): seg[g] & ~aff for g in SEGMENTS})
        groups[key] = SimpleNamespace(usersets=us, t_union=tu)
    for s in sels:
        s.us = groups[(s.budget, s.scope)].usersets
        s.t_union = groups[(s.budget, s.scope)].t_union

    pop = B["pop"]
    row_of = np.full(n_users, -1, np.int64); row_of[users] = np.arange(len(users))
    tr, ti = row_of[B["test_u"]], B["test_i"]
    ok = tr >= 0
    first = next(iter(variants))
    D = SimpleNamespace(ds=ds, users=users, base=base, sels=variants[first], variants=variants,
                        usersets=usersets, groups=groups, members=members,
                        t_all=t_all, pop=pop, n_train=int(pop.sum()), catalog=n_items - 1,
                        n_items=n_items, tail=tail_items(pop, args.head_share),
                        test_rows=tr[ok], test_items=ti[ok], K=base.top.shape[1], row_of=row_of)
    cp = L / "candidates.npz"
    D.cand = None
    if cp.exists() and not args.only_tables:
        z = np.load(cp)
        D.cand = SimpleNamespace(**{k: z[k] for k in ("src", "row", "item_ptr", "items", "user_ptr",
                                                      "users", "n_edges")})
        D.cand.n = len(D.cand.src)
    else:
        log(f"   {cp} non trovato: salto l'analisi della selezione (rilancia dump)")
    log(f"[{ds}] {len(users)} utenti di test, {int(members[users].sum())} membri esclusi, "
        + ", ".join(f"{v}: {len(ss)} selezioni" for v, ss in variants.items()))
    log(f"   utenti affetti (top-{args.affected_k} del base) per scope e budget:")
    log("     " + f"{'budget':8s}{'scope':10s}{'item':>7s}" + "".join(f"{g:>22s}" for g in SEGMENTS))
    for (budget, scope), gr in groups.items():
        cells = "".join(f"{int(gr.usersets[('aff', g)].sum()):>9d} / {int(seg[g].sum()):<7d}"
                        f"{100 * gr.usersets[('aff', g)].sum() / max(seg[g].sum(), 1):4.0f}%" for g in SEGMENTS)
        log(f"     {budget:8s}{scope:10s}{int(gr.t_union.sum()):7d}" + cells)
    return D


def merit_for(D, mask):
    sel = mask[D.test_rows]
    return np.bincount(D.test_items[sel], minlength=D.n_items).astype(float) / max(mask.sum(), 1)


def targeting(D, args, K):
    """Indici di targeting @K con IC bootstrap appaiato (stesso ricampionamento per tutti i metodi)."""
    rows = []
    rng = np.random.default_rng(args.seed)
    for (budget, scope), sels in by_budget_scope(D.sels).items():
        us = sels[0].us                                   # affetti di questo scope e budget
        idx = np.flatnonzero(us[("aff", "all")])
        if len(idx) == 0:
            continue
        lt = us[("aff", "lt")][idx]
        ac = us[("aff", "active")][idx]
        E_0 = (lambda s: s.Eku0[K]) if args.tr_targets == "union" else (lambda s: s.Ek0[K])
        E_1 = (lambda s: s.Eku1[K]) if args.tr_targets == "union" else (lambda s: s.Ek1[K])
        dE = np.stack([(E_0(s) - E_1(s))[idx] for s in sels], 1)  # utenti x metodi
        E0 = np.stack([E_0(s)[idx] for s in sels], 1)

        def stats(w):
            def mean(X, m):
                ww = w * m
                return (ww @ X) / max(ww.sum(), 1e-12)
            one = np.ones(len(idx), bool)
            d_all, d_lt, d_ac = mean(dE, one), mean(dE, lt), mean(dE, ac)
            e_all, e_lt, e_ac = mean(E0, one), mean(E0, lt), mean(E0, ac)
            with np.errstate(divide="ignore", invalid="ignore"):
                return dict(dE_all=d_all, dE_active=d_ac, dE_lt=d_lt,
                            rr_all=d_all / e_all, rr_active=d_ac / e_ac, rr_lt=d_lt / e_lt,
                            lift_active=d_ac / d_all, lift_lt=d_lt / d_all, TR=d_lt / d_ac,
                            pre_lift_active=e_ac / e_all, pre_lift_lt=e_lt / e_all,
                            pre_TR=e_lt / e_ac)

        point = stats(np.ones(len(idx)))
        boots = {k: [] for k in point}
        for _ in range(args.boot):
            w = np.bincount(rng.integers(0, len(idx), len(idx)), minlength=len(idx)).astype(float)
            for k, v in stats(w).items():
                boots[k].append(v)
        boots = {k: np.array(v) for k, v in boots.items()}
        base_of = {}
        for j, s in enumerate(sels):
            want = args.baseline.get(s.scope, "stratified" if s.scope == "all" else "heuristic")
            if s.method == want:
                base_of[s.scope] = j
        for j, s in enumerate(sels):
            r = dict(dataset=D.ds, k=K, budget=budget, scope=s.scope, method=s.method, sel=s.sel,
                     targets=args.tr_targets, n_aff=len(idx), n_aff_lt=int(lt.sum()), n_aff_active=int(ac.sum()))
            for k in point:
                lo_, hi_ = np.nanpercentile(boots[k][:, j], [2.5, 97.5])
                r[k], r[k + "_lo"], r[k + "_hi"] = float(point[k][j]), float(lo_), float(hi_)
            b = base_of.get(s.scope)
            if b is not None and b != j:
                with np.errstate(divide="ignore", invalid="ignore"):
                    diff = np.log(boots["TR"][:, j]) - np.log(boots["TR"][:, b])
                    r["dlogTR_vs_base"] = float(np.log(point["TR"][j]) - np.log(point["TR"][b]))
                r["dlogTR_lo"], r["dlogTR_hi"] = (float(x) for x in np.nanpercentile(diff, [2.5, 97.5]))
                r["baseline"] = sels[b].method
            rows.append(r)
            log(f"  targeting [{s.variant}] @{K} {budget} {s.scope:8s} {s.method:12s} TR={r['TR']:.3f} "
                f"[{r['TR_lo']:.3f},{r['TR_hi']:.3f}]  lift_lt={r['lift_lt']:.2f} lift_active={r['lift_active']:.2f}")
    return rows


def ref_method(seg):
    return REF_OF_SEGMENT[seg]


def by_budget_scope(sels):
    g = {}
    for s in sels:
        g.setdefault((s.budget, s.scope), []).append(s)
    return {k: sorted(v, key=lambda s: method_key(s.method)) for k, v in sorted(g.items())}


def paired_methods(D, args, K):
    """@K: Delta_u(ref) - Delta_u(altro) sugli stessi affetti; ref = MAB del segmento."""
    rows = []
    rng = np.random.default_rng(args.seed + 1)
    for (budget, scope), sels in by_budget_scope(D.sels).items():
        meths = {s.method: s for s in sels}
        for seg in ("all", "lt", "active"):
            ref = meths.get(ref_method(seg))
            if ref is None:
                continue
            m = ref.us[("aff", seg)]                     # affetti di questo scope e budget
            for o in sels:
                if o is ref:
                    continue
                for kind, a0, a1, b0, b1 in (("own", ref.Ek0[K], ref.Ek1[K], o.Ek0[K], o.Ek1[K]),
                                            ("union", ref.Eku0[K], ref.Eku1[K], o.Eku0[K], o.Eku1[K])):
                    dr, do = (a0 - a1)[m], (b0 - b1)[m]
                    d = dr - do
                    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(args.boot)] if len(d) else [np.nan]
                    rows.append(dict(dataset=D.ds, k=K, budget=budget, scope=scope, segment=seg, ref=ref.method,
                                     other=o.method, targets=kind, n_users=int(m.sum()),
                                     dE_ref=float(dr.mean()), dE_other=float(do.mean()), diff=float(d.mean()),
                                     diff_lo=float(np.nanpercentile(bs, 2.5)), diff_hi=float(np.nanpercentile(bs, 97.5)),
                                     ratio=float(dr.mean() / do.mean()) if do.mean() != 0 else np.nan,
                                     p_wilcoxon=wilcoxon_p(dr, do)))
    return rows


def ugf(D, K):
    """User-oriented fairness (Li et al. 2021): gap active - lt, prima e dopo."""
    rows = []
    for s in D.sels:
        for uset in ("aff", "seg"):
            ma, ml = s.us[(uset, "active")], s.us[(uset, "lt")]
            if ma.sum() == 0 or ml.sum() == 0:
                continue
            for metric in (f"ndcg@{K}", f"recall@{K}"):
                b = s.acc0[metric][ma].mean() - s.acc0[metric][ml].mean()
                a = s.acc1[metric][ma].mean() - s.acc1[metric][ml].mean()
                rows.append(dict(dataset=D.ds, budget=s.budget, scope=s.scope, method=s.method, userset=uset,
                                 metric=metric, gap_before=float(b), gap_after=float(a), delta_gap=float(a - b)))
    return rows


def candidate_scores(D, K, min_users):
    """Per ogni candidato c e segmento g (membri di c esclusi):
         reach_g(c) = quota di utenti di g con almeno un item di c in top-K
         pos_g(c)   = rank medio del PRIMO item di c tra gli utenti raggiunti (1 = in cima)
         exp_g(c)   = esposizione di posizione (Singh & Joachims), media sugli utenti di g
       pos è NaN se gli utenti raggiunti sono meno di min_users."""
    import scipy.sparse as sp
    C = D.cand
    out = {}
    for g in ("lt", "active"):
        rows = np.flatnonzero(D.usersets[("seg", g)])
        ng = max(len(rows), 1)
        segrow = np.full(len(D.users), -1, np.int64); segrow[rows] = np.arange(len(rows))
        Rk = sp.csc_matrix((np.tile(np.arange(1, K + 1), len(rows)),
                            (np.repeat(np.arange(len(rows)), K), D.base.top[rows, :K].ravel())),
                           shape=(len(rows), D.n_items))
        reach, pos, exp = np.zeros(C.n), np.full(C.n, np.nan), np.zeros(C.n)
        dsc = disc(K)
        for c in range(C.n):
            it = C.items[C.item_ptr[c]:C.item_ptr[c + 1]]
            if len(it) == 0:
                continue
            sub = Rk[:, it]
            r, rk = sub.indices, sub.data
            if len(r) == 0:
                continue
            mem = D.row_of[C.users[C.user_ptr[c]:C.user_ptr[c + 1]]]
            mem = segrow[mem[mem >= 0]]
            mem = mem[mem >= 0]
            if len(mem):
                keep = ~np.isin(r, mem)
                r, rk = r[keep], rk[keep]
                if len(r) == 0:
                    continue
            o = np.argsort(r, kind="stable")
            r, rk = r[o], rk[o]
            uniq, start = np.unique(r, return_index=True)
            best = np.minimum.reduceat(rk, start)
            reach[c] = len(uniq) / ng
            exp[c] = dsc[rk - 1].sum() / ng
            if len(uniq) >= min_users:
                pos[c] = best.mean()
        out[g] = dict(reach=reach, pos=pos, exp=exp)
    return out


def cliffs_delta_p(a, b):
    from scipy.stats import mannwhitneyu
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan, np.nan
    u, p = mannwhitneyu(a, b, alternative="two-sided")
    return float(2 * u / (len(a) * len(b)) - 1), float(p)


def holm_adjust(ps):
    ps = np.asarray(ps, float)
    out = np.full(len(ps), np.nan)
    ok = np.flatnonzero(np.isfinite(ps))
    order = ok[np.argsort(ps[ok])]
    run = 0.0
    for r, j in enumerate(order):
        run = max(run, min(1.0, (len(order) - r) * ps[j]))
        out[j] = run
    return out


SEL_METRICS = ("reach", "pos")          # misure dei difetti candidati usate per le classifiche


def selection(D, args, K):
    """
    Prima della rimozione, sul modello base. Due domande separate:
      (1) REACH:    i difetti scelti raggiungono più utenti del segmento?
      (2) POSIZIONE: i loro item stanno più in alto nelle liste del segmento?
    Livello DIFETTO: percentili dei difetti scelti nella classifica di tutti i candidati.
    Livello ITEM:    per gli item rimossi dal metodo, reach (T-HR@K) e best rank per utente.
    """
    out = {k: [] for k in ("selection", "selection_tests", "selection_scores", "thr_curves", "item_rank_ecdf")}
    rng = np.random.default_rng(args.seed + 2)
    seg_rows = {g: np.flatnonzero(D.usersets[("seg", g)]) for g in ("lt", "active")}
    item_best = {}                                   # (sel, g) -> best rank per utente raggiunto

    # ── livello item ─────────────────────────────────────────────────────────
    for s in D.sels:
        nt = max(int(s.tmask.sum()), 1)
        r = dict(dataset=D.ds, budget=s.budget, scope=s.scope, method=s.method, sel=s.sel, n_items=nt)
        boots = {}
        for g, rows in seg_rows.items():
            T = s.tmask[D.base.top[rows, :K]]
            anyT = T.any(1)
            best = (T.argmax(1) + 1)[anyT].astype(float)
            item_best[(s.sel, g)] = best
            r[f"item_reach_{g}"] = float(anyT.mean())                     # T-HR@K sul segmento
            r[f"item_reach_per_item_{g}"] = float(T.sum(1).mean() / nt)
            r[f"item_bestrank_{g}_mean"] = float(best.mean()) if len(best) else np.nan
            r[f"item_bestrank_{g}_median"] = float(np.median(best)) if len(best) else np.nan
            r[f"item_top10pos_{g}"] = float((best <= 10).mean()) if len(best) else np.nan
            r[f"item_top20pos_{g}"] = float((best <= 20).mean()) if len(best) else np.nan
            bs = [best[rng.integers(0, len(best), len(best))].mean() for _ in range(args.boot)] if len(best) else [np.nan]
            r[f"item_bestrank_{g}_lo"], r[f"item_bestrank_{g}_hi"] = (float(x) for x in np.nanpercentile(bs, [2.5, 97.5]))
            curve = T.cumsum(1).mean(0) / nt
            ecdf = np.array([(best <= k).mean() for k in range(1, K + 1)]) if len(best) else np.full(K, np.nan)
            for k in range(K):
                out["thr_curves"].append(dict(dataset=D.ds, budget=s.budget, scope=s.scope, method=s.method,
                                              segment=g, k=k + 1, thr_per_item=float(curve[k])))
                out["item_rank_ecdf"].append(dict(dataset=D.ds, budget=s.budget, scope=s.scope, method=s.method,
                                                  segment=g, k=k + 1, share_best_rank_le_k=float(ecdf[k])))
            e = s.E0[rows] / nt
            r[f"exp_{g}"] = float(e.mean()) if len(e) else np.nan
            boots[g] = np.array([e[rng.integers(0, len(e), len(e))].mean() for _ in range(args.boot)]) \
                if len(e) else np.array([np.nan])
        with np.errstate(divide="ignore", invalid="ignore"):
            r["disparity"] = r["exp_lt"] / r["exp_active"]
            q = boots["lt"] / boots["active"]
        r["disparity_lo"], r["disparity_hi"] = (float(x) for x in np.nanpercentile(q, [2.5, 97.5]))
        out["selection"].append(r)

    # ── livello difetto (servono i candidati) ────────────────────────────────
    vals = {}
    if D.cand is not None:
        C = D.cand
        log("  reach e posizione dei difetti candidati ...")
        sc = candidate_scores(D, K, args.min_reach_users)
        from scipy.stats import rankdata
        valid = C.n_edges > 0
        cache = {}

        def pct_for(scope):
            if scope not in cache:
                pool = valid & ((C.src == scope) if scope in set(C.src.tolist()) else True)
                pc = {}
                for g in ("lt", "active"):
                    for m, sign in (("reach", 1), ("pos", -1)):          # pos: rank basso = percentile alto
                        v = np.full(C.n, np.nan)
                        idx = np.flatnonzero(pool & np.isfinite(sc[g][m]))
                        if len(idx):
                            v[idx] = 100 * (rankdata(sign * sc[g][m][idx]) - 0.5) / len(idx)
                        pc[(m, g)] = v
                cache[scope] = (pool, pc)
            return cache[scope]

        sel_rows = {r["sel"]: r for r in out["selection"]}
        for s in D.sels:
            if s.chosen is None:
                log(f"   {s.sel}: difetti scelti non salvati (rilancia dump)")
                continue
            pool, pc = pct_for(s.scope)
            ch = s.chosen[pool[s.chosen]]
            r = sel_rows[s.sel]
            r.update(n_chosen=len(ch), n_pool=int(pool.sum()))
            for g in ("lt", "active"):
                r[f"reach_{g}_mean"] = float(sc[g]["reach"][ch].mean()) if len(ch) else np.nan
                pv = sc[g]["pos"][ch]
                r[f"pos_{g}_mean"] = float(np.nanmean(pv)) if np.isfinite(pv).any() else np.nan
                for m in SEL_METRICS:
                    v = pc[(m, g)][ch]
                    v = v[np.isfinite(v)]
                    vals[(s.sel, m, g)] = v
                    r[f"pct_{m}_{g}_mean"] = float(v.mean()) if len(v) else np.nan
                    r[f"pct_{m}_{g}_median"] = float(np.median(v)) if len(v) else np.nan
                    r[f"top10_{m}_{g}"] = float((v >= 90).mean()) if len(v) else np.nan
                    r[f"n_{m}_{g}"] = len(v)
            for c in ch:
                row = dict(dataset=D.ds, budget=s.budget, scope=s.scope, method=s.method,
                           source=str(C.src[c]), defect_row=int(C.row[c]), n_edges=int(C.n_edges[c]))
                for g in ("lt", "active"):
                    row.update({f"reach_{g}": float(sc[g]["reach"][c]), f"pos_{g}": float(sc[g]["pos"][c]),
                                f"exp_{g}": float(sc[g]["exp"][c])})
                    for m in SEL_METRICS:
                        row[f"pct_{m}_{g}"] = float(pc[(m, g)][c])
                out["selection_scores"].append(row)

    # ── test: riferimento (mab_lt / mab_active, o solo mab_all) contro gli altri metodi ──
    for (budget, scope), sels in by_budget_scope(D.sels).items():
        meths = {s.method: s for s in sels}
        for refm in dict.fromkeys(REF_OF_SEGMENT[g] for g in ("lt", "active")):
            ref = meths.get(refm)
            if ref is None:
                continue
            for g in ("lt", "active"):
                for level, m, getv, flip in (("defect", "reach", lambda s_: vals.get((s_.sel, "reach", g)), 1),
                                             ("defect", "pos", lambda s_: vals.get((s_.sel, "pos", g)), 1),
                                             ("item", "bestrank", lambda s_: item_best.get((s_.sel, g)), -1)):
                    a = getv(ref)
                    if a is None:
                        continue
                    block = []
                    for o in sels:
                        b = getv(o)
                        if o is ref or b is None:
                            continue
                        cd, p = cliffs_delta_p(flip * a, flip * b)     # >0: ref meglio (più reach / più in alto)
                        block.append(dict(dataset=D.ds, budget=budget, scope=scope, level=level, measure=m,
                                          segment=g, ref=refm, other=o.method, mean_ref=float(np.mean(a)),
                                          mean_other=float(np.mean(b)), cliffs_delta=cd, p=p))
                    for b_, ph in zip(block, holm_adjust([b_["p"] for b_ in block])):
                        b_["p_holm"] = float(ph)
                    out["selection_tests"] += block

    for r in out["selection"]:
        msg = (f"  selezione {r['budget']} {r['scope']:8s} {r['method']:12s} "
               f"item best rank lt={r['item_bestrank_lt_mean']:.1f} active={r['item_bestrank_active_mean']:.1f}")
        if "pct_reach_lt_mean" in r:
            msg += (f" | pct reach lt={r['pct_reach_lt_mean']:.1f} act={r['pct_reach_active_mean']:.1f}"
                    f" | pct pos lt={r['pct_pos_lt_mean']:.1f} act={r['pct_pos_active_mean']:.1f}")
        log(msg)
    return out


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _f(x, d=4):
    return "--" if x is None or not np.isfinite(x) else f"{x:.{d}f}"


def _ttest_row(a, b):
    """t-test appaiato su a - b: media, IC 95%, t, p, d_z (Cohen, appaiato)."""
    from scipy.stats import t as tdist, ttest_rel
    ok = np.isfinite(a) & np.isfinite(b)          # T-Rank: solo utenti con valore definito
    a, b = a[ok], b[ok]
    d = a - b
    n = len(d)
    out = dict(n_users=n, mean_a=float(a.mean()), mean_b=float(b.mean()), diff=float(d.mean()))
    sd = d.std(ddof=1) if n > 1 else np.nan
    if n < 3 or not np.isfinite(sd):
        return dict(out, ci_lo=np.nan, ci_hi=np.nan, t=np.nan, p=np.nan, d_z=np.nan)
    if sd == 0:
        return dict(out, ci_lo=out["diff"], ci_hi=out["diff"], t=np.nan,
                    p=1.0 if out["diff"] == 0 else 0.0, d_z=np.nan)
    h = tdist.ppf(0.975, n - 1) * sd / np.sqrt(n)
    tt = ttest_rel(a, b)
    return dict(out, ci_lo=out["diff"] - h, ci_hi=out["diff"] + h, t=float(tt.statistic),
                p=float(tt.pvalue), d_z=float(out["diff"] / sd))


def ttests(D, args, ks):
    """
    t-test appaiati sugli stessi utenti (affetti dello scope 'aff' e segmento intero 'seg'):
      before_after  ogni metodo, dopo vs prima: accuratezza + T-HR/T-NDCG (item del metodo)
      ref_vs_other  MAB del segmento (all->mab_all, active->mab_active, lt->mab_lt) vs ogni altro
                    metodo, entrambi dopo: accuratezza (+ novelty per utente) + T-HR/T-NDCG
                    sull'unione degli item rimossi nello scope (_u: stesso insieme di item per tutti)
      ref_vs_other_removed  MAB del segmento vs ogni altro metodo: esposizione TOLTA (prima - dopo)
                    di T-NDCG/T-HR sugli item dei difetti rimossi da ciascuno
    Holm per famiglia (tipo, budget, scope, userset, segmento, metrica), sui metodi confrontati.
    Diff = a - b: before_after a = dopo, b = prima; ref_vs_other a = MAB, b = altro.
    """
    acc = [f"{m}@{k}" for k in ks for m in ("ndcg", "recall", "hit", "novelty_pu")]
    own_main = [f"{m}@{k}" for k in ks for m in ("t_ndcg", "t_hr")]
    own = [f"{m}{suf}@{k}" for k in ks for m in ("t_hr", "t_ndcg") for suf in ("", "_niche", "_head")]
    uni = [f"{m}{suf}@{k}" for k in ks for m in ("t_hr", "t_ndcg") for suf in ("_u", "_u_niche", "_u_head")]
    uni += [f"t_rank_u@{k}" for k in ks]
    uni_main = [f"{m}@{k}" for k in ks for m in ("t_hr_u", "t_ndcg_u", "t_rank_u")]     # tabella dei valori assoluti
    rows = []
    for s in D.sels:
        for uset in ("aff", "seg"):
            for g in SEGMENTS:
                m = s.us[(uset, g)]
                if m.sum() < 3:
                    continue
                for metric in acc + own + uni_main:
                    b = (s.acc0 if metric in acc else s.tg0)[metric][m]
                    a = (s.acc1 if metric in acc else s.tg1)[metric][m]
                    rows.append(dict(kind="before_after", budget=s.budget, scope=s.scope, userset=uset,
                                     segment=g, metric=metric, a=s.method, b="before",
                                     **_ttest_row(a, b)))
    for (budget, scope), sels in by_budget_scope(D.sels).items():
        meths = {s.method: s for s in sels}
        for g in SEGMENTS:
            ref = meths.get(REF_OF_SEGMENT[g])
            if ref is None:
                continue
            for uset in ("aff", "seg"):
                m = ref.us[(uset, g)]
                if m.sum() < 3:
                    continue
                for o in sels:
                    if o is ref:
                        continue
                    for metric in acc + uni:
                        src = (lambda x: (x.acc1 if metric in acc else x.tg1)[metric][m])
                        rows.append(dict(kind="ref_vs_other", budget=budget, scope=scope, userset=uset,
                                         segment=g, metric=metric, a=ref.method, b=o.method,
                                         **_ttest_row(src(ref), src(o))))
                    for metric in own_main:          # esposizione TOLTA (prima - dopo), item di ciascuno
                        rem = (lambda x: (x.tg0[metric] - x.tg1[metric])[m])
                        rows.append(dict(kind="ref_vs_other_removed", budget=budget, scope=scope, userset=uset,
                                         segment=g, metric=metric, a=ref.method, b=o.method,
                                         **_ttest_row(rem(ref), rem(o))))
    fam = {}
    for i, r in enumerate(rows):
        fam.setdefault((r["kind"], r["budget"], r["scope"], r["userset"], r["segment"], r["metric"]), []).append(i)
    for idx in fam.values():
        for i, ph in zip(idx, holm_adjust([rows[i]["p"] for i in idx])):
            rows[i]["p_holm"] = float(ph)
            rows[i]["significant"] = bool(np.isfinite(ph) and ph < args.alpha)
    for r in rows:
        r["dataset"] = D.ds
    return rows
