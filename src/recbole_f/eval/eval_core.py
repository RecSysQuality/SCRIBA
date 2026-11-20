# Valutazione per il paper: costanti, etichette, caricamento delle liste e metriche per utente.

from __future__ import annotations

import fnmatch
import re
from types import SimpleNamespace as NS

import numpy as np


SEGMENTS = ("all", "active", "lt")
SEG_LABEL = {"all": "All", "active": "Active", "lt": "Long-tail"}


SCOPE_ORDER = ("group_f", "gsdb_f", "rev2", "all")
SCOPE_LABEL = {"group_f": "GSRank", "gsdb_f": "GSDB", "rev2": "Rev2", "all": "All"}
METHOD_ORDER = ("stratified", "heuristic", "random", "exposure", "regressor", "mab_all",
                "mab_active", "mab_lt")
METHOD_LABEL = {"stratified": "Stratified", "heuristic": "Heuristic", "random": "Random",
                "exposure": "Exp-total", "exposure_pe": "Exp",
                "regressor": "Regressor", "mab_all": "SCRIBA", "mab_active": "MAB-Active", "mab_lt": "MAB-LT"}
DEFAULT_METHODS = ("stratified", "heuristic", "random", "exposure*", "regressor*", "mab*")
# con --one-baseline restano SCRIBA, la baseline dello scope e questi (--keep-also)
DEFAULT_KEEP_ALSO = ("exposure", "random")
SEGMENT_MABS = ("mab_active", "mab_lt")

# stile: MAB-All = slot 1 (blu); baseline = slot 2-3 + grigi, sempre con marcatore diverso
# (scatter: oltre 3 colori la sola tinta non basta, il marcatore è la codifica secondaria)
STYLE = {"mab_all": ("#2a78d6", "o"), "stratified": ("#eb6834", "s"), "heuristic": ("#1baf7a", "D"),
         "regressor": ("#52514e", "^"), "random": ("#a3a29c", "v"),
         "exposure_pe": ("#c48a00", "p"), "exposure": ("#8a6d3b", "<"),
         "mab_active": ("#4a3aa7", "P"), "mab_lt": ("#e34948", "X")}
SEG_COLOR = {"active": "#2a78d6", "lt": "#eb6834", "all": "#52514e"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

TARGET_POOL = "detected"     # --target-pool: detected = tutti i difetti dei detector, selected = difetti scelti
POOL_TXT = {"detected": "all the items flagged by the detectors of the scope (including defects that no method "
                        "selected; same item set for every method)",
            "selected": "the union of the defective items removed in the scope (same item set for every method)"}


def log(msg):
    print(msg, flush=True)


def method_key(m):
    for i, p in enumerate(METHOD_ORDER):
        if m == p or (m.startswith(p + "_") and p == "regressor"):
            return (i, m)
    return (len(METHOD_ORDER), m)


SOURCE_BASELINE = False      # --one-baseline: la baseline di ogni scope si chiama come la sorgente (All, Group, ...)
BASELINES = ("stratified", "heuristic")


def is_baseline(m):
    return SOURCE_BASELINE and m in BASELINES


def method_label(m, scope=None, generic=False):
    """Nome per figure e tabelle. Con --one-baseline la baseline prende il nome dello scope (Stratified nello
    scope all -> "All", Heuristic in group_f -> "GSRank", ...); generic=True -> "Source" (legende condivise
    da pannelli di scope diversi)."""
    if is_baseline(m):
        return "Source" if generic or scope is None else SCOPE_LABEL.get(scope, scope)
    if m in METHOD_LABEL:
        return METHOD_LABEL[m]
    w = m.split("_")
    seg = lambda x: x.upper() if x == "lt" else x.capitalize()
    if w[0] == "mab":
        return "MAB-" + "-".join(seg(x) for x in w[1:])
    return w[0].capitalize() + "".join(f" ({seg(x)})" for x in w[1:])


def style(m):
    if is_baseline(m):                       # una sola baseline per scope: stesso stile ovunque
        return STYLE["stratified"]
    if m in STYLE:
        return STYLE[m]
    return STYLE["regressor"] if m.startswith("regressor") else ("#7a7974", "h")


def scope_key(s):
    return (SCOPE_ORDER.index(s) if s in SCOPE_ORDER else len(SCOPE_ORDER), s)


# Nome della selezione: <metodo>[__<scope>]_<budget>[K|k][_<tag>]
#   es. heuristic__gsdb_ff_50K_new_version, mab__rev2_50K_new_version, mab_all__gsdb_50K, random_50K
# Metodo, scope e budget si ricavano QUI dal nome, non da method/scope salvati dal dump: il dump li calcola con
# run_removed_eval.selection_parts, che non riconosce il budget se dopo c'è un tag (_new_version) e non conosce
# gli scope gsdb_ff / rev2_f.
SEL_RE = re.compile(r"^(?P<meth>.+?)(?:__(?P<scope>.+?))?_(?P<budget>\d+)(?P<unit>[kK]?)(?:_(?P<tag>.+))?$")
SCOPE_ALIASES = {"group": "group_f", "group_ff": "group_f",
                 "gsdb": "gsdb_f", "gsdb_ff": "gsdb_f","rev2_fff": "rev2",
                 "rev2_f": "rev2"}


def parse_sel(sel):
    """'heuristic__gsdb_ff_50K_new_version' -> ('heuristic', 'gsdb_f', '50K', 'new_version');
    'mab__lt_50K' -> ('mab_lt', 'all', '50K', ''); 'random_50K' -> ('random', 'all', '50K', '')."""
    m = SEL_RE.match(sel)
    if m is None:
        raise ValueError(f"nome di selezione non riconosciuto: {sel!r} (atteso <metodo>__<scope>_<budget>[_<tag>])")
    meth, scope = m["meth"], m["scope"] or "all"
    scope = SCOPE_ALIASES.get(scope, scope)
    if scope in ("active", "lt"):                    # mab__lt_50K: segmento obiettivo, candidati = tutti
        meth, scope = f"{meth}_{scope}", "all"
    if meth == "mab":
        meth = "mab_all"
    return meth, scope, m["budget"] + m["unit"].upper(), m["tag"] or ""


def budget_of(sel):
    return parse_sel(sel)[2]


def pick_versions(sels, prefer, ds):
    """Una sola selezione per (variante, budget, scope, metodo). I file possono avere o no un tag dopo il budget
    (heuristic__gsdb_ff_50K_new_version e mab__gsdb_50K si confrontano tra loro). Se per lo stesso metodo ci sono
    più versioni: si tiene quella con il tag --sel-tag ('' = quella senza tag); senza --sel-tag ci si ferma."""
    by = {}
    for s in sels:
        by.setdefault((s.variant, s.budget, s.scope, s.method), []).append(s)
    out, amb = [], {}
    for key, ss in by.items():
        if len(ss) == 1:
            out.append(ss[0])
            continue
        hit = [s for s in ss if prefer is not None and s.tag == prefer]
        if len(hit) == 1:
            out.append(hit[0])
            log(f"[{ds}] {key[0]} {key[1]} {key[2]} {key[3]}: uso {hit[0].sel}, scarto "
                + ", ".join(s.sel for s in ss if s is not hit[0]))
        else:
            amb[key] = sorted({s.sel for s in ss})
    if amb:
        raise ValueError(f"[{ds}] più versioni per lo stesso (budget, scope, metodo): "
                         + "; ".join(f"{k[2]}/{k[3]}/{k[1]} in lists/{'' if k[0] == 'retrain' else k[0] + '/'}: {v}"
                                     for k, v in amb.items())
                         + "  -> scegli con --sel-tag (es. --sel-tag new_version, oppure --sel-tag '' per i file "
                           "senza suffisso)")
    return out


def want_method(meth, a):
    if not a.segment_mabs and any(meth == p or meth.startswith(p + "_") for p in SEGMENT_MABS):
        return False
    return any(fnmatch.fnmatch(meth, p) for p in a.methods)


# ── caricamento: stesse regole di targeting_eval.load_dataset ────────────────────────────────────
def head_items(pop, pct):
    items = np.arange(1, len(pop))
    order = items[np.argsort(-pop[items], kind="stable")]
    head = np.zeros(len(pop), bool)
    head[order[:int(np.ceil(pct / 100.0 * len(items)))]] = True
    return head


def head_items_threshold(pop, quantile=None, max_pop=None):
    """Stessa regola di rev2_defects.py / GSDB: popolarità = interazioni di training dell'item (base.npz 'pop';
    = utenti distinti se il training non ha interazioni ripetute); esclusi gli item con popolarità > max_pop,
    oppure, se max_pop non è dato, > quantile `quantile` delle popolarità degli item con almeno un'interazione
    di training (come groupby sul train). Ritorna (maschera, soglia)."""
    pop = np.asarray(pop, float)
    seen = pop > 0
    seen[0] = False                                  # 0 = padding
    if max_pop is None:
        max_pop = float(np.quantile(pop[seen], quantile))
    head = seen & (pop > max_pop)
    return head, max_pop


def build_head(a, pop):
    """Item head da escludere dal pool secondo --exclude-head-pct, --head-quantile o --head-max-pop (o None)."""
    if a.exclude_head_pct:
        head = head_items(pop, a.exclude_head_pct)
        rule = f"top {a.exclude_head_pct:g}% item più popolari"
    elif a.head_quantile is not None or a.head_max_pop is not None:
        head, thr = head_items_threshold(pop, a.head_quantile, a.head_max_pop)
        rule = f"item con popolarità > {thr:g}" + (f" (quantile {a.head_quantile:g})" if a.head_max_pop is None else "")
    else:
        return None
    seen = np.asarray(pop) > 0
    seen[0] = False
    log(f"  head: {rule}: {int(head.sum()):,} item su {int(seen.sum()):,} ({100 * head.sum() / max(seen.sum(), 1):.1f}%), "
        f"che coprono {100 * pop[head].sum() / max(pop.sum(), 1):.1f}% delle interazioni di training")
    return head


def load_candidates(path):
    """lists/candidates.npz: sorgente (normalizzata come gli scope: gsdb_ff -> gsdb_f, ...), item e utenti di
    ogni difetto rilevato. item_src = sorgente di ogni voce di items (stessa lunghezza di items).
    users = utenti (ID del modello base) di tutti i difetti, concatenati (None nei dump vecchi)."""
    z = np.load(path)
    src = np.array([SCOPE_ALIASES.get(x, x) for x in z["src"].astype(str)])
    ptr = z["item_ptr"].astype(np.int64)
    users = z["users"].astype(np.int64) if "users" in z.files else None
    return NS(src=src, ptr=ptr, items=z["items"].astype(np.int64), item_src=np.repeat(src, np.diff(ptr)),
              users=users)


def _cand_items(cand, idx):
    """Item (ID del modello base) dei difetti candidati idx, concatenati."""
    if len(idx) == 0:
        return np.zeros(0, np.int64)
    return np.concatenate([cand.items[cand.ptr[c]:cand.ptr[c + 1]] for c in idx]).astype(np.int64)


def nohead_mask(s, head, sources, cand, n_items):
    """(--target-pool selected) Item head da togliere dai target della selezione s.
    Senza sorgenti (--nohead-sources non dato): tutti gli item head.
    Con sorgenti: scope di una sola sorgente -> tutti gli head se la sorgente è tra quelle indicate, altrimenti
    nessuno; scope 'all' -> solo gli head che stanno negli item dei difetti scelti delle sorgenti indicate e in
    NESSUN difetto scelto di un'altra sorgente (un item head segnalato anche da GSRank o GSDB resta)."""
    if not sources:
        return head
    if s.scope != "all":
        return head if s.scope in sources else np.zeros(n_items, bool)
    if cand is None or s.chosen is None or len(s.chosen) == 0:
        log(f"   ATTENZIONE {s.sel}: difetti scelti o candidates.npz mancanti, nohead per sorgente non "
            "applicabile allo scope all (nessun item escluso; rilancia targeting_eval dump)")
        return np.zeros(n_items, bool)
    ch = np.asarray(s.chosen, np.int64)
    from_src = np.isin(cand.src[ch], list(sources))
    inn = np.zeros(n_items, bool)
    inn[_cand_items(cand, ch[from_src])] = True
    out = np.zeros(n_items, bool)
    out[_cand_items(cand, ch[~from_src])] = True
    return head & inn & ~out


def detected_pool(scope, cand, n_items, head=None, sources=()):
    """(--target-pool detected) Item di TUTTI i difetti rilevati per lo scope: scope di una sorgente -> tutti i
    difetti di quella sorgente; 'all' -> tutti i difetti di tutte le sorgenti. Anche quelli che nessun metodo
    ha scelto. Nohead come nohead_mask, ma sui difetti rilevati invece che su quelli scelti.
    Ritorna (maschera degli item, numero di item head tolti)."""
    isrc = cand.item_src
    sel = np.ones(len(cand.items), bool) if scope == "all" else isrc == scope
    if not sel.any():
        raise ValueError(f"scope {scope!r}: nessun difetto con item in candidates.npz "
                         f"(sorgenti presenti: {sorted(set(cand.src.tolist()))})")
    pool = np.zeros(n_items, bool)
    pool[cand.items[sel]] = True
    pool[0] = False                                     # 0 = padding
    ex = np.zeros(n_items, bool)
    if head is not None:
        if not sources:
            ex = head.copy()
        elif scope != "all":
            ex = head.copy() if scope in sources else ex
        else:                                           # head segnalati SOLO dalle sorgenti indicate
            fs = np.isin(isrc, list(sources))
            inn = np.zeros(n_items, bool)
            inn[cand.items[fs]] = True
            out = np.zeros(n_items, bool)
            out[cand.items[~fs]] = True
            ex = head & inn & ~out
        ex &= pool
        pool &= ~ex
    return pool, int(ex.sum())


def load(ds, a):
    L = a.out / ds / "lists"
    B = np.load(L / "base.npz")
    sels = []
    for v in a.load_variants:
        vdir = L if v == "retrain" else L / v
        if not vdir.is_dir():
            continue
        for f in sorted(vdir.glob("*.npz")):
            if f.name in ("base.npz", "candidates.npz") or f.name.endswith(".tmp.npz"):
                continue
            z = np.load(f)
            sel = str(z["sel"])
            meth, scope, budget, tag = parse_sel(sel)       # dal nome, non da method/scope salvati dal dump
            if not want_method(meth, a):
                continue
            sels.append(NS(variant=v, sel=sel, method=meth, scope=scope, budget=budget, tag=tag,
                           users=z["users"], top=z["top"], hits=z["hits"], npos=z["npos"],
                           targets=z["targets"], members=z["members"], n_edges=int(z["n_edges"]),
                           chosen=z["chosen"] if "chosen" in z.files else None))
    sels = pick_versions(sels, a.sel_tag, ds)
    found = sorted({s.method for s in sels}, key=method_key)
    log(f"[{ds}] metodi nelle liste: {found}")
    for pat in (x for x in a.methods if x.startswith(("exposure", "random"))):   # baseline aggiunte di recente
        if not any(fnmatch.fnmatch(m, pat) for m in found):
            log(f"[{ds}] ATTENZIONE: nessuna lista per '{pat}' in {L}: il dump di targeting_eval filtra i metodi "
                f"con il suo --methods (default senza exposure e random). Rilancia il dump con --methods ... "
                f"'{pat}'")
    if not any(s.variant == a.variant for s in sels):
        raise FileNotFoundError(f"nessuna lista '{a.variant}' in {L}")
    n_users, n_items = int(B["n_users"]), int(B["n_items"])
    head = build_head(a, B["pop"])
    sources = {SCOPE_ALIASES.get(x, x) for x in (a.nohead_sources or [])}
    cp = L / "candidates.npz"
    cand = load_candidates(cp) if cp.exists() else None
    if a.target_pool == "detected":
        if cand is None:
            raise FileNotFoundError(f"{cp} non trovato: serve per il pool di tutti i difetti rilevati "
                                    "(rilancia targeting_eval dump --task 0, oppure usa --target-pool selected)")
        srcs, cnt = np.unique(cand.src, return_counts=True)
        log(f"  pool: tutti i difetti rilevati in candidates.npz ("
            + ", ".join(f"{s}={n}" for s, n in zip(srcs, cnt)) + ")")
    else:
        log("  pool: unione dei difetti scelti dai metodi caricati (--target-pool selected)")
        if head is not None and sources and cand is None:
            log(f"  ATTENZIONE: {cp} non trovato: nello scope all il nohead per sorgente non si può applicare")
    if head is not None and sources:
        log(f"  nohead: item head esclusi solo se segnalati da "
            f"{', '.join(sorted(sources))} (e da nessun'altra sorgente nello scope all)")
    users = B["users"]
    keep = np.ones(len(users), bool)
    for s in sels:                                   # stessi utenti per tutte le varianti caricate
        pos = np.full(n_users, -1, np.int64)
        pos[s.users] = np.arange(len(s.users))
        s.rows = pos[users]
        keep &= s.rows >= 0
    users = users[keep]
    members = np.zeros(n_users, bool)
    tg = {}
    for s in sels:
        r = s.rows[keep]
        s.top, s.hits, s.npos = s.top[r], s.hits[r], s.npos[r]
        members[s.members] = True                    # utenti dei difetti rimossi (sempre esclusi)
    n_removed_mem = int(members[users].sum())
    if a.exclude_members == "candidates":            # + utenti di TUTTI i difetti candidati
        if cand is None or cand.users is None:
            raise FileNotFoundError(f"{cp}: servono gli utenti dei difetti candidati per --exclude-members "
                                    "candidates (rilancia targeting_eval dump --task 0 --overwrite, oppure "
                                    "--exclude-members removed)")
        cu = cand.users[(cand.users >= 0) & (cand.users < n_users)]
        members[cu] = True
        log(f"  membri esclusi: utenti di tutti i {len(cand.src)} difetti candidati "
            f"({int(members[users].sum())} utenti di test; {n_removed_mem} erano già membri dei difetti rimossi)")
    else:
        log(f"  membri esclusi: utenti dei difetti rimossi ({n_removed_mem} utenti di test)")
    for s in sels:
        if a.target_pool != "selected":
            continue
        s.tmask = np.zeros(n_items, bool)
        s.tmask[s.targets] = True
        if head is not None:
            ex = nohead_mask(s, head, sources, cand, n_items) & s.tmask
            if sources and s.variant == a.variant:
                log(f"   nohead {s.sel:40s} item tolti {int(ex.sum()):6d} su {int(s.tmask.sum()):6d}")
            s.tmask &= ~ex
        tg.setdefault((s.budget, s.scope), np.zeros(n_items, bool))
        tg[(s.budget, s.scope)] |= s.tmask
    if a.target_pool == "detected":                  # stesso pool per ogni budget dello scope
        pools = {}
        for s in sels:
            if s.scope not in pools:
                pools[s.scope], n_ex = detected_pool(s.scope, cand, n_items, head, sources)
                if head is not None:
                    log(f"   nohead {s.scope:8s} item head tolti dal pool {n_ex:6d}")
            tg[(s.budget, s.scope)] = pools[s.scope]
    base = NS(top=B["top"][keep], hits=B["hits"][keep], npos=B["npos"][keep])
    nonmem = ~members[users]
    seg = {"all": nonmem}
    for g in ("active", "lt"):
        m = np.zeros(n_users, bool)
        m[B[f"seg_{g}"]] = True
        seg[g] = m[users] & nonmem
    groups = {}
    for key, tu in tg.items():
        aff = nonmem & tu[base.top[:, :a.affected_k]].any(1)
        groups[key] = NS(t_union=tu, aff={g: seg[g] & aff for g in SEGMENTS})
    sels = [s for s in sels if s.variant == a.variant]
    log(f"[{ds}] selezioni: " + ", ".join(f"{s.sel} -> {s.method}/{s.scope}/{s.budget}" for s in sels))
    if a.one_baseline:                     # solo MAB-All e la baseline dello scope (all -> stratified, altri -> heuristic)
        sels = [s for s in sels if s.method == "mab_all" or s.method == a.baseline.get(
            s.scope, "stratified" if s.scope == "all" else "heuristic")
            or any(fnmatch.fnmatch(s.method, p_) for p_ in a.keep_also)]
    sels = sorted(sels,
                  key=lambda s: (s.budget, scope_key(s.scope), method_key(s.method)))
    log(f"[{ds}] {len(users)} utenti, {int(members[users].sum())} membri esclusi, "
        f"{len(sels)} selezioni ({a.variant})")
    for (b, sc), gr in sorted(groups.items(), key=lambda x: (x[0][0], scope_key(x[0][1]))):
        log(f"   {b} {sc:8s} item segnalati {int(gr.t_union.sum()):6d}  affetti "
            + "  ".join(f"{g}={int(gr.aff[g].sum())}" for g in SEGMENTS))
    return NS(ds=ds, users=users, base=base, sels=sels, groups=groups, n_items=n_items, lists_dir=L,
              seg=seg, pop=B["pop"], K=base.top.shape[1])


# ── metriche per utente (stesse formule di targeting_eval) ───────────────────────────────────────
def disc(k):
    return 1.0 / np.log2(np.arange(2, k + 2))


def per_user(top, hits, npos, tmask, k):
    """NDCG@k, la sua parte dovuta agli item segnalati, T-NDCG@k (pool), T-DCG@k e conteggi."""
    d = disc(k)
    cum = np.cumsum(d)
    h = hits[:, :k].astype(float)
    T = tmask[top[:, :k]]
    idcg = cum[np.clip(npos, 1, k).astype(int) - 1]
    n_t = int(tmask.sum())
    tdcg = (T * d).sum(1)
    return dict(ndcg=(h * d).sum(1) / idcg, ndcg_flag=(h * T * d).sum(1) / idcg,
                tndcg=tdcg / cum[max(min(k, n_t), 1) - 1], tdcg=tdcg,
                nflag=T.sum(1), hflag=(h * T).sum(1), hother=(h * ~T).sum(1))


def user_js(top0, top1, k, n_items):
    """Per utente: Jensen-Shannon (base 2, in [0, 1]) tra la distribuzione dell'esposizione di posizione
    sugli item della lista prima e quella dopo. 0 = stessa lista nello stesso ordine."""
    n = len(top0)
    if n == 0:
        return np.zeros(0)
    w = disc(k) / disc(k).sum()
    r = np.repeat(np.arange(n, dtype=np.int64), k)
    key = np.concatenate([r * n_items + top0[:, :k].ravel(), r * n_items + top1[:, :k].ravel()])
    p = np.concatenate([np.tile(w, n), np.zeros(n * k)])
    q = np.concatenate([np.zeros(n * k), np.tile(w, n)])
    u, inv = np.unique(key, return_inverse=True)
    P, Q = np.bincount(inv, p, len(u)), np.bincount(inv, q, len(u))
    M = (P + Q) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        c = np.where(P > 0, P * np.log2(P / M), 0) + np.where(Q > 0, Q * np.log2(Q / M), 0)
    return np.bincount(u // n_items, 0.5 * c, n)


def exposure(top, k, n_items, mask=None):
    t = top[:, :k]
    e = np.bincount(t.ravel(), weights=np.tile(disc(k), len(t)), minlength=n_items).astype(float)
    if mask is not None:
        e = e * mask
    return e


def kl(p, q, eps=1e-9):
    """KL(p || q) in bit; q lisciata con eps (KL è infinita se q = 0 dove p > 0)."""
    p = p / p.sum()
    q = (q + eps) / (q + eps).sum()
    m = p > 0
    return float((p[m] * np.log2(p[m] / q[m])).sum())


def js(p, q):
    p, q = p / p.sum(), q / q.sum()
    m = (p + q) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(0.5 * np.nansum(np.where(p > 0, p * np.log2(p / m), 0))
                     + 0.5 * np.nansum(np.where(q > 0, q * np.log2(q / m), 0)))


def dist_divergence(before, after, n_bins=30):
    """KL(dopo || prima) e JS (base 2) tra le distribuzioni PER UTENTE di una metrica (NDCG o T-NDCG).
    Istogrammi con gli stessi bin per prima e dopo: un bin a parte per il valore esattamente 0 (utente senza
    hit / senza item difettosi in lista), poi n_bins bin uguali fino al 99.5° percentile (oltre: ultimo bin).
    Lisciatura di Laplace (+0.5 per bin) perché la KL sia finita. Non tiene conto dell'abbinamento degli
    utenti né del verso (dice QUANTO cambia la distribuzione, non se migliora): il verso lo danno le medie."""
    x = np.concatenate([before, after])
    hi = float(np.percentile(x[x > 0], 99.5)) if (x > 0).any() else 1.0
    edges = np.concatenate([[-np.inf, 1e-12], np.linspace(1e-12, hi, n_bins + 1)[1:-1], [np.inf]])
    p = np.histogram(after, edges)[0] + 0.5
    q = np.histogram(before, edges)[0] + 0.5
    p, q = p / p.sum(), q / q.sum()
    m = (p + q) / 2
    return (float((p * np.log2(p / q)).sum()),
            float(0.5 * (p * np.log2(p / m)).sum() + 0.5 * (q * np.log2(q / m)).sum()))


def boot(X, n_boot, rng):
    """Medie bootstrap delle colonne di X (utenti × misure), stesso ricampionamento per tutte."""
    n = len(X)
    out = np.empty((n_boot, X.shape[1]))
    for b in range(n_boot):
        out[b] = X[rng.integers(0, n, n)].mean(0)
    return out


def ci(v):
    lo, hi = np.nanpercentile(v, [2.5, 97.5])
    return float(lo), float(hi)


USET_TXT = {"aff": "affected users", "seg": "all users (defect members excluded)"}
USET_SUFFIX = {"aff": "", "seg": "_seg"}
