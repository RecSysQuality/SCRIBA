#!/usr/bin/env python3
"""
paper_figures.py
================
Figure e tabelle per il paper sul trade-off accuratezza / esposizione agli item difettosi.
Legge le liste salvate da `targeting_eval.py dump` (<out>/<ds>/lists/) e ricostruisce gli stessi
utenti, membri esclusi e utenti affetti di targeting_eval. Il pool degli item difettosi dello scope
(T-NDCG, T-HR, T-Rank, utenti affetti) è di default quello di TUTTI i difetti rilevati dai detector
(vedi POOL DEI DIFETTI). Nessun modello, solo numpy/scipy/matplotlib. Per le tabelle tables_abs importa
le funzioni di targeting_eval.py (deve stare nella stessa cartella): stesse formule e stessi test.

Tutte le variazioni sono in VALORE ASSOLUTO (NDCG e T-NDCG sono entrambi DCG normalizzati, stessa
scala [0, 1]); le variazioni relative restano nelle tabelle tra parentesi.

POOL DEI DIFETTI (item "difettosi" per T-NDCG, T-HR, T-Rank, utenti affetti, item segnalati)
  --target-pool detected  (default) item di TUTTI i difetti trovati dai detector, letti da lists/candidates.npz
                          (tutte le righe di <ds>.<sorgente>.csv; item = item con almeno un arco di training
                          degli utenti del difetto). Scope group_f / gsdb_f / rev2: tutti i difetti di quella
                          sorgente; scope all: tutti i difetti di tutte le sorgenti. Comprende i difetti che
                          nessun metodo ha scelto. Stesso pool per tutti i metodi e tutti i budget dello scope.
  --target-pool selected  come prima: unione degli item dei difetti scelti dai metodi caricati (cartelle
                          di output con suffisso _selpool).
  MEMBRI ESCLUSI (--exclude-members, vale per affetti, segmenti interi e tabelle)
  --exclude-members candidates  (default) utenti di TUTTI i difetti candidati (lists/candidates.npz, tutte le
                          sorgenti del dump), più quelli dei difetti rimossi: stessi utenti per ogni metodo, scope e
                          budget. Cartelle con suffisso _candmem.
  --exclude-members removed  come prima: solo gli utenti dei difetti RIMOSSI da qualche metodo.

METODI
  di default anche random, exposure ed exposure_pe (Exp: esposizione per arco). Con --one-baseline restano
  SCRIBA, la baseline dello scope e i metodi di --keep-also (default exposure_pe random).
  Le liste devono esserci: il dump di targeting_eval filtra con il suo --methods, che di default NON
  comprende exposure e random -> dump con --methods stratified heuristic 'regressor*' 'mab*' random 'exposure*'.

FIGURE (per budget e k; una colonna per scope)
  fig_tradeoff     piano del trade-off: x = ΔNDCG@k (dopo - prima), y = T-NDCG@k tolto (prima - dopo),
                   media sugli utenti affetti (All) con IC bootstrap 95%. Banda grigia = margine TOST
                   (accuratezza equivalente al Base). Il riquadro dei t-test SCRIBA vs baseline sta SOTTO
                   ogni pannello (sotto l'asse x), così non copre le frecce.
                   Con --tradeoff-usersets aff seg (default) anche fig_tradeoff_seg_*, fig_tradeoff_grid_seg_* e
                   fig_tradeoff_pairs_seg_*: stesso piano su TUTTI gli utenti (membri dei difetti esclusi).
                   tradeoff.csv ha la colonna userset (aff / seg).
  fig_decomp       perché Rev2 è diverso. Sinistra: ΔNDCG diviso in (a) rilevanza persa sugli item
                   SEGNALATI come difettosi (hit di test su item del pool) e (b) resto della lista.
                   Destra: precisione (hit di test) delle posizioni occupate da item segnalati vs dalle
                   altre, nelle liste del Base. Se gli item segnalati sono rilevanti quanto gli altri, le
                   etichette della sorgente contengono item buoni: la perdita di NDCG è un costo delle
                   etichette, non del metodo.
  fig_users        utenti aggregati per numero di item difettosi nella loro top-k del Base (1, 2, 3, 4, >=5):
                   T-NDCG tolto e ΔNDCG per gruppo. Mostra dove si concentra il guadagno.
  fig_dist         distribuzioni per utente (non la media): ECDF di ΔNDCG e di JS(lista prima || lista dopo).
  fig_fairness     Active vs Long-tail: riduzione relativa di T-NDCG@k (Σ (T-NDCG prima - dopo) / Σ T-NDCG
                   prima; T-NDCG = NDCG degli item target, metrica standard della letteratura sul poisoning)
                   e variazione relativa di NDCG per segmento, IC bootstrap 95% sugli utenti affetti.
                   Nessun rapporto tra segmenti in figura: il confronto è tra le barre affiancate.
  fig_kl           KL "non sulla media": x = KL(distribuzione per utente di NDCG dopo || prima),
                   y = KL(distribuzione per utente di T-NDCG dopo || prima). Alto e a sinistra = cambia
                   l'esposizione difettosa, non l'accuratezza. Anche in griglia e in tables_kl_k<k>.tex.
  fig_tradeoff_pairs  (con più dataset e --one-baseline) versione COMPATTA: un pannello per dataset, per ogni
                   scope una freccia dalla baseline della sorgente (vuoto) a SCRIBA (pieno).
  tables_kl_ndcg_k<k>.tex  solo KL della distribuzione di NDCG (5 decimali), righe = dataset x scope, colonne =
                   Source, SCRIBA, rumore. tables_kl_k<k>.tex: KL di NDCG e di T-NDCG per metodo
  tables_shares_k<k>.tex  PER IL PAPER: quota di utenti affetti a cui scende l'esposizione difettosa e a cui
                   scende l'NDCG, sorgente vs SCRIBA (con più dataset: tabella unica nella cartella comune)
  fig_sources      scope All: quanti difetti di GSRank, GSDB e Rev2 sceglie ogni metodo (e quota degli archi);
                   con più dataset anche fig_sources_grid (un pannello per dataset). CSV: sources_all.csv
  fig_pareto       (se esiste analysis_<tag>/curves.csv) traiettoria del trade-off al crescere degli archi
                   rimossi: x = ΔNDCG, y = T-NDCG tolto, un punto per prefisso. Variante prop. NB: le curve sono
                   calcolate da targeting_eval sull'unione dei difetti scelti, non sul pool dei detector.

TABELLE DEI VALORI ASSOLUTI (come targeting_eval, ma con gli utenti, il pool e il nohead di questo script)
  tables_abs_k<k>_<us>.tex  per ogni k di --k e us in --abs-usersets (default aff e seg): valori ASSOLUTI
                   (Base + metodi) per blocchi All / Active / Long-tail: NDCG, HR | Coverage, Novelty, Gini |
                   T-NDCG, T-HR, T-Rank sul POOL degli item difettosi dello scope. dagger = metodo vs Base,
                   * = SCRIBA vs metodo (t-test appaiati, Holm), approx = TOST equivalente a Base (solo aff).
                   Stessa funzione (latex_abs_tables) e stessi test di targeting_eval; i metodi sono quelli di
                   questo script (con --one-baseline: SCRIBA e la baseline dello scope, Holm su questi).
                   CSV: abs_metrics.csv (prima/dopo per metrica), abs_ttests.csv, abs_tost.csv
  --no-abs-tables  per saltarle.

NOHEAD
  --exclude-head-pct P            toglie dal pool i P% di item più popolari nel training (tutti gli scope)
  --head-quantile Q               toglie dal pool gli item con popolarità di training > quantile Q delle popolarità
                                  (es. 0.80: stessa regola con cui GSDB e rev2_defects.py escludono gli item head;
                                  applicata a tutte le sorgenti allinea GSRank alle altre due). Tag _headq<100Q>
  --head-max-pop N                come sopra con soglia fissa: più di N interazioni di training. Tag _headmax<N>
  (una sola delle tre regole; popolarità = interazioni di training in base.npz)
  --exclude-head-pct P --nohead-sources rev2
                                  toglie solo gli item head che vengono da difetti di Rev2: nello scope rev2 tutti i
                                  suoi item head; nello scope all solo gli item head dei difetti Rev2 che NON sono
                                  segnalati anche da un difetto di un'altra sorgente (quelli restano); group_f e
                                  gsdb_f invariati. Con --target-pool detected si guardano tutti i difetti rilevati,
                                  con selected solo quelli scelti. Serve lists/candidates.npz.
  L'esclusione cambia il pool degli item difettosi dello scope e quindi gli utenti affetti; le liste "dopo"
  non cambiano (gli archi restano rimossi): è un filtro sulla misura.

CSV / LaTeX
  tradeoff.csv     per (k, budget, scope, metodo, segmento): medie, IC, scomposizione, quote di utenti che
                   perdono / guadagnano NDCG, efficienza
  divergence.csv   KL e Jensen-Shannon (base 2) "sulle distribuzioni, non sulla media":
                     exp_*   distribuzione dell'esposizione sugli item (Σ_u 1/log2(1+pos)) prima vs dopo, tutti
                             gli item e solo gli item NON segnalati (il resto della lista cambia?)
                     ndcg_*  distribuzione per utente di NDCG@k prima vs dopo (JS su istogramma, KS, Wasserstein)
                     user_js distribuzione della JS per utente tra la lista prima e la lista dopo
  label_check.csv  precisione delle posizioni segnalate vs altre nel Base, per scope
  fairness.csv     tndcg_red_active, tndcg_red_lt (riduzione relativa di T-NDCG@k), ΔNDCG relativo per segmento,
                   gap UGF prima/dopo
  tables_tradeoff_k<k>.tex

Uso
  python paper_figures.py --datasets yelp --variant retrain --k 20
  python paper_figures.py --datasets yelp Books --variant prop --k 10 20 --exclude-head-pct 5
  python paper_figures.py --datasets Books yelp <terzo> --k 10 20 --one-baseline   (per il paper)
  python paper_figures.py --datasets Books yelp <terzo> --k 10 20 --one-baseline \\
                          --exclude-head-pct 5 --nohead-sources rev2
  python paper_figures.py --datasets yelp --k 20 --target-pool selected   (vecchio pool: difetti scelti)
Con più dataset: anche <out>/paper_<tag>_<variant>/fig_tradeoff_grid_<budget>_k<k>.pdf, UNA figura con
righe = dataset e colonne = scope (GSRank, GSDB, Rev2, All).
Output in <out>/<ds>/final_paper_<tag>_<variant>/ (tag come targeting_eval:
affk<affected-k>[_nohead<P>[_<sorgenti>]]; con --target-pool selected si aggiunge _selpool).
"""
from __future__ import annotations

import argparse
import csv
import fnmatch
import re
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np

SEGMENTS = ("all", "active", "lt")
SEG_LABEL = {"all": "All", "active": "Active", "lt": "Long-tail"}
SCOPE_ORDER = ("group_ff", "group_f","gsdb_ff", "rev2_fff", "all")
SCOPE_LABEL = {"group_ff": "GSRank","group_f": "GSRank0", "rev2_fff": "Rev2", "all": "All", "gsdb_ff": "GSDB"}
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


# ── analisi ──────────────────────────────────────────────────────────────────────────────────────
USET_TXT = {"aff": "affected users", "seg": "all users (defect members excluded)"}
USET_SUFFIX = {"aff": "", "seg": "_seg"}


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
    """Importa targeting_eval.py dalla cartella di questo script."""
    import importlib
    import sys
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    return importlib.import_module("targeting_eval_head")


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
    te = _te()
    te.set_segment_mabs(a.segment_mabs)
    rows_long, rows_tt, rows_test = abs_rows(D, a, ks, te)
    write_csv(outdir / "abs_metrics.csv", rows_long)
    write_csv(outdir / "abs_ttests.csv", rows_tt)
    write_csv(outdir / "abs_tost.csv", rows_test)
    old = te.method_label
    te.method_label = lambda m: method_label(m)              # SCRIBA; con --one-baseline la baseline è "Source"
    old_sl = dict(te.SCOPE_LABEL)
    te.SCOPE_LABEL.update(SCOPE_LABEL)                       # nomi degli scope di questo script (GSRank, ...)
    try:
        for k in ks:
            for us in a.abs_usersets:
                txt = te.latex_abs_tables(rows_long, rows_tt, rows_test, k, us, vlabel + nohead_note(a), a.alpha, D.K)
                txt = txt.replace("MAB-All", METHOD_LABEL["mab_all"])
                if TARGET_POOL == "detected":                # didascalia: pool di tutti i difetti rilevati
                    txt = txt.replace("the union of the defective items removed in the scope "
                                      "(same item set for every method)", POOL_TXT["detected"])
                if SOURCE_BASELINE:
                    txt = txt.replace("Base = before removal;", "Base = before removal; Source = the defects ranked "
                                      "by the source itself;")
                if a.exclude_members == "candidates":
                    txt = txt.replace("defect members excluded", "users of any candidate defect excluded")
                (outdir / f"tables_abs_k{k}_{us}.tex").write_text(txt)
    finally:
        te.method_label = old
        te.SCOPE_LABEL.clear()
        te.SCOPE_LABEL.update(old_sl)
    log(f"  tables_abs: k={list(ks)}, utenti={list(a.abs_usersets)} -> {outdir}")


# ── composizione per sorgente dei difetti scelti nello scope "all" ──────────────────────────────
SOURCE_ORDER = ("group_f", "gsdb_f", "rev2")


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
                     f"propagation\nT-NDCG on {POOL_TXT[TARGET_POOL]}; each marker = one prefix of the removal list; "
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
                f"rest of the list; T-NDCG removed = before $-$ after on {POOL_TXT[TARGET_POOL]}; "
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


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], epilog=__doc__,
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
    global SOURCE_BASELINE, TARGET_POOL
    SOURCE_BASELINE = a.one_baseline
    TARGET_POOL = a.target_pool
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