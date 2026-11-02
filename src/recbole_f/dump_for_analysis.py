#!/usr/bin/env python3
"""
targeting_eval.py
=================
Effetto della rimozione dei difetti sulle liste degli utenti AFFETTI, per segmento
(all, active, lt), sempre sullo STESSO insieme di utenti per tutti i metodi e tutti
gli scope (group_f, rev2, all). Due fasi:

  dump      (GPU) per ogni dataset: top-K del modello base e, per ogni file di rimozione,
            le top-K "dopo" in due varianti (--variants, default entrambe):
              retrain  modello riallenato sul dataset ripulito (checkpoint di
                       run_removed_eval.py; con --no-train i file senza checkpoint si saltano)
              prop     embedding del modello base propagati sul grafo senza gli archi
                       rimossi (come la variante 'prop' di loo_defects.py): nessun training,
                       isola l'effetto strutturale della rimozione dal rumore del retrain.
            Output: <out>/<ds>/lists/base.npz, candidates.npz, <sel>.npz (retrain),
                    prop/<sel>.npz
  analyze   (CPU) tutte le metriche dai file salvati.

UTENTI
  membri  = unione degli utenti elencati in TUTTI i difetti rimossi di TUTTI i file
            -> esclusi sempre (il loro training cambia direttamente); così il segmento
            intero è identico per tutti gli scope.
  segmenti all / active / lt (file rec_test_<seg>), membri esclusi.
  affetti = PER SCOPE E BUDGET: non membri con almeno un item rimosso da un metodo di quello
            scope e budget (unione su metodi e varianti) nella top-`--affected-k` (default 20)
            del modello BASE. Es. yelp, rev2, lt: utenti lt esposti ai difetti rimossi in rev2.
            Stessi utenti per tutti i metodi (e le varianti) dello stesso scope e budget:
            i metodi si confrontano dentro lo scope.
            Userset 'aff' = affetti del segmento, 'seg' = intero segmento,
            'unaff' = non affetti (rumore di fondo nell'analisi di sostituzione).
  Le metriche "_u" e la FoE 'def' usano l'unione degli item rimossi nello scope e budget.

METRICHE (prima = modello base, dopo = retrain o prop, stessi utenti)
  accuratezza     NDCG@k, Recall@k, Hit@k            (formule di run_removed_eval)
  beyond          Coverage@k, Novelty@k, Entropy@k, Gini@k, APLT@k
  item difettosi  T-HR@k   quota di utenti con almeno un item target in top-k
                  T-NDCG@k NDCG con gli item target come "rilevanti"
                           (IDCG su min(k, |target|))
                  T-Prec@k quota di posizioni occupate da item target
                  Target = item toccati dagli archi rimossi DA QUEL METODO (proprio);
                  con suffisso _u: unione degli item rimossi dai metodi dello stesso scope
                  e budget (stesso "righello" per tutti, controllo).
                  Suffissi _niche / _head (e _u_niche / _u_head): stessi indici ristretti agli
                  item target long-tail / head per popolarità di training (--head-share).
                  Stessi esperimenti, nessun retraining: separano l'effetto sugli item
                  difettosi di nicchia da quello sugli item difettosi popolari.
                  Riferimenti: Mobasher et al. 2007 (TOIT); Fang et al. 2018 (ACSAC);
                  Tang et al. 2020 (NeurIPS); Huang et al. 2021 (NDSS).
  rank medio      T-Rank@k (t_rank_u): per utente, rank medio degli item difettosi dell'UNIONE
                  dello scope che stavano nella top-k del base; prima = il loro rank nel base,
                  dopo = il loro rank nella lista dopo la rimozione. Letto dalla top-K salvata:
                  item uscito = K+1 (censura: il rank dopo è un limite inferiore). Più alto = item
                  spinto più giù. NaN (escluso dalle medie) per gli utenti senza item difettosi
                  nella top-k del base. Rank shift = T-Rank dopo - T-Rank prima.
  FoE             Fairness of Exposure (Singh & Joachims, KDD 2018). Esposizione di
                  posizione v_j = 1/log2(1+j); Exp(G) = media sugli item di G
                  dell'esposizione media per utente; merito U(G) = media sugli item di G
                  delle interazioni di test degli utenti valutati.
                    DP  = Exp(G)/Exp(resto)                              (demographic parity)
                    DTR = [Exp(G)/U(G)] / [Exp(resto)/U(resto)]          (disparate treatment)
                    exp_share = quota dell'esposizione totale che va a G
                  Gruppi di item: 'own' = item rimossi DAL METODO (misura principale);
                  'def' = unione degli item rimossi nello scope e budget (controllo);
                  'tail' = long-tail per popolarità di training (tutto tranne il top --head-share).
                  DTR_def > 1: gli item difettosi ricevono più esposizione del loro merito.

SELEZIONE (prima della rimozione, solo modello base). Due domande SEPARATE, per segmento g:
  (1) REACH     i difetti scelti dal MAB del segmento raggiungono più utenti di g?
  (2) POSIZIONE gli item difettosi stanno più in alto nelle liste di g?
  Livello DIFETTO, su tutti i candidati dello scope (file <ds>.<source>.csv; item = quelli con
  almeno un arco dei suoi utenti nel training; membri del difetto esclusi):
    reach_g(c) = quota di utenti di g con almeno un item di c in top-K
    pos_g(c)   = rank medio del PRIMO item di c tra gli utenti raggiunti (1 = in cima;
                 solo se raggiunge almeno --min-reach-users utenti)
    Classifica di tutti i candidati per ciascuna misura -> percentile dei difetti scelti
    (100 = più utenti / più in alto), quota nel top-10%, Cliff's delta (Holm) di mab_lt
    (classifiche lt) e mab_active (classifiche active) contro ogni altro metodo.
  Livello ITEM, per gli item rimossi dal metodo (G_m):
    item_reach_g      T-HR@K su g: quota di utenti con almeno un item di G_m in top-K
    item_bestrank_g   posizione media del primo item di G_m tra gli utenti raggiunti (IC)
    item_top10pos_g   quota di utenti raggiunti in cui è entro la posizione 10
    curva: quota di utenti raggiunti con best rank <= k, k = 1..K
  FAIRNESS: disparità di esposizione (Singh & Joachims 2018) Exp_lt(G_m)/Exp_active(G_m),
  esposizione media per item, IC bootstrap.

TARGETING ("ogni MAB agisce nel suo segmento")
  E_u = T-DCG@K (non normalizzato) degli item target del metodo nella lista di u.
  Delta_u = E_u(prima) - E_u(dopo)   esposizione difettosa rimossa dalla lista di u.
  Per i segmenti degli affetti:
    dE_<g>    media di Delta_u sugli affetti del segmento g
    rr_<g>    riduzione relativa  sum Delta / sum E(prima)
    lift_<g>  dE_<g> / dE_all     (>1: la riduzione si concentra su g)
    TR        dE_lt / dE_active   (targeting ratio: alto = lt, basso = active)
    pre_lift_<g>  stessa cosa per E(prima): dove pesano i difetti scelti (senza retrain)
    dlogTR_vs_base  log TR(metodo) - log TR(baseline dello scope), stessi utenti
  IC al 95% con bootstrap sugli utenti affetti (stesso ricampionamento per tutti i
  metodi: confronto appaiato).
  Atteso: TR(mab_lt) > TR(baseline) > TR(mab_active).

CONFRONTO APPAIATO TRA METODI (stessi utenti affetti)
  Delta_u(ref) - Delta_u(altro) sugli affetti lt (ref = mab_lt) e active (ref = mab_active):
  media, IC bootstrap, Wilcoxon. Versione 'own' (item del metodo) e 'union' (stesso insieme
  di item per tutti).

UGF (Li et al., WWW 2021): NDCG@K(active) - NDCG@K(lt), prima e dopo.

TEST APPAIATI (affetti, per segmento)
  TOST di equivalenza su NDCG@K e Recall@K (prima vs dopo), margine
  --tost-margin x media prima: p < 0.05 => "l'accuratezza non cambia".
  Wilcoxon su T-NDCG@k (prima vs dopo): la riduzione è significativa.

SOSTITUZIONE (cosa esce e cosa entra nella top-K)
  churn, quota di usciti che sono item target, precisione (hit di test) di entrati e
  usciti, quota long-tail e log-popolarità di entrati e usciti. Calcolato anche sui non
  affetti: è il rumore di fondo del retrain.

Uso
---
  python targeting_eval.py dump    --datasets Books yelp --removed-dir removed_defects
  python targeting_eval.py analyze --datasets Books yelp
  python targeting_eval.py dump    --datasets Books --removed-dir removed_defects --no-train
  python targeting_eval.py dump    --datasets Books --list-tasks       (mappa dei task)
  python targeting_eval.py dump    --datasets Books --no-train --task 0      (prima: base + candidati)
  python targeting_eval.py dump    --datasets Books --no-train --task 7      (poi, in parallelo: 1..2F)
  python targeting_eval.py dump    --datasets Books --removed-dir removed_defects --sources group_f rev2
  (un dump già fatto con la versione precedente viene completato: si aggiungono solo i
   candidati e i difetti scelti, senza ricaricare i modelli)
  python targeting_eval.py analyze --datasets Books --boot 2000 --tost-margin 0.01
  python targeting_eval.py analyze --datasets Books --ks 20 --only-tables   (solo tabelle, veloce)
  python targeting_eval.py dump    --datasets Books --no-train --curves 10      (+ curve vs archi, prop)
  python targeting_eval.py analyze --datasets Books --segment-mabs   (+ MAB-Active e MAB-LT)

MAB DI SEGMENTO (--segment-mabs)
  Di default mab_active e mab_lt sono ESCLUSI ovunque (dump, metriche, tabelle, figure), come se non
  fossero stati lanciati: non entrano nemmeno nell'unione degli item difettosi dello scope, negli utenti
  affetti né nei membri esclusi. Il riferimento ("MAB del segmento": grassetto, $^*$, confronti appaiati,
  Cliff's delta) è MAB-All per tutti i segmenti. Con --segment-mabs tutto torna come prima:
  active -> mab_active, lt -> mab_lt. Nel dump cambia la numerazione dei --task (--list-tasks).

Output di analyze in <out>/<ds>/analysis_<tag>/, tag di default affk<affected-k>
(es. analysis_affk20): job con soglie diverse sulle stesse liste non si sovrascrivono.
<out>/<ds>/analysis_<tag>/<variante>/ (retrain, prop) per le metriche "dopo";
<out>/<ds>/analysis_<tag>/ per la selezione (prima della rimozione, uguale per le varianti),
targeting_all_variants.csv e fig_targeting_variants_<budget>.png (TR delle due varianti).
Utenti identici per tutte le varianti, i metodi, gli scope e i budget.
Nelle cartelle delle varianti:
  metrics_long.csv   una riga per (budget, scope, metodo, userset, segmento, metrica)
  targeting.csv      indici di targeting con IC
  selection.csv      reach, posizione, percentili, top-10%, disparità, per metodo
  selection_tests.csv  Cliff's delta + Mann-Whitney (Holm), livello difetto e item
  selection_scores.csv reach/posizione/percentili di ogni difetto scelto
  thr_curves.csv, item_rank_ecdf.csv  curve per k
  paired_methods.csv confronto appaiato tra metodi
  ugf.csv            gap di NDCG active - lt
  tests.csv          TOST e Wilcoxon
  ttests.csv         t-test appaiati: dopo vs prima (ogni metodo) e MAB del segmento vs altri
                     metodi (dopo), su affetti e segmento intero, con IC, d_z e Holm
  replacement.csv    analisi di sostituzione
  Per ogni k in --ks (k = 10 20 50 100 consigliato), us = aff (affetti) | seg (tutto il segmento):
  tables_k<k>_<us>.tex  per ogni scope:
      Tab. 1  NDCG, Recall, Hit, Coverage, Novelty, Entropy: valore DOPO (dopo - prima); riga Base;
              dagger = diverso da Base, * = diverso dal MAB del segmento (t-test appaiati, Holm)
      Tab. 2  T-NDCG, T-HR, DTR sugli item dei difetti rimossi dal metodo: valore DOPO
              (prima - dopo = esposizione tolta, positivo = meglio); dagger = dopo diverso da prima,
              * = esposizione tolta diversa da quella del MAB del segmento
  tables_setup.tex      utenti affetti per scope e segmento; archi rimossi per metodo
  tables_targeting.tex  TR e lift con IC, per ogni k
  tables_extra.tex      confronto appaiato (paired_methods) e UGF, al k più grande
  (TR, lift e confronto appaiato: di default sull'UNIONE degli item difettosi dello scope, come
   T-NDCG/T-HR in tables_abs e la heatmap _pct_union; --tr-targets own = item del singolo metodo)
  fig_heatmap_<b>_k<k>_aff_values_union.png  VALORI ASSOLUTI: T-NDCG@k sull'unione e NDCG@k, righe
                        Base + metodi, colonne All / Active / Long-tail (stessi numeri di tables_abs)
  tables_abs_k<k>_<us>.tex  PER IL PAPER: valori ASSOLUTI (Base + metodi) per blocchi All / Active /
                        Long-tail: NDCG, Recall, Hit | Coverage, Novelty, Entropy, APLT | T-NDCG, T-HR
                        sull'UNIONE degli item difettosi dello scope. dagger = metodo vs Base, * = MAB del
                        segmento vs metodo (t-test appaiati, Holm), approx = TOST equivalente a Base
  fig_heatmap_<budget>_k<k>_<us>_pct.png   PRINCIPALE: quota del danno dello scope tolta dal metodo
                        = (T-DCG prima - dopo, item del metodo) / T-DCG prima di tutti gli item difettosi
                        dello scope (stesso denominatore per tutti) + (NDCG dopo - prima)/prima; in %
  fig_heatmap_<budget>_k<k>_<us>_abs.png   stessa figura in valore assoluto (T-DCG prima - dopo; NDCG dopo - prima)
  fig_heatmap_<budget>_k<k>_aff_pct_niche.png / _head.png   solo item di nicchia / popolari
  fig_heatmap_<budget>_k<k>_aff_pct_union.png  CONTROLLO: (T-NDCG prima - dopo)/prima sull'unione degli item
  Tutte le heatmap: più alto = meglio, blu = meglio; k da --heatmap-ks (default: tutti i --ks)
  paired_methods.csv    per k: esposizione tolta dal MAB del segmento vs ogni altro metodo, IC + Wilcoxon
  fig_targeting_<budget>_k<k>.png,
  fig_selection_{reach,pos}_<budget>.png, fig_topx_{reach,pos}_<budget>.png,
  fig_item_rank_<budget>.png, fig_thr_curve_<budget>.png
Nella cartella analysis_<tag>/ (uguali per le varianti):
  overlap.csv, overlap_venn_mab.csv, tables_overlap.tex, fig_overlap_<budget>.png
                        Jaccard tra le liste rimosse (difetti scelti e item) per scope; Venn dei tre MAB
  curves.csv, fig_curves_<budget>_<scope>_k<k>.png   (servono le curve del dump: --curves N, k da --curve-k)
                        Δ T-NDCG@k (unione) e Δ NDCG@k sugli affetti in funzione degli archi rimossi:
                        prefissi dei file (ordine di rimozione; stratified a sorgenti alternate), prop
  (le figure round per round dei run del MAB sono in mab_rounds.py)
e <out>/targeting_all_datasets_<tag>_<ds>.csv
"""
from __future__ import annotations

import argparse
import os
import csv
import fnmatch
import re
import time
from pathlib import Path
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


# ============================================================================ #
#  FASE 1: DUMP (richiede recbole/torch e i moduli del progetto)
# ============================================================================ #
def _project(module_name=None):
    """Importa run_removed_eval (e i moduli che usa). Mostra l'errore VERO se l'import fallisce."""
    import importlib
    import sys
    import traceback
    here = Path(__file__).resolve().parent
    for d in (here, here.parent):                     # src/recbole_f e src/ nel path
        if str(d) not in sys.path:
            sys.path.insert(0, str(d))
    names = [module_name] if module_name else ["run_removed_eval", "run_removed_eval_1"]
    errors = []
    for name in names:
        for prefix in ("", "recbole_f."):
            try:
                rre = importlib.import_module(prefix + name)
                log(f"  modulo del progetto: {rre.__file__}")
                return rre, rre.rc, rre.lo, rre.mrd
            except ModuleNotFoundError as e:
                if e.name in (prefix + name, (prefix + name).split(".")[0]):
                    errors.append(f"{prefix + name}: file non trovato")
                    continue
                errors.append(f"{prefix + name}: il file c'è ma un suo import fallisce -> "
                              + "".join(traceback.format_exception_only(type(e), e)).strip())
            except Exception as e:
                errors.append(f"{prefix + name}: errore durante l'import -> "
                              + "".join(traceback.format_exception(type(e), e, e.__traceback__)).strip())
    cands = sorted(p.name for p in here.glob("run_removed_eval*.py"))
    raise ImportError("impossibile importare run_removed_eval.\n  cartella dello script: "
                      f"{here}\n  file run_removed_eval*.py trovati lì: {cands or 'nessuno'}\n  "
                      + "\n  ".join(errors)
                      + "\n  (se il file ha un altro nome: --rre-module <nome senza .py>)")


def _test_pairs(env, u_map, i_map):
    """Coppie (utente, item) di test negli ID del modello base (per il merito della FoE)."""
    uid_field = env.dataset.uid_field
    U, I = [], []
    for interaction, _, positive_u, positive_i in env.test:
        uids = interaction[uid_field]
        U.append(uids[positive_u].cpu().numpy())
        I.append(positive_i.cpu().numpy())
    return u_map[np.concatenate(U)], i_map[np.concatenate(I)]


def _save_res(path, res, **extra):
    """Scrittura atomica: prima un file temporaneo, poi rename (job paralleli e interruzioni)."""
    path = Path(path)
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npz")      # un temporaneo per processo
    np.savez_compressed(tmp, users=res.users.astype(np.int64), top=res.top.astype(np.int32),
                        hits=res.hits.astype(bool), npos=res.npos.astype(np.int32), **extra)
    tmp.replace(path)


def _canon_method(meth):
    return "mab_all" if meth == "mab" else meth


def _toks(x):
    return frozenset(t for t in (x or "").split(",") if t)


def dump_candidates(ds, args, rc, lo, R, ref_uid, ref_iid, path, every=50000):
    """Difetti candidati di tutte le sorgenti, negli ID del modello base.
    Item del difetto = item con almeno un arco dei suoi utenti nel training (come gli archi rimossi).
    Per ogni candidato: righe di training dei suoi utenti, poi filtro sui suoi item (costo
    proporzionale agli archi dei suoi utenti, nessuna indicizzazione per colonne)."""
    R = R.tocsr()
    R.sort_indices()
    indptr, indices = R.indptr, R.indices
    src, row, ip, it, up, us, ne, ku, ki = [], [], [0], [], [0], [], [], [], []
    for source in args.sources:
        f = rc.DATASETS[ds] / f"{ds}.{source}.csv"
        if not f.exists():
            log(f"  candidati: {f.name} non trovato, salto la sorgente {source}")
            continue
        t0, n0 = time.time(), len(src)
        with open(f, newline="", encoding="utf-8") as fh:
            for k, r in enumerate(csv.DictReader(fh, delimiter="\t")):
                u = lo.to_ids(r["users"], ref_uid)
                i = lo.to_ids(r["items"], ref_iid)
                if len(u) and len(i):
                    cols = np.concatenate([indices[indptr[x]:indptr[x + 1]] for x in u])
                    hit = cols[np.isin(cols, i, assume_unique=False)]
                    e_items, n_e = np.unique(hit), len(hit)
                else:
                    e_items, n_e = np.zeros(0, np.int64), 0
                src.append(source); row.append(k); ne.append(int(n_e))
                it.extend(e_items); ip.append(len(it)); us.extend(u); up.append(len(us))
                ku.append(",".join(sorted(_toks(r["users"])))); ki.append(",".join(sorted(_toks(r["items"]))))
                if (k + 1) % every == 0:
                    log(f"    {source}: {k + 1} candidati letti ({time.time() - t0:.0f}s)")
        log(f"  candidati {source}: {len(src) - n0} difetti in {time.time() - t0:.0f}s")
    t0 = time.time()
    np.savez_compressed(path, src=np.array(src), row=np.array(row, np.int64), item_ptr=np.array(ip, np.int64),
                        items=np.array(it, np.int64), user_ptr=np.array(up, np.int64),
                        users=np.array(us, np.int64), n_edges=np.array(ne, np.int64),
                        key_users=np.array(ku), key_items=np.array(ki))
    log(f"  candidati salvati in {time.time() - t0:.0f}s -> {path}")


def candidate_index(cand):
    """Indici per riconoscere i difetti scelti: per token (users, items) e per (source, riga)."""
    by_tok = {}
    for j, (a, b) in enumerate(zip(cand["key_users"].tolist(), cand["key_items"].tolist())):
        by_tok.setdefault((a, b), j)
    by_id = {(a, int(b)): j for j, (a, b) in enumerate(zip(cand["src"].tolist(), cand["row"].tolist()))}
    return by_tok, by_id


def chosen_candidates(path, index):
    """Indici (nei candidati) dei difetti di un file removed_*: per token users/items se ci sono,
    altrimenti per source+defect_id (= riga nel file della sorgente)."""
    by_tok, by_id = index
    with open(path, newline="", encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    delim = "\t" if "\t" in lines[0] else ","
    out, miss = set(), 0
    for r in csv.DictReader(lines, delimiter=delim):
        j = None
        if r.get("users") and r.get("items"):
            j = by_tok.get((",".join(sorted(_toks(r["users"]))), ",".join(sorted(_toks(r["items"])))))
        if j is None and r.get("source") and str(r.get("defect_id", "")).strip() != "":
            j = by_id.get((r["source"], int(float(r["defect_id"]))))
        if j is None:
            miss += 1
        else:
            out.add(j)
    if miss:
        log(f"    ATTENZIONE: {miss} righe di {Path(path).name} non trovate tra i candidati")
    return np.array(sorted(out), np.int64)


def dump_dataset(ds, args):
    import gc
    import torch
    rre, rc, lo, mrd = _project(args.rre_module)
    staging = Path(args.staging_dir or rc.STAGING_DIR)
    removed_dir = Path(args.removed_dir or rc.BASE_DIR / "removed_defects")
    out = args.out / ds / "lists"
    out.mkdir(parents=True, exist_ok=True)
    files = []
    for f in rre.find_removed_files(removed_dir, ds, args.pattern):
        sel = rre.selection_name(f, ds)
        meth, scope = rre.selection_parts(sel)
        meth = _canon_method(meth)
        if args.scopes and not any(x in scope for x in args.scopes):
            continue
        if want_method(meth, args):
            files.append((f, sel, meth, scope))
    log(f"[{ds}] {len(files)} file selezionati: " + ", ".join(s for _, s, _, _ in files))
    if not files:
        return
    K = args.k
    F = len(files)
    all_files = list(files)                            # per le curve: unione degli item dello scope
    if args.list_tasks:
        log(f"  task 0: base + candidati + controllo propagazione")
        for j, (_, sel, _, _) in enumerate(files):
            log(f"  task {1 + j:3d}: retrain {sel}      task {1 + F + j:3d}: prop {sel}")
        log(f"  totale: {1 + 2 * F} task (0..{2 * F})")
        return
    variants = list(args.variants)
    only_base = False
    if args.task is not None:                          # un solo pezzo di lavoro (job array)
        t = args.task
        if t == 0:
            only_base = True
        elif 1 <= t <= F:
            files, variants = [files[t - 1]], ["retrain"]
        elif F < t <= 2 * F:
            files, variants = [files[t - F - 1]], ["prop"]
        else:
            log(f"  task {t} fuori range (0..{2 * F}): niente da fare")
            return
        log(f"  task {t}: " + ("base + candidati" if only_base else f"{variants[0]} {files[0][1]}"))

    # ── modello base ─────────────────────────────────────────────────────────
    dataset_dir = rc.DATASETS[ds]
    noise = rc.build_noise_registry(ds)["clean"]
    _, base_tag = rc.stage_dataset(ds, dataset_dir, "clean", "0.0", noise["levels"]["0.0"],
                                   noise["subdir"], staging)
    t0 = time.time()
    base = rre.build_env(ds, base_tag, staging, rc.SAVED_DIR / base_tag / rre.MODEL)
    log(f"  base: dataset RecBole pronto in {time.time() - t0:.0f}s")
    uid_f, iid_f = base.dataset.uid_field, base.dataset.iid_field
    ref_uid, ref_iid = base.dataset.field2token_id[uid_f], base.dataset.field2token_id[iid_f]
    n_users, n_items = base.dataset.user_num, base.dataset.item_num
    base_path = out / "base.npz"
    if args.task is not None and not only_base and not base_path.exists():
        raise FileNotFoundError(f"{base_path} non esiste: lancia prima il task 0 di {ds}")
    if not base_path.exists() or (args.overwrite and (args.task is None or only_base)):
        model0 = _load_trained(rre, rc, base, allow_train=True)
        t0 = time.time()
        res0 = rre.collect(model0, base, K, np.arange(n_users), np.arange(n_items))
        log(f"  base: top-{K} calcolate in {time.time() - t0:.0f}s")
        del model0
        gc.collect(); torch.cuda.empty_cache()
        items = base.train.dataset.inter_feat[iid_f].numpy()
        t0 = time.time()
        tu, ti = _test_pairs(base, np.arange(n_users), np.arange(n_items))
        log(f"  base: coppie di test lette in {time.time() - t0:.0f}s")
        segs = {}
        for seg in ("active", "lt"):
            try:
                segs[f"seg_{seg}"] = lo.load_segment(ds, seg, ref_uid).astype(np.int64)
            except FileNotFoundError as e:
                log(f"  ATTENZIONE: {e}")
                segs[f"seg_{seg}"] = np.zeros(0, np.int64)
        _save_res(base_path, res0, pop=np.bincount(items, minlength=n_items).astype(np.int64),
                  n_items=n_items, n_users=n_users, test_u=tu, test_i=ti, **segs)
        log(f"  base: {len(res0.users)} utenti, top-{K} salvate in {base_path}")

    # ── difetti candidati (per l'analisi della selezione) ────────────────────
    cand_path = out / "candidates.npz"
    if args.task is not None and not only_base and not cand_path.exists():
        raise FileNotFoundError(f"{cand_path} non esiste: lancia prima il task 0 di {ds}")
    if not cand_path.exists() or (args.overwrite and (args.task is None or only_base)):
        R = base.train.dataset.inter_matrix(form="csr").astype(np.float32)
        dump_candidates(ds, args, rc, lo, R, ref_uid, ref_iid, cand_path)
    t0 = time.time()
    cand = np.load(cand_path)
    cand_idx = candidate_index(cand)
    log(f"  candidati: indice pronto ({len(cand['src'])} difetti) in {time.time() - t0:.0f}s")

    if only_base:                                       # task 0: controllo della propagazione e stop
        if "prop" in args.variants:
            _prop_setup(rre, lo, base, np.load(base_path), K, check=True)
        return

    # ── rimozioni: propagazione (prop) e/o retraining (retrain) ──────────────
    train_src = dataset_dir / noise["subdir"] / noise["levels"]["0.0"]
    train_rows = rc._parse_inter(train_src, want_rating=False)
    u2i = mrd.user_items(train_rows)
    B = np.load(base_path)
    prop_dir = out / "prop"
    prop_dir.mkdir(exist_ok=True)
    prop = None                                          # modello base + grafo, caricati solo se servono

    group_cache = {}

    def group_info(scope, budget):
        """Unione degli item difettosi e archi massimi dei file dello stesso scope e budget."""
        key = (scope, budget)
        if key not in group_cache:
            items, emax = set(), 0
            for p2, s2, _, sc2 in all_files:
                if (sc2, budget_of(s2)) != key:
                    continue
                try:
                    dd = mrd.read_removed(p2, ds)
                except TypeError:
                    dd = mrd.read_removed(p2)
                rem = set()
                for d in dd:
                    rem.update(mrd.defect_edges(u2i, d))
                items |= {ref_iid[i] for _, i in rem if i in ref_iid}
                emax = max(emax, len(rem))
            group_cache[key] = (np.array(sorted(items), np.int64), emax)
        return group_cache[key]

    bad_ckpt = []                                      # file senza checkpoint coerente: da riallenare
    curve_dir = prop_dir / "curves_lists"
    if args.curves > 0 and "prop" in variants:
        curve_dir.mkdir(exist_ok=True)

    for path, sel, meth, scope in files:
        dst, dst_p = out / f"{sel}.npz", prop_dir / f"{sel}.npz"
        dst_c = curve_dir / f"{sel}.npz"
        if "retrain" in variants and dst.exists() and not args.overwrite:
            z = dict(np.load(dst))
            if "chosen" not in z:                       # dump vecchio: aggiungo solo i difetti scelti
                z["chosen"] = chosen_candidates(path, cand_idx)
                np.savez_compressed(dst, **z)
                log(f"  {sel}: aggiunti {len(z['chosen'])} difetti scelti")
        need_rt = "retrain" in variants and (not dst.exists() or args.overwrite)
        need_pr = "prop" in variants and (not dst_p.exists() or args.overwrite)
        need_cv = args.curves > 0 and "prop" in variants and (not dst_c.exists() or args.overwrite)
        if not (need_rt or need_pr or need_cv):
            log(f"  {sel}: già salvato")
            continue
        try:
            defects = mrd.read_removed(path, ds)
        except TypeError:
            defects = mrd.read_removed(path)
        removed = set()
        for d in defects:
            removed.update(mrd.defect_edges(u2i, d))
        t_items = np.array(sorted({ref_iid[i] for _, i in removed if i in ref_iid}), np.int64)
        meta = dict(targets=t_items, members=rre.member_ids(defects, ref_uid).astype(np.int64),
                    method=np.array(meth), scope=np.array(scope), sel=np.array(sel),
                    n_edges=len(removed), n_defects=len(defects), chosen=chosen_candidates(path, cand_idx))

        if need_pr:
            if prop is None:                             # con --task il controllo l'ha fatto il task 0
                prop = _prop_setup(rre, lo, base, B, K, check=args.task is None)
            ru = np.array([ref_uid.get(u, -1) for u, _ in removed], np.int64)
            ri = np.array([ref_iid.get(i, -1) for _, i in removed], np.int64)
            ok = (ru >= 0) & (ri >= 0)
            t0 = time.time()
            res_p = _prop_lists(prop, lo, ru[ok], ri[ok], K)
            log(f"    propagazione + top-{K} in {time.time() - t0:.0f}s")
            _save_res(dst_p, res_p, **meta)
            log(f"  {sel} [prop]: {int(ok.sum())} archi tolti dal grafo -> {dst_p}")

        if need_cv:
            if prop is None:
                prop = _prop_setup(rre, lo, base, B, K, check=args.task is None)
            u_items, emax = group_info(scope, budget_of(sel))
            t0 = time.time()
            dump_curves(path, meth, scope, sel, defects, u2i, mrd, prop, lo, ref_uid, ref_iid, n_items,
                        u_items, emax, args, dst_c)
            log(f"    curve in {time.time() - t0:.0f}s")

        if need_rt:
            tag = re.sub(r"[^A-Za-z0-9_.]+", "_", f"{ds}__rm__{sel}")      # come run_removed_eval
            saved_dir = rc.SAVED_DIR / tag / rre.MODEL
            if args.no_train and (not saved_dir.exists() or rc.find_checkpoint(saved_dir) is None):
                log(f"  {sel} [retrain]: nessun checkpoint in {saved_dir}, salto (--no-train)")
                bad_ckpt.append(sel)
                continue
            # staging SEMPRE dal file attuale: uno staging vecchio (file cambiato) darebbe un dataset
            # diverso da quello del checkpoint
            keep_rows = [r for r in train_rows if (r[0], r[1]) not in removed]
            rre.stage_removed(ds, tag, keep_rows, staging)
            t0 = time.time()
            env = rre.build_env(ds, tag, staging, saved_dir)
            log(f"    dataset RecBole pronto in {time.time() - t0:.0f}s")
            model1 = _load_trained(rre, rc, env, allow_train=not args.no_train)
            if model1 is None:
                log(f"  {sel} [retrain]: nessun checkpoint utilizzabile, salto")
                bad_ckpt.append(sel)
                continue
            u_map = rre.remap(env.dataset, env.dataset.uid_field, ref_uid)
            i_map = rre.remap(env.dataset, env.dataset.iid_field, ref_iid)
            t0 = time.time()
            res1 = rre.collect(model1, env, K, u_map, i_map)
            log(f"    top-{K} calcolate in {time.time() - t0:.0f}s")
            del model1
            gc.collect(); torch.cuda.empty_cache()
            _save_res(dst, res1, **meta)
            log(f"  {sel} [retrain]: {len(defects)} difetti, {len(removed)} archi, "
                f"{len(t_items)} item target -> {dst}")
    if bad_ckpt:
        tags = [re.sub(r"[^A-Za-z0-9_.]+", "_", f"{ds}__rm__{x}") for x in bad_ckpt]
        log(f"[{ds}] DA RIALLENARE (checkpoint mancante o allenato su un'altra lista di difetti): "
            + ", ".join(bad_ckpt))
        log(f"[{ds}] cartelle in {rc.SAVED_DIR}: " + ", ".join(tags))


def _load_trained(rre, rc, env, allow_train):
    """Carica il checkpoint SENZA la valutazione completa sul test che fa fit_or_load
    (al dump non serve e costa minuti). Se il checkpoint manca: fit_or_load (allena) se
    consentito, altrimenti None."""
    import torch
    ckpt = rc.find_checkpoint(env.saved_dir) if env.saved_dir.exists() else None
    if ckpt is None:
        if not allow_train:
            return None
        model, _ = rre.fit_or_load(env, force_train=False)
        return model
    cfg = env.cfg
    ck = torch.load(ckpt, map_location=cfg["device"])
    for key in ("n_layers", "embedding_size"):
        if ck["config"][key] != cfg[key]:
            raise ValueError(f"{key}: checkpoint={ck['config'][key]} config={cfg[key]} ({ckpt})")
    model = rre.get_model(rre.MODEL)(cfg, env.train.dataset).to(cfg["device"])
    try:
        model.load_state_dict(ck["state_dict"])
    except RuntimeError as e:
        shapes = {k: tuple(v.shape) for k, v in ck["state_dict"].items() if k.endswith("embedding.weight")}
        log(f"    ATTENZIONE: il checkpoint {ckpt} non corrisponde al file di rimozione attuale "
            f"(checkpoint {shapes}, dataset utenti={env.dataset.user_num} item={env.dataset.item_num}): "
            f"è stato allenato su un'altra lista di difetti. Va riallenato. ({str(e).splitlines()[0]})")
        return None
    if hasattr(model, "restore_user_e"):
        model.restore_user_e = None
        model.restore_item_e = None
    model.eval()
    log(f"    checkpoint: {Path(ckpt).name}")
    return model


def _prop_setup(rre, lo, base, B, K, check=True):
    """Modello base + grafo di training per la propagazione (come la variante 'prop' di loo_defects).
    Controlla che la propagazione sul grafo intero riproduca le top-K del modello base."""
    model = _load_trained(rre, rre.rc, base, allow_train=True)
    graph = lo.TrainGraph(base.train.dataset, base.cfg["device"])
    users = B["users"]
    keys = np.sort(B["test_u"].astype(np.int64) * graph.n_items + B["test_i"].astype(np.int64))
    # item da escludere oltre al training: quelli di validation (come la valutazione full-sort di
    # RecBole sul test, che esclude training + validation). Il modello base è valutato così.
    V = base.valid.dataset.inter_matrix(form="csr").astype(np.float32)
    V.data[:] = 1.0
    P = SimpleNamespace(model=model, graph=graph, users=users, npos=B["npos"], test_keys=keys, V=V)
    if not check:
        return P
    chk = _prop_lists(P, lo, np.zeros(0, np.int64), np.zeros(0, np.int64), K)
    n = min(2000, len(users))
    top0 = np.asarray(B["top"])          # una sola lettura (B è un file npz: ogni B[...] rilegge dal disco)
    ov = np.mean([len(np.intersect1d(chk.top[r], top0[r])) / K for r in range(n)])
    log(f"  [prop] controllo: overlap top-{K} propagazione su grafo intero vs base = {ov:.4f}")
    if ov < 0.95:
        log("  ATTENZIONE: la propagazione non riproduce il modello base (overlap < 0.95)")
    return P


def _prop_lists(P, lo, ru, ri, K):
    """Top-K degli utenti di test con gli embedding del modello base propagati sul grafo senza
    gli archi (ru, ri). Esclusi gli item di training (grafo ripulito, come nel retraining) e di
    validation (come la valutazione full-sort di RecBole sul test)."""
    import scipy.sparse as sp
    R = P.graph.R.tocoo()
    if len(ru):
        rm = np.isin(R.row.astype(np.int64) * P.graph.n_items + R.col, ru * P.graph.n_items + ri)
        R_mod = sp.csr_matrix((R.data[~rm], (R.row[~rm], R.col[~rm])), shape=R.shape)
    else:
        R_mod = P.graph.R
    ue, ie = lo.propagate(P.model, P.graph.norm_adj(R_mod))
    hist = R_mod + P.V if getattr(P, "V", None) is not None else R_mod     # training ripulito + validation
    top, _ = lo.topk_and_ranks(ue, ie, P.users, K, hist.tocsr())
    top = np.asarray(top)
    hits = np.isin(P.users[:, None].astype(np.int64) * P.graph.n_items + top, P.test_keys)
    return SimpleNamespace(users=P.users, top=top, hits=hits, npos=P.npos)


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


def analyze_dataset(ds, args):
    """Metriche 'dopo' per ogni variante (retrain, prop) in analysis_<tag>/<variante>/;
    selezione (prima della rimozione, uguale per le varianti) in analysis_<tag>/."""
    global EXPOSURE_TARGETS
    EXPOSURE_TARGETS = args.tr_targets
    D = load_dataset(ds, args)
    ks = [k for k in args.ks if k <= D.K]
    K = max(ks)
    root = args.out / ds / ("analysis" + (f"_{args.tag}" if args.tag else ""))
    root.mkdir(parents=True, exist_ok=True)
    log(f"[{ds}] output in {root}  (affected-k={args.affected_k}, ks={args.ks}, boot={args.boot})")
    rows_tgt_all = []
    for variant, sels in D.variants.items():
        log(f"[{ds}] ── variante: {VARIANT_LABEL[variant]} ──")
        D.sels = sels
        rows_tgt_all += analyze_variant(D, args, ks, K, root / variant, variant)
    if args.only_tables:
        log(f"[{ds}] --only-tables: selezione, figure, overlap e curve saltate. Tabelle in {root}/<variante>/")
        return rows_tgt_all
    D.sels = next(iter(D.variants.values()))       # la selezione usa solo il modello base
    sel_out = selection(D, args, K)
    for name, rows in sel_out.items():
        write_csv(root / f"{name}.csv", rows)
    (root / "tables_selection.tex").write_text(latex_selection(sel_out, [], []))
    write_csv(root / "targeting_all_variants.csv", rows_tgt_all)
    for fn, a in ((figures_selection, (sel_out, root)), (figure_variants, (rows_tgt_all, root))):
        try:
            fn(*a)
        except Exception as e:
            log(f"  figure non generate ({fn.__name__}): {e}")
    for fn, a in ((overlap_analysis, (D, root)), (curves_analysis, (D, args, root))):
        try:
            fn(*a)
        except Exception as e:
            log(f"  {fn.__name__} non eseguita: {e}")
    log(f"[{ds}] risultati in {root}")
    return rows_tgt_all


def analyze_variant(D, args, ks, K, outdir, variant):
    outdir.mkdir(parents=True, exist_ok=True)
    rows_long, rows_test, rows_rep = [], [], []

    # metriche del modello base per utente (uguali per tutti i metodi)
    acc0 = {}
    for k in ks:
        acc0.update(accuracy_pu(D.base.hits, D.base.npos, k))
        acc0.update(novelty_pu(D.base.top, D.pop, D.n_train, k))
    merit_cache = {}

    def merit_of(s, uset, g):
        key = (s.budget, s.scope, uset, g) if uset != "seg" else ("seg", g)
        if key not in merit_cache:
            merit_cache[key] = merit_for(D, s.us[(uset, g)])
        return merit_cache[key]

    for s in D.sels:
        acc1 = {}
        for k in ks:
            acc1.update(accuracy_pu(s.hits, s.npos, k))
            acc1.update(novelty_pu(s.top, D.pop, D.n_train, k))
        tg0, tg1 = {}, {}
        for k in ks:
            tg0.update(target_pu(D.base.top, s.tmask, k, int(s.tmask.sum())))
            tg1.update(target_pu(s.top, s.tmask, k, int(s.tmask.sum())))
            tg0.update(target_pu(D.base.top, s.t_union, k, int(s.t_union.sum()), "_u"))
            tg1.update(target_pu(s.top, s.t_union, k, int(s.t_union.sum()), "_u"))
            for suf, tm in target_splits(s, D.tail).items():          # nicchia / head
                tg0.update(target_pu(D.base.top, tm, k, int(tm.sum()), suf))
                tg1.update(target_pu(s.top, tm, k, int(tm.sum()), suf))
            tg0[f"t_rank_u@{k}"] = target_rank_pu(D.base.top, D.base.top, s.t_union, k)   # rank medio prima
            tg1[f"t_rank_u@{k}"] = target_rank_pu(D.base.top, s.top, s.t_union, k)        # e dopo (unione)
        s.Ek0, s.Ek1 = {k: tg0[f"t_dcg@{k}"] for k in ks}, {k: tg1[f"t_dcg@{k}"] for k in ks}
        s.Eku0, s.Eku1 = {k: tg0[f"t_dcg_u@{k}"] for k in ks}, {k: tg1[f"t_dcg_u@{k}"] for k in ks}
        s.E0, s.E1 = s.Ek0[K], s.Ek1[K]                     # selezione: k più grande
        s.acc0, s.acc1 = acc0, acc1
        keep_t = [f"{m}@{k}" for k in ks for m in ("t_hr", "t_ndcg", "t_hr_u", "t_ndcg_u")]
        keep_t += [f"{m}{suf}@{k}" for k in ks for m in ("t_hr", "t_ndcg") for suf in SPLIT_SUFFIXES]
        keep_t += [f"t_rank_u@{k}" for k in ks]
        n_t, n_h = int(s.tmask.sum()), int((s.tmask & ~D.tail).sum())
        log(f"  {s.sel:40s} item target: {n_t}, head {n_h} ({n_h / max(n_t, 1):.0%}), "
            f"nicchia {n_t - n_h}")
        s.tg0 = {k_: tg0[k_] for k_ in keep_t}
        s.tg1 = {k_: tg1[k_] for k_ in keep_t}
        meta = dict(dataset=D.ds, variant=variant, budget=s.budget, scope=s.scope, method=s.method, sel=s.sel,
                    n_edges=s.n_edges)

        for (uset, g), m in s.us.items():
            if uset == "unaff" or m.sum() == 0:
                continue
            base_kw = dict(meta, userset=uset, segment=g, n_users=int(m.sum()))
            for name in list(acc0):
                if name.startswith("novelty_pu"):          # solo per i test (= novelty@k di beyond)
                    continue
                b, a = float(acc0[name][m].mean()), float(acc1[name][m].mean())
                rows_long.append(dict(base_kw, metric=name, before=b, after=a))
            for k in ks:
                b0 = beyond(D.base.top[m], D.pop, D.n_train, D.catalog, D.tail, k)
                b1 = beyond(s.top[m], D.pop, D.n_train, D.catalog, D.tail, k)
                for name in b0:
                    rows_long.append(dict(base_kw, metric=name, before=b0[name], after=b1[name]))
                for gm, tag in ((s.tmask, "own"), (s.t_union, "def"), (D.tail, "tail")):
                    f0 = foe(D.base.top[m], k, gm, merit_of(s, uset, g), D.n_items, tag)
                    f1 = foe(s.top[m], k, gm, merit_of(s, uset, g), D.n_items, tag)
                    for name in f0:
                        rows_long.append(dict(base_kw, metric=name, before=f0[name], after=f1[name]))
            for name in tg0:
                rows_long.append(dict(base_kw, metric=name, before=_nmean(tg0[name][m]),
                                      after=_nmean(tg1[name][m])))

            # test appaiati sugli affetti
            if uset == "aff":
                for name in [f"{mm}@{kk}" for kk in ks for mm in ("ndcg", "recall")]:
                    b, a = acc0[name][m], acc1[name][m]
                    mean_d, se, p = tost(b, a, args.tost_margin * b.mean())
                    rows_test.append(dict(meta, segment=g, n_users=int(m.sum()), metric=name, test="TOST",
                                          before=b.mean(), after=a.mean(), mean_diff=mean_d, se=se,
                                          margin=args.tost_margin * b.mean(), p=p, equivalent=p < args.alpha))
                for k, suf in ((k, suf) for k in ks for suf in ("", "_niche", "_head")):
                    name = f"t_ndcg{suf}@{k}"
                    b, a = tg0[name][m], tg1[name][m]
                    rows_test.append(dict(meta, segment=g, n_users=int(m.sum()), metric=name, test="Wilcoxon",
                                          before=b.mean(), after=a.mean(), mean_diff=float((a - b).mean()),
                                          se=np.nan, margin=np.nan, p=wilcoxon_p(b, a), equivalent=""))

        if args.only_tables:
            log(f"  {s.sel:40s} [{s.method}/{s.scope}] fatto")
            continue
        # sostituzione: affetti e non affetti (rumore di fondo)
        n = len(D.users)
        key0 = np.arange(n)[:, None] * D.n_items + D.base.top[:, :K]
        key1 = np.arange(n)[:, None] * D.n_items + s.top[:, :K]
        stay1 = np.isin(key1, key0)          # in 'dopo' e già in 'prima'
        stay0 = np.isin(key0, key1)
        ent, ext = ~stay1, ~stay0
        lp = np.log1p(D.pop)
        for (uset, g), m in s.us.items():
            if uset == "seg" or m.sum() == 0:
                continue
            e, x = ent[m], ext[m]
            ne, nx = e.sum(), x.sum()
            ti0, ti1 = D.base.top[m, :K], s.top[m, :K]
            rows_rep.append(dict(meta, userset=uset, segment=g, k=K, n_users=int(m.sum()),
                                 churn=float(e.sum(1).mean() / K),
                                 unchanged_users=float((e.sum(1) == 0).mean()),
                                 exited_target_share=float(s.tmask[ti0][x].sum() / nx) if nx else np.nan,
                                 entered_target_share=float(s.tmask[ti1][e].sum() / ne) if ne else np.nan,
                                 exited_precision=float(D.base.hits[m, :K][x].mean()) if nx else np.nan,
                                 entered_precision=float(s.hits[m, :K][e].mean()) if ne else np.nan,
                                 exited_tail_share=float(D.tail[ti0][x].mean()) if nx else np.nan,
                                 entered_tail_share=float(D.tail[ti1][e].mean()) if ne else np.nan,
                                 exited_logpop=float(lp[ti0][x].mean()) if nx else np.nan,
                                 entered_logpop=float(lp[ti1][e].mean()) if ne else np.nan))
        log(f"  {s.sel:40s} [{s.method}/{s.scope}] fatto")

    for r in rows_long:
        r["delta"] = r["after"] - r["before"]
        r["rel_delta"] = r["delta"] / r["before"] if r["before"] not in (0, None) and np.isfinite(r["before"]) else np.nan
    if args.only_tables:                         # bootstrap e indici di targeting: non servono alle tabelle
        rows_tgt, rows_pair, rows_ugf = [], [], []
    else:
        rows_tgt = [r for k in ks for r in targeting(D, args, k)]
        rows_pair = [r for k in ks for r in paired_methods(D, args, k)]
        rows_ugf = ugf(D, K)
    rows_tt = ttests(D, args, ks)
    for r in rows_tt:
        r["variant"] = variant
    write_csv(outdir / "ttests.csv", rows_tt)
    for rows in (rows_tgt, rows_pair, rows_ugf):
        for r in rows:
            r["variant"] = variant
    write_csv(outdir / "paired_methods.csv", rows_pair)
    write_csv(outdir / "ugf.csv", rows_ugf)
    write_csv(outdir / "metrics_long.csv", rows_long)
    write_csv(outdir / "tests.csv", rows_test)
    write_csv(outdir / "replacement.csv", rows_rep)
    write_csv(outdir / "targeting.csv", rows_tgt)
    vl = VARIANT_LABEL[variant]
    for k in ks:
        for us in ("aff", "seg"):
            (outdir / f"tables_k{k}_{us}.tex").write_text(latex_paper_tables(rows_long, rows_tt, k, us, vl, args.alpha))
            (outdir / f"tables_abs_k{k}_{us}.tex").write_text(
                latex_abs_tables(rows_long, rows_tt, rows_test, k, us, vl, args.alpha, D.K))
    (outdir / "tables_setup.tex").write_text(latex_setup(rows_long, ks[0]))
    if args.only_tables:
        log(f"  [{variant}] tabelle scritte in {outdir}")
        return rows_tgt
    (outdir / "tables_targeting.tex").write_text(latex_targeting(rows_tgt, vl))
    (outdir / "tables_extra.tex").write_text(
        latex_selection({}, [r for r in rows_pair if r["k"] == K], rows_ugf).replace("\\caption{", f"\\caption{{[@{K}, {vl}] "))
    try:
        figures(rows_long, rows_tgt, outdir, vl, heat_ks=sorted(set(args.heatmap_ks or ks) & set(ks)))
    except Exception as e:                       # le figure non devono bloccare i CSV
        log(f"  figure non generate (figures): {e}")
    try:
        heatmap_values(rows_long, rows_tt, outdir, vl, heat_ks=sorted(set(args.heatmap_ks or ks) & set(ks)),
                       alpha=args.alpha)
    except Exception as e:
        log(f"  figure non generate (heatmap_values): {e}")
    return rows_tgt


TARGETS_LABEL = {"union": "defective items: union of the items removed in the scope (same set for every method)",
                 "own": "defective items: the items removed by each method"}
EXPOSURE_TARGETS = "union"          # impostato da --tr-targets in analyze (TR, lift, confronto appaiato)


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


# ============================================================================ #
#  OUTPUT
# ============================================================================ #
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


# ── tabelle per il paper ──────────────────────────────────────────────────────
# (metrica in metrics_long, etichetta, metrica per utente usata nei test o None)
ACC_COLS = (("ndcg", "NDCG", "ndcg"), ("recall", "Recall", "recall"), ("hit", "Hit", "hit"),
            ("coverage", "Coverage", None), ("novelty", "Novelty", "novelty_pu"), ("entropy", "Entropy", None), ("gini", "Gini", None))
EXP_COLS = (("t_ndcg", "T-NDCG", "t_ndcg"), ("t_hr", "T-HR", "t_hr"), ("foe_own_dtr", "DTR", None))
USERSET_LABEL = {"aff": "affected users", "seg": "whole segment (defect members excluded)",
                 "unaff": "unaffected users"}


def _tt_index(rows_tt):
    key = lambda r, who: (r["budget"], r["scope"], r["userset"], r["segment"], r["metric"], r[who])
    return ({key(r, "a"): r for r in rows_tt if r["kind"] == "before_after"},
            {key(r, "b"): r for r in rows_tt if r["kind"] == "ref_vs_other"},
            {key(r, "b"): r for r in rows_tt if r["kind"] == "ref_vs_other_removed"})


def _sig(r):
    return r is not None and bool(r.get("significant"))


def _num(x):
    return "--" if x is None or not np.isfinite(x) else (f"{x:.4f}" if abs(x) < 10 else f"{x:.2f}")


def _dnum(x):
    return "" if x is None or not np.isfinite(x) else f"$({x:+.4f})$"


def _table(caption, cols, body):
    return (["", "\\begin{table}[t]\\centering\\scriptsize", f"\\caption{{{caption}}}",
             "\\resizebox{\\ifdim\\width>\\linewidth\\linewidth\\else\\width\\fi}{!}{%",   # riduce solo se serve
             "\\begin{tabular}{ll" + "c" * len(cols) + "}", "\\toprule",
             "Segment & Method & " + " & ".join(cols) + " \\\\", "\\midrule"]
            + body + ["\\bottomrule", "\\end{tabular}}", "\\end{table}"])


def latex_paper_tables(rows_long, rows_tt, k, userset, vlabel, alpha):
    """
    Per ogni (budget, scope), utenti `userset` (aff | seg), @k:
      Tabella 1  accuratezza e beyond: valore DOPO e, tra parentesi, dopo - prima; riga Base = prima.
                 dagger = diverso da Base, * = diverso dal MAB del segmento (t-test appaiati, Holm).
      Tabella 2  T-NDCG, T-HR, DTR sugli item dei difetti rimossi dal metodo: valore DOPO e, tra
                 parentesi, prima - dopo (esposizione tolta, positivo = meglio).
                 dagger = dopo diverso da prima, * = esposizione tolta diversa da quella del MAB.
    """
    idx = {(r["budget"], r["scope"], r["method"], r["segment"], r["metric"]): r
           for r in rows_long if r["userset"] == userset}
    ba, rv, rr = _tt_index(rows_tt)
    ul = USERSET_LABEL[userset]
    out = [f"% generated by targeting_eval.py — {ul}, @{k}, {vlabel}",
           "% requires \\usepackage{booktabs,multirow,graphicx}"]

    def cell(r, delta, me, g, ref, tm, sig_ref):
        if r is None:
            return "--"
        c = f"{_num(r['after'])} {_dnum(delta)}"
        if me == ref:
            c = f"\\textbf{{{c}}}"
        sup = ""
        if tm:
            key = (r["budget"], r["scope"], userset, g, f"{tm}@{k}", me)
            sup += "\\dagger" if _sig(ba.get(key)) else ""
            sup += "*" if me != ref and _sig(sig_ref.get(key)) else ""
        return c + (f"$^{{{sup}}}$" if sup else "")

    for budget, scope in sorted({(b, sc) for b, sc, *_ in idx}):
        meths = sorted({m for b, sc, m, *_ in idx if (b, sc) == (budget, scope)}, key=method_key)
        cap = f"{SCOPE_LABEL.get(scope, scope)}, budget {budget}, @{k}, {ul} ({vlabel})"
        body1, body2 = [], []
        for g in SEGMENTS:
            if not any((budget, scope, me, g, f"ndcg@{k}") in idx for me in meths):
                continue
            ref = REF_OF_SEGMENT[g]
            first = next(idx[(budget, scope, me, g, f"ndcg@{k}")] for me in meths
                         if (budget, scope, me, g, f"ndcg@{k}") in idx)
            base = []
            for m, _, _ in ACC_COLS:
                r = next((idx[(budget, scope, me, g, f"{m}@{k}")] for me in meths
                          if (budget, scope, me, g, f"{m}@{k}") in idx), None)
                base.append(_num(r["before"]) if r else "--")
            lines1 = ["Base & " + " & ".join(base)]
            lines2 = []
            for me in meths:
                c1, c2 = [], []
                for m, _, tm in ACC_COLS:
                    r = idx.get((budget, scope, me, g, f"{m}@{k}"))
                    c1.append(cell(r, r["after"] - r["before"] if r else None, me, g, ref, tm, rv))
                for m, _, tm in EXP_COLS:
                    r = idx.get((budget, scope, me, g, f"{m}@{k}"))
                    c2.append(cell(r, r["before"] - r["after"] if r else None, me, g, ref, tm, rr))
                lines1.append(method_label(me) + " & " + " & ".join(c1))
                lines2.append(method_label(me) + " & " + " & ".join(c2))
            seg = f"{SEG_LABEL[g]} (n={first['n_users']})"
            for body, lines in ((body1, lines1), (body2, lines2)):
                body.append(f"\\multirow{{{len(lines)}}}{{*}}{{{seg}}} & " + lines[0] + " \\\\")
                body += [" & " + l + " \\\\" for l in lines[1:]]
                body.append("\\midrule")
        if not body1:
            continue
        body1[-1] = body2[-1] = ""
        out += _table(
            f"Accuracy and beyond-accuracy metrics, {cap}. Value after removal and, in parentheses, the "
            f"change (after $-$ before); Base = before removal. Bold: {_ref_txt()}. "
            f"$^\\dagger$: different from Base; $^*$: different from {_ref_txt()} (paired t-tests over "
            f"users, Holm correction, $\\alpha={alpha}$). Coverage, Entropy and Gini are computed on the whole set "
            "of lists and have no per-user value: not tested.",
            [l for _, l, _ in ACC_COLS], body1)
        out += _table(
            f"Exposure of defective items, {cap}. T-NDCG, T-HR and DTR are computed on the items of the "
            "defects removed by each method. Value after removal and, in parentheses, the exposure removed "
            f"(before $-$ after: positive = better). Bold: {_ref_txt()}. $^\\dagger$: after differs "
            f"from before; $^*$: exposure removed differs from that of {_ref_txt()} (paired t-tests over "
            f"users, Holm, $\\alpha={alpha}$). DTR has no per-user value: not tested.",
            [f"{l}@{k}" for _, l, _ in EXP_COLS], body2)
    return "\n".join(out) + "\n"


def latex_setup(rows_long, k):
    """Quanti utenti affetti per scope e segmento, e quanti archi ha rimosso ogni metodo."""
    rs = [r for r in rows_long if r["metric"] == f"ndcg@{k}"]
    n = {(r["budget"], r["scope"], r["userset"], r["segment"]): r["n_users"] for r in rs}
    edges = {(r["budget"], r["scope"], r["method"]): r["n_edges"] for r in rs}
    out = ["% generated by targeting_eval.py", "", "\\begin{table}[t]\\centering\\small",
           "\\caption{Affected users: users (defect members excluded) with at least one item removed in "
           "the scope among the top-$k$ of the base model.}",
           "\\begin{tabular}{lllrrr}", "\\toprule",
           "Budget & Scope & Segment & Affected & Segment size & \\% \\\\", "\\midrule"]
    for budget, scope in sorted({(b, sc) for b, sc, *_ in n}):
        for g in SEGMENTS:
            a, t = n.get((budget, scope, "aff", g)), n.get((budget, scope, "seg", g))
            if a is None or t is None:
                continue
            out.append(f"{budget} & {SCOPE_LABEL.get(scope, scope)} & {SEG_LABEL[g]} & {a} & {t} & "
                       f"{100 * a / max(t, 1):.1f} \\\\")
    out += ["\\bottomrule", "\\end{tabular}", "\\end{table}", "", "\\begin{table}[t]\\centering\\small",
            "\\caption{Training edges removed by each method.}", "\\begin{tabular}{lllr}", "\\toprule",
            "Budget & Scope & Method & Edges \\\\", "\\midrule"]
    for (budget, scope, me) in sorted(edges, key=lambda x: (x[0], x[1], method_key(x[2]))):
        out.append(f"{budget} & {SCOPE_LABEL.get(scope, scope)} & {method_label(me)} & {edges[(budget, scope, me)]} \\\\")
    out += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(out) + "\n"


def latex_targeting(rows_tgt, vlabel):
    out = ["% generated by targeting_eval.py"]
    for k in sorted({r["k"] for r in rows_tgt}):
        R = sorted([r for r in rows_tgt if r["k"] == k],
                   key=lambda r: (r["budget"], r["scope"], method_key(r["method"])))
        out += ["", "\\begin{table}[t]\\centering\\small",
                f"\\caption{{Where the removed defective exposure is concentrated, @{k} ({vlabel}). "
                "$E$ = T-DCG of the defective items in a user's list; $\\Delta E$ = before $-$ after, averaged "
                "over the affected users of the segment. "
                "TR $=\\Delta E_{lt}/\\Delta E_{active}$; lift $=\\Delta E_g/\\Delta E_{all}$; 95\\% bootstrap CI; "
                f"{TARGETS_LABEL[EXPOSURE_TARGETS]}.}}",
                "\\begin{tabular}{lllccc}", "\\toprule",
                "Budget & Scope & Method & lift$_{lt}$ & lift$_{active}$ & TR \\\\", "\\midrule"]
        for r in R:
            ci = lambda c: f"{_f(r[c], 2)} [{_f(r[c + '_lo'], 2)}, {_f(r[c + '_hi'], 2)}]"
            out.append(f"{r['budget']} & {SCOPE_LABEL.get(r['scope'], r['scope'])} & {method_label(r['method'])} & "
                       f"{ci('lift_lt')} & {ci('lift_active')} & {ci('TR')} \\\\")
        out += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(out) + "\n"


# ── heatmap ───────────────────────────────────────────────────────────────────
def heatmap_specs(k):
    """
    (userset, nome del file, metrica, metrica del denominatore, tipo, titolo).
    Pannello sinistro, più alto = meglio:
      share  (T-DCG prima - dopo, item del metodo) / T-DCG prima di tutti gli item difettosi dello
             scope (stesso denominatore per tutti i metodi)                                   [%]
      abs    T-DCG prima - dopo, item del metodo                                    [valore assoluto]
      rel    (T-NDCG prima - dopo) / prima sull'unione degli item dello scope (controllo)     [%]
    Pannello destro: NDCG dopo - prima ([%] diviso per prima; [assoluto] con 'abs').
    """
    num = "Damage removed = (T-DCG before − after, method's items)"
    out = []
    for us in ("aff", "seg"):
        out.append((us, f"{us}_pct", f"t_dcg@{k}", f"t_dcg_u@{k}", "share",
                    f"{num}\n/ T-DCG before, all defective items of the scope"))
        out.append((us, f"{us}_abs", f"t_dcg@{k}", None, "abs", f"{num}\nabsolute value"))
    out += [("aff", "aff_pct_niche", f"t_dcg_niche@{k}", f"t_dcg_u_niche@{k}", "share",
             f"{num}, niche only\n/ T-DCG before, niche items of the scope"),
            ("aff", "aff_pct_head", f"t_dcg_head@{k}", f"t_dcg_u_head@{k}", "share",
             f"{num}, popular only\n/ T-DCG before, popular items of the scope"),
            ("aff", "aff_pct_union", f"t_ndcg_u@{k}", None, "rel",
             "T-NDCG reduction = (before − after) / before\non all defective items of the scope")]
    return out


def figures(rows_long, rows_tgt, outdir, vlabel="", heat_ks=()):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for budget in sorted({r["budget"] for r in rows_long}):
        idx = {(r["userset"], r["scope"], r["method"], r["segment"], r["metric"]): r
               for r in rows_long if r["budget"] == budget}
        scopes = sorted({key[1] for key in idx})
        for kk in heat_ks:
            for us, fname, tmetric, dmetric, kind, ttl in heatmap_specs(kk):
                if not any(key[0] == us and key[4] == tmetric for key in idx):
                    continue
                pct = kind != "abs"
                fig, axes = plt.subplots(len(scopes), 2, figsize=(11, 2.4 + 2.0 * len(scopes)), squeeze=False)
                for i, sc in enumerate(scopes):
                    meths = sorted({key[2] for key in idx if key[1] == sc}, key=method_key)
                    panels = ((tmetric, kind, ttl),
                              (f"ndcg@{kk}", "acc", "Δ NDCG = (after − before)" + (" / before" if pct else "")
                               + "\n" + ("relative change" if pct else "absolute value")))
                    for j, (metric, pk, title) in enumerate(panels):
                        M = np.full((len(meths), len(SEGMENTS)), np.nan)
                        for a_, me in enumerate(meths):
                            for b_, g in enumerate(SEGMENTS):
                                r = idx.get((us, sc, me, g, metric))
                                if r is None:
                                    continue
                                if pk == "share":
                                    den = idx.get((us, sc, me, g, dmetric))
                                    if den is not None and den["before"] > 0:
                                        M[a_, b_] = (r["before"] - r["after"]) / den["before"]
                                elif pk == "abs":
                                    M[a_, b_] = r["before"] - r["after"]
                                elif pk == "rel":
                                    M[a_, b_] = -r["rel_delta"]
                                else:
                                    M[a_, b_] = r["rel_delta"] if pct else r["after"] - r["before"]
                        lim = max(np.nanmax(np.abs(M)) if np.isfinite(M).any() else 1e-9, 1e-9)
                        if pk == "acc" and pct:
                            lim = max(lim, 0.05)       # accuratezza: scala minima ±5% (piatto = piatto)
                        ax = axes[i, j]
                        im = ax.imshow(M, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
                        ax.set_xticks(range(len(SEGMENTS)), [SEG_LABEL[g] for g in SEGMENTS])
                        ax.set_yticks(range(len(meths)), [method_label(m) for m in meths])
                        for a_ in range(M.shape[0]):
                            for b_ in range(M.shape[1]):
                                if np.isfinite(M[a_, b_]):
                                    v = M[a_, b_]
                                    txt = f"{100 * v:+.1f}%" if pct else (f"{v:+.3g}" if pk == "abs" else f"{v:+.4f}")
                                    ax.text(b_, a_, txt, ha="center", va="center", fontsize=7,
                                            color="white" if abs(v) > 0.6 * lim else "black")
                        ax.set_title(f"{SCOPE_LABEL.get(sc, sc)} — {title}\n(@{kk}; {USERSET_LABEL[us]}; "
                                     f"{vlabel}; higher = better)", fontsize=7)
                        fig.colorbar(im, ax=ax, fraction=0.046)
                fig.tight_layout()
                fig.savefig(outdir / f"fig_heatmap_{budget}_k{kk}_{fname}.png", dpi=200)
                plt.close(fig)

        for kk in sorted({r["k"] for r in rows_tgt if r["budget"] == budget}):
            T = sorted([r for r in rows_tgt if r["budget"] == budget and r["k"] == kk],
                       key=lambda r: (r["scope"], method_key(r["method"])))
            fig, ax = plt.subplots(figsize=(6, 0.35 * len(T) + 1.2))
            y = np.arange(len(T))[::-1]
            tr = np.array([r["TR"] for r in T])
            err = np.array([[r["TR"] - r["TR_lo"], r["TR_hi"] - r["TR"]] for r in T]).T
            colors = ["C3" if r["method"].endswith("_lt") else "C0" if r["method"].endswith("_active") else "0.5"
                      for r in T]
            ax.errorbar(tr, y, xerr=err, fmt="none", ecolor="0.4", capsize=2)
            ax.scatter(tr, y, c=colors, zorder=3)
            ax.set_yticks(y, [f"{SCOPE_LABEL.get(r['scope'], r['scope'])} · {method_label(r['method'])}" for r in T])
            ax.set_xscale("log")
            ax.axvline(1, color="0.7", ls="--", lw=0.8)
            ax.set_xlabel(f"TR = ΔE(lt) / ΔE(active) @{kk}  (log scale, 95% CI) — {vlabel}")
            fig.tight_layout()
            fig.savefig(outdir / f"fig_targeting_{budget}_k{kk}.png", dpi=200)
            plt.close(fig)


def heatmap_values(rows_long, rows_tt, outdir, vlabel="", heat_ks=(), alpha=0.05):
    """
    Heatmap con i VALORI ASSOLUTI nelle celle (stessi numeri di tables_abs): per ogni scope, a sinistra
    T-NDCG@k sull'UNIONE degli item difettosi dello scope, a destra NDCG@k; righe Base e metodi, colonne
    All / Active / Long-tail, utenti affetti.
    COLORE = variazione rispetto a Base, orientata (blu = meglio, rosso = peggio, bianco = invariato):
      T-NDCG  (Base - dopo) / Base, stessa scala per tutti gli scope della figura
      NDCG    (dopo - Base) / Base, scala fissa almeno +-5% (accuratezza invariata = bianco)
    dagger = diverso da Base, * = diverso dal MAB del segmento (t-test appaiati, Holm).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ba, rv, _ = _tt_index(rows_tt)
    us = "aff"
    for budget in sorted({r["budget"] for r in rows_long}):
        idx = {(r["scope"], r["method"], r["segment"], r["metric"]): r
               for r in rows_long if r["budget"] == budget and r["userset"] == us}
        scopes = sorted({key[0] for key in idx})
        for kk in heat_ks:
            if not any(key[3] == f"t_ndcg_u@{kk}" for key in idx):
                continue
            specs = ((f"t_ndcg_u@{kk}", -1, f"T-NDCG@{kk} on the union of defective items"),
                     (f"ndcg@{kk}", +1, f"NDCG@{kk}"))
            data = {}                                     # (scope, metrica) -> (valori, colore, marcatori, metodi)
            for sc in scopes:
                meths = sorted({key[1] for key in idx if key[0] == sc}, key=method_key)
                for metric, sign, _ in specs:
                    V = np.full((len(meths) + 1, len(SEGMENTS)), np.nan)
                    mark = np.full(V.shape, "", dtype=object)
                    for b_, g in enumerate(SEGMENTS):
                        r0 = next((idx[(sc, me, g, metric)] for me in meths if (sc, me, g, metric) in idx), None)
                        if r0 is not None:
                            V[0, b_] = r0["before"]
                        for a_, me in enumerate(meths, start=1):
                            r = idx.get((sc, me, g, metric))
                            if r is None:
                                continue
                            V[a_, b_] = r["after"]
                            key = (budget, sc, us, g, metric, me)
                            mark[a_, b_] = ("†" if _sig(ba.get(key)) else "") + \
                                           ("*" if me != REF_OF_SEGMENT[g] and _sig(rv.get(key)) else "")
                    with np.errstate(divide="ignore", invalid="ignore"):
                        C = sign * (V - V[0:1, :]) / V[0:1, :]       # > 0 = meglio
                    data[(sc, metric)] = (V, C, mark, meths)
            lims = {}
            for metric, sign, _ in specs:
                allc = np.concatenate([data[(sc, metric)][1].ravel() for sc in scopes])
                allc = allc[np.isfinite(allc)]
                m = float(np.abs(allc).max()) if len(allc) else 0.0
                lims[metric] = max(m, 0.05) if sign > 0 else max(m, 1e-6)
            fig, axes = plt.subplots(len(scopes), 2, figsize=(11, 2.6 + 2.2 * len(scopes)), squeeze=False)
            for i, sc in enumerate(scopes):
                for j, (metric, sign, title) in enumerate(specs):
                    V, C, mark, meths = data[(sc, metric)]
                    lim = lims[metric]
                    ax = axes[i, j]
                    im = ax.imshow(C, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
                    ax.set_xticks(range(len(SEGMENTS)), [SEG_LABEL[g] for g in SEGMENTS])
                    ax.set_yticks(range(len(meths) + 1), ["Base"] + [method_label(m) for m in meths])
                    ax.axhline(0.5, color="black", lw=1)            # separa Base dai metodi
                    for a_ in range(V.shape[0]):
                        for b_ in range(V.shape[1]):
                            if np.isfinite(V[a_, b_]):
                                c = C[a_, b_]
                                dark = np.isfinite(c) and abs(c) > 0.6 * lim
                                ax.text(b_, a_, f"{V[a_, b_]:.4f}{mark[a_, b_]}", ha="center", va="center",
                                        fontsize=7, color="white" if dark else "black")
                    ax.set_title(f"{SCOPE_LABEL.get(sc, sc)} — {title}\n(values: absolute; colour: change vs. Base, "
                                 f"blue = better; {USERSET_LABEL[us]}; {vlabel})", fontsize=7)
                    cb = fig.colorbar(im, ax=ax, fraction=0.046)
                    cb.ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{100 * x:+.0f}%"))
            fig.text(0.5, 0.005, f"†: different from Base; *: different from {_ref_txt()} "
                     "(paired t-tests, Holm correction)", ha="center", fontsize=7)
            fig.tight_layout(rect=(0, 0.015, 1, 1))
            fig.savefig(outdir / f"fig_heatmap_{budget}_k{kk}_aff_values_union.png", dpi=200)
            plt.close(fig)


def figure_variants(rows_tgt, outdir):
    """TR con IC per metodo: propagazione e retraining affiancati (stessi utenti)."""
    variants = sorted({r["variant"] for r in rows_tgt})
    if len(variants) < 2 or not rows_tgt:
        return
    kmax = max(r.get("k", 0) for r in rows_tgt)
    rows_tgt = [r for r in rows_tgt if r.get("k", 0) == kmax]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for budget in sorted({r["budget"] for r in rows_tgt}):
        keys = sorted({(r["scope"], r["method"]) for r in rows_tgt if r["budget"] == budget},
                      key=lambda x: (x[0], method_key(x[1])))
        fig, ax = plt.subplots(figsize=(6.5, 0.4 * len(keys) + 1.3))
        y = np.arange(len(keys))[::-1]
        for j, (v, mk) in enumerate(zip(variants, ("o", "s"))):
            off = 0.15 * (1 - 2 * j)
            for yy, (sc, me) in zip(y, keys):
                r = [r for r in rows_tgt if (r["budget"], r["scope"], r["method"], r["variant"]) == (budget, sc, me, v)]
                if not r:
                    continue
                r = r[0]
                ax.errorbar(r["TR"], yy + off, xerr=[[r["TR"] - r["TR_lo"]], [r["TR_hi"] - r["TR"]]],
                            fmt=mk, color=_mcolor(me), mfc="white" if v == "prop" else _mcolor(me),
                            capsize=2, ms=5, label=VARIANT_LABEL[v] if yy == y[0] else None)
        ax.set_yticks(y, [f"{SCOPE_LABEL.get(sc, sc)} · {method_label(me)}" for sc, me in keys])
        ax.set_xscale("log")
        ax.axvline(1, color="0.7", ls="--", lw=0.8)
        ax.set_xlabel("TR = ΔE(lt) / ΔE(active)   (log scale, 95% CI)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(outdir / f"fig_targeting_variants_{budget}.png", dpi=200)
        plt.close(fig)


REF_OF_SEGMENT = {"all": "mab_all", "active": "mab_active", "lt": "mab_lt"}


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


def latex_selection(sel_out, rows_pair, rows_ugf):
    """Tabelle della selezione (reach, posizione, item) + confronto appaiato + UGF."""
    out = []
    S = sel_out.get("selection", [])
    tests = {(t["budget"], t["scope"], t["level"], t["measure"], t["segment"], t["ref"], t["other"]): t
             for t in sel_out.get("selection_tests", [])}

    def delta(budget, scope, level, measure, g, method):
        refm = ref_method(g)
        if method == refm:
            return "--"
        t = tests.get((budget, scope, level, measure, g, refm, method))
        if t is None or not np.isfinite(t["cliffs_delta"]):
            return "--"
        return f"{t['cliffs_delta']:+.2f}" + ("$^*$" if t["p_holm"] < 0.05 else "")

    specs = [
        ("reach", "Reach of the selected defects: share of segment users reached (Us.\\,\\%) and percentile "
                  "in the reach ranking of all candidates (100 = reaches the most users)",
         lambda r, g: _f(100 * r.get(f"reach_{g}_mean", np.nan), 2), "Us.\\,\\%", "defect"),
        ("pos", "Position of the selected defects: mean rank of the first item among reached users and "
                "percentile in the position ranking of the candidates (100 = highest)",
         lambda r, g: _f(r.get(f"pos_{g}_mean", np.nan), 1), "Rank", "defect"),
    ]
    for budget, scope in sorted({(r["budget"], r["scope"]) for r in S}):
        R = sorted([r for r in S if (r["budget"], r["scope"]) == (budget, scope)], key=lambda r: method_key(r["method"]))
        cap0 = f"({SCOPE_LABEL.get(scope, scope)}, budget {budget}, before removal)"
        for m, cap, raw, rawlab, level in specs:
            if not any(f"pct_{m}_lt_mean" in r for r in R):
                continue
            out += ["", "\\begin{table}[t]\\centering\\scriptsize",
                    f"\\caption{{{cap} {cap0}. Top-10\\%: share of selected defects in the top-10\\% of the "
                    "ranking; $\\delta$: Cliff's delta of "
                    + (f"{method_label(REF_OF_SEGMENT['lt'])} (Long-tail columns) or "
                       f"{method_label(REF_OF_SEGMENT['active'])} (Active columns)"
                       if REF_OF_SEGMENT["lt"] != REF_OF_SEGMENT["active"] else method_label(REF_OF_SEGMENT["lt"]))
                    + " against the row, $>0$ = the MAB is better; $^*$ Holm $p<0.05$.}",
                    "\\begin{tabular}{lcccccccc}", "\\toprule",
                    " & \\multicolumn{4}{c}{Long-tail} & \\multicolumn{4}{c}{Active} \\\\",
                    "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}",
                    f"Method & {rawlab} & Pct. & Top-10\\% & $\\delta$ & {rawlab} & Pct. & Top-10\\% & $\\delta$ \\\\",
                    "\\midrule"]
            for r in R:
                cells = []
                for g in ("lt", "active"):
                    cells += [raw(r, g), _f(r.get(f"pct_{m}_{g}_mean"), 1), _f(r.get(f"top10_{m}_{g}"), 2),
                              delta(budget, scope, level, m, g, r["method"])]
                out.append(method_label(r["method"]) + " & " + " & ".join(cells) + " \\\\")
            out += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
        # livello item
        out += ["", "\\begin{table}[t]\\centering\\scriptsize",
                f"\\caption{{Items removed by each method in the segment's lists {cap0}. "
                "Reach: share of users with at least one of these items in the top-K (T-HR@K); "
                "Best rank: mean position of the first of these items among reached users (95\\% CI); "
                "Pos.$\\le$10: share of reached users for whom it is within the first 10 positions; "
                "$\\delta$: Cliff's delta on the best rank ($>0$ = the MAB ranks them higher); "
                "Exp$_{lt}$/Exp$_{act}$: per-item exposure disparity (Singh \\& Joachims, 2018).}",
                "\\begin{tabular}{lccccccccc}", "\\toprule",
                " & \\multicolumn{4}{c}{Long-tail} & \\multicolumn{4}{c}{Active} & \\\\",
                "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}",
                "Method & Reach & Best rank & Pos.$\\le$10 & $\\delta$ & Reach & Best rank & Pos.$\\le$10 & $\\delta$ & "
                "Exp$_{lt}$/Exp$_{act}$ \\\\", "\\midrule"]
        for r in R:
            cells = []
            for g in ("lt", "active"):
                cells += [_f(r[f"item_reach_{g}"], 3),
                          f"{_f(r[f'item_bestrank_{g}_mean'], 1)} [{_f(r[f'item_bestrank_{g}_lo'], 1)}, "
                          f"{_f(r[f'item_bestrank_{g}_hi'], 1)}]",
                          _f(r[f"item_top10pos_{g}"], 2), delta(budget, scope, "item", "bestrank", g, r["method"])]
            cells.append(f"{_f(r['disparity'], 2)} [{_f(r['disparity_lo'], 2)}, {_f(r['disparity_hi'], 2)}]")
            out.append(method_label(r["method"]) + " & " + " & ".join(cells) + " \\\\")
        out += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    P = [r for r in rows_pair if r["targets"] == EXPOSURE_TARGETS]
    if P:
        out += ["", "\\begin{table}[t]\\centering\\small",
                "\\caption{Defective exposure removed from the lists of the same affected users: "
                "reference method against the others ($\\Delta E$ = T-DCG before $-$ after; 95\\% bootstrap CI; "
                f"$^*$ Wilcoxon $p<0.05$; {TARGETS_LABEL[EXPOSURE_TARGETS]}).}}",
                "\\begin{tabular}{llllccc}", "\\toprule",
                "Budget & Scope & Segment & Comparison & $\\Delta E_{ref}$ & $\\Delta E_{other}$ & Difference \\\\",
                "\\midrule"]
        for r in sorted(P, key=lambda r: (r["budget"], r["scope"], r["segment"], method_key(r["other"]))):
            star = "$^*$" if np.isfinite(r["p_wilcoxon"]) and r["p_wilcoxon"] < 0.05 else ""
            out.append(f"{r['budget']} & {SCOPE_LABEL.get(r['scope'], r['scope'])} & {SEG_LABEL[r['segment']]} & "
                       f"{method_label(r['ref'])} vs {method_label(r['other'])} & {_f(r['dE_ref'], 3)} & "
                       f"{_f(r['dE_other'], 3)} & {_f(r['diff'], 3)} [{_f(r['diff_lo'], 3)}, {_f(r['diff_hi'], 3)}]{star} \\\\")
        out += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    U = [r for r in rows_ugf if r["userset"] == "aff" and r["metric"].startswith("ndcg")]
    if U:
        out += ["", "\\begin{table}[t]\\centering\\small",
                "\\caption{UGF (Li et al., 2021) on affected users: NDCG(active) $-$ NDCG(long-tail), before and after.}",
                "\\begin{tabular}{lllccc}", "\\toprule",
                "Budget & Scope & Method & Before & After & $\\Delta$ \\\\", "\\midrule"]
        for r in sorted(U, key=lambda r: (r["budget"], r["scope"], method_key(r["method"]))):
            out.append(f"{r['budget']} & {SCOPE_LABEL.get(r['scope'], r['scope'])} & {method_label(r['method'])} & "
                       f"{_f(r['gap_before'])} & {_f(r['gap_after'])} & {r['delta_gap']:+.4f} \\\\")
        out += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(out) + "\n"


def _mcolor(m):
    if m.endswith("_lt"):
        return "C3"
    if m.endswith("_active"):
        return "C0"
    return {"mab_all": "C2", "random": "0.75"}.get(m, "C4" if m.startswith("regressor") else "0.45")


def figures_selection(sel_out, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    S = sel_out.get("selection_scores", [])
    lw = lambda m: 2 if m.startswith("mab_") and m != "mab_all" else 1.2
    TITLE = {"reach": "reach (users reached)", "pos": "ranking position"}
    for budget in sorted({r["budget"] for r in S}):
        R = [r for r in S if r["budget"] == budget]
        scopes = sorted({r["scope"] for r in R})
        for m in SEL_METRICS:
            # boxplot dei percentili
            fig, axes = plt.subplots(len(scopes), 2, figsize=(10, 2.6 * len(scopes) + 0.6), squeeze=False)
            for i, sc in enumerate(scopes):
                meths = sorted({r["method"] for r in R if r["scope"] == sc}, key=method_key)
                for j, g in enumerate(("lt", "active")):
                    ax = axes[i, j]
                    data = [[r[f"pct_{m}_{g}"] for r in R if r["scope"] == sc and r["method"] == me
                             and np.isfinite(r[f"pct_{m}_{g}"])] for me in meths]
                    bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.6)
                    for patch, me in zip(bp["boxes"], meths):
                        patch.set_facecolor(_mcolor(me)); patch.set_alpha(0.55)
                    for k, d in enumerate(data):
                        ax.scatter(np.random.default_rng(k).normal(k + 1, 0.06, len(d)), d, s=4, c="k", alpha=0.3)
                    ax.axhline(50, color="0.6", ls="--", lw=0.8)
                    ax.set_xticks(range(1, len(meths) + 1), [method_label(me) for me in meths], rotation=20, fontsize=8)
                    ax.set_ylim(0, 101)
                    ax.set_ylabel("percentile")
                    ax.set_title(f"{SCOPE_LABEL.get(sc, sc)} — {TITLE[m]} on {SEG_LABEL[g]}", fontsize=9)
            fig.tight_layout()
            fig.savefig(outdir / f"fig_selection_{m}_{budget}.png", dpi=200)
            plt.close(fig)
            # quota dei scelti nel top-X% della classifica
            fig, axes = plt.subplots(len(scopes), 2, figsize=(9, 2.8 * len(scopes) + 0.4), squeeze=False)
            X = np.arange(0, 101)
            for i, sc in enumerate(scopes):
                meths = sorted({r["method"] for r in R if r["scope"] == sc}, key=method_key)
                for j, g in enumerate(("lt", "active")):
                    ax = axes[i, j]
                    for me in meths:
                        v = np.array([r[f"pct_{m}_{g}"] for r in R if r["scope"] == sc and r["method"] == me])
                        v = v[np.isfinite(v)]
                        if len(v):
                            ax.plot(X, [(v >= 100 - x).mean() for x in X], label=method_label(me),
                                    color=_mcolor(me), lw=lw(me))
                    ax.plot(X, X / 100, color="0.6", ls="--", lw=0.8, label="chance")
                    ax.set_xlabel(f"top-X% of candidates by {TITLE[m]} on {SEG_LABEL[g]}")
                    ax.set_ylabel("share of selected defects")
                    ax.set_title(SCOPE_LABEL.get(sc, sc), fontsize=9)
                    ax.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(outdir / f"fig_topx_{m}_{budget}.png", dpi=200)
            plt.close(fig)
    # livello item: T-HR@k per item e quota di utenti raggiunti con il primo item entro la posizione k
    for key, col, ylab, fname, logy in (("thr_curves", "thr_per_item", "T-HR@k per item", "fig_thr_curve", True),
                                        ("item_rank_ecdf", "share_best_rank_le_k",
                                         "share of reached users with best rank ≤ k", "fig_item_rank", False)):
        C = sel_out.get(key, [])
        for budget in sorted({r["budget"] for r in C}):
            R = [r for r in C if r["budget"] == budget]
            scopes = sorted({r["scope"] for r in R})
            fig, axes = plt.subplots(len(scopes), 2, figsize=(9, 2.8 * len(scopes) + 0.4), squeeze=False)
            for i, sc in enumerate(scopes):
                meths = sorted({r["method"] for r in R if r["scope"] == sc}, key=method_key)
                for j, g in enumerate(("lt", "active")):
                    ax = axes[i, j]
                    for me in meths:
                        pts = sorted((r["k"], r[col]) for r in R
                                     if r["scope"] == sc and r["method"] == me and r["segment"] == g)
                        ax.plot([p[0] for p in pts], [p[1] for p in pts], label=method_label(me),
                                color=_mcolor(me), lw=lw(me))
                    if logy:
                        ax.set_yscale("log")
                    ax.set_xlabel("k")
                    ax.set_ylabel(f"{ylab} ({SEG_LABEL[g]})", fontsize=8)
                    ax.set_title(f"{SCOPE_LABEL.get(sc, sc)} — before removal", fontsize=9)
                    ax.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(outdir / f"{fname}_{budget}.png", dpi=200)
            plt.close(fig)


# ============================================================================ #
#  PER IL PAPER: tabella dei valori assoluti, overlap, curve per budget, round del MAB
# ============================================================================ #
ABS_COLS = (  # (metrica in metrics_long, etichetta, metrica dei t-test o None, gruppo, decimali, verso)
    ("ndcg", "NDCG", "ndcg", "acc", 4, "$\\uparrow$"),
    ("hit", "HR", "hit", "acc", 4, "$\\uparrow$"),
    ("coverage", "Coverage", None, "beyond", 4, "$\\uparrow$"),
    ("novelty", "Novelty", "novelty_pu", "beyond", 3, "$\\uparrow$"),
    ("gini", "Gini", None, "beyond", 4, "$\\downarrow$"),
    ("t_ndcg_u", "T-NDCG", "t_ndcg_u", "exp", 4, "$\\downarrow$"),
    ("t_hr_u", "T-HR", "t_hr_u", "exp", 4, "$\\downarrow$"),
    ("t_rank_u", "T-Rank", "t_rank_u", "exp", 1, "$\\uparrow$"))
ABS_GROUPS = (("acc", "Accuracy"), ("beyond", "Beyond-accuracy"), ("exp", "Defective items"))


def _fmt(x, d):
    return "--" if x is None or not np.isfinite(x) else f"{x:.{d}f}"


def _pct(after, before):
    """Variazione percentuale rispetto a Base, tra parentesi (vuota se Base è 0 o non definito)."""
    if before is None or after is None or not np.isfinite(before) or not np.isfinite(after) or before == 0:
        return ""
    return f" $({100 * (after - before) / abs(before):+.1f}\\%)$"


def latex_abs_tables(rows_long, rows_tt, rows_test, k, userset, vlabel, alpha, K_list=None):
    """
    Valori ASSOLUTI, una tabella per (budget, scope): blocchi All / Active / Long-tail, righe Base
    (prima della rimozione) e metodi (dopo). T-NDCG, T-HR e T-Rank sull'UNIONE degli item difettosi dello
    scope (stesso insieme per tutti i metodi: la riga Base ha un solo valore).
      T-Rank  rank medio degli item difettosi che stavano nella top-k del base: Base = dove stavano,
              metodi = dove stanno dopo (fuori dalla top-K salvata = K+1, limite inferiore).
      dagger  metodo dopo vs Base (t-test appaiato sugli utenti, Holm)
      *       MAB del segmento vs metodo, entrambi dopo (t-test appaiato, Holm)
      approx  TOST: NDCG equivalente a Base entro il margine (solo utenti affetti)
    Coverage e Gini: nessun test (calcolati sull'insieme delle liste, nessun valore per utente).
    """
    idx = {(r["budget"], r["scope"], r["method"], r["segment"], r["metric"]): r
           for r in rows_long if r["userset"] == userset}
    ba, rv, _ = _tt_index(rows_tt)
    eq = {(r["budget"], r["scope"], r["method"], r["segment"], r["metric"]): bool(r["equivalent"])
          for r in rows_test if r.get("test") == "TOST"} if userset == "aff" else {}
    ul = USERSET_LABEL[userset]
    out = [f"% generated by targeting_eval.py — absolute values, {ul}, @{k}, {vlabel}",
           "% requires \\usepackage{booktabs,multirow,graphicx}"]
    ncol = len(ABS_COLS)
    cens = f"{K_list + 1}" if K_list else "$K+1$"
    for budget, scope in sorted({(b, sc) for b, sc, *_ in idx}):
        meths = sorted({m for b, sc, m, *_ in idx if (b, sc) == (budget, scope)}, key=method_key)
        body = []
        for g in SEGMENTS:
            if not any((budget, scope, me, g, f"ndcg@{k}") in idx for me in meths):
                continue
            ref = REF_OF_SEGMENT[g]
            n_us = next(idx[(budget, scope, me, g, f"ndcg@{k}")]["n_users"] for me in meths
                        if (budget, scope, me, g, f"ndcg@{k}") in idx)
            base = []
            for m, _, _, _, dg, _ in ABS_COLS:
                r = next((idx[(budget, scope, me, g, f"{m}@{k}")] for me in meths
                          if (budget, scope, me, g, f"{m}@{k}") in idx), None)
                base.append(_fmt(r["before"], dg) if r else "--")
            lines = ["Base & " + " & ".join(base)]
            for me in meths:
                cells = []
                for m, _, tm, _, dg, _ in ABS_COLS:
                    r = idx.get((budget, scope, me, g, f"{m}@{k}"))
                    if r is None:
                        cells.append("--")
                        continue
                    sup = ""
                    if tm:
                        key = (budget, scope, userset, g, f"{tm}@{k}", me)
                        sup += "\\dagger" if _sig(ba.get(key)) else ""
                        sup += "*" if me != ref and _sig(rv.get(key)) else ""
                    if eq.get((budget, scope, me, g, f"{m}@{k}")):
                        sup += "\\approx"
                    cells.append(_fmt(r["after"], dg) + (f"$^{{{sup}}}$" if sup else "") + _pct(r["after"], r["before"]))
                lab = method_label(me)
                lines.append((f"\\textbf{{{lab}}}" if me == ref else lab) + " & " + " & ".join(cells))
            seg = f"{SEG_LABEL[g]} ($n$={n_us})"
            body.append(f"\\multirow{{{len(lines)}}}{{*}}{{{seg}}} & " + lines[0] + " \\\\")
            body += [" & " + l + " \\\\" for l in lines[1:]]
            body.append("\\midrule")
        if not body:
            continue
        body[-1] = ""
        spans, c0 = [], 3
        for gk, gl in ABS_GROUPS:
            w = sum(1 for c in ABS_COLS if c[3] == gk)
            spans.append((c0, c0 + w - 1, gl))
            c0 += w
        cap = (f"Absolute values after removal, {SCOPE_LABEL.get(scope, scope)}, budget {budget}, @{k}, {ul} "
               f"({vlabel}). Base = before removal; in parentheses, relative change with respect to Base. "
               "Arrows give the direction of improvement. "
               "T-NDCG, T-HR and T-Rank are computed on the union of the defective items removed in the scope "
               "(same item set for every method). T-Rank: mean rank of the defective items that were in the "
               f"base top-{k}, averaged over the users who had at least one (Base: their rank before removal; "
               "methods: their rank after removal, so the difference from Base is the rank shift); items "
               f"that leave the saved lists count as rank {cens}, so T-Rank after removal is a lower bound. Gini: concentration of recommendations over items (lower = more "
               f"even). Bold: {_ref_txt()}. $^\\dagger$: different from Base; $^*$: different from "
               f"{_ref_txt()} (paired t-tests over users, Holm correction, "
               f"$\\alpha={alpha}$)"
               + ("; $^\\approx$: equivalent to Base within the TOST margin" if eq else "")
               + ". Coverage and Gini are computed on the whole set of lists and are not tested.")
        out += ["", "\\begin{table*}[t]\\centering\\scriptsize", f"\\caption{{{cap}}}",
                "\\resizebox{\\linewidth}{!}{%",
                "\\begin{tabular}{ll" + "c" * ncol + "}", "\\toprule",
                " & & " + " & ".join(f"\\multicolumn{{{b - a + 1}}}{{c}}{{{gl}}}" for a, b, gl in spans) + " \\\\",
                "".join(f"\\cmidrule(lr){{{a}-{b}}}" for a, b, _ in spans),
                "Segment & Method & " + " & ".join(f"{l}@{k}{ar}" for _, l, _, _, _, ar in ABS_COLS) + " \\\\",
                "\\midrule"] + body + ["\\bottomrule", "\\end{tabular}}", "\\end{table*}"]
    return "\n".join(out) + "\n"


# ── overlap delle liste rimosse ──────────────────────────────────────────────
def _jaccard(a, b):
    a, b = set(map(int, a)), set(map(int, b))
    return len(a & b) / len(a | b) if (a or b) else np.nan


def overlap_analysis(D, outdir):
    """Jaccard tra i metodi dello stesso (budget, scope): difetti scelti e item target.
    Per i tre MAB anche le regioni del diagramma di Venn (difetti)."""
    rows, venn = [], []
    sels = next(iter(D.variants.values()))
    for (budget, scope), ss in sorted(by_budget_scope(sels).items()):
        ss = sorted(ss, key=lambda s: method_key(s.method))
        for i, a in enumerate(ss):
            for b in ss[i + 1:]:
                rows.append(dict(dataset=D.ds, budget=budget, scope=scope, a=a.method, b=b.method,
                                 jaccard_defects=_jaccard(a.chosen, b.chosen)
                                 if a.chosen is not None and b.chosen is not None else np.nan,
                                 jaccard_items=_jaccard(a.targets, b.targets),
                                 n_defects_a=len(a.chosen) if a.chosen is not None else np.nan,
                                 n_defects_b=len(b.chosen) if b.chosen is not None else np.nan))
        mabs = {s.method: set(map(int, s.chosen)) for s in ss
                if s.method in ("mab_all", "mab_active", "mab_lt") and s.chosen is not None}
        if len(mabs) == 3:
            A, C, L = mabs["mab_all"], mabs["mab_active"], mabs["mab_lt"]
            venn.append(dict(dataset=D.ds, budget=budget, scope=scope,
                             only_all=len(A - C - L), only_active=len(C - A - L), only_lt=len(L - A - C),
                             all_active=len((A & C) - L), all_lt=len((A & L) - C),
                             active_lt=len((C & L) - A), all_three=len(A & C & L)))
    write_csv(outdir / "overlap.csv", rows)
    write_csv(outdir / "overlap_venn_mab.csv", venn)
    (outdir / "tables_overlap.tex").write_text(latex_overlap(rows))
    try:
        figure_overlap(rows, outdir)
    except Exception as e:
        log(f"  figure non generate (overlap): {e}")
    return rows


def latex_overlap(rows):
    meths = {x for r in rows for x in (r["a"], r["b"])}
    if all(m in meths for m in ("mab_all",) + SEGMENT_MABS):       # i tre MAB
        pairs = (("mab_all", "mab_active"), ("mab_all", "mab_lt"), ("mab_active", "mab_lt"))
        lab = lambda x, y: f"{method_label(x)[4:]}/{method_label(y)[4:]}"
        cap = ("Overlap between the removal lists of the three MABs (Jaccard index), at the level of "
               "selected defects and of defective items. Low values mean that each MAB selects different "
               "defects.")
    else:                                                          # solo MAB-All: contro ogni altro metodo
        pairs = tuple(("mab_all", m) for m in sorted(meths - {"mab_all"}, key=method_key)) if "mab_all" in meths else ()
        lab = lambda x, y: method_label(y)
        cap = ("Overlap between the removal list of MAB-All and that of each other method (Jaccard index), "
               "at the level of selected defects and of defective items.")
    if not pairs:
        return "% generated by targeting_eval.py: no MAB-All in the selections, no overlap table\n"
    n = len(pairs)
    J = {(r["budget"], r["scope"], r["a"], r["b"]): r for r in rows}
    get = lambda b, sc, x, y: J.get((b, sc, x, y)) or J.get((b, sc, y, x))
    out = ["% generated by targeting_eval.py", "", "\\begin{table}[t]\\centering\\small",
           f"\\caption{{{cap}}}",
           "\\begin{tabular}{ll" + "c" * (2 * n) + "}", "\\toprule",
           f" & & \\multicolumn{{{n}}}{{c}}{{Defects}} & \\multicolumn{{{n}}}{{c}}{{Items}} \\\\",
           f"\\cmidrule(lr){{3-{2 + n}}}\\cmidrule(lr){{{3 + n}-{2 + 2 * n}}}",
           "Budget & Scope & " + " & ".join([lab(x, y) for x, y in pairs] * 2)
           + " \\\\", "\\midrule"]
    for budget, scope in sorted({(r["budget"], r["scope"]) for r in rows}):
        cells = []
        for col in ("jaccard_defects", "jaccard_items"):
            for x, y in pairs:
                r = get(budget, scope, x, y)
                cells.append(_f(r[col], 2) if r else "--")
        out.append(f"{budget} & {SCOPE_LABEL.get(scope, scope)} & " + " & ".join(cells) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(out) + "\n"


def figure_overlap(rows, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for budget in sorted({r["budget"] for r in rows}):
        R = [r for r in rows if r["budget"] == budget]
        scopes = sorted({r["scope"] for r in R})
        fig, axes = plt.subplots(len(scopes), 2, figsize=(10, 3.4 * len(scopes) + 0.4), squeeze=False)
        for i, sc in enumerate(scopes):
            meths = sorted({x for r in R if r["scope"] == sc for x in (r["a"], r["b"])}, key=method_key)
            for j, (col, title) in enumerate((("jaccard_defects", "selected defects"),
                                              ("jaccard_items", "defective items"))):
                M = np.eye(len(meths))
                for r in R:
                    if r["scope"] != sc:
                        continue
                    a, b = meths.index(r["a"]), meths.index(r["b"])
                    M[a, b] = M[b, a] = r[col]
                ax = axes[i, j]
                im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1)
                labs = [method_label(m) for m in meths]
                ax.set_xticks(range(len(meths)), labs, rotation=30, ha="right", fontsize=8)
                ax.set_yticks(range(len(meths)), labs, fontsize=8)
                for a in range(len(meths)):
                    for b in range(len(meths)):
                        if np.isfinite(M[a, b]):
                            ax.text(b, a, f"{M[a, b]:.2f}", ha="center", va="center", fontsize=7,
                                    color="white" if M[a, b] > 0.6 else "black")
                ax.set_title(f"{SCOPE_LABEL.get(sc, sc)} — Jaccard, {title}", fontsize=9)
                fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        fig.savefig(outdir / f"fig_overlap_{budget}.png", dpi=200)
        plt.close(fig)


# ── curve: esposizione e accuratezza in funzione degli archi rimossi (dump, variante prop) ──
def _curve_order(defects, meth):
    """Ordine di rimozione: quello del file (classifica o round del MAB). Stratified è scritto una
    sorgente dopo l'altra: si alternano le sorgenti (quote uguali a ogni budget)."""
    if not meth.startswith("stratified"):
        return list(defects)
    by = {}
    for d in defects:
        by.setdefault(d.source, []).append(d)
    out, i = [], 0
    while any(i < len(v) for v in by.values()):
        for v in by.values():
            if i < len(v):
                out.append(v[i])
        i += 1
    return out


def dump_curves(path, meth, scope, sel, defects, u2i, mrd, prop, lo, ref_uid, ref_iid, n_items,
                union_items, emax, args, dst):
    """Per n_points+1 budget (0 .. emax archi, stessa griglia per tutti i metodi dello scope e budget)
    rimuove il PREFISSO del file con quegli archi, propaga (modello base, grafo ripulito) e salva:
      top, hits  [punti, utenti, Kc]  liste e hit di test di ogni punto: le metriche sugli item difettosi
                                      si calcolano a valle con il pool che si vuole (paper_figures.py)
      ndcg_<k>, t_ndcg_u_<k>, t_hr_u_<k>  come prima (unione degli item dei difetti scelti nello scope)."""
    ks = sorted(k for k in set(args.curve_k) if k <= args.k) or [min(args.k, 20)]
    Kc = max(ks)
    order = _curve_order(defects, meth)
    seen, per_def, cum = set(), [], []
    for d in order:
        new = [e for e in mrd.defect_edges(u2i, d) if e not in seen]
        seen.update(new)
        per_def.append(new)
        cum.append(len(seen))
    cum = np.array(cum, np.int64)
    grid = np.linspace(0, emax, args.curves + 1)
    umask = np.zeros(n_items, bool)
    umask[union_items] = True
    res_e, res_n, last, tops, hits = [], [], None, [], []
    nd, tn, th = ({k: [] for k in ks} for _ in range(3))
    for gpt in grid:
        n = int(np.searchsorted(cum, gpt, side="right"))       # difetti con archi cumulati <= gpt
        if last is not None and n == last:
            continue                                              # file finito: niente punti doppi
        last = n
        edges = [e for blk in per_def[:n] for e in blk]
        ru = np.array([ref_uid.get(u, -1) for u, _ in edges], np.int64)
        ri = np.array([ref_iid.get(i, -1) for _, i in edges], np.int64)
        ok = (ru >= 0) & (ri >= 0)
        res = _prop_lists(prop, lo, ru[ok], ri[ok], Kc)
        res_e.append(len(edges)); res_n.append(n)
        tops.append(np.asarray(res.top[:, :Kc], np.int32))
        hits.append(np.asarray(res.hits[:, :Kc], bool))
        for k in ks:
            acc = accuracy_pu(res.hits, res.npos, k)
            tg = target_pu(res.top, umask, k, int(umask.sum()), "_u")
            nd[k].append(acc[f"ndcg@{k}"]); tn[k].append(tg[f"t_ndcg_u@{k}"]); th[k].append(tg[f"t_hr_u@{k}"])
    tmp = Path(dst).with_name(f"{Path(dst).stem}.{os.getpid()}.tmp.npz")
    np.savez_compressed(tmp, users=prop.users.astype(np.int64), edges=np.array(res_e, np.int64),
                        n_defects=np.array(res_n, np.int64), ks=np.array(ks, np.int64), n_union=int(umask.sum()),
                        method=np.array(meth), scope=np.array(scope), sel=np.array(sel),
                        top=np.stack(tops), hits=np.stack(hits),
                        **{f"ndcg_{k}": np.array(nd[k], np.float32) for k in ks},
                        **{f"t_ndcg_u_{k}": np.array(tn[k], np.float32) for k in ks},
                        **{f"t_hr_u_{k}": np.array(th[k], np.float32) for k in ks})
    tmp.replace(dst)
    log(f"  {sel} [curve]: {len(res_e)} punti fino a {res_e[-1]} archi (griglia 0..{emax}), "
        f"liste top-{Kc} salvate -> {dst}")


def curves_analysis(D, args, outdir):
    """Medie per segmento sugli utenti AFFETTI dello scope e budget (gli stessi di analyze)."""
    cdir = args.out / D.ds / "lists" / "prop" / "curves"
    if not cdir.is_dir():
        return []
    sels = {s.sel: s for s in next(iter(D.variants.values()))}
    rows = []
    for f in sorted(cdir.glob("*.npz")):
        if f.name.endswith(".tmp.npz"):
            continue
        z = np.load(f)
        s = sels.get(str(z["sel"]))
        if s is None:                                   # metodo filtrato o file non caricato
            continue
        pos = np.full(int(max(z["users"].max(), D.users.max())) + 1, -1, np.int64)
        pos[z["users"]] = np.arange(len(z["users"]))
        cols = pos[D.users]
        if (cols < 0).any():
            log(f"  curve {f.name}: utenti diversi da quelli dell'analisi, salto")
            continue
        if "ks" in z:
            arr = {int(k): (z[f"ndcg_{k}"], z[f"t_ndcg_u_{k}"], z[f"t_hr_u_{k}"]) for k in z["ks"]}
        else:                                           # file con un solo k
            arr = {int(z["k"]): (z["ndcg"], z["t_ndcg_u"], z["t_hr_u"])}
        for k, (A_nd, A_tn, A_th) in arr.items():
            for g in SEGMENTS:
                m = s.us[("aff", g)]
                if m.sum() == 0:
                    continue
                c = cols[m]
                for j in range(len(z["edges"])):
                    rows.append(dict(dataset=D.ds, budget=s.budget, scope=s.scope, method=s.method, segment=g,
                                     userset="aff", k=k, n_users=int(m.sum()), point=j,
                                     edges=int(z["edges"][j]), n_defects=int(z["n_defects"][j]),
                                     ndcg=float(A_nd[j][c].mean()), t_ndcg_u=float(A_tn[j][c].mean()),
                                     t_hr_u=float(A_th[j][c].mean())))
    first = {(x["budget"], x["scope"], x["method"], x["segment"], x["k"]): x for x in rows if x["point"] == 0}
    for r in rows:
        r0 = first[(r["budget"], r["scope"], r["method"], r["segment"], r["k"])]
        for m in ("ndcg", "t_ndcg_u", "t_hr_u"):
            r[f"{m}_rel"] = (r[m] - r0[m]) / r0[m] if r0[m] else np.nan
    write_csv(outdir / "curves.csv", rows)
    try:
        figure_curves(rows, outdir)
    except Exception as e:
        log(f"  figure non generate (curves): {e}")
    return rows


def figure_curves(rows, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for budget, scope, k in sorted({(r["budget"], r["scope"], r["k"]) for r in rows}):
        R = [r for r in rows if (r["budget"], r["scope"], r["k"]) == (budget, scope, k)]
        meths = sorted({r["method"] for r in R}, key=method_key)
        fig, axes = plt.subplots(2, 3, figsize=(13, 6.4), sharex=True, squeeze=False)
        for j, g in enumerate(SEGMENTS):
            for i, (col, ylab) in enumerate(((f"t_ndcg_u_rel", f"Δ T-NDCG@{k} (%)  — lower = better"),
                                             (f"ndcg_rel", f"Δ NDCG@{k} (%)"))):
                ax = axes[i, j]
                for me in meths:
                    P = sorted((r["edges"], r[col]) for r in R if r["method"] == me and r["segment"] == g)
                    if P:
                        ax.plot([p[0] / 1000 for p in P], [100 * p[1] for p in P], marker="o", ms=3,
                                color=_mcolor(me), lw=2 if me.startswith("mab_") and me != "mab_all" else 1.3,
                                label=method_label(me))
                ax.axhline(0, color="0.6", lw=0.8, ls="--")
                ax.set_ylabel(ylab, fontsize=8)
                if i == 0:
                    ax.set_title(f"{SEG_LABEL[g]} (affected users)", fontsize=9)
                if i == 1:
                    ax.set_xlabel("training edges removed (thousands)")
        axes[0, 0].legend(fontsize=7)
        fig.suptitle(f"{SCOPE_LABEL.get(scope, scope)} — defective exposure and accuracy vs. edges removed "
                     f"(propagation, budget {budget})", fontsize=10)
        fig.tight_layout()
        fig.savefig(outdir / f"fig_curves_{budget}_{scope}_k{k}.png", dpi=200)
        plt.close(fig)


# ============================================================================ #
def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("phase", choices=["dump", "analyze", "all"])
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--out", type=Path, default=Path("results/targeting"))
    p.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS),
                   help="pattern dei metodi (dopo la normalizzazione mab -> mab_all)")
    p.add_argument("--segment-mabs", action="store_true",
                   help="includi anche i MAB ottimizzati sui segmenti (mab_active, mab_lt) in dump, metriche, "
                        "tabelle e figure; ciascuno è il riferimento del proprio segmento. Default: spenti, "
                        "solo MAB-All, che fa da riferimento ($^*$, grassetto) per tutti i segmenti")
    # dump
    p.add_argument("--removed-dir", default=None)
    p.add_argument("--rre-module", default=None,
                   help="nome del modulo run_removed_eval da importare (es. run_removed_eval_2)")
    p.add_argument("--pattern", default="removed_*.tsv")
    p.add_argument("--scopes", nargs="+", default=None,
                   help="dump: solo i file di questi scope (sottostringa, es. group gsdb). "
                        "ATTENZIONE: cambia la numerazione dei --task, controlla con --list-tasks")
    p.add_argument("--staging-dir", default=None)
    p.add_argument("--k", type=int, default=100, help="lunghezza delle top salvate")
    p.add_argument("--sources", nargs="+", default=list(DEFAULT_SOURCES),
                   help="sorgenti dei difetti candidati: file <ds>.<source>.csv")
    p.add_argument("--no-train", action="store_true", help="retrain: salta i file senza checkpoint")
    p.add_argument("--variants", nargs="+", choices=["retrain", "prop"], default=["retrain", "prop"],
                   help="liste 'dopo': retraining (checkpoint di run_removed_eval) e/o sola propagazione "
                        "degli embedding del modello base sul grafo ripulito")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--task", type=int, default=None,
                   help="dump di un solo pezzo (job array): 0 = base + candidati + controllo prop; "
                        "1..F = retrain del file 1..F; F+1..2F = prop del file 1..F (F = file selezionati)")
    p.add_argument("--list-tasks", action="store_true", help="stampa la mappa dei task del dump ed esce")
    # analyze
    p.add_argument("--ks", nargs="+", type=int, default=[20, 50, 100])
    p.add_argument("--affected-k", type=int, default=20,
                   help="un utente è affetto se ha un item rimosso (dallo scope) nella top-k del modello base")
    p.add_argument("--min-reach-users", type=int, default=5,
                   help="posizione di un candidato calcolata solo se raggiunge almeno questi utenti del segmento")
    p.add_argument("--head-share", type=float, default=0.2)
    p.add_argument("--tr-targets", choices=["union", "own"], default="union",
                   help="item difettosi per TR, lift e confronto appaiato: union = unione degli item rimossi "
                        "nello scope (stessi per tutti i metodi, default), own = item del singolo metodo")
    p.add_argument("--curves", type=int, default=0,
                   help="dump: punti della curva esposizione/accuratezza vs archi rimossi (variante prop; "
                        "es. 10; 0 = niente curve)")
    p.add_argument("--curve-k", nargs="+", type=int, default=[20, 50, 100],
                   help="k delle curve (NDCG@k, T-NDCG@k, T-HR@k), tutti dalla stessa propagazione")
    p.add_argument("--heatmap-ks", nargs="+", type=int, default=None,
                   help="k per cui disegnare le heatmap (default: tutti i --ks); "
                        "vengono aggiunti automaticamente a --ks")
    p.add_argument("--only-tables", action="store_true",
                   help="analyze: solo metriche, t-test e tabelle .tex (tables_abs, tables_k, tables_setup). "
                        "Salta selezione dei candidati, targeting/bootstrap, confronto appaiato, UGF, "
                        "sostituzione, figure, overlap e curve. I CSV di quelle analisi non vengono toccati")
    p.add_argument("--boot", type=int, default=1000)
    p.add_argument("--tag", default=None,
                   help="nome della configurazione: risultati in analysis_<tag>/. Default: affk<affected-k> "
                        "(es. analysis_affk20), così job con soglie diverse non si sovrascrivono")
    p.add_argument("--seed", type=int, default=2020)
    p.add_argument("--tost-margin", type=float, default=0.01, help="margine relativo del TOST (0.01 = ±1%%)")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--baseline", nargs="*", default=[], metavar="SCOPE=METODO",
                   help="baseline per il confronto del TR (default: all=stratified, altri=heuristic)")
    a = p.parse_args()
    a.baseline = dict(x.split("=", 1) for x in a.baseline)
    if a.heatmap_ks:
        a.heatmap_ks = sorted(set(a.heatmap_ks))
        a.ks = list(a.ks) + a.heatmap_ks
    a.ks = sorted(set(a.ks))
    if a.tag is None:
        a.tag = f"affk{a.affected_k}"
    return a


def main():
    args = parse_args()
    set_segment_mabs(args.segment_mabs)
    log("MAB di segmento (mab_active, mab_lt): " + ("inclusi" if args.segment_mabs
                                                     else "esclusi, riferimento = MAB-All (--segment-mabs per includerli)"))
    all_rows = []
    for ds in args.datasets:
        if args.phase in ("dump", "all"):
            dump_dataset(ds, args)
        if args.phase in ("analyze", "all"):
            all_rows += analyze_dataset(ds, args)
    if all_rows:
        name = "targeting_all_datasets" + (f"_{args.tag}" if args.tag else "")
        if len(args.datasets) == 1:
            name += f"_{args.datasets[0]}"          # job array: un file per dataset, nessuna sovrascrittura
        write_csv(args.out / f"{name}.csv", all_rows)


if __name__ == "__main__":
    main()