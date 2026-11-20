# Dump per l'analisi del targeting dei difetti: fase 'dump' (liste top-K) e fase 'analyze'.
# Uso: python3 dump_for_analysis.py {dump,analyze,all} --datasets ...

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import targeting_report as report
from targeting_core import (
    DEFAULT_METHODS, DEFAULT_SOURCES, SEGMENTS, SPLIT_SUFFIXES, VARIANT_LABEL, _nmean, accuracy_pu, beyond,
    budget_of, by_budget_scope, foe, load_dataset, log, merit_for, method_key, novelty_pu, paired_methods,
    selection, set_segment_mabs, target_pu, target_rank_pu, target_splits, targeting, tost, ttests, ugf,
    want_method, wilcoxon_p, write_csv
)
from targeting_report import (
    figure_curves, figure_overlap, figure_variants, figures, figures_selection, heatmap_values,
    latex_abs_tables, latex_overlap, latex_paper_tables, latex_selection, latex_setup, latex_targeting
)


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


def analyze_dataset(ds, args):
    """Metriche 'dopo' per ogni variante (retrain, prop) in analysis_<tag>/<variante>/;
    selezione (prima della rimozione, uguale per le varianti) in analysis_<tag>/."""
    report.EXPOSURE_TARGETS = args.tr_targets
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


# ============================================================================ #
def parse_args():
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0],
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
