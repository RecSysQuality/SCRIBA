
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import scipy.sparse as sp
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))   # per trovare removed_eval_core/tables

try:
    import run_my_eval as rc
    import loo_defects as lo
    import make_removed_defects as mrd
except ImportError:
    from recbole_f import run_my_eval as rc
    from recbole_f import loo_defects as lo
    from recbole_f import make_removed_defects as mrd

import removed_eval_core as core
import removed_eval_tables as tables
from removed_eval_core import (MODEL, SEGMENTS, SUBSETS, affected_users, all_subset_metrics, build_env,
                               collect, comparison_group, defect_exposure, delta, find_removed_files,
                               fit_or_load, log, member_ids, overlap_with, per_user, remap,
                               selection_label, selection_name, selection_parts, stage_removed,
                               tail_items)
from removed_eval_tables import load_records, write_summary


# ============================================================================ #
#  PIPELINE PER DATASET
# ============================================================================ #
def run_dataset(ds: str, args) -> List[dict]:
    staging = Path(args.staging_dir)
    out_ds = args.out / ds
    out_ds.mkdir(parents=True, exist_ok=True)
    files = find_removed_files(args.removed_dir, ds, args.pattern)
    if not files:
        log(f"  [{ds}] nessun file {ds}.{args.pattern} in {args.removed_dir}: salto")
        return []
    all_files = list(files)                              # per --exclude-members union
    if args.file_index is not None:                      # un solo file (job array)
        if not 0 <= args.file_index < len(files):
            log(f"  [{ds}] --file-index {args.file_index} fuori range (0..{len(files) - 1}): salto")
            return []
        files = [files[args.file_index]]
    res_dir = out_ds / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    if args.skip_done:                                   # ripresa: salta i file già valutati
        todo = [f for f in files if not (res_dir / f"{selection_name(f, ds)}.json").exists()]
        if len(todo) < len(files):
            log(f"  [{ds}] già valutati, saltati: {[f.name for f in files if f not in todo]}")
        files = todo
        if not files:
            return []
    log(f"  [{ds}] {len(files)} file di difetti rimossi:")
    for f in files:
        log(f"      {f.resolve()}")
    ks = tuple(sorted(set(args.ks)))
    max_k = max(max(ks), args.topk, args.affected_k)

    # ── 1. Modello base ──────────────────────────────────────────────────────
    dataset_dir = rc.DATASETS[ds]
    noise = rc.build_noise_registry(ds)["clean"]
    _, base_tag = rc.stage_dataset(ds, dataset_dir, "clean", "0.0",
                                   noise["levels"]["0.0"], noise["subdir"], staging)
    base = build_env(ds, base_tag, staging, rc.SAVED_DIR / base_tag / MODEL)
    uid_f, iid_f = base.dataset.uid_field, base.dataset.iid_field
    ref_uid = base.dataset.field2token_id[uid_f]
    ref_iid = base.dataset.field2token_id[iid_f]
    uid_tok = base.dataset.field2id_token[uid_f]
    n_users, n_items = base.dataset.user_num, base.dataset.item_num
    catalog = n_items - 1

    model0, info0 = fit_or_load(base, force_train=args.retrain_base)
    res0 = collect(model0, base, max_k, np.arange(n_users), np.arange(n_items))
    graph = None
    if args.mode == "prop":
        # PROPAGAZIONE: parametri del modello base FISSI; dopo la rimozione si ricalcola solo la
        # propagazione di LightGCN sul training ripulito (come l'ambiente del MAB / loo_defects).
        graph = lo.TrainGraph(base.train.dataset, base.cfg["device"])
        lo.propagate(model0, graph.norm_adj(graph.R))     # controllo: grafo intero = modello base
        res_chk = collect(model0, base, max_k, np.arange(n_users), np.arange(n_items))
        same = float((res_chk.top[:, :20] == res0.top[:, :20]).all(1).mean())
        log(f"  [{ds}] modalità PROP: controllo sul grafo intero, top-20 identiche al modello base "
            f"per il {100 * same:.2f}% degli utenti (atteso ~100%)")
        if same < 0.95:
            log("  ATTENZIONE: la propagazione sul grafo intero non riproduce il modello base: "
                "controlla che TrainGraph.norm_adj corrisponda a LightGCN.get_norm_adj_mat")
        del res_chk
    else:
        del model0
    gc.collect(); torch.cuda.empty_cache()
    base_items = base.train.dataset.inter_feat[iid_f].numpy()
    pop0 = np.bincount(base_items, minlength=n_items)
    n_train0 = len(base_items)
    log(f"  [{ds}] base: {len(res0.users)} utenti di test, {catalog} item, {n_train0} interazioni di training")

    masks = {"all": np.ones(n_users, dtype=bool)}
    masks["all"][0] = False
    for seg in ("active", "lt"):
        m = np.zeros(n_users, dtype=bool)
        try:
            m[lo.load_segment(ds, seg, ref_uid)] = True
        except FileNotFoundError as e:
            log(f"  ATTENZIONE: {e}")
        masks[seg] = m
    tail0 = tail_items(pop0, args.head_share)            # long-tail fisso (training base) per tutti i modelli
    kw = dict(ks=ks, topk=args.topk, catalog=catalog, tail=tail0)

    # ── Gruppi di confronto ─────────────────────────────────────────────────
    # Un gruppo = i file confrontati nella stessa tabella: stessa sorgente (rev2, group_f, ...)
    # e stesso budget. Esempio Books/rev2/50K: heuristic, regressor, mab_all, mab_lt, ...
    # Membri esclusi (--exclude-members union) e utenti a rischio (--eval-users risk) si
    # calcolano SOLO sui file del gruppo: stessi utenti per i metodi confrontati tra loro.
    by_group: Dict[tuple, List[Path]] = {}
    for f in all_files:
        by_group.setdefault(comparison_group(selection_name(f, ds)), []).append(f)
    group_defs = {}
    for g_, fs in by_group.items():
        dd_all = []
        for f in fs:
            try:
                dd_all.append(mrd.read_removed(f, ds))
            except TypeError:
                dd_all.append(mrd.read_removed(f))
        group_defs[g_] = dd_all
    log(f"  [{ds}] gruppi di confronto (sorgente, budget): "
        + "; ".join(f"{g_[0]}/{g_[1]}: {[selection_parts(selection_name(f, ds))[0] for f in fs]}"
                    for g_, fs in sorted(by_group.items())))

    union_excl = {}
    if args.exclude_members == "union":                  # stessi utenti valutati per i metodi del gruppo
        for g_, dd_all in group_defs.items():
            ex = np.zeros(n_users, dtype=bool)
            for dd in dd_all:
                ex[member_ids(dd, ref_uid)] = True
            union_excl[g_] = ex
            log(f"  [{ds}] {g_[0]}/{g_[1]}: esclusi i membri di {len(dd_all)} file: "
                f"{int(ex[res0.users].sum())} utenti di test su {len(res0.users)}")

    # ── Utenti a rischio (--eval-users risk) ────────────────────────────────
    # Sul modello BASE: utenti con almeno --risk-min item sospetti nella top---risk-k.
    # Item sospetti = item dei difetti dei file DEL GRUPPO (i metodi confrontati tra loro).
    # Con --risk-min 1 --risk-k 20: utenti affetti da almeno una delle liste del gruppo.
    risk = {}
    if args.eval_users == "risk":
        if args.exclude_members != "union":
            log("  ATTENZIONE: con --eval-users risk usa anche --exclude-members union "
                "(altrimenti gli utenti valutati non sono gli stessi per tutti i metodi)")
        for g_, dd_all in group_defs.items():
            susp = np.zeros(n_items, dtype=bool)
            for dd in dd_all:
                for d in dd:
                    susp[lo.to_ids(",".join(d.items), ref_iid)] = True
            cnt = susp[res0.top[:, :args.risk_k]].sum(1)      # item sospetti per utente di test
            rk = np.zeros(n_users, dtype=bool)
            rk[res0.users[cnt >= args.risk_min]] = True
            risk[g_] = rk
            ok = ~union_excl[g_] if g_ in union_excl else np.ones(n_users, dtype=bool)
            log(f"  [{ds}] {g_[0]}/{g_[1]}: UTENTI A RISCHIO (>= {args.risk_min} item sospetti in "
                f"top-{args.risk_k}, {len(dd_all)} file, {int(susp.sum())} item sospetti): "
                + ", ".join(f"{s_}={int((masks[s_] & rk & ok)[res0.users].sum())}"
                            f"/{int((masks[s_] & ok)[res0.users].sum())}" for s_ in SEGMENTS)
                + "  (a rischio / totale)")

    peru_dir = out_ds / "peruser"                        # metriche per utente (test appaiati)
    peru_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(peru_dir / "base.npz", tokens=uid_tok[res0.users], **per_user(res0, ks))

    # training in token, per trovare gli archi da togliere
    train_src = dataset_dir / noise["subdir"] / noise["levels"]["0.0"]
    train_rows = rc._parse_inter(train_src, want_rating=False)
    u2i = mrd.user_items(train_rows)

    records = []
    for path in files:
        sel = selection_name(path, ds)
        log("=" * 72)
        meth, scope = selection_parts(sel)
        log(f"  {ds} · rimozione {sel}  (metodo={meth}, scope={scope}, tabella='{selection_label(sel)}')")
        log(f"  file: {path.resolve()}")
        log("=" * 72)
        t0 = time.time()
        try:
            defects = mrd.read_removed(path, ds)
        except TypeError:                                # make_removed_defects.py vecchio: read_removed(path)
            defects = mrd.read_removed(path)

        # ── 2. Archi da togliere e utenti affetti ────────────────────────────
        removed = set()
        for d in defects:
            removed.update(mrd.defect_edges(u2i, d))
        keep_rows = [r for r in train_rows if (r[0], r[1]) not in removed]
        n_rows_removed = len(train_rows) - len(keep_rows)
        if n_rows_removed == 0:
            log("  ATTENZIONE: nessuna riga del training rimossa! I token del file non corrispondono "
                "a quelli di rec_train.csv: il retrain sarebbe identico al base.")
        # utenti esclusi dalla valutazione (membri dei difetti), prima e dopo
        if args.exclude_members == "own":
            excl = np.zeros(n_users, dtype=bool)
            excl[member_ids(defects, ref_uid)] = True
        elif args.exclude_members == "union":
            excl = union_excl[comparison_group(sel)]
        else:
            excl = np.zeros(n_users, dtype=bool)
        n_excl = int(excl[res0.users].sum())
        seg_masks = {s_: masks[s_] & ~excl for s_ in SEGMENTS}
        if risk:                                         # valutazione solo sugli utenti a rischio del gruppo
            seg_masks = {s_: m_ & risk[comparison_group(sel)] for s_, m_ in seg_masks.items()}
            log(f"  utenti a rischio del gruppo {comparison_group(sel)}: "
                + ", ".join(f"{s_}={int(seg_masks[s_][res0.users].sum())}" for s_ in SEGMENTS))
        if args.exclude_members != "none":
            log(f"  esclusi dalla valutazione ({args.exclude_members}): {n_excl} membri dei difetti · "
                + ", ".join(f"{s_}={int(seg_masks[s_][res0.users].sum())}" for s_ in SEGMENTS) + " utenti valutati")
        aff_ids = affected_users(res0, defects, ref_uid, ref_iid, n_items,
                                 args.affected_k, args.include_members)
        aff = np.zeros(n_users, dtype=bool)
        aff[aff_ids] = True
        sub = dict(seg_masks)
        sub.update({f"aff_{s}": seg_masks[s] & aff for s in SEGMENTS})
        n_aff = {s: int(sub[f"aff_{s}"][res0.users].sum()) for s in SEGMENTS}
        log(f"  {len(defects)} difetti · {len(removed)} archi ({n_rows_removed} righe di training, "
            f"{100 * n_rows_removed / max(len(train_rows), 1):.2f}% del training) · "
            f"affetti: " + ", ".join(f"{s}={v}" for s, v in n_aff.items()))
        with open(out_ds / f"affected_{sel}.txt", "w") as f:
            f.write("\n".join(uid_tok[aff_ids]))

        # ── 3. Prima ─────────────────────────────────────────────────────────
        before = all_subset_metrics(res0, sub, SUBSETS, pop=pop0, n_train=n_train0, **kw)
        # item toccati dagli archi rimossi: la loro esposizione è l'effetto diretto della rimozione
        dmask = np.zeros(n_items, dtype=bool)
        dmask[[ref_iid[i] for i in {i for _, i in removed} if i in ref_iid]] = True
        for s_ in SUBSETS:
            before[s_].update(defect_exposure(res0, sub[s_][res0.users], dmask, args.topk))

        if args.mode == "prop":
            # ── 4-5. Rimozione + PROPAGAZIONE (parametri fissi, nessun retrain) ──
            t1 = time.time()
            ru = np.array([ref_uid.get(u, -1) for u, _ in removed], dtype=np.int64)
            ri = np.array([ref_iid.get(i, -1) for _, i in removed], dtype=np.int64)
            ok_ = (ru > 0) & (ri > 0)
            coo = graph.R.tocoo()
            keys = coo.row.astype(np.int64) * n_items + coo.col
            keep = ~np.isin(keys, ru[ok_] * n_items + ri[ok_])
            R_mod = sp.csr_matrix((coo.data[keep], (coo.row[keep], coo.col[keep])), shape=graph.R.shape)
            lo.propagate(model0, graph.norm_adj(R_mod))
            res1 = collect(model0, base, max_k, np.arange(n_users), np.arange(n_items))
            info1 = {"mode": "prop", "edges_removed_from_graph": int((~keep).sum()),
                     "elapsed_s": round(time.time() - t1, 1)}
            log(f"  ✓ propagazione: {int((~keep).sum())} archi tolti dal grafo "
                f"({int((~ok_).sum())} coppie non presenti nel modello) in {time.time() - t1:.0f}s")
            if args.novelty_pop == "own":
                pop1 = np.asarray(R_mod.sum(0)).ravel().astype(np.int64)
                n_train1 = int(R_mod.nnz)
            else:
                pop1, n_train1 = pop0, n_train0
            del R_mod, coo, keys, keep
        else:
            # ── 4-5. Rimozione + retrain ─────────────────────────────────────
            tag = re.sub(r"[^A-Za-z0-9_.]+", "_", f"{ds}__rm__{sel}")
            stage_removed(ds, tag, keep_rows, staging)
            env = build_env(ds, tag, staging, rc.SAVED_DIR / tag / MODEL)
            model1, info1 = fit_or_load(env, force_train=not args.skip_existing)
            u_map = remap(env.dataset, env.dataset.uid_field, ref_uid)
            i_map = remap(env.dataset, env.dataset.iid_field, ref_iid)
            res1 = collect(model1, env, max_k, u_map, i_map)
            del model1
            gc.collect(); torch.cuda.empty_cache()
            if args.novelty_pop == "own":
                items1 = i_map[env.train.dataset.inter_feat[env.dataset.iid_field].numpy()]
                pop1, n_train1 = np.bincount(items1, minlength=n_items), len(items1)
            else:
                pop1, n_train1 = pop0, n_train0

        # ── 6. Dopo ──────────────────────────────────────────────────────────
        after = all_subset_metrics(res1, sub, SUBSETS, pop=pop1, n_train=n_train1, **kw)
        np.savez_compressed(peru_dir / f"{sel}.npz", tokens=uid_tok[res1.users], **per_user(res1, ks),
                            **{f"in_{s_}": sub[s_][res1.users] for s_ in SEGMENTS})
        for s_ in SUBSETS:
            m1 = sub[s_][res1.users]
            after[s_].update(defect_exposure(res1, m1, dmask, args.topk))
            after[s_].update(overlap_with(res0, res1, m1, args.topk))

        rec = {"dataset": ds, "selection": sel, "label": selection_label(sel),
               "method": selection_parts(sel)[0], "scope": selection_parts(sel)[1], "file": str(path),
               "n_defects": len(defects),
               "defects_per_source": {s: sum(d.source == s for d in defects)
                                      for s in sorted({d.source for d in defects})},
               "n_edges_removed": len(removed), "n_train_rows_removed": n_rows_removed,
               "n_train_rows": len(train_rows), "n_affected": n_aff,
               "removed_share": n_rows_removed / max(len(train_rows), 1),
               "n_defect_items": int(dmask.sum()),
               "affected_k": args.affected_k, "include_members": args.include_members,
               "exclude_members": args.exclude_members, "n_excluded_test_users": n_excl,
               "eval_users": args.eval_users, "risk_k": args.risk_k, "risk_min": args.risk_min,
               "topk": args.topk, "ks": list(ks), "novelty_pop": args.novelty_pop,
               "head_share": args.head_share, "n_tail_items": int(tail0.sum()),
               "base": info0, "retrained": info1, "mode": args.mode,
               "before": before, "after": after, "delta": delta(after, before),
               "elapsed_s": round(time.time() - t0, 1)}
        records.append(rec)
        tk = args.table_k
        for s in SUBSETS:
            b, a = before[s], after[s]
            if a.get("n_users", 0) and b.get("n_users", 0):
                log(f"    [{s:10s}] n={a['n_users']:6d}  ndcg@{tk} {b[f'ndcg@{tk}']:.4f}→{a[f'ndcg@{tk}']:.4f}  "
                    f"entropy {b['entropy']:.3f}→{a['entropy']:.3f}  "
                    f"coverage {b['coverage']:.4f}→{a['coverage']:.4f}  "
                    f"novelty {b['novelty']:.3f}→{a['novelty']:.3f}  "
                    f"def.exp {b.get('defect_exposure', 0):.4f}→{a.get('defect_exposure', 0):.4f}  "
                    f"overlap {a.get('overlap_base', float('nan')):.3f}")
        rb, ra = info0.get("recbole_test", {}), info1.get("recbole_test", {})
        common_k = [k_ for k_ in rb if k_ in ra]
        if common_k:
            log("    RecBole test  " + "  ".join(f"{k_} {rb[k_]:.4f}→{ra[k_]:.4f}" for k_ in common_k[:6]))
        with open(res_dir / f"{sel}.json", "w") as f:        # un file per selezione: i job non si pestano
            json.dump(rec, f)
    return records


# ============================================================================ #
#  CLI
# ============================================================================ #
def parse_args():
    p = argparse.ArgumentParser(description="Metriche prima/dopo la rimozione dei difetti + retrain",
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--datasets", nargs="+", default=list(rc.DATASETS.keys()))
    p.add_argument("--removed-dir", type=Path, default=rc.BASE_DIR / "removed_defects",
                   help="cartella con i file <ds>.removed_*.tsv (anche in sottocartelle <ds>/)")
    p.add_argument("--pattern", default="*_ablation.tsv",
                   help="quali file usare, dopo '<ds>.' (es. 'removed_random_*.tsv')")
    p.add_argument("--topk", type=int, default=50, help="lunghezza delle liste per entropia/coverage/novelty")
    p.add_argument("--ks", nargs="+", type=int, default=[10, 20, 50, 100],
                   help="cutoff per NDCG/Recall/Hit/MRR e per la beyond-accuracy (<metrica>@k)")
    p.add_argument("--head-share", type=float, default=0.2,
                   help="quota di item più popolari (training base) considerati 'head'; il resto è long-tail (APLT)")
    p.add_argument("--paper-beyond-ks", nargs="+", type=int, default=[20, 100],
                   help="cutoff della tabella beyond-accuracy del paper")
    p.add_argument("--affected-k", type=int, default=20,
                   help="top-k del modello base usata per trovare gli utenti affetti (come loo_defects)")
    p.add_argument("--exclude-members", choices=["none", "own", "union"], default="none",
                   help="toglie dagli utenti valutati (tutti i gruppi, prima e dopo) chi compare nei difetti: "
                        "own = membri dei difetti di quel file; union = membri di tutti i file trovati "
                        "(stessi utenti per tutti i metodi: usare questo per confrontare i metodi). "
                        "Senza --out, i risultati vanno in results/removed_nomembers_<modo>/")
    p.add_argument("--methods", nargs="+", default=None,
                   help="valuta solo questi metodi (es. heuristic regressor stratified mab_all); i file degli "
                        "altri metodi vengono ignorati, anche per l'esclusione dei membri e le tabelle")
    p.add_argument("--paper-segments", nargs="+", choices=["all", "active", "lt"], default=["all", "active", "lt"],
                   help="gruppi di utenti nelle tabelle del paper (es. solo 'all')")
    p.add_argument("--mode", choices=["retrain", "prop"], default="retrain",
                   help="retrain = LightGCN riallenato sul training ripulito; prop = parametri del modello "
                        "base fissi, si ricalcola solo la propagazione sul grafo ripulito (nessun rumore "
                        "di training, come l'ambiente del MAB)")
    p.add_argument("--eval-users", choices=["segment", "risk"], default="segment",
                   help="segment = tutti gli utenti di ogni segmento; risk = solo gli utenti a rischio "
                        "(stessi per tutti i metodi, vedi --risk-k e --risk-min)")
    p.add_argument("--risk-k", type=int, default=10,
                   help="top-K del modello base usata per definire gli utenti a rischio")
    p.add_argument("--risk-min", type=int, default=2,
                   help="numero minimo di item sospetti nella top---risk-k per essere a rischio")
    p.add_argument("--include-members", action="store_true",
                   help="conta come affetti anche gli utenti membri dei difetti")
    p.add_argument("--novelty-pop", choices=["base", "own"], default="base",
                   help="popolarità per la novelty: training base (default) o training di ogni modello")
    p.add_argument("--paper-rank", nargs="+", default=None, metavar="PREFIX",
                   help="tabelle del paper: grassetto/sottolineato solo tra i metodi con questi prefissi "
                        "(es. --paper-rank mab); gli altri restano come riferimento. Default: tutti")
    p.add_argument("--paper-scopes", nargs="+", default=None,
                   help="per quali candidati fare le tabelle del paper (group_f gsdb_f rev2 all); default: tutti")
    p.add_argument("--sig-ref", nargs="+", default=["all=mab_all", "active=mab_active", "lt=mab_lt"],
                   metavar="GRUPPO=METODO",
                   help="riferimento del test appaiato per ogni gruppo di utenti (default: all=mab_all "
                        "active=mab_active lt=mab_lt); 'none' per disattivarlo")
    p.add_argument("--correction", choices=["holm", "none"], default="holm",
                   help="correzione per confronti multipli dentro ogni (gruppo, metrica)")
    p.add_argument("--alpha", type=float, default=0.05, help="soglia del test appaiato")
    p.add_argument("--table-k", type=int, default=20, help="k delle metriche di accuratezza in tabella")
    p.add_argument("--compact-k", type=int, default=100,
                   help="k della tabella compatta (Hit, Recall, Entropy, Coverage, Novelty): paper_compact.tex")
    p.add_argument("--retrain-base", action="store_true",
                   help="riallena anche il modello base invece di usare il checkpoint di run_my_eval")
    p.add_argument("--skip-existing", action="store_true",
                   help="se il modello riallenato ha già un checkpoint lo riusa invece di riallenare")
    p.add_argument("--table-only", action="store_true",
                   help="non allena: rifà tabella e CSV dai risultati salvati in --out")
    p.add_argument("--list-files", action="store_true",
                   help="stampa i file trovati per ogni dataset con il loro indice ed esce")
    p.add_argument("--file-index", type=int, default=None,
                   help="valuta solo il file con questo indice (vedi --list-files): per i job array. "
                        "Default: $SLURM_ARRAY_TASK_ID se definita e --array, altrimenti tutti")
    p.add_argument("--array", action="store_true",
                   help="usa $SLURM_ARRAY_TASK_ID come --file-index")
    p.add_argument("--skip-done", action="store_true",
                   help="salta i file che hanno già un risultato in --out (ripresa dopo un'interruzione)")
    p.add_argument("--staging-dir", default=str(rc.STAGING_DIR))
    p.add_argument("--out", type=Path, default=None,
                   help="default results/removed (o results/removed_nomembers_<modo> con --exclude-members)")
    return p.parse_args()


def main():
    args = parse_args()
    if args.methods:
        core.METHODS_FILTER = set(args.methods)          # letto da keep_method (removed_eval_core)
    tables.PAPER_SEGMENTS = [(s_, lab) for s_, lab in tables.PAPER_SEGMENTS if s_ in args.paper_segments]
    if [x.lower() for x in args.sig_ref] == ["none"]:
        args.sig_refs = None
    else:
        args.sig_refs = dict(x.split("=", 1) for x in args.sig_ref)
    if args.out is None:                                 # risultati separati: non sovrascrivono gli altri
        name = "removed" if args.exclude_members == "none" else f"removed_nomembers_{args.exclude_members}"
        if args.eval_users == "risk":
            name += f"_risk{args.risk_k}x{args.risk_min}"
        if args.mode == "prop":
            name += "_prop"
        if args.methods:                                 # un sottoinsieme di metodi: cartella a parte
            name += "_m-" + "+".join(sorted(args.methods))
        args.out = rc.BASE_DIR / "results" / name
    args.out.mkdir(parents=True, exist_ok=True)
    if args.table_k not in args.ks or args.compact_k not in args.ks:
        args.ks = sorted(set(args.ks) | {args.table_k, args.compact_k})
    if args.array and args.file_index is None:
        args.file_index = int(os.environ["SLURM_ARRAY_TASK_ID"])
    if args.list_files:
        print("dataset\tindex\tmethod\tscope\tlabel\tpath")
        for ds in args.datasets:
            if ds not in rc.DATASETS:
                continue
            files = find_removed_files(args.removed_dir, ds, args.pattern)
            for i, f in enumerate(files):
                sel = selection_name(f, ds)
                meth, scope = selection_parts(sel)
                print(f"{ds}\t{i}\t{meth}\t{scope}\t{selection_label(sel)}\t{f.resolve()}")
            print(f"# {ds}: {len(files)} file → {len(files)} retrain (--array=0-{max(len(files) - 1, 0)})")
        return
    if not args.table_only:
        for ds in args.datasets:
            if ds not in rc.DATASETS:
                log(f"Dataset sconosciuto '{ds}', saltato.")
                continue
            try:
                run_dataset(ds, args)
            except Exception as exc:                     # un dataset fallito non ferma gli altri
                rc.log.exception(f"  ✗ {ds} fallito: {exc}")
    if args.file_index is not None and not args.table_only:
        log("Job su un solo file: tabella non scritta (a job finiti: --table-only)")
        return                                           # più job in parallelo non scrivono la stessa tabella
    write_summary(load_records(args.out), args.out, args.table_k, args.paper_rank,
                  sorted(set(args.ks)), args.paper_beyond_ks, args.sig_refs, args.alpha, args.paper_scopes,
                  args.correction, args.compact_k)


if __name__ == "__main__":
    main()
