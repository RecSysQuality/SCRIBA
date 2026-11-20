
from __future__ import annotations

import copy
import csv
import json
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np

# la cartella di questo file deve essere nel path (mab_core, mab_env, mab_regressor, mab_report)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mab_core import (GROUPS, LOO_REWARD_COL, PAIRS, STATE_METRICS, Defect, RewardScaler,
                      SegmentLinUCB, _extra, _fmt, _jac, _loo, _sp, _tok, fit_online_features,
                      greedy_fill, offline_raws, pool_edges, rank_norm, read_defect_rows,
                      subsample_defects)
from mab_env import (Environment, Holdout, _log_round, _save_reg, _write_rounds, build_offline_prior,
                     load_offline, measure_holdout, offline_check, parse_args)


# ============================================================================ #
#  EPISODIO ONLINE
# ============================================================================ #
@dataclass
class RunConfig:
    target: str
    policy: str
    seed: int
    budget_per_round: int
    total_budget: int
    rounds: int
    max_per_round: int = 0
    max_overlap: float = 0.5
    probe: int = 20
    probe_update: bool = False
    interactive: bool = False
    joint_effect: bool = True
    online_scale: bool = True
    reg_train_on: str = "chosen"
    sage_every: int = 1
    init_rounds: int = 3
    pool: str = "none"          # none | regressor | random
    pool_size: int = 0          # se > 0: N difetti; altrimenti pool_mult x budget del round (archi)
    pool_mult: float = 3.0
    only_touch: bool = False    # candidati = solo difetti che toccano il target (exposure_target > 0)
    hurdle: bool = False        # theta_g aggiornato solo con difetti che toccano g: E[r_g | tocca g]
    random_accept: bool = False # utente simulato: rimuove un numero casuale (1..tutti) dei difetti scelti
    mab_stop: bool = False      # dal round 1: se nessun candidato ha UCB > 0 l'episodio finisce
    patience: int = 0           # > 0: stop se la cumulata congiunta del target non migliora per N round
    patience_rel: float = 0.001 # miglioramento minimo, relativo al miglior valore raggiunto (0.001 = 0.1%)
    bandit_update: bool = True


def run_episode(env, defects, fb, bandit0, scaler0, cfg: RunConfig, out_dir: Path, log,
                reg0=None, embedder=None, holdout: Optional[Holdout] = None, bandit_cold=None):
    """Un esperimento (target, politica, seed) dal grafo iniziale. Ritorna le righe per round."""
    out_dir.mkdir(parents=True, exist_ok=True)
    env.reset()
    rng = np.random.default_rng(cfg.seed)
    acc_rng = np.random.default_rng(cfg.seed + 7777)   # RNG separato: non altera pool e sonda
    # mab      = LinUCB con il prior offline
    # mab_cold = LinUCB SENZA prior, che nei primi cfg.init_rounds round sceglie a caso
    #            (e impara da quei difetti), poi passa all'UCB
    bandit = copy.deepcopy(bandit0) if cfg.policy == "mab" else \
        copy.deepcopy(bandit_cold) if cfg.policy == "mab_cold" else None
    scaler = copy.deepcopy(scaler0)
    reg = copy.deepcopy(reg0) if reg0 is not None else None
    tgt = cfg.target
    hold_keys = holdout.keys if holdout else set()
    pool = [d for d in defects if d.key not in hold_keys]
    for d in pool:
        d.yhat = {}
    pool_pos = {d.key: k for k, d in enumerate(pool)}
    if cfg.only_touch:                   # item di tutti i difetti concatenati: filtro vettoriale
        it_len = np.array([len(d.items) for d in pool], dtype=np.int64)
        it_cat = np.concatenate([d.items for d in pool]) if pool else np.zeros(0, np.int64)
        it_off = np.r_[0, np.cumsum(it_len)[:-1]] if pool else np.zeros(0, np.int64)

    rec_f = open(out_dir / "recommendations.tsv", "w", newline="", encoding="utf-8")
    rec_w = csv.DictWriter(rec_f, delimiter="\t", lineterminator="\n", fieldnames=[
        "round", "group", "rank", "executed", "source", "defect_id", "cost",
        "mean", "bonus", "ucb", "y_hat", "push"])
    rec_w.writeheader()
    mea_f = open(out_dir / "measurements.tsv", "w", newline="", encoding="utf-8")
    mea_fields = (["round", "kind", "removed", "source", "defect_id", "cost", "n_aff_total"]
                  + [f"{a}_{g}" for a in ("r", "raw", "loc", "n_aff", "pred", "bonus", "y_hat", "push")
                     for g in GROUPS])
    mea_w = csv.DictWriter(mea_f, delimiter="\t", lineterminator="\n", fieldnames=mea_fields, restval="")
    mea_w.writeheader()
    jl = open(out_dir / "rounds.jsonl", "w", encoding="utf-8")
    # tutti i difetti effettivamente rimossi, round per round (scritto e salvato a ogni round)
    rm_f = open(out_dir / "removed_defects.tsv", "w", newline="", encoding="utf-8")
    rm_w = csv.DictWriter(rm_f, delimiter="\t", lineterminator="\n", restval="", fieldnames=(
        ["round", "order", "source", "defect_id", "cost", "n_users", "n_items",
         "edges_before_round", "edges_after_round",
         f"mean_{tgt}", f"bonus_{tgt}", f"ucb_{tgt}", f"y_hat_{tgt}"]
        + [f"r_{g}" for g in GROUPS] + [f"n_aff_{g}" for g in GROUPS]
        + [f"joint_round_{tgt}", f"cum_joint_{tgt}", "users", "items"]))
    rm_w.writeheader()
    reg_dir = out_dir / "regressor"
    if reg is not None:
        reg_dir.mkdir(exist_ok=True)
    reg_log, reg_models = [], []

    env.begin_round()
    ref_top = env.top_all.copy()
    state0 = None
    removed: set = set()
    edges, t = 0, 0
    cum = {g: 0.0 for g in GROUPS}
    cum_joint = {g: 0.0 for g in GROUPS}   # somma degli effetti CONGIUNTI dei batch rimossi
    joint_fallback = 0                     # round in cui il congiunto non c'era (si usa la somma dei singoli)
    stop = {"reason": "rounds", "round": None}
    best_cj, best_round, no_improve, patience_hit = -np.inf, None, 0, False
    prev_theta = bandit.theta.copy() if bandit else None
    prev_mean: Dict[tuple, float] = {}
    aff_mask = np.zeros(env.n_users, bool)      # utenti affetti dai difetti RIMOSSI finora
    emb_cache: Dict[tuple, np.ndarray] = {}
    rows = []

    def state_columns():
        """Metriche di stato sul grafo ATTUALE (su tutti gli utenti e sugli affetti)."""
        nonlocal state0
        seg_state, struct = env.state_metrics(ref_top)
        if state0 is None:
            state0 = seg_state
        row = {f"struct_{k}": v for k, v in struct.items()}
        for g in GROUPS:
            for m in STATE_METRICS:
                row[f"{g}_{m}"] = seg_state[g][m]
                row[f"{g}_d_{m}"] = seg_state[g][m] - state0[g][m]
        # stesse metriche solo sugli utenti affetti dai difetti rimossi finora: grafo attuale
        # vs grafo iniziale, sullo STESSO insieme di utenti (Δ = effetto su chi era esposto)
        aff_masks = {g: env.masks[g] & aff_mask for g in GROUPS}
        if aff_mask.any():
            cur_a = env.metrics_on(aff_masks, env.top_all, ref_top)
            ini_a = env.metrics_on(aff_masks, ref_top, ref_top)
        for g in GROUPS:
            row[f"{g}_aff_n"] = int(aff_masks[g].sum())
            for m in STATE_METRICS:
                v = cur_a[g][m] if aff_mask.any() else np.nan
                row[f"{g}_aff_{m}"] = v
                row[f"{g}_aff_d_{m}"] = v - ini_a[g][m] if aff_mask.any() else np.nan
        return row

    def hold_eval(bnd, rg):
        """Holdout fisso (grafo iniziale, mai usato per allenare): MAB e/o regressore."""
        out = {}
        if holdout is None:
            return out
        if bnd is not None and holdout.X_ctx is not None:
            for g in GROUPS:
                mu_h, gl_h = bnd.ucb(holdout.X_ctx[g], g)[0], holdout.glob(g)
                out[f"sp_hold_{g}"] = _sp(mu_h, gl_h)
                tch = np.array([i["n_aff"][g] for i in holdout.info]) > 0
                if tch.sum() >= 5:       # solo i difetti che toccano g (quelli tra cui si sceglie)
                    out[f"sp_hold_t_{g}"] = _sp(mu_h[tch], gl_h[tch])
        if rg is not None and holdout.X_emb is not None:
            for key, v in rg.evaluate(holdout.X_emb, {g: holdout.loc(g) for g in GROUPS}).items():
                out[f"reg_hold_{key}"] = v
        return out

    # Righe di rounds.tsv:
    #   round = -1  stato INIZIALE (grafo originale, nessuna rimozione)
    #   round = t   scelte e reward del round t + stato DOPO la rimozione dei difetti del round t
    while True:
        t0 = time.time()
        if t > 0:
            env.begin_round()                    # propagazione sul grafo senza i difetti rimossi
        state = state_columns()
        if t == 0:
            rows.append({"round": -1, "initial": 1, "edges_after": 0, **state, **hold_eval(bandit, reg)})
            jl.write(json.dumps(rows[-1], default=float) + "\n")
        else:
            rows[-1].update(state)               # stato dopo la rimozione del round t-1
            jl.write(json.dumps(rows[-1], default=float) + "\n")
            jl.flush()
            _log_round(log, rows[-1], cfg, reg is not None)
        if patience_hit:
            stop.update(reason="pazienza", round=t, best_round=best_round, best_cum_joint=best_cj)
            log.info(f"  Pazienza esaurita al round {t}: cumulata congiunta {tgt} ferma da "
                     f"{cfg.patience} round (migliore {best_cj:.3f} al round {best_round})")
            break
        if edges >= cfg.total_budget or t >= cfg.rounds:
            stop.update(reason="total_budget" if edges >= cfg.total_budget else "rounds", round=t)
            break
        row = {"round": t, "edges_before": edges}
        budget = min(cfg.budget_per_round, cfg.total_budget - edges)
        cands = [d for d in pool if d.key not in removed]

        # ── regressore: embedding sul grafo del round e y_hat ─────────────────
        if reg is not None and embedder is not None:
            if t % cfg.sage_every == 0 or any(d.key not in emb_cache for d in cands):
                E = embedder.embed(cands, env.removed_pairs)
                emb_cache = {d.key: e for d, e in zip(cands, E)}
            E = np.array([emb_cache[d.key] for d in cands])
            yh = {g: rank_norm(v) for g, v in reg.predict(E).items()}   # scala fissa
            for i, d in enumerate(cands):
                d.yhat = {g: float(yh[g][i]) for g in reg.groups}
                if "all" in d.yhat:
                    for g in GROUPS:
                        d.yhat.setdefault(g, d.yhat["all"])

        # ── pool: il MAB (o la baseline) sceglie solo tra N candidati ─────────
        #    regressor = i primi N per y_hat_all (il tuo regressore, sul grafo del round)
        #    random    = N candidati estratti a caso
        #    La sonda viene estratta da TUTTI i candidati rimasti, così il percentile
        #    è confrontabile tra pool diversi.
        n_all = len(cands)
        if cfg.only_touch:
            # limite superiore veloce: almeno un item del difetto nella top-K di un utente del
            # target (il controllo esatto, che esclude i membri, è l'exposure più sotto)
            hits = env.group_item_hits(tgt)
            tp = np.maximum.reduceat(hits[it_cat], it_off) > 0 if len(pool) else np.zeros(0, bool)
            elig = tp[np.array([pool_pos[d.key] for d in cands], dtype=np.int64)] if n_all else np.zeros(0, bool)
        else:
            elig = np.ones(n_all, bool)
        use_pool = cfg.pool != "none"
        pre_probe = set()
        if use_pool:
            # ordine: y_hat del regressore (decrescente) oppure casuale; si prendono i difetti
            # in quest'ordine finché la somma dei loro archi attuali arriva a pool_mult x budget
            # (o i primi pool_size difetti, se dato). Il costo si calcola SOLO per i difetti
            # visitati (prima: per tutti i candidati a ogni round, la parte lenta del round).
            if cfg.pool == "regressor":
                # con --yhat seg il pool è ordinato per l'impatto previsto sul TARGET
                order = np.argsort(-np.array([d.yhat.get(tgt, d.yhat.get("all", 0.0)) for d in cands]),
                                   kind="stable")
            else:
                order = rng.permutation(n_all)
            if cfg.only_touch:
                order = order[elig[order]]
            kk, tot = [], 0
            for j in order:
                c = env.R[cands[j].users][:, cands[j].items].nnz
                if c <= 0:
                    continue
                if cfg.pool_size > 0:
                    kk.append(int(j))
                    if len(kk) >= cfg.pool_size:
                        break
                else:
                    if kk and tot + c > cfg.pool_mult * budget:
                        break
                    kk.append(int(j))
                    tot += c
            keep = set(kk)
            if cfg.probe and cfg.only_touch:     # sonda tra i difetti che toccano il target
                ei = np.flatnonzero(elig)
                pre_probe = set(ei[rng.choice(len(ei), min(cfg.probe, len(ei)), replace=False)].tolist())
            elif cfg.probe:
                pre_probe = set(rng.choice(n_all, min(cfg.probe, n_all), replace=False).tolist())
            use = sorted(keep | pre_probe)
            in_pool = np.array([j in keep for j in use], dtype=bool)
            is_pre_probe = np.array([j in pre_probe for j in use], dtype=bool)
            cands = [cands[j] for j in use]
        elif cfg.only_touch:
            cands = [d for d, e in zip(cands, elig) if e]
        exposure = env.exposure(cands) if (fb.use_exposure or cfg.only_touch) else None
        if cfg.only_touch:                       # controllo esatto: exposure_target > 0
            k = np.flatnonzero(exposure[tgt] > 0)
            cands = [cands[j] for j in k]
            exposure = {g: np.asarray(v)[k] for g, v in exposure.items()}
            if use_pool:
                in_pool, is_pre_probe = in_pool[k], is_pre_probe[k]
        row["n_touch"] = int(elig.sum())
        X, keys, costs, ok, push = fb.transform(env.R, cands, env.masks, exposure)
        cands = [cands[j] for j in ok]
        n = len(cands)
        allowed = in_pool[ok] if use_pool else None
        row["n_cands_total"] = n_all
        row["pool_size"] = int(allowed.sum()) if use_pool else n
        if not n:
            log.info("  Nessun candidato rimasto.")
            stop.update(reason="no_candidates", round=t)
            break

        # ── una lista per OGNI gruppo; si esegue quella del target ───────────
        lists, mean, bonus = {}, {}, {}
        sel_stop = None
        perm = rng.permutation(n)
        random_phase = cfg.policy == "mab_cold" and t < cfg.init_rounds
        row["random_phase"] = int(random_phase)

        # ── ORACOLO: reward vera (LOO) di tutti i candidati del pool, su G_t ──
        if cfg.policy == "oracle":
            t_or = time.time()
            idx_o = np.flatnonzero(allowed) if allowed is not None else np.arange(n)
            true_r = {g: np.full(n, np.nan) for g in GROUPS}
            for j in idx_o:
                info = env.reward(cands[j])
                for g in GROUPS:
                    true_r[g][j] = info["glob"][g]
            row["oracle_n_measured"] = int(len(idx_o))
            row["oracle_seconds"] = round(time.time() - t_or, 1)

        for g in GROUPS:
            if bandit is not None and random_phase:
                lists[g] = greedy_fill(perm, keys, costs, budget, cfg.max_overlap, cfg.max_per_round,
                                       allowed)
                mean[g], bonus[g], _ = bandit.ucb(X[g], g)
            elif bandit is not None:
                lists[g], mean[g], bonus[g] = bandit.select(X[g], g, keys, costs, budget,
                                                            cfg.max_overlap, cfg.max_per_round, allowed,
                                                            force_one=not (cfg.mab_stop and t > 0))
                if g == tgt:
                    sel_stop = bandit.last_stop
            elif cfg.policy == "oracle":  # ← NUOVO RAMO
                sc = true_r[g] / np.maximum(costs, 1)  # reward vera / archi
                ok_o = np.isfinite(sc) & (sc > 0)  # solo difetti utili
                if allowed is not None:
                    ok_o &= allowed
                order = np.argsort(-np.nan_to_num(sc, nan=-np.inf), kind="stable")
                lists[g] = greedy_fill(order, keys, costs, budget, cfg.max_overlap,
                                       cfg.max_per_round, ok_o)
                mean[g], bonus[g] = np.nan_to_num(sc, nan=0.0), np.zeros(n)
            else:
                if cfg.policy == "push":
                    sc = push[g] / np.maximum(costs, 1)
                elif cfg.policy == "regressor":
                    sc = np.array([d.yhat.get(g, 0.0) for d in cands])
                else:
                    sc = np.zeros(n)
                order = perm if cfg.policy == "random" else np.argsort(-sc, kind="stable")
                lists[g] = greedy_fill(order, keys, costs, budget, cfg.max_overlap, cfg.max_per_round,
                                       allowed)
                mean[g], bonus[g] = sc, np.zeros(n)
        chosen = lists[tgt]
        if not chosen:
            if bandit is not None and not random_phase and sel_stop == "saturo":
                log.info(f"  MAB saturo al round {t} dopo {edges} archi rimossi: "
                         f"nessun candidato con UCB > 0 per {tgt}. Episodio finito.")
                stop.update(reason="mab_saturo", round=t)
            else:
                log.info("  Nessun difetto selezionabile con il budget del round.")
                stop.update(reason="no_feasible", round=t)
            break
        for g in GROUPS:
            for rank, j in enumerate(lists[g]):
                d = cands[j]
                rec_w.writerow({"round": t, "group": g, "rank": rank, "executed": int(g == tgt),
                                "source": d.source, "defect_id": d.id, "cost": int(costs[j]),
                                "mean": _fmt(mean[g][j]), "bonus": _fmt(bonus[g][j]),
                                "ucb": _fmt(mean[g][j] + bonus[g][j]),
                                "y_hat": _fmt(d.yhat.get(g, np.nan)), "push": _fmt(push[g][j])})
        rec_f.flush()

        if cfg.interactive:
            for i, j in enumerate(chosen):
                log.info(f"  [{i}] {cands[j].source} id={cands[j].id} archi={costs[j]} "
                         f"media={mean[tgt][j]:.3f} bonus={bonus[tgt][j]:.3f}")
            ans = input("Indici da rimuovere (es. 0,2 | 'all'): ").strip()
            pick = list(range(len(chosen))) if ans in ("all", "") else \
                [int(x) for x in ans.split(",") if x.strip()]
            to_remove = [chosen[i] for i in (pick or [0])]
        elif cfg.random_accept:
            k = int(acc_rng.integers(1, len(chosen) + 1))              # quanti: da 1 a tutti
            pick = np.sort(acc_rng.choice(len(chosen), k, replace=False))  # quali: a caso
            to_remove = [chosen[i] for i in pick]
        else:
            to_remove = list(chosen)

        # ── misure: scelti + sonda casuale (stesso grafo) ────────────────────
        if use_pool:
            probe = np.setdiff1d(np.flatnonzero(is_pre_probe[ok]), chosen)
        else:
            others = np.setdiff1d(np.arange(n), chosen)
            probe = rng.choice(others, min(cfg.probe, len(others)), replace=False) if cfg.probe else []
        meas = [("chosen", int(j), env.reward(cands[j])) for j in chosen]
        meas += [("probe", int(j), env.reward(cands[j])) for j in probe]
        if t == 0 and cfg.online_scale:
            scaler.fit_online({g: [m[2]["glob"][g] for m in meas] for g in GROUPS})
            log.info("  Scala reward online: " + ", ".join(f"{g}={s:.2e}" for g, s in scaler.scale.items()))
        R = {kind: {g: [] for g in GROUPS} for kind in ("chosen", "probe")}
        P = {kind: {g: [] for g in GROUPS} for kind in ("chosen", "probe")}
        rm_set = set(to_remove)
        for kind, j, info in meas:
            d = cands[j]
            rec = {"round": t, "kind": kind, "removed": int(j in rm_set), "source": d.source,
                   "defect_id": d.id, "cost": int(costs[j]), "n_aff_total": info["n_aff_total"]}
            for g in GROUPS:
                r = float(scaler(g, info["glob"][g]))
                R[kind][g].append(r)
                P[kind][g].append(mean[g][j])
                rec.update({f"r_{g}": _fmt(r), f"raw_{g}": f"{info['glob'][g]:.4e}",
                            f"loc_{g}": _fmt(info["loc"][g]), f"n_aff_{g}": info["n_aff"][g],
                            f"pred_{g}": _fmt(mean[g][j]), f"bonus_{g}": _fmt(bonus[g][j]),
                            f"y_hat_{g}": _fmt(d.yhat.get(g, np.nan)), f"push_{g}": _fmt(push[g][j])})
            mea_w.writerow(rec)
        mea_f.flush()
        R = {k: {g: np.array(v) for g, v in d.items()} for k, d in R.items()}
        P = {k: {g: np.array(v) for g, v in d.items()} for k, d in P.items()}

        # ── reward, percentile rispetto alla sonda ──────────────────────────
        rm_mask = np.array([j in rm_set for j in chosen])
        for g in GROUPS:
            rc = R["chosen"][g]
            cum[g] += float(rc[rm_mask].sum())
            row.update({f"r_{g}_sum": rc.sum(), f"r_{g}_mean": rc.mean(), f"cum_{g}": cum[g]})
            pr = R["probe"][g]
            if len(pr) >= 5:
                row[f"pct_{g}"] = float(np.mean([(pr < x).mean() + 0.5 * (pr == x).mean() for x in rc]))
                row[f"top20_{g}"] = float((rc > np.quantile(pr, 0.8)).mean())
                row[f"probe_mean_{g}"] = pr.mean()
        row.update(n_cands=row["pool_size"], n_chosen=len(chosen), n_removed=len(to_remove), budget=budget,
                   cost_chosen=int(costs[chosen].sum()))

        # ── MAB: capacità predittiva (prima dell'aggiornamento) ──────────────
        if bandit is not None:
            for g in GROUPS:
                row[f"sp_probe_{g}"] = _sp(P["probe"][g], R["probe"][g])
                row[f"sp_chosen_{g}"] = _sp(P["chosen"][g], R["chosen"][g])
                row[f"mae_chosen_{g}"] = float(np.abs(P["chosen"][g] - bandit.tr(R["chosen"][g])).mean())
                if len(R["probe"][g]):         # calibrazione: media prevista vs misurata sulla sonda
                    row[f"calib_pred_{g}"] = float(np.mean(P["probe"][g]))
                    row[f"calib_obs_{g}"] = float(np.mean(bandit.tr(R["probe"][g])))
                row[f"sigma_{g}"] = bandit.sigma(g)
            ucb_t = mean[tgt][chosen] + bonus[tgt][chosen]
            row["bonus_mean"] = float(bonus[tgt][chosen].mean())
            row["bonus_frac"] = float(np.mean(bonus[tgt][chosen] / np.maximum(np.abs(ucb_t), 1e-12)))
            row["unc_mean"] = float(bonus[tgt].mean())
            cur = {d.key: m for d, m in zip(cands, mean[tgt])}
            common = [k for k in cur if k in prev_mean]
            if len(common) >= 10:
                row["rank_stability"] = _sp([cur[k] for k in common], [prev_mean[k] for k in common])
            prev_mean = cur
            upd = list(chosen) + (list(probe) if cfg.probe_update else [])
            r_upd = {g: np.r_[R["chosen"][g], R["probe"][g] if cfg.probe_update else []] for g in GROUPS}
            Xu = {g: X[g][upd] for g in GROUPS}
            if cfg.hurdle:                        # theta_g impara solo dai difetti che toccano g
                na = {g: np.array([m[2]["n_aff"][g] for m in meas if m[0] == "chosen"]
                                  + ([m[2]["n_aff"][g] for m in meas if m[0] == "probe"]
                                     if cfg.probe_update else []), float) > 0 for g in GROUPS}
                Xu = {g: Xu[g][na[g]] for g in GROUPS}
                r_upd = {g: np.asarray(r_upd[g])[na[g]] for g in GROUPS}
                for g in GROUPS:
                    row[f"n_upd_{g}"] = int(na[g].sum())
            #bandit.update(Xu, r_upd)
            if cfg.bandit_update:
                bandit.update(Xu, r_upd)
            th = bandit.theta
            n_prev = np.linalg.norm(prev_theta)       # θ=0 (mab_cold al round 0): Δ relativo non definito
            row["theta_delta"] = float(np.linalg.norm(th - prev_theta) / n_prev) if n_prev > 1e-12 else np.nan
            row["theta_norm"] = float(np.linalg.norm(th))
            prev_theta = th.copy()
            row.update(hold_eval(bandit, None))              # holdout: modello a FINE round

        # ── liste per segmento: quanto sono diverse ─────────────────────────
        for a, b in PAIRS:
            row[f"jac_{a}_{b}"] = _jac(lists[a], lists[b])

        # ── regressore: test-then-train ─────────────────────────────────────
        if reg is not None:
            idx = {"chosen": [m[1] for m in meas if m[0] == "chosen"],
                   "probe": [m[1] for m in meas if m[0] == "probe"]}
            infos = {"chosen": [m[2] for m in meas if m[0] == "chosen"],
                     "probe": [m[2] for m in meas if m[0] == "probe"]}
            Xe = {k: np.array([emb_cache[cands[j].key] for j in v]).reshape(len(v), reg.dim)
                  for k, v in idx.items()}
            Yl = {k: {g: np.array([i["loc"][g] for i in infos[k]]) for g in GROUPS} for k in idx}
            for k in ("chosen", "probe"):
                for key, v in reg.evaluate(Xe[k], Yl[k]).items():
                    row[f"reg_{k}_{key}"] = v
            kinds = ["chosen"] + (["probe"] if cfg.reg_train_on == "chosen+probe" else [])
            for k in kinds:
                reg.add(Xe[k], Yl[k])
            reg.update()
            row.update(hold_eval(None, reg))                 # holdout: modello a FINE round
            row["reg_n_online"] = reg.n_online("all")
            for k in ("chosen", "probe"):
                for i, (j, info) in enumerate(zip(idx[k], infos[k])):
                    d = cands[j]
                    reg_log.append({"round": t, "kind": k, "trained": int(k in kinds),
                                    "source": d.source, "defect_id": d.id, "X": Xe[k][i],
                                    **{f"loc_{g}": info["loc"][g] for g in GROUPS},
                                    **{f"glob_{g}": info["glob"][g] for g in GROUPS},
                                    **{f"n_aff_{g}": info["n_aff"][g] for g in GROUPS}})
            reg_models.append({"round": t, "bundles": reg.snapshot()})

        # ── rimozione ed effetto congiunto ───────────────────────────────────
        batch = [cands[j] for j in to_remove]
        je = env.joint_effect(batch) if cfg.joint_effect else None
        for g in GROUPS:
            ind = float(scaler(g, sum(m[2]["glob"][g] for m in meas if m[0] == "chosen" and m[1] in rm_set)))
            row[f"sumind_{g}"] = ind
            if je is not None:
                row[f"joint_{g}"] = float(scaler(g, je["glob"][g]))
                cum_joint[g] += row[f"joint_{g}"]
            else:                                # congiunto non disponibile: ripiego sulla somma dei singoli
                cum_joint[g] += ind
            row[f"cum_joint_{g}"] = cum_joint[g]
        if je is None:
            joint_fallback += 1
        row["joint_fallback"] = joint_fallback
        for d in batch:                          # affetti misurati sul grafo PRIMA della rimozione
            aff_mask[env.affected_users(d)] = True
        n_rm = env.remove(batch)
        edges_before = edges
        edges += n_rm
        removed |= {d.key for d in batch}
        info_by_j = {m[1]: m[2] for m in meas if m[0] == "chosen"}
        for order, j in enumerate(to_remove):
            d, info = cands[j], info_by_j[j]
            u = d.u_tok if len(d.u_tok) else [str(x) for x in d.users]
            i = d.i_tok if len(d.i_tok) else [str(x) for x in d.items]
            rm_w.writerow({
                "round": t, "order": order, "source": d.source, "defect_id": d.id,
                "cost": int(costs[j]), "n_users": len(d.users), "n_items": len(d.items),
                "edges_before_round": edges_before, "edges_after_round": edges,
                f"mean_{tgt}": _fmt(mean[tgt][j]), f"bonus_{tgt}": _fmt(bonus[tgt][j]),
                f"ucb_{tgt}": _fmt(mean[tgt][j] + bonus[tgt][j]),
                f"y_hat_{tgt}": _fmt(d.yhat.get(tgt, np.nan)),
                **{f"r_{g}": _fmt(float(scaler(g, info["glob"][g]))) for g in GROUPS},
                **{f"n_aff_{g}": info["n_aff"][g] for g in GROUPS},
                f"joint_round_{tgt}": _fmt(row.get(f"joint_{tgt}", np.nan)),
                f"cum_joint_{tgt}": _fmt(cum_joint[tgt]),
                "users": ",".join(map(str, u)), "items": ",".join(map(str, i))})
        rm_f.flush()
        if cfg.patience > 0:                     # pazienza sulla cumulata CONGIUNTA del target
            cj = cum_joint[tgt]
            thr = best_cj + cfg.patience_rel * abs(best_cj) if np.isfinite(best_cj) else -np.inf
            if cj > thr:
                best_cj, best_round, no_improve = cj, t, 0
            else:
                no_improve += 1
            row["patience_count"] = no_improve
            patience_hit = no_improve >= cfg.patience
        row.update(edges_after=edges, seconds=round(time.time() - t0, 1))
        rows.append(row)                         # lo stato dopo la rimozione arriva al prossimo giro
        t += 1

    rec_f.close()
    mea_f.close()
    rm_f.close()
    jl.close()
    _write_rounds(rows, out_dir / "rounds.tsv")
    stop.update(edges_removed=int(edges), rounds_done=sum(1 for r in rows if r.get("round", -1) >= 0),
                cum={g: cum[g] for g in GROUPS}, cum_joint={g: cum_joint[g] for g in GROUPS},
                joint_fallback_rounds=joint_fallback, impact=env.impact)
    with open(out_dir / "stop.json", "w") as f:
        json.dump(stop, f, indent=1, default=float)
    log.info(f"  Fine episodio: motivo={stop['reason']} · round {stop['round']} · archi rimossi {edges} · "
             f"cumulata {tgt} {cum[tgt]:.3f} · cumulata congiunta {tgt} {cum_joint[tgt]:.3f}"
             + (f" · round senza congiunto: {joint_fallback}" if joint_fallback else ""))
    if reg is not None:
        _save_reg(reg_log, reg_models, reg_dir)
    if bandit is not None:
        with open(out_dir / "bandit_final.json", "w") as f:
            json.dump({"names": fb.names, **{k: {kk: float(vv) for kk, vv in v.items()}
                                             for k, v in bandit.explain(fb.names).items()}}, f, indent=1)
    return rows


# ============================================================================ #
#  MAIN
# ============================================================================ #
def main():
    ROOT_DIR = Path(__file__).resolve().parent
    EMBEDDINGS_DIR = (ROOT_DIR / "../node_embeddings/defects/").resolve()

    args = parse_args()
    # default solo se non dati da riga di comando (prima sovrascrivevano sempre --emb-dir ecc.)
    args.sage_encoder = args.sage_encoder or (ROOT_DIR / "../node_embeddings/weights/sage_shared.pt").resolve()
    args.emb_dir = args.emb_dir or EMBEDDINGS_DIR
    args.feature_cache = args.feature_cache or (ROOT_DIR / "../mab/features/").resolve()
    REG_DIR = (ROOT_DIR / "../impact_regressor/scriba_regressor_out/final_model.pkl").resolve()
    REG_DIR_LT = (ROOT_DIR / "../impact_regressor/scriba_regressor_out/final_model_lt.pkl").resolve()
    REG_DIR_ACTIVE = (ROOT_DIR / "../impact_regressor/scriba_regressor_out/final_model_active.pkl").resolve()
    #args.regressor = REG_DIR
    #args.regressor_lt = REG_DIR_LT
    #args.regressor_active = REG_DIR_ACTIVE

    if args.defect_type:
        args.defect_files = [args.defect_type]
    if args.target:
        args.targets = [args.target]
    if args.policy:
        args.policies = [args.policy]
    if args.loo_reward_col is None:
        args.loo_reward_col = LOO_REWARD_COL[args.impact]
    if "{g}" not in args.loo_reward_col:
        raise ValueError(f"--loo-reward-col deve contenere {{g}} (es. {LOO_REWARD_COL[args.impact]})")
    print(f"ARGS: {args}")
    lo = _loo()
    log, rc = lo.log, lo.rc
    log.info(f"  Impatto: {args.impact}"
             + (f" (NDCG sul ranking intero, rilevanti = item del difetto; affetti = item del "
                f"difetto in top-{args.topk})" if args.impact == "tndcg"
                else " (1/rank del miglior item del difetto)")
             + f" · reward offline dalla colonna {args.loo_reward_col}")
    staging = Path(args.staging_dir or rc.STAGING_DIR)
    loo_dir = Path(args.loo_dir or rc.BASE_DIR / "results" / "loo")

    use_reg = args.regressor is not None or args.sage_encoder is not None or "regressor" in args.policies \
        or "regressor" in (args.pools or [])
    yhat_mode = args.yhat if use_reg and args.yhat != "none" else None
    reg_groups = list(GROUPS) if yhat_mode == "seg" else ["all"]
    fb_kwargs = dict(extra_cols=list(args.extra_cols), use_seg=not args.no_seg_features,
                     use_exposure=args.use_exposure, yhat_mode=yhat_mode)

    # ── Offline: difetti + reward del LOO (+ y_hat out-of-fold) ─────────────
    off = {n: load_offline(n, staging, loo_dir, args.extra_cols, args.loo_reward_col)
           for n in args.offline_datasets}
    data_off, bundle = None, None
    if use_reg:
        import mab_regressor as mr
        if not args.emb_dir:
            raise ValueError("il regressore richiede --emb-dir")
        data_off = {n: mr.load_emb_pickle(args.emb_dir, n) for n in args.offline_datasets}
        if args.regressor:
            with open(args.regressor, "rb") as f:
                bundle = pickle.load(f)
        seg_bundles = {}
        for g, path in (("active", args.regressor_active), ("lt", args.regressor_lt)):
            if path:
                with open(path, "rb") as f:
                    seg_bundles[g] = pickle.load(f)
        ln = (bundle or {}).get("label_norm", "rank")
        ng, ns = (bundle or {}).get("n_group", 500), (bundle or {}).get("n_gsdb", 500)
        if yhat_mode:
            for g in reg_groups:
                oof = mr.regressor_oof(data_off, list(data_off), mr.LABEL_OF[g], ln, ng, ns)
                for n in off:
                    vals = [oof[n].get(d.key) for d in off[n][2]]
                    ok_v = np.array([v is not None for v in vals], bool)
                    z = np.zeros(len(vals))                 # mancanti = 0 (la mediana)
                    if ok_v.any():
                        z[ok_v] = rank_norm([v for v in vals if v is not None])
                    for d, v in zip(off[n][2], z):
                        d.yhat[g] = float(v)
                    log.info(f"  [offline {n}] y_hat_{g} out-of-fold: {int(ok_v.sum())}/{len(vals)}")
            if yhat_mode == "common":
                for n in off:
                    for d in off[n][2]:
                        for g in GROUPS:
                            d.yhat.setdefault(g, d.yhat["all"])
    raws = offline_raws(off, log)             # la parte lenta, una volta per dataset
    if args.offline_check:
        rows = offline_check(off, fb_kwargs, log, raws)
        out = Path(args.out or rc.BASE_DIR / "results" / "mab") / f"offline_check_{args.impact}.tsv"
        out.parent.mkdir(parents=True, exist_ok=True)
        _write_rounds(rows, out)
    if not args.online_dataset:
        return

    scaler = RewardScaler().fit({g: np.concatenate([off[n][3][g] for n in off]) for g in GROUPS})
    log.info("  Scala reward offline: " + ", ".join(f"{g}={s:.2e}" for g, s in scaler.scale.items()))

    # ── Online: ambiente + candidati ─────────────────────────────────────────
    ds = args.online_dataset
    # una sottocartella per variante: configurazioni che differiscono solo per --yhat,
    # --reg-update, --impact o i filtri non si sovrascrivono (e il REPORT non le mescola)
    variant = (f"yhat-{args.yhat}_upd-{args.reg_update}"
               + (f"_imp-{args.impact}" if args.impact != "rr" else "")
               + (f"_src-{args.defect_type}" if args.defect_type else "")
               + ("_touch" if args.only_touching else "") + ("_hurdle" if args.hurdle else "")
               + ("_racc" if args.random_accept else "")
               + (f"_max{args.max_defects}" if args.max_defects else "")
               + (f"_rt-{args.reward_transform}" if args.reward_transform != "none" else "")
               + ("_stop" if args.mab_stop else "")
               + (f"_pat{args.patience}" if args.patience else "")
               + (f"_{args.tag}" if args.tag else ""))
    OUT_DIR = (ROOT_DIR / f"../mab/results/{ds}/{variant}").resolve()

    args.out = OUT_DIR
    env = Environment(ds, staging, args.topk, args.score_batch, args.test_file, args.head_frac, log,
                      impact=args.impact)
    defects = []
    for s in args.defect_files:
        path = rc.DATASETS[ds] / f"{ds}.{s}.csv"
        if not path.exists():
            log.info(f"  ATTENZIONE: {path} non trovato, tipo {s} saltato")
            continue
        n0 = len(defects)
        for k, row in enumerate(read_defect_rows(path)):
            users, items = lo.to_ids(row["users"], env.uid), lo.to_ids(row["items"], env.iid)
            if len(users) and len(items):
                defects.append(Defect(k, users, items, _extra(row, args.extra_cols), s,
                                      _tok(row["users"]), _tok(row["items"])))
        log.info(f"  [online {ds}] {len(defects) - n0} difetti {s}")
    if args.max_defects:
        defects = subsample_defects(defects, env.R, args.max_defects, tuple(args.keep_all_sources), log=log)
    env.set_defect_items(defects)

    reg0, embedder = None, None
    if use_reg:
        import mab_regressor as mr
        reg0 = mr.OnlineRegressor(data_off, reg_groups, label_norm=ln, n_group=ng, n_gsdb=ns,
                                  mode=args.reg_update, online_weight=args.reg_online_weight,
                                  init_bundle=({"all": bundle, **seg_bundles} if seg_bundles else bundle),
                                  log=log)
        if args.sage_encoder:
            train_csv = args.train_csv or rc.DATASETS[ds] / f"{ds}.train.csv"
            embedder = mr.SageEmbedder(args.sage_encoder, Path(args.emb_dir) / "std_stats.pkl", train_csv, log)
            log.info(f"  GraphSAGE: embedding ricalcolati a ogni round (dim {embedder.dim})")
            pk = Path(args.emb_dir) / f"{ds}.pkl"
            if pk.exists():                      # controllo: round 0 == pickle
                E0 = embedder.embed(defects[:500], set())
                P0 = mr.PickleEmbedder(args.emb_dir, ds).embed(defects[:500])
                okr = np.isfinite(P0).all(1)
                if okr.any():
                    log.info(f"  Controllo embedding round 0 vs pickle: max |diff| = "
                             f"{np.abs(E0[okr] - P0[okr]).max():.2e}")
        else:
            embedder = mr.PickleEmbedder(args.emb_dir, ds)
            log.info("  Embedding statici dal pickle (senza --sage-encoder non cambiano con il grafo)")

    # FeatureBuilder adattato al grafo iniziale (y_hat del modello di partenza)
    env.begin_round()
    if reg0 is not None:
        E = embedder.embed(defects, set())
        yh = {g: rank_norm(v) for g, v in reg0.predict(E).items()}  # scala fissa
        for i, d in enumerate(defects):
            d.yhat = {g: float(yh[g][i]) for g in reg0.groups}
            for g in GROUPS:
                d.yhat.setdefault(g, d.yhat["all"])
    fb = fit_online_features(env, defects, fb_kwargs, args.use_exposure, ds,
                             args.feature_cache, args.fit_sample, log)
    if args.features_only:
        log.info("  --features-only: feature online pronte, esco.")
        return

    bandit_kwargs = dict(lam_shared=args.lam_shared, lam_seg=args.lam_seg, alpha=args.alpha,
                         reward_transform=args.reward_transform)
    log.info(f"  Reward per il LinUCB: {args.reward_transform}"
             + (" (sign(r)*log(1+|r|), solo per imparare)" if args.reward_transform == "slog" else ""))
    bandit0 = build_offline_prior(off, fb_kwargs, scaler, bandit_kwargs, args.warm_weight,
                                  args.warm_neff, log, raws, hurdle=args.hurdle)
    bandit_cold0 = SegmentLinUCB(fb.p, **bandit_kwargs)     # nessun prior
    holdout = measure_holdout(env, defects, args.holdout, args.seeds[0], fb, embedder, log) \
        if args.holdout else None

    # ── budget totale vs archi disponibili nel pool ─────────────────────────
    # pool = candidati selezionabili (holdout escluso, come in run_episode), grafo iniziale
    hold_keys = holdout.keys if holdout else set()
    args.pool_edges = pool_edges(env.R0, [d for d in defects if d.key not in hold_keys])
    src = "/".join(args.defect_files)
    if args.budget_cut > 0 and args.pool_edges < args.total_budget:
        new_budget = max(1, int(args.pool_edges * (1 - args.budget_cut)))
        log.info(f"  Pool {src}: {args.pool_edges} archi distinti < budget totale {args.total_budget}: "
                 f"budget totale ridotto a {new_budget} (-{args.budget_cut:.0%} degli archi del pool)")
        args.total_budget = new_budget
        if args.budget_per_round > args.total_budget:
            log.info(f"  ATTENZIONE: budget per round {args.budget_per_round} > budget totale "
                     f"{args.total_budget}: il primo round può esaurire tutto il budget")
    else:
        log.info(f"  Pool {src}: {args.pool_edges} archi distinti, budget totale {args.total_budget}")

    out_root = Path(args.out) if args.out else rc.BASE_DIR / "results" / "mab" / ds
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "config.json", "w") as f:
        json.dump({k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}, f, indent=1)
    if args.pools is None:
        args.pools = ["regressor"] if reg0 is not None else ["none"]
    log.info(f"  Pool di candidati: {args.pools} (pool = difetti fino a {args.pool_mult} x budget del round)"
             if args.pool_size <= 0 else f"  Pool di candidati: {args.pools} ({args.pool_size} difetti)")
    for target in args.targets:
        for pool in args.pools:
            if pool == "regressor" and reg0 is None:
                log.info("  --pools regressor richiede il regressore (--regressor/--emb-dir): saltato")
                continue
            for policy in args.policies:
                if policy == "regressor" and reg0 is None:
                    continue
                for seed in args.seeds:
                    cfg = RunConfig(target, policy, seed, args.budget_per_round, args.total_budget,
                                    args.rounds, args.max_per_round, args.max_overlap, args.probe,
                                    args.probe_update, args.interactive, not args.no_joint,
                                    not args.no_online_scale, args.reg_train_on, args.sage_every,
                                    args.init_rounds, pool, args.pool_size, args.pool_mult,
                                    only_touch=args.only_touching, hurdle=args.hurdle,
                                    random_accept=args.random_accept, mab_stop=args.mab_stop,
                                    patience=args.patience, patience_rel=args.patience_rel,bandit_update=not args.no_bandit_update)
                    name = f"{target}_{policy}" + (f"_pool-{pool}" if pool != "none" else "")
                    run_dir = out_root / f"{name}_s{seed}"
                    log.info(f"\n=== {ds}: target={target} politica={policy} pool={pool} seed={seed} "
                             f"impatto={args.impact} → {run_dir}")
                    run_episode(env, defects, fb, bandit0, scaler, cfg, run_dir, log, reg0, embedder, holdout,
                                bandit_cold0)
                    with open(run_dir / "run.json", "w") as f:
                        json.dump({"dataset": ds, "impact": args.impact, **cfg.__dict__}, f, indent=1)
    import mab_report
    mab_report.build_report(out_root)


if __name__ == "__main__":
    main()
