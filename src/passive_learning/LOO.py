
from __future__ import annotations

import copy
import json
import os
import random
from collections import defaultdict
from functools import lru_cache
from typing import Literal, Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import time
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.general_recommender import LightGCN
from recbole.quick_start import load_data_and_model
from recbole.trainer import Trainer
from recbole.utils import get_trainer
from recbole.utils.case_study import full_sort_topk, full_sort_scores

import sys

sys.stdout.reconfigure(line_buffering=True)

# ── Paths & seed ──────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
_rng = np.random.default_rng(SEED)
from collections import Counter
target_users = None
# =============================================================================
# Helpers interni
# =============================================================================

def _resolve_internal_ids(uid_series_external, dataset) -> list[int]:
    uid_field = dataset.uid_field
    out = []
    for u in uid_series_external:
        try:
            out.append(dataset.token2id(uid_field, str(u)))
        except ValueError:
            pass
    return out


def _build_uid_to_gt(uid_col: np.ndarray, iid_col: np.ndarray) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for u, i in zip(uid_col, iid_col):
        if u not in mapping:
            mapping[u] = int(i)
    return mapping






# =============================================================================
# [NUOVO] Adj matrix caching  (ottimizzazione B + G)
# =============================================================================

# Dizionario dataset_id → (A_coo, u_all, i_all, N, n_users, n_items, scale)
# Evita di ricostruire A completa per ogni difetto.
_ADJ_CACHE: dict = {}


def _get_or_build_base_adj(dataset) -> dict:
    """
    Costruisce (o recupera da cache) la matrice di adiacenza completa
    D^{-1/2} A D^{-1/2} come COO numpy + metadati utili.
    """
    key = id(dataset)
    if key in _ADJ_CACHE:
        return _ADJ_CACHE[key]

    n_users = dataset.user_num
    n_items = dataset.item_num
    N       = n_users + n_items

    uid_field = dataset.uid_field
    iid_field = dataset.iid_field
    u_all = dataset.inter_feat[uid_field].numpy()
    i_all = dataset.inter_feat[iid_field].numpy()

    row  = np.concatenate([u_all, i_all + n_users])
    col  = np.concatenate([i_all + n_users, u_all])
    data = np.ones(len(row), dtype=np.float32)

    A   = sp.coo_matrix((data, (row, col)), shape=(N, N)).tocsr()
    deg = np.asarray(A.sum(axis=1)).ravel() + 1e-7

    cache = dict(
        A=A, deg=deg, u_all=u_all, i_all=i_all,
        n_users=n_users, n_items=n_items, N=N,
        scale=int(i_all.max()) + 1,
    )
    _ADJ_CACHE[key] = cache
    return cache


# =============================================================================
# Metriche  (ottimizzazione C: max_pool_size)
# =============================================================================

def compute_metrics_on_users(
    uid_series_internal,
    model,
    test_data,
    dataset,topk_np,
    metrics: tuple[str, ...] = ("mrr", "ndcg"),filter=False,
    max_pool_size: int = 100,          # [OTT. C] cap al pool
) -> dict:
    """
    Calcola MRR e/o NDCG in un singolo forward pass.

    max_pool_size limita il pool a un campione fisso di item negativi +
    il pos-item dell'utente. Questo mantiene O(U · max_pool_size) invece
    di O(U · |all_pos_items|), riducendo drasticamente il costo GPU.
    """
    uid_field = dataset.uid_field
    iid_field = dataset.iid_field
    device    = next(model.parameters()).device

    uid_col = test_data.dataset.inter_feat[uid_field].numpy()
    iid_col = test_data.dataset.inter_feat[iid_field].numpy()
    uid_to_gt = _build_uid_to_gt(uid_col, iid_col)

    valid_uids = [u for u in uid_series_internal if u in uid_to_gt]
    if not valid_uids:
        print('not valid')
        return {"mrr": 0.0, "ndcg": 0.0, "details": []}

    # ── [OTT. C] Pool: pos-item di ciascun utente + negativi campionati ───────


    rr_list, ndcg_list, recall_list = [], [], []

    for idx, uid in enumerate(valid_uids):
        pos_item = uid_to_gt[uid]
        ranked   = topk_np[idx]
        hits = np.where(ranked == pos_item)[0]


        if len(hits):
            rank = int(hits[0]) + 1
            rr   = 1.0 / rank
            ndcg = 1.0 / np.log2(rank + 1)
            hit = 1.0
        else:
            rank, rr, ndcg, hit = None, 0.0, 0.0, 0.0

        rr_list.append(rr)
        ndcg_list.append(ndcg)
        recall_list.append(hit)
    print('\n\n')
    return {
        "mrr":     float(np.mean(rr_list))   if "mrr"  in metrics else None,
        "recall":     float(np.mean(recall_list))   if "recall"  in metrics else None,
        "ndcg":    float(np.mean(ndcg_list)) if "ndcg" in metrics else None    }






# =============================================================================
# Gestione difetti / grafo
# =============================================================================

def get_defect_pairs_int(dataset, df_train: pd.DataFrame,
                         defect_group_id) -> set[tuple[int, int]]:
    uid_field = dataset.uid_field
    iid_field = dataset.iid_field
    rows  = df_train[df_train["group_id:token"] == defect_group_id]
    pairs: set[tuple[int, int]] = set()
    for uid_tok, iid_tok in zip(
        rows["user_id:token"].astype(str),
        rows["item_id:token"].astype(str),
    ):
        try:
            u = dataset.token2id(uid_field, uid_tok)
            i = dataset.token2id(iid_field, iid_tok)
            pairs.add((u, i))
        except ValueError:
            pass
    return pairs





# =============================================================================
# Ranking & selezione utenti target
# =============================================================================

def precompute_ranking(model, test_data, dataset, config, k_scan: int = 20,
                       filter_items: Optional[set] = None,chunked = False):
    """
    [OTT. F] Se filter_items è fornito, esegue full_sort_topk solo sugli utenti
    che hanno almeno un item di interesse nel loro set di test,
    anziché su tutto il dataset.
    """
    uid_field     = dataset.uid_field
    iid_field     = dataset.iid_field
    all_users_int = test_data.dataset.inter_feat[uid_field].unique().numpy()
    all_users_int = np.sort(all_users_int)
    chunked = True
    if filter_items:
        # Tieni solo utenti il cui gt-item è tra quelli di interesse
        iid_col   = test_data.dataset.inter_feat[iid_field].numpy()
        uid_col   = test_data.dataset.inter_feat[uid_field].numpy()
        uid_to_gt = _build_uid_to_gt(uid_col, iid_col)
        fi_arr    = np.array(list(filter_items), dtype=np.int64)
        all_users_int = np.array(
            [u for u in all_users_int if uid_to_gt.get(u) in filter_items],
            dtype=np.int64,
        )

    if len(all_users_int) == 0:
        return all_users_int, np.empty((0, k_scan), dtype=np.int64)

    if not chunked:
        _, topk_item_ids = full_sort_topk(
            uid_series=all_users_int,
            model=model,
            test_data=test_data,
            k=k_scan,
            device=config["device"],
        )
        return all_users_int, topk_item_ids.cpu().numpy()

    # ── BATCH MODE ────────────────────────────────────────────────────────────
    all_topk = []
    n_users  = len(all_users_int)
    batch_size = 10000

    c = 0
    for start in range(0, n_users, batch_size):
        batch_users = all_users_int[start : start + batch_size]
        c+=1
        _, topk_item_ids = full_sort_topk(
            uid_series=batch_users,
            model=model,
            test_data=test_data,
            k=k_scan,
            device=config["device"],
        )

        all_topk.append(topk_item_ids.cpu().numpy())

        # Libera memoria GPU esplicitamente ad ogni batch
        del topk_item_ids
        torch.cuda.empty_cache()

    topk_matrix = np.concatenate(all_topk, axis=0)  # shape: [n_users, k_scan]
    return all_users_int, topk_matrix


def calculate_coverage(recs_list,num_total_items):
    # catalogue = build_catalogue(recs_list)
    all_recommended_items = set([item for _, items in recs_list for item in items])
    return len(all_recommended_items) / num_total_items


def calculate_novelty(recs_list, df, dataset):
    # Estrai tutti gli item raccomandati

    item_users = df.groupby("item_id:token")["user_id:token"].nunique()
    total_users = df["user_id:token"].nunique()
    item_popularity = item_users / total_users
    uid_field = dataset.uid_field
    iid_field = dataset.iid_field
    novelty_scores = []

    for _, items in recs_list:
        for item in items:
            item = dataset.id2token(iid_field, item)
            # skip item sconosciuti
            if item not in item_popularity:
                continue

            # popularity globale
            pop = item_popularity[item]

            # evita log(0)
            pop = max(pop, 1e-12)

            novelty_scores.append(-np.log2(pop))

    return np.mean(novelty_scores)


def calculate_entropy(recs_list, num_total_items):
    # Estrai solo gli item consigliati (la seconda parte della tupla)
    all_recommended_items = [item for _, items in recs_list for item in items]
    #
    # # Conta frequenze
    counts = Counter(all_recommended_items)
    #all_recommended_items = np.concatenate([items for _, items in recs_list])
    #_, counts = np.unique(all_recommended_items, return_counts=True)




    # Calcola probabilità (p_i)
    total_recs = len(all_recommended_items)
    probs = np.array([count / total_recs for count in counts.values()])
    #probs = counts / total_recs
    # Entropia = -sum(p * log2(p))
    # Aggiungiamo un piccolo epsilon per evitare log(0)
    entropy = -np.sum(probs * np.log2(probs + 1e-12))
    return entropy

def build_catalogue(recs):
    cat = set()
    for _, items in recs:
        cat.update(items)
    return cat

def entropy_stats(base_recs, loo_recs):
    catalogue = build_catalogue(base_recs + loo_recs)

    H_base = calculate_entropy(base_recs, num_total_items=len(catalogue))
    H_loo = calculate_entropy(loo_recs, num_total_items=len(catalogue))

    delta = H_loo - H_base
    return H_base, H_loo, delta



from collections import defaultdict

def get_target_users_by_rank(
    defect_group_id,
    df_train: pd.DataFrame,
    df_noise: pd.DataFrame,
    dataset,
    all_users_int: np.ndarray,
    topk_np: np.ndarray,
    top_users: int = 100,
    return_all: bool = False,
) :
    uid_field = dataset.uid_field
    iid_field = dataset.iid_field

    if return_all:
        return [dataset.id2token(uid_field, uid) for uid in all_users_int]

    noise_df = (
        df_noise[df_noise["group_id"].isin(defect_group_id)]
        [["user_id", "item_id"]]
        .drop_duplicates()
    )
    users_noise = noise_df["user_id"].unique()
    mask = df_train["user_id:token"].isin(users_noise)
    # mask = df_train.set_index(
    #     ["user_id:token", "item_id:token"]
    # ).index.isin(
    #     noise_df.set_index(["user_id", "item_id"]).index
    # )
    if len(defect_group_id) > 1:
        print('target')
        mask &= (df_train["casistica:token"] == "target")
    # if len(defect_group_id) > 1:
    #     mask = (
    #                df_train.set_index(
    #                    ["user_id:token", "item_id:token"]
    #                ).index.isin(
    #                    noise_df.set_index(["user_id", "item_id"]).index
    #                )
    #            ) & (
    #                    df_train["casistica:token"] == "target"
    #            )

    defect_items_int: set[int] = set()
    #print(df_train[mask]["item_id:token"].unique().tolist())
    #print(df_train[mask]["user_id:token"].unique().tolist())
    for ext_id in df_train[mask]["item_id:token"].astype(str):
        try:
            defect_items_int.add(dataset.token2id(iid_field, ext_id))
        except ValueError:
            pass


    defect_arr  = np.array(list(defect_items_int), dtype=np.int64)

    hit_matrix  = np.isin(topk_np, defect_arr)          # (U, K) bool
    has_hit     = hit_matrix.any(axis=1)                 # (U,) bool
    first_hit   = np.where(hit_matrix, np.arange(topk_np.shape[1]), topk_np.shape[1]).min(axis=1)

    valid_mask  = has_hit
    sorted_idx  = np.argsort(first_hit[valid_mask])
    if len(defect_group_id) > 1:
        sorted_users = all_users_int[valid_mask][sorted_idx]
    else:
        sorted_users = all_users_int[valid_mask][sorted_idx][:50000]


    target_users = [dataset.id2token(uid_field, uid) for uid in sorted_users]
    tgt = 'user'
    if tgt == 'item':
        ITEM = "item_id:token"
        USER = "user_id:token"

        df = df_train[df_train[USER].isin(target_users)][[USER, ITEM]]

        item_counts = df_train[ITEM].value_counts()

        q25 = item_counts.quantile(0.25)
        q50 = item_counts.quantile(0.50)
        q70 = item_counts.quantile(0.70)

        low_items = set(item_counts[item_counts <= q25].index)
        mid_items = set(item_counts[item_counts < q50].index)
        high_items = set(item_counts[item_counts >= q70].index)

        # LOW USERS
        mask_low = df[ITEM].isin(low_items)
        bad_low_users = df.loc[~mask_low, USER].unique()
        low_users = np.setdiff1d(df[USER].unique(), bad_low_users)

        # fallback
        if len(low_users) < 10:
            mask_mid = df[ITEM].isin(mid_items)
            bad_mid_users = df.loc[~mask_mid, USER].unique()
            low_users = np.setdiff1d(df[USER].unique(), bad_mid_users)

        # HIGH USERS
        mask_high = df[ITEM].isin(high_items)
        bad_high_users = df.loc[~mask_high, USER].unique()
        high_users = np.setdiff1d(df[USER].unique(), bad_high_users)





    else:
        ITEM = "item_id:token"
        USER = "user_id:token"

        #df = df_train[df_train[USER].isin(target_users)][[USER, ITEM]]

        # interazioni per utente (su tutto il training, non solo target)
        user_counts = df_train.groupby(USER).size()

        q25 = user_counts.quantile(0.25)
        q50 = user_counts.quantile(0.50)
        q70 = user_counts.quantile(0.70)

        # low = poche interazioni (long-tail users)
        low_users = set(user_counts[user_counts <= q25].index)
        mid_users = set(user_counts[user_counts <= q50].index)
        high_users = set(user_counts[user_counts >= q70].index)

        # intersezione con target_users
        low_users = np.array([u for u in target_users if u in low_users])
        high_users = np.array([u for u in target_users if u in high_users])
        mid_users = np.array([u for u in target_users if u in mid_users])
        # if len(low_users)<10:
        #     low_users = mid_users


    if len(defect_group_id) == 1:
        high_users = high_users[:100]
        low_users = low_users[:100]
        sorted_users = sorted_users[:5000]
    else:
        print('dento')
        high_users = high_users[:1000]
        low_users = low_users[:1000]
        sorted_users = sorted_users[:10000]
        
    high_users_ids = [dataset.token2id(uid_field, uid) for uid in high_users]
    low_users_ids = [dataset.token2id(uid_field, uid) for uid in low_users]
    target_users = [dataset.id2token(uid_field, uid) for uid in sorted_users]
    return [target_users,high_users,low_users],[sorted_users,high_users_ids,low_users_ids]


# =============================================================================
# Dataset helpers
# =============================================================================

def load_dataset(path: str, out_dir: str, save: bool = True,
                 exclude_defect: list = [],target_users: list = []) -> pd.DataFrame:
    df = pd.read_csv(path)
    st = time.time()

    if "group_id"  not in df.columns:
        df["group_id"]  = "original"
    if "casistica" not in df.columns:
        df["casistica"] = "original"

    df = df[["user_id", "item_id", "rating", "timestamp", "group_id", "casistica"]]
    if exclude_defect:
        print(f'dimensioni vecchied: {len(df)}')
        try:
            df_defects = pd.read_csv(f"{PARENT_DIR}/data/noisy/{dataset_name}_5/groups/merged_filtered.csv")
            #df_defects = pd.read_csv(f"{PARENT_DIR}/data/noisy/{dataset_name}_5/injected_noise_new.csv")

        except Exception as e:
            df_defects = pd.read_csv(f"{PARENT_DIR}/data/noisy/{dataset_name}_5/injected_noise_new.csv")

        user_defects = df_defects[df_defects['group_id'].isin(exclude_defect)]['user_id'].unique().tolist()
        df = df[~df["user_id"].isin(user_defects)]
        print(f'dimensioni nuove: {len(df)}')


    if len(target_users) > 0:
        df = df[df["user_id"].isin(target_users)]
    df = df.rename(columns={
        "user_id":   "user_id:token",
        "item_id":   "item_id:token",
        "rating":    "rating:float",
        "timestamp": "timestamp:float",
        "group_id":  "group_id:token",
        "casistica": "casistica:token",
    })
    df["timestamp:float"] = (
        pd.to_datetime(df["timestamp:float"]).astype("int64") // 10 ** 9
    )
    if save:
        df.to_csv(out_dir, index=False, sep="\t")
    return df


# =============================================================================
# Modello
# =============================================================================


def build_catalogue(recs):
    cat = set()
    for _, items in recs:
        cat.update(items)
    return cat

def train_base_model(config_path: str, dataset_name: str,local=False,relevance=False):
    config = Config(
        model="LightGCN",
        dataset=dataset_name,
        config_file_list=[config_path],
        config_dict={"data_path": f"{BASE_DIR}/dataset"},
    )

    dataset    = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    model   = LightGCN(config, train_data.dataset).to(config["device"])
    trainer = Trainer(config, model)
    if not os.path.exists(f"{PARENT_DIR}/checkpoints/{dataset_name}/LightGCN/base_weights_new.pth"):
        print('NOT found model')
        trainer.fit(train_data, valid_data=None, saved=True, show_progress=True)
        base_weights = copy.deepcopy(model.state_dict())
        os.makedirs(f"{PARENT_DIR}/checkpoints/{dataset_name}/", exist_ok=True)
        os.makedirs(f"{PARENT_DIR}/checkpoints/{dataset_name}/LightGCN/", exist_ok=True)
        torch.save(base_weights, f"{PARENT_DIR}/checkpoints/{dataset_name}/LightGCN/base_weights_new.pth")
    else:
        print('found model')

        print('found model')
        checkpoint_path = f"{PARENT_DIR}/checkpoints/{dataset_name}/LightGCN/base_weights_new.pth"
        checkpoint = torch.load(checkpoint_path, map_location=config["device"])
        try:
            base_weights = checkpoint['state_dict']
        except Exception as e:
            base_weights =checkpoint

        model.load_state_dict(base_weights)
        trainer.saved_model_file = checkpoint_path



        #checkpoint_path = f"{PARENT_DIR}/checkpoints/{dataset_name}/LightGCN/base_weights_new.pth"
        # torch.save({
        #     'state_dict': model.state_dict()
        # }, checkpoint_path)
        #checkpoint = torch.load(checkpoint_path, map_location=config["device"])
        #base_weights = checkpoint
        #model.load_state_dict(base_weights)
        #trainer.saved_model_file = checkpoint_path
        #global_t = "global" if local == False else "local"
        #base_weights = checkpoint["state_dict"]
        # if relevance:
        #
        #     result = trainer.evaluate(valid_data)
        #     print(f"BASE {global_t} NDCG@20:   {result['ndcg@20']:.4f}")
        #     print(f"BASE {global_t} Recall@20: {result['recall@20']:.4f}")
        #     print(f"BASE {global_t} MRR@20: {result['mrr@20']:.4f}")


    return (model, config, dataset, train_data, valid_data, test_data,
            base_weights, trainer.saved_model_file)


def load_and_evaluate(checkpoint_path: str, valid_data=None):
    config, model, dataset, train_data, valid_data_loaded, test_data = \
        load_data_and_model(model_file=checkpoint_path)
    if valid_data is None:
        valid_data = valid_data_loaded

    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    result  = trainer.evaluate(valid_data)
    print(f"NDCG@20:   {result['ndcg@20']:.4f}")
    print(f"Recall@20: {result['recall@20']:.4f}")
    return result, config, model, train_data, valid_data, test_data


def remap_weights(base_weights: dict, old_dataset, new_dataset) -> dict:
    new_weights = {k: v.clone() for k, v in base_weights.items()}
    for field, key in [
        (old_dataset.uid_field, "user_embedding.weight"),
        (old_dataset.iid_field, "item_embedding.weight"),
    ]:
        if key not in base_weights:
            continue
        old_id2token  = old_dataset.field2id_token[field]
        new_token2id  = new_dataset.field2token_id[field]
        old_emb       = base_weights[key]
        n_new         = len(new_token2id)
        new_emb       = torch.zeros(n_new, old_emb.shape[1], device=old_emb.device)
        for old_id, token in enumerate(old_id2token):
            if token in new_token2id:
                new_emb[new_token2id[token]] = old_emb[old_id]
        new_weights[key] = new_emb
    return new_weights


def remove_defect_from_dataset(config, dataset_name: str, defect_group_id: list,target_users = []):
    train_path = f"{PARENT_DIR}/data/noisy/{dataset_name}_5/dataset_dirty_new.csv"
    vali_path = f"{PARENT_DIR}/data/original/split/{dataset_name}/vali_{dataset_name}_5.csv"
    test_path = f"{PARENT_DIR}/data/original/split/{dataset_name}/test_{dataset_name}_5.csv"

    out_dir   = f"{BASE_DIR}/dataset/{dataset_name}_tmp_13/{dataset_name}_tmp_13.train.inter"
    out_v_dir = f"{BASE_DIR}/dataset/{dataset_name}_tmp_13/{dataset_name}_tmp_13.valid.inter"
    out_t_dir = f"{BASE_DIR}/dataset/{dataset_name}_tmp_13/{dataset_name}_tmp_13.test.inter"
    os.makedirs(os.path.dirname(out_dir), exist_ok=True)

    load_dataset(train_path, out_dir, exclude_defect=defect_group_id,save=True)
    load_dataset(vali_path,  out_v_dir,save=True,target_users=target_users)
    load_dataset(test_path,  out_t_dir,save=True,target_users=target_users)

    new_config = copy.deepcopy(config)
    new_config["dataset"] = f"{BASE_DIR}/dataset/{dataset_name}_tmp_13/{dataset_name}_tmp_13"
    new_dataset = create_dataset(new_config)
    # if len(defect_group_id) > 1:
    #     torch.save(dataset, f"{BASE_DIR}/dataset/{dataset_name}_{len(defect_group_id)}.pt")
    # else:
    #     torch.save(dataset, f"{BASE_DIR}/dataset/{dataset_name}_{defect_group_id[0]}.pt")

    return data_preparation(new_config, new_dataset)


# =============================================================================
# LOO impact  (ottimizzazione D + E)
# =============================================================================


def loo_impact(
    dataset_name: str,
    defects: list,
    base_weights: dict,
    config,
    train_data,
    valid_data,
    df_train: pd.DataFrame,
    all_users_int: np.ndarray,
    topk_np: np.ndarray,
        target_users,target_id,
    base_model: Optional[LightGCN] = None,
    method: Literal["influence", "local", "warm"] = "influence", n=0, # [OTT. E]
) -> tuple[dict, dict]:
    """
    LOO impact.

    method="influence"  → influence_unlearning  O(D·d)      veloce, approssimato
    method="local"      → local_unlearn_recbole O(k·E·d)    esatto, lento

    Usa "influence" per lo screening su tutti i difetti, poi "local" solo
    sul sottoinsieme top-K che ti interessa davvero.
    """


    # ── 1. Base model (riusato se fornito) ────────────────────────────────────
    if base_model is None:
        base_model = LightGCN(config, train_data.dataset).to(config["device"])
        base_model.load_state_dict(base_weights)


    # ── 2. LOO model ──────────────────────────────────────────────────────────
    loo_model = LightGCN(config, train_data.dataset).to(config["device"])
    loo_model.load_state_dict(base_weights)
    # defect_pairs_int = set()
    # for defect in defects:
    #     defect_pairs_int.update(get_defect_pairs_int(
    #         train_data.dataset, df_train, defect["group_id"]
    #     ))
    # if len(defect_pairs_int) == 0:
    #     print(f"  Nessuna coppia trovata skip.")
    #     return 0.0, 0
    results_obj = {}
    results_obj['defect_id'] = defects_list[0]
    if method == 'warm':
        if base_model is None:
            base_model = LightGCN(config, train_data.dataset).to(config["device"])
            base_model.load_state_dict(base_weights)



        st = time.time()
        ft_config = copy.deepcopy(config)
        ft_config["epochs"] = 3
        if dataset_name == 'Books':
            ft_config["epochs"] = 1

        ft_config["train_batch_size"] = 8192 #1024
        #if dataset_name == 'Sports_and_Outdoors':


        # print(ft_config)
        ft_config["learning_rate"] = 5e-4
        ft_config["stopping_step"] = 5
        # ── 2. Dataset LOO (senza il difetto) ─────────────────────────────────────
        defec = [d['group_id'] for d in defects]




        train, valid, _ = remove_defect_from_dataset(
            ft_config, dataset_name, defec
        )

        remapped = remap_weights(base_weights, train_data.dataset, train.dataset)

        loo_model = LightGCN(config, train.dataset).to(config["device"])
        loo_model.load_state_dict(remapped)


        trainer = get_trainer(ft_config["MODEL_TYPE"], ft_config["model"])(ft_config, loo_model)

        trainer.fit(train, valid_data=None, saved=True, show_progress=True)

        loo_model.eval()
        all_users_int_loo, topk_np_loo = precompute_ranking(
                            model=loo_model,
                            test_data=valid_data,
                            dataset=valid_data.dataset,
                            config=ft_config,
                            k_scan=k_scan,
                            filter_items=None
                        )
        num_total_items = len(np.arange(1, dataset.item_num, dtype=np.int64))

        if len(defects) > 1:
            results_obj['global'] = {}
            base_recs = list(zip(all_users_int, topk_np[:,:20]))
            loo_recs = list(zip(all_users_int_loo, topk_np_loo[:,:20]))
            H_base, H_loo, delta = entropy_stats(base_recs, loo_recs)
            print(f"GLOBAL defect type: global {len(tid)}")

            print(f"\n[{n} - GLOBAL]")
            print(f"Entropia Base: {H_base:.4f}")
            print(f"Entropia LOO:  {H_loo:.4f}")
            print(f"Delta Entropia: {delta:.4f}")
            results_obj['global']['entropy'] = {'base':H_base,'loo':H_loo,'delta':delta}
            coverage_base = calculate_coverage(recs_list=base_recs, num_total_items=num_total_items)
            coverage_loo = calculate_coverage(recs_list=loo_recs, num_total_items=num_total_items)
            delta = coverage_loo - coverage_base
            print(f"Coverage Base: {coverage_base:.4f}")
            print(f"Coverage LOO:  {coverage_loo:.4f}")
            print(f"Delta Coverage: {delta:.4f}\n")
            results_obj['global']['coverage'] = {'base':coverage_base,'loo':coverage_loo,'delta':delta}

            novelty_base = calculate_novelty(recs_list=base_recs, df=df_train, dataset=dataset)
            novelty_loo = calculate_novelty(recs_list=loo_recs, df=df_train, dataset=dataset)
            delta_nov = novelty_loo - novelty_base
            print(f"Novelty Base: {novelty_base:.4f}")
            print(f"Novelty LOO:  {novelty_loo:.4f}")
            print(f"Delta Novelty: {delta_nov:.4f}\n")
            results_obj['global']['novelty'] = {'base':novelty_base,'loo':novelty_loo,'delta':delta_nov}

            res20loo = compute_metrics_on_users(all_users_int, loo_model, valid_data, dataset, topk_np_loo[:, :20],('mrr', 'ndcg', 'recall'))
            print(f"GLOBAL@20 LOO:{res20loo}")
            res20base = compute_metrics_on_users(all_users_int, model, valid_data, dataset, topk_np[:, :20],
                                                  ('mrr', 'ndcg', 'recall'))
            print(f"GLOBAL@20 BASE: {res20base}\n")


            res100loo = compute_metrics_on_users(all_users_int, loo_model, valid_data, dataset, topk_np_loo[:, :100],
                                                ('mrr', 'ndcg', 'recall'))
            print(f"GLOBAL@100 LOO:{res100loo}")
            res100base = compute_metrics_on_users(all_users_int, model, valid_data, dataset, topk_np[:, :100],
                                                 ('mrr', 'ndcg', 'recall'))
            print(f"GLOBAL@100 BASE: {res100base}\n")


            # res10local = compute_metrics_on_users(all_users_int, loo_model, valid_data, dataset, topk_np[:, :10],('mrr', 'ndcg', 'recall'))
            # print(f"GLOBAL@10 LOO: {res10local}")
            results_obj['global']['relevance'] = {'loo':res20loo}

            # res20local = compute_metrics_on_users(all_users_int, model, valid_data, dataset, topk_np,('mrr', 'ndcg', 'recall'))
            # print(f"GLOBAL@20 BASE:{res20local}")
            # res10local = compute_metrics_on_users(all_users_int, model, valid_data, dataset, topk_np[:, :10],('mrr', 'ndcg', 'recall'))
            # print(f"GLOBAL@10 BASE: {res10local}")




        # ---------------- LOCAL ----------------
        results_obj['local'] = {}

        for i,users in enumerate(target_id):

            if i == 0:
                tipo = 'all'
            elif i == 1:
                tipo = 'high'
            else:
                tipo = 'low'
            print(f"LOCAL defect type: {tipo} {len(users)}")

            base_recs_local = [(i, j) for i, j in zip(all_users_int, topk_np[:,:20]) if i in users]
            loo_recs_local = [(i, j) for i, j in zip(all_users_int_loo, topk_np_loo[:,:20]) if i in users]

            H_base, H_loo, delta = entropy_stats(base_recs_local, loo_recs_local)
            novelty_base = calculate_novelty(recs_list=base_recs_local, df=df_train, dataset=dataset)
            novelty_loo = calculate_novelty(recs_list=loo_recs_local, df=df_train, dataset=dataset)
            delta_nov = novelty_loo - novelty_base

            results_obj['local'][tipo] = {}


            print(f"\nLOCAL {tipo}")
            print('users: ',len(users))
            print(f"Entropia Base: {H_base:.4f}")
            print(f"Entropia LOO:  {H_loo:.4f}")
            print(f"Delta Entropia: {delta:.4f}\n")
            results_obj['local'][tipo]['entropy'] = {'base':H_base,'loo':H_loo,'delta':delta}

            print(f"Novelty Base: {novelty_base:.4f}")
            print(f"Novelty LOO:  {novelty_loo:.4f}")
            print(f"Delta Novelty: {delta_nov:.4f}\n")
            results_obj['local'][tipo]['novelty'] = {'base':novelty_base,'loo':novelty_loo,'delta':delta_nov}


            coverage_base = calculate_coverage(recs_list=base_recs_local, num_total_items=num_total_items)
            coverage_loo = calculate_coverage(recs_list=loo_recs_local, num_total_items=num_total_items)
            delta = coverage_loo - coverage_base
            print(f"Coverage Base: {coverage_base:.4f}")
            print(f"Coverage LOO:  {coverage_loo:.4f}")
            print(f"Delta Coverage: {delta:.4f}\n")
            results_obj['local'][tipo]['coverage'] = {'base':coverage_base,'loo':coverage_loo,'delta':delta}

            filtered_topk_loo = topk_np_loo[list(users)]
            filtered_topk = topk_np[list(users)]
            res20local_loo = compute_metrics_on_users(users, loo_model, valid_data, dataset, filtered_topk_loo[:, :20],('mrr', 'ndcg', 'recall'))
            res20local_base = compute_metrics_on_users(users, model, valid_data, dataset, filtered_topk[:, :20],
                                                  ('mrr', 'ndcg', 'recall'))
            print(f"LOCAL@20 BASE: {res20local_base}")
            print(f"LOCAL@20 LOO: {res20local_loo}\n")

            filtered_topk_loo = topk_np_loo[list(users)]
            filtered_topk = topk_np[list(users)]
            res100local_loo = compute_metrics_on_users(users, loo_model, valid_data, dataset, filtered_topk_loo[:, :100],('mrr', 'ndcg', 'recall'),filter=True)
            res100local_base = compute_metrics_on_users(users, model, valid_data, dataset, filtered_topk[:, :100],
                                                  ('mrr', 'ndcg', 'recall'),filter=True)
            print(f"LOCAL@100 BASE: {res100local_base}")
            print(f"LOCAL@100 LOO: {res100local_loo}\n")



            results_obj['local'][tipo]['relevance'] = {'base':res20local_base,'loo':res20local_loo}
            print('end training', time.time()-st)

            with open(f"{BASE_DIR}/files/{dataset_name}_labels.jsonl","a") as g:
                json.dump(results_obj, g)
                g.write("\n")

        print("\n\n\n")

    return results_obj,{}


def compute_all_loo_impacts(
    dataset_name: str,
    defects: list[dict],
    base_weights: dict,
    config,
    train_data,
    valid_data,
    df_train: pd.DataFrame,
    all_users_int, topk_np,
    method: Literal["influence", "local", "warm"] = "warm",  # [OTT. E]
    n = 0,target_users=[],target_id=[],results=dict,
) -> tuple[dict, dict, dict]:
    """
    Calcola LOO impact per tutti i difetti.

    Ottimizzazioni applicate:
    - base_model costruito UNA SOLA VOLTA                    [OTT. D2]
    - target_users calcolato PER DIFETTO (non unione)        [OTT. D1]
    - precomputed_E0 costruito UNA SOLA VOLTA                [OTT. A]
    - _get_or_build_base_adj() chiamato PRIMA del loop       [OTT. G]
    - precompute_ranking filtra utenti per item rilevanti    [OTT. F]
    """
    defects = [d for d in defects if d["group_id"] != "original"]
    #random.shuffle(defects)
    # ── Preriscalda la cache adj (O(E) pagato una sola volta) ─────────────────
    _get_or_build_base_adj(train_data.dataset)


    # [OTT. F] full_sort_topk solo sugli utenti rilevanti

    # ── Base model costruito UNA SOLA VOLTA [OTT. D2] ─────────────────────────
    base_model = LightGCN(config, train_data.dataset).to(config["device"])
    base_model.load_state_dict(base_weights)
    base_model.eval()
    st = time.time()
    # all_users_int, topk_np = precompute_ranking(
    #     model=base_model,
    #     test_data=valid_data,
    #     dataset=valid_data.dataset,
    #     config=config,
    #     k_scan=k_scan,
    #     filter_items=None,
    # )


    st = time.time()
    obj_res,_ = loo_impact(
        dataset_name=dataset_name,
        defects=defects,
        base_weights=base_weights,
        config=config,
        train_data=train_data,
        valid_data=valid_data,
        df_train=df_train,
        all_users_int=all_users_int,
        topk_np=topk_np,target_users=target_users,target_id = target_id,results=results,
        base_model=base_model,
        method=method,     n=n                 # [OTT. E]
    )
    print('difetto in: ',time.time()-st)

    return obj_res


def LOO(defect_to_keep,prev_list,dataset_name,keep_all=True,single_eval=False,iteration=0):
    print(f"--- ITERATION: {iteration} ---")
    os.makedirs(f"{PARENT_DIR}/recbole/dataset/{dataset_name}", exist_ok=True)

    path = f"{PARENT_DIR}/data/noisy/{dataset_name}_5/dataset_dirty_new.csv"

    out_dir = f"{BASE_DIR}/dataset/{dataset_name}/{dataset_name}.train.inter"
    df_train = load_dataset(path, out_dir)
    df_defects = pd.read_csv(f"{PARENT_DIR}/data/noisy/{dataset_name}_5/groups/merged_filtered.csv")
    #df_defects = pd.read_csv(f"{PARENT_DIR}/data/noisy/{dataset_name}_5/injected_noise_new.csv")

    (model, config, dataset, train_data, valid_data, test_data,
     base_weights, checkpoint_path) = train_base_model(
        f"{BASE_DIR}/config/lightgcn_{dataset_name}.yaml", dataset_name, local=False,
        relevance=len(defects_regressor) > 0
    )

    model.eval()
    all_users_int, topk_np = precompute_ranking(
        model=model,
        test_data=valid_data,
        dataset=valid_data.dataset,
        config=config,
        k_scan=k_scan,
        filter_items=None, chunked=True

    )
    defects = [
        {"id": gid, "group_id": gid}
        for gid in df_defects["group_id"].unique() if gid in defect_to_keep + prev_list
    ]

    if keep_all:
        defects = [defects]
        if single_eval:
            defects_one_by_one = [[d] for d in defects if d['id'] not in prev_list]
            defects.extend(defects_one_by_one)

    res_ob = {}
    for defects_list in defects:
        print("----" * 50)
        print(f"{defects_list} defectslist")
        if type(defects_list) != list:
            defects_list = [defects_list]
        n = len(defects_list)

        target_users, target_id = get_target_users_by_rank(
            defect_group_id=[d['group_id'] for d in defects[-1]],
            df_train=df_train,
            df_noise=df_defects,
            dataset=valid_data.dataset,
            all_users_int=all_users_int,
            topk_np=topk_np,
            top_users=5000,
        )

        res20global = compute_metrics_on_users(all_users_int, model, valid_data, dataset, topk_np[:, :20],
                                               ('mrr', 'ndcg', 'recall'))
        print(f"GLOBAL@20 BASE {tipo}: {res20global}\n")



        obj = compute_all_loo_impacts(
            dataset_name=dataset_name,
            defects=defects_list,
            base_weights=base_weights,
            config=config,
            train_data=train_data,
            valid_data=valid_data,
            df_train=df_train,all_users_int=all_users_int,topk_np=topk_np,
            method="warm", n=n, target_users=target_users, target_id=target_id
            # cambia in "local" per risultati più precisi
        )

        for ii, tid in enumerate(target_id):

            if ii == 0:
                tipo = 'all'
            elif ii == 1:
                tipo = 'high'
            else:
                tipo = 'low'
            print(f"defect type: {tipo} {len(tid)}")
            filtered_topk = topk_np[list(tid)]
            res20local = compute_metrics_on_users(tid, model, valid_data, dataset, filtered_topk[:, :20],
                                                  ('mrr', 'ndcg', 'recall'))
            obj['local'][tipo]['relevance']['base'] = res20local
            print(f"LOCAL@20 BASE {tipo}: {res20local}\n")

        obj['global']['relevance']['base'] = res20global

        if len(defects_list) == 1:
            res_ob[defects_list[0]['id']] = obj
        else:
            res_ob['all'] = obj


        print("----" * 50)
    f = open(f"{BASE_DIR}/results/loo_{dataset_name}_results_{iteration}.json",'w')
    json.dump(res_ob,f,indent=4)
    return res_ob



# =============================================================================
# Main
# =============================================================================

def LOO_eval():
    import sys
    import os
    try:
        dataset = os.environ.get("DATASET")
        print(dataset)
    except Exception as e:
        dataset = "Office_Products"



    group_by = True
    baseline = True
    k_scan = 20
    if group_by or baseline:
        k_scan = 100
    defects_regressor = []

    defect_to_keep = []

    for dataset_name in ["Toys_and_Games","Pet_Supplies","Office_Products"]:

        print(f"dataset={dataset}")

        os.makedirs(f"{PARENT_DIR}/recbole/dataset/{dataset_name}", exist_ok=True)

        path      = f"{PARENT_DIR}/data/noisy/{dataset_name}_5/dataset_dirty_new.csv"
        vali_path = f"{PARENT_DIR}/data/original/split/{dataset_name}/vali_{dataset_name}_5.csv"
        test_path = f"{PARENT_DIR}/data/original/split/{dataset_name}/test_{dataset_name}_5.csv"

        out_dir   = f"{BASE_DIR}/dataset/{dataset_name}/{dataset_name}.train.inter"
        out_v_dir = f"{BASE_DIR}/dataset/{dataset_name}/{dataset_name}.valid.inter"
        out_t_dir = f"{BASE_DIR}/dataset/{dataset_name}/{dataset_name}.test.inter"

        df_train = load_dataset(path,      out_dir)
        df_val   = load_dataset(vali_path, out_v_dir)
        df_test  = load_dataset(test_path, out_t_dir)

        df_defects = pd.read_csv(f"{PARENT_DIR}/data/noisy/{dataset_name}_5/injected_noise_new.csv")

        (model, config, dataset, train_data, valid_data, test_data,
         base_weights, checkpoint_path) = train_base_model(
            f"{BASE_DIR}/config/lightgcn_{dataset_name}.yaml", dataset_name,local=False,relevance=len(defects_regressor) > 0
        )

        valid_data_local = None


        model.eval()
        all_users_int, topk_np = precompute_ranking(
            model=model,
            test_data=valid_data,
            dataset=valid_data.dataset,
            config=config,
            k_scan=k_scan,
            filter_items=None, chunked=True

        )

        defects = [
            {"id": gid, "group_id": gid}
            for gid in df_defects["group_id"].unique()
        ]
        defects_regressor_obj = [
            {"id": gid, "group_id": gid}
            for gid in defects_regressor
        ]

        if (len(defects_regressor) > 0):
            defects_list = [
                {"id": gid, "group_id": gid}
                for gid in defects_regressor
            ]


            from itertools import accumulate

            defects_list = []
            current = []
            defects = []
            for dl in lista_difetti:
                current = [ {                  
                        "id": el,
                        "group_id": el
                    }
                    for el in dl]


                defects.append(current.copy())





        print(f"totale difetti evaluated: {len(defects)}")


        print("compute loo impacts", flush=True)

        n = 0

        if len(defects_regressor) == 0:
            random.shuffle(defects)


        save = True

        if save:
            results = {}
            for defects_list in defects:

                print("----" * 50)
                print(f"{defects_list} defectslist")
                if type(defects_list) != list:
                    results[defects_list] = {}
                    defects_list = [defects_list]
                n = len(defects_list)

                target_users, target_id = get_target_users_by_rank(
                    defect_group_id=[d['group_id'] for d in defects_list],
                    df_train=df_train,
                    df_noise=df_defects,
                    dataset=valid_data.dataset,
                    all_users_int=all_users_int,
                    topk_np=topk_np[:, :20],
                    top_users=5000,
                )

                
                for ii,tid in enumerate(target_id):

                    if ii == 0:
                        tipo = 'all'
                    elif ii == 1:
                        tipo = 'high'
                    else:
                        tipo = 'low'
                    print(f"defect type: {tipo} {len(tid)}")

                obj = compute_all_loo_impacts(
                    dataset_name=dataset_name,
                    defects=defects_list,
                    base_weights=base_weights,
                    config=config,
                    train_data=train_data,
                    valid_data=valid_data,
                    df_train=df_train,all_users_int=all_users_int,topk_np=topk_np,
                    method="warm",n=n,target_users=target_users,target_id=target_id,results=results,
                    # cambia in "local" per risultati più precisi
                )
                print("----" * 50)

