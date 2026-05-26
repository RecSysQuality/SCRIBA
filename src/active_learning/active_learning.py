"""
train_sgd_regressor.py

Fase 1 — fit iniziale: SGDRegressor su tutti i dataset insieme
          (unico batch = tutto il training set, molti epoch).
Fase 2 — aggiornamento incrementale: partial_fit su nuovi dataset.
"""
import json
import os
import random
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor

import pandas as pd
import numpy as np
import torch
from lightgbm import LGBMRegressor
from scipy.stats import spearmanr, kendalltau
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_regression, VarianceThreshold
from sklearn.linear_model import SGDRegressor, SGDClassifier, Ridge, BayesianRidge, ElasticNet, HuberRegressor, Lasso
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, f1_score, \
    mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import cross_val_predict, GroupKFold, cross_val_score, learning_curve
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import joblib

from sklearn.ensemble import HistGradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import cross_validate

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
_rng = np.random.default_rng(SEED)
NUM_SAMPLES = 300  # NUMBER OF SAMPELS FOR PASSIVE LEARNIGN FOR EACH DATASET INVOLVED
import torch
import torch.nn as nn


def load_embeddings(pt_path: str, drop_features=None):
    checkpoint = torch.load(pt_path, map_location="cpu")

    X = checkpoint["features"].numpy()
    col_names = checkpoint["col_names"]
    defect_ids = checkpoint["defect_ids"]

    group = [
        x.rsplit("_", 1)[0]
        for x in checkpoint["defect_ids"]
    ]

    # keep_mask = [
    #     ("hijacking" not in g) and ("bridge" not in g)
    #     for g in group
    # ]
    #
    # # applica filtro
    # X = X[keep_mask]
    # defect_ids = [d for d, k in zip(defect_ids, keep_mask) if k]
    # group = [g for g, k in zip(group, keep_mask) if k]

    if drop_features is not None:
        keep_idx = [i for i, c in enumerate(col_names) if c not in drop_features]
        X = X[:, keep_idx]
        col_names = [col_names[i] for i in keep_idx]
    return X, defect_ids, col_names, group


def load_labels(json_path: str) -> dict[str, float]:
    with open(json_path) as f:
        data = json.load(f)
        # return data
    return {str(e["defect_id"]): float(e["delta_entropia 0"]) for e in data}


def align_pairs(features, defect_ids, labels, min_abs=0.1, max_abs=0.1, drop_ids=None, to_keep_ids=None):
    rows, ys, ids = [], [], []
    drop_ids = set(drop_ids or [])

    labels_dict = {tuple(l[0]): float(l[1]) for l in labels}

    for i, did in enumerate(defect_ids):
        did = tuple(did)
        did = (did[1], did[0])  # più veloce e pulito

        if did in labels_dict:
            y = labels_dict[did]

            if min_abs <= abs(y) <= max_abs:
                if to_keep_ids is None or (did in to_keep_ids):
                    rows.append(features[i])
                    ys.append(y)
                    ids.append(did)

    if not rows:
        raise ValueError("Nessun defect_id dopo filtro range label.")

    return (
        np.array(rows, dtype=np.float32),
        np.array(ys, dtype=np.float32),
        ids,
    )


def normalize_graph_features(X, feature_names):
    X = X.copy()

    for i, name in enumerate(feature_names):

        if name in ["n_edges", "avg_degree", "density"]:
            # log scaling
            X[:, i] = np.log1p(X[:, i])

        elif name in ["avg_degree"]:
            # normalizzazione per possibile scala grafo
            X[:, i] = X[:, i] / (np.max(X[:, i]) + 1e-8)

    return X


def align_for_inference(features, defect_ids):
    rows, ys, ids, groupsf = [], [], [], []

    for i, did in enumerate(defect_ids):
        rows.append(features[i])
        ids.append(did)

    if not rows:
        raise ValueError("Nessun defect_id dopo filtro range label.")

    return (
        np.array(rows, dtype=np.float32),
        ids
    )


def align(features, defect_ids, groups, labels, min_abs=0.0, max_abs=100.0, drop_ids=None, to_keep_ids=None):
    rows, ys, ids, groupsf = [], [], [], []
    drop_ids = set(drop_ids or [])

    for i, did in enumerate(defect_ids):

        if did in drop_ids:
            continue

        if did in labels:
            y = labels[did]

            if min_abs <= abs(y) <= max_abs:
                if to_keep_ids is None or (did in to_keep_ids):
                    rows.append(features[i])
                    ys.append(y)
                    ids.append(did)
                    groupsf.append(groups[i])

    if not rows:
        raise ValueError("Nessun defect_id dopo filtro range label.")

    return (
        np.array(rows, dtype=np.float32),
        np.array(ys, dtype=np.float32),
        ids, groupsf
    )


def split_dataset(X, y, ids, train_ratio=0.8, val_ratio=0.1, random_state=42, test_ids_set=None):
    n = len(X)
    rng = np.random.default_rng(random_state)
    idx = rng.permutation(n)

    if not test_ids_set:
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        tr = idx[:n_train]
        va = idx[n_train: n_train + n_val]
        te = idx[n_train + n_val:]

        ids_arr = np.array(ids)
    else:
        rng = np.random.default_rng(random_state)
        ids_arr = np.array(ids)
        test_ids_set = set(test_ids_set)

        te = np.array([i for i, did in enumerate(ids_arr)
                       if did[0] in test_ids_set])

        rest = np.array([i for i, did in enumerate(ids_arr)
                         if did[0] not in test_ids_set])

        # shuffle solo il resto
        rest = rng.permutation(rest)

        n_train = int(len(rest) * train_ratio)
        n_val = int(len(rest) * val_ratio)

        tr = rest[:n_train]
        va = rest[n_train: n_train + n_val]

    return (
        X[tr], y[tr], ids_arr[tr].tolist(),
        X[va], y[va], ids_arr[va].tolist(),
        X[te], y[te], ids_arr[te].tolist(),
    )


from sklearn.model_selection import train_test_split


def prepare_single_datasets_for_inference(dataset, train_ratio=0.8, val_ratio=0.1, random_state=42):
    """
    Carica, allinea e splitta tutti i dataset.
    Restituisce i tre split globali (concatenazione di tutti i dataset).
    """

    print("Loading and split...\n")
    drop = []
    features, defect_ids, col_names, group = load_embeddings(dataset["pt_path"])

    X, ids = align_for_inference(features, defect_ids)

    return X, ids


def prepare_single_datasets(dataset, train_ratio=0.8, val_ratio=0.1, random_state=42):
    """
    Carica, allinea e splitta tutti i dataset.
    Restituisce i tre split globali (concatenazione di tutti i dataset).
    """

    print("Loading and split...\n")
    drop = []
    features, defect_ids, col_names, group = load_embeddings(dataset["pt_path"])
    labels = load_labels(dataset["json_path"])

    X, y, ids, group = align(features, defect_ids, group, labels, drop_ids=drop)

    y = np.abs(y)

    X_tr, X_te, y_tr, y_te, ids_tr, ids_te, group_tr, group_te = train_test_split(
        X, y, ids, group,
        test_size=val_ratio,
        random_state=random_state, stratify=group
    )

    name = dataset.get("name", dataset["pt_path"])
    print(f"  {name:<30}  totale={len(ids)}  "
          f"train={len(ids_tr)}    test={len(ids_te)}")

    print(f"\n  Totale  →  train={len(X_tr)}  test={len(X_te)}\n")
    print("percentuale ~0:", np.mean(np.abs(y_tr) < 1e-4))
    print("y_train:", y_tr.mean(), y_tr.std(), y_tr.min(), y_tr.max())
    correlations = pd.DataFrame(X_tr, columns=col_names).corrwith(pd.Series(y_tr)).abs().sort_values(
        ascending=False)
    print(correlations.head(20))

    return X_tr, y_tr, X_te, y_te, ids_te, ids_tr


def prepare_datasets(datasets, train_ratio=0.8, val_ratio=0.1, random_state=42):
    """
    Carica, allinea e splitta tutti i dataset.
    Restituisce i tre split globali (concatenazione di tutti i dataset).
    """
    all_tr_X, all_tr_y = [], []
    all_va_X, all_va_y = [], []
    all_te_X, all_te_y = [], []
    all_te_ids, all_tr_ids = [], []

    print("Loading and split...\n")
    drop = []
    for ds in datasets:
        features, defect_ids, col_names, group = load_embeddings(ds["pt_path"])
        labels = load_labels(ds["json_path"])
        X, y, ids, group = align(features, defect_ids, group, labels, drop_ids=drop)

        y = np.abs(y)

        X_tr, X_te, y_tr, y_te, ids_tr, ids_te, group_tr, group_te = train_test_split(
            X, y, ids, group,
            test_size=0.3,
            random_state=random_state, stratify=group
        )

        name = ds.get("name", ds["pt_path"])
        print(f"  {name:<30}  totale={len(ids)}  "
              f"train={len(ids_tr)}    test={len(ids_te)}")

        all_tr_X.append(X_tr)
        all_tr_y.append(y_tr)
        all_te_X.append(X_te)
        all_te_y.append(y_te)
        all_te_ids.append(ids_te)
        all_tr_ids.append(ids_tr)

    X_train = np.concatenate(all_tr_X[0:3])
    y_train = np.concatenate(all_tr_y[0:3])

    X_test = np.concatenate(all_te_X[3:4])
    y_test = np.concatenate(all_te_y[3:4])
    all_te_ids = all_te_ids[3:4]

    print(f"\n  Totale  →  train={len(X_train)}  test={len(X_test)}\n")
    print("percentuale ~0:", np.mean(np.abs(y_train) < 1e-4))
    print("y_train:", y_train.mean(), y_train.std(), y_train.min(), y_train.max())
    correlations = pd.DataFrame(X_train, columns=col_names).corrwith(pd.Series(y_train)).abs().sort_values(
        ascending=False)
    print(correlations.head(20))

    return X_train, y_train, X_test, y_test, all_te_ids


def diagnose(X_train, y_train):
    spearman_corrs = [abs(spearmanr(X_train[:, j], y_train)[0])
                      for j in range(X_train.shape[1])]
    max_s = max(spearman_corrs)
    #
    # print("mean:", np.mean(spearman_corrs))
    # print("median:", np.median(spearman_corrs))
    # print("top10 mean:", np.mean(np.sort(spearman_corrs)[-10:]))
    #
    # print(f"Max Spearman: {max_s:.4f}")

    if max_s < 0.10:
        print("→ Il problema è nelle FEATURE, non nel modello")
    elif max_s < 0.20:
        print("→ Segnale debole, migliora le feature")
    else:
        print("→ Segnale ok, il problema è nel modello")


# =============================================================================
# Fase 1 — fit iniziale su tutto il training set
# =============================================================================
from sklearn.model_selection import KFold


def initial_fit_sgd(
        datasets: list[dict],
        n_epochs: int = 100,
        loss: str = "huber",
        alpha: float = 1e-4,
        learning_rate: str = "invscaling",
        eta0: float = 0.01,
        random_state: int = 42,
        train_ratio: float = 0.8,
        val_ratio: float = 0.2,
        save_path: str | None = None,
):
    """
    Allena SGDRegressor passando tutto il training set come unico batch
    per n_epochs epoch. Equivale a fare gradient descent sull'intero dataset.
    """
    # X_train, y_train, X_val, y_val, X_test, y_test, te_ids = prepare_datasets(
    #     datasets, train_ratio, val_ratio, random_state=42
    # )
    all_tr_X, all_tr_y = [], []
    all_va_X, all_va_y = [], []
    all_te_X, all_te_y = [], []
    all_te_ids, all_tr_ids = [], []

    for dataset in datasets:
        X_tr, y_tr, X_te, y_te, ids_te, ids_tr = prepare_single_datasets(
            dataset, train_ratio, val_ratio=val_ratio, random_state=42
        )
        shuffle(X_tr, y_tr, ids_tr)
        shuffle(X_te, y_te, ids_te)
        X_tr, y_tr, ids_tr = X_tr[:NUM_SAMPLES], y_tr[:NUM_SAMPLES], ids_tr[:NUM_SAMPLES]
        X_te, y_te, ids_te = X_te[:NUM_SAMPLES], y_te[:NUM_SAMPLES], ids_te[:NUM_SAMPLES]
        all_tr_X.append(X_tr)
        all_tr_y.append(y_tr)
        all_te_X.append(X_te)
        all_te_y.append(y_te)
        all_te_ids.append(ids_te)
        all_tr_ids.append(ids_tr)

    X_train = np.concatenate(all_tr_X[0:3])
    y_train = np.concatenate(all_tr_y[0:3])

    X_test = np.concatenate(all_tr_X[3:4] + all_te_X[3:4])
    y_test = np.concatenate(all_tr_y[3:4] + all_te_y[3:4])
    all_test_ids = all_tr_ids[3:4] + all_te_ids[3:4]

    # Scaler fittato solo sul training
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = SGDRegressor(
        loss=loss,
        alpha=alpha,
        learning_rate=learning_rate,
        eta0=eta0,
        random_state=random_state,
        max_iter=50,
        tol=None,
    )

    model.fit(X_train, y_train)
    scores = cross_validate(
        model, X_train, y_train,
        cv=5,
        scoring=['r2', 'neg_mean_absolute_error']
    )

    print(scores)
    # # Valutazione sul test set
    y_pred = model.predict(X_test)
    #
    # # costruisci dataframe con ranking
    # df_rank = pd.DataFrame({
    #     "test_id": all_te_ids[0],  # <-- oppure X_test.index se è DataFrame
    #     "y_true": y_test,
    #     "y_pred": y_pred
    # })
    #
    # # ordina per score predetto (decrescente)
    # df_rank = df_rank.sort_values("y_pred", ascending=False)
    # df_rank.to_csv(f"{BASE_DIR}/prediction_sports.csv",index=False)
    # print(df_rank['test_id'].tolist())
    # print(df_rank[df_rank['y_pred']>0.3]['test_id'].tolist())
    #
    #
    #
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    # spearman_corr, _ = spearmanr(y_test, y_pred)

    print(
        # f"Spearman {spearman_corr:.4f} | "
        f"MSE={mse:.4f} | "
        f"MAE={mae:.4f} | "
        f"RMSE={rmse:.4f} | "
        f"R2={r2:.4f}"
    )

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump({"model": model, "scaler": scaler}, save_path)
        print(f"Modello salvato in {save_path}")

    return model, scaler, X_train, y_train


# =============================================================================
# Fase 2 — aggiornamento incrementale su nuovo dataset
# =============================================================================

# =============================================================================
# Inferenza
# =============================================================================

def predict(pt_path: str, model: SGDRegressor,
            scaler: StandardScaler) -> dict[str, float]:
    features, defect_ids, _ = load_embeddings(pt_path)
    X_scaled = scaler.transform(features)
    preds = model.predict(X_scaled)
    return dict(zip(defect_ids, preds.tolist()))


# =============================================================================
# Main
# =============================================================================
from sklearn.utils import shuffle


def active_learning(dataset, model, scaler, X_test, y_test, save_path, labels=None):

    features, defect_ids, col_names, group = load_embeddings(dataset["pt_path"])

    X, y, ids, group = align(features, defect_ids, group, labels)
    X = scaler(X)
    y = np.abs(y)

    mae_list = []
    mse_list = []
    rmse_list = []
    r2_list = []
    spearman_list = []


    for start in range(0, len(X), batch_size):

        X_batch = X
        y_batch = y

        model.partial_fit(X_batch, y_batch)

        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        spearman_corr, _ = spearmanr(y_test, y_pred)
        mse_list.append(mse)
        mae_list.append(mae)
        rmse_list.append(rmse)
        r2_list.append(r2)
        spearman_list.append(spearman_corr)
        print(
            f"Batch {start // batch_size + 1} | "
            f"Spearman {spearman_corr:.4f} | "
            f"MSE={mse:.4f} | "
            f"MAE={mae:.4f} | "
            f"RMSE={rmse:.4f} | "
            f"R2={r2:.4f}"
        )
    joblib.dump({"model": model, "scaler": scaler}, save_path)



def create_groups(df, path_csv, budget=10000):
    df_noise = pd.read_csv(path_csv)
    defects = df[df['y_pred'] > 0.3]['test_id'].unique().tolist()

    # edges per defect
    edge_count = (
        df_noise.groupby('group_id')
        .size()
        .to_dict()
    )

    remaining_defects = defects.copy()

    groups = []

    while remaining_defects:

        current_group = []
        current_budget = 0

        used = []

        for did in remaining_defects:

            n_edges = edge_count[did]

            # salta se da solo supera budget
            if n_edges > budget:
                continue

            # aggiungi se ci sta
            if current_budget + n_edges <= budget:
                current_group.append(did)
                current_budget += n_edges
                used.append(did)

        # salva gruppo
        groups.append(current_group)

        # rimuovi quelli usati
        remaining_defects = [
            did for did in remaining_defects
            if did not in used
        ]

    return groups



def run_active_learning(dataset_active,save_path):
    data = joblib.load(save_path)

    model = data["model"]
    scaler = data["scaler"]

    active_learning(
        dataset=dataset_active,
        model=model,
        scaler=scaler, X_test=X, y_test=y, save_path= save_path, labels=labels
    )


def predict_impact(dataset):
    features, defect_ids, col_names, group = load_embeddings(dataset["pt_path"])

    X, ids = align_for_inference(features, defect_ids)
    y_pred = model.predict(X)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    spearman_corr, _ = spearmanr(y_test, y_pred)
    mse_list.append(mse)
    mae_list.append(mae)
    rmse_list.append(rmse)
    r2_list.append(r2)
    spearman_list.append(spearman_corr)
    print(
        f"Batch {start // batch_size + 1} | "
        f"Spearman {spearman_corr:.4f} | "
        f"MSE={mse:.4f} | "
        f"MAE={mae:.4f} | "
        f"RMSE={rmse:.4f} | "
        f"R2={r2:.4f}"
    )
    # costruisci dataframe con ranking
    mask = y_pred > 0.3

    X = X[mask]
    y = y[mask]
    ids = ids[mask]
    group = group[mask]
    y_pred = y_pred[mask]
    return X,y_pred,ids







