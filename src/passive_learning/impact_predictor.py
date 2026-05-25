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


def active_learning(dataset, model, scaler, X_test, y_test, te_ids, labels=None):
    # arr = ['group_badnwagon_0', 'group_badnwagon_1', 'group_badnwagon_10', 'group_badnwagon_100', 'group_badnwagon_101', 'group_badnwagon_102', 'group_badnwagon_103', 'group_badnwagon_104', 'group_badnwagon_105', 'group_badnwagon_106', 'group_badnwagon_107', 'group_badnwagon_108', 'group_badnwagon_109', 'group_badnwagon_11', 'group_badnwagon_110', 'group_badnwagon_111', 'group_badnwagon_112', 'group_badnwagon_113', 'group_badnwagon_114', 'group_badnwagon_115', 'group_badnwagon_116', 'group_badnwagon_117', 'group_badnwagon_118', 'group_badnwagon_119', 'group_badnwagon_12', 'group_badnwagon_120', 'group_badnwagon_121', 'group_badnwagon_122', 'group_badnwagon_123', 'group_badnwagon_124', 'group_badnwagon_125', 'group_badnwagon_126', 'group_badnwagon_127', 'group_badnwagon_128', 'group_badnwagon_129', 'group_badnwagon_13', 'group_badnwagon_130', 'group_badnwagon_131', 'group_badnwagon_132', 'group_badnwagon_133', 'group_badnwagon_134', 'group_badnwagon_135', 'group_badnwagon_136', 'group_badnwagon_137', 'group_badnwagon_138', 'group_badnwagon_139', 'group_badnwagon_14', 'group_badnwagon_140', 'group_badnwagon_141', 'group_badnwagon_142', 'group_badnwagon_143', 'group_badnwagon_144', 'group_badnwagon_145', 'group_badnwagon_146', 'group_badnwagon_147', 'group_badnwagon_148', 'group_badnwagon_149', 'group_badnwagon_15', 'group_badnwagon_150', 'group_badnwagon_151', 'group_badnwagon_152', 'group_badnwagon_153', 'group_badnwagon_154', 'group_badnwagon_155', 'group_badnwagon_156', 'group_badnwagon_157', 'group_badnwagon_158', 'group_badnwagon_159', 'group_badnwagon_16', 'group_badnwagon_160', 'group_badnwagon_161', 'group_badnwagon_162', 'group_badnwagon_163', 'group_badnwagon_164', 'group_badnwagon_165', 'group_badnwagon_166', 'group_badnwagon_167', 'group_badnwagon_168', 'group_badnwagon_169', 'group_badnwagon_17', 'group_badnwagon_170', 'group_badnwagon_171', 'group_badnwagon_172', 'group_badnwagon_173', 'group_badnwagon_174', 'group_badnwagon_175', 'group_badnwagon_176', 'group_badnwagon_177', 'group_badnwagon_178', 'group_badnwagon_179', 'group_badnwagon_18', 'group_badnwagon_180', 'group_badnwagon_181', 'group_badnwagon_182', 'group_badnwagon_183', 'group_badnwagon_184', 'group_badnwagon_185', 'group_badnwagon_186', 'group_badnwagon_187', 'group_badnwagon_188', 'group_badnwagon_189', 'group_badnwagon_19', 'group_badnwagon_190', 'group_badnwagon_191', 'group_badnwagon_192', 'group_badnwagon_193', 'group_badnwagon_194', 'group_badnwagon_195', 'group_badnwagon_196', 'group_badnwagon_197', 'group_badnwagon_198', 'group_badnwagon_199', 'group_badnwagon_2', 'group_badnwagon_20', 'group_badnwagon_21', 'group_badnwagon_22', 'group_badnwagon_23', 'group_badnwagon_24', 'group_badnwagon_25', 'group_badnwagon_26', 'group_badnwagon_27', 'group_badnwagon_28', 'group_badnwagon_29', 'group_badnwagon_3', 'group_badnwagon_30', 'group_badnwagon_31', 'group_badnwagon_32', 'group_badnwagon_33', 'group_badnwagon_34', 'group_badnwagon_35', 'group_badnwagon_36', 'group_badnwagon_37', 'group_badnwagon_38', 'group_badnwagon_39', 'group_badnwagon_4', 'group_badnwagon_40', 'group_badnwagon_41', 'group_badnwagon_42', 'group_badnwagon_43', 'group_badnwagon_44', 'group_badnwagon_45', 'group_badnwagon_46', 'group_badnwagon_47', 'group_badnwagon_48', 'group_badnwagon_49', 'group_badnwagon_5', 'group_badnwagon_50', 'group_badnwagon_51', 'group_badnwagon_52', 'group_badnwagon_53', 'group_badnwagon_54', 'group_badnwagon_55', 'group_badnwagon_56', 'group_badnwagon_57', 'group_badnwagon_58', 'group_badnwagon_59', 'group_badnwagon_6', 'group_badnwagon_60', 'group_badnwagon_61', 'group_badnwagon_62', 'group_badnwagon_63', 'group_badnwagon_64', 'group_badnwagon_65', 'group_badnwagon_66', 'group_badnwagon_67', 'group_badnwagon_68', 'group_badnwagon_69', 'group_badnwagon_7', 'group_badnwagon_70', 'group_badnwagon_71', 'group_badnwagon_72', 'group_badnwagon_73', 'group_badnwagon_74', 'group_badnwagon_75', 'group_badnwagon_76', 'group_badnwagon_77', 'group_badnwagon_78', 'group_badnwagon_79', 'group_badnwagon_8', 'group_badnwagon_80', 'group_badnwagon_81', 'group_badnwagon_82', 'group_badnwagon_83', 'group_badnwagon_84', 'group_badnwagon_85', 'group_badnwagon_86', 'group_badnwagon_87', 'group_badnwagon_88', 'group_badnwagon_89', 'group_badnwagon_9', 'group_badnwagon_90', 'group_badnwagon_91', 'group_badnwagon_92', 'group_badnwagon_93', 'group_badnwagon_94', 'group_badnwagon_95', 'group_badnwagon_96', 'group_badnwagon_97', 'group_badnwagon_98', 'group_badnwagon_99', 'group_camouflage_0', 'group_camouflage_1', 'group_camouflage_10', 'group_camouflage_100', 'group_camouflage_101', 'group_camouflage_102', 'group_camouflage_103', 'group_camouflage_104', 'group_camouflage_105', 'group_camouflage_106', 'group_camouflage_107', 'group_camouflage_108', 'group_camouflage_109', 'group_camouflage_11', 'group_camouflage_110', 'group_camouflage_111', 'group_camouflage_112', 'group_camouflage_113', 'group_camouflage_114', 'group_camouflage_115', 'group_camouflage_116', 'group_camouflage_117', 'group_camouflage_118', 'group_camouflage_119', 'group_camouflage_12', 'group_camouflage_120', 'group_camouflage_121', 'group_camouflage_122', 'group_camouflage_123', 'group_camouflage_124', 'group_camouflage_125', 'group_camouflage_126', 'group_camouflage_127', 'group_camouflage_128', 'group_camouflage_129', 'group_camouflage_13', 'group_camouflage_130', 'group_camouflage_131', 'group_camouflage_132', 'group_camouflage_133', 'group_camouflage_134', 'group_camouflage_135', 'group_camouflage_136', 'group_camouflage_137', 'group_camouflage_138', 'group_camouflage_139', 'group_camouflage_14', 'group_camouflage_140', 'group_camouflage_141', 'group_camouflage_142', 'group_camouflage_143', 'group_camouflage_144', 'group_camouflage_145', 'group_camouflage_146', 'group_camouflage_147', 'group_camouflage_148', 'group_camouflage_149', 'group_camouflage_15', 'group_camouflage_150', 'group_camouflage_151', 'group_camouflage_152', 'group_camouflage_153', 'group_camouflage_154', 'group_camouflage_155', 'group_camouflage_156', 'group_camouflage_157', 'group_camouflage_158', 'group_camouflage_159', 'group_camouflage_16', 'group_camouflage_160', 'group_camouflage_161', 'group_camouflage_162', 'group_camouflage_163', 'group_camouflage_164', 'group_camouflage_165', 'group_camouflage_166', 'group_camouflage_167', 'group_camouflage_168', 'group_camouflage_169', 'group_camouflage_17', 'group_camouflage_170', 'group_camouflage_171', 'group_camouflage_172', 'group_camouflage_173', 'group_camouflage_174', 'group_camouflage_175', 'group_camouflage_176', 'group_camouflage_177', 'group_camouflage_178', 'group_camouflage_179', 'group_camouflage_18', 'group_camouflage_180', 'group_camouflage_181', 'group_camouflage_182', 'group_camouflage_183', 'group_camouflage_184', 'group_camouflage_185', 'group_camouflage_186', 'group_camouflage_187', 'group_camouflage_188', 'group_camouflage_189', 'group_camouflage_19', 'group_camouflage_190', 'group_camouflage_191', 'group_camouflage_192', 'group_camouflage_193', 'group_camouflage_194', 'group_camouflage_195', 'group_camouflage_196', 'group_camouflage_197', 'group_camouflage_198', 'group_camouflage_199', 'group_camouflage_2', 'group_camouflage_20', 'group_camouflage_21', 'group_camouflage_22', 'group_camouflage_23', 'group_camouflage_24', 'group_camouflage_25', 'group_camouflage_26', 'group_camouflage_27', 'group_camouflage_28', 'group_camouflage_29', 'group_camouflage_3', 'group_camouflage_30', 'group_camouflage_31', 'group_camouflage_32', 'group_camouflage_33', 'group_camouflage_34', 'group_camouflage_35', 'group_camouflage_36', 'group_camouflage_37', 'group_camouflage_38', 'group_camouflage_39', 'group_camouflage_4', 'group_camouflage_40', 'group_camouflage_41', 'group_camouflage_42', 'group_camouflage_43', 'group_camouflage_44', 'group_camouflage_45', 'group_camouflage_46', 'group_camouflage_47', 'group_camouflage_48', 'group_camouflage_49', 'group_camouflage_5', 'group_camouflage_50', 'group_camouflage_51', 'group_camouflage_52', 'group_camouflage_53', 'group_camouflage_54', 'group_camouflage_55', 'group_camouflage_56', 'group_camouflage_57', 'group_camouflage_58', 'group_camouflage_59', 'group_camouflage_6', 'group_camouflage_60', 'group_camouflage_61', 'group_camouflage_62', 'group_camouflage_63', 'group_camouflage_64', 'group_camouflage_65', 'group_camouflage_66', 'group_camouflage_67', 'group_camouflage_68', 'group_camouflage_69', 'group_camouflage_7', 'group_camouflage_70', 'group_camouflage_71', 'group_camouflage_72', 'group_camouflage_73', 'group_camouflage_74', 'group_camouflage_75', 'group_camouflage_76', 'group_camouflage_77', 'group_camouflage_78', 'group_camouflage_79', 'group_camouflage_8', 'group_camouflage_80', 'group_camouflage_81', 'group_camouflage_82', 'group_camouflage_83', 'group_camouflage_84', 'group_camouflage_85', 'group_camouflage_86', 'group_camouflage_87', 'group_camouflage_88', 'group_camouflage_89', 'group_camouflage_9', 'group_camouflage_90', 'group_camouflage_91', 'group_camouflage_92', 'group_camouflage_93', 'group_camouflage_94', 'group_camouflage_95', 'group_camouflage_96', 'group_camouflage_97', 'group_camouflage_98', 'group_camouflage_99', 'group_dense_cluster_0', 'group_dense_cluster_1', 'group_dense_cluster_10', 'group_dense_cluster_100', 'group_dense_cluster_101', 'group_dense_cluster_102', 'group_dense_cluster_103', 'group_dense_cluster_104', 'group_dense_cluster_105', 'group_dense_cluster_106', 'group_dense_cluster_107', 'group_dense_cluster_108', 'group_dense_cluster_109', 'group_dense_cluster_11', 'group_dense_cluster_110', 'group_dense_cluster_111', 'group_dense_cluster_112', 'group_dense_cluster_113', 'group_dense_cluster_114', 'group_dense_cluster_115', 'group_dense_cluster_116', 'group_dense_cluster_117', 'group_dense_cluster_118', 'group_dense_cluster_119', 'group_dense_cluster_12', 'group_dense_cluster_120', 'group_dense_cluster_121', 'group_dense_cluster_122', 'group_dense_cluster_123', 'group_dense_cluster_124', 'group_dense_cluster_125', 'group_dense_cluster_126', 'group_dense_cluster_127', 'group_dense_cluster_128', 'group_dense_cluster_129', 'group_dense_cluster_13', 'group_dense_cluster_130', 'group_dense_cluster_131', 'group_dense_cluster_132', 'group_dense_cluster_133', 'group_dense_cluster_134', 'group_dense_cluster_135', 'group_dense_cluster_136', 'group_dense_cluster_137', 'group_dense_cluster_138', 'group_dense_cluster_139', 'group_dense_cluster_14', 'group_dense_cluster_140', 'group_dense_cluster_141', 'group_dense_cluster_142', 'group_dense_cluster_143', 'group_dense_cluster_144', 'group_dense_cluster_145', 'group_dense_cluster_146', 'group_dense_cluster_147', 'group_dense_cluster_148', 'group_dense_cluster_149', 'group_dense_cluster_15', 'group_dense_cluster_150', 'group_dense_cluster_151', 'group_dense_cluster_152', 'group_dense_cluster_153', 'group_dense_cluster_154', 'group_dense_cluster_155', 'group_dense_cluster_156', 'group_dense_cluster_157', 'group_dense_cluster_158', 'group_dense_cluster_159', 'group_dense_cluster_16', 'group_dense_cluster_160', 'group_dense_cluster_161', 'group_dense_cluster_162', 'group_dense_cluster_163', 'group_dense_cluster_164', 'group_dense_cluster_165', 'group_dense_cluster_166', 'group_dense_cluster_167', 'group_dense_cluster_168', 'group_dense_cluster_169', 'group_dense_cluster_17', 'group_dense_cluster_170', 'group_dense_cluster_171', 'group_dense_cluster_172', 'group_dense_cluster_173', 'group_dense_cluster_174', 'group_dense_cluster_175', 'group_dense_cluster_176', 'group_dense_cluster_177', 'group_dense_cluster_178', 'group_dense_cluster_179', 'group_dense_cluster_18', 'group_dense_cluster_180', 'group_dense_cluster_181', 'group_dense_cluster_182', 'group_dense_cluster_183', 'group_dense_cluster_184', 'group_dense_cluster_185', 'group_dense_cluster_186', 'group_dense_cluster_187', 'group_dense_cluster_188', 'group_dense_cluster_189', 'group_dense_cluster_19', 'group_dense_cluster_190', 'group_dense_cluster_191', 'group_dense_cluster_192', 'group_dense_cluster_193', 'group_dense_cluster_194', 'group_dense_cluster_195', 'group_dense_cluster_196', 'group_dense_cluster_197', 'group_dense_cluster_198', 'group_dense_cluster_199', 'group_dense_cluster_2', 'group_dense_cluster_20', 'group_dense_cluster_21', 'group_dense_cluster_22', 'group_dense_cluster_23', 'group_dense_cluster_24', 'group_dense_cluster_25', 'group_dense_cluster_26', 'group_dense_cluster_27', 'group_dense_cluster_28', 'group_dense_cluster_29', 'group_dense_cluster_3', 'group_dense_cluster_30', 'group_dense_cluster_31', 'group_dense_cluster_32', 'group_dense_cluster_33', 'group_dense_cluster_34', 'group_dense_cluster_35', 'group_dense_cluster_36', 'group_dense_cluster_37', 'group_dense_cluster_38', 'group_dense_cluster_39', 'group_dense_cluster_4', 'group_dense_cluster_40', 'group_dense_cluster_41', 'group_dense_cluster_42', 'group_dense_cluster_43', 'group_dense_cluster_44', 'group_dense_cluster_45', 'group_dense_cluster_46', 'group_dense_cluster_47', 'group_dense_cluster_48', 'group_dense_cluster_49', 'group_dense_cluster_5', 'group_dense_cluster_50', 'group_dense_cluster_51', 'group_dense_cluster_52', 'group_dense_cluster_53', 'group_dense_cluster_54', 'group_dense_cluster_55', 'group_dense_cluster_56', 'group_dense_cluster_57', 'group_dense_cluster_58', 'group_dense_cluster_59', 'group_dense_cluster_6', 'group_dense_cluster_60', 'group_dense_cluster_61', 'group_dense_cluster_62', 'group_dense_cluster_63', 'group_dense_cluster_64', 'group_dense_cluster_65', 'group_dense_cluster_66', 'group_dense_cluster_67', 'group_dense_cluster_68', 'group_dense_cluster_69', 'group_dense_cluster_7', 'group_dense_cluster_70', 'group_dense_cluster_71', 'group_dense_cluster_72', 'group_dense_cluster_73', 'group_dense_cluster_74', 'group_dense_cluster_75', 'group_dense_cluster_76', 'group_dense_cluster_77', 'group_dense_cluster_78', 'group_dense_cluster_79', 'group_dense_cluster_8', 'group_dense_cluster_80', 'group_dense_cluster_81', 'group_dense_cluster_82', 'group_dense_cluster_83', 'group_dense_cluster_84', 'group_dense_cluster_85', 'group_dense_cluster_86', 'group_dense_cluster_87', 'group_dense_cluster_88', 'group_dense_cluster_89', 'group_dense_cluster_9', 'group_dense_cluster_90', 'group_dense_cluster_91', 'group_dense_cluster_92', 'group_dense_cluster_93', 'group_dense_cluster_94', 'group_dense_cluster_95', 'group_dense_cluster_96', 'group_dense_cluster_97', 'group_dense_cluster_98', 'group_dense_cluster_99', 'shilling_bridge_0', 'shilling_bridge_1', 'shilling_bridge_10', 'shilling_bridge_11', 'shilling_bridge_12', 'shilling_bridge_13', 'shilling_bridge_14', 'shilling_bridge_15', 'shilling_bridge_16', 'shilling_bridge_17', 'shilling_bridge_18', 'shilling_bridge_19', 'shilling_bridge_2', 'shilling_bridge_20', 'shilling_bridge_21', 'shilling_bridge_22', 'shilling_bridge_23', 'shilling_bridge_24', 'shilling_bridge_25', 'shilling_bridge_26', 'shilling_bridge_27', 'shilling_bridge_28', 'shilling_bridge_29', 'shilling_bridge_3', 'shilling_bridge_30', 'shilling_bridge_31', 'shilling_bridge_32', 'shilling_bridge_33', 'shilling_bridge_34', 'shilling_bridge_35', 'shilling_bridge_36', 'shilling_bridge_37', 'shilling_bridge_38', 'shilling_bridge_39', 'shilling_bridge_4', 'shilling_bridge_5', 'shilling_bridge_6', 'shilling_bridge_7', 'shilling_bridge_8', 'shilling_bridge_9', 'shilling_high_deg_0', 'shilling_high_deg_1', 'shilling_high_deg_10', 'shilling_high_deg_11', 'shilling_high_deg_12', 'shilling_high_deg_13', 'shilling_high_deg_14', 'shilling_high_deg_15', 'shilling_high_deg_16', 'shilling_high_deg_17', 'shilling_high_deg_18', 'shilling_high_deg_19', 'shilling_high_deg_2', 'shilling_high_deg_20', 'shilling_high_deg_21', 'shilling_high_deg_22', 'shilling_high_deg_23', 'shilling_high_deg_24', 'shilling_high_deg_25', 'shilling_high_deg_26', 'shilling_high_deg_27', 'shilling_high_deg_28', 'shilling_high_deg_29', 'shilling_high_deg_3', 'shilling_high_deg_30', 'shilling_high_deg_31', 'shilling_high_deg_32', 'shilling_high_deg_33', 'shilling_high_deg_34', 'shilling_high_deg_35', 'shilling_high_deg_36', 'shilling_high_deg_37', 'shilling_high_deg_38', 'shilling_high_deg_39', 'shilling_high_deg_4', 'shilling_high_deg_40', 'shilling_high_deg_41', 'shilling_high_deg_42', 'shilling_high_deg_43', 'shilling_high_deg_44', 'shilling_high_deg_45', 'shilling_high_deg_46', 'shilling_high_deg_47', 'shilling_high_deg_48', 'shilling_high_deg_49', 'shilling_high_deg_5', 'shilling_high_deg_50', 'shilling_high_deg_51', 'shilling_high_deg_52', 'shilling_high_deg_53', 'shilling_high_deg_54', 'shilling_high_deg_55', 'shilling_high_deg_56', 'shilling_high_deg_57', 'shilling_high_deg_58', 'shilling_high_deg_59', 'shilling_high_deg_6', 'shilling_high_deg_7', 'shilling_high_deg_8', 'shilling_high_deg_9', 'shilling_hijacking_0', 'shilling_hijacking_1', 'shilling_hijacking_10', 'shilling_hijacking_100', 'shilling_hijacking_101', 'shilling_hijacking_102', 'shilling_hijacking_103', 'shilling_hijacking_104', 'shilling_hijacking_105', 'shilling_hijacking_106', 'shilling_hijacking_107', 'shilling_hijacking_108', 'shilling_hijacking_109', 'shilling_hijacking_11', 'shilling_hijacking_110', 'shilling_hijacking_111', 'shilling_hijacking_112', 'shilling_hijacking_113', 'shilling_hijacking_114', 'shilling_hijacking_115', 'shilling_hijacking_116', 'shilling_hijacking_117', 'shilling_hijacking_118', 'shilling_hijacking_119', 'shilling_hijacking_12', 'shilling_hijacking_120', 'shilling_hijacking_121', 'shilling_hijacking_122', 'shilling_hijacking_123', 'shilling_hijacking_124', 'shilling_hijacking_125', 'shilling_hijacking_126', 'shilling_hijacking_127', 'shilling_hijacking_128', 'shilling_hijacking_129', 'shilling_hijacking_13', 'shilling_hijacking_130', 'shilling_hijacking_131', 'shilling_hijacking_132', 'shilling_hijacking_133', 'shilling_hijacking_134', 'shilling_hijacking_135', 'shilling_hijacking_136', 'shilling_hijacking_137', 'shilling_hijacking_138', 'shilling_hijacking_139', 'shilling_hijacking_14', 'shilling_hijacking_140', 'shilling_hijacking_141', 'shilling_hijacking_142', 'shilling_hijacking_143', 'shilling_hijacking_144', 'shilling_hijacking_145', 'shilling_hijacking_146', 'shilling_hijacking_147', 'shilling_hijacking_148', 'shilling_hijacking_149', 'shilling_hijacking_15', 'shilling_hijacking_150', 'shilling_hijacking_151', 'shilling_hijacking_152', 'shilling_hijacking_153', 'shilling_hijacking_154', 'shilling_hijacking_155', 'shilling_hijacking_156', 'shilling_hijacking_157', 'shilling_hijacking_158', 'shilling_hijacking_159', 'shilling_hijacking_16', 'shilling_hijacking_160', 'shilling_hijacking_161', 'shilling_hijacking_162', 'shilling_hijacking_163', 'shilling_hijacking_164', 'shilling_hijacking_165', 'shilling_hijacking_166', 'shilling_hijacking_167', 'shilling_hijacking_168', 'shilling_hijacking_169', 'shilling_hijacking_17', 'shilling_hijacking_170', 'shilling_hijacking_171', 'shilling_hijacking_172', 'shilling_hijacking_173', 'shilling_hijacking_174', 'shilling_hijacking_175', 'shilling_hijacking_176', 'shilling_hijacking_177', 'shilling_hijacking_178', 'shilling_hijacking_179', 'shilling_hijacking_18', 'shilling_hijacking_180', 'shilling_hijacking_181', 'shilling_hijacking_182', 'shilling_hijacking_183', 'shilling_hijacking_184', 'shilling_hijacking_185', 'shilling_hijacking_186', 'shilling_hijacking_187', 'shilling_hijacking_188', 'shilling_hijacking_189', 'shilling_hijacking_19', 'shilling_hijacking_190', 'shilling_hijacking_191', 'shilling_hijacking_192', 'shilling_hijacking_193', 'shilling_hijacking_194', 'shilling_hijacking_195', 'shilling_hijacking_196', 'shilling_hijacking_197', 'shilling_hijacking_198', 'shilling_hijacking_199', 'shilling_hijacking_2', 'shilling_hijacking_20', 'shilling_hijacking_21', 'shilling_hijacking_22', 'shilling_hijacking_23', 'shilling_hijacking_24', 'shilling_hijacking_25', 'shilling_hijacking_26', 'shilling_hijacking_27', 'shilling_hijacking_28', 'shilling_hijacking_29', 'shilling_hijacking_3', 'shilling_hijacking_30', 'shilling_hijacking_31', 'shilling_hijacking_32', 'shilling_hijacking_33', 'shilling_hijacking_34', 'shilling_hijacking_35', 'shilling_hijacking_36', 'shilling_hijacking_37', 'shilling_hijacking_38', 'shilling_hijacking_39', 'shilling_hijacking_4', 'shilling_hijacking_40', 'shilling_hijacking_41', 'shilling_hijacking_42', 'shilling_hijacking_43', 'shilling_hijacking_44', 'shilling_hijacking_45', 'shilling_hijacking_46', 'shilling_hijacking_47', 'shilling_hijacking_48', 'shilling_hijacking_49', 'shilling_hijacking_5', 'shilling_hijacking_50', 'shilling_hijacking_51', 'shilling_hijacking_52', 'shilling_hijacking_53', 'shilling_hijacking_54', 'shilling_hijacking_55', 'shilling_hijacking_56', 'shilling_hijacking_57', 'shilling_hijacking_58', 'shilling_hijacking_59', 'shilling_hijacking_6', 'shilling_hijacking_60', 'shilling_hijacking_61', 'shilling_hijacking_62', 'shilling_hijacking_63', 'shilling_hijacking_64', 'shilling_hijacking_65', 'shilling_hijacking_66', 'shilling_hijacking_67', 'shilling_hijacking_68', 'shilling_hijacking_69', 'shilling_hijacking_7', 'shilling_hijacking_70', 'shilling_hijacking_71', 'shilling_hijacking_72', 'shilling_hijacking_73', 'shilling_hijacking_74', 'shilling_hijacking_75', 'shilling_hijacking_76', 'shilling_hijacking_77', 'shilling_hijacking_78', 'shilling_hijacking_79', 'shilling_hijacking_8', 'shilling_hijacking_80', 'shilling_hijacking_81', 'shilling_hijacking_82', 'shilling_hijacking_83', 'shilling_hijacking_84', 'shilling_hijacking_85', 'shilling_hijacking_86', 'shilling_hijacking_87', 'shilling_hijacking_88', 'shilling_hijacking_89', 'shilling_hijacking_9', 'shilling_hijacking_90', 'shilling_hijacking_91', 'shilling_hijacking_92', 'shilling_hijacking_93', 'shilling_hijacking_94', 'shilling_hijacking_95', 'shilling_hijacking_96', 'shilling_hijacking_97', 'shilling_hijacking_98', 'shilling_hijacking_99']
    # shuffle(arr)
    features, defect_ids, col_names, group = load_embeddings(dataset["pt_path"])
    # labels                  = load_labels(dataset["json_path"])

    X, y, ids, group = align(features, defect_ids, group, labels)

    y = np.abs(y)

    mae_list = []
    mse_list = []
    rmse_list = []
    r2_list = []
    spearman_list = []
    batch_size = 50

    # spearman_list.append(spearman_corr)

    for start in range(0, len(X), batch_size):
        end = start + batch_size
        X_batch = X[start:end]
        y_batch = X[start:end]
        ids_batch = ids[start:end]

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

    # groups = [arr[i:i + 20] for i in range(0, len(arr), 10)]
    groups_ative_baselines = [
        ['group_badnwagon_119', 'group_camouflage_152', 'group_badnwagon_164', 'group_dense_cluster_32',
         'group_badnwagon_174', 'shilling_high_deg_54', 'group_camouflage_38', 'shilling_hijacking_89',
         'shilling_high_deg_18', 'group_badnwagon_171', 'shilling_hijacking_77', 'group_dense_cluster_173',
         'group_dense_cluster_2', 'group_dense_cluster_4', 'group_dense_cluster_172', 'shilling_hijacking_94',
         'shilling_hijacking_46', 'shilling_hijacking_132', 'group_badnwagon_43', 'group_camouflage_41',
         'shilling_bridge_25', 'shilling_hijacking_64', 'shilling_hijacking_161', 'shilling_bridge_2',
         'shilling_hijacking_9', 'shilling_hijacking_199', 'shilling_bridge_29', 'shilling_hijacking_62',
         'shilling_hijacking_38', 'shilling_hijacking_7', 'shilling_hijacking_23', 'shilling_hijacking_5',
         'shilling_hijacking_128', 'shilling_hijacking_164', 'shilling_hijacking_126', 'shilling_hijacking_138',
         'shilling_hijacking_24', 'shilling_bridge_13', 'shilling_hijacking_171', 'shilling_hijacking_32',
         'shilling_hijacking_122', 'shilling_hijacking_130', 'shilling_bridge_26', 'shilling_hijacking_137',
         'shilling_hijacking_197', 'shilling_hijacking_55', 'shilling_hijacking_121', 'shilling_hijacking_56',
         'shilling_hijacking_153', 'shilling_hijacking_73', 'shilling_hijacking_146', 'shilling_hijacking_183'],
        ['shilling_high_deg_44', 'group_badnwagon_40', 'group_camouflage_18', 'group_badnwagon_150',
         'group_camouflage_171', 'group_dense_cluster_12', 'group_camouflage_57', 'group_badnwagon_99',
         'group_dense_cluster_73', 'group_camouflage_115', 'group_badnwagon_92', 'group_badnwagon_199',
         'group_dense_cluster_24', 'shilling_high_deg_2', 'group_dense_cluster_170', 'group_badnwagon_77',
         'shilling_hijacking_112', 'shilling_hijacking_118', 'shilling_hijacking_150', 'shilling_hijacking_18'],
        ['group_badnwagon_156', 'shilling_high_deg_10', 'group_badnwagon_116', 'group_dense_cluster_159',
         'group_camouflage_0', 'shilling_high_deg_35', 'group_badnwagon_143', 'group_badnwagon_73',
         'group_camouflage_96', 'group_badnwagon_68', 'group_camouflage_188', 'shilling_high_deg_24',
         'group_dense_cluster_52', 'group_badnwagon_187', 'group_badnwagon_22', 'group_dense_cluster_121',
         'shilling_bridge_31', 'shilling_hijacking_188', 'shilling_hijacking_76', 'shilling_hijacking_119',
         'shilling_hijacking_101', 'shilling_hijacking_27', 'shilling_hijacking_142', 'shilling_hijacking_68',
         'shilling_hijacking_2', 'shilling_hijacking_135', 'shilling_bridge_34', 'shilling_hijacking_37',
         'shilling_bridge_33', 'shilling_hijacking_145', 'shilling_bridge_36', 'shilling_bridge_15',
         'shilling_hijacking_131', 'shilling_bridge_24', 'shilling_hijacking_111', 'shilling_hijacking_53',
         'shilling_hijacking_113', 'shilling_bridge_19', 'shilling_bridge_7', 'shilling_hijacking_99',
         'shilling_hijacking_175', 'shilling_hijacking_91', 'shilling_bridge_21', 'shilling_hijacking_58',
         'shilling_hijacking_87', 'shilling_hijacking_185', 'shilling_bridge_23'],
        ['group_badnwagon_139', 'group_badnwagon_1', 'group_badnwagon_42', 'group_badnwagon_134', 'group_badnwagon_179',
         'group_dense_cluster_102', 'group_camouflage_59', 'group_badnwagon_188', 'group_camouflage_52',
         'group_dense_cluster_139', 'shilling_high_deg_13', 'shilling_high_deg_27', 'shilling_high_deg_42',
         'group_dense_cluster_153', 'group_camouflage_42', 'shilling_hijacking_15', 'shilling_hijacking_114',
         'shilling_hijacking_10', 'shilling_hijacking_93', 'shilling_hijacking_177', 'shilling_hijacking_28',
         'shilling_hijacking_70', 'shilling_hijacking_115', 'shilling_hijacking_86', 'shilling_hijacking_110',
         'shilling_hijacking_66', 'shilling_hijacking_84', 'shilling_hijacking_4', 'shilling_hijacking_129',
         'shilling_bridge_37'],
        ['shilling_high_deg_57', 'group_badnwagon_186', 'group_dense_cluster_195', 'group_badnwagon_59',
         'group_camouflage_140', 'group_badnwagon_11', 'group_badnwagon_41', 'group_badnwagon_93',
         'group_camouflage_170', 'group_camouflage_199', 'group_dense_cluster_178', 'group_dense_cluster_142',
         'group_dense_cluster_146', 'group_badnwagon_35', 'shilling_bridge_14', 'shilling_hijacking_178',
         'shilling_hijacking_148', 'shilling_hijacking_78', 'shilling_hijacking_71', 'shilling_bridge_6',
         'shilling_hijacking_41', 'shilling_bridge_39', 'shilling_hijacking_67', 'shilling_hijacking_33',
         'shilling_hijacking_156', 'shilling_hijacking_6'],
        ['group_badnwagon_74', 'group_badnwagon_175', 'group_camouflage_15', 'group_camouflage_154',
         'group_camouflage_70', 'group_badnwagon_6', 'group_camouflage_65', 'group_dense_cluster_26',
         'shilling_high_deg_40', 'group_badnwagon_113', 'group_badnwagon_61', 'group_dense_cluster_7',
         'group_dense_cluster_186', 'shilling_high_deg_16', 'shilling_high_deg_48', 'group_badnwagon_54',
         'group_badnwagon_97', 'shilling_hijacking_95', 'shilling_hijacking_144', 'shilling_hijacking_102',
         'shilling_bridge_32', 'shilling_hijacking_123', 'shilling_hijacking_79', 'shilling_bridge_5',
         'shilling_hijacking_47', 'shilling_hijacking_26', 'shilling_hijacking_75', 'shilling_hijacking_43',
         'shilling_hijacking_191', 'shilling_hijacking_81', 'shilling_hijacking_125', 'shilling_hijacking_72',
         'shilling_hijacking_11', 'shilling_hijacking_69', 'shilling_hijacking_40', 'shilling_hijacking_97'],
        ['shilling_high_deg_53', 'group_dense_cluster_181', 'group_dense_cluster_87', 'shilling_high_deg_45',
         'group_dense_cluster_138', 'group_dense_cluster_123', 'group_dense_cluster_63', 'group_dense_cluster_18',
         'group_badnwagon_124', 'group_dense_cluster_151', 'group_badnwagon_3', 'group_dense_cluster_115',
         'group_badnwagon_166', 'group_dense_cluster_98', 'group_badnwagon_82', 'shilling_hijacking_127',
         'shilling_hijacking_172', 'shilling_bridge_3', 'shilling_hijacking_49', 'shilling_hijacking_8'],
        ['group_camouflage_71', 'group_camouflage_46', 'group_dense_cluster_90', 'group_badnwagon_162',
         'group_camouflage_131', 'group_badnwagon_57', 'group_dense_cluster_128', 'group_badnwagon_151',
         'group_camouflage_103', 'group_camouflage_196', 'group_dense_cluster_124', 'group_dense_cluster_182',
         'group_badnwagon_127', 'group_dense_cluster_44', 'group_badnwagon_112', 'group_dense_cluster_25',
         'group_dense_cluster_48', 'group_camouflage_186', 'shilling_hijacking_31', 'shilling_hijacking_180',
         'shilling_hijacking_116', 'shilling_bridge_38', 'shilling_bridge_8', 'shilling_bridge_22',
         'shilling_hijacking_59', 'shilling_hijacking_60', 'shilling_hijacking_57', 'shilling_hijacking_30',
         'shilling_hijacking_107', 'shilling_hijacking_19', 'shilling_hijacking_54', 'shilling_hijacking_133',
         'shilling_hijacking_189', 'shilling_hijacking_63', 'shilling_hijacking_44', 'shilling_bridge_10',
         'shilling_hijacking_103', 'shilling_hijacking_143', 'shilling_bridge_18'],
        ['group_dense_cluster_145', 'group_dense_cluster_59', 'group_badnwagon_58', 'group_dense_cluster_127',
         'group_badnwagon_173', 'group_camouflage_67', 'group_camouflage_155', 'group_camouflage_5',
         'group_badnwagon_133', 'group_camouflage_116', 'group_camouflage_51', 'group_badnwagon_111',
         'shilling_high_deg_4', 'group_camouflage_73', 'group_camouflage_37', 'group_camouflage_129',
         'group_dense_cluster_152', 'shilling_hijacking_29', 'shilling_hijacking_34', 'shilling_hijacking_3',
         'shilling_hijacking_187', 'shilling_hijacking_108', 'shilling_hijacking_186', 'shilling_hijacking_92',
         'shilling_hijacking_134', 'shilling_hijacking_176', 'shilling_hijacking_196', 'shilling_hijacking_35',
         'shilling_hijacking_160', 'shilling_hijacking_45', 'shilling_bridge_9'],
        ['group_badnwagon_28', 'group_camouflage_43', 'group_dense_cluster_89', 'group_dense_cluster_17',
         'group_camouflage_149', 'group_badnwagon_172', 'group_dense_cluster_1', 'group_camouflage_30',
         'group_dense_cluster_45', 'group_camouflage_124', 'group_camouflage_82', 'group_camouflage_39',
         'group_dense_cluster_120', 'group_camouflage_138', 'group_badnwagon_71', 'group_camouflage_166',
         'group_camouflage_106', 'group_dense_cluster_133', 'group_camouflage_85', 'shilling_hijacking_105',
         'shilling_bridge_28'],
        ['group_badnwagon_7', 'group_badnwagon_178', 'group_dense_cluster_39', 'group_dense_cluster_157',
         'group_dense_cluster_61', 'group_camouflage_193', 'group_badnwagon_8', 'group_badnwagon_104',
         'group_badnwagon_39', 'group_camouflage_27', 'group_badnwagon_30', 'group_badnwagon_136',
         'group_camouflage_55', 'group_badnwagon_105', 'group_badnwagon_148', 'group_dense_cluster_105',
         'group_camouflage_146', 'group_badnwagon_51', 'shilling_hijacking_0', 'shilling_hijacking_141',
         'shilling_hijacking_48', 'shilling_hijacking_192', 'shilling_hijacking_162', 'shilling_hijacking_182',
         'shilling_hijacking_17', 'shilling_hijacking_25', 'shilling_hijacking_149', 'shilling_bridge_0',
         'shilling_hijacking_1', 'shilling_hijacking_21', 'shilling_hijacking_195', 'shilling_hijacking_151',
         'shilling_hijacking_147', 'shilling_hijacking_13', 'shilling_hijacking_117', 'shilling_hijacking_159',
         'shilling_hijacking_16', 'shilling_hijacking_152', 'shilling_hijacking_36', 'shilling_bridge_11',
         'shilling_hijacking_74', 'shilling_bridge_27'],
        ['shilling_high_deg_55', 'group_camouflage_151', 'group_camouflage_68', 'group_camouflage_78',
         'group_badnwagon_83', 'group_badnwagon_176', 'group_badnwagon_96', 'group_badnwagon_31', 'group_badnwagon_10',
         'group_camouflage_174', 'group_badnwagon_138', 'shilling_high_deg_3', 'group_badnwagon_154',
         'shilling_high_deg_8', 'group_camouflage_56', 'group_camouflage_139'],
        ['group_dense_cluster_95', 'group_dense_cluster_22', 'group_camouflage_118', 'group_dense_cluster_35',
         'group_badnwagon_49', 'group_dense_cluster_92', 'group_camouflage_26', 'group_badnwagon_14',
         'group_dense_cluster_49', 'group_dense_cluster_161', 'group_camouflage_48', 'group_dense_cluster_107',
         'group_camouflage_119', 'group_badnwagon_94', 'group_camouflage_147', 'group_badnwagon_118',
         'group_badnwagon_62', 'group_camouflage_161', 'shilling_hijacking_12'],
        ['group_badnwagon_135', 'group_camouflage_32', 'group_dense_cluster_194', 'group_badnwagon_163',
         'group_badnwagon_195', 'group_camouflage_64', 'group_dense_cluster_14', 'group_badnwagon_15',
         'group_camouflage_184', 'group_dense_cluster_175', 'group_dense_cluster_38', 'group_camouflage_189',
         'group_camouflage_175', 'shilling_high_deg_30', 'group_dense_cluster_136', 'shilling_high_deg_49',
         'group_camouflage_148', 'group_camouflage_145'],
        ['shilling_high_deg_26', 'group_dense_cluster_28', 'group_dense_cluster_192', 'group_dense_cluster_148',
         'group_badnwagon_55', 'group_badnwagon_153', 'group_badnwagon_33', 'shilling_high_deg_15',
         'group_dense_cluster_36', 'group_camouflage_158', 'shilling_high_deg_14', 'group_camouflage_31',
         'group_dense_cluster_76', 'group_badnwagon_60', 'group_badnwagon_34', 'group_badnwagon_84'],
        ['group_camouflage_40', 'group_dense_cluster_166', 'group_dense_cluster_13', 'group_camouflage_12',
         'group_dense_cluster_15', 'group_dense_cluster_132', 'group_dense_cluster_185', 'group_dense_cluster_69',
         'group_badnwagon_114', 'group_camouflage_153', 'group_camouflage_50', 'group_dense_cluster_141',
         'group_badnwagon_72', 'group_camouflage_74', 'group_camouflage_34', 'group_dense_cluster_174',
         'group_dense_cluster_188', 'group_dense_cluster_77'],
        ['group_dense_cluster_81', 'group_badnwagon_26', 'group_camouflage_36', 'group_camouflage_16',
         'group_camouflage_88', 'group_dense_cluster_78', 'group_dense_cluster_91', 'group_badnwagon_23',
         'group_dense_cluster_189', 'group_badnwagon_79', 'group_badnwagon_29', 'shilling_high_deg_58',
         'group_dense_cluster_64', 'group_dense_cluster_158', 'group_badnwagon_63', 'group_camouflage_128',
         'group_badnwagon_130'],
        ['group_dense_cluster_130', 'group_dense_cluster_113', 'group_dense_cluster_94', 'group_dense_cluster_27',
         'group_dense_cluster_11', 'group_dense_cluster_19', 'group_camouflage_53', 'group_camouflage_90',
         'group_badnwagon_87', 'group_dense_cluster_30', 'group_badnwagon_181', 'group_dense_cluster_41',
         'group_dense_cluster_131', 'group_dense_cluster_122', 'shilling_high_deg_47', 'group_dense_cluster_51'],
        ['group_dense_cluster_37', 'group_dense_cluster_68', 'group_camouflage_79', 'group_badnwagon_169',
         'shilling_high_deg_36', 'group_camouflage_136', 'group_camouflage_19', 'group_badnwagon_52',
         'group_camouflage_45', 'group_camouflage_178', 'group_dense_cluster_165', 'group_badnwagon_110',
         'group_badnwagon_194', 'group_dense_cluster_40', 'group_dense_cluster_111', 'group_camouflage_14',
         'group_dense_cluster_53', 'group_badnwagon_107', 'group_camouflage_86'],
        ['group_badnwagon_9', 'group_badnwagon_158', 'group_camouflage_168', 'group_dense_cluster_86',
         'group_camouflage_33', 'group_camouflage_130', 'group_camouflage_22', 'group_badnwagon_190',
         'group_dense_cluster_140', 'group_dense_cluster_135', 'group_camouflage_197', 'group_dense_cluster_0',
         'group_dense_cluster_154', 'group_dense_cluster_114', 'group_camouflage_17', 'group_badnwagon_27',
         'group_dense_cluster_118', 'group_badnwagon_50'],
        ['group_camouflage_75', 'group_dense_cluster_119', 'group_badnwagon_147', 'group_camouflage_109',
         'group_camouflage_176', 'shilling_high_deg_52', 'group_camouflage_135', 'group_camouflage_24',
         'group_camouflage_125', 'group_camouflage_83', 'group_dense_cluster_196', 'group_dense_cluster_79',
         'group_badnwagon_24', 'group_camouflage_187', 'group_camouflage_91', 'group_camouflage_164',
         'group_badnwagon_182', 'group_badnwagon_192', 'group_dense_cluster_191'],
        ['group_camouflage_63', 'group_dense_cluster_88', 'group_dense_cluster_156', 'group_camouflage_81',
         'group_badnwagon_106', 'group_badnwagon_66', 'group_camouflage_162', 'group_dense_cluster_6',
         'group_badnwagon_197', 'group_camouflage_104', 'group_camouflage_69', 'group_dense_cluster_9',
         'group_dense_cluster_34', 'shilling_high_deg_0', 'group_camouflage_134', 'group_camouflage_60'],
        ['group_badnwagon_38', 'shilling_high_deg_37', 'group_badnwagon_76', 'group_dense_cluster_72',
         'group_badnwagon_108', 'group_badnwagon_157', 'shilling_high_deg_46', 'group_dense_cluster_74',
         'group_camouflage_185', 'shilling_high_deg_25', 'group_camouflage_169', 'group_badnwagon_88',
         'group_camouflage_192', 'group_dense_cluster_190'],
        ['shilling_high_deg_39', 'group_badnwagon_64', 'group_camouflage_121', 'group_badnwagon_160',
         'group_dense_cluster_150', 'group_dense_cluster_179', 'shilling_high_deg_5', 'shilling_high_deg_59',
         'group_dense_cluster_58', 'group_badnwagon_131', 'group_badnwagon_65', 'group_camouflage_181',
         'shilling_high_deg_21', 'group_badnwagon_126', 'group_badnwagon_19'],
        ['group_dense_cluster_162', 'group_camouflage_87', 'group_dense_cluster_106', 'group_dense_cluster_100',
         'group_dense_cluster_103', 'group_badnwagon_170', 'group_dense_cluster_137', 'shilling_high_deg_38',
         'group_dense_cluster_167', 'group_dense_cluster_117', 'group_badnwagon_185', 'group_camouflage_92',
         'group_badnwagon_144', 'group_camouflage_191', 'group_camouflage_93', 'group_dense_cluster_56',
         'group_camouflage_25'],
        ['group_badnwagon_85', 'shilling_high_deg_31', 'group_badnwagon_183', 'group_badnwagon_18',
         'group_dense_cluster_126', 'group_dense_cluster_8', 'group_badnwagon_46', 'group_camouflage_3',
         'group_camouflage_113', 'group_dense_cluster_96', 'group_dense_cluster_62', 'shilling_high_deg_51',
         'group_badnwagon_4', 'group_camouflage_198', 'group_badnwagon_168', 'group_badnwagon_5',
         'group_badnwagon_161'],
        ['group_dense_cluster_184', 'group_badnwagon_123', 'group_camouflage_194', 'shilling_high_deg_11',
         'group_camouflage_120', 'group_dense_cluster_112', 'group_camouflage_94', 'group_badnwagon_101',
         'group_badnwagon_196', 'group_dense_cluster_125', 'group_dense_cluster_109', 'group_dense_cluster_97',
         'group_camouflage_182', 'group_dense_cluster_66', 'shilling_high_deg_43', 'group_dense_cluster_33'],
        ['shilling_high_deg_17', 'group_badnwagon_44', 'group_badnwagon_129', 'group_camouflage_10',
         'group_badnwagon_32', 'group_dense_cluster_42', 'group_camouflage_84', 'group_camouflage_9',
         'group_badnwagon_12', 'group_dense_cluster_65', 'group_camouflage_66', 'group_badnwagon_2',
         'group_badnwagon_36', 'group_camouflage_160', 'group_dense_cluster_55', 'group_badnwagon_86',
         'group_badnwagon_48'],
        ['group_badnwagon_152', 'group_camouflage_58', 'group_badnwagon_117', 'group_camouflage_80',
         'shilling_high_deg_19', 'group_camouflage_100', 'group_dense_cluster_84', 'group_badnwagon_47',
         'group_badnwagon_69', 'group_camouflage_107', 'shilling_high_deg_29', 'shilling_high_deg_56',
         'group_badnwagon_70', 'group_camouflage_179', 'group_badnwagon_128', 'group_badnwagon_37',
         'group_camouflage_77'],
        ['group_dense_cluster_199', 'group_camouflage_157', 'group_camouflage_114', 'group_badnwagon_146',
         'group_dense_cluster_187', 'group_camouflage_195', 'group_camouflage_61', 'group_badnwagon_25',
         'group_camouflage_144', 'group_dense_cluster_198', 'group_camouflage_35', 'group_badnwagon_122',
         'group_badnwagon_45', 'group_dense_cluster_176', 'group_dense_cluster_21', 'group_badnwagon_159'],
        ['group_camouflage_47', 'group_dense_cluster_54', 'group_camouflage_49', 'group_badnwagon_53',
         'group_camouflage_98', 'group_dense_cluster_85', 'group_camouflage_97', 'group_badnwagon_80',
         'group_camouflage_122', 'group_dense_cluster_46', 'group_dense_cluster_93', 'group_dense_cluster_16',
         'group_badnwagon_177', 'group_dense_cluster_82', 'group_dense_cluster_160', 'group_camouflage_123']]
    groups_ative_baselines = [
        ['group_badnwagon_119', 'group_camouflage_152', 'group_badnwagon_164', 'group_dense_cluster_32',
         'group_badnwagon_174', 'shilling_high_deg_54', 'group_camouflage_38', 'shilling_hijacking_89',
         'shilling_high_deg_18', 'group_badnwagon_171', 'shilling_hijacking_77', 'group_dense_cluster_173',
         'group_dense_cluster_2', 'group_dense_cluster_4', 'group_dense_cluster_172', 'shilling_hijacking_94',
         'shilling_hijacking_46', 'shilling_hijacking_132', 'group_badnwagon_43', 'group_camouflage_41',
         'shilling_bridge_25', 'shilling_hijacking_64', 'shilling_hijacking_161', 'shilling_bridge_2',
         'shilling_hijacking_9', 'shilling_hijacking_199', 'shilling_bridge_29', 'shilling_hijacking_62',
         'shilling_hijacking_38', 'shilling_hijacking_7', 'shilling_hijacking_23', 'shilling_hijacking_5',
         'shilling_hijacking_128', 'shilling_hijacking_164', 'shilling_hijacking_126', 'shilling_hijacking_138',
         'shilling_hijacking_24', 'shilling_bridge_13', 'shilling_hijacking_171', 'shilling_hijacking_32',
         'shilling_hijacking_122', 'shilling_hijacking_130', 'shilling_bridge_26', 'shilling_hijacking_137',
         'shilling_hijacking_197', 'shilling_hijacking_55', 'shilling_hijacking_121', 'shilling_hijacking_56',
         'shilling_hijacking_153', 'shilling_hijacking_73', 'shilling_hijacking_146', 'shilling_hijacking_183'],
        ['shilling_high_deg_44', 'group_badnwagon_40', 'group_camouflage_18', 'group_badnwagon_150',
         'group_camouflage_171', 'group_dense_cluster_12', 'group_camouflage_57', 'group_badnwagon_99',
         'group_dense_cluster_73', 'group_camouflage_115', 'group_badnwagon_92', 'group_badnwagon_199',
         'group_dense_cluster_24', 'shilling_high_deg_2', 'group_dense_cluster_170', 'group_badnwagon_77',
         'shilling_hijacking_112', 'shilling_hijacking_118', 'shilling_hijacking_150', 'shilling_hijacking_18'],
        ['group_badnwagon_156', 'shilling_high_deg_10', 'group_badnwagon_116', 'group_dense_cluster_159',
         'group_camouflage_0', 'shilling_high_deg_35', 'group_badnwagon_143', 'group_badnwagon_73',
         'group_camouflage_96', 'group_badnwagon_68', 'group_camouflage_188', 'shilling_high_deg_24',
         'group_dense_cluster_52', 'group_badnwagon_187', 'group_badnwagon_22', 'group_dense_cluster_121',
         'shilling_bridge_31', 'shilling_hijacking_188', 'shilling_hijacking_76', 'shilling_hijacking_119',
         'shilling_hijacking_101', 'shilling_hijacking_27', 'shilling_hijacking_142', 'shilling_hijacking_68',
         'shilling_hijacking_2', 'shilling_hijacking_135', 'shilling_bridge_34', 'shilling_hijacking_37',
         'shilling_bridge_33', 'shilling_hijacking_145', 'shilling_bridge_36', 'shilling_bridge_15',
         'shilling_hijacking_131', 'shilling_bridge_24', 'shilling_hijacking_111'],
        ['group_badnwagon_139', 'group_badnwagon_1', 'group_badnwagon_42', 'group_badnwagon_134', 'group_badnwagon_179',
         'group_dense_cluster_102', 'group_camouflage_59', 'group_badnwagon_188', 'group_camouflage_52',
         'group_dense_cluster_139', 'shilling_high_deg_13', 'shilling_high_deg_27', 'shilling_high_deg_42',
         'group_dense_cluster_153', 'group_camouflage_42'],
        ['shilling_high_deg_57', 'group_badnwagon_186', 'group_dense_cluster_195', 'group_badnwagon_59',
         'group_camouflage_140', 'group_badnwagon_11', 'group_badnwagon_41', 'group_badnwagon_93',
         'group_camouflage_170', 'group_camouflage_199', 'group_dense_cluster_178', 'group_dense_cluster_142',
         'group_dense_cluster_146', 'group_badnwagon_35'],
        ['group_badnwagon_74', 'group_badnwagon_175', 'group_camouflage_15', 'group_camouflage_154',
         'group_camouflage_70', 'group_badnwagon_6', 'group_camouflage_65', 'group_dense_cluster_26',
         'shilling_high_deg_40', 'group_badnwagon_113', 'group_badnwagon_61', 'group_dense_cluster_7',
         'group_dense_cluster_186', 'shilling_high_deg_16', 'shilling_high_deg_48', 'group_badnwagon_54',
         'group_badnwagon_97'],
        ['shilling_high_deg_53', 'group_dense_cluster_181', 'group_dense_cluster_87', 'shilling_high_deg_45',
         'group_dense_cluster_138', 'group_dense_cluster_123', 'group_dense_cluster_63', 'group_dense_cluster_18',
         'group_badnwagon_124', 'group_dense_cluster_151', 'group_badnwagon_3', 'group_dense_cluster_115',
         'group_badnwagon_166', 'group_dense_cluster_98', 'group_badnwagon_82'],
        ['group_camouflage_71', 'group_camouflage_46', 'group_dense_cluster_90', 'group_badnwagon_162',
         'group_camouflage_131', 'group_badnwagon_57', 'group_dense_cluster_128', 'group_badnwagon_151',
         'group_camouflage_103', 'group_camouflage_196', 'group_dense_cluster_124', 'group_dense_cluster_182',
         'group_badnwagon_127', 'group_dense_cluster_44', 'group_badnwagon_112', 'group_dense_cluster_25',
         'group_dense_cluster_48', 'group_camouflage_186'],
        ['group_dense_cluster_145', 'group_dense_cluster_59', 'group_badnwagon_58', 'group_dense_cluster_127',
         'group_badnwagon_173', 'group_camouflage_67', 'group_camouflage_155', 'group_camouflage_5',
         'group_badnwagon_133', 'group_camouflage_116', 'group_camouflage_51', 'group_badnwagon_111',
         'shilling_high_deg_4', 'group_camouflage_73', 'group_camouflage_37', 'group_camouflage_129',
         'group_dense_cluster_152'],
        ['group_badnwagon_28', 'group_camouflage_43', 'group_dense_cluster_89', 'group_dense_cluster_17',
         'group_camouflage_149', 'group_badnwagon_172', 'group_dense_cluster_1', 'group_camouflage_30',
         'group_dense_cluster_45', 'group_camouflage_124', 'group_camouflage_82', 'group_camouflage_39',
         'group_dense_cluster_120', 'group_camouflage_138', 'group_badnwagon_71', 'group_camouflage_166',
         'group_camouflage_106', 'group_dense_cluster_133', 'group_camouflage_85'],
        ['group_badnwagon_7', 'group_badnwagon_178', 'group_dense_cluster_39', 'group_dense_cluster_157',
         'group_dense_cluster_61', 'group_camouflage_193', 'group_badnwagon_8', 'group_badnwagon_104',
         'group_badnwagon_39', 'group_camouflage_27', 'group_badnwagon_30', 'group_badnwagon_136',
         'group_camouflage_55', 'group_badnwagon_105', 'group_badnwagon_148', 'group_dense_cluster_105',
         'group_camouflage_146', 'group_badnwagon_51'],
        ['shilling_high_deg_55', 'group_camouflage_151', 'group_camouflage_68', 'group_camouflage_78',
         'group_badnwagon_83', 'group_badnwagon_176', 'group_badnwagon_96', 'group_badnwagon_31', 'group_badnwagon_10',
         'group_camouflage_174', 'group_badnwagon_138', 'shilling_high_deg_3', 'group_badnwagon_154',
         'shilling_high_deg_8', 'group_camouflage_56'],
        ['group_dense_cluster_95', 'group_dense_cluster_22', 'group_camouflage_118', 'group_dense_cluster_35',
         'group_badnwagon_49', 'group_dense_cluster_92', 'group_camouflage_26']]

    groups_active_mab = [
        ['shilling_hijacking_132', 'group_dense_cluster_63', 'shilling_bridge_13', 'group_dense_cluster_172',
         'group_dense_cluster_36', 'group_dense_cluster_56', 'group_dense_cluster_24', 'group_dense_cluster_7',
         'shilling_bridge_22', 'shilling_hijacking_0', 'shilling_high_deg_55', 'shilling_hijacking_110',
         'shilling_hijacking_21', 'shilling_bridge_7', 'shilling_hijacking_135', 'group_dense_cluster_30',
         'group_badnwagon_106', 'shilling_high_deg_49', 'shilling_hijacking_118', 'group_dense_cluster_44',
         'group_dense_cluster_48', 'shilling_hijacking_123', 'group_dense_cluster_61', 'shilling_high_deg_45',
         'shilling_bridge_31', 'shilling_hijacking_125'],
        ['group_dense_cluster_175', 'group_dense_cluster_190', 'shilling_hijacking_101', 'group_camouflage_64',
         'shilling_hijacking_108', 'group_camouflage_158', 'group_dense_cluster_154', 'shilling_hijacking_148',
         'shilling_hijacking_99', 'shilling_hijacking_111', 'shilling_bridge_32', 'shilling_bridge_11',
         'shilling_bridge_38', 'shilling_high_deg_57', 'shilling_bridge_6', 'shilling_hijacking_126',
         'shilling_hijacking_119', 'shilling_hijacking_117', 'group_camouflage_70', 'group_camouflage_97',
         'group_camouflage_93', 'shilling_hijacking_144', 'group_camouflage_174', 'group_dense_cluster_14',
         'shilling_high_deg_48', 'shilling_hijacking_3', 'group_camouflage_164', 'group_dense_cluster_89',
         'shilling_bridge_18', 'group_dense_cluster_62', 'shilling_bridge_9', 'group_dense_cluster_35',
         'group_dense_cluster_192'],
        ['shilling_hijacking_133', 'group_dense_cluster_173', 'group_dense_cluster_113', 'shilling_bridge_33',
         'group_dense_cluster_15', 'group_camouflage_176', 'shilling_hijacking_131', 'group_dense_cluster_198',
         'shilling_hijacking_156', 'group_camouflage_91', 'group_camouflage_96', 'group_camouflage_19',
         'group_camouflage_181', 'group_camouflage_35', 'group_camouflage_175', 'shilling_bridge_34',
         'group_camouflage_75', 'shilling_bridge_24', 'group_camouflage_155', 'group_dense_cluster_138',
         'group_dense_cluster_1', 'group_dense_cluster_146', 'shilling_hijacking_134', 'shilling_hijacking_145',
         'shilling_bridge_29', 'shilling_hijacking_141', 'group_dense_cluster_186'],
        ['group_dense_cluster_111', 'shilling_bridge_37', 'shilling_bridge_14', 'group_dense_cluster_131',
         'group_camouflage_65', 'shilling_bridge_28', 'group_camouflage_195', 'shilling_hijacking_151',
         'group_camouflage_152', 'group_camouflage_79', 'shilling_hijacking_10', 'group_dense_cluster_139',
         'shilling_bridge_26', 'shilling_hijacking_9', 'group_camouflage_189', 'shilling_hijacking_79',
         'group_camouflage_186', 'shilling_hijacking_12', 'group_dense_cluster_145', 'group_dense_cluster_182',
         'shilling_hijacking_137', 'shilling_hijacking_112', 'group_camouflage_153', 'shilling_hijacking_182',
         'group_dense_cluster_194', 'group_camouflage_31', 'shilling_hijacking_153', 'group_camouflage_32',
         'shilling_hijacking_138', 'group_dense_cluster_142', 'shilling_hijacking_146', 'shilling_hijacking_15',
         'group_camouflage_33', 'group_camouflage_73'],
        ['group_camouflage_88', 'group_dense_cluster_156', 'group_dense_cluster_130', 'group_camouflage_193',
         'group_camouflage_25', 'group_camouflage_184', 'group_camouflage_185', 'group_dense_cluster_152',
         'group_camouflage_50', 'group_camouflage_17', 'group_dense_cluster_115', 'group_camouflage_80',
         'group_camouflage_36', 'shilling_hijacking_186', 'group_dense_cluster_176', 'group_dense_cluster_174',
         'group_dense_cluster_122', 'group_dense_cluster_19', 'group_dense_cluster_165'],
        ['group_dense_cluster_11', 'group_dense_cluster_153', 'group_camouflage_59', 'shilling_hijacking_5',
         'shilling_hijacking_66', 'shilling_hijacking_34', 'shilling_hijacking_41', 'shilling_hijacking_8',
         'group_camouflage_179', 'shilling_hijacking_48', 'shilling_hijacking_78', 'group_camouflage_42',
         'shilling_hijacking_47', 'group_dense_cluster_150', 'shilling_hijacking_68', 'shilling_hijacking_4',
         'group_dense_cluster_187', 'shilling_hijacking_40', 'shilling_hijacking_32', 'group_camouflage_171',
         'group_dense_cluster_120', 'shilling_hijacking_53', 'shilling_hijacking_92', 'shilling_hijacking_72',
         'shilling_hijacking_7', 'group_dense_cluster_179', 'group_dense_cluster_148', 'shilling_bridge_23',
         'shilling_hijacking_31', 'group_dense_cluster_17', 'shilling_hijacking_33', 'group_dense_cluster_137',
         'shilling_hijacking_93', 'group_camouflage_169', 'shilling_hijacking_97', 'group_dense_cluster_121',
         'group_camouflage_188', 'group_camouflage_161'],
        ['shilling_hijacking_69', 'shilling_hijacking_6', 'shilling_hijacking_113', 'shilling_hijacking_81',
         'shilling_hijacking_56', 'group_camouflage_47', 'shilling_hijacking_95', 'group_camouflage_166',
         'shilling_hijacking_43', 'shilling_hijacking_54', 'group_dense_cluster_196', 'shilling_bridge_2',
         'shilling_hijacking_121', 'group_dense_cluster_191', 'group_camouflage_26', 'shilling_hijacking_37',
         'shilling_hijacking_35', 'shilling_hijacking_46', 'group_camouflage_45', 'shilling_hijacking_122',
         'shilling_hijacking_45', 'shilling_hijacking_59', 'group_dense_cluster_151', 'group_camouflage_22',
         'shilling_hijacking_75', 'shilling_hijacking_150', 'shilling_hijacking_13', 'shilling_hijacking_185',
         'group_dense_cluster_126', 'shilling_hijacking_62', 'shilling_hijacking_94', 'group_dense_cluster_140',
         'shilling_bridge_25', 'group_camouflage_61', 'shilling_hijacking_77', 'group_camouflage_46',
         'group_dense_cluster_159', 'group_dense_cluster_109', 'group_camouflage_41', 'group_dense_cluster_128',
         'shilling_hijacking_199', 'shilling_hijacking_177'],
        ['group_camouflage_49', 'group_camouflage_3', 'group_dense_cluster_167', 'group_dense_cluster_195',
         'group_camouflage_55', 'group_camouflage_162', 'group_dense_cluster_141', 'group_camouflage_24',
         'group_camouflage_66', 'group_camouflage_67', 'group_camouflage_39', 'group_dense_cluster_132',
         'group_dense_cluster_185', 'group_dense_cluster_189', 'group_dense_cluster_184', 'group_dense_cluster_136',
         'group_dense_cluster_135', 'group_camouflage_81', 'group_camouflage_98'],
        ['shilling_hijacking_63', 'group_camouflage_86', 'group_dense_cluster_107', 'shilling_hijacking_89',
         'group_camouflage_57', 'group_camouflage_85', 'group_camouflage_69', 'shilling_hijacking_87',
         'shilling_hijacking_74', 'group_dense_cluster_12', 'group_camouflage_82', 'group_dense_cluster_100',
         'group_camouflage_40', 'shilling_hijacking_84', 'group_dense_cluster_162', 'shilling_hijacking_58',
         'shilling_bridge_10', 'shilling_hijacking_67', 'group_camouflage_160', 'group_camouflage_92',
         'group_dense_cluster_161', 'group_camouflage_157', 'shilling_hijacking_91', 'group_dense_cluster_106',
         'group_dense_cluster_188', 'shilling_hijacking_76', 'group_camouflage_52', 'group_dense_cluster_112',
         'shilling_hijacking_64', 'shilling_hijacking_55', 'shilling_hijacking_86', 'shilling_hijacking_70'],
        ['group_camouflage_34', 'group_camouflage_51', 'shilling_hijacking_57', 'group_camouflage_83',
         'shilling_bridge_0', 'shilling_hijacking_36', 'shilling_hijacking_171', 'group_camouflage_30',
         'group_dense_cluster_157', 'shilling_hijacking_176', 'group_camouflage_58', 'group_dense_cluster_18',
         'shilling_hijacking_60', 'group_camouflage_60', 'group_dense_cluster_114', 'group_camouflage_182',
         'group_dense_cluster_118', 'group_camouflage_5', 'shilling_hijacking_44', 'group_camouflage_37',
         'shilling_hijacking_114', 'shilling_bridge_21', 'group_camouflage_87', 'group_dense_cluster_119',
         'group_camouflage_77', 'shilling_hijacking_38', 'shilling_hijacking_49', 'shilling_hijacking_30',
         'group_camouflage_84', 'shilling_hijacking_143'],
        ['group_dense_cluster_0', 'group_camouflage_56', 'shilling_hijacking_130', 'shilling_bridge_36',
         'shilling_hijacking_187', 'shilling_hijacking_73', 'shilling_bridge_15', 'shilling_bridge_8',
         'shilling_hijacking_152', 'shilling_hijacking_128', 'shilling_hijacking_102', 'shilling_hijacking_105',
         'shilling_hijacking_195', 'shilling_bridge_3', 'group_camouflage_199', 'shilling_hijacking_107',
         'shilling_hijacking_127', 'shilling_hijacking_116', 'shilling_hijacking_191', 'group_camouflage_48',
         'group_dense_cluster_166', 'group_camouflage_63', 'shilling_hijacking_161', 'shilling_hijacking_172',
         'shilling_bridge_5', 'shilling_hijacking_129', 'shilling_hijacking_149', 'shilling_hijacking_71',
         'shilling_hijacking_115', 'shilling_hijacking_11', 'group_camouflage_194', 'group_camouflage_178',
         'shilling_hijacking_26', 'group_camouflage_53', 'group_dense_cluster_103', 'shilling_bridge_39',
         'group_dense_cluster_133', 'group_camouflage_18', 'shilling_hijacking_16', 'group_camouflage_74',
         'group_dense_cluster_158', 'shilling_bridge_27', 'group_camouflage_192', 'shilling_hijacking_1',
         'shilling_hijacking_147', 'group_dense_cluster_117', 'group_camouflage_78', 'shilling_hijacking_188',
         'group_camouflage_94', 'group_camouflage_27'],
        ['group_dense_cluster_160', 'shilling_hijacking_29', 'shilling_hijacking_160', 'group_camouflage_170',
         'shilling_hijacking_178', 'shilling_hijacking_183', 'shilling_hijacking_2', 'group_dense_cluster_170',
         'group_dense_cluster_178', 'group_dense_cluster_181', 'shilling_hijacking_28', 'shilling_hijacking_142',
         'group_camouflage_196', 'group_dense_cluster_124', 'group_dense_cluster_102', 'shilling_hijacking_162',
         'group_dense_cluster_13', 'shilling_hijacking_192', 'group_camouflage_9', 'shilling_hijacking_159',
         'shilling_hijacking_23', 'shilling_hijacking_19', 'shilling_hijacking_175', 'shilling_hijacking_18',
         'shilling_hijacking_189', 'shilling_hijacking_196', 'shilling_hijacking_24', 'shilling_hijacking_197',
         'group_dense_cluster_105', 'group_dense_cluster_125', 'shilling_hijacking_27', 'group_camouflage_154',
         'shilling_hijacking_17', 'shilling_bridge_19', 'shilling_hijacking_164', 'shilling_hijacking_180',
         'shilling_hijacking_103', 'shilling_hijacking_25', 'group_dense_cluster_199'],
        ['group_camouflage_43', 'group_camouflage_68', 'group_dense_cluster_123', 'group_camouflage_38',
         'group_camouflage_168', 'group_camouflage_187', 'group_dense_cluster_96', 'group_camouflage_90',
         'group_camouflage_16', 'group_camouflage_14', 'group_dense_cluster_82', 'group_camouflage_198',
         'group_badnwagon_112', 'group_camouflage_71', 'group_camouflage_197', 'group_dense_cluster_16',
         'group_dense_cluster_127', 'group_camouflage_147', 'group_camouflage_191', 'group_dense_cluster_42'],
        ['group_dense_cluster_55', 'group_dense_cluster_53', 'group_dense_cluster_72', 'group_dense_cluster_76',
         'group_dense_cluster_8', 'group_dense_cluster_39', 'shilling_high_deg_18', 'group_dense_cluster_21',
         'shilling_high_deg_0', 'shilling_high_deg_15', 'shilling_high_deg_11', 'group_dense_cluster_81',
         'shilling_high_deg_19'],
        ['shilling_high_deg_56', 'group_dense_cluster_68', 'shilling_high_deg_40', 'group_dense_cluster_9',
         'shilling_high_deg_38', 'shilling_high_deg_58', 'group_dense_cluster_45', 'shilling_high_deg_5',
         'group_dense_cluster_87', 'group_dense_cluster_27', 'shilling_high_deg_2', 'group_camouflage_139',
         'shilling_high_deg_35'],
        ['group_dense_cluster_90', 'shilling_high_deg_59', 'group_dense_cluster_98', 'shilling_high_deg_52',
         'shilling_high_deg_54', 'group_dense_cluster_51', 'group_dense_cluster_84', 'group_dense_cluster_6',
         'group_dense_cluster_49', 'shilling_high_deg_46', 'shilling_high_deg_4', 'group_dense_cluster_46',
         'group_badnwagon_117'],
        ['shilling_high_deg_51', 'group_badnwagon_196', 'group_dense_cluster_2', 'group_dense_cluster_34',
         'group_dense_cluster_78', 'shilling_high_deg_8', 'group_dense_cluster_59', 'group_dense_cluster_40',
         'group_dense_cluster_86', 'group_badnwagon_187', 'group_dense_cluster_54', 'group_dense_cluster_79',
         'shilling_high_deg_10', 'group_dense_cluster_28'],
        ['shilling_high_deg_25', 'group_dense_cluster_92', 'group_dense_cluster_88', 'shilling_high_deg_31',
         'shilling_high_deg_47', 'group_dense_cluster_93', 'group_dense_cluster_25', 'shilling_high_deg_37',
         'group_dense_cluster_22', 'shilling_high_deg_44', 'shilling_high_deg_24', 'group_dense_cluster_26',
         'shilling_high_deg_14'],
        ['group_badnwagon_64', 'group_dense_cluster_33', 'group_badnwagon_86', 'group_camouflage_119',
         'group_camouflage_130', 'group_badnwagon_59', 'shilling_high_deg_30', 'shilling_high_deg_13',
         'group_camouflage_136', 'group_badnwagon_71', 'group_badnwagon_171', 'group_dense_cluster_74',
         'group_badnwagon_96', 'group_dense_cluster_52', 'group_dense_cluster_73', 'shilling_high_deg_29'],
        ['group_badnwagon_80', 'group_camouflage_144', 'group_dense_cluster_64', 'group_badnwagon_136',
         'group_badnwagon_161', 'group_dense_cluster_41', 'group_camouflage_149', 'group_dense_cluster_95',
         'group_dense_cluster_32', 'group_badnwagon_108', 'group_dense_cluster_94', 'group_dense_cluster_66',
         'group_badnwagon_66', 'shilling_high_deg_26', 'group_dense_cluster_38', 'group_badnwagon_105',
         'group_badnwagon_162'],
        ['group_dense_cluster_97', 'group_dense_cluster_77', 'shilling_high_deg_3', 'shilling_high_deg_16',
         'shilling_high_deg_17', 'group_dense_cluster_85', 'shilling_high_deg_21', 'shilling_high_deg_53',
         'shilling_high_deg_27', 'shilling_high_deg_36', 'group_badnwagon_55', 'group_dense_cluster_91'],
        ['group_badnwagon_11', 'group_badnwagon_185', 'group_dense_cluster_69', 'group_badnwagon_127',
         'group_camouflage_113', 'group_badnwagon_25', 'shilling_high_deg_39', 'group_badnwagon_183',
         'group_badnwagon_68', 'group_badnwagon_195', 'group_camouflage_146', 'group_badnwagon_34',
         'group_badnwagon_169', 'group_dense_cluster_58', 'group_dense_cluster_65', 'group_dense_cluster_4',
         'group_badnwagon_156'],
        ['group_badnwagon_188', 'group_badnwagon_73', 'group_badnwagon_175', 'group_badnwagon_197',
         'group_badnwagon_122', 'shilling_high_deg_43', 'group_badnwagon_61', 'group_badnwagon_39',
         'group_badnwagon_124', 'group_badnwagon_134', 'group_badnwagon_31', 'group_badnwagon_179',
         'group_dense_cluster_37', 'shilling_high_deg_42', 'group_badnwagon_36'],
        ['group_badnwagon_45', 'group_camouflage_128', 'group_badnwagon_77', 'group_badnwagon_74',
         'group_badnwagon_154', 'group_badnwagon_111', 'group_badnwagon_10', 'group_camouflage_122',
         'group_badnwagon_186', 'group_badnwagon_51', 'group_badnwagon_146', 'group_badnwagon_97',
         'group_camouflage_121', 'group_badnwagon_23', 'group_badnwagon_43', 'group_badnwagon_3', 'group_badnwagon_65'],
        ['group_camouflage_118', 'group_camouflage_10', 'group_camouflage_116', 'group_badnwagon_182',
         'group_badnwagon_143', 'group_camouflage_129', 'group_badnwagon_18', 'group_badnwagon_139',
         'group_badnwagon_5', 'group_badnwagon_128', 'group_badnwagon_199', 'group_badnwagon_2', 'group_badnwagon_57',
         'group_badnwagon_26', 'group_badnwagon_113', 'group_badnwagon_1', 'group_badnwagon_178', 'group_badnwagon_69',
         'group_badnwagon_147', 'group_badnwagon_123'],
        ['group_badnwagon_104', 'group_badnwagon_129', 'group_badnwagon_133', 'group_badnwagon_99',
         'group_badnwagon_130', 'group_camouflage_120', 'group_badnwagon_148', 'group_badnwagon_85',
         'group_badnwagon_138', 'group_badnwagon_44', 'group_badnwagon_37', 'group_camouflage_115',
         'group_badnwagon_192', 'group_badnwagon_9', 'group_badnwagon_151', 'group_camouflage_148',
         'group_badnwagon_176', 'group_badnwagon_135', 'group_badnwagon_53', 'group_badnwagon_84'],
        ['group_badnwagon_119', 'group_camouflage_124', 'group_badnwagon_46', 'group_badnwagon_87',
         'group_badnwagon_157', 'group_badnwagon_58', 'group_badnwagon_144', 'group_camouflage_125',
         'group_camouflage_135', 'group_badnwagon_70', 'group_badnwagon_47', 'group_badnwagon_150',
         'group_badnwagon_166', 'group_camouflage_134', 'group_badnwagon_164', 'group_badnwagon_126',
         'group_badnwagon_83', 'group_badnwagon_33'],
        ['group_badnwagon_177', 'group_badnwagon_42', 'group_camouflage_100', 'group_badnwagon_63',
         'group_badnwagon_14', 'group_badnwagon_160', 'group_camouflage_123', 'group_badnwagon_92',
         'group_badnwagon_159', 'group_badnwagon_49', 'group_badnwagon_7', 'group_badnwagon_173', 'group_badnwagon_54',
         'group_badnwagon_76', 'group_badnwagon_88', 'group_badnwagon_107', 'group_badnwagon_163'],
        ['group_camouflage_131', 'group_badnwagon_82', 'group_camouflage_107', 'group_badnwagon_50',
         'group_badnwagon_110', 'group_camouflage_106', 'group_camouflage_138', 'group_badnwagon_170',
         'group_badnwagon_8', 'group_camouflage_15', 'group_badnwagon_24', 'group_badnwagon_153',
         'group_camouflage_103', 'group_badnwagon_27', 'group_badnwagon_152', 'group_badnwagon_181',
         'group_badnwagon_19', 'group_badnwagon_38', 'group_badnwagon_131'],
        ['group_badnwagon_190', 'group_camouflage_140', 'group_badnwagon_48', 'group_badnwagon_93',
         'group_badnwagon_118', 'group_badnwagon_168', 'group_badnwagon_172', 'group_badnwagon_30',
         'group_badnwagon_174', 'group_camouflage_145', 'group_badnwagon_79', 'group_badnwagon_158',
         'group_camouflage_0', 'group_badnwagon_72', 'group_badnwagon_116', 'group_badnwagon_15', 'group_badnwagon_12',
         'group_badnwagon_35', 'group_camouflage_109', 'group_camouflage_12'],
        ['group_badnwagon_4', 'group_camouflage_104', 'group_badnwagon_62', 'group_badnwagon_60', 'group_badnwagon_29',
         'group_badnwagon_22', 'group_badnwagon_6', 'group_badnwagon_114', 'group_badnwagon_94', 'group_camouflage_114',
         'group_badnwagon_41', 'group_badnwagon_32', 'group_badnwagon_28', 'group_badnwagon_101', 'group_badnwagon_40',
         'group_badnwagon_52', 'group_badnwagon_194'], ['group_camouflage_151']]
    groups_active_mab = [
        ['shilling_bridge_25', 'shilling_bridge_15', 'group_badnwagon_54', 'shilling_bridge_2', 'group_badnwagon_175',
         'shilling_hijacking_197', 'shilling_bridge_24', 'group_camouflage_155', 'group_badnwagon_164',
         'shilling_bridge_29', 'shilling_hijacking_188', 'shilling_bridge_13', 'group_badnwagon_174',
         'group_badnwagon_124', 'group_dense_cluster_127', 'shilling_bridge_34', 'shilling_bridge_36',
         'shilling_hijacking_94', 'group_dense_cluster_87', 'group_badnwagon_11', 'shilling_bridge_26',
         'group_dense_cluster_35', 'shilling_high_deg_16', 'shilling_bridge_33', 'group_dense_cluster_92',
         'group_badnwagon_74', 'group_camouflage_52', 'group_dense_cluster_151', 'shilling_hijacking_183',
         'group_dense_cluster_95', 'shilling_bridge_31', 'group_camouflage_131', 'group_dense_cluster_102'],
        ['group_dense_cluster_26', 'group_badnwagon_143', 'shilling_hijacking_101', 'group_dense_cluster_128',
         'group_badnwagon_93', 'group_badnwagon_99', 'group_badnwagon_150', 'group_camouflage_5',
         'shilling_high_deg_27', 'group_badnwagon_61', 'group_camouflage_140', 'group_badnwagon_113',
         'shilling_hijacking_9', 'group_badnwagon_92', 'group_camouflage_27', 'group_badnwagon_3',
         'group_dense_cluster_89', 'group_camouflage_55', 'group_camouflage_67', 'shilling_hijacking_171',
         'shilling_hijacking_132', 'group_badnwagon_151'],
        ['shilling_hijacking_137', 'shilling_hijacking_37', 'group_badnwagon_41', 'group_camouflage_116',
         'shilling_hijacking_68', 'shilling_hijacking_121', 'shilling_hijacking_23', 'shilling_hijacking_56',
         'shilling_hijacking_5', 'group_badnwagon_30', 'shilling_hijacking_142', 'shilling_hijacking_111',
         'group_dense_cluster_105', 'shilling_hijacking_62', 'shilling_hijacking_76', 'group_badnwagon_172',
         'shilling_hijacking_138', 'group_dense_cluster_22', 'shilling_hijacking_126', 'shilling_hijacking_161',
         'shilling_hijacking_118', 'shilling_hijacking_135', 'group_camouflage_166', 'group_camouflage_196',
         'shilling_hijacking_199', 'shilling_hijacking_89', 'shilling_hijacking_122', 'group_badnwagon_119',
         'group_dense_cluster_1', 'shilling_hijacking_2', 'shilling_hijacking_73', 'shilling_hijacking_64',
         'shilling_hijacking_38', 'shilling_hijacking_7', 'group_camouflage_59', 'group_camouflage_151',
         'group_badnwagon_73', 'shilling_hijacking_128', 'group_badnwagon_39', 'shilling_hijacking_153',
         'shilling_hijacking_164', 'shilling_hijacking_112', 'shilling_hijacking_27', 'shilling_hijacking_32',
         'group_dense_cluster_121', 'group_badnwagon_77', 'shilling_hijacking_18', 'shilling_hijacking_77',
         'group_dense_cluster_90', 'shilling_hijacking_145', 'shilling_hijacking_130'],
        ['group_camouflage_124', 'group_dense_cluster_115', 'group_badnwagon_134', 'group_camouflage_71',
         'group_badnwagon_176', 'group_dense_cluster_32', 'group_badnwagon_104', 'group_badnwagon_96',
         'group_badnwagon_83', 'group_badnwagon_139', 'shilling_hijacking_131', 'shilling_hijacking_119',
         'shilling_hijacking_150', 'group_camouflage_174', 'shilling_high_deg_10', 'shilling_hijacking_24',
         'group_dense_cluster_152', 'group_camouflage_42', 'group_badnwagon_148', 'shilling_hijacking_55',
         'group_badnwagon_6', 'shilling_hijacking_146', 'group_camouflage_82', 'shilling_hijacking_46',
         'group_camouflage_39', 'group_camouflage_193'],
        ['group_badnwagon_82', 'group_dense_cluster_142', 'group_badnwagon_138', 'group_camouflage_129',
         'group_camouflage_65', 'shilling_high_deg_13', 'group_camouflage_154', 'group_dense_cluster_59',
         'group_badnwagon_22', 'shilling_high_deg_18', 'group_dense_cluster_39', 'group_camouflage_118',
         'group_camouflage_38', 'group_camouflage_146', 'group_camouflage_103', 'group_dense_cluster_181',
         'group_badnwagon_68'],
        ['group_badnwagon_59', 'group_dense_cluster_4', 'group_camouflage_171', 'group_badnwagon_156',
         'group_badnwagon_43', 'group_dense_cluster_2', 'group_badnwagon_49', 'group_camouflage_26',
         'group_badnwagon_136', 'group_dense_cluster_7', 'group_camouflage_15', 'group_camouflage_68',
         'group_dense_cluster_48', 'group_camouflage_37', 'group_dense_cluster_186', 'group_camouflage_96'],
        ['group_camouflage_170', 'group_badnwagon_35', 'group_dense_cluster_178', 'group_camouflage_186',
         'group_dense_cluster_63', 'group_dense_cluster_18', 'group_badnwagon_133', 'group_dense_cluster_182',
         'group_badnwagon_97', 'group_badnwagon_112', 'shilling_high_deg_24', 'group_dense_cluster_145',
         'group_dense_cluster_159', 'group_dense_cluster_138', 'group_badnwagon_166'],
        ['group_camouflage_78', 'group_badnwagon_171', 'group_badnwagon_7', 'group_badnwagon_57', 'group_camouflage_70',
         'group_camouflage_30', 'group_badnwagon_28', 'group_dense_cluster_124', 'group_camouflage_46',
         'group_dense_cluster_170', 'group_camouflage_56', 'group_badnwagon_8', 'group_badnwagon_40',
         'group_badnwagon_173', 'group_badnwagon_186', 'group_dense_cluster_123', 'group_badnwagon_127',
         'group_dense_cluster_153'],
        ['group_camouflage_73', 'group_camouflage_0', 'group_dense_cluster_173', 'group_badnwagon_58',
         'group_dense_cluster_73', 'group_dense_cluster_139', 'group_badnwagon_187', 'group_badnwagon_116',
         'group_dense_cluster_44', 'group_camouflage_43', 'group_dense_cluster_61', 'group_camouflage_57',
         'group_camouflage_188', 'group_dense_cluster_157', 'group_camouflage_18', 'group_badnwagon_179'],
        ['group_badnwagon_51', 'group_badnwagon_31', 'group_dense_cluster_25', 'shilling_high_deg_2',
         'group_dense_cluster_52', 'group_dense_cluster_17', 'group_camouflage_41', 'group_badnwagon_71',
         'group_camouflage_152', 'group_badnwagon_199', 'group_dense_cluster_133', 'group_badnwagon_42',
         'group_dense_cluster_24', 'group_dense_cluster_45', 'group_badnwagon_188', 'group_dense_cluster_98',
         'group_badnwagon_10'],
        ['group_camouflage_138', 'group_dense_cluster_120', 'group_dense_cluster_12', 'group_camouflage_85',
         'group_camouflage_149', 'group_camouflage_106', 'group_badnwagon_154', 'group_dense_cluster_146',
         'group_dense_cluster_172', 'group_badnwagon_162', 'group_dense_cluster_195', 'group_badnwagon_1',
         'group_camouflage_115', 'group_badnwagon_111', 'group_badnwagon_105'],
        ['shilling_high_deg_48', 'shilling_high_deg_54', 'shilling_high_deg_45', 'shilling_high_deg_53',
         'group_camouflage_51', 'group_camouflage_199', 'shilling_high_deg_4', 'group_badnwagon_178',
         'shilling_high_deg_42', 'shilling_high_deg_8', 'shilling_high_deg_57'],
        ['shilling_high_deg_44', 'shilling_high_deg_55', 'shilling_high_deg_35', 'shilling_high_deg_3',
         'shilling_high_deg_40'], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],
        [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
    groups_active_mab = [
        ['shilling_bridge_25', 'group_dense_cluster_87', 'shilling_hijacking_183', 'group_badnwagon_174',
         'group_badnwagon_74', 'shilling_bridge_13', 'shilling_bridge_2', 'shilling_bridge_24', 'group_badnwagon_124',
         'group_dense_cluster_92', 'shilling_high_deg_16', 'group_dense_cluster_35', 'shilling_hijacking_197',
         'shilling_hijacking_188', 'shilling_hijacking_94', 'group_badnwagon_164', 'group_dense_cluster_127',
         'shilling_bridge_15', 'group_dense_cluster_151', 'shilling_bridge_26', 'group_dense_cluster_102',
         'group_badnwagon_11', 'shilling_bridge_29', 'group_camouflage_155', 'group_camouflage_52',
         'shilling_bridge_36', 'shilling_bridge_33', 'shilling_bridge_34', 'group_camouflage_131',
         'group_badnwagon_175', 'group_dense_cluster_95', 'group_badnwagon_54', 'shilling_bridge_31'],
        ['shilling_hijacking_118', 'group_dense_cluster_22', 'shilling_hijacking_112', 'shilling_hijacking_56',
         'group_badnwagon_136', 'shilling_hijacking_142', 'group_camouflage_27', 'group_dense_cluster_89',
         'group_camouflage_67', 'group_badnwagon_150', 'shilling_hijacking_9', 'shilling_hijacking_132',
         'group_badnwagon_113', 'shilling_hijacking_153', 'shilling_hijacking_2', 'shilling_hijacking_135',
         'group_badnwagon_93', 'group_camouflage_55', 'group_dense_cluster_2', 'shilling_hijacking_37',
         'group_badnwagon_143', 'group_badnwagon_151', 'group_dense_cluster_26', 'group_camouflage_5',
         'shilling_hijacking_77', 'shilling_high_deg_27', 'shilling_hijacking_171', 'group_badnwagon_61',
         'group_badnwagon_3', 'shilling_hijacking_101', 'shilling_hijacking_161', 'shilling_hijacking_130'],
        ['shilling_hijacking_121', 'shilling_hijacking_23', 'shilling_hijacking_126', 'shilling_hijacking_38',
         'shilling_hijacking_62', 'shilling_hijacking_111', 'shilling_hijacking_7', 'shilling_hijacking_128',
         'shilling_hijacking_73', 'group_badnwagon_139', 'shilling_hijacking_137', 'group_camouflage_59',
         'shilling_hijacking_199', 'group_badnwagon_39', 'shilling_hijacking_68', 'group_badnwagon_119',
         'group_dense_cluster_90', 'shilling_hijacking_164', 'group_camouflage_166', 'group_badnwagon_77',
         'shilling_hijacking_138', 'shilling_hijacking_64', 'group_dense_cluster_105', 'group_badnwagon_30',
         'group_camouflage_116', 'group_dense_cluster_1', 'shilling_hijacking_5', 'group_badnwagon_22',
         'group_camouflage_124', 'shilling_hijacking_32', 'group_badnwagon_172', 'group_badnwagon_35',
         'shilling_hijacking_27', 'shilling_hijacking_145', 'group_dense_cluster_121', 'shilling_hijacking_18',
         'group_badnwagon_73'],
        ['group_camouflage_129', 'group_camouflage_193', 'group_badnwagon_6', 'group_camouflage_146',
         'shilling_hijacking_76', 'group_dense_cluster_4', 'group_badnwagon_99', 'group_dense_cluster_152',
         'group_camouflage_82', 'group_camouflage_42', 'shilling_hijacking_89', 'shilling_high_deg_13',
         'group_badnwagon_176', 'group_camouflage_71', 'group_dense_cluster_32', 'group_badnwagon_96',
         'group_camouflage_174', 'group_dense_cluster_142', 'group_camouflage_196', 'group_dense_cluster_115'],
        ['group_camouflage_118', 'shilling_hijacking_146', 'group_dense_cluster_39', 'group_dense_cluster_123',
         'group_dense_cluster_181', 'group_camouflage_151', 'group_badnwagon_148', 'group_camouflage_103',
         'shilling_high_deg_10', 'shilling_hijacking_55', 'group_camouflage_140', 'group_badnwagon_104',
         'group_dense_cluster_128', 'shilling_hijacking_119', 'group_badnwagon_166', 'shilling_hijacking_24',
         'shilling_hijacking_46', 'shilling_hijacking_150', 'group_badnwagon_134', 'group_badnwagon_68',
         'group_dense_cluster_59', 'group_badnwagon_43', 'shilling_hijacking_131', 'group_badnwagon_83',
         'shilling_hijacking_122'],
        ['group_badnwagon_171', 'group_badnwagon_156', 'group_camouflage_15', 'group_camouflage_154',
         'group_camouflage_170', 'group_camouflage_39', 'group_dense_cluster_178', 'group_dense_cluster_153',
         'shilling_high_deg_18', 'group_camouflage_65', 'group_dense_cluster_145', 'group_badnwagon_92',
         'group_camouflage_73', 'group_camouflage_0', 'group_camouflage_38', 'group_camouflage_57',
         'group_badnwagon_41', 'group_badnwagon_173'],
        ['group_dense_cluster_48', 'group_dense_cluster_63', 'group_badnwagon_112', 'group_badnwagon_187',
         'group_badnwagon_97', 'group_badnwagon_59', 'group_dense_cluster_159', 'group_camouflage_96',
         'group_camouflage_68', 'group_badnwagon_133', 'group_dense_cluster_138', 'group_camouflage_30',
         'group_dense_cluster_182', 'group_badnwagon_116', 'group_badnwagon_138', 'group_camouflage_26',
         'group_camouflage_186'],
        ['group_dense_cluster_73', 'group_camouflage_46', 'group_camouflage_56', 'group_dense_cluster_139',
         'shilling_high_deg_24', 'group_badnwagon_186', 'group_dense_cluster_186', 'group_camouflage_171',
         'group_dense_cluster_18', 'group_badnwagon_49', 'group_badnwagon_127', 'group_dense_cluster_173',
         'group_badnwagon_40', 'group_dense_cluster_7', 'group_dense_cluster_170'],
        ['group_dense_cluster_124', 'group_camouflage_70', 'group_camouflage_37', 'group_camouflage_43',
         'group_dense_cluster_157', 'group_badnwagon_10', 'group_badnwagon_8', 'group_dense_cluster_25',
         'group_badnwagon_82', 'group_badnwagon_199', 'group_dense_cluster_44', 'group_badnwagon_71',
         'group_badnwagon_7', 'group_badnwagon_51', 'group_badnwagon_31', 'group_dense_cluster_45',
         'group_badnwagon_28', 'group_camouflage_18', 'group_dense_cluster_133'],
        ['group_badnwagon_111', 'group_dense_cluster_61', 'group_badnwagon_58', 'group_dense_cluster_12',
         'group_dense_cluster_52', 'group_camouflage_41', 'group_dense_cluster_98', 'group_camouflage_188',
         'group_camouflage_106', 'group_camouflage_152', 'shilling_high_deg_2', 'group_badnwagon_42',
         'group_dense_cluster_195', 'group_camouflage_85', 'group_camouflage_149'],
        ['group_dense_cluster_24', 'group_badnwagon_1', 'group_badnwagon_188', 'group_dense_cluster_172',
         'group_badnwagon_154', 'group_badnwagon_162', 'group_camouflage_199', 'group_badnwagon_179',
         'group_badnwagon_105', 'group_dense_cluster_120', 'group_camouflage_78', 'group_camouflage_115',
         'group_badnwagon_57', 'group_dense_cluster_146', 'group_dense_cluster_17', 'group_camouflage_138'],
        ['shilling_high_deg_57', 'group_badnwagon_178', 'group_camouflage_51', 'shilling_high_deg_42',
         'shilling_high_deg_35', 'shilling_high_deg_4', 'shilling_high_deg_8', 'shilling_high_deg_53',
         'shilling_high_deg_45', 'shilling_high_deg_48', 'shilling_high_deg_54'],
        ['shilling_high_deg_44', 'shilling_high_deg_3', 'shilling_high_deg_40', 'shilling_high_deg_55'], [], [], [], [],
        [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],
        [], [], [], [], []]
    # groups_active_mab = [['shilling_hijacking_183', 'shilling_hijacking_101', 'shilling_hijacking_146', 'group_dense_cluster_151', 'group_camouflage_124', 'group_camouflage_196', 'shilling_hijacking_27', 'shilling_hijacking_55', 'shilling_hijacking_9', 'group_badnwagon_30', 'group_badnwagon_172', 'shilling_hijacking_94', 'shilling_hijacking_188', 'shilling_hijacking_76', 'shilling_hijacking_142', 'shilling_hijacking_18', 'shilling_hijacking_119', 'shilling_hijacking_130', 'group_badnwagon_77', 'shilling_high_deg_53', 'group_dense_cluster_90', 'shilling_hijacking_126', 'shilling_hijacking_171', 'shilling_hijacking_24', 'shilling_hijacking_118', 'shilling_hijacking_128', 'group_dense_cluster_63', 'shilling_hijacking_137', 'shilling_hijacking_46', 'group_dense_cluster_1', 'group_dense_cluster_105', 'shilling_high_deg_55', 'shilling_hijacking_56', 'shilling_hijacking_64', 'group_camouflage_71', 'shilling_hijacking_32', 'group_camouflage_116', 'group_camouflage_151'], ['group_badnwagon_119', 'group_badnwagon_138', 'shilling_high_deg_16', 'shilling_hijacking_77', 'shilling_hijacking_121', 'group_badnwagon_124', 'group_badnwagon_54', 'shilling_bridge_24', 'shilling_hijacking_153', 'group_dense_cluster_121', 'group_badnwagon_3', 'shilling_bridge_13', 'group_badnwagon_127', 'shilling_hijacking_122', 'group_camouflage_59', 'group_camouflage_103', 'shilling_hijacking_150', 'shilling_bridge_33', 'shilling_hijacking_132', 'shilling_bridge_15', 'shilling_bridge_2', 'shilling_hijacking_135', 'shilling_hijacking_138', 'shilling_hijacking_145', 'shilling_hijacking_199', 'shilling_bridge_25', 'group_badnwagon_11', 'shilling_hijacking_197', 'shilling_hijacking_89', 'group_camouflage_166', 'shilling_hijacking_5', 'shilling_hijacking_161', 'shilling_hijacking_2', 'group_dense_cluster_127', 'shilling_bridge_29', 'group_dense_cluster_92', 'group_badnwagon_139', 'shilling_bridge_26', 'shilling_bridge_34', 'shilling_hijacking_23', 'shilling_bridge_36', 'shilling_hijacking_7', 'group_camouflage_131', 'shilling_hijacking_62', 'group_badnwagon_74', 'shilling_bridge_31', 'shilling_hijacking_73', 'shilling_hijacking_38', 'shilling_hijacking_68', 'shilling_hijacking_112', 'shilling_hijacking_111', 'group_badnwagon_111', 'group_camouflage_155', 'shilling_hijacking_164', 'shilling_hijacking_131', 'shilling_hijacking_37'], ['group_dense_cluster_32', 'group_badnwagon_6', 'group_dense_cluster_115', 'group_dense_cluster_152', 'group_camouflage_171', 'group_badnwagon_176', 'group_badnwagon_39', 'group_camouflage_118', 'group_camouflage_193', 'group_camouflage_43', 'group_badnwagon_104', 'shilling_high_deg_10', 'group_badnwagon_96', 'group_badnwagon_83', 'group_dense_cluster_39', 'group_badnwagon_148', 'group_badnwagon_73', 'group_camouflage_82'], ['group_badnwagon_134', 'group_badnwagon_57', 'group_badnwagon_59', 'group_dense_cluster_7', 'group_dense_cluster_146', 'group_dense_cluster_95', 'group_camouflage_96', 'group_camouflage_5', 'group_dense_cluster_186', 'group_camouflage_37', 'group_dense_cluster_48', 'group_dense_cluster_22', 'group_camouflage_78', 'group_badnwagon_22', 'group_dense_cluster_142', 'group_dense_cluster_128'], ['group_camouflage_129', 'group_badnwagon_112', 'group_camouflage_55', 'group_dense_cluster_102', 'group_badnwagon_61', 'group_badnwagon_175', 'group_badnwagon_143', 'group_camouflage_52', 'group_badnwagon_97', 'group_badnwagon_166', 'group_dense_cluster_35', 'group_dense_cluster_2', 'group_camouflage_42', 'group_badnwagon_136', 'group_badnwagon_151', 'group_camouflage_30', 'group_camouflage_26', 'group_dense_cluster_26', 'group_camouflage_174'], ['group_badnwagon_68', 'group_camouflage_15', 'group_camouflage_70', 'group_camouflage_188', 'group_dense_cluster_4', 'group_badnwagon_40', 'group_badnwagon_116', 'group_badnwagon_179', 'group_camouflage_146', 'shilling_high_deg_27', 'group_camouflage_138', 'group_badnwagon_58', 'group_dense_cluster_159', 'shilling_high_deg_13', 'group_dense_cluster_17'], ['group_camouflage_56', 'group_camouflage_170', 'group_badnwagon_133', 'group_camouflage_39', 'group_dense_cluster_170', 'group_camouflage_68', 'group_badnwagon_41', 'group_badnwagon_35', 'group_badnwagon_28', 'group_camouflage_186', 'group_dense_cluster_182', 'group_camouflage_154', 'group_dense_cluster_138', 'group_dense_cluster_145', 'group_badnwagon_49', 'shilling_high_deg_24'], ['group_camouflage_46', 'group_camouflage_41', 'group_dense_cluster_178', 'group_badnwagon_150', 'group_badnwagon_186', 'group_badnwagon_82', 'group_dense_cluster_59', 'group_badnwagon_43', 'group_badnwagon_187', 'group_badnwagon_93', 'group_badnwagon_42', 'group_dense_cluster_73', 'group_badnwagon_164', 'group_badnwagon_105', 'group_badnwagon_174', 'shilling_high_deg_18', 'group_dense_cluster_181'], ['group_dense_cluster_24', 'group_badnwagon_1', 'group_badnwagon_171', 'group_dense_cluster_139', 'group_badnwagon_7', 'group_badnwagon_8', 'group_dense_cluster_89', 'group_dense_cluster_25', 'group_dense_cluster_173', 'group_dense_cluster_61', 'group_badnwagon_156', 'group_dense_cluster_18', 'group_camouflage_65', 'group_dense_cluster_44', 'group_badnwagon_188', 'group_dense_cluster_124', 'group_badnwagon_173'], ['group_camouflage_57', 'group_dense_cluster_153', 'group_dense_cluster_87', 'group_camouflage_0', 'group_camouflage_67', 'group_camouflage_73', 'group_camouflage_140', 'group_dense_cluster_123', 'group_badnwagon_113', 'group_camouflage_152', 'shilling_high_deg_2', 'group_camouflage_18', 'group_camouflage_27', 'group_badnwagon_31', 'group_badnwagon_99', 'group_badnwagon_92', 'group_badnwagon_51', 'group_camouflage_38'], ['group_dense_cluster_120', 'group_badnwagon_162', 'group_dense_cluster_12', 'group_badnwagon_10', 'group_badnwagon_199', 'group_dense_cluster_133', 'group_camouflage_106', 'group_dense_cluster_52', 'group_badnwagon_154', 'group_dense_cluster_157', 'group_dense_cluster_98', 'group_dense_cluster_45', 'group_badnwagon_71', 'group_badnwagon_178', 'group_camouflage_85', 'group_camouflage_51', 'group_camouflage_149'], ['shilling_high_deg_4', 'shilling_high_deg_54', 'shilling_high_deg_42', 'group_dense_cluster_172', 'shilling_high_deg_45', 'group_dense_cluster_195', 'group_camouflage_115', 'shilling_high_deg_57', 'shilling_high_deg_48', 'shilling_high_deg_8', 'group_camouflage_199'], ['shilling_high_deg_40', 'shilling_high_deg_3', 'shilling_high_deg_44', 'shilling_high_deg_35'], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
    groups_active_mab = [
        ['group_dense_cluster_105', 'shilling_hijacking_183', 'shilling_hijacking_56', 'shilling_hijacking_101',
         'shilling_hijacking_76', 'shilling_hijacking_119', 'shilling_hijacking_137', 'shilling_hijacking_128',
         'shilling_hijacking_118', 'shilling_hijacking_9', 'shilling_hijacking_126', 'shilling_hijacking_130',
         'shilling_hijacking_24', 'shilling_hijacking_188', 'shilling_hijacking_142', 'shilling_hijacking_171',
         'group_badnwagon_172', 'shilling_hijacking_18', 'group_camouflage_196', 'group_camouflage_151',
         'shilling_hijacking_27', 'shilling_hijacking_55', 'group_dense_cluster_90', 'group_dense_cluster_1',
         'shilling_hijacking_64', 'shilling_hijacking_32', 'shilling_high_deg_55', 'shilling_hijacking_46',
         'group_badnwagon_77', 'shilling_hijacking_94', 'shilling_hijacking_146', 'group_dense_cluster_63',
         'group_camouflage_116', 'shilling_high_deg_53', 'group_badnwagon_30', 'group_dense_cluster_151',
         'group_camouflage_71', 'group_camouflage_124'],
        ['group_badnwagon_74', 'group_badnwagon_111', 'shilling_hijacking_121', 'group_camouflage_59',
         'shilling_bridge_15', 'shilling_bridge_36', 'shilling_bridge_26', 'group_camouflage_155',
         'shilling_hijacking_161', 'group_badnwagon_124', 'shilling_hijacking_7', 'shilling_hijacking_164',
         'shilling_bridge_2', 'group_badnwagon_11', 'shilling_hijacking_112', 'shilling_bridge_13',
         'shilling_hijacking_23', 'group_dense_cluster_121', 'shilling_hijacking_153', 'shilling_hijacking_89',
         'shilling_bridge_31', 'shilling_hijacking_2', 'shilling_bridge_33', 'shilling_bridge_24',
         'group_dense_cluster_92', 'group_dense_cluster_73', 'shilling_high_deg_16', 'group_badnwagon_127',
         'shilling_hijacking_131', 'shilling_hijacking_5', 'shilling_bridge_34', 'group_dense_cluster_127',
         'shilling_hijacking_132', 'shilling_hijacking_73', 'group_badnwagon_138', 'shilling_bridge_29',
         'shilling_hijacking_135', 'group_badnwagon_3', 'shilling_hijacking_122', 'shilling_hijacking_150',
         'shilling_hijacking_62', 'shilling_hijacking_38', 'shilling_hijacking_77', 'group_badnwagon_139',
         'shilling_bridge_25', 'shilling_hijacking_111', 'group_camouflage_166', 'shilling_hijacking_68',
         'shilling_hijacking_197', 'group_camouflage_131', 'shilling_hijacking_138', 'shilling_hijacking_37',
         'group_badnwagon_54'],
        ['group_dense_cluster_115', 'group_badnwagon_134', 'group_camouflage_43', 'group_dense_cluster_128',
         'group_badnwagon_96', 'group_dense_cluster_32', 'group_camouflage_193', 'group_badnwagon_39',
         'group_badnwagon_176', 'group_badnwagon_119', 'shilling_high_deg_10', 'group_badnwagon_83',
         'group_camouflage_171', 'group_dense_cluster_35', 'group_camouflage_5', 'group_dense_cluster_95',
         'group_dense_cluster_22', 'group_badnwagon_148'],
        ['group_dense_cluster_2', 'group_dense_cluster_102', 'group_badnwagon_61', 'group_badnwagon_143',
         'group_dense_cluster_26', 'shilling_hijacking_199', 'group_badnwagon_164', 'group_camouflage_30',
         'group_camouflage_55', 'group_badnwagon_113', 'group_badnwagon_175', 'group_badnwagon_151',
         'group_badnwagon_93', 'group_badnwagon_136', 'group_camouflage_52', 'group_dense_cluster_89',
         'group_camouflage_26', 'group_badnwagon_150', 'group_camouflage_129', 'shilling_hijacking_145'],
        ['group_badnwagon_104', 'group_badnwagon_166', 'group_camouflage_174', 'group_badnwagon_6',
         'group_camouflage_68', 'group_camouflage_118', 'group_dense_cluster_142', 'group_camouflage_146',
         'group_badnwagon_97', 'group_camouflage_96', 'group_dense_cluster_4', 'shilling_high_deg_27',
         'group_dense_cluster_138', 'group_dense_cluster_152', 'group_camouflage_82', 'group_badnwagon_35'],
        ['group_badnwagon_112', 'group_badnwagon_174', 'group_camouflage_154', 'group_camouflage_42',
         'group_dense_cluster_145', 'group_camouflage_39', 'group_dense_cluster_181', 'group_badnwagon_133',
         'group_dense_cluster_170', 'shilling_high_deg_24', 'shilling_high_deg_18', 'group_badnwagon_73',
         'group_dense_cluster_182', 'group_camouflage_103', 'group_badnwagon_22', 'group_camouflage_170'],
        ['group_badnwagon_187', 'group_dense_cluster_87', 'group_camouflage_67', 'group_badnwagon_43',
         'group_dense_cluster_7', 'group_camouflage_46', 'group_camouflage_140', 'group_camouflage_73',
         'group_camouflage_41', 'group_badnwagon_116', 'shilling_high_deg_13', 'group_badnwagon_92',
         'group_badnwagon_41', 'group_camouflage_56', 'group_badnwagon_171', 'group_camouflage_188',
         'group_dense_cluster_178', 'group_camouflage_27'],
        ['group_badnwagon_173', 'group_camouflage_186', 'group_badnwagon_82', 'group_camouflage_70',
         'group_dense_cluster_186', 'group_camouflage_65', 'group_dense_cluster_124', 'group_camouflage_37',
         'group_dense_cluster_59', 'group_dense_cluster_44', 'group_dense_cluster_159', 'group_badnwagon_49',
         'group_dense_cluster_48', 'group_badnwagon_57', 'group_badnwagon_186', 'group_dense_cluster_39',
         'group_dense_cluster_139'],
        ['group_dense_cluster_173', 'group_badnwagon_40', 'group_badnwagon_28', 'group_badnwagon_7',
         'group_camouflage_78', 'group_camouflage_18', 'group_badnwagon_31', 'group_camouflage_38',
         'group_badnwagon_156', 'group_badnwagon_59', 'group_dense_cluster_24', 'group_badnwagon_51',
         'group_camouflage_15', 'group_badnwagon_42', 'group_badnwagon_58', 'group_dense_cluster_18',
         'group_badnwagon_8'],
        ['group_dense_cluster_25', 'group_badnwagon_68', 'group_badnwagon_10', 'group_camouflage_0',
         'group_dense_cluster_61', 'group_dense_cluster_123', 'group_badnwagon_188', 'group_dense_cluster_133',
         'group_camouflage_57', 'group_camouflage_149', 'group_badnwagon_99', 'group_camouflage_152',
         'group_badnwagon_105', 'group_dense_cluster_153', 'group_badnwagon_199', 'group_dense_cluster_17',
         'shilling_high_deg_2'],
        ['group_dense_cluster_52', 'group_dense_cluster_45', 'group_dense_cluster_195', 'group_dense_cluster_146',
         'group_camouflage_85', 'group_badnwagon_179', 'group_camouflage_138', 'group_camouflage_51',
         'group_dense_cluster_157', 'group_dense_cluster_12', 'group_badnwagon_71', 'group_dense_cluster_120',
         'group_dense_cluster_98', 'group_camouflage_106', 'group_badnwagon_154'],
        ['group_dense_cluster_172', 'group_badnwagon_162', 'shilling_high_deg_8', 'shilling_high_deg_57',
         'shilling_high_deg_42', 'shilling_high_deg_45', 'group_badnwagon_178', 'shilling_high_deg_4',
         'group_badnwagon_1', 'group_camouflage_115', 'group_camouflage_199', 'shilling_high_deg_54',
         'shilling_high_deg_48'],
        ['shilling_high_deg_35', 'shilling_high_deg_3', 'shilling_high_deg_40', 'shilling_high_deg_44'], [], [], [], [],
        [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],
        [], [], [], [], []]
    groups_active_mab = [
        ['group_camouflage_71', 'shilling_hijacking_101', 'shilling_hijacking_56', 'shilling_hijacking_46',
         'shilling_hijacking_142', 'group_badnwagon_172', 'shilling_hijacking_171', 'shilling_hijacking_128',
         'shilling_hijacking_24', 'shilling_high_deg_53', 'group_dense_cluster_151', 'shilling_hijacking_126',
         'shilling_hijacking_76', 'group_badnwagon_30', 'group_camouflage_116', 'group_camouflage_196',
         'shilling_hijacking_27', 'group_dense_cluster_105', 'shilling_hijacking_137', 'group_dense_cluster_1',
         'shilling_hijacking_183', 'shilling_hijacking_146', 'shilling_hijacking_18', 'group_badnwagon_77',
         'group_dense_cluster_63', 'group_camouflage_124', 'shilling_hijacking_94', 'shilling_high_deg_55',
         'shilling_hijacking_55', 'shilling_hijacking_64', 'group_camouflage_151', 'shilling_hijacking_9',
         'shilling_hijacking_188', 'shilling_hijacking_32', 'shilling_hijacking_119', 'shilling_hijacking_130',
         'group_dense_cluster_90', 'shilling_hijacking_118'],
        ['shilling_hijacking_197', 'group_badnwagon_139', 'shilling_hijacking_112', 'shilling_bridge_33',
         'shilling_hijacking_2', 'shilling_hijacking_77', 'shilling_hijacking_138', 'shilling_bridge_24',
         'shilling_hijacking_38', 'shilling_bridge_15', 'group_dense_cluster_22', 'group_dense_cluster_92',
         'group_badnwagon_74', 'group_camouflage_155', 'group_badnwagon_127', 'group_camouflage_59',
         'group_dense_cluster_121', 'shilling_high_deg_16', 'group_badnwagon_124', 'group_dense_cluster_138',
         'shilling_hijacking_5', 'shilling_bridge_31', 'shilling_hijacking_62', 'shilling_bridge_25',
         'shilling_bridge_13', 'shilling_hijacking_89', 'shilling_bridge_26', 'group_badnwagon_119',
         'group_badnwagon_39', 'shilling_bridge_36', 'group_camouflage_131', 'shilling_hijacking_68',
         'group_badnwagon_138', 'group_badnwagon_111', 'shilling_bridge_34', 'group_camouflage_103',
         'shilling_hijacking_7', 'shilling_bridge_29', 'group_camouflage_166', 'group_badnwagon_54',
         'shilling_hijacking_23', 'shilling_hijacking_132', 'shilling_bridge_2'],
        ['shilling_high_deg_10', 'group_badnwagon_73', 'shilling_hijacking_122', 'group_badnwagon_3',
         'group_badnwagon_148', 'group_camouflage_174', 'shilling_hijacking_73', 'group_dense_cluster_7',
         'group_dense_cluster_127', 'group_dense_cluster_152', 'group_dense_cluster_48', 'group_badnwagon_83',
         'group_camouflage_171', 'group_badnwagon_22', 'group_dense_cluster_32', 'group_camouflage_70',
         'shilling_hijacking_161', 'group_camouflage_118', 'shilling_hijacking_199', 'shilling_hijacking_37',
         'shilling_hijacking_153', 'group_camouflage_193', 'group_badnwagon_134'],
        ['group_camouflage_154', 'group_badnwagon_104', 'group_badnwagon_6', 'group_dense_cluster_159',
         'group_badnwagon_175', 'group_badnwagon_96', 'group_dense_cluster_39', 'shilling_hijacking_111',
         'shilling_hijacking_131', 'group_dense_cluster_115', 'group_camouflage_68', 'shilling_hijacking_121',
         'group_camouflage_5', 'shilling_hijacking_135', 'shilling_high_deg_24', 'group_badnwagon_97',
         'shilling_hijacking_164', 'group_badnwagon_41', 'shilling_hijacking_145', 'group_camouflage_43',
         'shilling_hijacking_150', 'group_dense_cluster_186', 'group_camouflage_42', 'group_badnwagon_57'],
        ['group_badnwagon_40', 'group_badnwagon_176', 'group_camouflage_96', 'group_badnwagon_59', 'group_badnwagon_28',
         'group_dense_cluster_17', 'group_badnwagon_105', 'group_camouflage_186', 'group_dense_cluster_146',
         'group_badnwagon_58', 'group_camouflage_82', 'group_camouflage_78', 'group_badnwagon_112',
         'group_camouflage_37', 'group_badnwagon_1', 'group_badnwagon_49', 'group_dense_cluster_142'],
        ['group_camouflage_41', 'group_dense_cluster_128', 'group_dense_cluster_35', 'group_camouflage_129',
         'group_dense_cluster_2', 'group_dense_cluster_182', 'group_camouflage_26', 'shilling_high_deg_18',
         'group_dense_cluster_4', 'group_camouflage_138', 'group_badnwagon_68', 'group_camouflage_170',
         'group_badnwagon_166', 'group_badnwagon_42', 'group_badnwagon_82', 'shilling_high_deg_27'],
        ['group_badnwagon_11', 'group_dense_cluster_95', 'group_dense_cluster_178', 'group_camouflage_146',
         'group_badnwagon_133', 'group_badnwagon_136', 'group_dense_cluster_145', 'group_camouflage_30',
         'group_badnwagon_187', 'group_camouflage_55', 'group_badnwagon_43', 'group_dense_cluster_181',
         'group_badnwagon_61', 'group_camouflage_52', 'shilling_high_deg_13', 'group_badnwagon_151',
         'group_badnwagon_143'],
        ['group_badnwagon_35', 'group_camouflage_15', 'group_dense_cluster_170', 'group_dense_cluster_24',
         'group_badnwagon_116', 'group_camouflage_46', 'group_dense_cluster_73', 'group_dense_cluster_61',
         'group_camouflage_56', 'group_badnwagon_179', 'group_badnwagon_173', 'group_dense_cluster_120',
         'group_badnwagon_188', 'group_camouflage_39', 'group_dense_cluster_59', 'group_badnwagon_186'],
        ['group_badnwagon_93', 'group_badnwagon_99', 'group_badnwagon_8', 'group_badnwagon_199', 'group_badnwagon_7',
         'group_camouflage_188', 'group_dense_cluster_173', 'group_badnwagon_171', 'group_dense_cluster_139',
         'group_dense_cluster_124', 'group_badnwagon_156', 'group_dense_cluster_26', 'group_badnwagon_51',
         'group_camouflage_57', 'group_dense_cluster_18', 'group_dense_cluster_25', 'group_dense_cluster_102'],
        ['group_dense_cluster_89', 'group_camouflage_152', 'group_camouflage_140', 'group_badnwagon_178',
         'group_dense_cluster_157', 'group_camouflage_18', 'shilling_high_deg_2', 'group_camouflage_67',
         'group_badnwagon_150', 'group_camouflage_85', 'group_camouflage_38', 'group_camouflage_65',
         'group_badnwagon_92', 'group_badnwagon_164', 'group_dense_cluster_123', 'group_dense_cluster_44',
         'group_dense_cluster_153', 'group_badnwagon_174'],
        ['group_dense_cluster_87', 'group_dense_cluster_45', 'group_badnwagon_154', 'group_dense_cluster_12',
         'group_dense_cluster_98', 'group_badnwagon_10', 'group_camouflage_106', 'group_dense_cluster_52',
         'group_camouflage_73', 'group_camouflage_149', 'group_badnwagon_31', 'group_dense_cluster_172',
         'group_camouflage_51', 'group_badnwagon_162', 'group_camouflage_0', 'group_badnwagon_113'],
        ['group_dense_cluster_195', 'shilling_high_deg_45', 'shilling_high_deg_54', 'group_camouflage_27',
         'group_badnwagon_71', 'shilling_high_deg_57', 'shilling_high_deg_35', 'shilling_high_deg_3',
         'shilling_high_deg_4', 'group_dense_cluster_133', 'shilling_high_deg_8', 'group_camouflage_199',
         'group_camouflage_115'],
        ['shilling_high_deg_42', 'shilling_high_deg_48', 'shilling_high_deg_40', 'shilling_high_deg_44'], [], [], [],
        [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],
        [], [], [], [], [], []]
    groups_active_mab = [
        ['shilling_hijacking_171', 'shilling_hijacking_137', 'shilling_high_deg_53', 'group_dense_cluster_63',
         'shilling_hijacking_94', 'group_badnwagon_172', 'shilling_hijacking_128', 'shilling_hijacking_27',
         'shilling_hijacking_46', 'shilling_hijacking_64', 'shilling_hijacking_24', 'shilling_hijacking_119',
         'shilling_hijacking_56', 'shilling_hijacking_142', 'group_dense_cluster_1', 'shilling_hijacking_183',
         'shilling_high_deg_55', 'shilling_hijacking_101', 'group_camouflage_196', 'shilling_hijacking_32',
         'shilling_hijacking_9', 'group_camouflage_116', 'shilling_hijacking_146', 'group_dense_cluster_105',
         'shilling_hijacking_18', 'shilling_hijacking_130', 'shilling_hijacking_76', 'shilling_hijacking_188',
         'group_badnwagon_77', 'shilling_hijacking_55', 'group_dense_cluster_151', 'group_camouflage_151',
         'group_dense_cluster_90', 'group_badnwagon_30', 'group_camouflage_71', 'shilling_hijacking_118',
         'group_camouflage_124', 'shilling_hijacking_126'],
        ['group_camouflage_59', 'shilling_bridge_31', 'shilling_hijacking_2', 'shilling_hijacking_5',
         'shilling_hijacking_112', 'shilling_hijacking_38', 'group_camouflage_131', 'shilling_hijacking_132',
         'shilling_hijacking_73', 'shilling_hijacking_150', 'shilling_bridge_24', 'shilling_bridge_15',
         'shilling_hijacking_135', 'shilling_hijacking_89', 'group_badnwagon_111', 'group_badnwagon_124',
         'shilling_bridge_36', 'shilling_hijacking_197', 'group_camouflage_166', 'shilling_hijacking_23',
         'group_badnwagon_139', 'shilling_hijacking_122', 'shilling_hijacking_77', 'shilling_hijacking_111',
         'group_dense_cluster_127', 'shilling_bridge_33', 'shilling_hijacking_138', 'group_camouflage_155',
         'shilling_hijacking_161', 'group_dense_cluster_92', 'group_badnwagon_74', 'group_dense_cluster_121',
         'shilling_bridge_34', 'shilling_bridge_29', 'shilling_hijacking_164', 'group_dense_cluster_73',
         'shilling_hijacking_131', 'group_camouflage_52', 'shilling_hijacking_121', 'group_badnwagon_127',
         'group_badnwagon_54', 'shilling_hijacking_7', 'shilling_bridge_13', 'group_badnwagon_138',
         'shilling_bridge_25', 'group_badnwagon_11', 'shilling_high_deg_16', 'shilling_bridge_2',
         'shilling_hijacking_62', 'shilling_bridge_26', 'shilling_hijacking_37'],
        ['group_badnwagon_134', 'group_dense_cluster_115', 'shilling_hijacking_68', 'shilling_hijacking_153',
         'group_dense_cluster_35', 'group_badnwagon_96', 'group_badnwagon_3', 'group_badnwagon_176',
         'group_badnwagon_83', 'group_camouflage_193', 'group_camouflage_5', 'shilling_high_deg_10',
         'group_badnwagon_148', 'group_dense_cluster_95', 'group_camouflage_43', 'group_dense_cluster_32',
         'group_camouflage_82', 'group_badnwagon_119', 'group_dense_cluster_128', 'group_badnwagon_39'],
        ['group_badnwagon_61', 'shilling_high_deg_27', 'group_dense_cluster_102', 'group_badnwagon_150',
         'group_dense_cluster_89', 'group_camouflage_55', 'group_dense_cluster_26', 'group_dense_cluster_22',
         'group_badnwagon_175', 'group_camouflage_26', 'group_badnwagon_93', 'group_badnwagon_113',
         'group_camouflage_30', 'group_badnwagon_164', 'group_badnwagon_136', 'group_badnwagon_143',
         'group_badnwagon_151', 'shilling_hijacking_199', 'group_camouflage_129'],
        ['group_dense_cluster_2', 'group_camouflage_68', 'group_badnwagon_97', 'group_dense_cluster_4',
         'group_badnwagon_166', 'group_camouflage_171', 'group_camouflage_174', 'group_camouflage_103',
         'group_camouflage_154', 'group_camouflage_96', 'shilling_hijacking_145', 'group_badnwagon_73',
         'group_badnwagon_35', 'group_badnwagon_6', 'group_dense_cluster_138', 'group_dense_cluster_152',
         'group_camouflage_27'],
        ['group_camouflage_146', 'group_badnwagon_22', 'group_camouflage_56', 'group_badnwagon_174',
         'group_dense_cluster_142', 'group_badnwagon_112', 'group_camouflage_118', 'group_badnwagon_133',
         'group_dense_cluster_145', 'group_dense_cluster_39', 'group_dense_cluster_87', 'group_camouflage_39',
         'group_badnwagon_43', 'group_dense_cluster_170', 'group_dense_cluster_181', 'group_badnwagon_104',
         'group_camouflage_42'],
        ['group_dense_cluster_59', 'group_badnwagon_171', 'group_badnwagon_57', 'group_badnwagon_41',
         'group_dense_cluster_48', 'group_dense_cluster_139', 'group_camouflage_73', 'group_camouflage_170',
         'group_camouflage_41', 'group_camouflage_46', 'shilling_high_deg_24', 'group_dense_cluster_7',
         'group_camouflage_38', 'group_camouflage_65', 'group_camouflage_67', 'group_camouflage_186',
         'shilling_high_deg_18', 'group_badnwagon_187'],
        ['group_camouflage_15', 'shilling_high_deg_13', 'group_badnwagon_59', 'group_dense_cluster_124',
         'group_camouflage_78', 'group_dense_cluster_182', 'group_badnwagon_186', 'group_camouflage_37',
         'group_dense_cluster_44', 'group_dense_cluster_178', 'group_camouflage_70', 'group_badnwagon_8',
         'group_badnwagon_49', 'group_dense_cluster_159', 'group_dense_cluster_186'],
        ['group_badnwagon_92', 'group_dense_cluster_173', 'group_badnwagon_82', 'group_badnwagon_156',
         'group_camouflage_188', 'group_badnwagon_116', 'group_badnwagon_40', 'group_camouflage_152',
         'group_badnwagon_7', 'group_badnwagon_31', 'group_dense_cluster_133', 'group_badnwagon_58',
         'group_dense_cluster_123', 'group_badnwagon_42', 'group_camouflage_18', 'group_badnwagon_28',
         'group_camouflage_57', 'group_badnwagon_173', 'group_camouflage_140'],
        ['group_badnwagon_10', 'group_dense_cluster_18', 'group_dense_cluster_25', 'group_camouflage_0',
         'group_dense_cluster_153', 'group_dense_cluster_45', 'group_camouflage_149', 'group_badnwagon_99',
         'group_badnwagon_68', 'group_badnwagon_51', 'shilling_high_deg_2', 'group_camouflage_106',
         'group_camouflage_85', 'group_camouflage_138', 'group_dense_cluster_195', 'group_badnwagon_71',
         'group_dense_cluster_98'],
        ['group_badnwagon_154', 'group_dense_cluster_24', 'group_dense_cluster_12', 'group_dense_cluster_120',
         'group_badnwagon_178', 'group_badnwagon_188', 'group_badnwagon_105', 'group_dense_cluster_52',
         'group_camouflage_115', 'group_camouflage_51', 'group_badnwagon_179', 'group_dense_cluster_146',
         'group_dense_cluster_61', 'group_badnwagon_199', 'group_dense_cluster_157'],
        ['group_camouflage_199', 'group_dense_cluster_17', 'shilling_high_deg_4', 'shilling_high_deg_48',
         'group_badnwagon_1', 'shilling_high_deg_42', 'shilling_high_deg_45', 'shilling_high_deg_54',
         'group_dense_cluster_172', 'shilling_high_deg_57', 'shilling_high_deg_8', 'group_badnwagon_162'],
        ['shilling_high_deg_40', 'shilling_high_deg_3', 'shilling_high_deg_44', 'shilling_high_deg_35']]
    groups_active_mab = [
        ['shilling_hijacking_126', 'shilling_hijacking_188', 'shilling_hijacking_64', 'shilling_hijacking_183',
         'group_dense_cluster_63', 'shilling_hijacking_119', 'group_camouflage_124', 'group_dense_cluster_105',
         'shilling_hijacking_27', 'shilling_hijacking_9', 'shilling_hijacking_24', 'group_camouflage_196',
         'shilling_hijacking_142', 'shilling_hijacking_32', 'group_badnwagon_172', 'shilling_hijacking_118',
         'shilling_hijacking_18', 'shilling_high_deg_55', 'group_dense_cluster_90', 'group_dense_cluster_151',
         'shilling_hijacking_101', 'shilling_high_deg_53', 'shilling_hijacking_130', 'shilling_hijacking_76',
         'group_dense_cluster_1', 'group_camouflage_71', 'shilling_hijacking_146', 'shilling_hijacking_171',
         'shilling_hijacking_137', 'group_badnwagon_30', 'shilling_hijacking_128', 'shilling_hijacking_94',
         'shilling_hijacking_56', 'shilling_hijacking_46', 'group_camouflage_116', 'group_camouflage_151',
         'group_badnwagon_77', 'shilling_hijacking_55'],
        ['shilling_hijacking_138', 'shilling_hijacking_122', 'shilling_hijacking_121', 'group_badnwagon_54',
         'group_dense_cluster_127', 'shilling_bridge_29', 'shilling_bridge_36', 'shilling_hijacking_77',
         'shilling_hijacking_73', 'group_badnwagon_111', 'shilling_hijacking_68', 'group_camouflage_59',
         'shilling_bridge_15', 'group_camouflage_155', 'shilling_bridge_25', 'shilling_hijacking_23',
         'shilling_hijacking_37', 'shilling_hijacking_150', 'shilling_bridge_31', 'shilling_hijacking_2',
         'shilling_hijacking_161', 'group_badnwagon_3', 'shilling_hijacking_135', 'group_badnwagon_139',
         'group_badnwagon_74', 'shilling_hijacking_132', 'group_camouflage_131', 'shilling_bridge_2',
         'group_dense_cluster_73', 'shilling_high_deg_16', 'shilling_bridge_13', 'shilling_hijacking_62',
         'shilling_bridge_24', 'shilling_hijacking_153', 'group_dense_cluster_121', 'group_dense_cluster_92',
         'group_camouflage_166', 'group_badnwagon_11', 'shilling_hijacking_112', 'group_badnwagon_124',
         'group_badnwagon_138', 'shilling_bridge_33', 'shilling_hijacking_7', 'shilling_hijacking_131',
         'shilling_hijacking_38', 'shilling_hijacking_164', 'shilling_hijacking_111', 'shilling_hijacking_89',
         'group_badnwagon_127', 'shilling_bridge_34', 'shilling_bridge_26', 'shilling_hijacking_5',
         'shilling_hijacking_197'],
        ['group_camouflage_171', 'group_badnwagon_96', 'group_badnwagon_176', 'group_dense_cluster_32',
         'group_dense_cluster_128', 'group_badnwagon_39', 'group_camouflage_5', 'group_dense_cluster_95',
         'group_badnwagon_119', 'group_dense_cluster_115', 'group_badnwagon_148', 'group_camouflage_193',
         'shilling_high_deg_10', 'group_camouflage_43', 'group_dense_cluster_35', 'group_dense_cluster_22',
         'group_badnwagon_134', 'group_badnwagon_83'],
        ['group_badnwagon_164', 'group_badnwagon_136', 'group_dense_cluster_102', 'group_badnwagon_143',
         'group_dense_cluster_2', 'group_camouflage_30', 'shilling_hijacking_145', 'group_badnwagon_175',
         'group_camouflage_52', 'group_badnwagon_151', 'group_dense_cluster_26', 'shilling_hijacking_199',
         'group_camouflage_55', 'group_badnwagon_61', 'group_dense_cluster_89', 'group_badnwagon_93',
         'group_badnwagon_113', 'group_badnwagon_150', 'group_camouflage_26', 'group_camouflage_129'],
        ['group_camouflage_96', 'group_camouflage_118', 'group_badnwagon_166', 'group_camouflage_82',
         'group_dense_cluster_152', 'group_dense_cluster_142', 'group_dense_cluster_4', 'group_badnwagon_104',
         'group_camouflage_68', 'group_dense_cluster_138', 'group_badnwagon_97', 'group_badnwagon_6',
         'group_camouflage_146', 'group_camouflage_174', 'group_badnwagon_35', 'shilling_high_deg_27'],
        ['group_dense_cluster_182', 'group_badnwagon_73', 'group_badnwagon_112', 'shilling_high_deg_18',
         'group_badnwagon_22', 'group_dense_cluster_181', 'group_dense_cluster_145', 'shilling_high_deg_24',
         'group_camouflage_170', 'group_camouflage_39', 'group_camouflage_42', 'group_camouflage_154',
         'group_dense_cluster_170', 'group_badnwagon_174', 'group_camouflage_103', 'group_badnwagon_133'],
        ['group_badnwagon_171', 'group_camouflage_56', 'shilling_high_deg_13', 'group_badnwagon_116',
         'group_dense_cluster_178', 'group_dense_cluster_7', 'group_dense_cluster_87', 'group_badnwagon_92',
         'group_camouflage_27', 'group_camouflage_46', 'group_camouflage_67', 'group_camouflage_140',
         'group_badnwagon_187', 'group_camouflage_73', 'group_camouflage_188', 'group_badnwagon_43',
         'group_camouflage_41', 'group_badnwagon_41'],
        ['group_dense_cluster_44', 'group_badnwagon_173', 'group_camouflage_186', 'group_camouflage_65',
         'group_camouflage_37', 'group_badnwagon_82', 'group_dense_cluster_186', 'group_badnwagon_49',
         'group_badnwagon_186', 'group_dense_cluster_39', 'group_dense_cluster_48', 'group_dense_cluster_59',
         'group_dense_cluster_159', 'group_camouflage_70', 'group_dense_cluster_139', 'group_dense_cluster_124',
         'group_badnwagon_57'],
        ['group_camouflage_18', 'group_dense_cluster_24', 'group_badnwagon_156', 'group_dense_cluster_173',
         'group_badnwagon_51', 'group_badnwagon_58', 'group_badnwagon_28', 'group_badnwagon_8', 'group_badnwagon_42',
         'group_camouflage_15', 'group_camouflage_78', 'group_badnwagon_59', 'group_badnwagon_31',
         'group_dense_cluster_18', 'group_badnwagon_7', 'group_badnwagon_40', 'group_camouflage_38'],
        ['group_badnwagon_68', 'group_camouflage_57', 'group_dense_cluster_61', 'group_camouflage_152',
         'group_dense_cluster_17', 'group_badnwagon_188', 'group_badnwagon_105', 'group_camouflage_149',
         'group_badnwagon_99', 'group_badnwagon_10', 'group_badnwagon_199', 'group_camouflage_0',
         'group_dense_cluster_123', 'group_dense_cluster_25', 'shilling_high_deg_2', 'group_dense_cluster_133',
         'group_dense_cluster_153'],
        ['group_dense_cluster_120', 'group_badnwagon_179', 'group_camouflage_106', 'group_dense_cluster_52',
         'group_dense_cluster_195', 'group_dense_cluster_157', 'group_camouflage_51', 'group_dense_cluster_45',
         'group_dense_cluster_98', 'group_dense_cluster_146', 'group_camouflage_138', 'group_camouflage_85',
         'group_dense_cluster_12', 'group_badnwagon_154', 'group_badnwagon_71'],
        ['shilling_high_deg_42', 'group_dense_cluster_172', 'group_camouflage_115', 'shilling_high_deg_54',
         'group_camouflage_199', 'shilling_high_deg_45', 'shilling_high_deg_4', 'group_badnwagon_1',
         'shilling_high_deg_57', 'group_badnwagon_178', 'group_badnwagon_162', 'shilling_high_deg_48',
         'shilling_high_deg_8'],
        ['shilling_high_deg_44', 'shilling_high_deg_35', 'shilling_high_deg_40', 'shilling_high_deg_3']]

    groups_active_mab_0 = [
        ['shilling_hijacking_146', 'shilling_hijacking_128', 'shilling_hijacking_130', 'shilling_hijacking_46',
         'group_camouflage_124', 'shilling_hijacking_24', 'group_camouflage_71', 'shilling_hijacking_119',
         'group_badnwagon_77', 'shilling_hijacking_55', 'shilling_hijacking_18', 'shilling_high_deg_53',
         'group_dense_cluster_90', 'shilling_hijacking_126', 'shilling_hijacking_32', 'shilling_hijacking_142',
         'group_camouflage_116', 'group_camouflage_151', 'group_dense_cluster_105', 'shilling_hijacking_188',
         'shilling_hijacking_27', 'shilling_hijacking_183', 'shilling_hijacking_101', 'group_dense_cluster_1',
         'shilling_hijacking_56', 'shilling_hijacking_76', 'shilling_hijacking_118', 'shilling_hijacking_64',
         'shilling_high_deg_55', 'group_dense_cluster_151', 'shilling_hijacking_137', 'shilling_hijacking_9',
         'group_badnwagon_172', 'shilling_hijacking_94', 'group_badnwagon_30', 'shilling_hijacking_171',
         'group_camouflage_196', 'group_dense_cluster_63'],
        ['shilling_hijacking_111', 'shilling_bridge_34', 'shilling_hijacking_7', 'shilling_hijacking_37',
         'shilling_high_deg_16', 'shilling_hijacking_2', 'shilling_hijacking_132', 'shilling_hijacking_89',
         'shilling_bridge_36', 'shilling_hijacking_5', 'shilling_hijacking_138', 'group_badnwagon_139',
         'group_dense_cluster_121', 'group_badnwagon_54', 'shilling_bridge_2', 'shilling_hijacking_38',
         'shilling_hijacking_112', 'shilling_hijacking_62', 'shilling_hijacking_121', 'shilling_hijacking_161',
         'group_badnwagon_111', 'shilling_hijacking_73', 'group_camouflage_59', 'shilling_bridge_25',
         'shilling_hijacking_150', 'shilling_hijacking_122', 'shilling_hijacking_131', 'group_badnwagon_138',
         'group_badnwagon_11', 'group_badnwagon_127', 'shilling_hijacking_77', 'shilling_hijacking_164',
         'shilling_bridge_29', 'shilling_bridge_26', 'group_dense_cluster_127', 'group_camouflage_131',
         'group_badnwagon_124', 'shilling_bridge_31', 'group_dense_cluster_92', 'shilling_hijacking_23',
         'group_camouflage_155', 'group_camouflage_52', 'shilling_bridge_33', 'shilling_bridge_13',
         'shilling_hijacking_197', 'group_dense_cluster_73', 'group_badnwagon_74', 'shilling_bridge_24',
         'shilling_bridge_15', 'shilling_hijacking_135', 'group_camouflage_166'],
        ['group_camouflage_82', 'group_dense_cluster_115', 'group_dense_cluster_35', 'group_badnwagon_39',
         'shilling_hijacking_68', 'group_camouflage_5', 'group_dense_cluster_32', 'group_badnwagon_176',
         'group_badnwagon_96', 'group_badnwagon_134', 'group_dense_cluster_95', 'group_badnwagon_3',
         'group_camouflage_193', 'group_badnwagon_83', 'group_badnwagon_119', 'shilling_high_deg_10',
         'group_dense_cluster_128', 'shilling_hijacking_153', 'group_camouflage_43', 'group_badnwagon_148'],
        ['group_camouflage_26', 'group_badnwagon_93', 'shilling_high_deg_27', 'group_badnwagon_61',
         'group_camouflage_129', 'shilling_hijacking_199', 'group_badnwagon_143', 'group_dense_cluster_26',
         'group_dense_cluster_102', 'group_dense_cluster_22', 'group_camouflage_55', 'group_badnwagon_175',
         'group_badnwagon_164', 'group_badnwagon_136', 'group_badnwagon_151', 'group_badnwagon_150',
         'group_badnwagon_113', 'group_dense_cluster_89', 'group_camouflage_30'],
        ['group_camouflage_174', 'group_badnwagon_166', 'group_badnwagon_73', 'group_camouflage_96',
         'group_dense_cluster_4', 'group_dense_cluster_2', 'group_dense_cluster_138', 'group_dense_cluster_152',
         'group_badnwagon_35', 'group_camouflage_27', 'group_camouflage_154', 'shilling_hijacking_145',
         'group_camouflage_171', 'group_camouflage_103', 'group_badnwagon_6', 'group_badnwagon_97',
         'group_camouflage_68'],
        ['group_badnwagon_104', 'group_badnwagon_174', 'group_badnwagon_43', 'group_dense_cluster_87',
         'group_dense_cluster_170', 'group_badnwagon_22', 'group_dense_cluster_145', 'group_camouflage_146',
         'group_camouflage_56', 'group_badnwagon_133', 'group_camouflage_42', 'group_camouflage_39',
         'group_dense_cluster_39', 'group_camouflage_118', 'group_badnwagon_112', 'group_dense_cluster_181',
         'group_dense_cluster_142'],
        ['group_camouflage_46', 'group_badnwagon_171', 'group_dense_cluster_48', 'shilling_high_deg_18',
         'group_camouflage_170', 'shilling_high_deg_24', 'group_dense_cluster_139', 'group_camouflage_38',
         'group_dense_cluster_7', 'group_badnwagon_57', 'group_camouflage_65', 'group_camouflage_41',
         'group_camouflage_186', 'group_badnwagon_187', 'group_dense_cluster_59', 'group_badnwagon_41',
         'group_camouflage_73', 'group_camouflage_67'],
        ['group_dense_cluster_178', 'group_badnwagon_59', 'group_camouflage_78', 'group_camouflage_70',
         'shilling_high_deg_13', 'group_badnwagon_186', 'group_badnwagon_49', 'group_dense_cluster_124',
         'group_dense_cluster_182', 'group_dense_cluster_44', 'group_camouflage_37', 'group_dense_cluster_186',
         'group_dense_cluster_159', 'group_camouflage_15', 'group_badnwagon_8'],
        ['group_badnwagon_7', 'group_camouflage_140', 'group_badnwagon_116', 'group_badnwagon_173',
         'group_camouflage_188', 'group_badnwagon_92', 'group_badnwagon_82', 'group_dense_cluster_133',
         'group_badnwagon_156', 'group_dense_cluster_173', 'group_badnwagon_31', 'group_camouflage_152',
         'group_badnwagon_40', 'group_badnwagon_28', 'group_badnwagon_58', 'group_badnwagon_42', 'group_camouflage_18',
         'group_dense_cluster_123', 'group_camouflage_57'],
        ['group_dense_cluster_45', 'group_dense_cluster_195', 'group_dense_cluster_25', 'group_camouflage_106',
         'group_camouflage_138', 'group_camouflage_85', 'group_camouflage_149', 'group_dense_cluster_98',
         'shilling_high_deg_2', 'group_camouflage_0', 'group_dense_cluster_153', 'group_badnwagon_10',
         'group_dense_cluster_18', 'group_badnwagon_71', 'group_badnwagon_68', 'group_badnwagon_51',
         'group_badnwagon_99'],
        ['group_badnwagon_199', 'group_camouflage_51', 'group_dense_cluster_61', 'group_badnwagon_105',
         'group_badnwagon_178', 'group_camouflage_115', 'group_badnwagon_179', 'group_dense_cluster_24',
         'group_dense_cluster_146', 'group_dense_cluster_120', 'group_badnwagon_188', 'group_dense_cluster_52',
         'group_dense_cluster_157', 'group_dense_cluster_12', 'group_badnwagon_154'],
        ['group_badnwagon_1', 'group_camouflage_199', 'shilling_high_deg_54', 'group_dense_cluster_17',
         'group_dense_cluster_172', 'shilling_high_deg_57', 'shilling_high_deg_48', 'shilling_high_deg_42',
         'shilling_high_deg_8', 'group_badnwagon_162', 'shilling_high_deg_45', 'shilling_high_deg_4'],
        ['shilling_high_deg_35', 'shilling_high_deg_3', 'shilling_high_deg_44', 'shilling_high_deg_40']]
    groups_active_mab_2 = [
        ['shilling_hijacking_142', 'group_dense_cluster_1', 'group_camouflage_116', 'shilling_hijacking_118',
         'shilling_hijacking_56', 'group_badnwagon_30', 'shilling_hijacking_64', 'group_camouflage_151',
         'shilling_hijacking_171', 'shilling_high_deg_53', 'group_dense_cluster_105', 'shilling_hijacking_24',
         'shilling_hijacking_146', 'shilling_hijacking_46', 'shilling_hijacking_76', 'shilling_hijacking_101',
         'group_badnwagon_77', 'group_camouflage_196', 'shilling_hijacking_128', 'shilling_hijacking_94',
         'group_dense_cluster_63', 'shilling_hijacking_188', 'shilling_hijacking_55', 'shilling_hijacking_126',
         'shilling_hijacking_18', 'group_camouflage_71', 'group_dense_cluster_90', 'shilling_hijacking_32',
         'shilling_hijacking_137', 'group_badnwagon_172', 'shilling_high_deg_55', 'shilling_hijacking_119',
         'group_dense_cluster_151', 'shilling_hijacking_27', 'group_camouflage_124', 'shilling_hijacking_183',
         'shilling_hijacking_9', 'shilling_hijacking_130'],
        ['shilling_hijacking_37', 'shilling_bridge_34', 'group_badnwagon_3', 'shilling_hijacking_7',
         'shilling_hijacking_62', 'shilling_hijacking_161', 'group_badnwagon_139', 'group_camouflage_155',
         'group_badnwagon_74', 'shilling_hijacking_23', 'shilling_hijacking_135', 'shilling_bridge_15',
         'group_badnwagon_127', 'group_badnwagon_54', 'group_badnwagon_11', 'shilling_hijacking_38',
         'group_dense_cluster_121', 'shilling_hijacking_68', 'group_badnwagon_138', 'shilling_hijacking_138',
         'shilling_hijacking_111', 'shilling_high_deg_16', 'shilling_bridge_33', 'group_camouflage_59',
         'shilling_hijacking_5', 'shilling_hijacking_112', 'shilling_hijacking_2', 'shilling_hijacking_197',
         'shilling_bridge_26', 'shilling_bridge_25', 'group_badnwagon_124', 'shilling_hijacking_121',
         'group_camouflage_131', 'group_badnwagon_111', 'shilling_hijacking_199', 'shilling_bridge_24',
         'group_dense_cluster_73', 'shilling_bridge_31', 'shilling_hijacking_164', 'shilling_hijacking_73',
         'shilling_hijacking_132', 'shilling_bridge_29', 'shilling_bridge_13', 'shilling_hijacking_77',
         'shilling_hijacking_153', 'group_dense_cluster_127', 'group_camouflage_166', 'shilling_hijacking_89',
         'shilling_bridge_2', 'shilling_hijacking_145', 'group_dense_cluster_92', 'shilling_hijacking_122',
         'shilling_bridge_36'],
        ['group_dense_cluster_32', 'group_camouflage_171', 'group_dense_cluster_128', 'group_badnwagon_39',
         'group_dense_cluster_152', 'group_dense_cluster_115', 'group_badnwagon_119', 'shilling_hijacking_150',
         'group_badnwagon_96', 'shilling_hijacking_131', 'shilling_high_deg_10', 'group_camouflage_193',
         'group_camouflage_118', 'group_badnwagon_148', 'group_badnwagon_134', 'group_camouflage_43',
         'group_badnwagon_83', 'group_dense_cluster_22', 'group_camouflage_5', 'group_badnwagon_176'],
        ['group_dense_cluster_26', 'group_badnwagon_93', 'group_camouflage_129', 'group_camouflage_174',
         'group_dense_cluster_95', 'group_dense_cluster_35', 'group_badnwagon_143', 'group_badnwagon_97',
         'group_camouflage_55', 'group_camouflage_52', 'group_camouflage_82', 'group_camouflage_30',
         'group_camouflage_26', 'group_badnwagon_175', 'group_dense_cluster_102', 'group_badnwagon_61',
         'group_badnwagon_6', 'group_badnwagon_166'],
        ['group_badnwagon_22', 'group_dense_cluster_4', 'group_dense_cluster_89', 'group_camouflage_42',
         'group_badnwagon_73', 'group_dense_cluster_142', 'group_dense_cluster_2', 'group_dense_cluster_39',
         'group_badnwagon_136', 'group_camouflage_103', 'group_dense_cluster_138', 'group_badnwagon_164',
         'group_badnwagon_113', 'group_camouflage_68', 'group_badnwagon_151', 'group_camouflage_96',
         'group_badnwagon_104'],
        ['group_camouflage_39', 'group_badnwagon_43', 'shilling_high_deg_27', 'group_dense_cluster_145',
         'group_badnwagon_133', 'group_dense_cluster_48', 'group_camouflage_154', 'group_camouflage_170',
         'group_camouflage_56', 'group_dense_cluster_170', 'group_badnwagon_112', 'group_badnwagon_35',
         'group_badnwagon_173', 'group_dense_cluster_181', 'group_badnwagon_150', 'group_badnwagon_174'],
        ['group_camouflage_146', 'group_dense_cluster_7', 'group_badnwagon_41', 'group_dense_cluster_159',
         'group_camouflage_70', 'group_camouflage_41', 'group_dense_cluster_178', 'group_camouflage_78',
         'shilling_high_deg_24', 'group_camouflage_46', 'group_dense_cluster_182', 'group_camouflage_186',
         'group_camouflage_37', 'shilling_high_deg_18', 'group_dense_cluster_186', 'group_dense_cluster_59',
         'group_badnwagon_57'],
        ['group_camouflage_188', 'group_dense_cluster_146', 'group_badnwagon_116', 'group_badnwagon_186',
         'group_badnwagon_28', 'group_badnwagon_49', 'group_badnwagon_42', 'group_badnwagon_187', 'group_camouflage_15',
         'shilling_high_deg_13', 'group_badnwagon_105', 'group_badnwagon_40', 'group_badnwagon_59',
         'group_camouflage_140', 'group_badnwagon_58', 'group_camouflage_67', 'group_badnwagon_82'],
        ['group_dense_cluster_173', 'group_badnwagon_156', 'group_camouflage_65', 'group_dense_cluster_124',
         'group_dense_cluster_17', 'group_badnwagon_8', 'group_dense_cluster_25', 'group_dense_cluster_18',
         'group_dense_cluster_44', 'group_dense_cluster_24', 'group_badnwagon_171', 'group_camouflage_57',
         'group_dense_cluster_87', 'group_dense_cluster_139', 'group_camouflage_27', 'group_badnwagon_92'],
        ['group_badnwagon_10', 'group_dense_cluster_61', 'group_badnwagon_68', 'shilling_high_deg_2',
         'group_badnwagon_7', 'group_badnwagon_99', 'group_dense_cluster_133', 'group_badnwagon_199',
         'group_dense_cluster_157', 'group_camouflage_0', 'group_camouflage_38', 'group_camouflage_18',
         'group_badnwagon_31', 'group_camouflage_152', 'group_dense_cluster_123', 'group_dense_cluster_153',
         'group_camouflage_73', 'group_badnwagon_51'],
        ['group_dense_cluster_98', 'group_dense_cluster_195', 'group_badnwagon_154', 'group_camouflage_149',
         'group_dense_cluster_52', 'group_dense_cluster_45', 'group_camouflage_51', 'group_dense_cluster_12',
         'group_badnwagon_71', 'group_camouflage_85', 'group_camouflage_106', 'group_badnwagon_188',
         'group_dense_cluster_120', 'group_badnwagon_162', 'group_camouflage_138', 'group_badnwagon_179'],
        ['shilling_high_deg_4', 'shilling_high_deg_45', 'shilling_high_deg_42', 'group_camouflage_199',
         'group_dense_cluster_172', 'group_badnwagon_178', 'group_badnwagon_1', 'shilling_high_deg_8',
         'shilling_high_deg_57', 'shilling_high_deg_54', 'shilling_high_deg_48', 'group_camouflage_115'],
        ['shilling_high_deg_44', 'shilling_high_deg_40', 'shilling_high_deg_3', 'shilling_high_deg_35']]
    groups_active_mab_4 = [
        ['shilling_hijacking_9', 'shilling_hijacking_146', 'group_dense_cluster_105', 'shilling_hijacking_94',
         'shilling_hijacking_24', 'shilling_hijacking_64', 'shilling_hijacking_142', 'shilling_hijacking_56',
         'group_camouflage_116', 'shilling_hijacking_119', 'shilling_high_deg_53', 'shilling_hijacking_130',
         'group_badnwagon_77', 'shilling_hijacking_128', 'group_camouflage_151', 'group_dense_cluster_1',
         'group_dense_cluster_90', 'group_camouflage_196', 'shilling_hijacking_126', 'shilling_hijacking_118',
         'shilling_hijacking_171', 'group_badnwagon_30', 'shilling_hijacking_32', 'group_camouflage_71',
         'group_badnwagon_172', 'shilling_hijacking_46', 'shilling_hijacking_18', 'group_dense_cluster_151',
         'shilling_hijacking_101', 'shilling_hijacking_76', 'shilling_hijacking_183', 'shilling_hijacking_137',
         'shilling_hijacking_188', 'shilling_high_deg_55', 'group_camouflage_124', 'shilling_hijacking_27',
         'shilling_hijacking_55', 'group_dense_cluster_63'],
        ['group_camouflage_131', 'group_badnwagon_127', 'shilling_bridge_13', 'shilling_bridge_2', 'shilling_bridge_36',
         'group_badnwagon_139', 'group_badnwagon_138', 'shilling_bridge_26', 'shilling_bridge_34',
         'group_badnwagon_124', 'group_badnwagon_3', 'group_badnwagon_119', 'group_camouflage_155',
         'shilling_bridge_24', 'shilling_high_deg_16', 'group_dense_cluster_138', 'group_badnwagon_111',
         'group_camouflage_59', 'shilling_bridge_25', 'group_dense_cluster_92', 'group_dense_cluster_121',
         'group_badnwagon_54', 'shilling_bridge_29', 'group_badnwagon_39', 'group_badnwagon_74', 'group_camouflage_166',
         'shilling_bridge_15', 'group_dense_cluster_127', 'shilling_bridge_33'],
        ['group_badnwagon_11', 'shilling_high_deg_10', 'group_camouflage_171', 'group_dense_cluster_115',
         'group_badnwagon_83', 'group_dense_cluster_22', 'shilling_hijacking_197', 'shilling_hijacking_7',
         'group_badnwagon_96', 'shilling_hijacking_62', 'group_camouflage_52', 'group_badnwagon_148',
         'group_camouflage_174', 'group_dense_cluster_35', 'group_dense_cluster_73', 'shilling_hijacking_112',
         'group_camouflage_5', 'group_badnwagon_73', 'shilling_hijacking_132', 'group_camouflage_193',
         'group_camouflage_43', 'group_dense_cluster_32', 'shilling_bridge_31'],
        ['group_dense_cluster_4', 'group_badnwagon_97', 'group_dense_cluster_26', 'group_camouflage_103',
         'shilling_hijacking_38', 'group_dense_cluster_152', 'group_camouflage_154', 'group_badnwagon_99',
         'group_dense_cluster_170', 'group_dense_cluster_7', 'group_badnwagon_22', 'group_camouflage_129',
         'group_camouflage_30', 'group_badnwagon_175', 'group_camouflage_68', 'group_badnwagon_35',
         'shilling_hijacking_153', 'shilling_hijacking_2', 'group_dense_cluster_95'],
        ['group_badnwagon_136', 'group_badnwagon_104', 'group_badnwagon_166', 'shilling_hijacking_89',
         'shilling_hijacking_37', 'shilling_hijacking_23', 'shilling_hijacking_73', 'shilling_hijacking_131',
         'group_camouflage_82', 'shilling_hijacking_164', 'group_camouflage_26', 'group_dense_cluster_128',
         'group_dense_cluster_39', 'group_camouflage_118', 'group_dense_cluster_2', 'shilling_hijacking_122',
         'group_dense_cluster_48', 'group_badnwagon_134', 'shilling_hijacking_199', 'shilling_hijacking_111',
         'shilling_hijacking_138', 'shilling_hijacking_145', 'group_camouflage_96', 'group_camouflage_42',
         'shilling_hijacking_121', 'group_badnwagon_112', 'shilling_hijacking_5', 'shilling_hijacking_135',
         'shilling_hijacking_77', 'group_camouflage_55', 'shilling_high_deg_27', 'group_badnwagon_6',
         'shilling_hijacking_68', 'shilling_hijacking_161'],
        ['group_dense_cluster_186', 'group_camouflage_186', 'group_badnwagon_49', 'group_dense_cluster_145',
         'group_camouflage_170', 'group_badnwagon_133', 'shilling_hijacking_150', 'group_camouflage_78',
         'group_badnwagon_28', 'group_camouflage_70', 'group_camouflage_41', 'group_badnwagon_41',
         'group_camouflage_37', 'group_dense_cluster_159', 'group_camouflage_39', 'group_camouflage_56',
         'shilling_high_deg_24', 'group_badnwagon_57'],
        ['group_badnwagon_58', 'group_badnwagon_43', 'group_dense_cluster_102', 'group_dense_cluster_178',
         'group_camouflage_146', 'group_badnwagon_40', 'group_badnwagon_143', 'group_dense_cluster_182',
         'group_dense_cluster_142', 'group_dense_cluster_59', 'group_dense_cluster_181', 'shilling_high_deg_18',
         'group_badnwagon_61', 'group_badnwagon_151', 'group_badnwagon_187', 'group_badnwagon_176',
         'group_badnwagon_174'],
        ['group_dense_cluster_89', 'group_camouflage_46', 'group_badnwagon_116', 'group_badnwagon_59',
         'shilling_high_deg_13', 'group_badnwagon_42', 'group_badnwagon_82', 'group_camouflage_15',
         'group_dense_cluster_146', 'group_badnwagon_164', 'group_badnwagon_93', 'group_badnwagon_105',
         'group_badnwagon_173', 'group_camouflage_188', 'group_badnwagon_186', 'group_badnwagon_150',
         'group_dense_cluster_124'],
        ['group_dense_cluster_25', 'group_dense_cluster_44', 'group_camouflage_67', 'group_dense_cluster_173',
         'group_dense_cluster_24', 'group_badnwagon_171', 'group_dense_cluster_139', 'group_badnwagon_188',
         'group_badnwagon_7', 'group_camouflage_57', 'group_dense_cluster_18', 'group_badnwagon_179',
         'group_camouflage_140', 'group_dense_cluster_17', 'group_camouflage_138', 'group_badnwagon_92',
         'group_badnwagon_113'],
        ['group_camouflage_27', 'group_dense_cluster_87', 'group_dense_cluster_61', 'group_badnwagon_51',
         'group_dense_cluster_120', 'group_badnwagon_1', 'group_badnwagon_8', 'group_badnwagon_199',
         'shilling_high_deg_2', 'group_camouflage_38', 'group_dense_cluster_153', 'group_camouflage_65',
         'group_camouflage_18', 'group_badnwagon_31', 'group_camouflage_152', 'group_badnwagon_156'],
        ['group_camouflage_106', 'group_dense_cluster_12', 'group_camouflage_0', 'group_badnwagon_154',
         'group_dense_cluster_157', 'group_dense_cluster_123', 'group_camouflage_85', 'group_dense_cluster_45',
         'group_camouflage_73', 'group_dense_cluster_52', 'group_camouflage_51', 'group_badnwagon_178',
         'group_badnwagon_68', 'group_dense_cluster_133', 'group_camouflage_149', 'group_badnwagon_10',
         'group_dense_cluster_98'],
        ['group_badnwagon_162', 'shilling_high_deg_4', 'shilling_high_deg_45', 'shilling_high_deg_48',
         'group_badnwagon_71', 'group_camouflage_115', 'shilling_high_deg_54', 'group_dense_cluster_172',
         'group_dense_cluster_195', 'shilling_high_deg_8', 'shilling_high_deg_57', 'group_camouflage_199'],
        ['shilling_high_deg_44', 'shilling_high_deg_40', 'shilling_high_deg_3', 'shilling_high_deg_42',
         'shilling_high_deg_35']]
    groups_active_mab_6 = [
        ['shilling_high_deg_55', 'shilling_hijacking_118', 'shilling_hijacking_119', 'shilling_hijacking_183',
         'shilling_hijacking_76', 'group_dense_cluster_90', 'group_dense_cluster_63', 'shilling_hijacking_46',
         'shilling_high_deg_53', 'shilling_hijacking_18', 'shilling_hijacking_146', 'group_badnwagon_30',
         'shilling_hijacking_128', 'shilling_hijacking_55', 'group_badnwagon_77', 'group_camouflage_116',
         'shilling_hijacking_27', 'shilling_hijacking_137', 'shilling_hijacking_56', 'group_camouflage_196',
         'group_badnwagon_172', 'group_camouflage_151', 'shilling_hijacking_171', 'group_dense_cluster_1',
         'shilling_hijacking_188', 'shilling_hijacking_101', 'shilling_hijacking_130', 'group_camouflage_124',
         'shilling_hijacking_142', 'shilling_hijacking_9', 'shilling_hijacking_94', 'shilling_hijacking_24',
         'group_camouflage_71', 'shilling_hijacking_126', 'shilling_hijacking_32', 'shilling_hijacking_64',
         'group_dense_cluster_151', 'group_dense_cluster_105'],
        ['shilling_bridge_25', 'shilling_bridge_29', 'group_camouflage_155', 'group_dense_cluster_127',
         'shilling_bridge_26', 'group_badnwagon_138', 'group_badnwagon_119', 'group_badnwagon_124', 'group_badnwagon_3',
         'group_badnwagon_127', 'group_dense_cluster_22', 'group_badnwagon_54', 'shilling_bridge_13',
         'shilling_bridge_15', 'group_camouflage_103', 'shilling_high_deg_16', 'group_dense_cluster_92',
         'group_camouflage_131', 'shilling_bridge_24', 'group_dense_cluster_121', 'shilling_bridge_31',
         'group_badnwagon_39', 'shilling_bridge_2', 'shilling_bridge_34', 'group_camouflage_59', 'group_badnwagon_139',
         'group_badnwagon_111', 'group_badnwagon_74', 'group_camouflage_166', 'shilling_bridge_33',
         'shilling_bridge_36'],
        ['group_camouflage_118', 'shilling_hijacking_199', 'shilling_hijacking_122', 'group_badnwagon_148',
         'shilling_hijacking_89', 'shilling_hijacking_132', 'group_badnwagon_96', 'group_badnwagon_6',
         'group_badnwagon_11', 'group_camouflage_193', 'group_dense_cluster_128', 'group_badnwagon_104',
         'shilling_hijacking_2', 'shilling_hijacking_38', 'group_dense_cluster_152', 'group_badnwagon_176',
         'group_camouflage_5', 'group_dense_cluster_32', 'shilling_hijacking_112', 'shilling_high_deg_10',
         'group_dense_cluster_115', 'group_badnwagon_83', 'shilling_hijacking_197', 'group_camouflage_43',
         'shilling_hijacking_37', 'group_camouflage_171', 'shilling_hijacking_68', 'group_camouflage_82',
         'shilling_hijacking_62', 'shilling_hijacking_7', 'shilling_hijacking_138'],
        ['group_camouflage_30', 'group_dense_cluster_7', 'group_camouflage_42', 'group_camouflage_96',
         'group_dense_cluster_142', 'group_camouflage_129', 'group_badnwagon_134', 'group_camouflage_26',
         'group_badnwagon_166', 'group_dense_cluster_39', 'group_badnwagon_59', 'group_camouflage_70',
         'group_dense_cluster_146', 'group_dense_cluster_95', 'shilling_hijacking_23', 'group_camouflage_55',
         'group_dense_cluster_186', 'group_badnwagon_73', 'shilling_hijacking_5', 'shilling_hijacking_77'],
        ['group_camouflage_37', 'group_dense_cluster_48', 'group_badnwagon_58', 'shilling_hijacking_135',
         'group_badnwagon_105', 'shilling_hijacking_150', 'group_dense_cluster_2', 'shilling_hijacking_73',
         'shilling_hijacking_131', 'shilling_hijacking_121', 'group_badnwagon_57', 'group_camouflage_78',
         'shilling_hijacking_164', 'group_camouflage_174', 'shilling_hijacking_161', 'group_dense_cluster_35',
         'shilling_hijacking_153', 'group_camouflage_138', 'group_badnwagon_22', 'group_badnwagon_41',
         'shilling_hijacking_111', 'group_dense_cluster_159', 'shilling_hijacking_145', 'group_badnwagon_68',
         'group_badnwagon_116', 'group_badnwagon_97', 'group_camouflage_52'],
        ['group_camouflage_68', 'group_dense_cluster_17', 'group_camouflage_146', 'group_dense_cluster_182',
         'group_badnwagon_40', 'group_badnwagon_49', 'group_badnwagon_175', 'group_badnwagon_112',
         'group_camouflage_15', 'group_dense_cluster_138', 'shilling_high_deg_24', 'group_badnwagon_28',
         'shilling_high_deg_27', 'group_dense_cluster_4'],
        ['group_dense_cluster_73', 'shilling_high_deg_18', 'group_dense_cluster_178', 'group_camouflage_170',
         'group_camouflage_186', 'group_badnwagon_61', 'group_badnwagon_136', 'group_dense_cluster_145',
         'group_dense_cluster_181', 'group_badnwagon_35', 'group_camouflage_154', 'group_camouflage_39',
         'group_camouflage_56', 'group_dense_cluster_26', 'group_badnwagon_133', 'group_camouflage_41'],
        ['group_badnwagon_173', 'shilling_high_deg_13', 'group_badnwagon_150', 'group_badnwagon_82',
         'group_badnwagon_186', 'group_dense_cluster_170', 'group_badnwagon_99', 'group_badnwagon_93',
         'group_badnwagon_187', 'group_badnwagon_42', 'group_badnwagon_143', 'group_dense_cluster_102',
         'group_dense_cluster_59', 'group_camouflage_46', 'group_badnwagon_43', 'group_badnwagon_151',
         'group_camouflage_188', 'group_camouflage_57', 'group_dense_cluster_24'],
        ['group_badnwagon_51', 'group_badnwagon_156', 'group_badnwagon_8', 'group_dense_cluster_173',
         'group_dense_cluster_61', 'group_badnwagon_7', 'group_dense_cluster_120', 'group_badnwagon_174',
         'group_badnwagon_188', 'group_badnwagon_1', 'group_badnwagon_179', 'group_dense_cluster_44',
         'group_dense_cluster_18', 'group_dense_cluster_124', 'group_dense_cluster_25'],
        ['group_camouflage_38', 'group_camouflage_67', 'group_badnwagon_199', 'shilling_high_deg_2',
         'group_camouflage_18', 'group_badnwagon_178', 'group_badnwagon_113', 'group_dense_cluster_89',
         'group_camouflage_65', 'group_badnwagon_171', 'group_dense_cluster_157', 'group_dense_cluster_139',
         'group_badnwagon_92', 'group_dense_cluster_153', 'group_dense_cluster_123', 'group_camouflage_152',
         'group_badnwagon_164', 'group_dense_cluster_87'],
        ['group_dense_cluster_12', 'group_camouflage_27', 'group_dense_cluster_133', 'group_camouflage_140',
         'group_dense_cluster_98', 'group_camouflage_0', 'group_badnwagon_154', 'group_badnwagon_71',
         'group_dense_cluster_45', 'group_badnwagon_31', 'group_camouflage_149', 'group_camouflage_73',
         'group_camouflage_106', 'group_dense_cluster_52', 'group_dense_cluster_172', 'group_camouflage_85',
         'group_badnwagon_10'],
        ['shilling_high_deg_54', 'shilling_high_deg_48', 'group_camouflage_115', 'group_camouflage_199',
         'group_dense_cluster_195', 'shilling_high_deg_3', 'shilling_high_deg_8', 'group_camouflage_51',
         'group_badnwagon_162', 'shilling_high_deg_4', 'shilling_high_deg_57', 'shilling_high_deg_45'],
        ['shilling_high_deg_40', 'shilling_high_deg_44', 'shilling_high_deg_42', 'shilling_high_deg_35']]
    groups_active_mab_8 = [
        ['group_dense_cluster_105', 'shilling_hijacking_128', 'shilling_high_deg_55', 'shilling_hijacking_32',
         'shilling_hijacking_56', 'shilling_hijacking_183', 'shilling_hijacking_101', 'shilling_hijacking_64',
         'group_badnwagon_30', 'group_badnwagon_77', 'group_camouflage_124', 'shilling_hijacking_55',
         'shilling_hijacking_94', 'shilling_hijacking_118', 'shilling_hijacking_188', 'group_camouflage_116',
         'shilling_high_deg_53', 'shilling_hijacking_9', 'group_dense_cluster_63', 'shilling_hijacking_46',
         'shilling_hijacking_146', 'shilling_hijacking_171', 'group_camouflage_196', 'shilling_hijacking_126',
         'group_dense_cluster_151', 'group_camouflage_151', 'group_badnwagon_172', 'shilling_hijacking_142',
         'shilling_hijacking_130', 'shilling_hijacking_76', 'shilling_hijacking_119', 'shilling_hijacking_24',
         'group_camouflage_71', 'group_dense_cluster_1', 'shilling_hijacking_18', 'shilling_hijacking_137',
         'shilling_hijacking_27', 'group_dense_cluster_90'],
        ['group_camouflage_103', 'shilling_hijacking_5', 'group_camouflage_59', 'shilling_high_deg_16',
         'shilling_bridge_24', 'shilling_hijacking_197', 'group_badnwagon_54', 'shilling_bridge_2',
         'group_badnwagon_124', 'shilling_hijacking_62', 'shilling_hijacking_68', 'shilling_hijacking_7',
         'group_badnwagon_39', 'shilling_bridge_15', 'shilling_bridge_33', 'group_camouflage_155', 'shilling_bridge_26',
         'shilling_bridge_36', 'group_dense_cluster_22', 'shilling_bridge_13', 'shilling_bridge_34',
         'group_badnwagon_119', 'shilling_hijacking_23', 'group_badnwagon_138', 'group_badnwagon_139',
         'shilling_hijacking_77', 'shilling_hijacking_38', 'group_dense_cluster_121', 'group_dense_cluster_92',
         'group_badnwagon_111', 'shilling_hijacking_2', 'shilling_hijacking_132', 'group_camouflage_131',
         'shilling_hijacking_89', 'group_badnwagon_127', 'group_badnwagon_74', 'group_dense_cluster_138',
         'shilling_bridge_25', 'shilling_bridge_31', 'shilling_bridge_29', 'group_camouflage_166',
         'shilling_hijacking_112', 'shilling_hijacking_138'],
        ['group_badnwagon_83', 'group_badnwagon_73', 'group_dense_cluster_152', 'group_camouflage_193',
         'group_badnwagon_148', 'group_dense_cluster_115', 'shilling_hijacking_153', 'group_badnwagon_134',
         'group_dense_cluster_32', 'group_camouflage_43', 'group_camouflage_174', 'group_camouflage_171',
         'shilling_high_deg_10', 'group_dense_cluster_48', 'group_badnwagon_3', 'group_dense_cluster_7',
         'shilling_hijacking_161', 'shilling_hijacking_199', 'shilling_hijacking_122', 'shilling_hijacking_37',
         'group_camouflage_118', 'group_dense_cluster_127', 'shilling_hijacking_73'],
        ['group_camouflage_5', 'group_badnwagon_6', 'group_camouflage_96', 'group_dense_cluster_186',
         'group_camouflage_129', 'group_badnwagon_104', 'group_badnwagon_96', 'group_badnwagon_22',
         'group_badnwagon_11', 'group_badnwagon_176', 'group_badnwagon_166', 'group_badnwagon_97',
         'group_camouflage_42', 'shilling_high_deg_24', 'group_camouflage_70', 'group_dense_cluster_159',
         'group_camouflage_82', 'shilling_hijacking_164', 'group_badnwagon_82', 'group_dense_cluster_39'],
        ['group_camouflage_170', 'shilling_hijacking_121', 'group_badnwagon_40', 'group_badnwagon_7',
         'group_badnwagon_57', 'shilling_hijacking_150', 'group_camouflage_154', 'group_badnwagon_8',
         'group_camouflage_186', 'group_badnwagon_1', 'shilling_hijacking_111', 'group_badnwagon_41',
         'shilling_hijacking_131', 'group_badnwagon_28', 'group_dense_cluster_170', 'shilling_hijacking_135',
         'group_camouflage_37', 'group_badnwagon_49', 'group_camouflage_39', 'group_badnwagon_35',
         'group_camouflage_78', 'group_badnwagon_58', 'shilling_hijacking_145'],
        ['group_dense_cluster_4', 'group_badnwagon_59', 'group_dense_cluster_142', 'shilling_high_deg_18',
         'shilling_high_deg_27', 'group_dense_cluster_145', 'group_badnwagon_42', 'group_dense_cluster_17',
         'group_dense_cluster_35', 'group_badnwagon_175', 'group_camouflage_68', 'group_badnwagon_112',
         'group_camouflage_41'],
        ['group_dense_cluster_178', 'shilling_high_deg_13', 'group_dense_cluster_128', 'group_camouflage_15',
         'group_dense_cluster_95', 'group_badnwagon_105', 'group_dense_cluster_2', 'group_camouflage_26',
         'group_badnwagon_133', 'group_camouflage_138', 'group_camouflage_56', 'group_badnwagon_186',
         'group_dense_cluster_182', 'group_camouflage_30', 'group_camouflage_146', 'group_camouflage_46',
         'group_dense_cluster_59'],
        ['group_badnwagon_143', 'group_badnwagon_187', 'group_dense_cluster_102', 'group_dense_cluster_181',
         'group_badnwagon_93', 'group_dense_cluster_26', 'group_dense_cluster_89', 'group_badnwagon_43',
         'group_camouflage_52', 'group_dense_cluster_73', 'group_badnwagon_151', 'group_badnwagon_150',
         'group_camouflage_55', 'group_camouflage_57', 'group_badnwagon_92', 'group_badnwagon_136',
         'group_badnwagon_61', 'group_badnwagon_173'],
        ['group_dense_cluster_124', 'group_badnwagon_156', 'group_dense_cluster_25', 'group_badnwagon_188',
         'group_dense_cluster_139', 'group_badnwagon_116', 'group_badnwagon_171', 'group_dense_cluster_173',
         'group_badnwagon_179', 'group_dense_cluster_18', 'group_dense_cluster_61', 'group_badnwagon_164',
         'group_camouflage_188', 'group_dense_cluster_44', 'group_camouflage_67', 'group_dense_cluster_24'],
        ['group_camouflage_38', 'group_camouflage_18', 'group_badnwagon_99', 'group_camouflage_73',
         'group_badnwagon_68', 'group_camouflage_65', 'group_badnwagon_199', 'group_badnwagon_174',
         'group_camouflage_152', 'group_dense_cluster_153', 'shilling_high_deg_2', 'group_dense_cluster_87',
         'group_badnwagon_31', 'group_dense_cluster_157', 'group_dense_cluster_123', 'group_badnwagon_51',
         'group_dense_cluster_120', 'group_badnwagon_113'],
        ['group_dense_cluster_12', 'group_camouflage_85', 'group_badnwagon_154', 'group_dense_cluster_45',
         'group_badnwagon_10', 'group_dense_cluster_172', 'group_camouflage_0', 'group_badnwagon_178',
         'group_camouflage_51', 'group_camouflage_27', 'group_camouflage_149', 'group_camouflage_140',
         'group_dense_cluster_146', 'group_dense_cluster_98', 'group_dense_cluster_52', 'group_camouflage_106'],
        ['group_badnwagon_71', 'shilling_high_deg_35', 'group_camouflage_199', 'shilling_high_deg_3',
         'shilling_high_deg_45', 'shilling_high_deg_8', 'shilling_high_deg_57', 'group_dense_cluster_133',
         'group_dense_cluster_195', 'group_camouflage_115', 'shilling_high_deg_54', 'shilling_high_deg_4',
         'group_badnwagon_162'],
        ['shilling_high_deg_40', 'shilling_high_deg_48', 'shilling_high_deg_44', 'shilling_high_deg_42']]
    groups_active_mab_1 = [
        ['shilling_hijacking_46', 'shilling_hijacking_171', 'shilling_hijacking_56', 'shilling_hijacking_146',
         'shilling_hijacking_118', 'shilling_hijacking_142', 'shilling_hijacking_18', 'shilling_hijacking_101',
         'shilling_hijacking_24', 'group_badnwagon_172', 'group_dense_cluster_63', 'group_camouflage_124',
         'shilling_hijacking_27', 'shilling_hijacking_126', 'shilling_hijacking_9', 'group_camouflage_151',
         'group_camouflage_196', 'shilling_hijacking_76', 'shilling_hijacking_94', 'shilling_hijacking_119',
         'shilling_hijacking_183', 'group_badnwagon_77', 'shilling_hijacking_55', 'shilling_hijacking_64',
         'shilling_hijacking_128', 'shilling_high_deg_53', 'group_camouflage_71', 'group_dense_cluster_105',
         'shilling_hijacking_32', 'group_dense_cluster_1', 'group_dense_cluster_151', 'group_dense_cluster_90',
         'shilling_high_deg_55', 'shilling_hijacking_137', 'shilling_hijacking_130', 'group_camouflage_116',
         'group_badnwagon_30', 'shilling_hijacking_188'],
        ['shilling_hijacking_68', 'group_camouflage_155', 'group_badnwagon_111', 'shilling_bridge_34',
         'shilling_bridge_15', 'shilling_bridge_31', 'shilling_hijacking_197', 'group_dense_cluster_22',
         'group_badnwagon_74', 'shilling_bridge_26', 'group_camouflage_103', 'shilling_bridge_24',
         'shilling_hijacking_112', 'group_camouflage_131', 'shilling_bridge_36', 'shilling_hijacking_89',
         'shilling_high_deg_16', 'shilling_hijacking_62', 'shilling_hijacking_138', 'shilling_hijacking_132',
         'shilling_hijacking_5', 'group_badnwagon_138', 'shilling_hijacking_38', 'group_badnwagon_124',
         'shilling_bridge_33', 'group_badnwagon_119', 'shilling_hijacking_77', 'group_dense_cluster_138',
         'group_camouflage_166', 'group_badnwagon_127', 'shilling_bridge_13', 'shilling_bridge_2', 'group_badnwagon_54',
         'shilling_hijacking_2', 'shilling_bridge_25', 'group_badnwagon_139', 'group_badnwagon_39',
         'group_camouflage_59', 'group_dense_cluster_121', 'shilling_hijacking_7', 'shilling_hijacking_23',
         'group_dense_cluster_92', 'shilling_bridge_29'],
        ['group_badnwagon_73', 'group_dense_cluster_127', 'group_badnwagon_134', 'shilling_hijacking_199',
         'shilling_hijacking_161', 'group_badnwagon_83', 'group_dense_cluster_152', 'shilling_high_deg_10',
         'shilling_hijacking_153', 'group_badnwagon_148', 'group_camouflage_70', 'shilling_hijacking_122',
         'group_camouflage_171', 'shilling_hijacking_37', 'group_badnwagon_22', 'group_camouflage_118',
         'group_dense_cluster_32', 'group_camouflage_174', 'group_badnwagon_3', 'group_camouflage_193',
         'group_dense_cluster_7', 'group_dense_cluster_48', 'shilling_hijacking_73'],
        ['group_badnwagon_104', 'shilling_hijacking_131', 'shilling_hijacking_164', 'group_camouflage_42',
         'shilling_hijacking_135', 'group_badnwagon_57', 'group_badnwagon_6', 'group_camouflage_154',
         'shilling_hijacking_111', 'group_badnwagon_96', 'group_badnwagon_41', 'group_dense_cluster_159',
         'shilling_hijacking_150', 'group_dense_cluster_115', 'shilling_hijacking_145', 'group_dense_cluster_186',
         'group_dense_cluster_39', 'group_badnwagon_97', 'group_camouflage_68', 'group_camouflage_5',
         'shilling_high_deg_24', 'group_camouflage_43', 'group_badnwagon_175', 'shilling_hijacking_121'],
        ['group_badnwagon_176', 'group_badnwagon_105', 'group_badnwagon_58', 'group_badnwagon_1',
         'group_camouflage_186', 'group_camouflage_37', 'group_camouflage_78', 'group_badnwagon_49',
         'group_badnwagon_59', 'group_dense_cluster_142', 'group_camouflage_82', 'group_badnwagon_112',
         'group_dense_cluster_17', 'group_dense_cluster_146', 'group_badnwagon_28', 'group_camouflage_96',
         'group_badnwagon_40'],
        ['group_camouflage_129', 'group_dense_cluster_35', 'group_camouflage_170', 'group_dense_cluster_2',
         'shilling_high_deg_18', 'group_dense_cluster_4', 'group_badnwagon_42', 'group_dense_cluster_182',
         'shilling_high_deg_27', 'group_camouflage_26', 'group_camouflage_41', 'group_badnwagon_68',
         'group_badnwagon_82', 'group_dense_cluster_128', 'group_badnwagon_166', 'group_camouflage_138'],
        ['group_dense_cluster_145', 'group_camouflage_55', 'group_dense_cluster_181', 'group_badnwagon_187',
         'group_badnwagon_133', 'group_badnwagon_151', 'group_badnwagon_43', 'group_camouflage_146',
         'shilling_high_deg_13', 'group_badnwagon_136', 'group_dense_cluster_95', 'group_camouflage_30',
         'group_dense_cluster_178', 'group_badnwagon_11', 'group_badnwagon_61', 'group_badnwagon_143',
         'group_camouflage_52'],
        ['group_camouflage_39', 'group_badnwagon_35', 'group_dense_cluster_24', 'group_dense_cluster_120',
         'group_dense_cluster_170', 'group_camouflage_15', 'group_badnwagon_179', 'group_camouflage_56',
         'group_camouflage_46', 'group_dense_cluster_59', 'group_badnwagon_188', 'group_badnwagon_173',
         'group_badnwagon_116', 'group_badnwagon_186', 'group_dense_cluster_73', 'group_dense_cluster_61'],
        ['group_dense_cluster_173', 'group_badnwagon_93', 'group_dense_cluster_18', 'group_dense_cluster_124',
         'group_dense_cluster_102', 'group_badnwagon_7', 'group_camouflage_188', 'group_badnwagon_51',
         'group_dense_cluster_139', 'group_camouflage_57', 'group_badnwagon_171', 'group_badnwagon_99',
         'group_dense_cluster_26', 'group_dense_cluster_25', 'group_badnwagon_8', 'group_badnwagon_199',
         'group_badnwagon_156'],
        ['group_camouflage_85', 'group_camouflage_152', 'group_dense_cluster_89', 'group_camouflage_18',
         'shilling_high_deg_2', 'group_camouflage_67', 'group_camouflage_140', 'group_dense_cluster_157',
         'group_badnwagon_174', 'group_camouflage_38', 'group_camouflage_65', 'group_dense_cluster_123',
         'group_dense_cluster_44', 'group_badnwagon_178', 'group_dense_cluster_153', 'group_badnwagon_164',
         'group_badnwagon_92', 'group_badnwagon_150'],
        ['group_badnwagon_162', 'group_camouflage_51', 'group_badnwagon_10', 'group_dense_cluster_45',
         'group_dense_cluster_87', 'group_badnwagon_31', 'group_dense_cluster_12', 'group_dense_cluster_98',
         'group_camouflage_149', 'group_badnwagon_113', 'group_dense_cluster_52', 'group_badnwagon_154',
         'group_camouflage_106', 'group_camouflage_0', 'group_dense_cluster_172', 'group_camouflage_73'],
        ['group_camouflage_199', 'group_dense_cluster_195', 'shilling_high_deg_57', 'group_camouflage_115',
         'group_dense_cluster_133', 'shilling_high_deg_4', 'shilling_high_deg_35', 'group_badnwagon_71',
         'shilling_high_deg_3', 'shilling_high_deg_54', 'group_camouflage_27', 'shilling_high_deg_8',
         'shilling_high_deg_45'],
        ['shilling_high_deg_44', 'shilling_high_deg_40', 'shilling_high_deg_48', 'shilling_high_deg_42']]
    # groups_active_mab_1 = [['group_badnwagon_124', 'group_badnwagon_174', 'shilling_bridge_2', 'shilling_bridge_29', 'shilling_bridge_31', 'group_badnwagon_54', 'shilling_bridge_26', 'shilling_bridge_34', 'group_dense_cluster_35', 'group_dense_cluster_92', 'group_badnwagon_175', 'group_dense_cluster_87', 'shilling_bridge_15', 'group_camouflage_155', 'group_dense_cluster_95', 'group_badnwagon_11', 'shilling_bridge_24', 'group_dense_cluster_102', 'group_dense_cluster_151', 'group_camouflage_52', 'shilling_hijacking_197', 'shilling_hijacking_94', 'shilling_bridge_36', 'shilling_hijacking_188', 'group_dense_cluster_127', 'shilling_bridge_33', 'shilling_bridge_13', 'shilling_high_deg_16', 'group_camouflage_131', 'group_badnwagon_74', 'shilling_bridge_25', 'group_badnwagon_164', 'shilling_hijacking_183'], ['group_camouflage_27', 'shilling_hijacking_132', 'group_badnwagon_151', 'group_dense_cluster_26', 'group_camouflage_5', 'shilling_hijacking_9', 'group_camouflage_55', 'group_badnwagon_3', 'group_badnwagon_93', 'group_badnwagon_113', 'group_badnwagon_61', 'group_camouflage_140', 'group_badnwagon_92', 'group_dense_cluster_89', 'group_camouflage_67', 'group_badnwagon_150', 'group_badnwagon_143', 'shilling_hijacking_101', 'shilling_high_deg_27', 'shilling_hijacking_171', 'group_badnwagon_99', 'group_dense_cluster_128'], ['group_dense_cluster_39', 'group_badnwagon_73', 'group_dense_cluster_1', 'group_camouflage_116', 'group_badnwagon_30', 'group_dense_cluster_105', 'group_badnwagon_119', 'group_badnwagon_77', 'shilling_hijacking_56', 'group_camouflage_59', 'group_dense_cluster_22', 'group_camouflage_166', 'shilling_hijacking_112', 'shilling_hijacking_2', 'group_camouflage_151', 'group_badnwagon_172', 'shilling_hijacking_118', 'group_dense_cluster_90', 'shilling_hijacking_32', 'shilling_hijacking_142', 'group_badnwagon_41', 'group_dense_cluster_121', 'group_badnwagon_39'], ['group_camouflage_196', 'group_dense_cluster_115', 'shilling_hijacking_77', 'shilling_hijacking_37', 'group_dense_cluster_152', 'shilling_hijacking_73', 'shilling_hijacking_137', 'group_badnwagon_96', 'group_camouflage_118', 'shilling_hijacking_199', 'shilling_hijacking_76', 'shilling_hijacking_126', 'group_badnwagon_134', 'group_camouflage_124', 'shilling_hijacking_62', 'group_badnwagon_83', 'group_camouflage_37', 'shilling_high_deg_10', 'group_badnwagon_6', 'shilling_hijacking_128', 'group_camouflage_171', 'shilling_hijacking_130', 'shilling_hijacking_161', 'group_camouflage_174', 'shilling_hijacking_38', 'shilling_hijacking_18', 'group_badnwagon_104', 'shilling_hijacking_27', 'shilling_hijacking_64', 'shilling_hijacking_7', 'shilling_hijacking_135', 'shilling_hijacking_23', 'group_badnwagon_176', 'shilling_hijacking_68', 'shilling_hijacking_5', 'group_dense_cluster_48', 'group_camouflage_193', 'shilling_hijacking_138', 'shilling_hijacking_153'], ['shilling_hijacking_46', 'shilling_hijacking_121', 'shilling_high_deg_13', 'shilling_hijacking_131', 'group_dense_cluster_59', 'shilling_hijacking_111', 'group_camouflage_103', 'group_camouflage_154', 'group_dense_cluster_7', 'shilling_hijacking_89', 'group_camouflage_71', 'group_camouflage_65', 'shilling_high_deg_18', 'group_camouflage_82', 'group_camouflage_146', 'group_badnwagon_68', 'group_camouflage_39', 'shilling_hijacking_24', 'shilling_hijacking_119', 'group_badnwagon_22', 'group_dense_cluster_32', 'group_badnwagon_139', 'group_dense_cluster_4', 'shilling_hijacking_164', 'group_camouflage_42', 'shilling_hijacking_145', 'shilling_hijacking_122', 'shilling_hijacking_55', 'shilling_hijacking_150', 'shilling_hijacking_146', 'group_dense_cluster_142'], ['group_camouflage_68', 'group_camouflage_170', 'group_dense_cluster_181', 'group_badnwagon_49', 'group_camouflage_96', 'group_badnwagon_148', 'group_dense_cluster_186', 'shilling_high_deg_24', 'group_dense_cluster_2', 'group_camouflage_186', 'group_camouflage_129', 'group_dense_cluster_159', 'group_badnwagon_156', 'group_badnwagon_43', 'group_badnwagon_59'], ['group_badnwagon_82', 'group_dense_cluster_138', 'group_dense_cluster_182', 'group_badnwagon_58', 'group_camouflage_38', 'group_badnwagon_112', 'group_badnwagon_97', 'group_badnwagon_166', 'group_camouflage_26', 'group_dense_cluster_63', 'group_dense_cluster_18', 'group_badnwagon_40', 'group_camouflage_70', 'group_badnwagon_187', 'group_badnwagon_138', 'group_dense_cluster_145', 'group_dense_cluster_178'], ['group_camouflage_46', 'group_badnwagon_35', 'group_dense_cluster_173', 'group_badnwagon_171', 'group_dense_cluster_44', 'group_camouflage_30', 'group_badnwagon_133', 'group_badnwagon_136', 'group_dense_cluster_153', 'group_camouflage_56', 'group_badnwagon_186', 'group_dense_cluster_73', 'group_dense_cluster_123', 'group_dense_cluster_170', 'group_badnwagon_173', 'group_badnwagon_28', 'group_badnwagon_127', 'group_badnwagon_7'], ['group_badnwagon_51', 'group_badnwagon_31', 'group_camouflage_78', 'group_badnwagon_8', 'group_badnwagon_10', 'group_badnwagon_199', 'group_dense_cluster_24', 'group_camouflage_18', 'group_camouflage_0', 'group_camouflage_73', 'group_camouflage_15', 'group_dense_cluster_61', 'group_badnwagon_179', 'group_dense_cluster_139', 'group_camouflage_57', 'group_camouflage_43', 'group_dense_cluster_157'], ['group_badnwagon_42', 'group_dense_cluster_98', 'group_badnwagon_188', 'group_camouflage_188', 'group_dense_cluster_124', 'group_dense_cluster_25', 'shilling_high_deg_2', 'group_dense_cluster_45', 'group_badnwagon_116', 'group_camouflage_41', 'group_camouflage_106', 'group_dense_cluster_52', 'group_badnwagon_111', 'group_dense_cluster_120', 'group_badnwagon_105', 'group_badnwagon_57'], ['group_dense_cluster_12', 'group_dense_cluster_146', 'group_dense_cluster_133', 'group_camouflage_199', 'group_camouflage_152', 'group_dense_cluster_17', 'group_camouflage_138', 'group_badnwagon_1', 'group_camouflage_149', 'group_badnwagon_162', 'group_camouflage_115', 'group_badnwagon_71', 'group_badnwagon_178', 'group_camouflage_85', 'group_camouflage_51', 'group_dense_cluster_172', 'group_badnwagon_154'], ['shilling_high_deg_53', 'shilling_high_deg_4', 'shilling_high_deg_48', 'group_dense_cluster_195', 'shilling_high_deg_54', 'shilling_high_deg_8', 'shilling_high_deg_45', 'shilling_high_deg_35', 'shilling_high_deg_57', 'shilling_high_deg_42'], ['shilling_high_deg_40', 'shilling_high_deg_44', 'shilling_high_deg_3', 'shilling_high_deg_55']]

    flat_a = [x for sublist in groups_ative_baselines for x in sublist]
    flat_b = [x for sublist in groups_active_mab for x in sublist]
    overlap = set(flat_a) & set(flat_b)

    # groups = [['group_badnwagon_0', 'group_badnwagon_10', 'group_badnwagon_100', 'group_badnwagon_101', 'group_badnwagon_103', 'group_badnwagon_104', 'group_badnwagon_105', 'group_badnwagon_106', 'group_badnwagon_107', 'group_badnwagon_108', 'group_badnwagon_109', 'group_badnwagon_11', 'group_badnwagon_110', 'group_badnwagon_112', 'group_badnwagon_113', 'group_badnwagon_114', 'group_badnwagon_115', 'group_badnwagon_116', 'group_badnwagon_119', 'group_badnwagon_12']]
    for group_to_keep in groups_active_mab_1:
        groups = []
        features, defect_ids, col_names, group0 = load_embeddings(dataset["pt_path"])
        labels = load_labels(dataset["json_path"])
        # X_tr, y_tr, ids, group = align(features, defect_ids, group0, labels, drop_ids=te_ids)
        X_tr, y_tr, ids, group = align(features, defect_ids, group0, labels, drop_ids=te_ids)
        y_tr = np.abs(y_tr)
        # y_tr0 = np.abs(y_tr0)
        # print(ids)
        X_tr, y_tr, ids = shuffle(X_tr, y_tr, ids)
        X_tr = scaler.fit_transform(X_tr)
        # X_tr0 = scaler.fit_transform(X_tr0)
        # X_test = scaler.transform(X_test)
        mse_list = []
        mae_list = []
        rmse_list = []
        r2_list = []
        spearman_list = []
        batch_size = 50

        # spearman_list.append(spearman_corr)

        for start in range(0, len(X_tr), batch_size):
            end = start + batch_size
            X_batch = X_tr[start:end]
            y_batch = y_tr[start:end]
            ids_batch = ids[start:end]

            groups.append(ids_batch)

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
    # return model
    # plot = True
    # if plot:
    #     df = pd.DataFrame({
    #         "batch": np.arange(0, len(mse_list)),
    #         "mse": mse_list,
    #         "mae": mae_list,
    #         "rmse": rmse_list,
    #         "r2": r2_list,
    #         "spearman": spearman_list
    #     })
    #     df["train_samples"] = df["batch"] * 50
    #     import seaborn as sns
    #     import matplotlib.pyplot as plt
    #
    #     sns.set_theme(style="whitegrid")
    #
    #     plt.figure(figsize=(10, 5))
    #
    #     sns.lineplot(data=df, x="batch", y="mse", label="MSE")
    #     sns.lineplot(data=df, x="batch", y="mae", label="MAE")
    #     sns.lineplot(data=df, x="batch", y="rmse", label="RMSE")
    #     plt.xticks(np.arange(0, df["batch"].max()+1, 1))
    #     plt.xlabel("Batch")
    #     plt.ylabel("Error")
    #     plt.title("Error metrics over samples")
    #     plt.legend()
    #     plt.savefig("./analysis_learning/error_metrics_active_learning.pdf", format="pdf", bbox_inches="tight")
    #     # plt.show()
    #     # plt.show()
    #
    #     plt.figure(figsize=(10, 5))
    #
    #     sns.lineplot(data=df, x="batch", y="r2", label="R²")
    #     #sns.lineplot(data=df, x="batch", y="spearman", label="Spearman")
    #     plt.xticks(np.arange(0, df["batch"].max()+1, 1))
    #     plt.xlabel("Batch")
    #     plt.ylabel("Score")
    #     plt.title("Model quality over batches")
    #     plt.legend()
    #     plt.savefig("./analysis_learning/quality_metrics_active_learning.pdf", format="pdf", bbox_inches="tight")
    #     #plt.show()

    # fig, axs = plt.subplots(2, 1, figsize=(12, 8))
    #
    # # errori
    # axs[0].plot(batches, mse_list, label="MSE")
    # axs[0].plot(batches, mae_list, label="MAE")
    # axs[0].plot(batches, rmse_list, label="RMSE")
    # axs[0].set_title("Error metrics")
    # axs[0].legend()
    # axs[0].grid()
    #
    # # qualità modello
    # axs[1].plot(batches, r2_list, label="R²")
    # axs[1].plot(batches, spearman_list, label="Spearman")
    # axs[1].set_title("Correlation metrics")
    # axs[1].legend()
    # axs[1].grid()
    #
    # plt.tight_layout()
    # plt.show()


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


def train_regressor(datasets, dataset_active, path_csv):
    # sgdregressor
    model, scaler, X_train, y_train = initial_fit_sgd(
        datasets=datasets,
        loss="squared_error",
        alpha=1e-9,
        learning_rate="invscaling",
        random_state=42,
        save_path=f"{BASE_DIR}/results/sgd_regressor.joblib",
    )

    # testo il modello allenato su sports
    X, ids = prepare_single_datasets_for_inference(dataset_active)

    X = scaler.transform(X)
    y_pred = model.predict(X)






if __name__ == "__main__":
    initial_datasets = [
        {
            "name": "Office_Products",
            "pt_path": f"{BASE_DIR}/node_embeddings/defects_embeddings_Office_Products_sage_new_version_2.pt",
            "json_path": f"{BASE_DIR}/labels/labels_Office_Products.json",
        },
        {
            "name": "Toys_and_Games",
            "pt_path": f"{BASE_DIR}/node_embeddings/defects_embeddings_Toys_and_Games_sage_new_version_2.pt",
            "json_path": f"{BASE_DIR}/labels/labels_Toys_and_Games.json",
        },

        {
            "name": "Pet_Supplies",
            "pt_path": f"{BASE_DIR}/node_embeddings/defects_embeddings_Pet_Supplies_sage_new_version_2.pt",
            "json_path": f"{BASE_DIR}/labels/labels_Pet_Supplies.json",
        },


    ]



    train_regressor(initial_datasets[0:4], initial_datasets[4], path_csv)



