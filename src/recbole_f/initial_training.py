
import argparse
import json
import logging
import math
import time
from collections import Counter
from pathlib import Path
import torch
import torch.nn.functional as F
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import get_model, get_trainer, init_seed
import numpy as np

try:
    from sklearn.metrics import roc_auc_score
    _SKLEARN_OK = True
except ImportError:
    _SKLEARN_OK = False

# try:
#     from recbole.config import Config
#     from recbole.data import create_dataset, data_preparation
#     from recbole.utils import get_model, get_trainer, init_seed
#
#
# except ImportError as e:
#     print(e)
#     import sys
#     sys.exit("RecBole non installato. Esegui:  pip install recbole")

# Compatibilità PyTorch ≥ 2.6
_torch_load_orig = torch.load
def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _torch_load_orig(*args, **kwargs)
torch.load = _torch_load_compat

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)




# ==============================================================================
#  PATHS
# ==============================================================================
import os
try:
    method = os.environ.get("METHOD")
except Exception as e:
    print(e)
    method = "LightGCN"
print(f'METHOD: {method}')

BASE_DIR      = Path(__file__).resolve().parent
BASE_DATA_DIR = BASE_DIR.parent.parent / "data"
CONFIGS_DIR   = BASE_DIR
_ds_suffix = os.environ.get("DATASET_NAME", "all")
RESULTS_FILE  = BASE_DIR / "results" / f"results_summary_{_ds_suffix}_3v3t.jsonl"


ROOT_DIR      = Path(__file__).resolve().parent
print(f"ROOT_DIR {ROOT_DIR}")
BASE_DATA_DIR = (ROOT_DIR / "../data/original/").resolve()
CONFIGS_DIR   = ROOT_DIR



SAVED_DIR     = ROOT_DIR / "saved_cf"
STAGING_DIR   = ROOT_DIR / "staging_cf"

# ==============================================================================
#  DATASETS
# ==============================================================================
DATASETS = {
    "Books": BASE_DATA_DIR / "Books",
    "Toys_and_Games": BASE_DATA_DIR / "Toys_and_Games",
    "Baby_Products": BASE_DATA_DIR / "Baby_Products",
    "Beauty_and_Personal_Care": BASE_DATA_DIR / "Beauty_and_Personal_Care",
    "Pet_Supplies": BASE_DATA_DIR / "Pet_Supplies",
    "yelp": BASE_DATA_DIR / "yelp",
}



ALL_MODELS = ["LightGCN"]

# Modelli che fanno rating prediction invece di ranking top-k.
# Hanno pipeline diverso: niente neg sampling, niente full_sort_predict,
# metriche RMSE/MAE/AUC invece di Recall/NDCG.
RATING_MODELS = {}

def is_rating_model(model_name: str) -> bool:
    return model_name in RATING_MODELS


_COMMON_LIGHTGCN = {
    "n_layers":            3,
    "reg_weight":          1.0e-4,
    "learning_rate":       1.0e-3,
    "metrics":             ["Recall", "NDCG", "MRR", "Hit"],
    "topk":                [10, 20, 50],
    "valid_metric":        "NDCG@20",
    "valid_metric_bigger": True,
}

_TIERS = {
    "small": {    # < 1M interazioni, ≤ ~80K item
        "embedding_size":   64,
        "train_batch_size": 2048,
        "eval_batch_size":  4_000_000,   # ~50-200 utenti per batch
        "epochs":           300,
        "eval_step":        10,
        "stopping_step":    5,          # patience 20 epoche
    },
    "medium": {   # 1M-2M interazioni
        "embedding_size":   128,
        "train_batch_size": 4096,
        "eval_batch_size":  10_000_000,  # ~80-140 utenti per batch
        "epochs":           400,
        "eval_step":        10,
        "stopping_step":    5,           # patience 30 epoche
    },
    "large": {    # > 2M interazioni
        "embedding_size":   128,
        "train_batch_size": 8192,
        "eval_batch_size":  20_000_000,  # ~64 utenti per batch su Books
        "epochs":           500,
        "eval_step":        10,
        "stopping_step":    5,           # patience 30 epoche
    },
}

_DATASET_TIER = {
    "Baby_Products":            "small",   # 258K inter | 26K users | 19K items
    "Office_Products":          "small",   # 289K inter | 28K users | 30K items
    "Sports_and_Outdoors":      "small",   # DA VERIFICARE: il commento era copiato da Office
    "Toys_and_Games":           "small",   # 926K inter | 79K users | 83K items
    "Pet_Supplies":             "medium",  # 1.47M inter | 139K users | 71K items
    "Beauty_and_Personal_Care": "medium",  # 1.86M inter | 145K users | 123K items
    "Health_and_Household": "medium",  # 1.86M inter | 145K users | 123K items
    "Beauty_and_Personal_Care_injseg": "medium",  # 1.86M inter | 145K users | 123K items
    "Books":                    "large",   # 3.98M inter | 217K users | 311K items
    "Electronics":                    "large",   # 3.98M inter | 217K users | 311K items
    "Books_injseg":                    "large",   # 3.98M inter | 217K users | 311K items
    "yelp":                     "large",   # DA VERIFICARE
    "yelp_injseg": "large",  # DA VERIFICARE
}

MODEL_HPARAMS_CF: dict[str, dict[str, dict]] = {
    ds: {"LightGCN": {**_COMMON_LIGHTGCN, **_TIERS[tier]}}
    for ds, tier in _DATASET_TIER.items()
}
# MODEL_HPARAMS_CF: dict[str, dict[str, dict]] = {
#
#     # ── Dataset piccoli (<1M interazioni) ───────────────────────────────────
#     # embedding piccoli bastano, reg/lr standard
#     "Baby_Products": {
#         "LightGCN": {
#             "embedding_size": 64,
#             "n_layers":       2,
#             "reg_weight":     1.0e-4,
#             "learning_rate":  0.001,
#             "epochs":         500,
#         },
#         # train: 258,423 ratings | 25,809 users | 18,753 items | sparsity 99.947%
#     },
#     "Office_Products": {
#         "LightGCN": {
#             "embedding_size": 64,
#             "n_layers":       2,
#             "reg_weight":     1.0e-4,
#             "learning_rate":  0.001,
#             "epochs":         500,
#         },
#         # train: 288,942 ratings | 27,796 users | 29,918 items | sparsity 99.965%
#
#     },
#
#     "Sports_and_Outdoors": {
#         "LightGCN": {
#             "embedding_size": 64,
#             "n_layers": 2,
#             "reg_weight": 1.0e-4,
#             "learning_rate": 0.001,
#             "epochs": 500,
#         },
#         # train: 288,942 ratings | 27,796 users | 29,918 items | sparsity 99.965%
#     },
#     "Toys_and_Games": {
#         "LightGCN": {
#             "embedding_size": 128,
#             "n_layers":       3,
#             "reg_weight":     1.0e-4,
#             "learning_rate":  0.0005,
#             "epochs":         500,
#         },
#         # train: 926,333 ratings | 79,223 users | 82,649 items | sparsity 99.986%
#     },
#
#     # ── Dataset grandi (1M-5M interazioni) ──────────────────────────────────
#     # embedding più grandi, reg più alta, lr più basso (gradienti noisier),
#     # n_layers ridotto a 2 (da 3) per evitare over-smoothing su sparsità estrema
#     "Pet_Supplies": {
#         "LightGCN": {
#             "embedding_size": 128,
#             "n_layers":       3,
#             "reg_weight":     1.0e-3,
#             "learning_rate":  0.0005,
#             "epochs":         500,
#         },
#         # train: 1,474,606 ratings | 139,006 users | 71,264 items | sparsity 99.985%
#     },
#     "Beauty_and_Personal_Care": {
#         "LightGCN": {
#             "embedding_size": 128,
#             "n_layers":       3,
#             "reg_weight":     1.0e-3,
#             "learning_rate":  0.0005,
#             "epochs":         500,
#         },
#         # train: 1,859,237 ratings | 145,173 users | 123,447 items | sparsity 99.990%
#     },
#     "Books": {
#         "LightGCN": {
#             "embedding_size": 128,
#             "n_layers":       3,
#             "reg_weight":     1.0e-3,
#             "learning_rate":  0.0001,
#             "epochs":         500,
#         },
#         # train: 3,981,546 ratings | 216,588 users | 311,184 items | sparsity 99.994%
#     },
#     "yelp": {
#         "LightGCN": {
#             "embedding_size": 128,
#             "n_layers": 3,
#             "reg_weight": 1.0e-3,
#             "learning_rate": 0.0001,
#             "epochs": 500,
#         },
#         # train: 3,981,546 ratings | 216,588 users | 311,184 items | sparsity 99.994%
#     },
# }



def get_model_hparams(dataset_name: str, model: str) -> dict:
    """
    Restituisce gli iperparametri CF per la coppia (dataset, model).
    Se il dataset non è nel registry usa un dizionario vuoto
    (i valori restano quelli del file .yaml).
    Logga sempre quali parametri vengono applicati.
    """
    hparams = MODEL_HPARAMS_CF.get(dataset_name, {}).get(model, {})
    if hparams:
        log.info(
            f"  [hparams] {dataset_name}/{model} → "
            + ", ".join(f"{k}={v}" for k, v in hparams.items())
        )
    else:
        log.warning(
            f"  [hparams] Nessun override per {dataset_name}/{model}: "
            "uso i valori dal file .yaml"
        )
    return hparams


# ==============================================================================
#  NOISE REGISTRY — identico a run_experiments.py
# ==============================================================================
def build_noise_registry(dataset_name: str) -> dict:
    return {
        "clean": {
            "subdir": "",
            "levels": {"0.0": f"{dataset_name}.rec_train.csv"},
        },
    }

ALL_NOISE_TYPES = ["clean"]



# ==============================================================================
#  STAGING
# ==============================================================================
import re

def _parse_inter(
    path: Path,
    want_rating: bool = False,
) -> list[tuple]:
    """
    Parsa un file .inter e restituisce righe come tuple.
    Se want_rating=True restituisce (uid, iid, rating, ts).
    Altrimenti restituisce (uid, iid, ts).
    Se want_rating=True ma la colonna rating manca, lancia ValueError.
    """
    with path.open(encoding="utf-8") as f:
        raw_header = f.readline().rstrip("\n")
        lines = [l.rstrip("\n") for l in f if l.strip()]

    if "\t" in raw_header:
        sep = "\t"
    elif re.search(r"  +", raw_header):
        sep = None
    else:
        sep = ","

    def split_line(line):
        return line.split(sep) if sep else line.split()

    cols = [c.strip().split(":")[0].lower() for c in split_line(raw_header)]
    try:
        uid_i = cols.index("user_id")
        iid_i = cols.index("item_id")
        ts_i  = cols.index("timestamp")
    except ValueError as e:
        raise ValueError(f"Colonna mancante in {path.name}: {e}")

    rat_i = None
    if want_rating:
        # Accetta "rating" o "rate" come nome colonna
        for cand in ("rating","rating:float", "rate", "ratings"):
            if cand in cols:
                rat_i = cols.index(cand)
                break
        if rat_i is None:
            raise ValueError(
                f"Colonna rating non trovata in {path.name}. "
                f"Colonne disponibili: {cols}. "
                "GCMC richiede rating espliciti nel file .inter."
            )

    rows = []
    for line in lines:
        parts = split_line(line)
        needed = max(uid_i, iid_i, ts_i) if rat_i is None else max(uid_i, iid_i, ts_i, rat_i)
        if len(parts) <= needed:
            continue
        if want_rating:
            rows.append((
                parts[uid_i].strip(),
                parts[iid_i].strip(),
                parts[rat_i].strip(),
                parts[ts_i].strip(),
            ))
        else:
            rows.append((parts[uid_i].strip(), parts[iid_i].strip(), parts[ts_i].strip()))
    return rows


# ==============================================================================
#  STAGING — split già pronto (train / valid / test)
# ==============================================================================
# Legge i tuoi file:
#     <dataset>.rec_train.csv   <dataset>.rec_valid.csv   <dataset>.rec_test.csv
# e li copia in tre file con i nomi che vuole RecBole:
#     <tag>.train.inter   <tag>.valid.inter   <tag>.test.inter
# RecBole li usa così come sono, senza rifare lo split.

SPLITS = ("train", "valid", "test")


def _find_file(dataset_dir: Path, candidates: list[str]) -> Path:
    """Restituisce il primo file esistente tra i nomi candidati."""
    for name in candidates:
        p = dataset_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Nessuno di questi file esiste in {dataset_dir}: {candidates}")


def _write_inter(dest: Path, rows: list[tuple], include_rating: bool) -> None:
    with dest.open("w", encoding="utf-8") as f:
        if include_rating:
            f.write("user_id:token\titem_id:token\trating:float\ttimestamp:float\n")
            for uid, iid, rat, ts in rows:
                f.write(f"{uid}\t{iid}\t{rat}\t{ts}\n")
        else:
            f.write("user_id:token\titem_id:token\ttimestamp:float\n")
            for uid, iid, ts in rows:
                f.write(f"{uid}\t{iid}\t{ts}\n")


def stage_dataset(
    dataset_name: str,
    dataset_dir: Path,
    noise_type: str,
    level: str,
    train_filename: str,
    train_subdir: str,
    staging_root: Path,
    include_rating: bool = False,
) -> tuple[Path, str]:
    tag        = f"{dataset_name}__{noise_type}__{level}"
    tag_staged = f"{tag}__rated" if include_rating else tag

    # I tuoi file di partenza
    train_src = dataset_dir / train_subdir / train_filename
    if not train_src.exists():
        raise FileNotFoundError(f"File mancante: {train_src}")
    sources = {
        "train": train_src,
        "valid": _find_file(dataset_dir, [f"{dataset_name}.rec_valid.csv",
                                          f"{dataset_name}.rec_vali.csv"]),
        "test":  _find_file(dataset_dir, [f"{dataset_name}.rec_test.csv"]),
    }

    # I file che legge RecBole
    stage_r = staging_root / tag_staged
    stage_r.mkdir(parents=True, exist_ok=True)
    dests = {s: stage_r / f"{tag_staged}.{s}.inter" for s in SPLITS}

    # Se i tre file esistono già e sono aggiornati, non li riscrivo
    newest_src = max(p.stat().st_mtime for p in sources.values())
    if all(d.exists() for d in dests.values()) and \
            newest_src <= min(d.stat().st_mtime for d in dests.values()):
        return stage_r, tag_staged

    counts = {}
    for split in SPLITS:
        rows = _parse_inter(sources[split], want_rating=include_rating)
        _write_inter(dests[split], rows, include_rating)
        counts[split] = len(rows)

    # Il vecchio file unico non serve più
    old_single = stage_r / f"{tag_staged}.inter"
    if old_single.exists():
        old_single.unlink()

    log.info(f"  Staged {tag_staged}: train={counts['train']}  "
             f"valid={counts['valid']}  test={counts['test']}")
    return stage_r, tag_staged

def load_user_set(path: Path) -> set[str]:
    if not path.exists():
        return set()
    rows = _parse_inter(path, want_rating=False)
    return {row[0] for row in rows}


# ==============================================================================
#  UTILITY — recupera embedding item in modo robusto
#  LightGCN: model.item_embedding  (nn.Embedding)
#  NGCF:     model.embedding_dict['item_id']  (nn.Parameter)
#  GCMC:     N/A — task rating prediction, diversity non calcolata
# ==============================================================================
def get_item_emb(trained_model) -> np.ndarray | None:
    """
    Restituisce i pesi degli embedding item come numpy array.
    Gestisce LightGCN (item_embedding), NGCF (embedding_dict).
    Per GCMC restituisce None: il task è rating prediction,
    calcolare diversity sulle predizioni non ha senso.
    """
    # GCMC: cortocircuito esplicito
    if trained_model.__class__.__name__ == "GCMC":
        return None

    # LightGCN
    if hasattr(trained_model, "item_embedding"):
        emb = trained_model.item_embedding
        if hasattr(emb, "weight"):
            return emb.weight.detach().cpu().numpy()
        if isinstance(emb, torch.Tensor):
            return emb.detach().cpu().numpy()

    # NGCF
    if hasattr(trained_model, "embedding_dict"):
        ed = trained_model.embedding_dict
        if "item_id" in ed:
            return ed["item_id"].detach().cpu().numpy()

    # Fallback generico
    for attr_name in dir(trained_model):
        if "item" in attr_name.lower() and "emb" in attr_name.lower():
            try:
                attr = getattr(trained_model, attr_name)
                if hasattr(attr, "weight"):
                    return attr.weight.detach().cpu().numpy()
                if isinstance(attr, torch.Tensor):
                    return attr.detach().cpu().numpy()
            except Exception:
                pass

    log.warning("  ✗ Item embedding non trovato — diversity non calcolata")
    return None


# ==============================================================================
#  BEYOND-ACCURACY METRICS — identiche a run_experiments.py
# ==============================================================================
def calculate_entropy(topk_lists: list[list[int]]) -> float:

    all_rec = [iid for lst in topk_lists for iid in lst]
    if not all_rec:
        return 0.0
    counts     = Counter(all_rec)
    total_recs = len(all_rec)
    probs      = np.array([c / total_recs for c in counts.values()])
    entropy    = -np.sum(probs * np.log2(probs + 1e-12))
    return float(entropy)


def entropy_delta(
    base_topk_lists: list[list[int]],
    loo_topk_lists:  list[list[int]],
) -> dict:
    """
    Variazione di entropia tra raccomandazioni pre- e post- rimozione
    di un difetto (LOO), come definito nel paper (Sezione 4.1, P2):

        y_k = H(G \\ D_k) - H(G)

    IMPORTANTE: va chiamata passando SOLO le raccomandazioni degli utenti
    affetti dal difetto (U_k, cioè quelli che hanno un item del difetto
    in top-K), non l'intero test set — altrimenti l'effetto si diluisce
    nella popolazione generale (è il punto di cui parlavamo sopra).
    """
    h_base = calculate_entropy(base_topk_lists)
    h_loo  = calculate_entropy(loo_topk_lists)
    return {
        "entropy_base":  round(h_base, 4),
        "entropy_loo":   round(h_loo, 4),
        "entropy_delta": round(h_loo - h_base, 4),
    }


def compute_beyond_accuracy(
    topk_lists:           list[list[int]],
    item_embeddings:      np.ndarray | None,
    item_popularity:      dict[int, int],
    n_train_interactions: int,
    catalog_size:         int,
) -> dict:
    all_rec    = [iid for lst in topk_lists for iid in lst]
    unique_rec = set(all_rec)
    coverage   = len(unique_rec) / catalog_size if catalog_size > 0 else 0.0
    entropy    = calculate_entropy(topk_lists)   # <-- NUOVO

    nov_per_user = []
    for lst in topk_lists:
        if not lst:
            continue
        si = []
        for iid in lst:
            pop = item_popularity.get(iid, 1)
            p_i = pop / max(n_train_interactions, 1)
            si.append(-math.log2(p_i))
        nov_per_user.append(float(np.mean(si)))
    novelty = float(np.mean(nov_per_user)) if nov_per_user else 0.0

    counts = np.array(sorted(Counter(all_rec).values()), dtype=float)
    n = len(counts)
    if n > 1:
        counts /= counts.sum()
        counts.sort()
        idx  = np.arange(1, n + 1)
        gini = float((2 * np.sum(idx * counts) / n) - (n + 1) / n)
        gini = max(0.0, min(1.0, gini))
    else:
        gini = 0.0

    diversity = None
    if item_embeddings is not None:
        emb = F.normalize(
            torch.tensor(item_embeddings, dtype=torch.float32), dim=-1
        )
        div_per_user = []
        for lst in topk_lists:
            if len(lst) < 2:
                continue
            idx  = [min(i, emb.shape[0] - 1) for i in lst]
            vecs = emb[idx]
            sim  = (vecs @ vecs.T).numpy()
            k    = len(idx)
            mask = np.triu(np.ones((k, k), dtype=bool), k=1)
            avg_sim = sim[mask].mean() if mask.any() else 1.0
            div_per_user.append(1.0 - float(avg_sim))
        diversity = float(np.mean(div_per_user)) if div_per_user else 0.0

    result = {
        "coverage": round(coverage, 4),
        "novelty":  round(novelty,  4),
        "gini":     round(gini,     4),
        "entropy":  round(entropy,  4),   # <-- NUOVO
    }
    if diversity is not None:
        result["diversity"] = round(diversity, 4)
    return result


# def compute_beyond_accuracy(
#     topk_lists:           list[list[int]],
#     item_embeddings:      np.ndarray | None,
#     item_popularity:      dict[int, int],
#     n_train_interactions: int,
#     catalog_size:         int,
# ) -> dict:
#     all_rec    = [iid for lst in topk_lists for iid in lst]
#     unique_rec = set(all_rec)
#     coverage   = len(unique_rec) / catalog_size if catalog_size > 0 else 0.0
#
#     nov_per_user = []
#     for lst in topk_lists:
#         if not lst:
#             continue
#         si = []
#         for iid in lst:
#             pop = item_popularity.get(iid, 1)
#             p_i = pop / max(n_train_interactions, 1)
#             si.append(-math.log2(p_i))
#         nov_per_user.append(float(np.mean(si)))
#     novelty = float(np.mean(nov_per_user)) if nov_per_user else 0.0
#
#     counts = np.array(sorted(Counter(all_rec).values()), dtype=float)
#     n = len(counts)
#     if n > 1:
#         counts /= counts.sum()
#         counts.sort()
#         idx  = np.arange(1, n + 1)
#         gini = float((2 * np.sum(idx * counts) / n) - (n + 1) / n)
#         gini = max(0.0, min(1.0, gini))
#     else:
#         gini = 0.0
#
#     diversity = None
#     if item_embeddings is not None:
#         emb = F.normalize(
#             torch.tensor(item_embeddings, dtype=torch.float32), dim=-1
#         )
#         div_per_user = []
#         for lst in topk_lists:
#             if len(lst) < 2:
#                 continue
#             idx  = [min(i, emb.shape[0] - 1) for i in lst]
#             vecs = emb[idx]
#             sim  = (vecs @ vecs.T).numpy()
#             k    = len(idx)
#             mask = np.triu(np.ones((k, k), dtype=bool), k=1)
#             avg_sim = sim[mask].mean() if mask.any() else 1.0
#             div_per_user.append(1.0 - float(avg_sim))
#         diversity = float(np.mean(div_per_user)) if div_per_user else 0.0
#
#     result = {
#         "coverage": round(coverage, 4),
#         "novelty":  round(novelty,  4),
#         "gini":     round(gini,     4),
#     }
#     if diversity is not None:
#         result["diversity"] = round(diversity, 4)
#     return result


# ==============================================================================
#  GCMC — METRICHE RATING PREDICTION
#  RecBole calcola RMSE e MAE nativamente per GCMC via eval_args labeled.
#  AUC non è nativa per regression task: la calcoliamo in post-processing
#  binarizzando i rating con soglia POS_THRESHOLD (≥ soglia → positivo).
# ==============================================================================
POS_THRESHOLD = 4.0   # Rating ≥ 4 su scala 1-5 = interazione positiva


def compute_auc_from_ratings(
    true_ratings: np.ndarray,
    pred_ratings: np.ndarray,
) -> float | None:
    """
    Calcola AUC binarizzando i rating con soglia POS_THRESHOLD.
    Restituisce None se sklearn non è disponibile o se una classe
    è assente (tutti positivi o tutti negativi nel subset).
    """
    if not _SKLEARN_OK:
        log.warning("  sklearn non disponibile — AUC non calcolato. "
                    "Installa con: pip install scikit-learn")
        return None
    y_true = (true_ratings >= POS_THRESHOLD).astype(int)
    if len(np.unique(y_true)) < 2:
        log.warning("  AUC non calcolabile: una sola classe nel subset "
                    f"(threshold={POS_THRESHOLD}, "
                    f"positivi={y_true.sum()}/{len(y_true)})")
        return None
    try:
        return float(roc_auc_score(y_true, pred_ratings))
    except Exception as exc:
        log.warning(f"  AUC fallito: {exc}")
        return None


def eval_gcmc_subset(
    checkpoint_path: Path,
    model_name: str,
    tag: str,
    staging_root: Path,
    cfg_files: list,
    user_set: set[str],
    subset_label: str,
) -> dict:
    """
    Eval GCMC su un sottoinsieme di utenti (active o lt).
    Calcola RMSE, MAE e AUC manualmente filtrando per utente.
    Non usa full_sort_predict: itera sulle coppie (u,i) del test set.
    """
    if not user_set:
        return {"error": f"user_set vuoto per {subset_label}"}

    parameter_dict = {
        "data_path":    str(staging_root),
        "dataset":      tag,
        "save_dataset": False,
        "show_progress": False,
        "load_col":     {"inter": ["user_id", "item_id", "rating", "timestamp"]},
        # GCMC: niente neg sampling
        "train_neg_sample_args": None,
        "eval_args": {
            "group_by": "user",
            "order":    "TO",
            "mode":     "full",
            "split":    {"LS": "valid_and_test"},
        },
        "metrics":      ["RMSE", "MAE"],
        "valid_metric": "RMSE",
    }

    try:
        cfg = Config(
            model=model_name,
            dataset=tag,
            config_file_list=cfg_files,
            config_dict=parameter_dict,
        )
        print(f"CFG {cfg}")
        init_seed(cfg["seed"], cfg["reproducibility"])

        dataset = create_dataset(cfg)
        _, _, test_data = data_preparation(cfg, dataset)

        uid_field = dataset.uid_field
        token2id  = {
            dataset.id2token(uid_field, i): i
            for i in range(dataset.user_num)
        }
        valid_uids = {token2id[u] for u in user_set if u in token2id}
        if not valid_uids:
            return {"error": f"Nessun utente di {subset_label} trovato nel dataset"}

        checkpoint  = torch.load(checkpoint_path, map_location=cfg["device"])
        cfg_ckpt    = checkpoint["config"]
        model_class = get_model(cfg_ckpt["model"])
        model       = model_class(cfg_ckpt, dataset).to(cfg["device"])
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        all_preds  = []
        all_truths = []
        rating_field = dataset.label_field  # "rating" per context-aware models

        with torch.no_grad():
            for batch_tuple in test_data:
                batch = batch_tuple[0] if isinstance(batch_tuple, (tuple, list)) \
                        else batch_tuple
                batch = batch.to(cfg["device"])

                uids = batch[uid_field].cpu().tolist()
                mask = torch.tensor(
                    [u in valid_uids for u in uids], dtype=torch.bool
                )
                if not mask.any():
                    continue

                # GCMC.predict restituisce i rating predetti scalari
                preds  = model.predict(batch).cpu()
                truths = batch[rating_field].cpu()

                all_preds.extend(preds[mask].tolist())
                all_truths.extend(truths[mask].tolist())

        if not all_preds:
            return {"error": f"Nessuna predizione per {subset_label}"}

        preds_arr  = np.array(all_preds,  dtype=float)
        truths_arr = np.array(all_truths, dtype=float)

        rmse = float(np.sqrt(np.mean((preds_arr - truths_arr) ** 2)))
        mae  = float(np.mean(np.abs(preds_arr - truths_arr)))
        auc  = compute_auc_from_ratings(truths_arr, preds_arr)

        log.info(
            f"    [{subset_label}] RMSE={rmse:.4f}  MAE={mae:.4f}"
            + (f"  AUC={auc:.4f}" if auc is not None else "  AUC=N/A")
        )
        result = {"rmse": round(rmse, 4), "mae": round(mae, 4)}
        if auc is not None:
            result["auc"] = round(auc, 4)
        return result

    except Exception as exc:
        log.error(f"  ✗ Eval GCMC [{subset_label}] fallita: {exc}")
        return {"error": str(exc)}


# ==============================================================================
#  ACCURACY METRICS MANUALE (per subset di utenti)
# ==============================================================================
def _compute_accuracy_metrics(
    scores: "torch.Tensor",
    pos_items: "torch.Tensor",
    ks: list[int] = [50, 10, 20],
) -> dict:
    results = {}
    sorted_indices = torch.argsort(scores, dim=-1, descending=True)
    pos_items_exp  = pos_items.unsqueeze(1)
    rank_matrix    = (sorted_indices == pos_items_exp)
    rank           = rank_matrix.float().argmax(dim=1).float()

    for k in ks:
        hit    = (rank < k).float()
        ndcg   = (hit / torch.log2(rank + 2)).mean().item()
        recall = hit.mean().item()
        mrr    = (1.0 / (rank + 1) * hit).mean().item()

        results[f"recall@{k}"] = round(recall, 4)
        results[f"ndcg@{k}"]   = round(ndcg,   4)
        results[f"mrr@{k}"]    = round(mrr,     4)
        results[f"hit@{k}"]    = round(recall,  4)

    return results


# ==============================================================================
#  HELPER: mappa uid interno → test item
# ==============================================================================
def _build_uid_to_test_item(dataset) -> dict[int, int]:
    """
    Costruisce la mappa uid_interno → item_id del test item per ogni utente.

    Con eval_args mode=full (RS split) RecBole passa al modello solo user_id
    nel batch — il positive item non è mai presente nel batch stesso.
    Va estratto direttamente dalle interazioni del dataset.

    Con RS split ogni utente ha esattamente 1 test item (10% casuale).
    Non potendo identificarlo con certezza assoluta dall'esterno,
    usiamo l'item con indice/timestamp massimo come proxy, che è lo stesso
    criterio usato internamente da RecBole quando split=RS con ordine RO.
    In pratica, per RS puro l'ordine non conta — l'importante è avere
    UN item di riferimento per utente, e questo è coerente col trainer.
    """
    inter  = dataset.inter_feat
    uid_col = inter[dataset.uid_field]
    iid_col = inter[dataset.iid_field]

    uid_np = uid_col.numpy() if hasattr(uid_col, "numpy") else np.array(uid_col.tolist())
    iid_np = iid_col.numpy() if hasattr(iid_col, "numpy") else np.array(iid_col.tolist())

    # Cerca campo timestamp per ordinamento stabile
    ts_field = None
    for f in inter.interaction.keys():
        if "time" in f.lower() or "stamp" in f.lower():
            ts_field = f
            break

    if ts_field is not None:
        ts_col = inter[ts_field]
        ts_np  = ts_col.numpy() if hasattr(ts_col, "numpy") else np.array(ts_col.tolist())
        order  = np.lexsort((np.arange(len(uid_np)), ts_np, uid_np))
    else:
        order = np.arange(len(uid_np))

    uid_to_pos: dict[int, int] = {}
    for i in order:
        uid_to_pos[int(uid_np[i])] = int(iid_np[i])

    return uid_to_pos


def _topk_accuracy(hits: torch.Tensor, n_pos: torch.Tensor, ks) -> dict:
    """
    Metriche di accuratezza calcolate come in RecBole.

    hits:  [n_utenti, max_k] bool — hits[u, r] = True se l'item in posizione r
           della top-K dell'utente u è un item di test.
    n_pos: [n_utenti] — numero di item di test per utente (1 con split LS).
    """
    hits  = hits.float()
    n_pos = n_pos.float()
    max_k = hits.shape[1]

    ranks     = torch.arange(1, max_k + 1, dtype=torch.float)
    discounts = 1.0 / torch.log2(ranks + 1)          # 1/log2(r+1)
    cum_disc  = discounts.cumsum(0)                  # per l'IDCG

    results = {}
    for k in ks:
        h     = hits[:, :k]
        n_hit = h.sum(dim=1)

        recall = n_hit / n_pos
        hit    = (n_hit > 0).float()

        # MRR: reciproco della posizione del primo item corretto (0 se nessuno)
        first_pos = h.argmax(dim=1).float() + 1
        mrr       = torch.where(n_hit > 0, 1.0 / first_pos, torch.zeros_like(n_hit))

        # NDCG: DCG normalizzato per il DCG ideale (tutti i positivi in cima)
        dcg      = (h * discounts[:k]).sum(dim=1)
        ideal_n  = n_pos.clamp(max=k).long()
        idcg     = cum_disc[ideal_n - 1]
        ndcg     = dcg / idcg

        results[f"recall@{k}"] = round(recall.mean().item(), 4)
        results[f"ndcg@{k}"]   = round(ndcg.mean().item(),   4)
        results[f"mrr@{k}"]    = round(mrr.mean().item(),    4)
        results[f"hit@{k}"]    = round(hit.mean().item(),    4)
    return results


def eval_user_subset(
    checkpoint_path: Path,
    model_name: str,
    tag: str,
    staging_root: Path,
    cfg_files: list,
    user_set: set[str] | None,
    subset_label: str,
    base_parameter_dict: dict,
    topk: int = 10,
    ks: tuple[int, ...] = (10, 20, 50),
) -> dict:

    if user_set is not None and not user_set:
        return {"error": f"user_set vuoto per {subset_label}"}

    # Stessa configurazione del training: stesso split, stessi campi caricati
    parameter_dict = dict(base_parameter_dict)
    parameter_dict["show_progress"] = False

    try:
        cfg = Config(
            model=model_name,
            dataset=tag,
            config_file_list=cfg_files,
            config_dict=parameter_dict,
        )
        init_seed(cfg["seed"], cfg["reproducibility"])
        device = cfg["device"]

        dataset = create_dataset(cfg)
        train_data, _, test_data = data_preparation(cfg, dataset)
        train_ds  = train_data.dataset
        n_items   = dataset.item_num          # include l'item di padding con ID 0
        uid_field = dataset.uid_field
        iid_field = dataset.iid_field

        # ── Utenti da valutare: token esterni → ID interni ────────────────────
        if user_set is None:
            valid_uids = None                 # tutti gli utenti del test set
        else:
            token2id   = dataset.field2token_id[uid_field]
            valid_uids = {token2id[u] for u in user_set if u in token2id}
            n_unknown  = len(user_set) - len(valid_uids)
            if n_unknown:
                log.warning(f"  [{subset_label}] {n_unknown}/{len(user_set)} "
                            "utenti del subset non presenti nel dataset")
            if not valid_uids:
                return {"error": f"Nessun utente di {subset_label} trovato nel dataset"}

        # ── Modello ───────────────────────────────────────────────────────────
        checkpoint  = torch.load(checkpoint_path, map_location=device)
        cfg_ckpt    = checkpoint["config"]
        model_class = get_model(cfg_ckpt["model"])
        model       = model_class(cfg_ckpt, train_ds).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        # Non carichiamo "other_parameter": per LightGCN contiene gli embedding
        # propagati in cache (restore_user_e / restore_item_e), calcolati sul
        # grafo usato al momento del salvataggio. Li lasciamo ricalcolare.
        if hasattr(model, "restore_user_e"):
            model.restore_user_e = None
            model.restore_item_e = None
        model.eval()

        # ── Loop di valutazione ───────────────────────────────────────────────
        max_k      = max(max(ks), topk)
        topk_lists = []
        hits_all   = []
        npos_all   = []

        with torch.no_grad():
            for batched in test_data:
                if not (isinstance(batched, (tuple, list)) and len(batched) == 4):
                    raise RuntimeError(
                        "Il dataloader di test non è in modalità full-sort: "
                        "atteso (interaction, history_index, positive_u, positive_i)"
                    )
                interaction, history_index, positive_u, positive_i = batched
                interaction = interaction.to(device)
                uids    = interaction[uid_field]
                n_batch = uids.shape[0]

                # Score appiattiti → matrice [n_utenti, n_item]
                scores = model.full_sort_predict(interaction).view(-1, n_items)
                if scores.shape[0] != n_batch:
                    raise RuntimeError(
                        f"scores.shape={tuple(scores.shape)} incompatibile con "
                        f"{n_batch} utenti nel batch"
                    )

                scores[:, 0] = -float("inf")                  # item di padding
                if history_index is not None:
                    scores[history_index] = -float("inf")     # item già visti in training

                # Utenti del batch da tenere
                if valid_uids is None:
                    keep = torch.ones(n_batch, dtype=torch.bool, device=device)
                else:
                    keep = torch.tensor(
                        [u in valid_uids for u in uids.tolist()],
                        dtype=torch.bool, device=device,
                    )
                if not keep.any():
                    continue

                top_idx = torch.topk(scores, k=max_k, dim=-1).indices   # [B, max_k]

                # Matrice dei positivi: pos[u, i] = True se i è un item di test di u
                pos = torch.zeros((n_batch, n_items), dtype=torch.bool, device=device)
                pos[positive_u.to(device), positive_i.to(device)] = True
                hits  = pos.gather(1, top_idx)                   # [B, max_k]
                n_pos = pos.sum(dim=1)

                keep = keep & (n_pos > 0)
                if not keep.any():
                    continue

                topk_lists.extend(top_idx[keep, :topk].cpu().tolist())
                hits_all.append(hits[keep].cpu())
                npos_all.append(n_pos[keep].cpu())

        if not topk_lists:
            return {"error": f"Nessun utente valutato per {subset_label}"}

        # ── Accuratezza ───────────────────────────────────────────────────────
        hits_t = torch.cat(hits_all, dim=0)
        npos_t = torch.cat(npos_all, dim=0)
        acc = _topk_accuracy(hits_t, npos_t, ks)
        log.info(f"  [{subset_label}] valutati {hits_t.shape[0]} utenti  "
                 f"ndcg@20={acc.get('ndcg@20', float('nan')):.4f}  "
                 f"recall@20={acc.get('recall@20', float('nan')):.4f}")

        # ── Beyond-accuracy (popolarità dal solo training) ────────────────────
        train_items  = train_ds.inter_feat[iid_field]
        pop_counter  = Counter(train_items.tolist())
        n_train      = len(train_ds.inter_feat)
        catalog_size = n_items - 1                     # escluso l'item di padding

        ba = compute_beyond_accuracy(
            topk_lists           = topk_lists,
            item_embeddings      = get_item_emb(model),
            item_popularity      = pop_counter,
            n_train_interactions = n_train,
            catalog_size         = catalog_size,
        )
        log.info(f"  [{subset_label}] coverage={ba['coverage']:.4f}  "
                 f"novelty={ba['novelty']:.4f}  entropy={ba['entropy']:.4f}  "
                 f"gini={ba['gini']:.4f}")

        result = {"n_users": int(hits_t.shape[0])}
        result.update(acc)
        result.update(ba)
        return result

    except Exception as exc:
        log.exception(f"  ✗ Eval [{subset_label}] fallita: {exc}")
        return {"error": str(exc)}
    
def eval_user_subset_old(
    checkpoint_path: Path,
    model_name: str,
    tag: str,
    staging_root: Path,
    cfg_files: list,
    user_set: set[str],
    subset_label: str,
    base_parameter_dict: dict,
    topk: int = 10,
) -> dict:
    if not user_set:
        return {"error": f"user_set vuoto per {subset_label}"}

    # Stessa configurazione del training: stesso split, stessi campi caricati
    parameter_dict = dict(base_parameter_dict)
    parameter_dict["show_progress"] = False

    # parameter_dict = {
    #     "data_path":   str(staging_root),
    #     "dataset":     tag,
    #     "save_dataset": False,
    #     "show_progress": True,
    #     # BPR: neg sampling necessario anche in eval
    #     "train_neg_sample_args": {
    #         "distribution": "uniform",
    #         "sample_num": 3,
    #         "dynamic": False,
    #     },
    #     "eval_args": {
    #         "group_by": "user",
    #         "order":    "TO",
    #         "mode":     "full",
    #         "split":    {"LS": "valid_and_test"},
    #     },
    # }

    try:
        cfg = Config(
            model=model_name,
            dataset=tag,
            config_file_list=cfg_files,
            config_dict=parameter_dict,
        )
        init_seed(cfg["seed"], cfg["reproducibility"])

        # dataset = create_dataset(cfg)
        # _, _, test_data = data_preparation(cfg, dataset)
        dataset = create_dataset(cfg)
        train_data, _, test_data = data_preparation(cfg, dataset)

        uid_field = dataset.uid_field
        token2id  = {
            dataset.id2token(uid_field, i): i
            for i in range(dataset.user_num)
        }
        valid_uids = {token2id[u] for u in user_set if u in token2id}

        if not valid_uids:
            return {"error": f"Nessun utente di {subset_label} trovato nel dataset"}

        # ── Costruisci mappa uid → test item dal dataset (fix principale) ──────
        # Il batch full-sort contiene solo user_id; il ground-truth va letto
        # direttamente dalle interazioni del dataset.
        uid_to_pos = _build_uid_to_test_item(dataset)
        n_mapped   = sum(1 for u in valid_uids if u in uid_to_pos)
        log.info(
            f"  [{subset_label}] ground-truth mappati: "
            f"{n_mapped}/{len(valid_uids)} utenti del subset"
        )

        checkpoint  = torch.load(checkpoint_path, map_location=cfg["device"])
        cfg_ckpt    = checkpoint["config"]
        model_class = get_model(cfg_ckpt["model"])
        #model       = model_class(cfg_ckpt, dataset).to(cfg["device"])
        model = model_class(cfg_ckpt, train_data.dataset).to(cfg["device"])
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        topk_lists      = []
        all_scores_list = []
        all_pos_items   = []

        with torch.no_grad():
            for batch_tuple in test_data:
                batch = batch_tuple[0] if isinstance(batch_tuple, (tuple, list)) \
                        else batch_tuple
                batch = batch.to(cfg["device"])

                uids = batch[uid_field].cpu().tolist()
                mask = torch.tensor(
                    [u in valid_uids for u in uids], dtype=torch.bool
                )
                if not mask.any():
                    continue

                scores = model.full_sort_predict(batch)

                if scores.dim() == 1:
                    scores = scores.unsqueeze(0)

                # Se batch size e scores non coincidono, skip
                n_batch = len(uids)
                if scores.shape[0] != n_batch:
                    log.warning(
                        f"  scores.shape={scores.shape} != batch_size={n_batch}, skip batch"
                    )
                    continue

                scores[:, 0] = -float("inf")

                topk_ids          = torch.topk(scores, k=topk, dim=-1).indices
                topk_ids_filtered = topk_ids[mask]
                topk_lists.extend(topk_ids_filtered.cpu().tolist())

                # Recupera il positive item dalla mappa pre-costruita
                uids_masked    = [uids[i] for i, m in enumerate(mask.tolist()) if m]
                pos_items_vals = [uid_to_pos.get(u) for u in uids_masked]

                if all(p is not None for p in pos_items_vals):
                    pos_t = torch.tensor(pos_items_vals, dtype=torch.long)
                    all_scores_list.append(scores[mask].cpu())
                    all_pos_items.append(pos_t)
                else:
                    n_missing = sum(1 for p in pos_items_vals if p is None)
                    log.warning(
                        f"  [{subset_label}] {n_missing}/{len(uids_masked)} utenti "
                        f"senza ground-truth in questo batch — accuracy skip per il batch"
                    )

        if not topk_lists:
            return {"error": f"Nessun batch per {subset_label}"}

        acc_result = {}
        if all_scores_list and all_pos_items:
            all_scores_t    = torch.cat(all_scores_list, dim=0)
            all_pos_items_t = torch.cat(all_pos_items, dim=0)
            acc_result = _compute_accuracy_metrics(all_scores_t, all_pos_items_t, ks=[50, 10, 20])
            log.info(
                f"  [{subset_label}] accuracy calcolata su "
                f"{all_scores_t.shape[0]} utenti"
            )
        else:
            log.warning(
                f"  [{subset_label}] accuracy metrics non calcolate: "
                "nessun batch con ground-truth completo"
            )

        iid_tensor    = dataset.inter_feat[dataset.iid_field]
        pop_counter   = Counter(iid_tensor.tolist())
        n_total_inter = len(dataset.inter_feat)
        catalog_size  = dataset.item_num
        item_emb      = get_item_emb(model)

        ba = compute_beyond_accuracy(
            topk_lists           = topk_lists,
            item_embeddings      = item_emb,
            item_popularity      = pop_counter,
            n_train_interactions = n_total_inter,
            catalog_size         = catalog_size,
        )
        log.info(
            f"    [{subset_label}] coverage={ba['coverage']:.4f}  "
            f"novelty={ba['novelty']:.4f}  gini={ba['gini']:.4f}"
        )
        combined = dict(acc_result)
        combined.update(ba)
        return combined

    except Exception as exc:
        log.error(f"  ✗ Eval [{subset_label}] fallita: {exc}")
        return {"error": str(exc)}


# ==============================================================================
#  SINGLE RUN
# ==============================================================================
def run_single(
    model: str,
    dataset_name: str,
    dataset_dir: Path,
    noise_type: str,
    level: str,
    train_filename: str,
    train_subdir: str,
    staging_root: Path,
    topk: int = 10,
    eval_only: bool = False,

) -> dict:
    log.info("=" * 72)
    log.info(f"  Model={model}  Dataset={dataset_name}  "
             f"Noise={noise_type}  Level={level}")
    log.info("=" * 72)

    rating_task = is_rating_model(model)

    # ── 1. Staging ────────────────────────────────────────────────────────────
    # GCMC ha bisogno del campo rating nel file .inter.
    # LightGCN/NGCF usano solo user_id/item_id/timestamp.
    try:
        _, tag = stage_dataset(
            dataset_name, dataset_dir,
            noise_type, level,
            train_filename, train_subdir,
            staging_root,
            include_rating=rating_task,
        )
    except (FileNotFoundError, ValueError) as e:
        log.error(f"  ✗ {e}")
        return _err(model, dataset_name, noise_type, level, str(e), 0)

    run_saved_dir = SAVED_DIR / tag / model
    run_saved_dir.mkdir(parents=True, exist_ok=True)

    dataset_cfg = str(CONFIGS_DIR / "datasets" / f"{dataset_name}.yaml")
    model_cfg   = str(CONFIGS_DIR / "models"   / f"{model}.yaml")
    cfg_files   = [f for f in [dataset_cfg, model_cfg] if Path(f).exists()]

    # ── parameter_dict: biforcazione GCMC vs LightGCN/NGCF ───────────────────
    if rating_task:
        # GCMC: regressione, niente neg sampling, mode=labeled, valid_metric=RMSE
        parameter_dict = {
            "data_path":      str(staging_root),
            "dataset":        tag,
            "checkpoint_dir": str(run_saved_dir),
            "save_dataset":   False,
            "show_progress":  True,
            "load_col":       {"inter": ["user_id", "item_id", "rating", "timestamp"]},
            "train_neg_sample_args": None,
            "eval_args": {
                "group_by": "user",
                "order": "TO",
                "mode": "full",
                "split": {"LS": "valid_and_test"},
            },
            "metrics":      ["RMSE", "MAE"],
            "valid_metric": "RMSE",     # minimizzare (RecBole gestisce lower-is-better)
        }
    else:
        # LightGCN / NGCF: ranking, BPR, mode=full
        parameter_dict = {
            "data_path":      str(staging_root),
            "dataset":        tag,
            "checkpoint_dir": str(run_saved_dir),
            "save_dataset":   False,
            "show_progress":  True,
            "load_col":       {"inter": ["user_id", "item_id", "timestamp"]},


            "train_neg_sample_args": {
                "distribution": "uniform",
                "sample_num":   3,
                "dynamic":      False,
            },
            "benchmark_filename": ["train", "valid", "test"],
            "eval_args": {
                "group_by": "user",
                "order": "TO",
                "mode": "full",
            },
        }


    parameter_dict.update(get_model_hparams(dataset_name, model))

    t0 = time.time()
    try:
        cfg = Config(
            model=model,
            dataset=tag,
            config_file_list=cfg_files,
            config_dict=parameter_dict,
        )
        init_seed(cfg["seed"], cfg["reproducibility"])

        dataset = create_dataset(cfg)
        train_data, valid_data, test_data = data_preparation(cfg, dataset)

        model_class   = get_model(cfg["model"])
        trained_model = model_class(cfg, train_data.dataset).to(cfg["device"])
        trainer       = get_trainer(cfg["MODEL_TYPE"], cfg["model"])(cfg, trained_model)

        if eval_only:
            best_ckpt = find_checkpoint(run_saved_dir)
            if best_ckpt is None:
                return _err(model, dataset_name, noise_type, level,
                            f"nessun checkpoint in {run_saved_dir}", 0)
            log.info(f"  → Eval-only: carico {best_ckpt.name}")
            ckpt = torch.load(best_ckpt, map_location=cfg["device"])
            best_valid_score = ckpt.get("best_valid_score")
            test_result = trainer.evaluate(
                test_data, load_best_model=True,
                model_file=str(best_ckpt), show_progress=False,
            )
        else:
            best_valid_score, _ = trainer.fit(
                train_data, valid_data,
                saved=True, show_progress=cfg["show_progress"],
            )
            best_ckpt = Path(trainer.saved_model_file)
            test_result = trainer.evaluate(
                test_data, load_best_model=True, show_progress=False,
            )

        elapsed = time.time() - t0
        log.info(f"  ✓ {'Load' if eval_only else 'Training'} completato in {elapsed:.1f}s  "
                 f"best_valid={best_valid_score}")

        result = {
            "best_valid_score": best_valid_score,
            "test_result":      test_result,
        }

    except Exception as exc:
        elapsed = time.time() - t0
        log.error(f"  ✗ {'Load' if eval_only else 'Training'} FALLITO dopo {elapsed:.1f}s — {exc}")
        return _err(model, dataset_name, noise_type, level, str(exc), elapsed)
    # t0 = time.time()
    # try:
    #     cfg = Config(
    #         model=model,
    #         dataset=tag,
    #         config_file_list=cfg_files,
    #         config_dict=parameter_dict,
    #     )
    #     init_seed(cfg["seed"], cfg["reproducibility"])
    #     log.info(f"  load_col: {cfg['load_col']}")
    #
    #     dataset = create_dataset(cfg)
    #     train_data, valid_data, test_data = data_preparation(cfg, dataset)
    #
    #     model_class   = get_model(cfg["model"])
    #     trained_model = model_class(cfg, dataset).to(cfg["device"])
    #     trainer       = get_trainer(cfg["MODEL_TYPE"], cfg["model"])(cfg, trained_model)
    #
    #     best_valid_score, best_valid_result = trainer.fit(
    #         train_data, valid_data,
    #         saved=True, show_progress=cfg["show_progress"],
    #     )
    #     test_result = trainer.evaluate(
    #         test_data, load_best_model=True, show_progress=False
    #     )
    #
    #     elapsed = time.time() - t0
    #     log.info(f"  ✓ Training completato in {elapsed:.1f}s  "
    #              f"best_valid={best_valid_score:.4f}")
    #
    #     result = {
    #         "best_valid_score": best_valid_score,
    #         "test_result":      test_result,
    #     }
    #
    # except Exception as exc:
    #     elapsed = time.time() - t0
    #     log.error(f"  ✗ Training FALLITO dopo {elapsed:.1f}s — {exc}")
    #     return _err(model, dataset_name, noise_type, level, str(exc), elapsed)
    #
    # checkpoints = sorted(run_saved_dir.glob("*.pth"))
    # best_ckpt   = checkpoints[-1] if checkpoints else None
    # if best_ckpt is None:
    #     return _err(model, dataset_name, noise_type, level, "no checkpoint", elapsed)

    active_users = load_user_set(dataset_dir / f"{dataset_name}.rec_test_active.csv")
    lt_users     = load_user_set(dataset_dir / f"{dataset_name}.rec_test_lt.csv")

    # ── 3a. GCMC: eval subset con RMSE/MAE/AUC, niente beyond-accuracy ────────
    if rating_task:
        log.info(f"  → Eval active users GCMC ({len(active_users)} utenti)")
        result_active = eval_gcmc_subset(
            checkpoint_path = best_ckpt,
            model_name      = model,
            tag             = tag,
            staging_root    = staging_root,
            cfg_files       = cfg_files,
            user_set        = active_users,
            subset_label    = "active",
        )

        log.info(f"  → Eval long-tail users GCMC ({len(lt_users)} utenti)")
        result_lt = eval_gcmc_subset(
            checkpoint_path = best_ckpt,
            model_name      = model,
            tag             = tag,
            staging_root    = staging_root,
            cfg_files       = cfg_files,
            user_set        = lt_users,
            subset_label    = "lt",
        )

        # AUC sul test set standard (RecBole non lo calcola per regression)
        log.info("  → AUC sul test set standard")
        test_result_standard = dict(result.get("test_result", {}))
        try:
            cfg2 = Config(
                model=model, dataset=tag,
                config_file_list=cfg_files, config_dict=parameter_dict,
            )
            init_seed(cfg2["seed"], cfg2["reproducibility"])
            dataset2 = create_dataset(cfg2)
            _, _, test_data2 = data_preparation(cfg2, dataset2)

            ckpt2    = torch.load(best_ckpt, map_location=cfg2["device"])
            m2_class = get_model(ckpt2["config"]["model"])
            m2       = m2_class(ckpt2["config"], dataset2).to(cfg2["device"])
            m2.load_state_dict(ckpt2["state_dict"])
            m2.eval()

            all_preds, all_truths = [], []
            rating_field = dataset2.label_field
            with torch.no_grad():
                for batch_tuple in test_data2:
                    batch = batch_tuple[0] if isinstance(batch_tuple, (tuple, list)) \
                            else batch_tuple
                    batch = batch.to(cfg2["device"])
                    preds  = m2.predict(batch).cpu().tolist()
                    truths = batch[rating_field].cpu().tolist()
                    all_preds.extend(preds)
                    all_truths.extend(truths)

            auc_std = compute_auc_from_ratings(
                np.array(all_truths, dtype=float),
                np.array(all_preds,  dtype=float),
            )
            if auc_std is not None:
                test_result_standard["auc"] = round(auc_std, 4)
                log.info(f"  → AUC standard = {auc_std:.4f}")

        except Exception as exc:
            log.error(f"  ✗ AUC standard fallita: {exc}")

        return {
            "model":              model,
            "dataset":            dataset_name,
            "noise_type":         noise_type,
            "level":              level,
            "status":             "ok",
            "elapsed_s":          round(elapsed, 1),
            "best_valid_score":   result.get("best_valid_score"),
            "checkpoint_dir":     str(run_saved_dir),
            "test_result":        test_result_standard,
            "test_result_active": result_active,
            "test_result_lt":     result_lt,
        }

    # ── 3b. LightGCN/NGCF: eval subset con Recall/NDCG + beyond-accuracy ─────
    log.info(f"  → Eval active users ({len(active_users)} utenti)")
    # result_active = eval_user_subset(
    #     checkpoint_path = best_ckpt,
    #     model_name      = model,
    #     tag             = tag,
    #     staging_root    = staging_root,
    #     cfg_files       = cfg_files,
    #     user_set        = active_users,
    #     subset_label    = "active",
    #     topk            = topk,
    # )
    result_active = eval_user_subset(
        checkpoint_path     = best_ckpt,
        model_name          = model,
        tag                 = tag,
        staging_root        = staging_root,
        cfg_files           = cfg_files,
        user_set            = active_users,
        subset_label        = "active",
        base_parameter_dict = parameter_dict,
        topk                = topk,
    )

    log.info(f"  → Eval long-tail users ({len(lt_users)} utenti)")
    # result_lt = eval_user_subset(
    #     checkpoint_path = best_ckpt,
    #     model_name      = model,
    #     tag             = tag,
    #     staging_root    = staging_root,
    #     cfg_files       = cfg_files,
    #     user_set        = lt_users,
    #     subset_label    = "lt",
    #     topk            = topk,
    # )
    result_lt = eval_user_subset(
        checkpoint_path     = best_ckpt,
        model_name          = model,
        tag                 = tag,
        staging_root        = staging_root,
        cfg_files           = cfg_files,
        user_set            = lt_users,
        subset_label        = "lt",
        base_parameter_dict = parameter_dict,
        topk                = topk,
    )
    # ── 4. Beyond-accuracy sul test set standard (solo LightGCN/NGCF) ─────────
    # ── 4. Metriche su tutti gli utenti ───────────────────────────────────────
    log.info("  → Eval su tutti gli utenti")
    result_all = eval_user_subset(
        checkpoint_path     = best_ckpt,
        model_name          = model,
        tag                 = tag,
        staging_root        = staging_root,
        cfg_files           = cfg_files,
        user_set            = None,
        subset_label        = "all",
        base_parameter_dict = parameter_dict,
        topk                = topk,
    )
    test_result_standard = dict(result.get("test_result", {}))
    test_result_standard["manual_all"] = result_all
    # log.info("  → Beyond-accuracy sul test set standard")
    # try:
    #     cfg = Config(
    #         model=model,
    #         dataset=tag,
    #         config_file_list=cfg_files,
    #         config_dict=parameter_dict,
    #     )
    #     init_seed(cfg["seed"], cfg["reproducibility"])
    #     dataset = create_dataset(cfg)
    #     _, _, test_data = data_preparation(cfg, dataset)
    #
    #     checkpoint    = torch.load(best_ckpt, map_location=cfg["device"])
    #     cfg_ckpt      = checkpoint["config"]
    #     model_class   = get_model(cfg_ckpt["model"])
    #     trained_model = model_class(cfg_ckpt, dataset).to(cfg["device"])
    #     trained_model.load_state_dict(checkpoint["state_dict"])
    #     trained_model.eval()
    #
    #     topk_lists = []
    #     with torch.no_grad():
    #         for batch_tuple in test_data:
    #             batch = batch_tuple[0] if isinstance(batch_tuple, (tuple, list)) \
    #                     else batch_tuple
    #             batch = batch.to(cfg["device"])
    #             scores = trained_model.full_sort_predict(batch)
    #             if scores.dim() == 1:
    #                 scores = scores.unsqueeze(0)
    #             scores[:, 0] = -float("inf")
    #             topk_ids = torch.topk(scores, k=topk, dim=-1).indices
    #             topk_lists.extend(topk_ids.cpu().tolist())
    #
    #     iid_tensor    = dataset.inter_feat[dataset.iid_field]
    #     pop_counter   = Counter(iid_tensor.tolist())
    #     n_total_inter = len(dataset.inter_feat)
    #     catalog_size  = dataset.item_num
    #     item_emb      = get_item_emb(trained_model)
    #
    #     ba_standard = compute_beyond_accuracy(
    #         topk_lists           = topk_lists,
    #         item_embeddings      = item_emb,
    #         item_popularity      = pop_counter,
    #         n_train_interactions = n_total_inter,
    #         catalog_size         = catalog_size,
    #     )
    #
    #     test_result_standard = dict(result.get("test_result", {}))
    #     test_result_standard.update(ba_standard)

    # except Exception as exc:
    #     log.error(f"  ✗ Beyond-accuracy standard fallita: {exc}")
    #     test_result_standard = result.get("test_result", {})

    return {
        "model":              model,
        "dataset":            dataset_name,
        "noise_type":         noise_type,
        "level":              level,
        "status":             "ok",
        "elapsed_s":          round(elapsed, 1),
        "best_valid_score":   result.get("best_valid_score"),
        "checkpoint_dir":     str(run_saved_dir),
        "test_result":        test_result_standard,
        "test_result_active": result_active,
        "test_result_lt":     result_lt,
    }


def _err(model, dataset, noise, level, msg, elapsed):
    return {
        "model": model, "dataset": dataset,
        "noise_type": noise, "level": level,
        "status": "error", "error": msg,
        "elapsed_s": round(elapsed, 1),
    }


def save_result(m: dict):
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_FILE.open("a") as f:
        f.write(json.dumps(m) + "\n")

def find_checkpoint(run_saved_dir: Path) -> Path | None:
    """Restituisce il checkpoint più recente nella cartella del run."""
    ckpts = list(run_saved_dir.glob("*.pth"))
    return max(ckpts, key=lambda p: p.stat().st_mtime) if ckpts else None

def checkpoint_exists(dataset_name, noise_type, level, model):
    ckpt = SAVED_DIR / f"{dataset_name}__{noise_type}__{level}" / model
    return ckpt.exists() and any(ckpt.glob("*.pth"))

# def checkpoint_exists(dataset_name, noise_type, level, model):
#     tag  = f"{dataset_name}"
#     ckpt = SAVED_DIR / tag / model
#     return ckpt.exists() and any(ckpt.glob("*.pth"))


# ==============================================================================
#  CLI
# ==============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="RecBole CF (LightGCN, NGCF, GCMC) — RS split",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--datasets",      nargs="+", default=list(DATASETS.keys()))
    p.add_argument("--models",        nargs="+", default=ALL_MODELS)
    p.add_argument("--noise",         nargs="+", default=ALL_NOISE_TYPES)
    p.add_argument("--levels",        nargs="+", default=None)
    p.add_argument("--topk",          type=int,  default=50)
    p.add_argument("--staging-dir",   default=str(STAGING_DIR))
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--dry-run",       action="store_true")
    p.add_argument("--eval_only", action="store_true",
                   help="Non allena: carica il checkpoint esistente e calcola le metriche")
    return p.parse_args()


def main():
    args         = parse_args()
    staging_root = Path(args.staging_dir)
    staging_root.mkdir(parents=True, exist_ok=True)
    #args.eval_only = True
    runs = []
    for dataset_name in args.datasets:
        if dataset_name not in DATASETS:
            log.warning(f"Dataset sconosciuto '{dataset_name}', saltato.")
            continue
        dataset_dir = DATASETS[dataset_name]
        noise_reg   = build_noise_registry(dataset_name)


        for noise_type in args.noise:
            if noise_type not in noise_reg:
                continue
            noise_cfg  = noise_reg[noise_type]
            level_map  = noise_cfg["levels"]
            levels_run = args.levels if args.levels else list(level_map.keys())

            for level in levels_run:
                if level not in level_map:
                    continue
                for model in args.models:
                    runs.append({
                        "dataset_name": dataset_name,
                        "dataset_dir":  dataset_dir,
                        "model":        model,
                        "noise_type":   noise_type,
                        "level":        level,
                        "train_file":   level_map[level],
                        "train_subdir": noise_cfg["subdir"],
                    })

    total = len(runs)
    log.info(f"Esperimenti pianificati: {total}")

    if args.dry_run:
        print(f"\n{'Dataset':<20} {'Model':<10} {'Noise':<22} {'Level'}")
        print("-" * 60)
        for r in runs:
            print(f"{r['dataset_name']:<20} {r['model']:<10} "
                  f"{r['noise_type']:<22} {r['level']}")
        print(f"\nTotale: {total}\n")
        return

    done, errors = 0, 0
    for r in runs:
        ds, model, noise, level = (
            r["dataset_name"], r["model"], r["noise_type"], r["level"]
        )
        if args.skip_existing and checkpoint_exists(ds, noise, level, model):
            log.info(f"  → Skip: {ds}__{noise}__{level} / {model}")
            done += 1
            continue

        metrics = run_single(
            model          = model,
            dataset_name   = ds,
            dataset_dir    = r["dataset_dir"],
            noise_type     = noise,
            level          = level,
            train_filename = r["train_file"],
            train_subdir   = r["train_subdir"],
            staging_root   = staging_root,
            topk           = args.topk,
            eval_only      = args.eval_only,
        )
        save_result(metrics)
        done += 1
        if metrics["status"] != "ok":
            errors += 1
        log.info(f"  Progresso: {done}/{total}  (errori: {errors})")

    log.info("=" * 72)
    log.info(f"Completato.  {done - errors}/{total} riusciti.")
    log.info(f"Risultati  → {RESULTS_FILE}")
    log.info(f"Checkpoint → {SAVED_DIR}/")
    log.info("=" * 72)


if __name__ == "__main__":
    main()