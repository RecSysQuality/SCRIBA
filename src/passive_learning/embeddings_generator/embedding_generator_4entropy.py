import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch_geometric.data import Data
import os
import sys
sys.stdout.reconfigure(line_buffering=True)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import networkx as nx
import pandas as pd
import numpy as np
from scipy.stats import entropy


import pandas as pd
import numpy as np
import networkx as nx
from scipy.stats import entropy

def compute_receiver_features(pairs_df: pd.DataFrame,
                              reviews_df: pd.DataFrame,
                              defects_df: pd.DataFrame | None = None,
                              G_full: nx.Graph | None = None) -> pd.DataFrame:
    reviews = reviews_df.copy()
    reviews["user_id"] = reviews["user_id"].astype(str)
    reviews["item_id"] = reviews["item_id"].astype(str)
    reviews["timestamp"] = pd.to_datetime(reviews["timestamp"], errors="coerce")

    pairs = pairs_df.copy()
    pairs["user_id"] = pairs["user_id"].astype(str)

    if defects_df is not None:
        defects = defects_df.copy()
        defects["user_id"] = defects["user_id"].astype(str)
        defects["item_id"] = defects["item_id"].astype(str)
    else:
        defects = None

    total_users = reviews["user_id"].nunique()
    total_items = reviews["item_id"].nunique()

    item_pop = reviews.groupby("item_id").size()
    item_pop_share = item_pop / len(reviews)
    pop_q75 = item_pop.quantile(0.75)
    pop_q25 = item_pop.quantile(0.25)

    user_stats = reviews.groupby("user_id").agg(
        user_n_reviews=("item_id", "size"),
        user_n_items=("item_id", "nunique"),
        user_rating_mean=("rating", "mean"),
        user_rating_std=("rating", lambda x: x.std(ddof=0)),
        user_first_ts=("timestamp", "min"),
        user_last_ts=("timestamp", "max"),
    ).reset_index()
    user_stats["user_rating_std"] = user_stats["user_rating_std"].fillna(0)
    user_stats["user_span_days"] = (user_stats["user_last_ts"] - user_stats["user_first_ts"]).dt.total_seconds() / 86400.0
    user_stats["user_span_days"] = user_stats["user_span_days"].fillna(0)

    user_items = reviews.groupby("user_id")["item_id"].agg(list).to_dict()
    user_item_sets = reviews.groupby("user_id")["item_id"].agg(set).to_dict()

    item_users = reviews.groupby("item_id")["user_id"].agg(set).to_dict()

    if G_full is None:
        G_full = nx.from_pandas_edgelist(reviews[["user_id", "item_id"]], "user_id", "item_id")
        if len(G_full) > 0:
            try:
                pr_full = nx.pagerank(G_full)
            except Exception:
                pr_full = {n: 0.0 for n in G_full.nodes()}
        else:
            pr_full = {}
    else:
        try:
            pr_full = nx.pagerank(G_full)
        except Exception:
            pr_full = {n: 0.0 for n in G_full.nodes()}

    rows = []
    for _, row in pairs.iterrows():
        u = row["user_id"]
        target_items = []
        if "item_id" in pairs.columns and pd.notna(row.get("item_id", np.nan)):
            target_items = [str(row["item_id"])]
        elif "target_item_id" in pairs.columns and pd.notna(row.get("target_item_id", np.nan)):
            target_items = [str(row["target_item_id"])]

        us = user_stats[user_stats["user_id"] == u]
        if len(us) == 0:
            user_n_reviews = 0
            user_n_items = 0
            user_rating_mean = 0
            user_rating_std = 0
            user_span_days = 0
        else:
            us = us.iloc[0]
            user_n_reviews = float(us["user_n_reviews"])
            user_n_items = float(us["user_n_items"])
            user_rating_mean = float(us["user_rating_mean"]) if pd.notna(us["user_rating_mean"]) else 0
            user_rating_std = float(us["user_rating_std"]) if pd.notna(us["user_rating_std"]) else 0
            user_span_days = float(us["user_span_days"]) if pd.notna(us["user_span_days"]) else 0

        items = user_item_sets.get(u, set())
        item_list = list(items)

        if len(item_list) > 0:
            freqs = pd.Series(item_list).map(item_pop).fillna(0).values
            p = freqs / freqs.sum() if freqs.sum() > 0 else np.array([])
            user_entropy_profile = entropy(p, base=2) if len(p) > 0 else 0.0
            user_long_tail_ratio = float(np.mean([item_pop.get(i, 0) <= pop_q25 for i in item_list]))
            user_head_ratio = float(np.mean([item_pop.get(i, 0) >= pop_q75 for i in item_list]))
            user_avg_item_pop = float(np.mean([item_pop.get(i, 0) for i in item_list]))
            user_avg_pop_share = float(np.mean([item_pop_share.get(i, 0.0) for i in item_list]))
        else:
            user_entropy_profile = 0.0
            user_long_tail_ratio = 0.0
            user_head_ratio = 0.0
            user_avg_item_pop = 0.0
            user_avg_pop_share = 0.0

        if target_items:
            tgt_set = set(target_items)
            inter = len(items.intersection(tgt_set)) if isinstance(items, set) else len(set(item_list).intersection(tgt_set))
            union = len(set(item_list).union(tgt_set))
            sim_user_target = inter / union if union > 0 else 0.0

            target_pop = [item_pop.get(t, 0) for t in target_items]
            target_avg_pop = float(np.mean(target_pop)) if target_pop else 0.0
            target_avg_share = float(np.mean([item_pop_share.get(t, 0.0) for t in target_items])) if target_items else 0.0
        else:
            sim_user_target = 0.0
            target_avg_pop = 0.0
            target_avg_share = 0.0

        if G_full is not None and len(G_full) > 0:
            if u in G_full:
                neigh = list(G_full.neighbors(u))
                dists = []
                for t in target_items:
                    if t in G_full:
                        try:
                            dists.append(nx.shortest_path_length(G_full, u, t))
                        except Exception:
                            pass
                dist_user_defect = float(np.mean(dists)) if dists else np.inf
            else:
                dist_user_defect = np.inf
        else:
            dist_user_defect = np.nan

        recency_days = 0.0
        if pd.notna(us["user_last_ts"]) and pd.notna(us["user_first_ts"]):
            recency_days = (reviews["timestamp"].max() - us["user_last_ts"]).total_seconds() / 86400.0 if len(reviews) else 0.0

        row_out = {
            "defect_id": row["defect_id"],
            "user_id": u,
            "user_profile_size": user_n_reviews,
            "user_n_items": user_n_items,
            "user_entropy_profile": user_entropy_profile,
            "user_long_tail_ratio": user_long_tail_ratio,
            "user_head_ratio": user_head_ratio,
            "user_avg_item_pop": user_avg_item_pop,
            "user_avg_pop_share": user_avg_pop_share,
            "user_rating_mean": user_rating_mean,
            "user_rating_std": user_rating_std,
            "user_span_days": user_span_days,
            "user_activity_recency_days": recency_days,
            "user_similarity_to_target_set": sim_user_target,
            "target_avg_pop": target_avg_pop,
            "target_avg_share": target_avg_share,
            "distance_to_defect_subgraph": dist_user_defect,
            "already_exposed_to_target_items": len(set(item_list).intersection(set(target_items))) if target_items else 0,
        }
        rows.append(row_out)

    return pd.DataFrame(rows)



import numpy as np
from scipy.stats import skew, kurtosis

def distribution_stats(values):
    values = np.array(list(values))

    if len(values) == 0:
        return {
            "mean": 0, "std": 0, "min": 0, "max": 0,
            "skew": 0, "kurtosis": 0, "gini": 0
        }

    sorted_vals = np.sort(values)
    n = len(sorted_vals)

    # Gini coefficient
    cumvals = np.cumsum(sorted_vals)
    gini = (n + 1 - 2 * np.sum(cumvals) / cumvals[-1]) / n if cumvals[-1] != 0 else 0

    return {
        "mean": np.mean(values),
        "std": np.std(values),
        "min": np.min(values),
        "max": np.max(values),
        #"skew": skew(values),
        #"kurtosis": kurtosis(values),
        "gini": gini
    }



def compute_topology_features(defects_df: pd.DataFrame, reviews_df: pd.DataFrame) -> pd.DataFrame:
    results = {}

    da = json.load(open(f"./output.json",'r'))
    ids = [d['defect_id'] for d in da]
    # 1. Crea il grafo totale (opzionale: potresti volerlo passare già costruito)
    # Per semplicità, qui estraiamo il sottografo per ogni difetto
    for defect_id, group in defects_df.groupby("group_id"):
        print(defect_id)
        if defect_id not in ids:
            continue
        user_ids = group["user_id"].unique()
        item_ids = group["item_id"].unique()
        edges = len(group)
        # Filtra le recensioni rilevanti
        local_reviews = reviews_df[reviews_df["user_id"].isin(user_ids) & reviews_df["item_id"].isin(item_ids)]

        # 2. Costruisci il grafo locale (bipartito)
        G = nx.from_pandas_edgelist(local_reviews, 'user_id', 'item_id')

        # --- CALCOLO FEATURE TOPOLOGICHE ---

        # Clustering Coefficient (Locale)
        clustering = nx.average_clustering(G)

        # Degree Centrality (Media)
        degrees = dict(G.degree())
        avg_degree = sum(degrees.values()) / len(degrees) if len(degrees) > 0 else 0

        # Betweenness Centrality (degli item target)
        # Assumiamo che gli item target siano gli item nel set del difetto
        betw = nx.betweenness_centrality(G)
        target_betw = [betw[n] for n in item_ids if n in betw]
        avg_target_betw = np.mean(target_betw) if target_betw else 0

        # Density
        density = nx.density(G)

        # Average Path Length (locale)
        # Nota: per grafi non connessi, si usa la componente connessa più grande
        if nx.is_connected(G):
            avg_path_length = nx.average_shortest_path_length(G)
        else:
            avg_path_length = np.mean([nx.average_shortest_path_length(c) for c in nx.connected_components(G)])

        # Eigenvector Centrality
        eigen = nx.eigenvector_centrality(G, max_iter=1000)
        avg_eigen = np.mean(list(eigen.values())) if eigen else 0

        n_users = len(user_ids)
        n_items = len(item_ids)
        n_nodes = n_users + n_items
        n_reviews = len(local_reviews)

        # Densità del sottografo (connettività)
        density = n_reviews / (n_users * n_items) if (n_users * n_items) > 0 else 0

        # 2. Entropia Locale (Shannon) - "Il disordine interno al cluster"
        # La usiamo come misura della varietà imposta dal bot
        item_counts = local_reviews["item_id"].value_counts(normalize=True)
        local_entropy = entropy(item_counts.values, base=2)

        # 3. Feature di Centralità semplificata
        avg_degree = n_reviews / n_nodes if n_nodes > 0 else 0

        # --- FEATURE BONUS ---
        # 1. Assortatività
        try:
            assortativity = nx.degree_assortativity_coefficient(G)
        except:
            assortativity = 0

        # 2. Jaccard Similarity (Media tra i bot)
        similarities = []
        bot_list = list(user_ids)
        for i in range(len(bot_list)):
            for j in range(i + 1, len(bot_list)):
                n1, n2 = set(G.neighbors(bot_list[i])), set(G.neighbors(bot_list[j]))
                union_size = len(n1.union(n2))
                similarities.append(len(n1.intersection(n2)) / union_size if union_size > 0 else 0)
        avg_jaccard = np.mean(similarities) if similarities else 0

        try:
            communities = nx.community.greedy_modularity_communities(G)
            modularity = nx.community.modularity(G, communities)
        except:
            modularity = 0

            # Redundancy = 1 - (H / H_max) dove H_max è log2(numero di item unici)
        H = entropy(local_reviews["item_id"].value_counts(normalize=True), base=2)
        H_max = np.log2(len(item_ids)) if len(item_ids) > 1 else 1
        redundancy = 1 - (H / H_max)

        adj_matrix = nx.adjacency_matrix(G).toarray()
        eigenvalues = np.linalg.eigvalsh(adj_matrix)
        energy = np.sum(np.square(eigenvalues)) / len(G) if len(G) > 0 else 0
        core_numbers = nx.core_number(G)
        max_core = max(core_numbers.values()) if core_numbers else 0

        # eigen = nx.eigenvector_centrality(G, max_iter=1000)
        # eig_stats = distribution_stats(eigen.values())
        # betw = nx.betweenness_centrality(G)
        # betw_stats = distribution_stats(betw.values())
        # deg_values = [d for _, d in G.degree()]
        # deg_stats = distribution_stats(deg_values)
        # jaccard_stats = distribution_stats(similarities)



        try:
            alg_connectivity = nx.algebraic_connectivity(G)
        except:
            alg_connectivity = 0
        results[defect_id] = {
            #"clustering_coeff": clustering,
            "avg_degree": avg_degree,
            "avg_target_betweenness": avg_target_betw,
            "density": density,
            "avg_path_length": avg_path_length,
            "avg_eigenvector": avg_eigen,
            "n_users": n_users,
            "n_items": n_items,
            "n_edges":edges,
            "entropy": local_entropy,
            'avg_jaccard': avg_jaccard,
            'assortativity': assortativity,
            'modularity':modularity,'alg_connectivity':alg_connectivity,
            'redundancy':redundancy,'energy':energy,'max_core':max_core,

            # "deg_mean": deg_stats["mean"],
            # "deg_std": deg_stats["std"],
            # #"deg_skew": deg_stats["skew"],
            # #"deg_kurt": deg_stats["kurtosis"],
            # "deg_gini": deg_stats["gini"],
            #
            # "jacc_mean": jaccard_stats["mean"],
            # "jacc_std": jaccard_stats["std"],
            # "jacc_gini": jaccard_stats["gini"],
            #
            # "betw_mean": betw_stats["mean"],
            # "betw_std": betw_stats["std"],
            # "betw_gini": betw_stats["gini"],
            #
            # "eig_mean": eig_stats["mean"],
            # "eig_std": eig_stats["std"],
            # "eig_gini": eig_stats["gini"],
        }

    return pd.DataFrame.from_dict(results, orient='index')


def compute_bonus_features(G, group_users, target_items, G_pre=None):
    # 1. Assortatività (Degree Assortativity)
    # Indica se i nodi simili (grado) si collegano tra loro
    assortativity = nx.degree_assortativity_coefficient(G)

    # 2. Jaccard Similarity (tra bot)
    # Quanto sono simili le "liste dei desideri" dei bot del cluster?
    similarities = []
    bot_list = list(group_users)
    for i in range(len(bot_list)):
        for j in range(i + 1, len(bot_list)):
            n1, n2 = set(G.neighbors(bot_list[i])), set(G.neighbors(bot_list[j]))
            similarities.append(len(n1.intersection(n2)) / len(n1.union(n2)) if len(n1.union(n2)) > 0 else 0)
    avg_jaccard = np.mean(similarities) if similarities else 0

    # 3. PageRank Shift
    # Richiede il grafo pre-attacco. Se non lo hai, non calcolarlo.
    pagerank_shift = 0
    if G_pre:
        pr_pre = nx.pagerank(G_pre)
        pr_post = nx.pagerank(G)
        shifts = [pr_post.get(n, 0) - pr_pre.get(n, 0) for n in target_items if n in pr_post]
        pagerank_shift = np.mean(shifts) if shifts else 0

    return assortativity, avg_jaccard, pagerank_shift



# ═══════════════════════════════════════════════════════
# 1. SAVE EMBEDDINGS  →  .pt
# ═══════════════════════════════════════════════════════

def save_embeddings_pt(npy_path: str, out_path: str = None):
    """
    Converte .npy + .csv  →  un singolo .pt con:
      {
        "matrix": torch.Tensor  (n_nodes, 64)   float32,
        "node_ids": list[str]
      }
    """
    npy_path = Path(npy_path)
    csv_path = npy_path.with_suffix(".csv")
    out_path = Path(out_path) if out_path else npy_path.with_suffix(".pt")

    matrix = np.load(npy_path)  # (n, 64)
    node_ids = pd.read_csv(csv_path, index_col="node_id").index.tolist()

    payload = {
        "features": torch.tensor(matrix, dtype=torch.float32),
        "node_ids": node_ids,
    }
    torch.save(payload, out_path)
    print(f"  Salvato: {out_path}  shape={payload['features'].shape}")


# ═══════════════════════════════════════════════════════
# 1b. LOAD EMBEDDINGS  ←  .pt
# ═══════════════════════════════════════════════════════
def load_embeddings(pt_path: str) -> dict:
    payload = torch.load(pt_path, weights_only=True)

    if "features" in payload:
        # Formato nuovo (prodotto dai wrapper generate_*_embeddings)
        matrix   = payload["features"]
        node_ids = payload["node_ids"]

    elif "node_ids" in payload and payload["node_ids"]:
        # Formato con node_ids espliciti ma senza "features"
        matrix   = payload["node_embeddings"]
        node_ids = payload["node_ids"]
    else:
        # FIX 1 (robustezza): supporta sia top-level che annidato in "meta"
        matrix = payload["node_embeddings"]
        meta   = payload.get("meta", payload)          # fallback a top-level
        u2i    = meta.get("user2idx", payload.get("user2idx", {}))
        i2i    = meta.get("item2idx", payload.get("item2idx", {}))

        node_ids = [""] * matrix.shape[0]
        for uid, idx in u2i.items():
            node_ids[idx] = f"user_{uid}"
        for iid, idx in i2i.items():
            node_ids[idx] = f"item_{iid}"

    return {nid: matrix[i] for i, nid in enumerate(node_ids)}
# ═══════════════════════════════════════════════════════
# 2. AGGREGAZIONE EMBEDDING PER DIFETTO  (mode=all)
# ═══════════════════════════════════════════════════════
import torch
import pandas as pd



import numpy as np
import pandas as pd
from itertools import combinations





# ═══════════════════════════════════════════════════════
# 4. CONCAT FINALE
# ═══════════════════════════════════════════════════════

def build_feature_matrix(df_emb: pd.DataFrame,
                         df_hand: pd.DataFrame) -> pd.DataFrame:
    """
    Join su defect_id — solo i difetti presenti in entrambi.
    Embedding (192) + Handcraft (17) = 209 feature totali.
    """
    if df_emb.empty:
        df = df_hand
    else:
        df = df_emb.join(df_hand, how="inner")
    print(f"  Feature matrix: {df.shape}  "
          f"(emb={df_emb.shape[1]}, hand={df_hand.shape[1]})")
    return df
def load_graph_from_csv(csv_path):
    """
    Legge il CSV e costruisce un grafo bipartito utente-item come PyG Data.

    Nodi:  utenti (indici 0..U-1) + item (indici U..U+I-1)
    Archi: ogni riga del CSV → un arco bidirezionale
    Edge attr: rating normalizzato [0,1]

    Ritorna:
        graph  — PyG Data, con graph.meta già allegato
        meta   — dict con user2idx, item2idx, num_users, num_items
    """
    import pandas as pd

    df = pd.read_csv(csv_path, index_col=0)
    df.columns = df.columns.str.strip()

    # ── Mappa user_id / item_id a indici interi contigui ─────────────────────
    unique_users = df["user_id"].unique()
    unique_items = df["item_id"].unique()

    user2idx = {u: i              for i, u in enumerate(unique_users)}
    item2idx = {it: i + len(unique_users) for i, it in enumerate(unique_items)}

    num_users = len(unique_users)
    num_items = len(unique_items)
    num_nodes = num_users + num_items

    print(f"  Utenti: {num_users} | Item: {num_items} | Nodi totali: {num_nodes}")

    # ── Costruisci edge_index ─────────────────────────────────────────────────
    src = df["user_id"].map(user2idx).values
    dst = df["item_id"].map(item2idx).values

    edge_src = torch.tensor(list(src) + list(dst), dtype=torch.long)
    edge_dst = torch.tensor(list(dst) + list(src), dtype=torch.long)
    edge_index = torch.stack([edge_src, edge_dst], dim=0)  # [2, 2*E]

    # ── Edge attributes: rating normalizzato ──────────────────────────────────
    ratings      = torch.tensor(df["rating"].values, dtype=torch.float)
    ratings_norm = ratings / 5.0
    edge_attr    = torch.cat([ratings_norm, ratings_norm])  # [2*E]

    # ── Grafo PyG ─────────────────────────────────────────────────────────────
    # FIX 5: node_ids obbligatorio per export_embeddings
    node_ids = (
        [f"user_{u}"  for u in unique_users] +
        [f"item_{it}" for it in unique_items]
    )

    graph = Data(
        edge_index=edge_index,
        edge_attr=edge_attr.unsqueeze(1),  # [2*E, 1]
        num_nodes=num_nodes,
    )

    meta = {
        "user2idx": user2idx,
        "item2idx": item2idx,
        "num_users": num_users,
        "num_items": num_items,
    }

    graph.meta     = meta
    graph.node_ids = node_ids   # ← aggiunto

    return graph, meta

def aggregate_embeddings(emb, defects_df):
    dim = next(iter(emb.values())).shape[0]

    emb_keys = set(emb.keys())

    grouped = defects_df.groupby("group_id")

    results = {}

    for gid, group in grouped:
        print(gid)
        users = ["user:" + str(x) for x in set(group["user_id"])]
        items = ["item:" + str(x) for x in set(group["item_id"])]

        node_ids = users + items

        # filter in one step
        vecs = [emb[n] for n in node_ids if n in emb_keys]

        if not vecs:
            continue

        mat = torch.stack(vecs)
        results[gid] = mat.mean(0).cpu().numpy()

    return pd.DataFrame.from_dict(
        results,
        orient="index",
        columns=[f"emb_mean_{i}" for i in range(dim)]
    )

def aggregate_embeddings0(emb: dict, defects_df: pd.DataFrame) -> pd.DataFrame:
    dim = next(iter(emb.values())).shape[0]
    rows = {}

    for defect_id, group in defects_df.groupby("group_id"):
        print(defect_id)
        # Costruisce node_ids dai prefissi user_ e item_
        user_node_ids = ["user:" + str(u) for u in group["user_id"].unique()]
        item_node_ids = ["item:" + str(i) for i in group["item_id"].unique()]
        node_ids = user_node_ids + item_node_ids

        vecs = [emb[n] for n in node_ids if n in list(emb.keys())]

        if not vecs:
            print(f"  ⚠ {defect_id}: nessun nodo trovato, skip")
            continue

        mat = torch.stack(vecs)  # (k, 64)

        # rows[defect_id] = torch.cat([
        #     mat.mean(dim=0),
        #     mat.max(dim=0).values,
        #     mat.std(dim=0),
        # ]).numpy()  # → (192,)
        rows[defect_id] = torch.cat([
            mat.mean(dim=0)
        ]).numpy()  # → (192,)


    col_names = (
            [f"emb_mean_{i}" for i in range(dim)]     )
    df = pd.DataFrame.from_dict(rows, orient="index", columns=col_names)
    df.index.name = "defect_id"
    return df


import os
import json
def create_defects_embeddings():
    passive = ["Toys_and_Games", "Office_Products", "Pet_Supplies"]
    #    online = ["Books","Sports_and_Outdoors","Beauty_and_Personal_Care"]

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    for method in ['sage']:
        for dataset in passive:
            print(f"{dataset}-->{method}")

            defects_df = pd.read_csv(f"{PARENT_DIR}/data/noisy/{dataset}_5/injected_noise_new.csv")
            try:
                emb = load_embeddings(f"{BASE_DIR}/features_generator/node_embeddings/node_emb_{dataset}_final.pt")
            except Exception as e:
                emb = load_embeddings(
                    f"{BASE_DIR}/features_generator/node_embeddings/node_emb_inference_{dataset}_final.pt")

            df_emb = aggregate_embeddings(emb, defects_df)

            print('compute_hand')

            os.makedirs(f"{BASE_DIR}/features_generator/node_embeddings/", exist_ok=True)
            torch.save({
                "features": torch.tensor(df_emb.values, dtype=torch.float32),  # (n_difetti, 209)
                "defect_ids": df_emb.index.tolist(),
                "col_names": df_emb.columns.tolist(),
            },
                f"{BASE_DIR}/features_generator/node_embeddings/defects_embeddings_{dataset}_{method}_new_version_inference_2.pt")


# ═══════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════
import os
import json
if __name__ == "__main__":
