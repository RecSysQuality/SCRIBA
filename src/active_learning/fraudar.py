import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import heapq
import time
import numpy as np
import pandas as pd
import os
import networkx as nx
from networkx.algorithms.clique import find_cliques

import torch
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)


class FraudarTopK:
    def __init__(self, c=5.0):
        self.c = c

    def fit_graph(self, df, user_col="user_id", item_col="item_id"):
        df = df[[user_col, item_col]].drop_duplicates().copy()
        user_codes, user_uniques = pd.factorize(df[user_col], sort=False)
        item_codes, item_uniques = pd.factorize(df[item_col], sort=False)

        self.user_ids = user_uniques.to_numpy()
        self.item_ids = item_uniques.to_numpy()

        self.n_users = len(self.user_ids)
        self.n_items = len(self.item_ids)

        rows = user_codes.astype(np.int32)
        cols = item_codes.astype(np.int32)
        data = np.ones(len(df), dtype=np.float32)

        self.B = csr_matrix((data, (rows, cols)), shape=(self.n_users, self.n_items), dtype=np.float32)
        return self

    def _build_weights(self, B):
        item_deg = np.asarray(B.sum(axis=0)).ravel().astype(np.float32)
        edge_w = np.zeros_like(item_deg, dtype=np.float64)
        mask = item_deg > 0
        edge_w[mask] = 1.0 / np.log(item_deg[mask] + self.c)
        return edge_w

    def _run_once(self, B, min_nodes=10):
        Bt = B.transpose().tocsr()
        nU, nI = B.shape
        total_nodes = nU + nI

        edge_w = self._build_weights(B)

        active_u = np.ones(nU, dtype=bool)
        active_i = np.ones(nI, dtype=bool)

        user_score = np.zeros(nU, dtype=np.float64)
        item_score = np.zeros(nI, dtype=np.float64)

        weighted_data = edge_w[B.indices]
        weighted_B = csr_matrix(
            (weighted_data, B.indices, B.indptr),
            shape=B.shape
        )

        user_score = np.asarray(
            weighted_B.sum(axis=1)
        ).ravel()

        item_score = np.asarray(
            weighted_B.sum(axis=0)
        ).ravel()

        item_deg = Bt.getnnz(axis=1)
        item_score = item_deg * edge_w

        total_f = user_score.sum()
        active_count = total_nodes
        best_g = total_f / active_count if active_count > 0 else 0.0
        best_step = 0

        removed = []
        version = np.zeros(total_nodes, dtype=np.int64)
        heap = []

        def push_user(u):
            heapq.heappush(heap, (user_score[u], 0, u, version[u]))

        def push_item(i):
            heapq.heappush(heap, (item_score[i], 1, i, version[nU + i]))

        for u in range(nU):
            if active_u[u]:
                push_user(u)
        for i in range(nI):
            if active_i[i] and item_deg[i] > 0:
                push_item(i)

        step = 0
        while active_count > min_nodes and heap:
            score, typ, idx, ver = heapq.heappop(heap)

            if typ == 0:
                if not active_u[idx] or ver != version[idx]:
                    continue
                rem_score = user_score[idx]
                active_u[idx] = False
                removed.append((0, idx))
                total_f -= rem_score
                active_count -= 1

                s, e = B.indptr[idx], B.indptr[idx + 1]
                neigh_items = B.indices[s:e]
                for i in neigh_items:
                    if active_i[i]:
                        item_score[i] -= edge_w[i]
                        version[nU + i] += 1
                        push_item(i)
            else:
                if not active_i[idx] or ver != version[nU + idx]:
                    continue
                rem_score = item_score[idx]
                active_i[idx] = False
                removed.append((1, idx))
                total_f -= rem_score
                active_count -= 1

                s, e = Bt.indptr[idx], Bt.indptr[idx + 1]
                neigh_users = Bt.indices[s:e]
                for u in neigh_users:
                    if active_u[u]:
                        user_score[u] -= edge_w[idx]
                        version[u] += 1
                        push_user(u)

            step += 1
            g = total_f / active_count if active_count > 0 else 0.0
            if g > best_g:
                best_g = g
                best_step = step

        keep_u = np.ones(nU, dtype=bool)
        keep_i = np.ones(nI, dtype=bool)
        for t in range(best_step):
            typ, idx = removed[t]
            if typ == 0:
                keep_u[idx] = False
            else:
                keep_i[idx] = False

        suspicious_u = np.where(keep_u)[0]
        suspicious_i = np.where(keep_i)[0]

        if len(suspicious_u) == 0 or len(suspicious_i) == 0:
            return None

        sub_B = B[suspicious_u][:, suspicious_i]
        n_edges = sub_B.nnz
        if n_edges == 0:
            return None

        return {
            "score": float(best_g),
            "user_idx": suspicious_u,
            "item_idx": suspicious_i,
            "n_users": int(len(suspicious_u)),
            "n_items": int(len(suspicious_i)),
            "n_edges": int(n_edges),
        }

    def run_topk(self, k=10, min_nodes=10, min_edges=20, min_score=0.0, removal="hard", downweight=0.2):
        B = self.B.copy().tocsr().astype(np.float32)
        results = []

        for rank in range(k):
            print(rank)
            res = self._run_once(B, min_nodes=min_nodes)
            if res is None:
                break
            if res["n_edges"] < min_edges or res["score"] < min_score:
                break

            users = self.user_ids[res["user_idx"]]
            items = self.item_ids[res["item_idx"]]

            results.append({
                "rank": rank + 1,
                "score": res["score"],
                "n_users": res["n_users"],
                "n_items": res["n_items"],
                "n_edges": res["n_edges"],
                "users": users,
                "items": items,
            })

            u_idx = res["user_idx"]
            i_idx = set(res["item_idx"].tolist())

            if removal == "hard":
                mask_items = np.zeros(B.shape[1], dtype=bool)
                mask_items[list(i_idx)] = True

                for u in u_idx:
                    s, e = B.indptr[u], B.indptr[u + 1]
                    cols = B.indices[s:e]
                    remove_mask = mask_items[cols]
                    B.data[s:e][remove_mask] = 0.0
                B.eliminate_zeros()

            elif removal == "soft":
                for u in u_idx:
                    s, e = B.indptr[u], B.indptr[u + 1]
                    cols = B.indices[s:e]
                    vals = B.data[s:e]
                    for pos, it in enumerate(cols):
                        if it in i_idx:
                            vals[pos] *= downweight
                B.eliminate_zeros()

        return results

import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering
from collections import defaultdict


def ego_splitting_group(df, users, items, min_cluster_size=3):
    """
    Ego-splitting adattato a subgraph bipartito user-item.

    Input:
        df: dataframe completo o sub-df
        users: lista utenti del gruppo Fraudar
        items: lista item del gruppo Fraudar

    Output:
        lista di micro-gruppi:
        [
            {"users": [...], "items": [...]},
            ...
        ]
    """

    sub = df[df["user_id"].isin(users) & df["item_id"].isin(items)].copy()

    # mapping
    user2items = sub.groupby("user_id")["item_id"].apply(set).to_dict()
    item2users = sub.groupby("item_id")["user_id"].apply(set).to_dict()

    all_items = list(items)
    item_index = {i: idx for idx, i in enumerate(all_items)}
    n_items = len(all_items)


    cooc = np.zeros((n_items, n_items), dtype=np.float32)

    for u, its in user2items.items():
        its = list(its)
        for i in range(len(its)):
            for j in range(i + 1, len(its)):
                a, b = item_index[its[i]], item_index[its[j]]
                cooc[a, b] += 1
                cooc[b, a] += 1

    # evita casi vuoti
    if cooc.sum() == 0:
        return [{"users": users, "items": items}]


    k = max(2, len(items)//3)

    clustering = SpectralClustering(
        n_clusters=k,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=42
    )

    labels = clustering.fit_predict(cooc)


    clusters = defaultdict(set)

    for item, lab in zip(all_items, labels):
        clusters[lab].add(item)

    micro_groups = []

    for lab, item_set in clusters.items():
        if len(item_set) < min_cluster_size:
            continue

        # utenti che toccano questi item
        group_users = set()
        for it in item_set:
            group_users.update(item2users.get(it, []))

        # intersezione con utenti originali
        group_users = list(set(group_users) & set(users))

        if len(group_users) > 0:
            micro_groups.append({
                "users": group_users,
                "items": list(item_set)
            })

    # fallback
    if len(micro_groups) == 0:
        return [{"users": users, "items": items}]

    return micro_groups


def detect_defects():
    obj = {}
    datasets = ["Office_Products_5", "Beauty_and_Personal_Care", "Books"]

    start = True
    for dataset in datasets:
        if start:
            df = pd.read_csv(f"{PARENT_DIR}/data/noisy/{dataset}_5/dataset_dirty_new.csv",
                             usecols=["user_id", "item_id"])
            fraud = FraudarTopK(c=30.0).fit_graph(df)
            final_groups = []
            print('start fraudar')
            st = time.time()
            groups = fraud.run_topk(
                k=50,
                min_nodes=5,
                min_edges=10,
                min_score=0.25,
                removal="hard",  # oppure "hard"
                downweight=0.02
            )
            end = time.time()
            print(f'fraudar in {end - st}')

            for i, g in enumerate(groups):
                print('group: ', i)
                dfg = df[(df['user_id'].isin(g['users'])) & df['item_id'].isin(g['items'])]
                dfg.to_csv(f"{PARENT_DIR}/data/noisy/{dataset}_5/groups/dataset_dirty_new_g{i}_50_hard.csv",
                           index=False)

                print(g["rank"], g["score"], g["n_users"], g["n_items"], g["n_edges"])
                st = time.time()
                micro = ego_splitting_group(
                    df=df,
                    users=g["users"],
                    items=g["items"]
                )
                end = time.time()
                print(f'micro found in {end - st}')
                for j, group in enumerate(micro):
                    print('micro group: ', j)
                    dfg = df[(df['user_id'].isin(group['users'])) & df['item_id'].isin(group['items'])]
                    dfg.to_csv(f"{PARENT_DIR}/data/noisy/{dataset}_5/groups/mini_group_hard_{i}_{j}.csv", index=False)

            files = [file for file in os.listdir(f"{PARENT_DIR}/data/noisy/{dataset}_5/groups/") if
                     file.startswith("dense_extr_") or file.startswith('cam_extr_')]
            dfs = []
            for f in files:
                df = pd.read_csv(f"{PARENT_DIR}/data/noisy/{dataset}_5/groups/{f}")

                # 1. rename
                df = df.rename(columns={"defect": "group_id"})

                # 2. elimina gruppi troppo grandi
                group_sizes = df["group_id"].value_counts()
                valid_groups = group_sizes[group_sizes <= 1000].index

                df = df[df["group_id"].isin(valid_groups)]

                dfs.append(df)
            final_df = pd.concat(dfs, ignore_index=True)

            # salva
            final_df.to_csv(f"{PARENT_DIR}/data/noisy/{dataset}_5/groups/merged_filtered.csv", index=False)

            final_groups = []
            s = False
            if s:
                files = [file for file in os.listdir(f"{PARENT_DIR}/data/noisy/{dataset}_5/groups/") if
                         file.startswith("dataset_dirty_new_g") and file.endswith('_50_hard.csv')]
                for ind, file in enumerate(files):

                    df = pd.read_csv(f"{PARENT_DIR}/data/noisy/{dataset}_5/groups/{file}")
                    users = df['user_id'].unique().tolist()
                    items = df['item_id'].unique().tolist()

                    # grafo bipartito user-item
                    G = nx.Graph()

                    # aggiungi nodi
                    G.add_nodes_from(users, bipartite='users')
                    G.add_nodes_from(items, bipartite='items')

                    # aggiungi archi
                    G.add_edges_from(df[['user_id', 'item_id']].values)

                    # degree utenti
                    user_degree = dict(G.degree(users))
                    item_degree = dict(G.degree(items))

                    items_nodes = [n for n, d in G.nodes(data=True) if d.get('bipartite') == 'items']

                    dense_users = [u for u, d in user_degree.items() if d >= 20]

                    H = nx.Graph()
                    user_proj = nx.bipartite.weighted_projected_graph(G, dense_users)

                    for u, v, data in user_proj.edges(data=True):
                        if data["weight"] >= 9:
                            H.add_edge(u, v)

                    raw_cliques = list(find_cliques(H))
                    cliques = [set(c) for c in raw_cliques]

                    C = nx.Graph()

                    # aggiungi nodi (clique index)
                    for i in range(len(cliques)):
                        C.add_node(i)

                    # collega cliques con overlap
                    for i in range(len(cliques)):
                        for j in range(i + 1, len(cliques)):
                            if cliques[i] & cliques[j]:  # intersezione non vuota
                                C.add_edge(i, j)

                    merged_cliques = []

                    for component in nx.connected_components(C):
                        merged = set()

                        for idx in component:
                            merged |= cliques[idx]

                        merged_cliques.append(tuple(sorted(merged)))

                    defects = [
                        c for c in merged_cliques
                        if 4 <= len(c)
                    ]
                    user_to_defect = {}
                    if len(defects) > 0:
                        for i, clique in enumerate(defects):
                            defect_id = f"defect_dense_{ind}_{i}"

                            for u in clique:
                                user_to_defect[u] = defect_id

                        dense_df = df.copy()
                        dense_df["defect"] = dense_df["user_id"].map(user_to_defect)
                        dense_df = dense_df[dense_df["defect"].notna()]
                        dense_df.to_csv(f"{PARENT_DIR}/data/noisy/{dataset}_5/groups/dense_extr_{ind}.csv", index=False)

                    # =========================
                    # CAMOUFLAGE
                    # users con 14-32 archi
                    # =========================
                    camouflage_users = [
                        u for u, d in user_degree.items()
                        if 14 <= d <= 32
                    ]
                    H = nx.Graph()
                    user_proj = nx.bipartite.weighted_projected_graph(G, camouflage_users)

                    for u, v, data in user_proj.edges(data=True):
                        if data["weight"] >= 5 and data["weight"] <= 8:
                            H.add_edge(u, v)

                    raw_cliques = list(find_cliques(H))
                    cliques = [set(c) for c in raw_cliques]

                    C = nx.Graph()

                    # aggiungi nodi (clique index)
                    for i in range(len(cliques)):
                        C.add_node(i)

                    # collega cliques con overlap
                    for i in range(len(cliques)):
                        for j in range(i + 1, len(cliques)):
                            if cliques[i] & cliques[j]:  # intersezione non vuota
                                C.add_edge(i, j)
                    merged_cliques = []

                    for component in nx.connected_components(C):
                        merged = set()

                        for idx in component:
                            merged |= cliques[idx]

                        merged_cliques.append(tuple(sorted(merged)))

                    defects = [
                        c for c in merged_cliques
                        if 4 <= len(c) <= 21
                    ]
                    user_to_defect = {}
                    if len(defects) > 0:
                        for i, clique in enumerate(defects):
                            defect_id = f"defect_camouflage_{ind}_{i}"

                            for u in clique:
                                user_to_defect[u] = defect_id

                        camouflage_df = df.copy()
                        camouflage_df["defect"] = camouflage_df["user_id"].map(user_to_defect)
                        camouflage_df = camouflage_df[camouflage_df["defect"].notna()]
                        camouflage_df.to_csv(f"{PARENT_DIR}/data/noisy/{dataset}_5/groups/cam_extr_{ind}.csv",
                                             index=False)
