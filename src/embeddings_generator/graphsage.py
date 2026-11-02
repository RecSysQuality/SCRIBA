# ============================================================
# GraphSAGE induttivo per SCRIBA + defect embedding [utenti ; item]
# ============================================================
# - UN solo encoder condiviso, addestrato su piu' grafi (corrotti) del
#   passive phase e applicato in inferenza a grafi mai visti.
# - Feature di input solo strutturali e comparabili tra grafi diversi
#   (niente normalizzazioni per il massimo del singolo grafo).
# - Training BPR su triplette (u, i+, i-) con negativi campionati SOLO
#   tra gli item.
# - Inferenza full-neighborhood esatta e deterministica: lo stesso nodo
#   ha sempre lo stesso embedding (importante per LinUCB).
# - Defect embedding: concat(media z_u su U_k, media z_i su I_k)
#   -> dimensione 2 * out_dim  (es. out_dim=16 -> 32).
#
# Richiede torch_geometric con pyg-lib o torch-sparse (LinkNeighborLoader).
# ============================================================

import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import SAGEConv
from torch_geometric.sampler import NegativeSampling
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# GRAFO
# ============================================================

@dataclass
class Graph:
    data: Data                 # x [N, 3], edge_index [2, 2E] (simmetrico), su CPU
    ui_edges: torch.Tensor     # [2, E] solo archi user -> item (per la loss)
    is_item: torch.Tensor      # [N] bool
    meta: Dict[str, Dict[str, int]]   # {"user2idx": ..., "item2idx": ...}


def read_interactions(path: str) -> pd.DataFrame:
    """
    Legge le interazioni con gli id come STRINGHE. Senza dtype=str pandas
    converte gli ISBN numerici in interi (0060577312 -> 60577312) e gli id
    non corrispondono piu' a quelli dei file dei gruppi e del LOO.
    """
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df.columns = df.columns.str.strip()
    if "rating" in df.columns:
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    return df


def build_graph(df: pd.DataFrame) -> Graph:
    """
    Grafo bipartito user-item. Indici: utenti 0..U-1, item U..U+I-1.
    Gli id vengono convertiti in stringa, cosi' il lookup dei defect e' robusto.
    """
    df = df.copy()
    df.columns = df.columns.str.strip()
    df["user_id"] = df["user_id"].astype(str)
    df["item_id"] = df["item_id"].astype(str)

    has_rating = "rating" in df.columns
    if has_rating:
        df = df.groupby(["user_id", "item_id"], as_index=False)["rating"].mean()
    else:
        df = df.drop_duplicates(["user_id", "item_id"])

    users = df["user_id"].unique()
    items = df["item_id"].unique()
    U, I = len(users), len(items)
    N = U + I

    user2idx = {u: k for k, u in enumerate(users)}
    item2idx = {it: U + k for k, it in enumerate(items)}

    src = torch.tensor(df["user_id"].map(user2idx).to_numpy(), dtype=torch.long)
    dst = torch.tensor(df["item_id"].map(item2idx).to_numpy(), dtype=torch.long)
    ui_edges = torch.stack([src, dst])
    edge_index = torch.cat([ui_edges, ui_edges.flip(0)], dim=1)

    rating = (torch.tensor(df["rating"].to_numpy(), dtype=torch.float) / 5.0
              if has_rating else None)

    is_item = torch.zeros(N, dtype=torch.bool)
    is_item[U:] = True

    x = node_features(edge_index, ui_edges, rating, is_item, N)
    data = Data(x=x, edge_index=edge_index, num_nodes=N)
    return Graph(data, ui_edges, is_item,
                 {"user2idx": user2idx, "item2idx": item2idx})


def node_features(edge_index, ui_edges, rating, is_item, N) -> torch.Tensor:
    """
    [is_user, log(1+grado), rating medio]  -> in_dim = 3.
    Tutte comparabili tra grafi: il grado NON e' diviso per il massimo del
    grafo, altrimenti lo stesso nodo avrebbe feature diverse in grafi diversi.
    Il flag is_user permette al modello di trattare utenti e item in modo diverso.
    """
    deg = torch.bincount(edge_index[0], minlength=N).float()
    if rating is not None:
        rsum = torch.zeros(N)
        rsum.index_add_(0, ui_edges[0], rating)
        rsum.index_add_(0, ui_edges[1], rating)
        avg_rating = rsum / deg.clamp(min=1)
    else:
        avg_rating = torch.zeros(N)
    is_user = (~is_item).float()
    return torch.stack([is_user, torch.log1p(deg), avg_rating], dim=1)


# ============================================================
# MODELLO
# ============================================================

class GraphSAGE(nn.Module):
    def __init__(self, in_dim: int = 3, hid: int = 64, out_dim: int = 16,
                 dropout: float = 0.3):
        super().__init__()
        self.config = {"in_dim": in_dim, "hid": hid, "out_dim": out_dim,
                       "dropout": dropout}
        self.convs = nn.ModuleList([SAGEConv(in_dim, hid), SAGEConv(hid, out_dim)])
        self.dropout = dropout

    @property
    def out_dim(self) -> int:
        return self.config["out_dim"]

    def forward(self, x, edge_index):
        x = F.relu(self.convs[0](x, edge_index))
        x = F.dropout(x, self.dropout, self.training)
        return self.convs[1](x, edge_index)

    @torch.no_grad()
    def full_inference(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Inferenza layer-wise sul grafo intero, identica a forward() in eval mode
        ma senza materializzare un messaggio per arco: la media dei vicini e'
        una moltiplicazione per la matrice di adiacenza normalizzata per riga.
        Nessun campionamento -> embedding deterministici.
        """
        was_training = self.training
        self.eval()
        A = mean_adjacency(edge_index, x.size(0)).to(x.device)
        h = x
        for k, conv in enumerate(self.convs):
            h = conv.lin_l(torch.sparse.mm(A, h)) + conv.lin_r(h)
            if k < len(self.convs) - 1:
                h = F.relu(h)
        self.train(was_training)
        return h


def mean_adjacency(edge_index: torch.Tensor, N: int) -> torch.Tensor:
    """A[dst, src] = 1 / grado(dst): riproduce l'aggregazione 'mean' di SAGEConv."""
    src, dst = edge_index
    deg = torch.bincount(dst, minlength=N).float().clamp(min=1)
    A = torch.sparse_coo_tensor(torch.stack([dst, src]), 1.0 / deg[dst], (N, N))
    return A.coalesce().to_sparse_csr()


# ============================================================
# TRAINING
# ============================================================

def bpr_loss(z, src, pos, neg) -> torch.Tensor:
    """BPR su triplette. neg puo' essere [B] o [B, k]."""
    neg = neg.view(src.size(0), -1)
    s_pos = (z[src] * z[pos]).sum(-1, keepdim=True)          # [B, 1]
    s_neg = (z[src].unsqueeze(1) * z[neg]).sum(-1)            # [B, k]
    return -F.logsigmoid(s_pos - s_neg).mean()


def train(
    model: GraphSAGE,
    graphs: List[Graph],
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    batch_size: int = 4096,
    num_neighbors: Sequence[int] = (15, 10),
    steps_per_epoch: Optional[int] = None,
    num_workers: int = 4,
) -> GraphSAGE:
    """
    Mini-batch sugli archi user -> item. Per ogni arco (u, i+) il loader
    campiona un negativo i- SOLO tra gli item (dst_weight = is_item) e il
    vicinato a 2 hop di u, i+, i-: gli embedding usati nella loss sono quindi
    calcolati con il vicinato completo (campionato), non troncato.

    steps_per_epoch: se impostato, ogni "epoca" usa solo quel numero di batch
    per grafo (un sottoinsieme casuale degli archi). Utile su Books & co.
    """
    model = model.to(device)
    loaders = [
        LinkNeighborLoader(
            g.data,
            num_neighbors=list(num_neighbors),
            edge_label_index=g.ui_edges,
            neg_sampling=NegativeSampling("triplet", amount=1,
                                          dst_weight=g.is_item.float()),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
        )
        for g in graphs
    ]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    for ep in range(epochs):
        model.train()
        tot, n = 0.0, 0
        for loader in loaders:
            for step, b in enumerate(loader):
                if steps_per_epoch is not None and step >= steps_per_epoch:
                    break
                b = b.to(device)
                z = model(b.x, b.edge_index)
                loss = bpr_loss(z, b.src_index, b.dst_pos_index, b.dst_neg_index)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tot += loss.item()
                n += 1
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"Epoch {ep:3d} | BPR {tot / max(n, 1):.4f}")
    return model


# ============================================================
# SALVATAGGIO / CARICAMENTO (la config viaggia con i pesi)
# ============================================================

def save_encoder(model: GraphSAGE, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"config": model.config, "state_dict": model.state_dict()}, path)


def load_encoder(path: str) -> GraphSAGE:
    ckpt = torch.load(path, map_location="cpu")
    model = GraphSAGE(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()


# ============================================================
# EMBEDDING DEI NODI E DEI DEFECT
# ============================================================

def embed_graph(model: GraphSAGE, graph: Graph) -> Dict:
    """Embedding di tutti i nodi di un grafo (visto o nuovo)."""
    z = model.full_inference(graph.data.x.to(device),
                             graph.data.edge_index.to(device)).cpu()
    return {"node_embeddings": z, "meta": graph.meta}


def defect_embeddings_old(
    emb: Dict,
    defects: List[Tuple[Sequence, Sequence]],
) -> torch.Tensor:
    """
    defects: lista di (utenti, item) con gli id ORIGINALI del dataframe.
    Restituisce [m, 2 * out_dim]: [media degli utenti ; media degli item].
    Gli embedding vanno calcolati sul grafo che CONTIENE il defect
    (il grafo corrotto G~ o G^(t)), altrimenti i fake user non esistono.
    """
    Z = emb["node_embeddings"]
    u2i, i2i = emb["meta"]["user2idx"], emb["meta"]["item2idx"]
    d = Z.size(1)
    V = torch.zeros(len(defects), 2 * d)
    missing = 0
    for k, (users, items) in enumerate(defects):
        uidx = [u2i[str(u)] for u in users if str(u) in u2i]
        iidx = [i2i[str(i)] for i in items if str(i) in i2i]
        missing += (len(users) - len(uidx)) + (len(items) - len(iidx))
        if uidx:
            V[k, :d] = Z[uidx].mean(0)
        if iidx:
            V[k, d:] = Z[iidx].mean(0)
    if missing:
        print(f"[defect_embeddings] attenzione: {missing} nodi non trovati nel grafo")
    return V

import math

def defect_extra_features(graph: Graph, defects: List[Tuple[Sequence, Sequence]]) -> torch.Tensor:
    """
    Due feature manuali per defect, comparabili tra grafi -> [m, 2]:
      0: densita'  = archi dentro il defect / (n utenti * n item)
      1: quota     = archi dentro il defect / interazioni totali degli item del defect
    Da calcolare sul grafo che CONTIENE il defect, come gli embedding.
    """
    u2i, i2i = graph.meta["user2idx"], graph.meta["item2idx"]
    src, dst = graph.ui_edges.tolist()
    adj = {}
    for u, i in zip(src, dst):
        adj.setdefault(u, set()).add(i)
    deg = torch.bincount(graph.data.edge_index[0], minlength=graph.data.num_nodes).float()

    F_ = torch.zeros(len(defects), 2)
    for k, (users, items) in enumerate(defects):
        U = [u2i[str(u)] for u in users if str(u) in u2i]
        I = [i2i[str(i)] for i in items if str(i) in i2i]
        if not U or not I:
            continue
        I_set = set(I)
        inner = sum(len(adj.get(u, set()) & I_set) for u in U)
        F_[k, 0] = inner / (len(U) * len(I))
        F_[k, 1] = inner / deg[I].sum().clamp(min=1).item()
    return F_

def defect_embeddings(
    emb: Dict,
    defects: List[Tuple[Sequence, Sequence]],
    with_size: bool = True,
) -> torch.Tensor:
    """
    defects: lista di (utenti, item) con gli id ORIGINALI del dataframe.
    Restituisce [m, 2 * out_dim]: [media degli utenti ; media degli item].
    Con with_size=True aggiunge 2 colonne in fondo: log(1 + n utenti), log(1 + n item),
    perche' la media da sola non dice quanto e' grande il defect.
    Gli embedding vanno calcolati sul grafo che CONTIENE il defect.
    """
    Z = emb["node_embeddings"]
    u2i, i2i = emb["meta"]["user2idx"], emb["meta"]["item2idx"]
    d = Z.size(1)
    V = torch.zeros(len(defects), 2 * d + (2 if with_size else 0))
    missing = 0
    for k, (users, items) in enumerate(defects):
        uidx = [u2i[str(u)] for u in users if str(u) in u2i]
        iidx = [i2i[str(i)] for i in items if str(i) in i2i]
        missing += (len(users) - len(uidx)) + (len(items) - len(iidx))
        if uidx:
            V[k, :d] = Z[uidx].mean(0)
        if iidx:
            V[k, d:2 * d] = Z[iidx].mean(0)
        if with_size:
            V[k, 2 * d] = math.log1p(len(uidx))
            V[k, 2 * d + 1] = math.log1p(len(iidx))
    if missing:
        print(f"[defect_embeddings] attenzione: {missing} nodi non trovati nel grafo")
    return V


# Standardizzazione: fit SOLO sui defect del passive phase, poi applicata
# sempre uguale (regressore e LinUCB devono vedere la stessa scala).

def fit_standardizer(V: torch.Tensor) -> Dict[str, torch.Tensor]:
    return {"mean": V.mean(0), "std": V.std(0).clamp(min=1e-6)}


def standardize(V: torch.Tensor, stats: Dict[str, torch.Tensor]) -> torch.Tensor:
    return (V - stats["mean"]) / stats["std"]


# ============================================================
# MAIN (esempio)
# ============================================================

# if __name__ == "__main__":
#     set_seed(42)
#
#     OUT_DIM = 16          # dimensione per blocco -> defect embedding da 2*OUT_DIM
#     HID = 64
#     ENC_PATH = "node_embeddings/weights/sage_shared.pt"
#
#     # Grafi CORROTTI (con i defect iniettati) del passive phase, solo train split.
#     # Adatta il percorso a dove salvi i grafi con le iniezioni.
#     passive = ["Office_Products", "Toys_and_Games", "Pet_Supplies"]
#     graphs = [build_graph(read_interactions(f"data/corrupted/{d}/{d}.train.csv"))
#               for d in passive]
#
#     # 1) un solo encoder condiviso su tutti i grafi
#     model = GraphSAGE(in_dim=3, hid=HID, out_dim=OUT_DIM)
#     model = train(model, graphs, epochs=50, steps_per_epoch=500)
#     save_encoder(model, ENC_PATH)
#
#     # 2) inferenza su un grafo mai visto (active phase)
#     enc = load_encoder(ENC_PATH)
#     g_new = build_graph(read_interactions("data/corrupted/Books/Books.train.csv"))
#     emb = embed_graph(enc, g_new)
#
#     # 3) defect embedding: defects = [(lista_user_id, lista_item_id), ...]
#     # V = defect_embeddings(emb, defects)          # [m, 2*OUT_DIM]
#     # V = standardize(V, stats_dal_passive_phase)  # stessa scala per f_theta e LinUCB
#     print("Node embeddings:", emb["node_embeddings"].shape)