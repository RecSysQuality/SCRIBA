# ============================================================
# GRAPH EMBEDDING PIPELINE (GraphSAGE + PinSAGE + PinSAGE v2)
# ============================================================
# OBIETTIVO:
# - Allenare un modello su più grafi (G1, G2, G3)
# - Ottenere embedding nello STESSO spazio
# - Salvare embedding separati per ogni grafo
# - Mantenere mapping nodo → embedding (fondamentale)
# - Permettere inference su nuovi grafi (G4, G5)
#
# PinSAGE v2 (Pinterest-style, ottimizzato per 9M+ edges):
# - Importance-based sampling via random walk (PPR locale)
# - Importance-weighted aggregation con BatchNorm
# - Hard negative mining (curriculum)
# - Mini-batch training con NeighborLoader → scalabile
# - AMP (Automatic Mixed Precision) → 2× speedup su GPU
# - Grad clipping + AdamW + CosineAnnealing
# ============================================================

import os
import random
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
# Compatibilità torch.amp tra PyTorch < 2.0 e >= 2.0
try:
    from torch.amp import autocast, GradScaler
    _AMP_DTYPE = {"device_type": "cuda", "dtype": torch.float16}
    _SCALER_KWARGS: dict = {"device": "cuda"}
except ImportError:
    from torch.cuda.amp import autocast, GradScaler  # type: ignore
    _AMP_DTYPE = {}
    _SCALER_KWARGS = {}



# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ============================================================
# SEED (riproducibilità)
# ============================================================

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# ============================================================
# GRAPH BUILDER
# ============================================================

def load_graph_from_df(df: pd.DataFrame) -> Data:

    df = df.copy()
    df.columns = df.columns.str.strip()

    users = df["user_id"].unique()
    items = df["item_id"].unique()

    user2idx = {u: i for i, u in enumerate(users)}
    item2idx = {it: i + len(users) for i, it in enumerate(items)}

    src = df["user_id"].map(user2idx).values
    dst = df["item_id"].map(item2idx).values

    edge_index = torch.tensor(
        [list(src) + list(dst), list(dst) + list(src)],
        dtype=torch.long
    )

    # edge_attr[:, 0] = rating normalizzato → usato per avg_rating feature
    ratings = torch.tensor(
        list(df["rating"].values / 5.0) * 2, dtype=torch.float
    ).unsqueeze(1)

    node_ids = (
        [f"user:{u}" for u in users] +
        [f"item:{i}" for i in items]
    )

    graph = Data(
        edge_index=edge_index,
        edge_attr=ratings,          # [E, 1] → verrà espanso a [E, 2] dopo
        num_nodes=len(node_ids)
    )
    graph.meta = {"user2idx": user2idx, "item2idx": item2idx}
    graph.node_ids = node_ids
    return graph

# ============================================================
# FEATURE ENGINEERING
# ============================================================

def extract_node_features(graph: Data) -> torch.Tensor:
    """
    Feature strutturali → consistenti tra grafi diversi.
    Usa edge_attr[:, 0] per il rating medio (compatibile sia con
    edge_attr [E,1] che [E,2]).
    """
    N = graph.num_nodes
    row, col = graph.edge_index

    deg = torch.zeros(N, device=row.device)
    deg.scatter_add_(0, row, torch.ones_like(row, dtype=torch.float))

    deg_norm  = deg / (deg.max() + 1e-8)
    activity  = torch.log1p(deg)

    if graph.edge_attr is not None:
        ratings = graph.edge_attr[:, 0]          # prima colonna = rating
        rsum = torch.zeros(N, device=row.device).scatter_add_(0, row, ratings)
        avg  = rsum / deg.clamp(min=1)
    else:
        avg = deg_norm

    x = torch.stack([deg_norm, avg, activity, deg_norm**2, avg**2], dim=1)
    return x.to(device)

# ============================================================
# IMPORTANCE WEIGHTS  (cuore del PinSAGE originale)
# ============================================================

def compute_importance_weights(
    edge_index: torch.Tensor,
    num_nodes: int,
    walk_length: int = 3,
    num_walks: int = 80,
    chunk_size: int = 10_000,
) -> torch.Tensor:

    row, col = edge_index[0].cpu().long(), edge_index[1].cpu().long()

    if not HAS_TC:
        # ── Fallback: normalized degree weights ──────────────────────────
        deg = torch.zeros(num_nodes, dtype=torch.float)
        deg.scatter_add_(0, row, torch.ones_like(row, dtype=torch.float))
        w = 1.0 / deg[row].clamp(min=1.0)
        s = torch.zeros(num_nodes, dtype=torch.float)
        s.scatter_add_(0, row, w)
        return (w / s[row].clamp(min=1e-8)).to(edge_index.device)

    # ── Random walk su CPU (stabile per grafi grandi) ────────────────────
    src_acc: List[torch.Tensor] = []
    dst_acc: List[torch.Tensor] = []
    cnt_acc: List[torch.Tensor] = []

    for chunk_start in range(0, num_nodes, chunk_size):
        chunk_end = min(chunk_start + chunk_size, num_nodes)
        seeds = torch.arange(chunk_start, chunk_end).repeat(num_walks)  # [C*nw]

        # walks: [C*nw, walk_length+1] — ogni riga = un cammino
        walks = _tc_rw(row, col, seeds,
                       walk_length=walk_length,
                       num_nodes=num_nodes,
                       coalesced=False)

        # Sorgente (colonna 0) replicata per ogni step 1..L
        wl = walk_length
        ws = walks[:, 0].unsqueeze(1).expand(-1, wl).reshape(-1)   # [C*nw*wl]
        wd = walks[:, 1:].reshape(-1)                               # [C*nw*wl]

        # Rimuovi dead-end (-1) e self-loop
        valid = (wd >= 0) & (ws != wd)
        ws, wd = ws[valid], wd[valid]
        if ws.numel() == 0:
            continue

        # Conta coppie uniche via unique su tensor [M, 2]
        pairs, counts = torch.stack([ws, wd], dim=1).unique(
            dim=0, return_counts=True
        )
        src_acc.append(pairs[:, 0])
        dst_acc.append(pairs[:, 1])
        cnt_acc.append(counts.float())

    if not src_acc:
        # Tutti i walk sono falliti (grafo disconnesso o sparsissimo)
        return torch.ones(edge_index.size(1), dtype=torch.float,
                          device=edge_index.device)

    rw_src = torch.cat(src_acc)   # [K]
    rw_dst = torch.cat(dst_acc)   # [K]
    rw_cnt = torch.cat(cnt_acc)   # [K]

    # Normalizzazione per riga: P(dst | src) = count(src,dst) / sum_d count(src,d)
    norm = torch.zeros(num_nodes, dtype=torch.float)
    norm.scatter_add_(0, rw_src, rw_cnt)
    rw_weights = rw_cnt / norm[rw_src].clamp(min=1e-8)

    rw_keys   = rw_src.long()  * num_nodes + rw_dst.long()
    orig_keys = row.long()     * num_nodes + col.long()

    sort_idx      = torch.argsort(rw_keys)
    rw_keys_s     = rw_keys[sort_idx]
    rw_weights_s  = rw_weights[sort_idx]

    pos   = torch.searchsorted(rw_keys_s, orig_keys).clamp(max=rw_keys_s.size(0) - 1)
    found = rw_keys_s[pos] == orig_keys

    # Fallback per edge mai visitati nei walk → degree-norm
    deg = torch.zeros(num_nodes, dtype=torch.float)
    deg.scatter_add_(0, row, torch.ones_like(row, dtype=torch.float))
    fb  = 1.0 / deg[row].clamp(min=1.0)
    fb_norm = torch.zeros(num_nodes, dtype=torch.float)
    fb_norm.scatter_add_(0, row, fb)
    fb = fb / fb_norm[row].clamp(min=1e-8)

    weights = torch.where(found, rw_weights_s[pos], fb)
    return weights.to(edge_index.device)


def add_importance_weights(graph: Data, **rw_kwargs) -> Data:


    imp = compute_importance_weights(
        graph.edge_index, graph.num_nodes, **rw_kwargs
    ).cpu()

    ratings = graph.edge_attr  # [E, 1]
    graph.edge_attr = torch.cat([ratings, imp.unsqueeze(1)], dim=1)  # [E, 2]
    return graph

class GraphSAGE(nn.Module):
    def __init__(self, in_dim: int = 5, hid: int = 128, out_dim: int = 64):
        super().__init__()
        self.c1 = SAGEConv(in_dim, hid)
        self.c2 = SAGEConv(hid, out_dim)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.relu(self.c1(x, edge_index))
        x = F.dropout(x, 0.3, self.training)
        return self.c2(x, edge_index)




def bpr_loss(z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """BPR loss base con negative sampling random."""
    src, pos = edge_index
    neg = pos[torch.randperm(pos.size(0), device=pos.device)]
    return -F.logsigmoid(
        (z[src] * z[pos]).sum(1) - (z[src] * z[neg]).sum(1)
    ).mean()

def _strip_meta(g: Data) -> Data:
    """Restituisce una copia con soli attributi tensor (x, edge_index, edge_attr)."""
    return Data(
        x=g.x,
        edge_index=g.edge_index,
        edge_attr=g.edge_attr,
        num_nodes=g.num_nodes,
    )

def hard_negative_bpr_loss(
    z: torch.Tensor,
    edge_index: torch.Tensor,
    pool_size: int = 10,
    hard_ratio: float = 0.5,
) -> torch.Tensor:

    src, pos = edge_index
    E = src.size(0)

    # ── Easy negatives (random) ──────────────────────────────
    neg_easy = pos[torch.randperm(E, device=pos.device)]

    # ── Hard negatives (pool → argmax score) ─────────────────
    pool_idx = torch.randint(0, E, (E * pool_size,), device=pos.device)
    neg_pool = pos[pool_idx].view(E, pool_size)                   # [E, pool]

    with torch.no_grad():
        # Score di ogni candidato con il nodo sorgente
        # z[src]: [E, D]  z[neg_pool]: [E, pool, D]
        scores = (z[src].unsqueeze(1) * z[neg_pool]).sum(-1)      # [E, pool]
        hardest = scores.argmax(dim=1)                            # [E]

    neg_hard = neg_pool[torch.arange(E, device=pos.device), hardest]

    # ── Mix: primi n_hard indici → hard, resto → easy ────────
    n_hard = int(E * hard_ratio)
    mask   = torch.arange(E, device=pos.device) < n_hard
    neg    = torch.where(mask, neg_hard, neg_easy)

    pos_score = (z[src] * z[pos]).sum(1)
    neg_score = (z[src] * z[neg]).sum(1)
    return -F.logsigmoid(pos_score - neg_score).mean()

# ============================================================
# TRAIN MULTI-GRAPH  (full-graph, originale)
# ============================================================

def train(
    model: nn.Module,
    graphs: List[Data],
    epochs: int = 100,
    lr: float = 1e-3,
) -> nn.Module:
    """Full-graph training (per grafi piccoli). Invariato dall'originale."""
    model  = model.to(device)
    graphs = [g.to(device) for g in graphs]
    opt    = torch.optim.Adam(model.parameters(), lr=lr)

    for ep in range(epochs):
        model.train()
        total = 0.0
        for g in graphs:
            opt.zero_grad()
            z    = model(g.x, g.edge_index)
            loss = bpr_loss(z, g.edge_index)
            loss.backward()
            opt.step()
            total += loss.item()
        if ep % 10 == 0:
            print(f"Epoch {ep:3d} | loss {total/len(graphs):.4f}")

    return model

# ============================================================
# TRAIN MINI-BATCH  (PinSAGE v2, scalabile a 9M edges)
# ============================================================

def train_minibatch(
    model: nn.Module,
    graphs: List[Data],
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 2048,
    num_neighbors: Tuple[int, ...] = (15, 10),
    use_amp: bool = True,
    num_workers: int = 4,
    hard_neg_pool: int = 10,
    hard_neg_ratio: float = 0.5,
    weight_decay: float = 1e-5,
) -> nn.Module:

    model = model.to(device)
    use_amp = use_amp and (device.type == "cuda")

    # ── Crea un loader per ogni grafo ────────────────────────
    loaders = [
        NeighborLoader(
            _strip_meta(g),
            num_neighbors=list(num_neighbors),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )
        for g in graphs
    ]

    opt = torch.optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs, eta_min=lr * 0.01
    )
    scaler = GradScaler(**_SCALER_KWARGS, enabled=use_amp)

    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches  = 0

        for loader in loaders:
            for batch in loader:
                batch = batch.to(device)

                # Importance weight: edge_attr[:, 1] se disponibile
                ew: Optional[torch.Tensor] = None
                if batch.edge_attr is not None and batch.edge_attr.size(1) >= 2:
                    ew = batch.edge_attr[:, 1]

                opt.zero_grad(set_to_none=True)

                with autocast(**_AMP_DTYPE, enabled=use_amp):
                    z    = model(batch.x, batch.edge_index, ew)
                    loss = hard_negative_bpr_loss(
                        z, batch.edge_index,
                        pool_size=hard_neg_pool,
                        hard_ratio=hard_neg_ratio,
                    )

                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()

                total_loss += loss.item()
                n_batches  += 1

        scheduler.step()

        if ep % 10 == 0:
            lr_now = scheduler.get_last_lr()[0]
            print(f"Epoch {ep:3d} | loss {total_loss/max(n_batches,1):.4f}"
                  f" | lr {lr_now:.5f}")

    return model

# ============================================================
# EXPORT EMBEDDINGS
# ============================================================

def _get_edge_weight(g: Data) -> Optional[torch.Tensor]:
    """Estrae importance weight da edge_attr se presente."""
    if g.edge_attr is not None and g.edge_attr.size(1) >= 2:
        return g.edge_attr[:, 1]
    return None



def inference(
        dataset_name: str,
        model_class: type,
        weight_path: str,
        df: pd.DataFrame,
        batch_size: int = 4096,
        num_neighbors: List[int] = [-1, -1],
        use_importance: bool = True,
        num_workers: int = 4,
        **rw_kwargs
) -> dict:
    graph = load_graph_from_df(df)
    graph.x = extract_node_features(graph)

    model = model_class()
    model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
    model = model.to(device).eval()

    loader = NeighborLoader(
        _strip_meta(graph),
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    out_dim = model.c2.out_channels
    z_all = torch.zeros(graph.num_nodes, out_dim, dtype=torch.float32, device="cpu")

    with torch.no_grad():
        g = graph.to(device)
        ew = _get_edge_weight(g)
        z_all = model(g.x, g.edge_index, ew).cpu()

        # for batch in loader:
        #     batch = batch.to(device)
        #     ew = batch.edge_attr[:, 1] if batch.edge_attr.size(1) >= 2 else None
        #     z_batch = model(batch.x, batch.edge_index, ew).cpu()
        #
        #     seed_idx = batch.n_id[:batch.batch_size]
        #     z_all[seed_idx] = z_batch[:batch.batch_size]

    ckpt = {
        "node_embeddings": z_all,
        "node_ids": graph.node_ids,
        "meta": graph.meta,
        "dataset_name": dataset_name,
    }

    out_dir = f"{BASE_DIR}/node_embeddings/"
    os.makedirs(out_dir, exist_ok=True)
    path = f"{out_dir}/node_emb_inference_{dataset_name}.pt"
    torch.save(ckpt, path)

    return ckpt


# ============================================================
# BATCH INFERENCE MULTI-GRAFO - PER GRAF GRANDE/NUOVI
# ============================================================
def batch_inference_multi_graph(
        model_path: str,
        model_class: type,
        dfs: List[pd.DataFrame],
        dataset_names: List[str],
        batch_size: int = 8192,
        num_neighbors: List[int] = [-1, -1],
        use_importance: bool = True,
        num_workers: int = 8,
        **rw_kwargs
) -> List[dict]:
    """Inference su MULTIPLI grafi nuovi/grandi."""
    model = model_class()
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device).eval()

    results = []
    for df, name in zip(dfs, dataset_names):
        print(f"🔄 Inference {name} ({df.shape[0]:,} edges)")
        emb = inference(name, model_class, model_path, df, batch_size,
                        num_neighbors, use_importance, num_workers, **rw_kwargs)
        results.append(emb)

    return results



# ============================================================
# PIPELINE
# ============================================================

def run_training(dfs: List[pd.DataFrame], use_pinsage: bool = False) -> nn.Module:
    """Pipeline originale full-graph (GraphSAGE o PinSAGE base)."""
    graphs = [load_graph_from_df(df) for df in dfs]
    for g in graphs:
        g.x = extract_node_features(g)

    model = GraphSAGE()
    model = train(model, graphs, epochs=200)

    os.makedirs("./weights", exist_ok=True)
    torch.save(model.state_dict(), "./weights/model.pt")
    export_embeddings(model, graphs, "node_embeddings",
                      "pinsage" if use_pinsage else "sage")
    return model



def run_training_sage_v2(dataset_names, dfs, epochs=100, lr=1e-3, **kwargs):
    graphs = [load_graph_from_df(df) for df in dfs]
    for g in graphs:
        g.x = extract_node_features(g)

    model = GraphSAGE()
    model = train_minibatch(model, graphs, epochs=epochs, lr=lr, **kwargs)

    # salva i pesi del modello condiviso
    torch.save(model.state_dict(), f"{BASE_DIR}/weights/all_SAGE_finale.pt")

    # salva embedding separati per dataset con nome corretto
    export_embeddings(model, graphs, dataset_names, f"{BASE_DIR}/node_embeddings")
    return model


def export_embeddings(model, graphs, dataset_names, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()

    with torch.no_grad():
        for g, name in zip(graphs, dataset_names):
            g = g.to(device)
            ew = _get_edge_weight(g)
            z = model(g.x, g.edge_index, ew).cpu()

            path = f"{out_dir}/node_emb_{name}_finale.pt"
            torch.save({
                "node_embeddings": z,
                "node_ids": g.node_ids,
                "meta": g.meta,
            }, path)
            print(f"Saved {path} | shape={z.shape}")


# ============================================================
# MAIN
# ============================================================

def run_graphsage(infere=False):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PARENT_DIR = os.path.dirname(BASE_DIR)

    m = 'graphSAGE'
    all_graphs, dataset_names = [],[]
    #,"Toys_and_Games","Office_Products"
    passive = ["Toys_and_Games","Office_Products","Pet_Supplies"]
    online = ["Books","Sports_and_Outdoors","Beauty_and_Personal_Care"]
    if not infer:
        for dataset in passive:
            print(f"dataset: {dataset}")
            df_train = pd.read_csv(f"{PARENT_DIR}/data/noisy/{dataset}_5/dataset_dirty_new.csv")[['user_id','item_id','rating']]
            df_vali  = pd.read_csv(f"{PARENT_DIR}/data/original/split/{dataset}/vali_{dataset}_5.csv")[['user_id','item_id','rating']]
            df_test  = pd.read_csv(f"{PARENT_DIR}/data/original/split/{dataset}/test_{dataset}_5.csv")[['user_id','item_id','rating']]
            df = pd.concat([df_train, df_vali, df_test])
            all_graphs.append(df)
            dataset_names.append(dataset)

        model = run_training_sage_v2(
            dataset_names, all_graphs,
            epochs=100,
            batch_size=2048,
            num_neighbors=(15, 10),  # default (15,10) → subgraph enorme, riducilo
            num_workers=0,  # dati in RAM
            use_amp=True,  # fondamentale con 2M nodi
        )
    print("Inference")
    if infere:
        for dataset in online:
            print(f"dataset: {dataset}")
            df_train = pd.read_csv(f"{PARENT_DIR}/data/noisy/{dataset}_5/dataset_dirty_new.csv")[['user_id','item_id','rating']]
            df_vali  = pd.read_csv(f"{PARENT_DIR}/data/original/split/{dataset}/vali_{dataset}_5.csv")[['user_id','item_id','rating']]
            df_test  = pd.read_csv(f"{PARENT_DIR}/data/original/split/{dataset}/test_{dataset}_5.csv")[['user_id','item_id','rating']]
            df = pd.concat([df_train, df_vali, df_test])
            all_graphs.append(df)
            dataset_names.append(dataset)
            emb = inference(
                dataset,
                GraphSAGE,
                f"{BASE_DIR}/weights/all_SAGE_finale.pt",
                df,
                use_importance=False,
            )



