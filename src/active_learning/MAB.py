
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import torch
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))




import pandas as pd
def load_embeddings(pt_path: str, path_csv:str):
    checkpoint = torch.load(pt_path, map_location="cpu")
    n_edges = []

    X = checkpoint["features"].numpy()
    col_names = checkpoint["col_names"]
    defect_ids = checkpoint["defect_ids"]
    rows,ids,ys,names = [],[],[],[]
    df = pd.read_csv(path_csv)

    for i, did in enumerate(defect_ids):

        rows.append(X[i])
        ys.append(y)
        ids.append(did)
        df_group = df[df['group_id'] == did]
        n_edges.append(len(df_group))


    return X, ids, col_names, n_edges

@dataclass
class Defect:
    defect_id: str
    embedding: np.ndarray
    n_edges: int          # quanti archi rimuove questo difetto
    #n_users: int          # quanti utenti impatta
    metadata: dict = field(default_factory=dict)

@dataclass
class SelectionResult:
    """Output del bandit per un singolo round."""
    selected_defects: list[Defect]
    ucb_scores: dict[str, float]   # defect_id → UCB score
    predicted_impacts: dict[str, float]
    uncertainties: dict[str, float]


def compute_reward(true_impact: float, n_users: int, n_users_total: int) -> float:
    """
    Reward pesato per cardinalità.
    Puoi scegliere la formulazione che preferisci:
    """
    cardinality = n_users / n_users_total      # normalizzata in [0,1]
    # Opzione A — moltiplicativa (semplice)
    return true_impact * cardinality



class SharedLinUCB:
    def __init__(self, d: int, alpha: float = 1.0, lambda_: float = 1.0):
        self.d = d
        self.alpha = alpha
        self.lambda_ = lambda_

        # Parametri del modello — persistono tra i round
        self.A = lambda_ * np.eye(d)   # (d x d) matrice di design
        self.b = np.zeros(d)           # (d,) vettore reward accumulato

        # Statistiche
        self.n_updates = 0
        self.total_reward = 0.0

    # ------------------------------------------------------------------
    # Proprietà derivate
    # ------------------------------------------------------------------

    @property
    def A_inv(self) -> np.ndarray:
        """Inversa di A (ricalcolata on-demand)."""
        return np.linalg.inv(self.A)

    @property
    def theta_hat(self) -> np.ndarray:
        """Stima corrente di theta: theta_hat = A^{-1} b."""
        return self.A_inv @ self.b

    def score(self, defect: Defect) -> tuple[float, float, float]:

        x = defect.embedding
        A_inv = self.A_inv
        theta = A_inv @ self.b

        predicted = float(theta @ x)
        uncertainty = float(self.alpha * np.sqrt(x @ A_inv @ x))
        ucb = predicted + uncertainty

        return ucb, predicted, uncertainty

    def select_defects(
        self,
        defects: list[Defect],
        k: int,
        return_all_scores: bool = False,
    ) -> SelectionResult:

        if not defects:
            return SelectionResult(
                selected_defects=[],
                ucb_scores={},
                predicted_impacts={},
                uncertainties={},
            )

        scores = []
        ucb_scores = {}
        predicted_impacts = {}
        uncertainties = {}

        for defect in defects:
            ucb, pred, unc = self.score(defect)
            scores.append((ucb, defect))
            if return_all_scores or True:
                ucb_scores[defect.defect_id] = ucb
                predicted_impacts[defect.defect_id] = pred
                uncertainties[defect.defect_id] = unc

        # Ordina per UCB score decrescente
        scores.sort(key=lambda t: t[0], reverse=True)
        selected = [d for _, d in scores[:k]]

        return SelectionResult(
            selected_defects=selected,
            ucb_scores=ucb_scores,
            predicted_impacts=predicted_impacts,
            uncertainties=uncertainties,
        )

    def select_defects_by_edges(self, defects, edge_budget):
        scored = []
        for d in defects:
            ucb, pred, unc = self.score(d)
            scored.append((ucb, pred, unc, d))

        scored.sort(reverse=True, key=lambda t: t[0])  # ordina per UCB

        selected, remaining = [], edge_budget
        ucb_scores, predicted_impacts, uncertainties = {}, {}, {}

        for ucb, pred, unc, d in scored:
            # print(d.defect_id)
            ucb_scores[d.defect_id] = ucb
            predicted_impacts[d.defect_id] = pred
            uncertainties[d.defect_id] = unc

            if d.n_edges <= remaining:
                # print('inserito')
                selected.append(d)
                remaining -= d.n_edges

        return SelectionResult(selected, ucb_scores, predicted_impacts, uncertainties)
    # ------------------------------------------------------------------
    # Step 6 — Batch update
    # ------------------------------------------------------------------

    def update(self, defect: Defect, reward: float) -> None:
        """
        Aggiorna A e b con un singolo (x, r) osservato.

        Il reward r misura quanto la rimozione del difetto ha migliorato
        le metriche recsys (es. delta NDCG, delta precision@k).
        """
        x = defect.embedding
        self.A += np.outer(x, x)   # A ← A + x x^T
        self.b += reward * x       # b ← b + r x
        self.n_updates += 1
        self.total_reward += reward

    def batch_update(
        self,
        defects: list[Defect],
        rewards: list[float],
    ) -> None:
        """
        Aggiorna il modello con un batch di (difetto, reward) — Step 6.

        Parametri
        ---------
        defects : list[Defect]
            Difetti rimossi nel round precedente.
        rewards : list[float]
            Reward osservati per ciascun difetto rimosso.
            Esempio: delta_ndcg[i] dopo la rimozione di defects[i].
        """
        assert len(defects) == len(rewards), \
            "defects e rewards devono avere la stessa lunghezza"

        for defect, reward in zip(defects, rewards):
            self.update(defect, reward)

    # ------------------------------------------------------------------
    # Utilità
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reimposta il modello (utile per ablation study)."""
        self.A = self.lambda_ * np.eye(self.d)
        self.b = np.zeros(self.d)
        self.n_updates = 0
        self.total_reward = 0.0

    def state_dict(self) -> dict:
        """Serializza lo stato per checkpoint / batch update remoto."""
        return {
            "A": self.A.tolist(),
            "b": self.b.tolist(),
            "alpha": self.alpha,
            "lambda_": self.lambda_,
            "n_updates": self.n_updates,
            "total_reward": self.total_reward,
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "SharedLinUCB":
        """Ripristina il modello da un checkpoint."""
        d = len(state["b"])
        model = cls(d=d, alpha=state["alpha"], lambda_=state["lambda_"])
        model.A = np.array(state["A"])
        model.b = np.array(state["b"])
        model.n_updates = state["n_updates"]
        model.total_reward = state["total_reward"]
        return model

    def __repr__(self) -> str:
        return (
            f"SharedLinUCB(d={self.d}, alpha={self.alpha}, "
            f"lambda_={self.lambda_}, n_updates={self.n_updates})"
        )



def MAB_group(defects,dataset):

    EMBEDDING_DIM = 64
    ALPHA = 1.0       # esplorazione
    LAMBDA = 1.0      # regularizzazione


    path = f"{BASE_DIR}/node_embeddings/defects_embeddings_{dataset}_sage_new_version_2.pt"
    path_csv = f"{BASE_DIR}/../data/noisy/{dataset}_5/injected_noise_new.csv"


    bandit = SharedLinUCB(d=EMBEDDING_DIM, alpha=ALPHA, lambda_=LAMBDA)
    X, defect_ids, col_names, n_edges = load_embeddings(path,path_csv)
    defects = [
        Defect(defect_id=defect_ids[i], embedding=X[i], n_edges=n_edges[i])
        for i in range(len(defect_ids)) if defect_ids[i] in defects
    ]
    result = bandit.select_defects_by_edges(defects, edge_budget=5000)
    return result.selected_defects,bandit




