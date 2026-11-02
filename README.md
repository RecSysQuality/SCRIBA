# SCRIBA

This repository contains the code for **SCRIBA: A Budget-Aware Active Learning Framework for Defect Removal in Recommender Systems**, presented as a full paper at the 2027 ACM The Web Conference.

## How SCRIBA works

![Pipeline overview](./pipeline1.png)

Recommender systems learn from user interactions, which makes them exposed to noise and manipulation (e.g., fake reviews). Anomaly detectors flag groups of suspicious interactions, called **defects**, but they produce false positives and platforms can only afford to verify a limited number of them. SCRIBA answers the question: *which detector-flagged defects should be removed first to minimise users' exposure to corrupted items under a limited verification budget?*

The pipeline can be paired with any anomaly detector and comprises two main phases: *passive learning* and *active learning*.

### Passive Learning

Performed once, offline, on training graphs that are disjoint from the ones seen at deployment.

**Defect detection and processing**
Anomaly detectors are applied to each training graph to obtain candidate defects (user–item bicliques, item-centred stars, user-centred stars). A GraphSAGE encoder, trained once on all training graphs with a BPR loss, embeds users and items in a shared latent space. Each defect is represented by concatenating the mean-pooled embeddings of its users and of its items.

**Impact attribution**
A LightGCN model is trained once per graph and its input embeddings are stored. The impact of a defect is computed with a leave-one-out (LOO) strategy, without retraining: the stored embeddings are re-propagated on the graph with and without the defect, and the label is the average reduction in the Target NDCG (T-NDCG) of the users affected by it, i.e. the exposure of those users to the defect's target items. Labels are rank-normalised within each graph.

**Predictor training**
The defect representations and their normalised impact labels are used to train a linear regressor (MSE + L2 regularisation) that estimates the impact of a defect without computing the exact LOO. The predictor can then be applied to previously unseen graphs.

### Active Learning

Performed on a new, unseen graph, under a total edge-removal budget `B` and a per-round budget `b`. At each round:

**Detection and representation**
LightGCN is trained on the new graph and the detectors are applied to obtain the candidate set. At every round, candidates are re-encoded with the pretrained GraphSAGE encoder on the current (partially cleaned) graph.

**Impact prediction**
The predictor estimates the impact of each candidate, and the top-ranked ones (up to `m · b` edges) form a pool.

**Bandit-based selection**
A LinUCB contextual bandit selects a batch of defects (up to `b` edges) from the pool, balancing exploitation (high predicted impact) and exploration (uncertain estimates). Selection stops early if no candidate has a positive UCB score.

**Human validation and removal**
The batch is submitted to human moderators, who confirm which defects are actually removed, filtering out false positives.

**Online update**
The exact LOO impact is computed only for the validated defects and used as feedback to update both the bandit and the impact predictor (refitted with ridge regression) before the next round. When the budget is exhausted, LightGCN is retrained from scratch on the cleaned graph.

## Before Getting Started

1. Clone the project.

2. Download the datasets: the Amazon Reviews 2023 subsets ([Amazon Reviews 2023 collection](https://amazon-reviews-2023.github.io/)) and the [Yelp Open Dataset](https://www.yelp.com/dataset).

3. Extract the interaction `.jsonl` files into the following directory:

```text
src/data/original/
```

4. From the root directory of the project, run:

```bash
pip install -r requirements.txt
```

## Extraction

## Training

## Embeddings

## Impact attribution

## Active learning

## Final evaluation

## Ablation study
