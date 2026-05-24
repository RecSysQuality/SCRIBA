This repository contains the code for **SCRIBA: A Budget-Aware Active Learning Framework for Defect Removal in Recommender Systems**, accepted as a full paper at the 2026 ACM CIKM conference.

## Before Getting Started

1. Download the six datasets from the Amazon Reviews 2023 collection:  
   :contentReference[oaicite:0]{index=0}

2. Extract the interaction `.jsonl` files into the following directory:

```text
src/data/original/
```

3. SCRIBA is deployed as a docker container.
From the root directory of the project, run:

```bash
docker build -t scriba .

## Passive learning

## Active learning 
