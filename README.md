# SCRIBA

This repository contains the code for **SCRIBA: A Budget-Aware Active Learning Framework for Defect Removal in Recommender Systems**, accepted as a full paper at the 2026 ACM CIKM conference.

## How SCRIBA works
![Screenshot](./pipeline1.png)

The pipeline comprises two main phases: *passive learning* and *active learning* phases.

### Passive 

## Before Getting Started

1. Clone the progject.

2. Download the six datasets from the Amazon Reviews 2023 collection:  
   :contentReference[oaicite:0]{index=0}

3. Extract the interaction `.jsonl` files into the following directory:

```text
src/data/original/
```

3. From the root directory of the project, run:

```bash
pip install -r requirements.txt
```

## Passive learning
1. To run the passive learning phases of the pipeline, from preprocessing to training of impact predictor, run:
   
```bash
python run src/passive_learning/main.py
```


## Active learning 
1. To run the active learning phases of the pipeline, run:
   
```bash
python run src/active_learning/main.py
```
