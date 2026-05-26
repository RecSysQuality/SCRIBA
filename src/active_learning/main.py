import os
from fraudar import *
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(".."))

from defect_injector.configs import *
from defect_injector.injector_implicit import *
from preprocess.splitter import *
from utils import *
from graph_stats import *
import time
from embeddings_generator.graphsage import run_graphsage
from embeddings_generator.embeddings_generator_4entropy import create_defects_embeddings


if __name__ == '__main__':

    # params
    N = 30 # number of rounds
    B = 150_000
    b = B // N # budget per round

    #---------

    # 1. inject noise
    datasets = ['Beauty_and_Personal_Care' ,'Sports_and_Outdoors', 'Books']

    # path of input graphs
    input_dir = f"{PARENT_DIR}/data/original"
    for dataset in datasets:
        print(f"PREPROCESSING {dataset}")
        preprocessing(input_dir, datasets, k=5)
        # generate defects for each dataset
        print(f"INJECTING NOISE {dataset}")
        inject_noise(datasets, k=5, overlap=0.25)

    # 3. extract defects
    detect_defects()

    # 2. create embeddings


    # start active learning
    for round in range(1,N):
        print()


