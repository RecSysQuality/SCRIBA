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
from embeddings_generator.embeddings_generator_4entropy import create_defects_embeddings, create_defects_embeddings_inference


if __name__ == '__main__':

    # params

    N = 30 # number of rounds
    B = 150_000
    b = B // N # budget per round
    dataset_name = 'Sports_and_Outdoors'

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


    dataset_def = {
            "name": dataset_name,
            "pt_path": f"{BASE_DIR}/node_embeddings/defects_embeddings_{dataset_name}_sage_new_version_2.pt",
        }
    # start active learning
    for round in range(0,N):
        # 3. extract defects
        detect_defects()

        # 2. create embeddings
        run_graphsage(infer=True)
        create_defects_embeddings_inference()

        # 3. predict
        X_filtered,y_filtered,ids_filtered = predict_impact(dataset_def)


        # 4. MAB
        defects_round = MAB_groups(defects=ids_filtered,dataset=dataset_name)
        with open(f"{BASE_DIR}/defects_round_{roudn}.json","a") as g:
            round_obj = {}
            round_obj[str(round)] = defects_round
            # salva i difetti

        # selected defects
        if os.path.exists(f"{BASE_DIR}/defects_round_{roudn}_selected.json"):
            selected_defects = json.load(open(f"{BASE_DIR}/defects_round_{roudn}_selected.json","r"))

            # 5. Reward
            compute_reward()



            # 6. active learning
            active_learning()



            # 7. update graph
            update_graph()



