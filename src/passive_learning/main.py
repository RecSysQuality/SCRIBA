import os

import pandas as pd

from defect_injector.configs import cfg_pet_supplies, cfg_baby_products, cfg_books, cfg_toys_games, \
    cfg_sports_outdoors
from defect_injector.injector_implicit import *
from preprocess.splitter import temporal_leave_one_out_split
from utils import *
from graph_stats import *
import time
from embeddings_generator.graphsage import run_graphsage
from embeddings_generator.embeddings_generator_4entropy import create_defects_embeddings


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def print_stats_noise():
    input_dir = f"{BASE_DIR}/data"

    for dataset in datasets:
        print(f"datast {dataset}")
        pathv = f"{input_dir}/original/split/{dataset}/vali_{dataset}_5.csv"
        patht = f"{input_dir}/original/split/{dataset}/test_{dataset}_5.csv"
        pathtr = f"{input_dir}/original/split/{dataset}/train_{dataset}_5.csv"
        pathi = f"{input_dir}/noisy/{dataset}_5/injected_noise.csv"
        pathn = f"{input_dir}/noisy/{dataset}_5/dataset_dirty.csv"
        injected_nodes = pd.read_csv(pathi)
        df_v = pd.read_csv(pathv)
        df_t = pd.read_csv(patht)
        df_tr = pd.read_csv(pathtr)
        noisy = pd.read_csv(pathn)
        len_all = len(df_v) + len(df_t) + len(df_tr) + len(injected_nodes)
        print(f"Users new: {injected_nodes['user_id'].nunique() / noisy['user_id'].nunique()}")
        print(f"items coverage: {injected_nodes['item_id'].nunique() / noisy['item_id'].nunique()}")
        print(f"edges new: {len(injected_nodes) / len_all}")

def preprocessing(input_dir, datasets, k=5):
    """Phase 1: preprocessing: preprocess jsonl file, extract kcore and compute stats"""

    print('entro qua')
    for dataset in datasets:
        st = time.time()
        print(dataset)
        df = load_jsonl(os.path.join(input_dir, f'{dataset}.jsonl'), k=k)
        end = time.time()

        print(f"{dataset} elaborated in {end - st}")
        if k > 1:
            split_data(input_dir=input_dir.split('original')[0], datasets=[dataset])
            print(f"{dataset} splitted")
            graph_stats(df, dataset=dataset, k=k)



def split_data(input_dir,datasets):
    for dataset in datasets:
        print('dataset',dataset)
        path = os.path.join(input_dir,'original', f'{dataset}_5.csv')
        path_out_train = os.path.join(input_dir,f'original/split/{dataset}', f'train_{dataset}_5.csv')
        path_out_vali = os.path.join(input_dir,f'original/split/{dataset}', f'vali_{dataset}_5.csv')
        path_out_test = os.path.join(input_dir,f'original/split/{dataset}', f'test_{dataset}_5.csv')
        os.makedirs(os.path.join(input_dir,f'original/split/{dataset}'),exist_ok=True)

        df = pd.read_csv(path)

        train, val, test = temporal_leave_one_out_split(df)
        train.to_csv(path_out_train,index=False)
        val.to_csv(path_out_vali,index=False)
        test.to_csv(path_out_test,index=False)
        print("Train:", len(train))
        print("Validation:", len(val))
        print("Test:", len(test))

def inject_noise(datasets,k,overlap):
    """Phase 2: inject noise"""


    for dataset in datasets:
        if dataset == 'All_Beauty' and k >1:
            k = 3
            cfg = cfg_all_beauty


        if 'office' in dataset.lower():
            cfg = cfg_baby_products
        elif "pet" in dataset.lower() or 'personal' in dataset.lower():
            cfg = cfg_pet_supplies
        elif 'books' in dataset.lower():
            cfg = cfg_books
        elif 'toys' in dataset.lower():
            cfg = cfg_sports_outdoors
        elif 'sports' in dataset.lower():
            cfg = cfg_sports_outdoors
        else:
            cfg = cfg_all_beauty
        #cfg = cfg_baby_products


        print(f"Elaborating dataset: {dataset}")
        run_injection(
            input_csv=f"{BASE_DIR}/data/original/split/{dataset}/train_{dataset}_5.csv",  # il tuo CSV
            output_dir=f"{BASE_DIR}/data/noisy/{dataset}_{k}/",
            config=cfg,overlap=overlap
        )



def print_stats(datasets):
    # # stats
    for dataset in datasets:
        df = pd.read_csv(f"{BASE_DIR}/data/original/{dataset}_1.csv")
        graph_stats(df, dataset=dataset, k=1)
        print(f'dataset {dataset}', flush=True)
        df = pd.read_csv(f"{BASE_DIR}/data/original/{dataset}_5.csv")
        graph_stats(df, dataset=dataset, k=5)



def train_predictor():
    initial_datasets = [
        {
            "name": "Office_Products",
            "pt_path": f"{BASE_DIR}/node_embeddings/defects_embeddings_Office_Products_sage_new_version_2.pt",
            "json_path": f"{BASE_DIR}/files/Office_Products_labels.jsonl",
        },
        {
            "name": "Toys_and_Games",
            "pt_path": f"{BASE_DIR}/node_embeddings/defects_embeddings_Toys_and_Games_sage_new_version_2.pt",
            "json_path": f"{BASE_DIR}/files/Toys_and_Games_labels.jsonl",
        },

        {
            "name": "Pet_Supplies",
            "pt_path": f"{BASE_DIR}/node_embeddings/defects_embeddings_Pet_Supplies_sage_new_version_2.pt",
            "json_path": f"{BASE_DIR}/files/Pet_Supplies_labels.jsonl",
        },


    ]



    train_regressor(initial_datasets[0:4], initial_datasets[4], path_csv)


if __name__ == '__main__':

    # PASSIVE LEARNING
    # graph loading
    datasets = ['Office_Products' ,'Pet_Supplies', 'Toys_and_Games']

    # path of input graphs
    input_dir = f"{BASE_DIR}/data/original"
    for dataset in datasets:
        print(f"PREPROCESSING {dataset}")
        preprocessing(input_dir, datasets, k=5)
        # generate defects for each dataset
        print(f"INJECTING NOISE {dataset}")
        inject_noise(datasets, k=5, overlap=0.25)

    print(f"GRAPHSAGE NODE EMBEDDINGS -  TRAINING PASSIVE")
    run_graphsage()
    print(f"GRAPHSAGE DEFECTS EMBEDDINGS - NODE AGGREGATION")
    create_defects_embeddings()

    print(f"LOO DEFECTS EVALUATION")



    print(f"TRAIN PREDICTOR")
    train_predictor()















