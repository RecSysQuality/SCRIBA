

if __name__ == '__main__':

    # params
    N = 30 # number of rounds
    B = 150_000
    b = B // N # budget per round

    #---------

    # 1. inject noise
    datasets = ['Beauty_and_Personal_Care' ,'Sports_and_Outdoors', 'Books']

    # path of input graphs
    input_dir = f"{BASE_DIR}/data/original"
    for dataset in datasets:
        print(f"PREPROCESSING {dataset}")
        preprocessing(input_dir, datasets, k=5)
        # generate defects for each dataset
        print(f"INJECTING NOISE {dataset}")
        inject_noise(datasets, k=5, overlap=0.25)

    # 3. extract defects


    # 2. create embeddings


    # start active learning
    for round in range(1,N):
        print()


