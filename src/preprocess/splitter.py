import pandas as pd


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


def temporal_leave_one_out_split(df):
    # Assicurati che il timestamp sia datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Ordina per utente e tempo
    df = df.sort_values(by=['user_id', 'timestamp'])

    train_list = []
    val_list = []
    test_list = []

    df = df.sort_values(['user_id', 'timestamp'])  # o la colonna temporale corretta

    # indice progressivo per ogni utente
    df['pos'] = df.groupby('user_id').cumcount()
    # dimensione del gruppo
    df['size'] = df.groupby('user_id')['user_id'].transform('size')
    df['item_id'] = df['parent_asin']
    # condizioni
    train_df = df[(df['size'] < 3) | (df['pos'] < df['size'] - 2)]
    val_df = df[(df['size'] >= 3) & (df['pos'] == df['size'] - 2)]
    test_df = df[(df['size'] >= 3) & (df['pos'] == df['size'] - 1)]

    # pulizia
    train_df = train_df.drop(columns=['pos', 'size','parent_asin']).reset_index(drop=True)
    val_df = val_df.drop(columns=['pos', 'size','parent_asin']).reset_index(drop=True)
    test_df = test_df.drop(columns=['pos', 'size','parent_asin']).reset_index(drop=True)

    # Raggruppa per utente
    # for user, group in df.groupby('user_id'):
    #     if len(group) < 3:
    #         # Se troppo pochi dati, metti tutto in train (oppure gestisci diversamente)
    #         train_list.append(group)
    #         continue
    #
    #     # Split
    #     test = group.iloc[-1:]
    #     val = group.iloc[-2:-1]
    #     train = group.iloc[:-2]
    #
    #     train_list.append(train)
    #     val_list.append(val)
    #     test_list.append(test)
    #
    # # Concatena tutto
    # train_df = pd.concat(train_list).reset_index(drop=True)
    # val_df = pd.concat(val_list).reset_index(drop=True) if val_list else pd.DataFrame()
    # test_df = pd.concat(test_list).reset_index(drop=True) if test_list else pd.DataFrame()

    return train_df, val_df, test_df