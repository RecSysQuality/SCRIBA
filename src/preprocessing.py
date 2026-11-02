import pandas as pd


import pandas as pd
import numpy as np
import time
import os
import json
import csv

def detect_low_helpfulness_users(df, min_reviews=30, fraud_thresh=0.25):
    """
    Utenti con almeno `min_reviews` interazioni la cui frazione di review
    con helpful_vote > 0 è <= fraud_thresh.

    Returns:
        suspicious: set degli user_id sospetti
        groups_df: DataFrame con group_id, user_id, item_id
    """
    df = df.copy()
    df['is_helpful'] = df['helpful_vote'] > 0

    stats = df.groupby('user_id').agg(
        n_reviews=('is_helpful', 'size'),
        n_helpful=('is_helpful', 'sum')
    )

    stats = stats[stats['n_reviews'] >= min_reviews]
    stats['fraction'] = stats['n_helpful'] / stats['n_reviews']

    suspicious = set(
        stats[stats['fraction'] <= fraud_thresh].index
    )

    # Solo le interazioni degli utenti sospetti
    suspicious_df = df[df['user_id'].isin(suspicious)]

    # Un group_id per ogni user
    user_groups = {
        user_id: group_id
        for group_id, user_id in enumerate(sorted(suspicious), start=1)
    }

    groups_df = suspicious_df[['user_id', 'item_id']].copy()

    groups_df['group_id'] = groups_df['user_id'].map(user_groups)

    groups_df = (
        groups_df[
            ['group_id', 'user_id', 'item_id']
        ]
        .drop_duplicates()
        .sort_values(['group_id', 'item_id'])
        .reset_index(drop=True)
    )

    return suspicious, groups_df
# def detect_low_helpfulness_users(df, min_reviews=30, fraud_thresh=0.25):
#     """
#     Utenti con almeno `min_reviews` interazioni la cui frazione di review
#     con helpful_vote > 0 è <= fraud_thresh.
#     """
#     df = df.copy()
#     df['is_helpful'] = df['helpful_vote'] > 0
#
#     stats = df.groupby('user_id').agg(
#         n_reviews=('is_helpful', 'size'),
#         n_helpful=('is_helpful', 'sum')
#     )
#     stats = stats[stats['n_reviews'] >= min_reviews]
#     stats['fraction'] = stats['n_helpful'] / stats['n_reviews']
#
#     suspicious = set(stats[stats['fraction'] <= fraud_thresh].index)
#     return suspicious


from collections import defaultdict, deque
from collections import defaultdict
import pandas as pd


def detect_burst_users(df, min_users_in_window=20, window_days=30):
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    window = pd.Timedelta(days=window_days)

    # Pre-filtro
    item_counts = df.groupby('item_id')['user_id'].transform('size')
    df = df[item_counts >= min_users_in_window]

    suspicious_users = set()
    groups = []

    group_id = 1

    for item_id, group in df.groupby('item_id', sort=False):
        group = group.sort_values('datetime').reset_index(drop=True)

        timestamps = group['datetime'].to_numpy()
        users = group['user_id'].to_numpy()
        n = len(group)

        counts = defaultdict(int)
        distinct = 0
        left = 0

        # Burst corrente
        current_group_users = set()
        current_burst_end = None

        for right in range(n):
            u = users[right]

            if counts[u] == 0:
                distinct += 1

            counts[u] += 1

            # Restringi la finestra
            while timestamps[right] - timestamps[left] > window:
                lu = users[left]

                counts[lu] -= 1
                if counts[lu] == 0:
                    distinct -= 1

                left += 1

            if distinct >= min_users_in_window:

                window_start = timestamps[left]
                window_end = timestamps[right]

                # Nuovo burst se non esiste un burst corrente
                # oppure la nuova finestra NON si sovrappone al burst precedente
                if (
                    current_burst_end is None
                    or window_start > current_burst_end
                ):
                    # Chiudi il gruppo precedente
                    if current_group_users:
                        for user_id in current_group_users:
                            groups.append({
                                'group_id': group_id,
                                'item_id': item_id,
                                'user_id': user_id
                            })

                        group_id += 1

                    # Nuovo gruppo
                    current_group_users = set()

                # Aggiungi tutti gli utenti della finestra
                current_group_users.update(users[left:right + 1])

                # Estendi la fine del burst
                current_burst_end = window_end

                # Utenti sospetti
                suspicious_users.update(users[left:right + 1])

        # Chiudi l'ultimo burst dell'item
        if current_group_users:
            for user_id in current_group_users:
                groups.append({
                    'group_id': group_id,
                    'item_id': item_id,
                    'user_id': user_id
                })

            group_id += 1

    groups_df = pd.DataFrame(
        groups,
        columns=['group_id', 'item_id', 'user_id']
    )

    return suspicious_users, groups_df
# def detect_burst_users(df, min_users_in_window=20, window_days=30):
#     df = df.copy()
#     df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
#     window = pd.Timedelta(days=window_days)
#
#     # pre-filtro: scarta subito gli item che non possono comunque
#     # raggiungere la soglia (anche con tutte le review nella stessa finestra)
#     item_counts = df.groupby('item_id')['user_id'].transform('size')
#     df = df[item_counts >= min_users_in_window]
#
#     suspicious_users = set()
#
#     for item_id, group in df.groupby('item_id', sort=False):
#         group = group.sort_values('datetime')
#         timestamps = group['datetime'].to_numpy()
#         users = group['user_id'].to_numpy()
#         n = len(group)
#
#         counts = defaultdict(int)
#         window_users = deque()  # utenti attualmente nella finestra, in ordine
#         distinct = 0
#         left = 0
#
#         for right in range(n):
#             u = users[right]
#             if counts[u] == 0:
#                 distinct += 1
#             counts[u] += 1
#             window_users.append(u)
#
#             # contrai la finestra da sinistra finché è entro i window_days
#             while timestamps[right] - timestamps[left] > window:
#                 lu = users[left]
#                 counts[lu] -= 1
#                 if counts[lu] == 0:
#                     distinct -= 1
#                 window_users.popleft()
#                 left += 1
#
#             if distinct >= min_users_in_window:
#                 # aggiungere elementi già presenti in un set è O(1) ammortizzato,
#                 # quindi questo è economico anche se ripetuto
#                 suspicious_users.update(window_users)
#
#     return suspicious_users

def find_suspicious_users(df, min_reviews=30, fraud_thresh=0.25,
                           min_users_in_window=20, window_days=30):
    low_helpful,low_helpful_df = detect_low_helpfulness_users(df, min_reviews=min_reviews, fraud_thresh=fraud_thresh)
    burst,burst_df = detect_burst_users(df, min_users_in_window=min_users_in_window, window_days=window_days)

    suspicious = low_helpful | burst
    print(f"Sospetti per bassa helpfulness: {len(low_helpful)}")
    print(f"Sospetti per burst coordinato: {len(burst)}")
    print(f"Totale sospetti (unione): {len(suspicious)}")
    return low_helpful,low_helpful_df,burst,burst_df,suspicious


def load_yelp_jsonl(path: str) -> pd.DataFrame:
    st = time.time()
    cols = {'user_id': 'user_id', 'business_id': 'item_id',
            'stars': 'rating', 'date': 'timestamp'}

    reader = pd.read_json(path, lines=True, chunksize=100_000,
                          convert_dates=False, dtype=False)
    parts = [chunk[list(cols)].rename(columns=cols) for chunk in reader]
    df = pd.concat(parts, ignore_index=True)
    print(f'file read in {time.time() - st:.1f}s')

    # "2018-07-07 22:09:11" -> Unix timestamp in secondi
    df['timestamp'] = (pd.to_datetime(df['timestamp'])
                       - pd.Timestamp('1970-01-01')) // pd.Timedelta('1s')

    df = df.drop_duplicates(subset=['user_id', 'item_id', 'rating', 'timestamp'])

    out_path = os.path.splitext(path)[0] + '_kcore1.csv'
    df.to_csv(out_path, index=False)
    return df

def load_jsonl(path: str, k: int = 5) -> pd.DataFrame:
    # Usa chunks per file grandi
    st = time.time()
    chunks = pd.read_json(path, lines=True, chunksize=100_000)
    df = pd.concat(chunks, ignore_index=True)
    end = time.time()
    print(f'file read in {end-st}')
    df = df[['user_id','parent_asin','rating','timestamp','helpful_vote']]
    df.rename(columns={"parent_asin": "item_id"}, inplace=True)
    df = df.drop_duplicates(subset=['user_id', 'item_id','rating','timestamp','helpful_vote'])
    df.to_csv(path.replace('.jsonl',f'_kcore1.csv'), index=False)
    return df




def label_users_by_helpfulness(df, min_reviews=50, benign_thresh=0.75, fraud_thresh=0.25):
    df = df.copy()
    df['is_helpful'] = df['helpful_vote'] >= 1

    stats = df.groupby('user_id').agg(
        n_reviews=('is_helpful', 'size'),
        n_helpful=('is_helpful', 'sum')
    )
    stats = stats[stats['n_reviews'] >= min_reviews]
    stats['fraction'] = stats['n_helpful'] / stats['n_reviews']

    def label(f):
        if f >= benign_thresh:
            return 'benign'
        elif f <= fraud_thresh:
            return 'fraudulent'
        else:
            return 'unlabeled'

    stats['label'] = stats['fraction'].apply(label)
    return stats.reset_index()


def gini_index(values):
    """Calcola l'indice di Gini per un array di valori (es. conteggi per item)."""
    values = np.sort(np.array(values, dtype=np.float64))
    n = len(values)
    if n == 0 or values.sum() == 0:
        return 0.0
    cumvals = np.cumsum(values)
    return (2 * np.sum((np.arange(1, n + 1)) * values) - (n + 1) * cumvals[-1]) / (n * cumvals[-1])


def kcore_filter(df, min_user_interactions=5, min_item_users=5, verbose=True):
    """
    Filtra il dataset in modo iterativo finché:
    - ogni utente ha almeno `min_user_interactions` rating
    - ogni item ha almeno `min_item_users` utenti distinti che lo hanno valutato
    """
    df = df.copy()
    it = 0
    while True:
        it += 1
        n_before = len(df)

        # conteggio interazioni per utente e per item
        user_counts = df.groupby('user_id').size()
        item_counts = df.groupby('item_id')['user_id'].nunique()

        valid_users = user_counts[user_counts >= min_user_interactions].index
        valid_items = item_counts[item_counts >= min_item_users].index

        df = df[df['user_id'].isin(valid_users) & df['item_id'].isin(valid_items)]

        n_after = len(df)

        if verbose:
            print(f"Iter {it}: {n_before} -> {n_after} ratings "
                  f"({df['user_id'].nunique()} users, {df['item_id'].nunique()} items)")

        if n_after == n_before or n_after == 0:
            break

    return df.reset_index(drop=True)

def analyse_dataset(path, sep=',', names=None):
    # Se il csv non ha header, passa names=['user_id','item_id','rating','timestamp']
    df = pd.read_csv(path, sep=sep, names=names, header=0 if names is None else None)
    df.columns = [c.lower() for c in df.columns]

    n_users = df['user_id'].nunique()
    n_items = df['item_id'].nunique()
    n_ratings = len(df)

    sparsity = 1 - (n_ratings / (n_users * n_items))

    ratings_per_user = df.groupby('user_id').size()
    ratings_per_item = df.groupby('item_id').size()

    med_r_u = ratings_per_user.median()
    med_r_i = ratings_per_item.median()

    user_p75 = ratings_per_user.quantile(0.75)
    item_p75 = ratings_per_item.quantile(0.75)

    item_gini = gini_index(ratings_per_item.values)

    rating_mean = df['rating'].mean()
    rating_std = df['rating'].std()

    stats = {
        'n_users': n_users,
        'n_items': n_items,
        'n_ratings': n_ratings,
        'sparsity_%': round(sparsity * 100, 4),
        'med_r_u': med_r_u,
        'med_r_i': med_r_i,
        'user_p75': user_p75,
        'item_p75': item_p75,
        'item_gini': round(item_gini, 4),
        'rating_mean': round(rating_mean, 4),
        'rating_std': round(rating_std, 4),
    }
    return stats


def analyse_dataset_from_df(df):
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    n_users = df['user_id'].nunique()
    n_items = df['item_id'].nunique()
    n_ratings = len(df)
    sparsity = 1 - (n_ratings / (n_users * n_items))

    ratings_per_user = df.groupby('user_id').size()
    ratings_per_item = df.groupby('item_id').size()

    return {
        'n_users': n_users,
        'n_items': n_items,
        'n_ratings': n_ratings,
        'sparsity_%': round(sparsity * 100, 4),
        'med_r_u': ratings_per_user.median(),
        'med_r_i': ratings_per_item.median(),
        'user_p75': ratings_per_user.quantile(0.75),
        'item_p75': ratings_per_item.quantile(0.75),
        'item_gini': round(gini_index(ratings_per_item.values), 4),
        'rating_mean': round(df['rating'].mean(), 4),
        'rating_std': round(df['rating'].std(), 4),
    }

def print_latex_row(dataset_name, stats):
    print(f"{dataset_name} & {stats['med_r_u']:.1f} & {stats['med_r_i']:.1f} & "
          f"{stats['user_p75']:.1f} & {stats['item_p75']:.1f} & "
          f"{stats['item_gini']:.3f} & {stats['rating_mean']:.2f} & "
          f"{stats['rating_std']:.2f} \\\\")



def temporal_holdout_split(df, n_val=3, n_test=3, min_interactions=None):
    """
    Split temporale leave-n-out per utente:
    - Le ultime `n_test` interazioni (per timestamp) -> test
    - Le `n_val` precedenti -> validation
    - Il resto -> training

    Se un utente ha troppe poche interazioni per avere sia val che test,
    quell'utente finisce interamente in training (a meno di min_interactions).
    """
    df = df.sort_values(['user_id', 'timestamp'], kind='mergesort').copy()

    train_parts, val_parts, test_parts = [], [], []

    for user_id, group in df.groupby('user_id', sort=False):
        n = len(group)
        needed = n_val + n_test

        if n <= needed:
            # non abbastanza interazioni per fare lo split: tutto in training
            train_parts.append(group)
            continue

        train_parts.append(group.iloc[: n - needed])
        val_parts.append(group.iloc[n - needed : n - n_test])
        test_parts.append(group.iloc[n - n_test :])

    train_df = pd.concat(train_parts).reset_index(drop=True)
    val_df = pd.concat(val_parts).reset_index(drop=True) if val_parts else pd.DataFrame(columns=df.columns)
    test_df = pd.concat(test_parts).reset_index(drop=True) if test_parts else pd.DataFrame(columns=df.columns)

    print(f"Train: {len(train_df)} ratings, {train_df['user_id'].nunique()} users")
    print(f"Val:   {len(val_df)} ratings, {val_df['user_id'].nunique()} users")
    print(f"Test:  {len(test_df)} ratings, {test_df['user_id'].nunique()} users")

    return train_df, val_df, test_df


def create_recbole_data():
    datasets = [
        'Baby_Products',
        'Sports_and_Outdoors',
        'Office_Products',
        'Toys_and_Games',
        'Pet_Supplies',
        'Beauty_and_Personal_Care',
        'Books',
        'yelp'
    ]
    datasets = ['Health_and_Household']
    datasets = ['Electronics']

    for dataset in datasets:
        print('DATASET', dataset)

        base_path = f'data/original/{dataset}'

        path_train = f'{base_path}/{dataset}.train.csv'
        path_vali = f'{base_path}/{dataset}.vali.csv'
        path_test = f'{base_path}/{dataset}.test.csv'

        path_train_rec = f'{base_path}/{dataset}.rec_train.csv'
        path_vali_rec = f'{base_path}/{dataset}.rec_vali.csv'
        path_test_rec = f'{base_path}/{dataset}.rec_test.csv'

        path_vali_rec_act = f'{base_path}/{dataset}.rec_vali_active.csv'
        path_test_rec_act = f'{base_path}/{dataset}.rec_test_active.csv'

        path_vali_rec_lt = f'{base_path}/{dataset}.rec_vali_lt.csv'
        path_test_rec_lt = f'{base_path}/{dataset}.rec_test_lt.csv'

        # ---------------------------------------------------------
        # Load
        # ---------------------------------------------------------
        df_train = pd.read_csv(path_train)
        df_vali = pd.read_csv(path_vali)
        df_test = pd.read_csv(path_test)

        # ---------------------------------------------------------
        # Timestamp: datetime -> Unix timestamp
        # ---------------------------------------------------------
        if dataset != 'yelp':
            for df in [df_train, df_vali, df_test]:
                df['timestamp'] = (
                    pd.to_datetime(df['timestamp'], utc=True)
                    .astype('int64') / 1e9
                )

        # ---------------------------------------------------------
        # Active / Long-tail users
        # Calcolati SOLO sul training set
        # ---------------------------------------------------------
        user_interactions = df_train.groupby('user_id').size()

        p75 = user_interactions.quantile(0.75)
        p25 = user_interactions.quantile(0.25)

        active_users = set(
            user_interactions[user_interactions > p75].index
        )

        long_tail_users = set(
            user_interactions[user_interactions < p25].index
        )

        print(f'  Users: {len(user_interactions)}')
        print(f'  P25: {p25}')
        print(f'  P75: {p75}')
        print(f'  Active users: {len(active_users)}')
        print(f'  Long-tail users: {len(long_tail_users)}')

        # ---------------------------------------------------------
        # RecBole columns
        # ---------------------------------------------------------
        rename_dict = {
            'user_id': 'user_id:token',
            'item_id': 'item_id:token',
            'rating': 'rating:float',
            'timestamp': 'timestamp:float',
        }

        df_train = df_train.rename(columns=rename_dict)
        df_vali = df_vali.rename(columns=rename_dict)
        df_test = df_test.rename(columns=rename_dict)

        columns = [
            'user_id:token',
            'item_id:token',
            'rating:float',
            'timestamp:float'
        ]

        # ---------------------------------------------------------
        # Full train / validation / test
        # ---------------------------------------------------------
        df_train[columns].to_csv(path_train_rec, sep='\t', index=False)
        df_vali[columns].to_csv(path_vali_rec, sep='\t', index=False)
        df_test[columns].to_csv(path_test_rec, sep='\t', index=False)

        # ---------------------------------------------------------
        # Active validation / test
        # ---------------------------------------------------------
        df_vali_active = df_vali[
            df_vali['user_id:token'].isin(active_users)
        ]

        df_test_active = df_test[
            df_test['user_id:token'].isin(active_users)
        ]

        df_vali_active[columns].to_csv(
            path_vali_rec_act,
            sep='\t', index=False
        )

        df_test_active[columns].to_csv(
            path_test_rec_act,
            sep='\t', index=False
        )

        # ---------------------------------------------------------
        # Long-tail validation / test
        # ---------------------------------------------------------
        df_vali_lt = df_vali[
            df_vali['user_id:token'].isin(long_tail_users)
        ]

        df_test_lt = df_test[
            df_test['user_id:token'].isin(long_tail_users)
        ]

        df_vali_lt[columns].to_csv(
            path_vali_rec_lt,
            sep='\t', index=False
        )

        df_test_lt[columns].to_csv(
            path_test_rec_lt,
            sep='\t', index=False
        )

        print(f'  Validation active: {len(df_vali_active)}')
        print(f'  Test active: {len(df_test_active)}')
        print(f'  Validation LT: {len(df_vali_lt)}')
        print(f'  Test LT: {len(df_test_lt)}')


def create_kcore1():
    datasets = ['yelp','Health_and_Household','Beauty_and_Personal_Care','Books']
    datasets = ['Pet_Supplies','Toys_and_Games','Baby_Products']

    for dataset in datasets:
        print('DATASET', dataset, flush=True)
        path = f'data/original/{dataset}/{dataset}.jsonl'
        if dataset =='yelp':
            path = f'data/original/{dataset}/yelp_academic_dataset_review.json'
            df = load_yelp_jsonl(path)
        else:
            df = load_jsonl(path)
        print(len(df))



def create_datasets():
    datasets = ['Baby_Products','Sports_and_Outdoors','Office_Products','Toys_and_Games','Pet_Supplies','Beauty_and_Personal_Care','Books']
    datasets = ['Health_and_Household']
    datasets = ['Electronics']
    datasets = ['Health_and_Household','Beauty_and_Personal_Care','Books']
    datasets = ['Books']
    for dataset in datasets:
        print('DATASET', dataset, flush=True)
        path = f'data/original/{dataset}/{dataset}.jsonl'
        path_train = f'data/original/{dataset}/{dataset}.train.csv'
        path_vali = f'data/original/{dataset}/{dataset}.vali.csv'
        path_test = f'data/original/{dataset}/{dataset}.test.csv'
        try:
            df = load_jsonl(path)
        except Exception as e:
            if dataset =='yelp':
                df = pd.read_csv(path.replace('.jsonl', '_merged2.csv'),sep='\t')
                df = df.rename(columns={'user_id:token':'user_id','item_id:token':'item_id','rating:float':'rating','timestamp:float':'timestamp'})
            else:
                df = pd.read_csv(path.replace('jsonl','csv'))
                print(len(df))

        df = df.rename(columns={'parent_asin':'item_id'})

        df.columns = [c.lower() for c in df.columns]
        stats_prev = analyse_dataset_from_df(df)
        print_latex_row(dataset, stats_prev)
        df_core = kcore_filter(df, min_user_interactions=10, min_item_users=5)
        df_core.to_csv(path.replace('jsonl','csv'),index=False)
        stats = analyse_dataset_from_df(df_core)
        print_latex_row(dataset, stats)
        train_df, val_df, test_df = temporal_holdout_split(df_core, n_val=3, n_test=3)
        # trovo users sospetti sul dataset k-core filtrato

        print(f"Val dopo filtro sospetti: {len(val_df)} ratings")
        print(f"Test dopo filtro sospetti: {len(test_df)} ratings")

        train_df.to_csv(path_train, index=False)
        val_df.to_csv(path_vali, index=False)
        test_df.to_csv(path_test, index=False)


def get_suspicious_users_from_csv():
    datasets = [
        'Baby_Products',
        'Sports_and_Outdoors',
        'Office_Products',
        'Toys_and_Games',
        'Pet_Supplies',
        'Beauty_and_Personal_Care',
        'Books',
        'yelp'
    ]

    for dataset in datasets:
        print('DATASET', dataset)

        path_train = f'data/original/{dataset}/{dataset}.train.csv'
        users_path = f'data/original/{dataset}/{dataset}.sus_users.csv'
        items_path = f'data/original/{dataset}/{dataset}.sus_items.csv'

        df = pd.read_csv(path_train)

        # Keep suspicious interactions
        df_sus = df[
            (df['low_helpful'] == True) |
            (df['burst'] == True)
            ]

        # -------------------------
        # Suspicious users
        # -------------------------
        df_users = df_sus[
            ['user_id', 'low_helpful', 'burst']
        ].drop_duplicates()

        df_users.to_csv(users_path, index=False)

        # -------------------------
        # Suspicious items
        # -------------------------
        df_items = (
            df_sus.groupby('item_id')
            .size()
            .reset_index(name='count')
        )

        # Keep items with >= 20 suspicious interactions
        df_items = df_items[df_items['count'] >= 20]
        df_items.to_csv(items_path, index=False)



if __name__ == '__main__':

    #create_datasets()
    #create_recbole_data()
    create_kcore1()
#get_suspicious_users_from_csv()

