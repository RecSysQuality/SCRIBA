import pandas as pd
import time
from datasets import load_dataset
import pandas as pd

def extract_kcore(df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """Estrae il k-core dal dataframe"""

    df.drop_duplicates(inplace=True)
    if k > 1:
        # Iterativamente rimuovi utenti e item con grado < k
        while True:
            print('kcore...')
            initial_len = len(df)
            user_counts = df['user_id'].value_counts()
            item_counts = df['item_id'].value_counts()

            df = df[df['user_id'].isin(user_counts[user_counts >= k].index)]
            df = df[df['item_id'].isin(item_counts[item_counts >= k].index)]

            if len(df) == initial_len:
                break  # Nessuna rimozione, esci

    # self._log_stats(df)
    return df


def download_data(name):
    print('Scaricamento in corso...')
    ds = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_All_Beauty", trust_remote_code=True)

    print(f'Recensioni totali: {len(ds)}')
    print('Conversione in CSV...')
    ds.to_pandas().to_csv(f'{name}.csv', index=False)
    print('Fatto! File salvato: all_beauty_reviews.csv')

def load_jsonl(path: str, k: int = 5) -> pd.DataFrame:
    # Usa chunks per file grandi
    st = time.time()
    chunks = pd.read_json(path, lines=True, chunksize=100_000)
    df = pd.concat(chunks, ignore_index=True)
    end = time.time()
    print(f'file read in {end-st}')
    df = df[['user_id','parent_asin','rating','timestamp']]
    df.rename(columns={"parent_asin": "item_id"}, inplace=True)
    print('ready to extract kcore')
    df = df.drop_duplicates(subset=['user_id', 'item_id','rating','timestamp'])
    df = extract_kcore(df,k)
    print('kcore extracted')
    df.to_csv(path.replace('.jsonl',f'_{k}.csv'), index=False)
    return df





