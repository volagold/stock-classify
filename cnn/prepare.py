# prepare data for deep learning training and prediction
import json
from tqdm import tqdm
import pandas as pd

def get_df(ticket: str):
    file_loc = f'../data/csv/{ticket}.csv'
    df = pd.read_csv(file_loc)
    df.columns = df.columns.str.lower()
    df['date'] = df['date'].map(lambda x: x.split(' ')[0])
    df['date'] = pd.to_datetime(df['date'], utc=False)
    return df

def get_series(ticket: str):
    df = get_df(ticket)
    s, lgth = df.close, 7650
    if s.size >= lgth:
        s = s[-lgth:]
    else:
        s = pd.concat([pd.Series([0.0] * (lgth-s.size)), s.astype(float)], ignore_index=True)
    return {ticket: s.to_list()}

def prepare(tks: set, filename: str):
    data = {}
    for tk in tqdm(tks):
        series = get_series(tk)
        data.update(series)
    df = pd.DataFrame(data)
    df.to_csv(f'{filename}.csv', index=False)


if __name__ == '__main__':
    with open('../label.json') as f:
        label_str = f.read()
        label = json.loads(label_str)
        labeled = set(label["1"] + label["0"])

    with open('../data/tickets.txt', 'r') as f:
        tks_all = set([line.strip() for line in f])
        unlabeled = tks_all - labeled

    prepare(labeled, 'training_data')
    prepare(unlabeled, 'unlabeled_data')