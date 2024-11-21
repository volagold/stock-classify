import json
import random
import numpy as np
import pandas as pd

# ------------------
# features
# ------------------
def f0(s: pd.Series):
    return s.size

def f1(s: pd.Series):
    """
    pos of max in array (%), small -> p(1) +
    """
    return s.idxmax() / s.size

def f2(s: pd.Series):
    """
    last avg / prev. max (%), small -> p(1) +
    """
    baseline = s[0] if s.size <= 10 else s[:-10].max()
    f = s[-10:].mean() / baseline
    return f

def f3(s: pd.Series):
    """
    size of max decr. subarray (%), large -> p(1) +
    """
    idx, j = 0, 1  # s[0] as base
    while j <= s.size-1:
        if s[j] >= s[idx] * (1 + 0.03)**(j//252):
            j = j + 1
        else:
            idx, j = j, j + 1
            
    return idx / s.size

def random_compare(s: pd.Series):
    """
    p(s[a] < s[b]), small -> p(1) +
    """
    random.seed(1)
    idxmax = s.idxmax()
    if s.size <= 10: return 0  # p=0 for no data
    if idxmax == 0: return 0  # p=0 for first point being max
    if s.size - idxmax <= 2: return 1  # p=1 for last(1-3) point being max
    count, trials = 0, 1000
    for n in range(trials):
        a = random.randint(0, idxmax-1)
        b = random.randint(idxmax+1, s.size - 1)
        if s[a] < s[b]:
            count = count + 1
    return count / trials

def f4(s: pd.Series):
    return random_compare(s[s.size//2:])

def f5(s: pd.Series):
    """
    pos of min in array (%), large -> p(1) +
    """
    return s.idxmin() / s.size

def f6(s: pd.Series):
    """
    measure of cumsum function convexity, small -> p(1) +
    """
    if s.size <= 30: return 0
    f = s.cumsum()
    
    tests, w = [], [0.3, 0.7]
    
    for (x, y) in [(0, s.size-1), (s.size//2, s.size-1)]: # 1.global 2.recent
        count, points = 0, 500
        for lam in np.linspace(0, 1, points):
            z = np.floor(lam * x + (1-lam) * y)
            if f[z] < lam * f[x] + (1-lam) * f[y]:
                count = count + 1
        tests.append(count / points)

    return np.dot(tests, w)

# ------------------
# extract and save
# ------------------
def get_df(ticket: str):
    file_loc = f'../data/csv/{ticket}.csv'
    df = pd.read_csv(file_loc)
    df.columns = df.columns.str.lower()
    df['date'] = df['date'].map(lambda x: x.split(' ')[0])
    df['date'] = pd.to_datetime(df['date'], utc=False)
    return df

def get_feature(ticket: str):
    df = get_df(ticket)
    s = df.close
    if s.empty:
        return {}
    return {'name':ticket, 
            'f0':f0(s), 
            'f1':f1(s), 'f2':f2(s), 'f3':f3(s), 'f4':f4(s), 'f5':f5(s), 'f6':f6(s),
           }

if __name__ == '__main__':
    with open('../label.json') as f: label_str = f.read()
    label = json.loads(label_str)

    df1 = pd.DataFrame([get_feature(tk) for tk in label["1"]])
    df0 = pd.DataFrame([get_feature(tk) for tk in label["0"]])
    
    df1['label'] = 1
    df0['label'] = 0

    pd.concat([df1, df0]).to_csv('features_labeled.csv', index=False)
    print('done for labeled data')

    with open('../data/tickets.txt', 'r') as f:
        tickets = [line.strip() for line in f] 
    rest = set(tickets) - (set(label["1"])|set(label["0"]))

    df = pd.DataFrame([get_feature(tk) for tk in rest])
    df.to_csv('features_unlabeled.csv', index=False)
    print('done for unlabeled data')