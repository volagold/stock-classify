import json
import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

data = pd.read_csv('features_unlabeled.csv')
data = data.dropna()
X = data.drop(['name'], axis=1)
X['f0'] = MinMaxScaler(feature_range=(0, 1)).fit_transform(X[['f0']])
y = model.predict(X)
data['label'] = y

prediction = {
    "1": data['name'][data['label'] == 1].tolist(),
    "0": data['name'][data['label'] == 0].tolist()
}

with open('prediction.json', 'w') as f:
    json.dump(prediction, f, indent=4)