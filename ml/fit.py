import fire
import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


def main(model='lr'):
    # load data
    data = pd.read_csv('features_labeled.csv')
    data = data.dropna()
    # separate features (X) and target (y)
    X = data.drop(['name','label'], axis=1)
    X['f0'] = MinMaxScaler(feature_range=(0, 1)).fit_transform(X[['f0']])
    y = data['label']

    # split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # select model
    coef = False
    if model == 'lr':
        model = LogisticRegression(solver='newton-cg')  
        coef = True
    if model == 'tree':
        model = DecisionTreeClassifier()
    if model == 'boost':
        model = HistGradientBoostingClassifier()

    # fit and evaluate
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    confusion_mat = confusion_matrix(y_test, y_pred)

    if coef:
        print(f'coefficients: {model.coef_[0]}')
    print(f'accuracy: {accuracy}')
    print('confusion matrix:\n', confusion_mat)
 
    # save model
    with open('model.pkl','wb') as f:
        pickle.dump(model, f)

if __name__ == '__main__':
    fire.Fire(main)