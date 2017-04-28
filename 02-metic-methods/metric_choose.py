import pandas
import sklearn
from numpy import linspace
from sklearn.datasets import load_boston
import pandas
from sklearn.model_selection import cross_val_score
from sklearn.cross_validation import KFold
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn import preprocessing

import sys

sys.path.append("..")
from util import print_answer

data = load_boston()
X = data.data
y = data.target

X = sklearn.preprocessing.scale(X)

def evaluate_accuracy(kf, x, y):
    scores = list()
    p_range = linspace(1, 10, 200)
    for p in p_range:
        model = KNeighborsRegressor(p=p, n_neighbors=5, weights='distance')
        scores.append(cross_val_score(model, x, y, cv=kf, scoring='neg_mean_squared_error'))

    return pandas.DataFrame(scores, p_range).max(axis=1).sort_values(ascending=False)

kf = KFold(len(y), n_folds=5, shuffle=True, random_state=42)
accuracy = evaluate_accuracy(kf, X, y)

top_accuracy = accuracy.head(1)
print_answer(5, top_accuracy.index[0])
