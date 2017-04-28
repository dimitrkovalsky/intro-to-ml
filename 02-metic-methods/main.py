import pandas
from sklearn.model_selection import cross_val_score
from sklearn.cross_validation import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing

data = pandas.read_csv('wine.data', header=None)

import sys

sys.path.append("..")
from util import print_answer

y = data[0]
x = data.loc[:, 1:]

kf = KFold(len(y), n_folds=5, shuffle=True, random_state=42)


def evaluate_accuracy(kf, x, y):
    scores = list()
    k_range = range(1, 51)
    for k in k_range:
        model = KNeighborsClassifier(n_neighbors=k)
        scores.append(cross_val_score(model, x, y, cv=kf, scoring='accuracy'))
    return pandas.DataFrame.from_records(scores, k_range).mean(axis=1).sort_values(ascending=False)


accuracy = evaluate_accuracy(kf, x, y)
top_accuracy = accuracy.head(1)
print_answer(1, top_accuracy.index[0])
print_answer(2, top_accuracy.values[0])
x_scaled = preprocessing.scale(x)

accuracy = evaluate_accuracy(kf, x_scaled, y)
top_accuracy_scaled = accuracy.head(1)
print_answer(3, top_accuracy_scaled.index[0])
print_answer(4, top_accuracy_scaled.values[0])

