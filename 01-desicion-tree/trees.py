import numpy as np
from sklearn.tree import DecisionTreeClassifier
import pandas
import sys
sys.path.append("..")
from util import print_answer

data = pandas.read_csv('titanic.csv', index_col='PassengerId')

labels = ['Pclass', 'Fare', 'Age', 'Sex']

x = data.loc[:, labels]
x['Sex'] = x['Sex'].map(lambda s: 1 if s == 'male' else 0)

y = data['Survived']

x = x.dropna()
y = y[x.index.values]

clf = DecisionTreeClassifier(random_state=241)
clf.fit(x, y)

importances = pandas.Series(clf.feature_importances_, index=labels)
joined = " ".join(importances.sort_values(ascending=False).head(2).index.values)

print_answer(7, joined)
