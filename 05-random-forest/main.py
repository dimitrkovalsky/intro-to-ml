import numpy
import matplotlib.pyplot as plt
import pandas
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor

from util import print_answer

data = pandas.read_csv('abalone.csv')
data['Sex'] = data['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))
x = data.loc[:, :'ShellWeight']
y = data['Rings']


kf = KFold(y.size, n_folds=5, shuffle=True, random_state=1)
scores = [0.0]
for i in range(1, 50):
    clf = RandomForestRegressor(n_estimators=i, random_state=1)
    score = numpy.mean(cross_val_score(clf, x, y, cv=kf, scoring='r2'))
    scores.append(score)
    print("Cross validation : {}, estimators : {}", score, i)

for n, score in enumerate(scores):
    if score > 0.52:
        print_answer(1, n)
        break

plt.plot(scores)
plt.xlabel('n_estimators')
plt.ylabel('score')
plt.savefig('estimators_score.png')