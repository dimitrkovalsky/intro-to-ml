import sys

import pandas

sys.path.append("..")
from util import print_answer
import numpy as np
from sklearn import svm, datasets
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

df_train = pandas.read_csv('svm-data.csv', header=None)
y = df_train[0]
x = df_train.loc[:, 1:]

clf = svm.SVC(kernel="rbf", C=100000.0, random_state=241)
clf.fit(x, y)

n_sv = clf.support_
n_sv.sort()
print_answer(1, ' '.join([str(n + 1) for n in n_sv]))
