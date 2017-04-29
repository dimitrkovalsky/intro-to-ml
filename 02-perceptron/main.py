# coding=utf-8
import pandas
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

import sys

sys.path.append("..")
from util import print_answer


df_train = pandas.read_csv('perceptron-train.csv', header=None)
y_train = df_train[0]
X_train = df_train.loc[:, 1:]

df_test = pandas.read_csv('perceptron-test.csv', header=None)
y_test = df_train[0]
X_test = df_train.loc[:, 1:]


model = Perceptron(random_state=241)
model.fit(X_train, y_train)


acc_before = accuracy_score(y_test, model.predict(X_test))


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = Perceptron(random_state=241)
model.fit(X_train_scaled, y_train)
acc_after = accuracy_score(y_test, model.predict(X_test_scaled))


print_answer(1, acc_after - acc_before)