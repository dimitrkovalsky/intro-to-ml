# coding=utf-8
import pandas
import sklearn.metrics as metrics

import sys
sys.path.append("..")
from util import print_answer

df = pandas.read_csv('classification.csv')

clf_table = {'tp': (1, 1), 'fp': (0, 1), 'fn': (1, 0), 'tn': (0, 0)}
for name, res in clf_table.items():
    clf_table[name] = len(df[(df['true'] == res[0]) & (df['pred'] == res[1])])

print_answer(1, '{tp} {fp} {fn} {tn}'.format(**clf_table))

acc = metrics.accuracy_score(df['true'], df['pred'])

pr = metrics.precision_score(df['true'], df['pred'])

rec = metrics.recall_score(df['true'], df['pred'])

f1 = metrics.f1_score(df['true'], df['pred'])

print_answer(2, '{:0.2f} {:0.2f} {:0.2f} {:0.2f}'.format(acc, pr, rec, f1))

df2 = pandas.read_csv('scores.csv')

scores = {}
for clf in df2.columns[1:]:
    scores[clf] = metrics.roc_auc_score(df2['true'], df2[clf])

print_answer(3, pandas.Series(scores).sort_values(ascending=False).head(1).index[0])

pr_scores = {}
for clf in df2.columns[1:]:
    pr_curve = metrics.precision_recall_curve(df2['true'], df2[clf])
    pr_curve_df = pandas.DataFrame({'precision': pr_curve[0], 'recall': pr_curve[1]})
    pr_scores[clf] = pr_curve_df[pr_curve_df['recall'] >= 0.7]['precision'].max()

print_answer(4, pandas.Series(pr_scores).sort_values(ascending=False).head(1).index[0])
