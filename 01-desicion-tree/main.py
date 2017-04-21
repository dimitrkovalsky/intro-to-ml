import re
import pandas
data = pandas.read_csv('titanic.csv', index_col='PassengerId')

import sys
sys.path.append("..")
from util import print_answer

print("#1")
sex_counts = data['Sex'].value_counts()
print_answer(1, '{} {}'.format(sex_counts['male'], sex_counts['female']))

print("#2")
surv_counts = data['Survived'].value_counts()
surv_percent = 100.0 * surv_counts[1] / surv_counts.sum()
print_answer(2, "{:0.2f}".format(surv_percent))


print("#3")

pclass_counts = data['Pclass'].value_counts()
pclass_percent = 100.0 * pclass_counts[1] / pclass_counts.sum()
print_answer(3, "{:0.2f}".format(pclass_percent))

print("#4")

ages = data['Age'].dropna()
print_answer(4, "{:0.2f} {:0.2f}".format(ages.mean(), ages.median()))

print("#5")
corr = data['SibSp'].corr(data['Parch'])
print_answer(5, "{:0.2f}".format(corr))

print("#6")

def preprocess_name(name):
    s = re.search('^[^,]+, (.*)', name)
    if s:
        name = s.group(1)
    s = re.search('\(([^)]+)\)', name)
    if s:
        name = s.group(1)
    name = re.sub('(Miss\. |Mrs\. |Ms\. )', '', name)
    name = name.split(' ')[0].replace('"', '')

    return name


names = data[data['Sex'] == 'female']['Name'].map(preprocess_name)
name_counts = names.value_counts()
print_answer(6, name_counts.head(1).index.values[0])

