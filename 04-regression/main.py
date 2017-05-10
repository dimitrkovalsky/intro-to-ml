import pandas
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer

import sys

from sklearn.linear_model import Ridge

sys.path.append("..")
from util import print_answer
from sklearn.feature_extraction import DictVectorizer

def text_transform(text):
    text = text.map(lambda t: t.lower())
    text = text.replace('[^a-zA-Z0-9]', ' ', regex=True)
    return text

data_train = pandas.read_csv('salary-train.csv')

vectorizer = TfidfVectorizer(min_df=5)


data_train['FullDescription'] = data_train['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True)
data_train['FullDescription'] = data_train['FullDescription'].str.lower()

x_train_text = vectorizer.fit_transform(data_train['FullDescription'])

# print(x_train_text)


data_train['LocationNormalized'].fillna('nan', inplace=True)
data_train['ContractTime'].fillna('nan', inplace=True)

enc = DictVectorizer()
X_train_cat = enc.fit_transform(data_train[['LocationNormalized', 'ContractTime']].to_dict('records'))

X_train = hstack([x_train_text, X_train_cat])

y_train = data_train['SalaryNormalized']
model = Ridge(alpha=1)
model.fit(X_train, y_train)

test = pandas.read_csv('salary-test-mini.csv')
X_test_text = vectorizer.transform(text_transform(test['FullDescription']))
X_test_cat = enc.transform(test[['LocationNormalized', 'ContractTime']].to_dict('records'))
X_test = hstack([X_test_text, X_test_cat])

y_test = model.predict(X_test)
print_answer(1, '{:0.2f} {:0.2f}'.format(y_test[0], y_test[1]))