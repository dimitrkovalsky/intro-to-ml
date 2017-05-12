import pandas
from numpy import corrcoef
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import sys

from sklearn.linear_model import Ridge

sys.path.append("..")
from util import print_answer
from sklearn.feature_extraction import DictVectorizer

data = pandas.read_csv('close_prices.csv')
x = data.loc[:, 'AXP':]
pca = PCA(n_components=10)
pca.fit(x.values)

variance = 0
i = 0
for v in pca.explained_variance_ratio_:
    i += 1
    variance += v
    if variance >= 0.9:
        break

print_answer(1, i)

df_comp = pandas.DataFrame(pca.transform(x))
comp0 = df_comp[0]

df2 = pandas.read_csv('djia_index.csv')
dji = df2['^DJI']
corr = corrcoef(comp0, dji)
print_answer(2, corr[1, 0])

comp0_w = pandas.Series(pca.components_[0])
comp0_w_top = comp0_w.sort_values(ascending=False).head(1).index[0]
company = x.columns[comp0_w_top]
print_answer(3, company)