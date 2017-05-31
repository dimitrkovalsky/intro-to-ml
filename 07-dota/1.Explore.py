import pandas

pandas.set_option('display.max_columns', None)
df = pandas.read_csv('./data/features.csv', index_col='match_id')

desc = df.describe()
print(desc)