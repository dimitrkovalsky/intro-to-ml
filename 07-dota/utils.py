import os
import pandas


def save_data(clean_function, X_train, y_train, X_test, directory):
    path = './data/clean/' + directory
    if not os.path.exists(path):
        os.makedirs(path)

    y_train.to_csv(path + '/y_train.csv')
    clean_function(X_train).to_csv(path + '/X_train.csv')
    clean_function(X_test).to_csv(path + '/X_test.csv')

def get_data(directory):
    path = './data/clean/' + directory
    X_train = pandas.read_csv(path + '/X_train.csv', index_col='match_id')
    y_train = pandas.read_csv(path + '/y_train.csv', index_col='match_id')
    X_test = pandas.read_csv(path + '/X_test.csv', index_col='match_id')
    return X_train, y_train['radiant_win'], X_test
