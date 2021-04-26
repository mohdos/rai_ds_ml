

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt
from enum import Enum

class encodingTypes(Enum):
    one_hot_encoding = 1
    categorization = 2


def categorized_df(df, columns):
    df_cpy = df.copy()
    for col in columns:
        df_cpy[col]=df[col].astype('category').cat.codes

    return df_cpy


def get_str_columns(df):
    cols = []
    i = 0
    for dt in df.dtypes:
        # print(dt)
        if dt == 'object':
            cols.append(df.columns[i])
        i+=1
    
    return cols

def normalized_df(df):
    normalized=(df-df.min())/(df.max()-df.min())
    return normalized

def one_hot_encoded(df, columns):
    df_cpy = df.copy()

    for column in columns:
        one_hot = pd.get_dummies(df[column])
        df_cpy = df_cpy.drop(column, axis = 1)
        df_cpy = df_cpy.join(one_hot)
    # for column in columns:
    #     df

    return df_cpy


def generate_train_test(df, xcolumns, ycolumn, encoding_type: encodingTypes=encodingTypes.one_hot_encoding, normalized=False, test_size=0.2):
    df_modified = df.dropna(how='any', subset=[column for column in xcolumns]) # drop rows where any of the chosen features are nan
    df_modified = df_modified[[column for column in xcolumns]]
    
    str_cols = list(set(get_str_columns(df)).intersection(set(xcolumns)))
    if len(str_cols) > 0:
        if encoding_type == encodingTypes.categorization:
            df_modified = categorized_df(df_modified, columns=str_cols)
        else: # one hot
            df_modified = one_hot_encoded(df_modified, columns=str_cols)
    
    if normalized:
        df_modified = normalized_df(df_modified)
            
    df_train, df_test = train_test_split(df_modified.join(df[ycolumn]), test_size=test_size)
    Y_train = df_train.pop(ycolumn)
    Y_test = df_test.pop(ycolumn)
    X_train = df_train
    X_test = df_test

    return X_train, X_test, Y_train, Y_test

def plot_graphs(x_lists, y_lists, labels=None, titles=None):
    n_ims = len(x_lists)
    if titles is None: 
        titles = ['(%d)' % i for i in range(1,n_ims + 1)]

    if labels is None: 
        labels = ['(%d)' % i for i in range(1,n_ims + 1)]

    fig = plt.figure()
    n = 1
    for x_list, y_list, label, title in zip(x_lists, y_lists, labels, titles):
        a = fig.add_subplot(1,n_ims,n)
        a.plot(x_list, y_list, label=label)
        a.set_title(title)
        a.legend()
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    plt.show()

def plot_metrics(df, xcolumns, ycolumn, encoding_type: encodingTypes = encodingTypes.one_hot_encoding, normalized=False, test_size=0.2, runtimes=5):
    mse_list = []
    r2_list = []
    for i in range(runtimes):
        X_train, X_test, Y_train, Y_test = generate_train_test(df, xcolumns=xcolumns, ycolumn=ycolumn, encoding_type=encoding_type, normalized=normalized, test_size=test_size)
        model = RandomForestRegressor(n_estimators=50, max_features=3)
        model.fit(X_train, Y_train)
        y_pred = model.predict(X_test)
        ytest_list = Y_test.to_list()
        r2 = r2_score(y_true=ytest_list, y_pred=y_pred)
        r2_list.append(r2)
        mse = mean_squared_error(ytest_list,y_pred, squared=False)
        mse_list.append(mse)

    print('Mean squared errors avg =', np.mean(mse_list))
    print('R2 Score avg =', np.mean(r2_list))
    x_plots = list(range(runtimes))
    # plt.plot(x_plots, r2_list, label='R2_Score')
    # plt.plot(x_plots, mse_list, label='MSE List')
    # plt.legend()
    plot_graphs([x_plots, x_plots], [r2_list, mse_list], labels=['R2_Score', 'MSE'], titles=['R2_Score', 'MSE'])
