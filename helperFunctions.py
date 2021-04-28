

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVR
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
    # normalized=(df-df.min())/(df.max()-df.min())
    # return normalized
    df_cpy = df.copy()
    df_cpy = StandardScaler().fit_transform(df_cpy)
    df_cpy = pd.DataFrame(df_cpy, index=df.index, columns=df.columns)
    return df_cpy

def one_hot_encoded(df, columns):
    df_cpy = df.copy()

    for column in columns:
        one_hot = pd.get_dummies(df[column])
        df_cpy = df_cpy.drop(column, axis = 1)
        df_cpy = df_cpy.join(one_hot)
    # for column in columns:
    #     df

    return df_cpy


def generate_train_test(df, xcolumns, ycolumn, encoding_type: encodingTypes=encodingTypes.one_hot_encoding, normalized=False, test_size=0.2, pca_n=None):
    df_modified = df.dropna(how='any', subset=[column for column in xcolumns]) # drop rows where any of the chosen features are nan
    df_modified = df_modified[[column for column in xcolumns]]
    
    str_cols = list(set(get_str_columns(df)).intersection(set(xcolumns)))
    if len(str_cols) > 0:
        if encoding_type == encodingTypes.categorization:
            df_modified = categorized_df(df_modified, columns=str_cols)
        else: # one hot
            df_modified = one_hot_encoded(df_modified, columns=str_cols)
    
    # if normalized:
    #     df_modified = normalized_df(df_modified)
    
    if pca_n is not None and pca_n > 0 and pca_n < len(xcolumns):
        pca = PCA(n_components=pca_n)
        pca.fit(df_modified)
        columns = ['pca_%i' % i for i in range(pca_n)]
        df_modified = pd.DataFrame(pca.transform(df_modified), columns=columns, index=df_modified.index)

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

def plot_metrics(df, xcolumns, ycolumn, encoding_type: encodingTypes = encodingTypes.one_hot_encoding, normalized=False, test_size=0.2, runtimes=5, title_suffix = ''):
    mse_list = []
    r2_list = []
    for i in range(runtimes):
        X_train, X_test, Y_train, Y_test = generate_train_test(df, xcolumns=xcolumns, ycolumn=ycolumn, encoding_type=encoding_type, normalized=normalized, test_size=test_size, pca_n=None)
        model = RandomForestRegressor(n_estimators=50, max_features=len(xcolumns))
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
    plot_graphs([x_plots, x_plots], [r2_list, mse_list], labels=['R2_Score ' + title_suffix, 'MSE ' + title_suffix], titles=['R2_Score ' + title_suffix, 'MSE ' + title_suffix])


# class SVMRegressor:

#     def __init__(self, df, xcolumns, ycolumn, encoding_type: encodingTypes=encodingTypes.one_hot_encoding, normalized=False, test_size=0.2):
#         # self.df = df
#         df_modified = df.dropna(how='any', subset=[column for column in xcolumns]) # drop rows where any of the chosen features are nan
#         df_modified = df_modified[[column for column in xcolumns]]
        
#         str_cols = list(set(get_str_columns(df)).intersection(set(xcolumns)))
#         if len(str_cols) > 0:
#             if encoding_type == encodingTypes.categorization:
#                 df_modified = categorized_df(df_modified, columns=str_cols)
#             else: # one hot
#                 df_modified = one_hot_encoded(df_modified, columns=str_cols)
        
#         if normalized:
#             self.sc_x = StandardScaler()
#             self.sc_y = StandardScaler()
#             df_x_scaled = self.sc_x.fit_transform(df_modified)
#             df_x_scaled = pd.DataFrame(df_x_scaled, index=df_modified.index, columns=df_modified.columns)
#             df_y_scaled = self.sc_y.fit_transform(df[ycolumn].to_numpy().reshape(-1, 1))
#             df_y_scaled = pd.DataFrame(df_y_scaled, index=df[ycolumn].index, columns=['proppantPerFoot'])
#             df_modified = df_x_scaled.join(df_y_scaled)
#         else:
#             self.sc_x = None
#             self.sc_y = None
#             df_modified = df_modified.join(df[ycolumn])

#         df_train, df_test = train_test_split(df_modified, test_size=test_size)
#         self.Y_train = df_train.pop(ycolumn)
#         self.Y_test = df_test.pop(ycolumn)
#         self.X_train = df_train
#         self.X_test = df_test

#         # print('X train = ')
#         # print(self.X_train.head(10))

#         # print('Y train = ')
#         # print(self.Y_train.head(10))

#         # print("X train size = {}, X test size = {}, Y train size = {}, Y test size = {}".format(len(self.X_train), len(self.X_test), len(self.X_train), len(self.X_test)))

#     def train_model(self):
#         self.regressor = SVR(kernel='rbf')
#         print('Fitting...')
#         self.regressor.fit(self.X_train, self.Y_train)
#         print("Done fitting")

#     def model_prediction(self):
#         print('Predicting...')
#         y_pred = self.regressor.predict(self.X_test)

#         print('Inverse Prediction...')
#         y_pred = self.sc_y.inverse_transform(y_pred)

#         print('Successfully done')
#         return self.Y_test, y_pred

