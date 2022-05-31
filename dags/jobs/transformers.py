import numpy as np
import pandas as pd
import datetime as dt
from sklearn.base import TransformerMixin
from sklearn.cluster import KMeans


class TypeRecast(TransformerMixin):
    """ Recast selected field to another type """
    def __init__(self, field, dtype):
        """
        :param field: target field
        :param dtype: target type
        """
        self.field = field
        self.dtype = dtype
        self.parser = None

    def fit(self, X, y=None, **fit_params):
        if self.dtype == 'datetime':
            if X['id'].dtype == 'int':
                self.parser = dt.datetime.fromtimestamp
            elif X['id'].dtype == 'str':
                self.parser = dt.datetime.fromisoformat
        return self

    def transform(self, X, y=None):
        df = X.copy()
        if self.field in df.columns:
            if self.dtype == 'datetime':
                values = np.vectorize(self.parser)(df[self.field])
            else:
                values = df[self.field].astype(self.dtype)
            df[self.field] = values
        return df


class ColumnsCorrector(TransformerMixin):
    """ Drop features, add missing filled with given value, order correction """
    def __init__(self, mode, cols, fill_value=0):
        """
        :param mode: 'drop' or 'keep'
        :param cols: columns to process
        :param fill_value: value for filling in missing columns
        """
        self.cols = cols
        self.mode = mode
        self.fill_value = fill_value
        self.required_order = None

    def fit(self, X, y=None, **fit_params):
        self.cols = [col for col in self.cols if col in X.columns]      # leave only the existing columns to be processed
        if self.mode == 'keep':
            self.required_order = [col for col in X.columns if col in self.cols]
        else:
            self.required_order = [col for col in X.columns if col not in self.cols]
        return self

    def transform(self, X):
        df = X.copy()
        absenting = list(set(X.columns) ^ set(self.required_order))
        df[absenting] = self.fill_value
        return df[self.required_order]


class TimeDifference(TransformerMixin):
    """ Calculate days difference between two given timestamps """
    def __init__(self, time_1, time_2):
        self.time_1 = time_1
        self.time_2 = time_2

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X):
        df = X.copy()        
        df['time_diff'] = (df[self.time_2].astype('datetime64[s]').dt.date - df[self.time_1].astype('datetime64[s]').dt.date).dt.days
        return df


class Clusterer(TransformerMixin):
    def __init__(self, columns, **params):
        self.columns = columns
        self.__kmeans = KMeans(**params)

    def fit(self, X, y=None, **fit_params):        
        self.__kmeans.fit(X[self.columns], y)
        return self
    
    def transform(self, X):
        df = X.copy()
        df['cluster'] = self.__kmeans.predict(X[self.columns])
        return df


class Merger(TransformerMixin):
    def __init__(self, features, method='backward', fillna='mean'):
        self.features = features.sort_values(by='buy_time').copy()
        self.method = method
        self.fillna = fillna

    def fit(self, X, y=None, **fit_params):
        return self
    
    def transform(self, X):
        df = pd.merge_asof(X.sort_values(by='buy_time').rename(columns={'buy_time': 'train_time'}), 
                               self.features.rename(columns={'buy_time': 'feats_time'}),
                               by='id', left_on='train_time', right_on='feats_time', direction='backward')
        if self.fillna == 'nearest':
            nan_rows = df.isna().any(axis=1)
            nan_columns = df.columns[df.isna().any()].to_list()

            df[nan_rows] = pd.merge_asof(df[nan_rows].drop(nan_columns, axis=1), 
                                            self.features.rename(columns={'buy_time': 'feats_time'}), 
                                            by='id', left_on='train_time', right_on='feats_time', direction='nearest').values
        else:
            df.fillna(df.mean(), inplace=True)
        
        return df
        