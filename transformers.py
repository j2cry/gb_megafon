import numpy as np
import datetime as dt
from sklearn.base import TransformerMixin


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
