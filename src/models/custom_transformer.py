from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder


class CustomOneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, C=1, handle_unknown='ignore'):
        self.C = C
        self.handle_unknown = handle_unknown
        self.l1_reg = LogisticRegression(penalty='l1', solver='saga', C=C, random_state=42)
        self.one_hot = OneHotEncoder(sparse=False, handle_unknown=handle_unknown)
        self.mask = None

    def fit(self, X, y):
        self.one_hot.fit(X)
        pred = self.one_hot.transform(X)
        self.l1_reg.fit(pred, y.values.reshape(-1))
        self.mask = (self.l1_reg.coef_ != 0).reshape(-1)
        return self

    def transform(self, X, *_):
        if self.mask is None:
            raise Exception('CustomColumnTransformer')
        data_copy = X.copy()
        data_copy = self.one_hot.transform(data_copy)
        return data_copy[:, self.mask]