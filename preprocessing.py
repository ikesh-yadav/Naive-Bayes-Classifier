import numpy as np
import pandas as pd
import csv
from sklearn.base import TransformerMixin
from collections import Counter 
  
def most_frequent(List): 
    occurence_count = Counter(List) 
    return occurence_count.most_common(1)[0][0] 


def categorical_impute(X):
    for i in range(len(X)):
        for j in range(len(X[0])):
            if not (pd.notna(X[i][j])):
                X[i][j]='No'
                
"""
class DataFrameImputer(TransformerMixin):

    def __init__(self):
        Impute missing values.

        Columns of dtype object are imputed with the most frequent value 
        in column.

        Columns of other types are imputed with mean of column.

    def fit(self, X, y=None):

        self.fill = pd.Series([X[c]
            if X[c].dtype == np.dtype('O') else X[c].most_frequent()
            for c in range(len(X))])

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)
"""

data = pd.read_csv('rawData.csv')
data_numerical = data.iloc[:, :-2].values
data_class = data.iloc[:, -1].values
data_categorical = data.iloc[:, -2:].values

from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(data_numerical[:, :])
data_numerical[:, :] = imputer.transform(data_numerical[:, :])

categorical_impute(data_categorical)

with open("output2.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerows(data_categorical)
    
#imputer1 = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
#imputer1 = imputer1.fit(data_categorical[:, :])
#data_categorical[:, :] = imputer1.transform(data_categorical[:, :])


#xt = DataFrameImputer().fit_transform(data_categorical)

    