# General Libraries
import time
import numpy as np
import csv
import pandas as pd
from feature_engineering import *
from sklearn.preprocessing import OneHotEncoder

# SK-learn libraries for learning.
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.grid_search import GridSearchCV

# SK-learn libraries for evaluation.
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import classification_report

# SK-learn libraries for feature extraction from text.
from sklearn.feature_extraction.text import *

# StandardScaler - used to scale numerical features
# DictVectorizer - used for all features, but it might be too slow
# OneHotEncoder - used for categorical features
# LabelBinarizer - used for labels only

# 1) Read all data
# 2) Convert categorical features into ints
# 3) Split into 2 ndarrays: numerical and categorical
# 4) For numerical, use StandardScaler to scale correctly
# 5) For categorical, use OneHotEncoder
# 6) Convert back into a single ndarray
# 7) Enable now to convert non-labeled data into
# 8) Keep
# OTHER -> Enable easy addition of new features that are derived: PCA, combining, clustering, etc.

def main():
    '''df = pd.read_csv('mini_train.csv')
    X = np.array(df)
    print X
    print X.shape
    return

    print np.unique(X[:,4]).tolist()
    print X[:,5]
    #print X
    for row in X:
        print type(row)
    #print X[:,:-2] # Leave last 2 elements off
    #print df.head()
    #print list(np.unique(df['Resolution']))
    '''

    f = FeatureEngineering()
    return

    series = pd.Series(data = {'col1': 1, 'col2': True, 'col3': 'Testing', 'col4': 12.3})

    #series = pd.Series(data = {'Name': [1], 'IsCategorical': [True], 'StartIndex': [1], 'EndIndex': [23]})
    df = pd.DataFrame({'Name': [1], 'IsCategorical': [True], 'StartIndex': [1], 'EndIndex': [23]})
    #df.append()
    print series
    print series['col4']
    print df



class FeatureInfo:
    class Feature:
        def __init__(self, name, is_categorical, start_index, end_index = None):
            self.name = name
            self.is_categorical = is_categorical
            self.start_index = start_index
            if end_index == None:
                self.end_index = start_index
            else:
                self.end_index = end_index

    def __init__(self):
        self.features = []

    def add_feature(self, name, is_categorical, start_index, end_index = None):
        #feature = {'name':name, }

        f = self.Feature(name, is_categorical, start_index, end_index)
        self.features += [f]

    def get_row_info(self, list):
        return 'Test'

if __name__ == '__main__':
    main()