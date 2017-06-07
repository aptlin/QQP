# Code inspired by @lystdo, @bradleypallen, @act444 and emanuele@github
# See https://www.kaggle.com/lystdo/lstm-with-word2vec-embeddings
# and https://github.com/bradleypallen/keras-quora-question-pairs
# and https://www.kaggle.com/act444/lb-0-158-xgb-handcrafted-leaky
# and https://github.com/emanuele/kaggle_pbr/blob/master/blend.py
# Thank you, @lystdo, @bradleypallen, @act444 and @emanuele.

# * Libraries
# import packages
import os
from os.path import exists
import time
import re
import csv
import codecs
import numpy as np
import pandas as pd
import pickle

import argparse
import functools
from collections import defaultdict

import xgboost as xgb

from collections import Counter
from sklearn.metrics import log_loss
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier

from gensim.models import KeyedVectors

# * Variables

# set directories and parameters

BASE_DIR = 'data/'
EMBEDDING_FILE = BASE_DIR + 'GoogleNews-vectors-negative300.bin'
TRAIN_DATA_FILENAME = "2017-05-24-1818-shorties_trimmed_train"
TRAIN_DATA_FILE = BASE_DIR + TRAIN_DATA_FILENAME + '.csv'
TEST_DATA_FILE = BASE_DIR + 'test.csv'

VALIDATION_SPLIT = 0.1
FOLDS = 10

ABHISHEK_TRAIN = 'abhishek/train_features.csv'
ABHISHEK_TEST = 'abhishek/test_features.csv'

MAGIC_TRAIN = 'magic/' + TRAIN_DATA_FILENAME + "-train.csv"
MAGIC_TEST = 'magic/test.csv'

MAGIC_II_TRAIN = 'magic2/train_ic.csv'
MAGIC_II_TEST = 'magic2/test_ic.csv'

CUSTOM_FEATURES_TRAIN = 'custom/' + TRAIN_DATA_FILENAME + "-train.csv"
CUSTOM_FEATURES_TEST = 'custom/test.csv'

# * Constructor

class Lamed:
    def __init__(self,
                 train_data_filename=TRAIN_DATA_FILENAME,
                 test_data_filename=TEST_DATA_FILE,
):
        self.TRAIN_DATA_FILENAME = train_data_filename
        self.TRAIN_DATA_FILE = BASE_DIR + self.TRAIN_DATA_FILENAME + '.csv'
        self.TEST_DATA_FILE = test_data_filename


        self.STAMP = 'lamed_xgboost'

        # a container for the computed model
        self.model = {}
        self.clfs = []

        # add custom-engineered features

        self.train_custom_features, self.test_custom_features = self._engineered_features()

# * Engineered Features

    def _engineered_features(self):
        # @abhishek's features
        # Thanks to @raddar and @abhishek for the data.
        # See https://www.kaggle.com/c/quora-question-pairs/discussion/31284

        abhishek_train = pd.read_csv(ABHISHEK_TRAIN, encoding = "ISO-8859-1")
        abhishek_test = pd.read_csv(ABHISHEK_TEST, encoding = "ISO-8859-1")

        abhishek_train_features = abhishek_train.ix[:404176, 9:30]\
                                                .replace([np.inf,-np.inf],0)\
                                                .drop('jaccard_distance', axis=1)\
                                                .drop('euclidean_distance', axis=1)

        abhishek_test_features = abhishek_test.ix[:, 9:30]\
                                              .replace([np.inf,-np.inf],0)\
                                              .drop('jaccard_distance', axis=1)\
                                              .drop('euclidean_distance', axis=1)
        # Krzysztof Dziedzic's magic feature II.
        # Data by @Justfor.
        # See https://www.kaggle.com/justfor/edges/code
        # and https://www.kaggle.com/c/quora-question-pairs/discussion/33287

        magic2_train_features =  pd.read_csv(MAGIC_II_TRAIN,
                                             encoding = "utf-8")


        magic2_test_features =  pd.read_csv(MAGIC_II_TEST,
                                            encoding = "utf-8")

        # @tarobxl kcore feature
        # See https://www.kaggle.com/c/quora-question-pairs/discussion/33371
        from kcore_decomposition import KCore_Decomposition
        kd = KCore_Decomposition(train_data_filename=TRAIN_DATA_FILENAME)
        kcore_train_features, kcore_test_features = kd.attach_max_kcore()

        # @jturkewitz's magic feature
        # See https://www.kaggle.com/jturkewitz/magic-features-0-03-gain
        magic_train_features =  pd.read_csv(MAGIC_TRAIN,
                                             encoding = "utf-8")
        magic_train_features = magic_train_features.ix[:, 3:5]

        magic_test_features =  pd.read_csv(MAGIC_TEST,
                                            encoding = "utf-8")
        magic_test_features = magic_test_features.ix[:, 3:5]

        custom_train_features = pd.read_csv(CUSTOM_FEATURES_TRAIN,
                                            encoding="utf-8")
        custom_test_features = pd.read_csv(CUSTOM_FEATURES_TEST,
                                           encoding="utf-8")

        train_features = pd.concat([custom_train_features,
                                    abhishek_train_features,
                                    magic_train_features,
                                    magic2_train_features,
                                    kcore_train_features], axis=1, join='inner').fillna(0)
        test_features = pd.concat([custom_test_features,
                                   abhishek_test_features,
                                   magic_test_features,
                                   magic2_test_features,
                                   kcore_test_features], axis=1, join='inner').fillna(0)

        return (train_features, test_features)

# * Model Constructor
    def _model_constructor(self):
        ########################################
        ## sample train/validation data
        ########################################
        print("Loading train data...")
        X_train = self.train_custom_features.as_matrix()
        print("X_train shape:", X_train.shape)

        df_train = pd.read_csv(TRAIN_DATA_FILE, encoding="utf-8")

        y_train = np.array(df_train['is_duplicate'].values)
        print("y_train shape:", y_train.shape)

        skf = list(StratifiedKFold(y_train, FOLDS))
        print("SKF partition:", skf)
        self.clfs = [RandomForestClassifier(n_estimators=100,
                                            n_jobs=-1,
                                            criterion='gini',
                                            verbose=30,
                                            class_weight={0: 1.309028344,
                                                          1: 0.472001959}),
                     RandomForestClassifier(n_estimators=100,
                                       n_jobs=-1,
                                       criterion='entropy',
                                       verbose=30,
                                       class_weight={0: 1.309028344,
                                                     1: 0.472001959}),
                     ExtraTreesClassifier(n_estimators=100,
                                          n_jobs=-1,
                                          criterion='gini',
                                          verbose=30,
                                          class_weight={0: 1.309028344,
                                                        1: 0.472001959}),
                     ExtraTreesClassifier(n_estimators=100,
                                          n_jobs=-1,
                                          criterion='entropy',
                                          verbose=30,
                                          class_weight={0: 1.309028344,
                                                        1: 0.472001959}),
                     GradientBoostingClassifier(learning_rate=0.05,
                                                subsample=0.5,
                                                max_depth=6,
                                                n_estimators=50,
                                                verbose=30)]

        dataset_blend_train = np.zeros((X_train.shape[0], len(self.clfs)))

        X_test = self.test_custom_features.as_matrix()

        dataset_blend_test = np.zeros((X_test.shape[0], len(self.clfs)))

        for j, clf in enumerate(self.clfs):
            print("Execution {}: {}".format(j, clf))
            dataset_blend_test_j = np.zeros((X_test.shape[0], len(skf)))
            for i, (train, test) in enumerate(skf):
                print("Fold:", i)
                X = X_train[train]
                y = y_train[train]
                X_val = X_train[test]
                y_val = y_train[test]
                clf.fit(X, y)
                y_submission = clf.predict_proba(X_val)[:, 1]
                print("Current loss:", log_loss(y_val, y_submission))
                dataset_blend_train[test, j] = y_submission
                dataset_blend_test_j[:, i] = clf.predict_proba(X_test)[:, 1]
            dataset_blend_test[:, j] = dataset_blend_test_j.mean(1)

        print("Started Blending...")
        clf = LogisticRegression()
        clf.fit(dataset_blend_train, y_train)
        bst_val_score = log_loss(y_val, y_submission)
        print("Best score: {}".format(bst_val_score))
        return (clf, dataset_blend_test, bst_val_score)



# * Prediction

    def predict(self):
        # bst, bst_val_score = self._model_constructor()
        clf, dataset_blend_test, bst_val_score = self._model_constructor()
        self.model = clf

        y_test = clf.predict_proba(dataset_blend_test)[:, 1]

        self.STAMP = str(bst_val_score) + "_" + self.STAMP

        print("Stretching the predictions linearly...")
        y_test = (y_test - y_test.min()) / (y_test.max() - y_test.min())
        tmp = np.vstack([range(0, len(y_test)), y_test]).T
        print("Saving predictions...")
        np.savetxt(fname=self.STAMP + ".csv", X=tmp, fmt='%d,%0.9f',
                   header='test_id,is_duplicate', comments='')

