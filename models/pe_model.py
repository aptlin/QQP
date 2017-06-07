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
from sklearn.metrics import log_loss
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from mlxtend.classifier import StackingClassifier

# * Variables

# set directories and parameters

BASE_DIR = 'data/'
EMBEDDING_FILE = BASE_DIR + 'GoogleNews-vectors-negative300.bin'
TRAIN_DATA_FILENAME = "2017-05-24-1818-shorties_trimmed_train"
TRAIN_DATA_FILE = BASE_DIR + TRAIN_DATA_FILENAME + '.csv'
TEST_DATA_FILE = BASE_DIR + 'test.csv'

VALIDATION_SPLIT = 0.1
FOLDS = 5

ABHISHEK_TRAIN = 'abhishek/train_features.csv'
ABHISHEK_TEST = 'abhishek/test_features.csv'

MAGIC_TRAIN = 'magic/' + TRAIN_DATA_FILENAME + "-train.csv"
MAGIC_TEST = 'magic/test.csv'

MAGIC_II_TRAIN = 'magic2/train_ic.csv'
MAGIC_II_TEST = 'magic2/test_ic.csv'

CUSTOM_FEATURES_TRAIN = 'custom/' + TRAIN_DATA_FILENAME + "-train.csv"
CUSTOM_FEATURES_TEST = 'custom/test.csv'

# * Constructor

class Pe:
    def __init__(self,
                 train_data_filename=TRAIN_DATA_FILENAME,
                 test_data_filename=TEST_DATA_FILE,
):
        self.TRAIN_DATA_FILENAME = train_data_filename
        self.TRAIN_DATA_FILE = BASE_DIR + self.TRAIN_DATA_FILENAME + '.csv'
        self.TEST_DATA_FILE = test_data_filename


        self.STAMP = 'pe_scikit'

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
        magic_train_features = pd.read_csv(MAGIC_TRAIN,
                                           encoding = "utf-8")
        magic_train_features = magic_train_features.ix[:, 3:5]

        magic_test_features = pd.read_csv(MAGIC_TEST,
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
        X_train = self.train_custom_features
        
        df_train = pd.read_csv(TRAIN_DATA_FILE, encoding="utf-8")

        y_train = df_train['is_duplicate'].values
        
        #UPDownSampling
        pos_train = X_train[y_train == 1]
        neg_train = X_train[y_train == 0]
        X_train = pd.concat((neg_train, pos_train.iloc[:int(0.8*len(pos_train))], neg_train)).as_matrix()
        y_train = np.array([0] * neg_train.shape[0]
                           + [1] * pos_train.iloc[:int(0.8*len(pos_train))].shape[0]
                           + [0] * neg_train.shape[0])
        print("New duplicate content:", np.mean(y_train))
        del pos_train, neg_train


        ESTIMATORS = 180
        self.clfs = [ # MLPClassifier(hidden_layer_sizes=(300, 200, 100),
                      #               activation="relu",
                      #               learning_rate="adaptive",
                      #               verbose=True,                                   
                      #               batch_size=128,
                      #               max_iter=10,
                      #               tol=0.001,
                      #               early_stopping=True,                                    
                      #               warm_start=False),
                      # MLPClassifier(hidden_layer_sizes=(128, 128, 128),
                      #              activation="relu",
                      #              learning_rate="adaptive",
                      #              verbose=True,                                   
                      #              batch_size=64,
                      #              early_stopping=True),
                      RandomForestClassifier(n_estimators=ESTIMATORS,
                                             n_jobs=-1,
                                             criterion='entropy',
                                             verbose=1),                      
                      RandomForestClassifier(n_estimators=ESTIMATORS,
                                             n_jobs=-1,
                                             criterion='gini',
                                             verbose=1,
                                             warm_start=True),
                      ExtraTreesClassifier(n_estimators=ESTIMATORS,
                                           n_jobs=-1,
                                           criterion='gini',
                                           verbose=1),
                      ExtraTreesClassifier(n_estimators=ESTIMATORS,
                                           n_jobs=-1,
                                           criterion='entropy',
                                           verbose=1,
                                           warm_start=True)]
        lr = LogisticRegression()

        sclf = StackingClassifier(classifiers=self.clfs,
                                  use_probas=True,
                                  average_probas=False,
                                  verbose=2,
                                  meta_classifier=lr)

        sclf.fit(X_train, y_train)

        self.model = sclf
        
        bst_val_score = log_loss(y_train, sclf.predict_proba(X_train)[:, 1])
        print("Model train loss:", bst_val_score)
                
        return (sclf, bst_val_score)



# * Prediction

    def predict(self):
        # bst, bst_val_score = self._model_constructor()
        clf, bst_val_score = self._model_constructor()        

        y_test = clf.predict_proba(self.test_custom_features.as_matrix())[:, 1]

        self.STAMP = str(bst_val_score) + "_" + self.STAMP

        print("Saving the predictions...")

        tmp = np.vstack([range(0, len(y_test)), y_test]).T
        print("Saving predictions...")
        np.savetxt(fname=self.STAMP + ".csv", X=tmp, fmt='%d,%0.9f',
                   header='test_id,is_duplicate', comments='')
