# Code inspired by @lystdo, @bradleypallen and @act444
# See https://www.kaggle.com/lystdo/lstm-with-word2vec-embeddings
# and https://github.com/bradleypallen/keras-quora-question-pairs
# and https://www.kaggle.com/act444/lb-0-158-xgb-handcrafted-leaky
# Thank you, @lystdo, @bradleypallen and @act444.

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
from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier

from gensim.models import KeyedVectors

# * Variables

# set directories and parameters

BASE_DIR = 'data/'
EMBEDDING_FILE = BASE_DIR + 'GoogleNews-vectors-negative300.bin'
TRAIN_DATA_FILENAME = "vanilla_train"
TRAIN_DATA_FILE = BASE_DIR + TRAIN_DATA_FILENAME + '.csv'
# TRAIN_DATA_FILE = BASE_DIR + 'train_small.csv'
TEST_DATA_FILE = BASE_DIR + 'test.csv'
# TEST_DATA_FILE = BASE_DIR + 'test_small.csv'
WORD_EMBEDDING_MATRIX_FILE = BASE_DIR + \
                             TRAIN_DATA_FILENAME + \
                             '-word_embedding_matrix.npy'

ABHISHEK_TRAIN = 'abhishek/train_features.csv'
ABHISHEK_TEST = 'abhishek/test_features.csv'

MAGIC_TRAIN = 'magic/' + TRAIN_DATA_FILENAME + "-train.csv"
MAGIC_TEST = 'magic/test.csv'

MAGIC_II_TRAIN = 'magic2/train_ic.csv'
MAGIC_II_TEST = 'magic2/test_ic.csv'

CUSTOM_FEATURES_TRAIN = 'custom/' + TRAIN_DATA_FILENAME + "-train.csv"
CUSTOM_FEATURES_TEST = 'custom/test.csv'


# * Constructor

class Quf:
    def __init__(self,
                 train_data_filename=TRAIN_DATA_FILE,
                 test_data_filename=TEST_DATA_FILE):
        self.TRAIN_DATA_FILE = train_data_filename
        self.TEST_DATA_FILE = test_data_filename        
        self.STAMP = 'quf_xgboost'

        # a container for the computed model
        self.model = {}

        # add custom-engineered features

        self.train_custom_features, self.test_custom_features = self._engineered_features()

# * Engineered Features

    def _engineered_features(self):
        # @abhishek's features
        # Thanks to @raddar and @abhishek for the data.
        # See https://www.kaggle.com/c/quora-question-pairs/discussion/31284

        abhishek_train = pd.read_csv(ABHISHEK_TRAIN, encoding = "ISO-8859-1")
        abhishek_test = pd.read_csv(ABHISHEK_TEST, encoding = "ISO-8859-1")

        abhishek_train_features = abhishek_train.ix[:, 9:30]\
                                                .replace([np.inf,-np.inf],0)\

        abhishek_test_features = abhishek_test.ix[:, 9:30]\
                                              .replace([np.inf,-np.inf],0)\

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
        # from kcore_decomposition import KCore_Decomposition
        # kd = KCore_Decomposition(train_data_filename=TRAIN_DATA_FILENAME)
        # kcore_train_features, kcore_test_features = kd.attach_max_kcore()

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
                                    #magic2_train_features,
                                    #kcore_train_features                                    
                                    magic_train_features], axis=1, join='inner').fillna(0)
        test_features = pd.concat([custom_test_features,
                                   abhishek_test_features,
                                   #magic2_test_features,
                                   #kcore_test_features                                   
                                   magic_test_features], axis=1, join='inner').fillna(0)

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

        X_train, X_valid, y_train, y_valid = train_test_split(X_train,
                                                              y_train,
                                                              test_size=0.1,
                                                              random_state=4242)

        #UPDownSampling
        pos_train = X_train[y_train == 1]
        neg_train = X_train[y_train == 0]
        X_train = pd.concat((neg_train, pos_train.iloc[:int(0.8*len(pos_train))], neg_train))
        y_train = np.array([0] * neg_train.shape[0] + [1] * pos_train.iloc[:int(0.8*len(pos_train))].shape[0] + [0] * neg_train.shape[0])
        print(np.mean(y_train))
        del pos_train, neg_train
    
        pos_valid = X_valid[y_valid == 1]
        neg_valid = X_valid[y_valid == 0]
        X_valid = pd.concat((neg_valid, pos_valid.iloc[:int(0.8 * len(pos_valid))], neg_valid))
        y_valid = np.array([0] * neg_valid.shape[0] + [1] * pos_valid.iloc[:int(0.8 * len(pos_valid))].shape[0] + [0] * neg_valid.shape[0])
        print(np.mean(y_valid))
        del pos_valid, neg_valid

        
        ########################################
        ## define the model structure
        ########################################

        params = {}
        params['objective'] = 'binary:logistic'
        params['eval_metric'] = 'logloss'
        params['eta'] = 0.02
        params['max_depth'] = 7
        params['subsample'] = 0.6
        params['base_score'] = 0.2
        
        d_train = xgb.DMatrix(X_train, label=y_train)
        d_valid = xgb.DMatrix(X_valid, label=y_valid)
    
        watchlist = [(d_train, 'train'), (d_valid, 'valid')]
        print("Training the model...")
        bst = xgb.train(params, d_train, 2500, watchlist, early_stopping_rounds=50, verbose_eval=50)
        self.model = bst
        
        print(log_loss(y_valid, bst.predict(d_valid)))
        bst_val_score = log_loss(y_valid, bst.predict(d_valid))
        self.STAMP = str(bst_val_score) + "_" + self.STAMP
        
        bst.save_model(self.STAMP + '.mdl')
        return (bst, bst_val_score)
# * Prediction
    def predict(self):
        bst, bst_val_score = self._model_constructor()

        self.bst_val_score = bst_val_score
        print('Building Test Features')
        
        X_test = self.test_custom_features
        d_test = xgb.DMatrix(X_test)
        print('Start making the submission before fine-tuning...')                
        p_test = bst.predict(d_test)

        df_test = pd.read_csv(TEST_DATA_FILE, encoding="utf-8")
        sub = pd.DataFrame()
        sub['test_id'] = df_test['test_id']
        sub['is_duplicate'] = p_test
        sub.to_csv(self.STAMP+'.csv', index=False)
        print("Plotting feature importance of the model...")
        import seaborn as sns
        sns.set(font_scale = 1.5)
        ax = xgb.plot_importance(bst)
        fig = ax.get_figure()
        fig.set_size_inches(17, 8)
        timestr = time.strftime("%Y-%m-%d-%H%M-")
        fig.savefig(timestr+self.STAMP+"-feature_importance.png")        

        print("Done.")
