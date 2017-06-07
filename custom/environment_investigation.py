# -*- coding: utf-8 -*-
# * Libraries

from os.path import exists
import numpy as np
import pandas as pd
from scipy.stats import kendalltau
import functools
import itertools
import time
import string

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

from collections import Counter

from nltk.corpus import wordnet as wn

import pickle

# * Variables

BASE_DIR = 'data/'
TRAIN_DATA_FILENAME = "vanilla_train"
TRAIN_DATA_FILE = BASE_DIR + TRAIN_DATA_FILENAME + '.csv'
TEST_DATA_FILE = BASE_DIR + 'test.csv'
PREPROCESSED = 'preprocessed/'
PREPROCESSED_TRAIN_WORDS = BASE_DIR + PREPROCESSED + TRAIN_DATA_FILENAME + '-' + 'words.pkl'
PREPROCESSED_TRAIN_DF = BASE_DIR + PREPROCESSED + TRAIN_DATA_FILENAME + '-' + 'dataframe.hdf'

PREPROCESSED_TEST_WORDS = BASE_DIR + PREPROCESSED + "test" + '-' + 'words.pkl'
PREPROCESSED_TEST_DF = BASE_DIR + PREPROCESSED + "test" + '-' + 'dataframe.hdf'

start_time = time.time()
# * Constructor

class EnvFeatures:
    def __init__(self,
                 train_data_filename=TRAIN_DATA_FILENAME,
                 test_data_filename=TEST_DATA_FILE):

        self.TRAIN_DATA_FILENAME = train_data_filename
        self.TRAIN_DATA_FILE = BASE_DIR + train_data_filename + '.csv'
        self.TEST_DATA_FILE = test_data_filename
        self.CUSTOM_FEATURES_TRAIN = 'custom/' + \
                                     train_data_filename + \
                                     "-env-train.csv"
        self.CUSTOM_FEATURES_TEST = 'custom/env-test.csv'

    def kendall_tau(self, row):
        try:
            if row['id'] % 10000 == 0:
                elapsed = time.time() - start_time
                print("Processed {:10.0f} questions in {:10.0f} s ".format(row['id'], elapsed))
        except KeyError:
            if row['test_id'] % 10000 == 0:
                elapsed = time.time() - start_time
                print("Processed {:10.0f} questions in {:10.0f} s ".format(row['test_id'], elapsed))

        stemmer = SnowballStemmer('english')

        q1 = []
        q2 = []

        stops = set(stopwords.words("english"))

        for word in row['question1']:
            if word not in stops:
                stem = stemmer.stem(word)
                q1.append(stem)
        for word in row['question2']:
            if word not in stops:
                stem = stemmer.stem(word)
                q2.append(stem)

        q1stemmed = [word for word in q1 if word in q2]
        q2stemmed = [word for word in q2 if word in q1]

        q1ranked = list(range(len(q1stemmed)))
        q2ranked = [q1stemmed.index(word) for word in q2stemmed]

        trim_length = min(len(q1ranked), len(q2ranked))
        q1ranked = q1ranked[:trim_length]
        q2ranked = q2ranked[:trim_length]

        if len(q1ranked) == 0 or len(q2ranked) == 0:
            return 0
        elif len(q1ranked) == 1 or len(q2ranked) == 1:
            return 1
        else:
            return kendalltau(q1ranked, q2ranked)[0]

    def kendall_p_value(self, row):
        try:
            if row['id'] % 10000 == 0:
                elapsed = time.time() - start_time
                print("Processed {:10.0f} questions in {:10.0f} s ".format(row['id'], elapsed))
        except KeyError:
            if row['test_id'] % 10000 == 0:
                elapsed = time.time() - start_time
                print("Processed {:10.0f} questions in {:10.0f} s ".format(row['test_id'], elapsed))

        stemmer = SnowballStemmer('english')

        q1 = []
        q2 = []

        stops = set(stopwords.words("english"))

        for word in row['question1']:
            if word not in stops:
                stem = stemmer.stem(word)
                q1.append(stem)
        for word in row['question2']:
            if word not in stops:
                stem = stemmer.stem(word)
                q2.append(stem)

        q1stemmed = [word for word in q1 if word in q2]
        q2stemmed = [word for word in q2 if word in q1]


        q1ranked = list(range(len(q1stemmed)))
        q2ranked = [q1stemmed.index(word) for word in q2stemmed]

        trim_length = min(len(q1ranked), len(q2ranked))
        q1ranked = q1ranked[:trim_length]
        q2ranked = q2ranked[:trim_length]

        if len(q1ranked) == 0 or len(q2ranked) == 0:
            return 0
        elif len(q1ranked) == 1 or len(q2ranked) == 1:
            return 1                
        else:
            return kendalltau(q1ranked, q2ranked)[1]

    def string_similarity(self, row):
        try:
            if row['id'] % 10000 == 0:
                elapsed = time.time() - start_time
                print("Processed {:10.0f} questions in {:10.0f} s ".format(row['id'], elapsed))
        except KeyError:
            if row['test_id'] % 10000 == 0:
                elapsed = time.time() - start_time
                print("Processed {:10.0f} questions in {:10.0f} s ".format(row['test_id'], elapsed))

        q1 = ''.join(row['question1'])\
               .translate({ord(c): None for c in string.punctuation})
        q2 = ''.join(row['question2'])\
               .translate({ord(c): None for c in string.punctuation})

        from difflib import SequenceMatcher
        sm = SequenceMatcher(None, q1, q2)

        return sm.ratio()

    def subset_count(self, row):
        try:
            if row['id'] % 10000 == 0:
                elapsed = time.time() - start_time
                print("Processed {:10.0f} questions in {:10.0f} s ".format(row['id'], elapsed))
        except KeyError:
            if row['test_id'] % 10000 == 0:
                elapsed = time.time() - start_time
                print("Processed {:10.0f} questions in {:10.0f} s ".format(row['test_id'], elapsed))

        q1 = ''.join(row['question1'])\
               .lower().translate({ord(c): None for c in string.punctuation})
        q2 = ''.join(row['question2'])\
               .lower().translate({ord(c): None for c in string.punctuation})

        q1_words = [word.lower().translate({ord(c): None
                                            for c in string.punctuation})
                    for word in row['question1']]
        q2_words = [word.lower().translate({ord(c): None
                                            for c in string.punctuation})
                    for word in row['question2']]

        def sliding(a, n):
            return (a[i:i+n] for i in range(len(a) - n + 1))

        def substring_count(a, b):
            return sum(s == b for s in sliding(a, len(b)))

        count = 0
        for word in q1_words:
            count += substring_count(q2, word)
        for word in q2_words:
            count += substring_count(q1, word)

        count /= len(q1_words) + len(q2_words)
        return count

    def build_features(self, data):
        X = pd.DataFrame()
        print("Calculating kendall_tau...")
        X['kendall_tau'] = data.apply(self.kendall_tau, axis=1, raw=True)
        print("Calculating kendall_p_value...")
        X['kendall_p_value'] = data.apply(self.kendall_p_value, axis=1, raw=True)
        print("Calculating string_similarity...")
        X['string_similarity'] = data.apply(self.string_similarity, axis=1, raw=True)
        print("Calculating subset_count...")
        X['subset_count'] = data.apply(self.subset_count, axis=1, raw=True)

        return X

    def run(self):
        if exists(self.CUSTOM_FEATURES_TRAIN) and exists(self.CUSTOM_FEATURES_TEST):
            print("Using cached nltk features for {}..."
                  .format(self.TRAIN_DATA_FILENAME))
        else:
            print("Processing the training data set...")


            if exists(self.CUSTOM_FEATURES_TRAIN):
                print("Using cached features for the training data set...")
            else:
                if exists(PREPROCESSED_TRAIN_DF):
                    df_train = pd.read_hdf(PREPROCESSED_TRAIN_DF, 'df_data')
                else:
                    df_train = pd.read_csv(self.TRAIN_DATA_FILE,
                                           encoding="utf-8").fillna(" ")
                    df_train['question1'] = df_train['question1']\
                                            .map(lambda x: str(x).strip().split())
                    df_train['question2'] = df_train['question2']\
                                            .map(lambda x: str(x).strip().split())

                    df_train.to_hdf(PREPROCESSED_TRAIN_DF,'df_data',mode='w')

                print("Computing features for the training data set...")
                X_train = self.build_features(df_train)
                print("Saving...")
                X_train.to_csv(self.CUSTOM_FEATURES_TRAIN, index=False)

            if exists(self.CUSTOM_FEATURES_TEST):
                print("Using cached features the test data set...")
            else:
                print("Processing the testing data set...")
                if exists(PREPROCESSED_TEST_DF):
                    df_test = pd.read_hdf(PREPROCESSED_TEST_DF, 'df_data')
                else:
                    df_test = pd.read_csv(self.TEST_DATA_FILE,
                                           encoding="utf-8").fillna(" ")
                    df_test['question1'] = df_test['question1']\
                                           .map(lambda x: str(x).strip().split())
                    df_test['question2'] = df_test['question2']\
                                           .map(lambda x: str(x).strip().split())
                    df_test.to_hdf(PREPROCESSED_TEST_DF,'df_data',mode='w')

                print("Computing features for the test data set...")
                X_test = self.build_features(df_test)
                print("Saving...")
                X_test.to_csv(self.CUSTOM_FEATURES_TEST, index=False)
