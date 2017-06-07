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
import re

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

LOCATIONS = BASE_DIR + "cities.csv"

start_time = time.time()
# * Constructor

class AzFeatures:
    def __init__(self,
                 train_data_filename=TRAIN_DATA_FILENAME,
                 test_data_filename=TEST_DATA_FILE):

        self.TRAIN_DATA_FILENAME = train_data_filename
        self.TRAIN_DATA_FILE = BASE_DIR + train_data_filename + '.csv'
        self.TEST_DATA_FILE = test_data_filename
        self.CUSTOM_FEATURES_TRAIN = 'custom/' + \
                                     train_data_filename + \
                                     "-az-train.csv"
        self.CUSTOM_FEATURES_TEST = 'custom/az-test.csv'

        locations = pd.read_csv(LOCATIONS, encoding="utf-8")
        countries = set(locations['Country'].dropna(inplace=False).values.tolist())
        cities = set(locations['City'].dropna(inplace=False).values.tolist())
        self.all_places = countries | cities
        self.regex = "|".join(sorted(set(self.all_places)))
        self.matches = {}

    def _get_matches(self, row):
        q1 = ' '.join(row['question1'])\
               .translate({ord(c): None for c in string.punctuation})
        q2 = ' '.join(row['question1'])\
               .translate({ord(c): None for c in string.punctuation})
        if len(q1) == 0 or len(q2) == 0:
            return (-1, -1)
        try:            
            return (self.matches[q1], self.matches[q2])
        except KeyError:
            self.matches[q1] = [i.group().lower() for i in re.finditer(self.regex, q1, flags=re.IGNORECASE)]
            self.matches[q2] = [i.group().lower() for i in re.finditer(self.regex, q2, flags=re.IGNORECASE)]
            return (self.matches[q1], self.matches[q2])

    def places_share(self, row):
        try:
            if row['id'] % 10000 == 0:
                elapsed = time.time() - start_time
                print("Processed {:10.0f} questions in {:10.0f} s ".format(row['id'], elapsed))
        except KeyError:
            if row['test_id'] % 10000 == 0:
                elapsed = time.time() - start_time
                print("Processed {:10.0f} questions in {:10.0f} s ".format(row['test_id'], elapsed))        

        q1_matches, q2_matches = self._get_matches(row)
        if q1_matches == -1 and q2_matches == -1:
            return 0
        else:
            return len(set(q1_matches).intersection(set(q2_matches)))

    def places_difference(self, row):
        try:
            if row['id'] % 10000 == 0:
                elapsed = time.time() - start_time
                print("Processed {:10.0f} questions in {:10.0f} s ".format(row['id'], elapsed))
        except KeyError:
            if row['test_id'] % 10000 == 0:
                elapsed = time.time() - start_time
                print("Processed {:10.0f} questions in {:10.0f} s ".format(row['test_id'], elapsed))

        q1_matches, q2_matches = self._get_matches(row)
        if q1_matches == -1 and q2_matches == -1:
            return 0
        else:
            return len(set(q1_matches).difference(set(q2_matches)))

    def places_prevalence(self, row):
        try:
            if row['id'] % 10000 == 0:
                elapsed = time.time() - start_time
                print("Processed {:10.0f} questions in {:10.0f} s ".format(row['id'], elapsed))
        except KeyError:
            if row['test_id'] % 10000 == 0:
                elapsed = time.time() - start_time
                print("Processed {:10.0f} questions in {:10.0f} s ".format(row['test_id'], elapsed))
        
        q1_matches, q2_matches = self._get_matches(row)
        if q1_matches == -1 and q2_matches == -1:
            return 0
        else:
            return len(q1_matches)*len(q2_matches)

    def build_features(self, data):
        X = pd.DataFrame()
        print("Calculating places_share...")
        X['places_share'] = data.apply(self.places_share, axis=1, raw=True)
        # print("Calculating places_difference...")
        # X['places_difference'] = data.apply(self.places_difference, axis=1, raw=True)
        print("Calculating places_prevalence...")
        X['places_prevalence'] = data.apply(self.places_prevalence, axis=1, raw=True)

        return X

    def run(self):
        if exists(self.CUSTOM_FEATURES_TRAIN) and exists(self.CUSTOM_FEATURES_TEST):
            print("Using cached az features for {}..."
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
