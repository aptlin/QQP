# -*- coding: utf-8 -*-
# * Libraries

import os
from os.path import exists
import numpy as np
import pandas as pd

import functools
import itertools

from nltk.corpus import stopwords
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
# * Constructor

class WordsFeatures:
    def __init__(self,
                 train_data_filename=TRAIN_DATA_FILENAME,
                 test_data_filename=TEST_DATA_FILE):

        self.TRAIN_DATA_FILENAME = train_data_filename
        self.TRAIN_DATA_FILE = BASE_DIR + train_data_filename + '.csv'
        self.TEST_DATA_FILE = test_data_filename
        self.CUSTOM_FEATURES_TRAIN = 'custom/' + \
                                     train_data_filename + \
                                     "-wordies-train.csv"
        self.CUSTOM_FEATURES_TEST = 'custom/wordies-test.csv'

        # containers for features

        if exists(PREPROCESSED_TRAIN_WORDS):
            with open(PREPROCESSED_TRAIN_WORDS, 'rb') as f:
                self.train_raw_words = pickle.load(f)
        else:
            print("Processing the train data file...")
            TRAIN_DATA = pd.read_csv(self.TRAIN_DATA_FILE, encoding="utf-8")
            self.train_raw_words = self._get_unique_words(TRAIN_DATA)
            with open(PREPROCESSED_TRAIN_WORDS, 'wb') as f:
                pickle.dump(self.train_raw_words, f, pickle.HIGHEST_PROTOCOL)
            del TRAIN_DATA
        if exists(PREPROCESSED_TEST_WORDS):
            with open(PREPROCESSED_TEST_WORDS, 'rb') as f:
                self.test_raw_words = pickle.load(f,encoding="UTF-8")
        else:
            print("Processing the test data file...")
            TEST_DATA = pd.read_csv(self.TEST_DATA_FILE, encoding="utf-8")
            self.test_raw_words = self._get_unique_words(TEST_DATA)
            with open(PREPROCESSED_TEST_WORDS, 'wb') as f:
                pickle.dump(self.test_raw_words, f, pickle.HIGHEST_PROTOCOL)
            del TEST_DATA


    def _get_unique_words(self, data):
        print("Getting unique words from the dataframe...")
        return set(list(data['question1'].str.split(' ', expand=True).stack().unique()) +
                   list(data['question2'].str.split(' ', expand=True).stack().unique()))

    def character_freq(self, row):
        try:
            if row['id'] % 50000 == 0:
                print("Processing {}".format(row['id']))
        except KeyError:
            if row['test_id'] % 50000 == 0:
                print("Processing {}".format(row['test_id']))

        q1characters = {}
        q2characters = {}
        for word in row['question1']:
            characters = list(word)
            for character in characters:
                try:
                    q1characters[character] += 1
                except KeyError:
                    q1characters[character] = 1

        for word in row['question2']:
            characters = list(word)
            for character in characters:
                try:
                    q2characters[character] += 1
                except KeyError:
                    q2characters[character] = 1                    
        if len(q1characters) == 0 or len(q2characters) == 0:
            return 0
        q1freqs = {}
        for character in q1characters:
            q1freqs[character] = q1characters[character] / len(row['question1'])
        q2freqs = {}
        for character in q2characters:
            q2freqs[character] = q2characters[character] / len(row['question2'])

        score = 0
        for character in q1freqs:
            if character in q2freqs and abs(q2freqs[character]-q1freqs[character]) <= 0.01:
                score += 1
            elif character not in q2freqs:
                score -= 1
        for character in q2freqs:
            if character in q1freqs and abs(q2freqs[character]-q1freqs[character]) <= 0.01:
                score += 1
            elif character not in q2freqs:
                score -= 1
                
        R = score / (len(q1freqs) + len(q2freqs))
        return R

    def syllable_similarity(self, row):
        try:
            if row['id'] % 50000 == 0:
                print("Processing {}".format(row['id']))
        except KeyError:
            if row['test_id'] % 50000 == 0:
                print("Processing {}".format(row['test_id']))

        q1syllables = {}
        q2syllables = {}
        for word in row['question1']:
            if len(word) > 1:
                characters = list(word)                
                for i in range(len(word)-1):
                    syllable = characters[i] + characters[i+1]                    
                    try:
                        q1syllables[syllable] += 1
                    except KeyError:
                        q1syllables[syllable] = 1
            else:
                try:
                    q1syllables[word] += 1
                except KeyError:
                    q1syllables[word] = 1

        for word in row['question2']:
            if len(word) > 1:
                characters = list(word)                
                for i in range(len(word)-1):
                    syllable = characters[i] + characters[i+1]                    
                    try:
                        q2syllables[syllable] += 1
                    except KeyError:
                        q2syllables[syllable] = 1
            else:
                try:
                    q2syllables[word] += 1
                except KeyError:
                    q2syllables[word] = 1

        if len(q1syllables) == 0 or len(q2syllables) == 0:
            return 0                    

        q1freqs = {}
        for syllable in q1syllables:
            q1freqs[syllable] = q1syllables[syllable] / len(row['question1'])
        q2freqs = {}
        for syllable in q2syllables:
            q2freqs[syllable] = q2syllables[syllable] / len(row['question2'])

        score = 0
        for syllable in q1freqs:
            if syllable in q2freqs and abs(q2freqs[syllable]-q1freqs[syllable]) <= 0.01:
                score += 0.3
            elif syllable not in q2freqs:
                score -= 1
        for syllable in q2freqs:
            if syllable in q1freqs and abs(q2freqs[syllable]-q1freqs[syllable]) <= 0.01:
                score += 1
            elif syllable not in q2freqs:
                score -= 0.3
                
        R = score / (len(q1freqs) + len(q2freqs))
        return R        


    def build_features(self, data):
        X = pd.DataFrame()
        print("Calculating character_freq...")
        X['character_freq'] = data.apply(self.character_freq, axis=1, raw=True)
        print("Calculating the similarity of syllables...")
        X['syllable_similarity'] = data.apply(self.syllable_similarity, axis=1, raw=True)

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
