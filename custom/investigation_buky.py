# -*- coding: utf-8 -*-
# * Libraries

from os.path import exists
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import kendalltau
from scipy.spatial.distance import directed_hausdorff
from scipy.spatial.distance import correlation
from scipy.spatial import procrustes
import functools
import itertools
import time
import string

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

from collections import Counter

from nltk.corpus import wordnet as wn

import gensim
from gensim.models.word2vec import Word2Vec


import pickle

# * Variables

BASE_DIR = 'data/'
TRAIN_DATA_FILENAME = "vanilla_train"
TRAIN_DATA_FILE = BASE_DIR + TRAIN_DATA_FILENAME + '.csv'
TEST_DATA_FILE = BASE_DIR + 'test.csv'
EMBEDDING_FILE = BASE_DIR + 'GoogleNews-vectors-negative300.bin'

PREPROCESSED = 'preprocessed/'
PREPROCESSED_TRAIN_WORDS = BASE_DIR + PREPROCESSED + TRAIN_DATA_FILENAME + '-' + 'words.pkl'
PREPROCESSED_TRAIN_DF = BASE_DIR + PREPROCESSED + TRAIN_DATA_FILENAME + '-' + 'dataframe.hdf'

PREPROCESSED_TEST_WORDS = BASE_DIR + PREPROCESSED + "test" + '-' + 'words.pkl'
PREPROCESSED_TEST_DF = BASE_DIR + PREPROCESSED + "test" + '-' + 'dataframe.hdf'
PREPROCESSED_WORDVECS = BASE_DIR + PREPROCESSED + TRAIN_DATA_FILENAME + '-word2vec-dict.pkl'
LOCATIONS = BASE_DIR + "cities.csv"

start_time = time.time()
# * Constructor

class BukyFeatures:
    def __init__(self,
                 train_data_filename=TRAIN_DATA_FILENAME,
                 test_data_filename=TEST_DATA_FILE):

        self.TRAIN_DATA_FILENAME = train_data_filename
        self.TRAIN_DATA_FILE = BASE_DIR + train_data_filename + '.csv'
        self.TEST_DATA_FILE = test_data_filename
        self.CUSTOM_FEATURES_TRAIN = 'custom/' + \
                                     train_data_filename + \
                                     "-buky-train.csv"
        self.CUSTOM_FEATURES_TEST = 'custom/buky-test.csv'
        self.model = gensim.models.KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)
        self.wordvecs = {}

    def getWordVecs(self, words):
        changed_words = []
        for word in words:
            word = word.translate({ord(c): None for c in string.punctuation})
            changed_words.append(word)
            try:
                if word not in self.wordvecs:
                    self.wordvecs[word] = self.model[word]
            except KeyError:
                self.wordvecs[word] = np.zeros(300)
        return changed_words

    def dot_mean_word2vec(self, row):
        try:
            if row['id'] % 10000 == 0:
                elapsed = time.time() - start_time
                print("Processed {:10.0f} questions in {:10.0f} s ".format(row['id'], elapsed))
        except KeyError:
            if row['test_id'] % 10000 == 0:
                elapsed = time.time() - start_time
                print("Processed {:10.0f} questions in {:10.0f} s ".format(row['test_id'], elapsed))

        q1 = self.getWordVecs(row['question1'])
        q2 = self.getWordVecs(row['question2'])

        if len(q1) == 0 or len(q2) == 0:
            return 0

        q1_vec = np.zeros(300)
        q2_vec = np.zeros(300)

        for word in q1:
            q1_vec += self.wordvecs[word]
        q1_vec /= len(q1)
        for word in q2:
            q2_vec += self.wordvecs[word]
        q2_vec /= len(q2)

        return np.dot(q1_vec, q2_vec)

    def euch_cross_mean_word2vec(self, row):
        try:
            if row['id'] % 10000 == 0:
                elapsed = time.time() - start_time
                print("Processed {:10.0f} questions in {:10.0f} s ".format(row['id'], elapsed))
        except KeyError:
            if row['test_id'] % 10000 == 0:
                elapsed = time.time() - start_time
                print("Processed {:10.0f} questions in {:10.0f} s ".format(row['test_id'], elapsed))
        q1 = self.getWordVecs(row['question1'])
        q2 = self.getWordVecs(row['question2'])

        if len(q1) == 0 or len(q2) == 0:
            return 0

        q1_vec = np.zeros(300)
        q2_vec = np.zeros(300)

        for word in q1:
            word = word.translate({ord(c): None for c in string.punctuation})
            q1_vec += self.wordvecs[word]
        q1_vec /= len(q1)
        for word in q2:
            word = word.translate({ord(c): None for c in string.punctuation})
            q2_vec += self.wordvecs[word]
        q2_vec /= len(q2)

        q1_vec = np.array([np.mean(q1_vec), np.var(q1_vec), np.median(q1_vec)])
        q2_vec = np.array([np.mean(q2_vec), np.var(q2_vec), np.median(q2_vec)])

        score = np.linalg.norm(np.cross(q1_vec, q2_vec))
        return score

    def one_cross_mean_word2vec(self, row):
        try:
            if row['id'] % 10000 == 0:
                elapsed = time.time() - start_time
                print("Processed {:10.0f} questions in {:10.0f} s ".format(row['id'], elapsed))
        except KeyError:
            if row['test_id'] % 10000 == 0:
                elapsed = time.time() - start_time
                print("Processed {:10.0f} questions in {:10.0f} s ".format(row['test_id'], elapsed))
        q1 = self.getWordVecs(row['question1'])
        q2 = self.getWordVecs(row['question2'])

        if len(q1) == 0 or len(q2) == 0:
            return 0

        q1_vec = np.zeros(300)
        q2_vec = np.zeros(300)

        for word in q1:
            word = word.translate({ord(c): None for c in string.punctuation})
            q1_vec += self.wordvecs[word]
        q1_vec /= len(q1)
        for word in q2:
            word = word.translate({ord(c): None for c in string.punctuation})
            q2_vec += self.wordvecs[word]
        q2_vec /= len(q2)

        q1_vec = np.array([np.mean(q1_vec), np.var(q1_vec), np.median(q1_vec)])
        q2_vec = np.array([np.mean(q2_vec), np.var(q2_vec), np.median(q2_vec)])

        score = np.linalg.norm(np.cross(q1_vec, q2_vec), ord=1)
        return score

    def unique_dot_mean_word2vec(self, row):
        try:
            if row['id'] % 10000 == 0:
                elapsed = time.time() - start_time
                print("Processed {:10.0f} questions in {:10.0f} s ".format(row['id'], elapsed))
        except KeyError:
            if row['test_id'] % 10000 == 0:
                elapsed = time.time() - start_time
                print("Processed {:10.0f} questions in {:10.0f} s ".format(row['test_id'], elapsed))
        q1 = self.getWordVecs(row['question1'])
        q2 = self.getWordVecs(row['question2'])

        if len(q1) == 0 or len(q2) == 0:
            return 0

        for word in q1:
            word = word.translate({ord(c): None for c in string.punctuation})
        for word in q2:
            word = word.translate({ord(c): None for c in string.punctuation})

        q1_vec = np.zeros(300)
        q2_vec = np.zeros(300)

        for word in q1:
            if word not in q2:
                q1_vec += self.wordvecs[word]
        q1_vec /= len(q1)
        for word in q2:
            if word not in q1:
                q2_vec += self.wordvecs[word]
        q2_vec /= len(q2)

        return np.dot(q1_vec, q2_vec)

    def unique_euch_cross_mean_word2vec(self, row):
        try:
            if row['id'] % 10000 == 0:
                elapsed = time.time() - start_time
                print("Processed {:10.0f} questions in {:10.0f} s ".format(row['id'], elapsed))
        except KeyError:
            if row['test_id'] % 10000 == 0:
                elapsed = time.time() - start_time
                print("Processed {:10.0f} questions in {:10.0f} s ".format(row['test_id'], elapsed))
        q1 = self.getWordVecs(row['question1'])
        q2 = self.getWordVecs(row['question2'])

        if len(q1) == 0 or len(q2) == 0:
            return 0

        for word in q1:
            word = word.translate({ord(c): None for c in string.punctuation})
        for word in q2:
            word = word.translate({ord(c): None for c in string.punctuation})

        q1_vec = np.zeros(300)
        q2_vec = np.zeros(300)

        for word in q1:
            if word not in q2:
                q1_vec += self.wordvecs[word]
        q1_vec /= len(q1)
        for word in q2:
            if word not in q1:
                q2_vec += self.wordvecs[word]
        q2_vec /= len(q2)

        q1_vec = np.array([np.mean(q1_vec), np.var(q1_vec), np.median(q1_vec)])
        q2_vec = np.array([np.mean(q2_vec), np.var(q2_vec), np.median(q2_vec)])

        score = np.linalg.norm(np.cross(q1_vec, q2_vec))
        return score

    def unique_one_cross_mean_word2vec(self, row):
        try:
            if row['id'] % 10000 == 0:
                elapsed = time.time() - start_time
                print("Processed {:10.0f} questions in {:10.0f} s ".format(row['id'], elapsed))
        except KeyError:
            if row['test_id'] % 10000 == 0:
                elapsed = time.time() - start_time
                print("Processed {:10.0f} questions in {:10.0f} s ".format(row['test_id'], elapsed))
        q1 = self.getWordVecs(row['question1'])
        q2 = self.getWordVecs(row['question2'])

        if len(q1) == 0 or len(q2) == 0:
            return 0

        q1_vec = np.zeros(300)
        q2_vec = np.zeros(300)

        for word in q1:
            if word not in q2:
                q1_vec += self.wordvecs[word]
        q1_vec /= len(q1)
        for word in q2:
            if word not in q1:
                q2_vec += self.wordvecs[word]
        q2_vec /= len(q2)

        q1_vec = np.array([np.mean(q1_vec), np.var(q1_vec), np.median(q1_vec)])
        q2_vec = np.array([np.mean(q2_vec), np.var(q2_vec), np.median(q2_vec)])

        score = np.linalg.norm(np.cross(q1_vec, q2_vec), ord=1)
        return score


    def log_diff_word2vec(self, row):
        try:
            if row['id'] % 10000 == 0:
                elapsed = time.time() - start_time
                print("Processed {:10.0f} questions in {:10.0f} s ".format(row['id'], elapsed))
        except KeyError:
            if row['test_id'] % 10000 == 0:
                elapsed = time.time() - start_time
                print("Processed {:10.0f} questions in {:10.0f} s ".format(row['test_id'], elapsed))
        q1 = self.getWordVecs(row['question1'])
        q2 = self.getWordVecs(row['question2'])

        if len(q1) == 0 or len(q2) == 0:
            return 0

        q1_vec = np.ones(300)
        q2_vec = np.ones(300)

        for word in q1:
            q1_vec += self.wordvecs[word]

        for word in q2:
            q2_vec += self.wordvecs[word]

        score = np.log(1 + np.linalg.norm(q2_vec - q1_vec) / (len(q1) + len(q2)))
        return score

    def naive_diff_word2vec(self, row):
        try:
            if row['id'] % 10000 == 0:
                elapsed = time.time() - start_time
                print("Processed {:10.0f} questions in {:10.0f} s ".format(row['id'], elapsed))
        except KeyError:
            if row['test_id'] % 10000 == 0:
                elapsed = time.time() - start_time
                print("Processed {:10.0f} questions in {:10.0f} s ".format(row['test_id'], elapsed))

        q1 = self.getWordVecs(row['question1'])
        q2 = self.getWordVecs(row['question2'])

        if len(q1) == 0 or len(q2) == 0:
            return 0

        q1_vec = np.ones(300)
        q2_vec = np.ones(300)

        for word in q1:
            q1_vec += self.wordvecs[word]

        for word in q2:
            q2_vec += self.wordvecs[word]

        score = abs(np.linalg.norm(q2_vec) / len(q2) - np.linalg.norm(q1_vec) / len(q1))
        return score

    def directed_hausdorff_word2vec(self, row):
        try:
            if row['id'] % 10000 == 0:
                elapsed = time.time() - start_time
                print("Processed {:10.0f} questions in {:10.0f} s ".format(row['id'], elapsed))
        except KeyError:
            if row['test_id'] % 10000 == 0:
                elapsed = time.time() - start_time
                print("Processed {:10.0f} questions in {:10.0f} s ".format(row['test_id'], elapsed))
        q1 = self.getWordVecs(row['question1'])
        q2 = self.getWordVecs(row['question2'])

        if len(q1) == 0 or len(q2) == 0:
            return 0

        q1_vecs = tuple(self.wordvecs[word].reshape((1, 300)) for word in q1)
        q2_vecs = tuple(self.wordvecs[word].reshape((1, 300)) for word in q2)

        q1_vecs = np.concatenate(q1_vecs)
        q2_vecs = np.concatenate(q2_vecs)

        score = directed_hausdorff(q1_vecs, q2_vecs)[0]
        return score

    def directed_hausdorff_unique_word2vec(self, row):
        try:
            if row['id'] % 10000 == 0:
                elapsed = time.time() - start_time
                print("Processed {:10.0f} questions in {:10.0f} s ".format(row['id'], elapsed))
        except KeyError:
            if row['test_id'] % 10000 == 0:
                elapsed = time.time() - start_time
                print("Processed {:10.0f} questions in {:10.0f} s ".format(row['test_id'], elapsed))

        q1_words = self.getWordVecs(row['question1'])
        q2_words = self.getWordVecs(row['question2'])

        q1 = [ word for word in q1_words if word not in q2_words]
        q2 = [ word for word in q2_words if word not in q1_words]


        if len(q1) == 0 or len(q2) == 0:
            return 0

        q1_vecs = tuple(self.wordvecs[word].reshape((1, 300)) for word in q1)
        q2_vecs = tuple(self.wordvecs[word].reshape((1, 300)) for word in q2)

        q1_vecs = np.concatenate(q1_vecs)
        q2_vecs = np.concatenate(q2_vecs)

        score = directed_hausdorff(q1_vecs, q2_vecs)[0]
        return score


    def procrustes_word2vec(self, row):
        try:
            if row['id'] % 10000 == 0:
                elapsed = time.time() - start_time
                print("Processed {:10.0f} questions in {:10.0f} s ".format(row['id'], elapsed))
        except KeyError:
            if row['test_id'] % 10000 == 0:
                elapsed = time.time() - start_time
                print("Processed {:10.0f} questions in {:10.0f} s ".format(row['test_id'], elapsed))

        stops = set(stopwords.words("english"))
        q1 = self.getWordVecs(row['question1'])
        q2 = self.getWordVecs(row['question2'])

        q1 = list(set(word for word in q1 if word not in stops))
        q2 = list(set(word for word in q2 if word not in stops))

        trim_length = min(4, min(len(q1), len(q2)))
        q1 = q1[:trim_length]
        q2 = q2[:trim_length]
        
        if len(q1) == 0 or len(q2) == 0 or len(q1) == 1 or len(q2) == 1:
            return 0

        q1_vecs = tuple(self.wordvecs[word].reshape((1, 300)) for word in q1)
        q2_vecs = tuple(self.wordvecs[word].reshape((1, 300)) for word in q2)

        q1_vecs = np.concatenate(q1_vecs, axis=0)
        q2_vecs = np.concatenate(q2_vecs, axis=0)

        score = procrustes(q1_vecs, q2_vecs)[2]
        return score

    def procrustes_unique_word2vec(self, row):
        try:
            if row['id'] % 10000 == 0:
                elapsed = time.time() - start_time
                print("Processed {:10.0f} questions in {:10.0f} s ".format(row['id'], elapsed))
        except KeyError:
            if row['test_id'] % 10000 == 0:
                elapsed = time.time() - start_time
                print("Processed {:10.0f} questions in {:10.0f} s ".format(row['test_id'], elapsed))

        stops = set(stopwords.words("english"))
        q1_words = self.getWordVecs(row['question1'])
        q2_words = self.getWordVecs(row['question2'])

        q1 = list(set( word for word in q1_words if word not in q2_words and word not in stops))
        q2 = list(set( word for word in q2_words if word not in q1_words and word not in stops))

        trim_length = min(3, min(len(q1), len(q2)))                
        q1 = q1[:trim_length]
        q2 = q2[:trim_length]

        if len(q1) == 0 or len(q2) == 0 or len(q1) == 1 or len(q2) == 1:
            return 0
        
        q1_vecs = tuple(self.wordvecs[word].reshape((1, 300)) for word in q1)
        q2_vecs = tuple(self.wordvecs[word].reshape((1, 300)) for word in q2)

        q1_vecs = np.concatenate(q1_vecs, axis=0)
        q2_vecs = np.concatenate(q2_vecs, axis=0)

        score = procrustes(q1_vecs, q2_vecs)[2]
        return score


    def correlation_unique_am_word2vec(self, row):
        try:
            if row['id'] % 10000 == 0:
                elapsed = time.time() - start_time
                print("Processed {:10.0f} questions in {:10.0f} s ".format(row['id'], elapsed))
        except KeyError:
            if row['test_id'] % 10000 == 0:
                elapsed = time.time() - start_time
                print("Processed {:10.0f} questions in {:10.0f} s ".format(row['test_id'], elapsed))
        q1 = self.getWordVecs(row['question1'])
        q2 = self.getWordVecs(row['question2'])

        if len(q1) == 0 or len(q2) == 0:
            return 0

        q1_vec = np.zeros(300)
        q2_vec = np.zeros(300)

        for word in q1:
            if word not in q2:
                q1_vec += self.wordvecs[word]
        q1_vec /= len(q1)
        for word in q2:
            if word not in q1:
                q2_vec += self.wordvecs[word]
        q2_vec /= len(q2)

        score = correlation(q1_vec, q2_vec)
        return score
    def correlation_am_word2vec(self, row):
        try:
            if row['id'] % 10000 == 0:
                elapsed = time.time() - start_time
                print("Processed {:10.0f} questions in {:10.0f} s ".format(row['id'], elapsed))
        except KeyError:
            if row['test_id'] % 10000 == 0:
                elapsed = time.time() - start_time
                print("Processed {:10.0f} questions in {:10.0f} s ".format(row['test_id'], elapsed))
        q1 = self.getWordVecs(row['question1'])
        q2 = self.getWordVecs(row['question2'])

        if len(q1) == 0 or len(q2) == 0:
            return 0

        q1_vec = np.zeros(300)
        q2_vec = np.zeros(300)

        for word in q1:
            q1_vec += self.wordvecs[word]
        q1_vec /= len(q1)
        for word in q2:
            q2_vec += self.wordvecs[word]
        q2_vec /= len(q2)

        score = correlation(q1_vec, q2_vec)
        return score

    def build_features(self, data):
        X = pd.DataFrame()
        # Commented features are too computationally expensive.
        # print("Calculating procrustes_word2vec...")
        # X['procrustes_word2vec'] = data.apply(self.procrustes_word2vec, axis=1, raw=True)
        # print("Calculating procrustes_unique_word2vec...")
        # X['procrustes_unique_word2vec'] = data.apply(self.procrustes_unique_word2vec, axis=1, raw=True)                
        print("Calculating dot_mean_word2vec...")
        X['dot_mean_word2vec'] = data.apply(self.dot_mean_word2vec, axis=1, raw=True)
        print("Calculating euch_cross_mean_word2vec...")
        X['euch_cross_mean_word2vec'] = data.apply(self.euch_cross_mean_word2vec, axis=1, raw=True)
        print("Calculating one_cross_mean_word2vec...")
        X['one_cross_mean_word2vec'] = data.apply(self.one_cross_mean_word2vec, axis=1, raw=True)
        print("Calculating unique_dot_mean_word2vec...")
        X['unique_dot_mean_word2vec'] = data.apply(self.unique_dot_mean_word2vec, axis=1, raw=True)
        print("Calculating unique_euch_cross_mean_word2vec...")
        X['unique_euch_cross_mean_word2vec'] = data.apply(self.unique_euch_cross_mean_word2vec, axis=1, raw=True)
        print("Calculating unique_one_cross_mean_word2vec...")
        X['unique_one_cross_mean_word2vec'] = data.apply(self.unique_one_cross_mean_word2vec, axis=1, raw=True)
        print("Calculating log_diff_word2vec...")
        X['log_diff_word2vec'] = data.apply(self.log_diff_word2vec, axis=1, raw=True)
        print("Calculating naive_diff_word2vec...")
        X['naive_diff_word2vec'] = data.apply(self.naive_diff_word2vec, axis=1, raw=True)
        print("Calculating directed_hausdorff_word2vec...")
        X['directed_hausdorff_word2vec'] = data.apply(self.directed_hausdorff_word2vec, axis=1, raw=True)
        print("Calculating directed_hausdorff_unique_word2vec...")
        X['directed_hausdorff_unique_word2vec'] = data.apply(self.directed_hausdorff_unique_word2vec, axis=1, raw=True)
        print("Calculating correlation_am_word2vec...")
        X['correlation_am_word2vec'] = data.apply(self.correlation_am_word2vec, axis=1, raw=True).fillna(0.0)
        print("Calculating correlation_unique_am_word2vec...")
        X['correlation_unique_am_word2vec'] = data.apply(self.correlation_unique_am_word2vec, axis=1, raw=True).fillna(0.0)

        return X

    def run(self):
        if exists(self.CUSTOM_FEATURES_TRAIN) and exists(self.CUSTOM_FEATURES_TEST):
            print("Using cached buky features for {}..."
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
                print("Using cached features for the test data set...")
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
