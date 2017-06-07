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
PREPROCESSED_TRAIN_SYNSETS = BASE_DIR + PREPROCESSED + TRAIN_DATA_FILENAME + '-' + 'synsets.pkl'
PREPROCESSED_TRAIN_HYPERNYMS = BASE_DIR + PREPROCESSED + TRAIN_DATA_FILENAME + '-' + 'hypernyms.pkl'
PREPROCESSED_TRAIN_DF = BASE_DIR + PREPROCESSED + TRAIN_DATA_FILENAME + '-' + 'dataframe.hdf'

PREPROCESSED_TEST_WORDS = BASE_DIR + PREPROCESSED + "test" + '-' + 'words.pkl'
PREPROCESSED_TEST_SYNSETS = BASE_DIR + PREPROCESSED + "test" + '-' + 'synsets.pkl'
PREPROCESSED_TEST_HYPERNYMS = BASE_DIR + PREPROCESSED + "test" + '-' + 'hypernyms.pkl'
PREPROCESSED_TEST_DF = BASE_DIR + PREPROCESSED + "test" + '-' + 'dataframe.hdf'
# * Constructor

class NLTKFeatures:
    def __init__(self,
                 train_data_filename=TRAIN_DATA_FILENAME,
                 test_data_filename=TEST_DATA_FILE):

        self.TRAIN_DATA_FILENAME = train_data_filename
        self.TRAIN_DATA_FILE = BASE_DIR + train_data_filename + '.csv'
        self.TEST_DATA_FILE = test_data_filename
        self.CUSTOM_FEATURES_TRAIN = 'custom/' + \
                                     train_data_filename + \
                                     "-nltk-train.csv"
        self.CUSTOM_FEATURES_TEST = 'custom/nltk-test.csv'

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

        print("Getting synsets, hypernyms and lemmas for the train data set...")
        self.train_synsets_dict = self._get_synsets(self.train_raw_words)
        self.train_hypernyms_dict = self._get_hypernyms(self.train_raw_words,
                                                        self.train_synsets_dict)
        self.train_lemmas_dict = self._get_lemmas(self.train_raw_words,
                                                  self.train_synsets_dict)
        print("Getting synsets, hypernyms and lemmas for the test data set...")
        self.test_synsets_dict = self._get_synsets(self.test_raw_words)
        self.test_hypernyms_dict = self._get_hypernyms(self.test_raw_words,
                                                       self.test_synsets_dict)
        self.test_lemmas_dict = self._get_lemmas(self.test_raw_words,
                                                 self.test_synsets_dict)
        # if exists(PREPROCESSED_SYNSETS):
        #     with open(PREPROCESSED_SYNSETS, 'rb') as f:
        #         self.synsets_dict = pickle.load(f)
        # else:
        #     self.synsets_dict = self._get_synsets()
        #     with open(PREPROCESSED_SYNSETS, 'wb') as f:
        #         pickle.dump(self.synsets_dict, f, pickle.HIGHEST_PROTOCOL)

        # if exists(PREPROCESSED_HYPERNYMS):
        #     with open(PREPROCESSED_HYPERNYMS, 'rb') as f:
        #         self.hypernyms_dict = pickle.load(f)
        # else:
        #     self.hypernyms_dict = self._get_hypernyms()
        #     with open(PREPROCESSED_HYPERNYMS, 'wb') as f:
        #         pickle.dump(self.hypernyms_dict, f, pickle.HIGHEST_PROTOCOL)

    def _get_unique_words(self, data):
        print("Getting unique words from the dataframe...")
        return set(list(data['question1'].str.split(' ', expand=True).stack().unique()) +
                   list(data['question2'].str.split(' ', expand=True).stack().unique()))
    def _get_synsets(self, raw_words):
        print("Processing synsets...")
        synsets_dict = {}
        for w in raw_words:
            synsets_dict[w] = wn.synsets(w)
        return synsets_dict
    def _get_hypernyms(self, raw_words, synsets_dict={}):
        print("Processing hypernyms...")
        hypernyms_dict = {}
        for w in raw_words:
            synsets = synsets_dict[w]
            hypernyms = []
            for synset in synsets:
                hypernyms.append(synset.hypernyms())
            hypernyms_dict[w] = hypernyms
        return hypernyms_dict
    def _get_lemmas(self, raw_words, synsets_dict={}):
        print("Processing lemmas...")
        lemmas_dict = {}
        for w in raw_words:
            synsets = synsets_dict[w]
            lemmas = []
            for synset in synsets:
                lemmas.append(synset.lemmas())
            lemmas_dict[w] = lemmas
        return lemmas_dict

    def _key_not_found(self, word):
        print("Could not find a word '{}'!".format(word))


    def linear_synonyms_count(self, row, synsets_dict={}):
        try:
            if row['id'] % 50000 == 0:
                print("Processing {}".format(row['id']))
        except KeyError:
            if row['test_id'] % 50000 == 0:
                print("Processing {}".format(row['test_id']))
        q1_syncount = 0
        q2_syncount = 0
        for word in row['question1']:
            try:
                q1_syncount += len(synsets_dict[word]) / (1+len(row['question1']))
            except KeyError:
                self._key_not_found(word)
        for word in row['question2']:
            try:
                q2_syncount += len(synsets_dict[word]) / (1+len(row['question2']))
            except KeyError:
                self._key_not_found(word)
        return abs(q2_syncount - q1_syncount)

    def smooth_synonyms_count(self, row, synsets_dict={}):
        try:
            if row['id'] % 50000 == 0:
                print("Processing {}".format(row['id']))
        except KeyError:
            if row['test_id'] % 50000 == 0:
                print("Processing {}".format(row['test_id']))

        q1_syncount = 0
        q2_syncount = 0
        for word in row['question1']:            
            try:
                q1_syncount += len(synsets_dict[word]) / (1+len(row['question1']))
            except KeyError:
                self._key_not_found(word)
        for word in row['question2']:            
            try:
                q2_syncount += len(synsets_dict[word]) / (1+len(row['question2']))
            except KeyError:
                self._key_not_found(word)                

        return np.log(1+abs((q2_syncount + q1_syncount)/(1+q2_syncount * q1_syncount)))

    def hypernyms_share(self, row, hypernyms_dict={}):
        try:
            if row['id'] % 50000 == 0:
                print("Processing {}".format(row['id']))
        except KeyError:
            if row['test_id'] % 50000 == 0:
                print("Processing {}".format(row['test_id']))
        q1hypernyms = {}
        q2hypernyms = {}
        for word in row['question1']:
            try:
                hypernyms = hypernyms_dict[word]
                for hypernyms_list in hypernyms:
                    for hypernym in hypernyms_list:
                        q1hypernyms[hypernym.name().split('.')[0]] = 1
            except KeyError:
                self._key_not_found(word)
        for word in row['question2']:
            try:
                hypernyms = hypernyms_dict[word]
                for hypernyms_list in hypernyms:
                    for hypernym in hypernyms_list:
                        q2hypernyms[hypernym.name().split('.')[0]] = 1
            except KeyError:
                self._key_not_found(word)
        if len(q1hypernyms) == 0 or len(q2hypernyms) == 0:
            return 0
        shared_hypernyms_in_q1 = [h for h in q1hypernyms.keys() if h in q2hypernyms]
        shared_hypernyms_in_q2 = [h for h in q2hypernyms.keys() if h in q1hypernyms]
        R = (len(shared_hypernyms_in_q1) +
             len(shared_hypernyms_in_q2))/(len(q1hypernyms) + len(q2hypernyms))
        return R
    def lemmas_share(self, row, lemmas_dict={}):
        try:
            if row['id'] % 50000 == 0:
                print("Processing {}".format(row['id']))
        except KeyError:
            if row['test_id'] % 50000 == 0:
                print("Processing {}".format(row['test_id']))
        q1lemmas = {}
        q2lemmas = {}
        for word in row['question1']:
            try:
                lemmas = lemmas_dict[word]
                for lemma_list in lemmas:
                    for lemma in lemma_list:
                        q1lemmas[lemma.name().split('.')[0]] = 1
            except KeyError:
                self._key_not_found(word)
        for word in row['question2']:
            try:
                lemmas = lemmas_dict[word]
                for lemma_list in lemmas:
                    for lemma in lemma_list:
                        q2lemmas[lemma.name().split('.')[0]] = 1
            except KeyError:
                self._key_not_found(word)
        if len(q1lemmas) == 0 or len(q2lemmas) == 0:
            return 0
        shared_lemmas_in_q1 = [l for l in q1lemmas.keys() if l in q2lemmas]
        shared_lemmas_in_q2 = [l for l in q2lemmas.keys() if l in q1lemmas]
        R = (len(shared_lemmas_in_q1) +
             len(shared_lemmas_in_q2))/(len(q1lemmas) + len(q2lemmas))
        return R

    def cross_path_similarity(self, row, synsets_dict={}):
        try:
            if row['id'] % 50000 == 0:
                print("Processing {}".format(row['id']))
        except KeyError:
            if row['test_id'] % 50000 == 0:
                print("Processing {}".format(row['test_id']))

        q1synsets = set()
        q2synsets = set()
        for word in row['question1']:
            try:
                synsets = synsets_dict[word]
                for synset in synsets:
                    q1synsets.add(synset)
            except KeyError:
                self._key_not_found(word)
        for word in row['question2']:
            try:
                synsets = synsets_dict[word]
                for synset in synsets:
                    q2synsets.add(synset)
            except KeyError:
                self._key_not_found(word)
        if len(q1synsets) <= len(q2synsets):
            q_min = q1synsets
            q_max = q2synsets
        else:
            q_min = q2synsets
            q_max = q1synsets

        combinations = [zip(x,q_min) for x in itertools.permutations(q_max,len(q_min))]

        cps_score = 0
        for cycle in combinations:
            for synset1, synset2 in cycle:
                cps_score += synset1.path_similarity(synset2)
        return cps_score

    def build_features(self,
                       data,
                       synsets_dict,
                       hypernyms_dict,
                       lemmas_dict):
        X = pd.DataFrame()
        print("Calculating hypernyms_share...")
        f = functools.partial(self.hypernyms_share,
                              hypernyms_dict=hypernyms_dict)
        X['hypernyms_share'] = data.apply(f, axis=1, raw=True)
        print("Calculating linear_synonyms_count...")
        f = functools.partial(self.linear_synonyms_count,
                              synsets_dict=synsets_dict)
        X['linear_synonyms_count'] = data.apply(f, axis=1, raw=True)
        print("Calculating smooth_synonyms_count...")
        f = functools.partial(self.smooth_synonyms_count,
                              synsets_dict=synsets_dict)
        X['smooth_synonyms_count'] = data.apply(f, axis=1, raw=True)
        print("Calculating lemmas_share...")
        f = functools.partial(self.lemmas_share,
                              lemmas_dict=lemmas_dict)
        X['lemmas_share'] = data.apply(f, axis=1, raw=True)
        # print("Calculating cross_path_similarity...")
        # X['cross_path_similarity'] =
        # data.apply(self.cross_path_similarity, axis=1, raw=True)

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
                X_train = self.build_features(df_train,
                                              self.train_synsets_dict,
                                              self.train_hypernyms_dict,
                                              self.train_lemmas_dict)
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
                X_test = self.build_features(df_test,
                                              self.test_synsets_dict,
                                              self.test_hypernyms_dict,
                                              self.test_lemmas_dict)
                print("Saving...")
                X_test.to_csv(self.CUSTOM_FEATURES_TEST, index=False)
