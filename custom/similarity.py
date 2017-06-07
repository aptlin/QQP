# * Libraries

import os
from os.path import exists
import numpy as np
import pandas as pd
import time
import functools

from custom.similarity_backbone import SimilarityBackbone

# * Variables

BASE_DIR = 'data/'
TRAIN_DATA_FILENAME = "2017-05-24-1818-shorties_trimmed_train"
TRAIN_DATA_FILE = BASE_DIR + TRAIN_DATA_FILENAME + '.csv'
TEST_DATA_FILE = BASE_DIR + 'test.csv'

INFO_CONTENT_NORMALISATION = True

DELTA = 0.85

# * Constructor


class SimilarityFeatures:
    def __init__(self,
                 train_data_filename=TRAIN_DATA_FILENAME,
                 test_data_filename=TEST_DATA_FILE):

        self.TRAIN_DATA_FILENAME = train_data_filename
        self.TRAIN_DATA_FILE = BASE_DIR + train_data_filename + '.csv'        
        self.TEST_DATA_FILE = test_data_filename
        self.TRAIN_DATA = pd.read_csv(self.TRAIN_DATA_FILE,
                                   encoding="utf-8").fillna(" ")
        self.TEST_DATA = pd.read_csv(self.TEST_DATA_FILE,
                                     encoding="utf-8").fillna(" ")
        
        self.CUSTOM_FEATURES_TRAIN = 'custom/' + \
                                     train_data_filename + \
                                     "-similarity-train.csv"
        self.CUSTOM_FEATURES_TEST = 'custom/similarity-test.csv'

        # Containers for the similarity backbone

        self.sb_train = None
        self.sb_test = None

        self.semantic_similarity_dict = {}
        self.word_order_similarity_dict = {}
        self.similarity_dict = {}

    def _get_unique_words(self, data):
        return set(list(data['question1'].str.split(' ', expand=True).stack().unique()) +
                   list(data['question2'].str.split(' ', expand=True).stack().unique()))
    def _init_backbone(self, data):
        words = self._get_unique_words(data)
        return SimilarityBackbone(words)        

    def semantic_similarity(self, row, backbone=SimilarityBackbone()):
        start_time = time.time()
        if row['id'] % 100 == 0:
            elapsed = time.time() - start_time
            print("Processed {} rows after {:10.0f} s...".format(row['id'], elapsed))
        if (row['question1'], row['question2']) in self.semantic_similarity_dict:
            return self.semantic_similarity_dict[(row['question1'], row['question2'])]
        elif (row['question2'], row['question1']) in self.semantic_similarity_dict:
            return self.semantic_similarity_dict[(row['question2'], row['question1'])]
        else:
            sem_sim = backbone.semantic_similarity(row['question1'],
                                                   row['question2'],
                                                   INFO_CONTENT_NORMALISATION)
            self.semantic_similarity_dict[(row['question1'],
                                           row['question2'])] = sem_sim
            self.semantic_similarity_dict[(row['question2'],
                                           row['question1'])] = sem_sim            
            return sem_sim

    def word_order_similarity(self, row, backbone=SimilarityBackbone()):
        start_time = time.time()
        if row['id'] % 100 == 0:
            elapsed = time.time() - start_time
            print("Processed {} rows after {:10.0f} s...".format(row['id'], elapsed))            
        if (row['question1'], row['question2']) in self.word_order_similarity_dict:
            return self.word_order_similarity_dict[(row['question1'], row['question2'])]
        elif (row['question2'], row['question1']) in self.word_order_similarity_dict:
            return self.word_order_similarity_dict[(row['question2'], row['question1'])]
        else:
            sim = backbone.word_order_similarity(row['question1'],
                                                   row['question2'],
                                                   INFO_CONTENT_NORMALISATION)
            self.word_order_similarity_dict[(row['question1'],
                                           row['question2'])] = sim
            self.semantic_similarity_dict[(row['question2'],
                                           row['question1'])] = sim            
            return sim
        
    def similarity(self, row, backbone=SimilarityBackbone()):
        start_time = time.time()
        if row['id'] % 100 == 0:
            elapsed = time.time() - start_time
            print("Processed {} rows after {:10.0f} s...".format(row['id'], elapsed))                    
        if (row['question1'], row['question2']) in self.similarity_dict:
            return self.similarity_dict[(row['question1'], row['question2'])]
        elif (row['question2'], row['question1']) in self.similarity_dict:
            return self.similarity_dict[(row['question2'], row['question1'])]
        else:
            sim = DELTA * self.semantic_similarity(row['question1'],
                                                   row['question2'],
                                                   INFO_CONTENT_NORMALISATION) + \
                                                   (1.0 - DELTA) * self.word_order_similarity(row['question1'],
                                                                                              row['question2'])
            self.semantic_similarity_dict[(row['question2'],
                                           row['question1'])] = sim            
            return sim        
        
    def build_features(self, data, backbone):
        X = pd.DataFrame()
        print("Building features...")
        print("Working on the word order similarity...")
        f = functools.partial(self.semantic_similarity, backbone=backbone)
        X['word_order_similarity'] = data.apply(f, axis=1, raw=True)
        print("Computing semantic similarity...")
        f = functools.partial(self.word_order_similarity, backbone=backbone)
        X['semantic_similarity'] = data.apply(f, axis=1, raw=True)
        print("Averaging the results...")
        f = functools.partial(self.similarity, backbone=backbone)        
        X['corpus_similarity'] = data.apply(f, axis=1, raw=True)        

        return X
        

    def run(self):
        if exists(self.CUSTOM_FEATURES_TRAIN) and exists(self.CUSTOM_FEATURES_TEST):
            print("Using cached similarity features for {}..."
                  .format(self.TRAIN_DATA_FILENAME))
        else:
            print("Processing the training data set...")
            
            df_train = self.TRAIN_DATA

            if exists(self.CUSTOM_FEATURES_TRAIN):
                print("Using cached similarity features the training data set...")
            else:
                print("Caching train words for processing...")
                sb_train = self._init_backbone(self.TRAIN_DATA)
                print("Computing features for the training data set...")
                X_train = self.build_features(df_train, sb_train)
                print("Saving...")
                X_train.to_csv(self.CUSTOM_FEATURES_TRAIN, index=False)
            if exists(self.CUSTOM_FEATURES_TEST):
                print("Using cached similarity features the test data set...")
            else:
                print("Caching test words for processing...")
                sb_test = self._init_backbone(self.TEST_DATA)                
                print("Processing the testing data set...")
                df_test = self.TEST_DATA
                print("Computing features for the test data set...")
                X_test = self.build_features(df_test, sb_test)
                print("Saving...")
                X_test.to_csv(self.CUSTOM_FEATURES_TEST, index=False)
                
