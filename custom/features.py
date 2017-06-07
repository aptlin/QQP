# The code is adapted from @act444's work.
# See https://www.kaggle.com/act444/lb-0-158-xgb-handcrafted-leaky

# * Libraries

import os
from os.path import exists
import numpy as np
import pandas as pd

import functools

from nltk.corpus import stopwords
from collections import Counter

# * Variables

BASE_DIR = 'data/'
TRAIN_DATA_FILENAME = "vanilla_train"
TRAIN_DATA_FILE = BASE_DIR + TRAIN_DATA_FILENAME + '.csv'
TEST_DATA_FILE = BASE_DIR + 'test.csv'

VALIDATION_SPLIT = 0.1

# * Constructor


class CustomFeatures:
    def __init__(self,
                 train_data_filename=TRAIN_DATA_FILENAME,
                 test_data_filename=TEST_DATA_FILE):

        self.TRAIN_DATA_FILENAME = train_data_filename
        self.TRAIN_DATA_FILE = BASE_DIR + train_data_filename + '.csv'        
        self.TEST_DATA_FILE = test_data_filename        
        self.CUSTOM_FEATURES_TRAIN = 'custom/' + \
                                     train_data_filename + \
                                     "-train.csv"
        self.CUSTOM_FEATURES_TEST = 'custom/test.csv'
        

    def word_match_share(self, row, stops=None):
        q1words = {}
        q2words = {}
        for word in row['question1']:
            if word not in stops:
                q1words[word] = 1
        for word in row['question2']:
            if word not in stops:
                q2words[word] = 1
        if len(q1words) == 0 or len(q2words) == 0:
            return 0
        shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
        shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
        R = (len(shared_words_in_q1) +
             len(shared_words_in_q2))/(len(q1words) + len(q2words))
        return R

    def jaccard(self, row):
        wic = set(row['question1']).intersection(set(row['question2']))
        uw = set(row['question1']).union(row['question2'])
        if len(uw) == 0:
            uw = [1]
        return (len(wic) / len(uw))

    def common_words(self, row):
        return len(set(row['question1']).intersection(set(row['question2'])))

    def total_unique_words(self, row):
        return len(set(row['question1']).union(row['question2']))

    def total_unq_words_stop(self, row, stops):
        return len([x for x
                    in set(row['question1']).union(row['question2'])
                    if x not in stops])

    def wc_diff(self, row):
        return abs(len(row['question1']) - len(row['question2']))

    def wc_ratio(self, row):
        l1 = len(row['question1']) * 1.0
        l2 = len(row['question2'])
        if l2 == 0:
            return np.nan
        if l1 / l2:
            return l2 / l1
        else:
            return l1 / l2

    def wc_diff_unique(self, row):
        return abs(len(set(row['question1'])) - len(set(row['question2'])))

    def wc_ratio_unique(self, row):
        l1 = len(set(row['question1'])) * 1.0
        l2 = len(set(row['question2']))
        if l2 == 0:
            return np.nan
        if l1 / l2:
            return l2 / l1
        else:
            return l1 / l2

    def wc_diff_unique_stop(self, row, stops=None):
        return abs(len([x for x in set(row['question1'])
                        if x not in stops]) -
                   len([x for x in set(row['question2']) if x not in stops]))

    def wc_ratio_unique_stop(self, row, stops=None):
        l1 = len([x for x in set(row['question1']) if x not in stops])*1.0
        l2 = len([x for x in set(row['question2']) if x not in stops])
        if l2 == 0:
            return np.nan
        if l1 / l2:
            return l2 / l1
        else:
            return l1 / l2

    def same_start_word(self, row):
        if not row['question1'] or not row['question2']:
            return np.nan
        return int(row['question1'][0] == row['question2'][0])

    def char_diff(self, row):
        return abs(len(''.join(row['question1'])) -
                   len(''.join(row['question2'])))

    def char_ratio(self, row):
        l1 = len(''.join(row['question1']))
        l2 = len(''.join(row['question2']))
        if l2 == 0:
            return np.nan
        if l1 / l2:
            return l2 / l1
        else:
            return l1 / l2

    def char_diff_unique_stop(self, row, stops=None):
        return abs(len(''.join([x for x in set(row['question1'])
                                if x not in stops])) -
                   len(''.join([x for x in set(row['question2'])
                                if x not in stops])))

    def get_weight(self, count, eps=10000, min_count=2):
        if count < min_count:
            return 0
        else:
            return 1 / (count + eps)

    def tfidf_word_match_share_stops(self, row, stops=None, weights=None):
        q1words = {}
        q2words = {}
        for word in row['question1']:
            if word not in stops:
                q1words[word] = 1
        for word in row['question2']:
            if word not in stops:
                q2words[word] = 1
        if len(q1words) == 0 or len(q2words) == 0:
            # The computer-generated chaff includes a few questions
            # that are nothing but stopwords
            return 0

        shared_weights = [weights.get(w, 0)
                          for w in q1words.keys()
                          if w in q2words] + \
                             [weights.get(w, 0)
                              for w in q2words.keys()
                              if w in q1words]
        total_weights = [weights.get(w, 0) for w in q1words] + \
                        [weights.get(w, 0) for w in q2words]

        R = np.sum(shared_weights) / np.sum(total_weights)
        return R

    def tfidf_word_match_share(self, row, weights=None):
        q1words = {}
        q2words = {}
        for word in row['question1']:
            q1words[word] = 1
        for word in row['question2']:
            q2words[word] = 1
        if len(q1words) == 0 or len(q2words) == 0:
            # The computer-generated chaff includes a few questions
            # that are nothing but stopwords
            return 0

        shared_weights = [weights.get(w, 0)
                          for w in q1words.keys()
                          if w in q2words] + \
                             [weights.get(w, 0)
                              for w in q2words.keys()
                              if w in q1words]
        total_weights = [weights.get(w, 0)
                         for w in q1words] + \
                             [weights.get(w, 0)
                              for w in q2words]

        R = np.sum(shared_weights) / np.sum(total_weights)
        return R

    def build_features(self, data, stops, weights):
        X = pd.DataFrame()
        f = functools.partial(self.word_match_share, stops=stops)
        X['word_match'] = data.apply(f, axis=1, raw=True)

        f = functools.partial(self.tfidf_word_match_share, weights=weights)
        X['tfidf_wm'] = data.apply(f, axis=1, raw=True)

        f = functools.partial(self.tfidf_word_match_share_stops,
                              stops=stops, weights=weights)
        X['tfidf_wm_stops'] = data.apply(f, axis=1, raw=True)

        X['jaccard'] = data.apply(self.jaccard, axis=1, raw=True)
        X['wc_diff'] = data.apply(self.wc_diff, axis=1, raw=True)
        X['wc_ratio'] = data.apply(self.wc_ratio, axis=1, raw=True)
        X['wc_diff_unique'] = data.apply(self.wc_diff_unique,
                                         axis=1, raw=True)
        X['wc_ratio_unique'] = data.apply(self.wc_ratio_unique,
                                          axis=1, raw=True)

        f = functools.partial(self.wc_diff_unique_stop, stops=stops)
        X['wc_diff_unq_stop'] = data.apply(f, axis=1, raw=True)

        f = functools.partial(self.wc_ratio_unique_stop, stops=stops)
        X['wc_ratio_unique_stop'] = data.apply(f, axis=1, raw=True)

        X['same_start'] = data.apply(self.same_start_word, axis=1, raw=True)

        X['char_diff'] = data.apply(self.char_diff, axis=1, raw=True)

        f = functools.partial(self.char_diff_unique_stop, stops=stops)
        X['char_diff_unq_stop'] = data.apply(f, axis=1, raw=True)

        X['total_unique_words'] = data.apply(self.total_unique_words,
                                             axis=1, raw=True)

        f = functools.partial(self.total_unq_words_stop, stops=stops)
        X['total_unq_words_stop'] = data.apply(f, axis=1, raw=True)

        X['char_ratio'] = data.apply(self.char_ratio, axis=1, raw=True)

        return X

    def run(self):
        if exists(self.CUSTOM_FEATURES_TRAIN) and exists(self.CUSTOM_FEATURES_TEST):
            print("Using cached features for {}..."
                  .format(self.TRAIN_DATA_FILENAME))
        else:
            print("Processing the training data set...")
            
            df_train = pd.read_csv(self.TRAIN_DATA_FILE,
                                   encoding="utf-8").fillna(" ")
            df_train['question1'] = df_train['question1']\
                                    .map(lambda x: str(x).lower().split())
            df_train['question2'] = df_train['question2']\
                                    .map(lambda x: str(x).lower().split())
    
            train_qs = pd.Series(df_train['question1'].tolist() +
                                 df_train['question2'].tolist())
    
            words = [x for y in train_qs for x in y]
            counts = Counter(words)
            weights = {word: self.get_weight(count)
                       for word, count in counts.items()}
    
            stops = set(stopwords.words("english"))
            if exists(self.CUSTOM_FEATURES_TRAIN):
                print("Using cached features the training data set...")
            else:
                print("Computing features for the training data set...")
                X_train = self.build_features(df_train, stops, weights)
                print("Saving...")
                X_train.to_csv(self.CUSTOM_FEATURES_TRAIN, index=False)
            if exists(self.CUSTOM_FEATURES_TEST):
                print("Using cached features the test data set...")
            else:
                print("Processing the testing data set...")
                df_test = pd.read_csv(self.TEST_DATA_FILE,
                                       encoding="utf-8").fillna(" ")
                df_test['question1'] = df_test['question1']\
                                        .map(lambda x: str(x).lower().split())
                df_test['question2'] = df_test['question2']\
                                        .map(lambda x: str(x).lower().split())
                print("Computing features for the test data set...")
                X_test = self.build_features(df_test, stops, weights)
                print("Saving...")
                X_test.to_csv(self.CUSTOM_FEATURES_TEST, index=False)
        
