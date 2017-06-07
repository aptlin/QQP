# -*- coding: utf-8 -*-
# * Libraries

import os
from os.path import exists
import numpy as np
import pandas as pd

import functools
import itertools
import time

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

class CountsFeatures:
    def __init__(self,
                 train_data_filename=TRAIN_DATA_FILENAME,
                 test_data_filename=TEST_DATA_FILE):

        self.TRAIN_DATA_FILENAME = train_data_filename
        self.TRAIN_DATA_FILE = BASE_DIR + train_data_filename + '.csv'
        self.TEST_DATA_FILE = test_data_filename
        self.CUSTOM_FEATURES_TRAIN = 'custom/' + \
                                     train_data_filename + \
                                     "-counts-train.csv"
        self.CUSTOM_FEATURES_TEST = 'custom/counts-test.csv'

    def stems_freq(self, row):
        
        try:
            if row['id'] % 10000 == 0:
                elapsed = time.time() - start_time
                print("Processed {:10.0f} questions in {:10.0f} s ".format(row['id'], elapsed))
        except KeyError:
            if row['test_id'] % 10000 == 0:
                elapsed = time.time() - start_time
                print("Processed {:10.0f} questions in {:10.0f} s ".format(row['test_id'], elapsed))                
        q1stems = {}
        q2stems = {}

        stemmer = SnowballStemmer('english')

        for word in row['question1']:
            stem = stemmer.stem(word)
            try:
                q1stems[stem] += 1
            except KeyError:
                q1stems[stem] = 1

        for word in row['question2']:
            stem = stemmer.stem(word)
            try:
                q2stems[stem] += 1
            except KeyError:
                q2stems[stem] = 1

        if len(q1stems) == 0 or len(q2stems) == 0:
            return 0
        q1freqs = {}
        for stem in q1stems:
            q1freqs[stem] = q1stems[stem] / len(row['question1'])
        q2freqs = {}
        for stem in q2stems:
            q2freqs[stem] = q2stems[stem] / len(row['question2'])

        score = 0
        for stem in q1freqs:
            if stem in q2freqs:
                score += q2freqs[stem]
            else:
                score -= q1freqs[stem]
        for stem in q2freqs:
            if stem in q1freqs:
                score += q1freqs[stem]
            else:
                score -= q2freqs[stem]

        R = score
        return R

    def stems_share(self, row):
        
        try:
            if row['id'] % 10000 == 0:
                elapsed = time.time() - start_time
                print("Processed {:10.0f} questions in {:10.0f} s ".format(row['id'], elapsed))
        except KeyError:
            if row['test_id'] % 10000 == 0:
                elapsed = time.time() - start_time
                print("Processed {:10.0f} questions in {:10.0f} s ".format(row['test_id'], elapsed))        

        q1stems = {}
        q2stems = {}

        stemmer = SnowballStemmer('english')

        for word in row['question1']:
            stem = stemmer.stem(word)
            try:
                q1stems[stem] += 1
            except KeyError:
                q1stems[stem] = 1

        for word in row['question2']:
            stem = stemmer.stem(word)
            try:
                q2stems[stem] += 1
            except KeyError:
                q2stems[stem] = 1

        if len(q1stems) == 0 or len(q2stems) == 0:
            return 0

        shared_stems_in_q1 = [s for s in q1stems.keys() if s in q2stems]
        shared_stems_in_q2 = [s for s in q2stems.keys() if s in q1stems]

        R = (len(shared_stems_in_q1) +
             len(shared_stems_in_q2))/(len(q1stems) + len(q2stems))
        return R

    def stems_weighted_difference(self, row):
        
        try:
            if row['id'] % 10000 == 0:
                elapsed = time.time() - start_time
                print("Processed {:10.0f} questions in {:10.0f} s ".format(row['id'], elapsed))
        except KeyError:
            if row['test_id'] % 10000 == 0:
                elapsed = time.time() - start_time
                print("Processed {:10.0f} questions in {:10.0f} s ".format(row['test_id'], elapsed))

        q1stems = {}
        q2stems = {}

        stemmer = SnowballStemmer('english')

        for word in row['question1']:
            stem = stemmer.stem(word)
            try:
                q1stems[stem] += 1
            except KeyError:
                q1stems[stem] = 1

        for word in row['question2']:
            stem = stemmer.stem(word)
            try:
                q2stems[stem] += 1
            except KeyError:
                q2stems[stem] = 1

        if len(q1stems) == 0 or len(q2stems) == 0:
            return 0

        q1freqs = {}
        for stem in q1stems:
            q1freqs[stem] = q1stems[stem] / len(row['question1'])
        q2freqs = {}
        for stem in q2stems:
            q2freqs[stem] = q2stems[stem] / len(row['question2'])

        unique_stems_in_q1 = [s for s in q1stems.keys() if s not in q2stems]
        unique_stems_in_q2 = [s for s in q2stems.keys() if s not in q1stems]

        score = 0

        rel_q1freq = sum([q1freqs[s] for s in unique_stems_in_q1])
        rel_q2freq = sum([q2freqs[s] for s in unique_stems_in_q2])
        for stem in unique_stems_in_q1:
            score += q1freqs[stem]

        for stem in unique_stems_in_q2:
            score += q2freqs[stem]

        score /= 2

        return score

    def stems_tversky_index(self, row):
        
        try:
            if row['id'] % 10000 == 0:
                elapsed = time.time() - start_time
                print("Processed {:10.0f} questions in {:10.0f} s ".format(row['id'], elapsed))
        except KeyError:
            if row['test_id'] % 10000 == 0:
                elapsed = time.time() - start_time
                print("Processed {:10.0f} questions in {:10.0f} s ".format(row['test_id'], elapsed))

        alpha = 0.7
        q1stems = {}
        q2stems = {}

        stemmer = SnowballStemmer('english')

        for word in row['question1']:
            stem = stemmer.stem(word)
            try:
                q1stems[stem] += 1
            except KeyError:
                q1stems[stem] = 1

        for word in row['question2']:
            stem = stemmer.stem(word)
            try:
                q2stems[stem] += 1
            except KeyError:
                q2stems[stem] = 1

        if len(q1stems) == 0 or len(q2stems) == 0:
            return 0

        unique_stems_in_q1 = [s for s in q1stems.keys() if s not in q2stems]
        unique_stems_in_q2 = [s for s in q2stems.keys() if s not in q1stems]
        shared_stems = [s for s in q1stems.keys() if s in q2stems]

        index = len(shared_stems) / (len(shared_stems)
                                     + (1-alpha) * (alpha * min(len(unique_stems_in_q1),
                                                                len(unique_stems_in_q2)) +
                                                    (1-alpha)*max(len(unique_stems_in_q1),
                                                                  len(unique_stems_in_q2))))

        return index

    def most_freq_2_sdf(self, row):
        
        try:
            if row['id'] % 10000 == 0:
                elapsed = time.time() - start_time
                print("Processed {:10.0f} questions in {:10.0f} s ".format(row['id'], elapsed))
        except KeyError:
            if row['test_id'] % 10000 == 0:
                elapsed = time.time() - start_time
                print("Processed {:10.0f} questions in {:10.0f} s ".format(row['test_id'], elapsed))

        # implementation of MostFreqKSDF taken from
        # https://rosettacode.org/wiki/Most_frequent_k_chars_distance#Python
        import collections
        def MostFreqKHashing(inputString, K):
            occuDict = collections.defaultdict(int)
            for c in inputString:
                occuDict[c] += 1
            occuList = sorted(occuDict.items(), key = lambda x: x[1], reverse = True)
            outputDict = collections.OrderedDict(occuList[:K])
            #Return OrdredDict instead of string for faster lookup.
            return outputDict

        def MostFreqKSimilarity(inputStr1, inputStr2):
            similarity = 0
            for c, cnt1 in inputStr1.items():
                #Reduce the time complexity of lookup operation to about O(1).
                if c in inputStr2:
                    cnt2 = inputStr2[c]
                    similarity += cnt1 + cnt2
            return similarity

        def MostFreqKSDF(inputStr1, inputStr2, K, maxDistance):
            return maxDistance - MostFreqKSimilarity(MostFreqKHashing(inputStr1,K),
                                                     MostFreqKHashing(inputStr2,K))

        q1stems = {}
        q2stems = {}
        stops = set(stopwords.words("english"))
        stemmer = SnowballStemmer('english')

        for word in row['question1']:
            if word not in stops:
                stem = stemmer.stem(word)
                try:
                    q1stems[stem] += 1
                except KeyError:
                    q1stems[stem] = 1

        for word in row['question2']:
            if word not in stops:
                stem = stemmer.stem(word)
                try:
                    q2stems[stem] += 1
                except KeyError:
                    q2stems[stem] = 1

        if len(q1stems) == 0 or len(q2stems) == 0:
            return 0

        q1freqs = {}
        for stem in q1stems:
            q1freqs[stem] = q1stems[stem] / len(row['question1'])
        q2freqs = {}
        for stem in q2stems:
            q2freqs[stem] = q2stems[stem] / len(row['question2'])

        import operator
        sorted_q1freqs_sample = sorted(q1freqs.items(), key=operator.itemgetter(1))[:4]
        sorted_q2freqs_sample = sorted(q1freqs.items(), key=operator.itemgetter(1))[:4]

        q1set = set()
        q2set = set()

        for word, _ in sorted_q1freqs_sample:
            q1set.add(word)
        for word, _ in sorted_q2freqs_sample:
            q2set.add(word)

        if len(q1set) == 0 or len(q2set) == 0:
            return 0
        elif len(q1set) <= len(q2set):
            q_min = q1set
            q_max = q2set
        else:
            q_min = q2set
            q_max = q1set

        combinations = [zip(x,q_min) for x in itertools.permutations(q_max,len(q_min))]

        R = 0
        maxDistance = 100
        K = 2

        for cycle in combinations:
            for word1, word2 in cycle:
                R += MostFreqKSDF(word1, word2, K, maxDistance)

        R /= len(q1set) + len(q2set)
        return R
    def most_freq_2_sdf_reverse(self, row):
        
        try:
            if row['id'] % 10000 == 0:
                elapsed = time.time() - start_time
                print("Processed {:10.0f} questions in {:10.0f} s ".format(row['id'], elapsed))
        except KeyError:
            if row['test_id'] % 10000 == 0:
                elapsed = time.time() - start_time
                print("Processed {:10.0f} questions in {:10.0f} s ".format(row['test_id'], elapsed))

        # implementation of MostFreqKSDF taken from
        # https://rosettacode.org/wiki/Most_frequent_k_chars_distance#Python
        import collections
        def MostFreqKHashing(inputString, K):
            occuDict = collections.defaultdict(int)
            for c in inputString:
                occuDict[c] += 1
            occuList = sorted(occuDict.items(), key = lambda x: x[1], reverse = True)
            outputDict = collections.OrderedDict(occuList[:K])
            #Return OrdredDict instead of string for faster lookup.
            return outputDict

        def MostFreqKSimilarity(inputStr1, inputStr2):
            similarity = 0
            for c, cnt1 in inputStr1.items():
                #Reduce the time complexity of lookup operation to about O(1).
                if c in inputStr2:
                    cnt2 = inputStr2[c]
                    similarity += cnt1 + cnt2
            return similarity

        def MostFreqKSDF(inputStr1, inputStr2, K, maxDistance):
            return maxDistance - MostFreqKSimilarity(MostFreqKHashing(inputStr1,K),
                                                     MostFreqKHashing(inputStr2,K))

        q1stems = {}
        q2stems = {}
        stops = set(stopwords.words("english"))
        stemmer = SnowballStemmer('english')

        for word in row['question1']:
            if word not in stops:
                stem = stemmer.stem(word)
                try:
                    q1stems[stem] += 1
                except KeyError:
                    q1stems[stem] = 1

        for word in row['question2']:
            if word not in stops:
                stem = stemmer.stem(word)
                try:
                    q2stems[stem] += 1
                except KeyError:
                    q2stems[stem] = 1

        if len(q1stems) == 0 or len(q2stems) == 0:
            return 0

        q1freqs = {}
        for stem in q1stems:
            q1freqs[stem] = q1stems[stem] / len(row['question1'])
        q2freqs = {}
        for stem in q2stems:
            q2freqs[stem] = q2stems[stem] / len(row['question2'])

        import operator
        sorted_q1freqs_sample = sorted(q1freqs.items(), key=operator.itemgetter(1), reverse=True)[:4]
        sorted_q2freqs_sample = sorted(q1freqs.items(), key=operator.itemgetter(1), reverse=True)[:4]

        q1set = set()
        q2set = set()

        for word, _ in sorted_q1freqs_sample:
            q1set.add(word)
        for word, _ in sorted_q2freqs_sample:
            q2set.add(word)

        if len(q1set) == 0 or len(q2set) == 0:
            return 0
        elif len(q1set) <= len(q2set):
            q_min = q1set
            q_max = q2set
        else:
            q_min = q2set
            q_max = q1set

        combinations = [zip(x,q_min) for x in itertools.permutations(q_max,len(q_min))]

        R = 0
        maxDistance = 100
        K = 2

        for cycle in combinations:
            for word1, word2 in cycle:
                R += MostFreqKSDF(word1, word2, K, maxDistance)

        R /= len(q1set) + len(q2set)
        return R    

    def wagner_fischer(self, row):
        
        try:
            if row['id'] % 10000 == 0:
                elapsed = time.time() - start_time
                print("Processed {:10.0f} questions in {:10.0f} s ".format(row['id'], elapsed))
        except KeyError:
            if row['test_id'] % 10000 == 0:
                elapsed = time.time() - start_time
                print("Processed {:10.0f} questions in {:10.0f} s ".format(row['test_id'], elapsed))
        from wagnerfischerpp import WagnerFischer
        q1stems = {}
        q2stems = {}
        stops = set(stopwords.words("english"))
        stemmer = SnowballStemmer('english')

        for word in row['question1']:
            if word not in stops:
                stem = stemmer.stem(word)
                try:
                    q1stems[stem] += 1
                except KeyError:
                    q1stems[stem] = 1

        for word in row['question2']:
            if word not in stops:
                stem = stemmer.stem(word)
                try:
                    q2stems[stem] += 1
                except KeyError:
                    q2stems[stem] = 1

        if len(q1stems) == 0 or len(q2stems) == 0:
            return 0

        q1freqs = {}
        for stem in q1stems:
            q1freqs[stem] = q1stems[stem] / len(row['question1'])
        q2freqs = {}
        for stem in q2stems:
            q2freqs[stem] = q2stems[stem] / len(row['question2'])

        import operator
        sorted_q1freqs_sample = sorted(q1freqs.items(), key=operator.itemgetter(1))[:3]
        sorted_q2freqs_sample = sorted(q1freqs.items(), key=operator.itemgetter(1))[:3]
        
        q1set = set()
        q2set = set()

        for word, _ in sorted_q1freqs_sample:
            q1set.add(word)
        for word, _ in sorted_q2freqs_sample:
            q2set.add(word)

        if len(q1set) == 0 or len(q2set) == 0:
            return 0
        elif len(q1set) <= len(q2set):
            q_min = q1set
            q_max = q2set
        else:
            q_min = q2set
            q_max = q1set

        combinations = [zip(x,q_min) for x in itertools.permutations(q_max,len(q_min))]

        R = 0
        for cycle in combinations:
            for word1, word2 in cycle:
                R += WagnerFischer(word1, word2).cost

        R /= len(q1set) + len(q2set)
        return R

    def wagner_fischer_reverse(self, row):
        
        try:
            if row['id'] % 10000 == 0:
                elapsed = time.time() - start_time
                print("Processed {:10.0f} questions in {:10.0f} s ".format(row['id'], elapsed))
        except KeyError:
            if row['test_id'] % 10000 == 0:
                elapsed = time.time() - start_time
                print("Processed {:10.0f} questions in {:10.0f} s ".format(row['test_id'], elapsed))
        from wagnerfischerpp import WagnerFischer
        q1stems = {}
        q2stems = {}
        stops = set(stopwords.words("english"))
        stemmer = SnowballStemmer('english')

        for word in row['question1']:
            if word not in stops:
                stem = stemmer.stem(word)
                try:
                    q1stems[stem] += 1
                except KeyError:
                    q1stems[stem] = 1

        for word in row['question2']:
            if word not in stops:
                stem = stemmer.stem(word)
                try:
                    q2stems[stem] += 1
                except KeyError:
                    q2stems[stem] = 1

        if len(q1stems) == 0 or len(q2stems) == 0:
            return 0

        q1freqs = {}
        for stem in q1stems:
            q1freqs[stem] = q1stems[stem] / len(row['question1'])
        q2freqs = {}
        for stem in q2stems:
            q2freqs[stem] = q2stems[stem] / len(row['question2'])

        import operator
        sorted_q1freqs_sample = sorted(q1freqs.items(), key=operator.itemgetter(1), reverse=True)[:3]
        sorted_q2freqs_sample = sorted(q1freqs.items(), key=operator.itemgetter(1), reverse=True)[:3]
        q1set = set()
        q2set = set()

        for word, _ in sorted_q1freqs_sample:
            q1set.add(word)
        for word, _ in sorted_q2freqs_sample:
            q2set.add(word)

        if len(q1set) == 0 or len(q2set) == 0:
            return 0
        elif len(q1set) <= len(q2set):
            q_min = q1set
            q_max = q2set
        else:
            q_min = q2set
            q_max = q1set

        combinations = [zip(x,q_min) for x in itertools.permutations(q_max,len(q_min))]

        R = 0
        for cycle in combinations:
            for word1, word2 in cycle:
                R += WagnerFischer(word1, word2).cost

        R /= len(q1set) + len(q2set)
        return R

    def build_features(self, data):
        X = pd.DataFrame()
        # commented features give poor differentiability results
        # print("Calculating wagner_fischer...")
        # X['wagner_fischer'] = data.apply(self.wagner_fischer, axis=1, raw=True)
        # print("Calculating wagner_fischer_reverse...")
        # X['wagner_fischer_reverse'] = data.apply(self.wagner_fischer_reverse, axis=1, raw=True)        
        # print("Calculating most_freq_2_sdf...")
        # X['most_freq_2_sdf'] = data.apply(self.most_freq_2_sdf, axis=1, raw=True)
        # print("Calculating most_freq_2_sdf_reverse...")
        # X['most_freq_2_sdf_reverse'] = data.apply(self.most_freq_2_sdf_reverse, axis=1, raw=True)
        print("Calculating stems_freq...")
        X['stems_freq'] = data.apply(self.stems_freq, axis=1, raw=True)
        print("Calculating stems_share...")
        X['stems_share'] = data.apply(self.stems_share, axis=1, raw=True)
        print("Calculating stems_weighted_difference...")
        X['stems_weighted_difference'] = data.apply(self.stems_weighted_difference, axis=1, raw=True)
        print("Calculating stems_tversky_index...")
        X['stems_tversky_index'] = data.apply(self.stems_tversky_index, axis=1, raw=True)

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
