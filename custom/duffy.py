# Thanks, Collins and Duffy.
# See https://github.com/goodmattg/quora_kaggle

# * Libraries

import os
from os.path import exists
import numpy as np
import pandas as pd

# * Variables

BASE_DIR = 'data/'
TRAIN_DATA_FILENAME = "vanilla_train"
TRAIN_DATA_FILE = BASE_DIR + TRAIN_DATA_FILENAME + '.csv'
TEST_DATA_FILE = BASE_DIR + 'test.csv'
DUFFY_FEATURES_TRAIN_DIR = 'duffy/' + 'train_collins_duffy/'
DUFFY_FEATURES_TEST_DIR = 'duffy/' + 'test_collins_duffy/'
DUFFY_FEATURES_TRAIN = 'custom/' + TRAIN_DATA_FILENAME + "-duffy-train.csv"
DUFFY_FEATURES_TEST = 'custom/duffy-test.csv'

# * Constructor

class Duffy:
    def __init__(self,
                 train_data_filename=TRAIN_DATA_FILENAME,
                 test_data_filename=TEST_DATA_FILE):
        self.TRAIN_DATA_FILENAME = train_data_filename
        self.TRAIN_DATA_FILE = BASE_DIR + train_data_filename + '.csv'
        self.TEST_DATA_FILE = test_data_filename
        self.DUFFY_TRAIN = 'custom/' + self.TRAIN_DATA_FILENAME + "-duffy-train.csv"
        self.DUFFY_TEST = 'custom/duffy-test.csv'


    def build(self):
        if exists(self.DUFFY_TRAIN) and exists(self.DUFFY_TEST):
            print("Duffy features for {} and {} have already been computed."\
                  .format(self.TRAIN_DATA_FILENAME,
                          self.TEST_DATA_FILE))
        else:
            df_train = pd.read_csv(self.TRAIN_DATA_FILE)
            flip = True
            for filename in os.listdir(DUFFY_FEATURES_TRAIN_DIR):
                if filename.endswith(".csv"):
                    tmpFrame = pd.read_csv(os.path.join(os.getcwd(),
                                                        DUFFY_FEATURES_TRAIN_DIR,
                                                        filename))
                    print("Found:", os.path.join(os.getcwd(),
                                                 DUFFY_FEATURES_TRAIN_DIR,
                                                 filename))
                    tmpFrame = tmpFrame.rename(columns={'cdNorm_st': filename.replace(".csv","")})
                    if flip:
                        duffy_train = tmpFrame
                        flip = False
                    else:
                        duffy_train = duffy_train.merge(tmpFrame,how='inner',on='id')
                    continue
                else:
                    continue
            del tmpFrame, flip

            idx = df_train['id']
            duffy_train = duffy_train.set_index(duffy_train['id'])
            duffy_train = duffy_train.reindex(idx, fill_value=0.0)\
                                     .drop('id', axis=1)

            df_test = pd.read_csv(self.TEST_DATA_FILE)
            flip = True
            for filename in os.listdir(DUFFY_FEATURES_TEST_DIR):
                if filename.endswith(".csv"):
                    tmpFrame = pd.read_csv(os.path.join(os.getcwd(),
                                                        DUFFY_FEATURES_TEST_DIR,
                                                        filename))
                    print("Found:", os.path.join(os.getcwd(),
                                                 DUFFY_FEATURES_TEST_DIR,
                                                 filename))
                    tmpFrame = tmpFrame.rename(columns={'cdNorm_st': filename.replace(".csv","")})
                    if flip:
                        duffy_test = tmpFrame
                        flip = False
                    else:
                        duffy_test = duffy_test.merge(tmpFrame,how='inner',on='id')
                    continue
                else:
                    continue
            del tmpFrame, flip

            idx = df_test['test_id']
            duffy_test = duffy_test.set_index(duffy_test['id'])
            duffy_test = duffy_test.reindex(idx, fill_value=0.0)\
                                   .drop('id', axis=1)


            duffy_train.to_csv(self.DUFFY_TRAIN,
                               sep=',',
                               encoding='utf-8',
                               index=False)
            print("Saved duffy features for the train set {}".format(self.TRAIN_DATA_FILENAME))
            duffy_test.to_csv(self.DUFFY_TEST,
                              sep=',',
                              encoding='utf-8',
                              index=False)
            print("Saved duffy features for the test set {}".format(self.TEST_DATA_FILE))
