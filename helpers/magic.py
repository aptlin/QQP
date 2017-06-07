# Code adapted from @jturkewitz's work.
# See https://www.kaggle.com/jturkewitz/magic-features-0-03-gain
# Thanks, Jared

# * Libraries

import os
from os.path import exists
import numpy as np
import pandas as pd

# * Variables

BASE_DIR = 'data/'
TRAIN_DATA_FILENAME = "stopword_clean_train"
TRAIN_DATA_FILE = BASE_DIR + TRAIN_DATA_FILENAME + '.csv'
TEST_DATA_FILE = BASE_DIR + 'test.csv'
MAGIC_TRAIN = 'magic/' + TRAIN_DATA_FILENAME + "-train.csv"
MAGIC_TEST = 'magic/test.csv'

# * Constructor

class Magic:
    def __init__(self,
                 train_data_filename=TRAIN_DATA_FILENAME,                 
                 test_data_filename=TEST_DATA_FILE):
        self.TRAIN_DATA_FILENAME = train_data_filename
        self.TRAIN_DATA_FILE = BASE_DIR + train_data_filename + '.csv'
        self.TEST_DATA_FILE = test_data_filename
        self.MAGIC_TRAIN = 'magic/' + self.TRAIN_DATA_FILENAME + "-train.csv"
        self.MAGIC_TEST = 'magic/test.csv'

    def _compute_freqs(self):
        print("Computing magic frequencies...")
        
        train_orig =  pd.read_csv(self.TRAIN_DATA_FILE, header=0)
        test_orig =  pd.read_csv(self.TEST_DATA_FILE, header=0)
        
        df1 = train_orig[['question1']].copy()
        df2 = train_orig[['question2']].copy()
        df1_test = test_orig[['question1']].copy()
        df2_test = test_orig[['question2']].copy()

        df2.rename(columns = {'question2':'question1'},inplace=True)
        df2_test.rename(columns = {'question2':'question1'},inplace=True)

        train_questions = df1.append(df2)
        train_questions = train_questions.append(df1_test)
        train_questions = train_questions.append(df2_test)
        
        train_questions.drop_duplicates(subset = ['question1'],inplace=True)

        train_questions.reset_index(inplace=True,drop=True)
        questions_dict = pd.Series(train_questions.index.values,index=train_questions.question1.values).to_dict()
        train_cp = train_orig.copy()
        test_cp = test_orig.copy()
        train_cp.drop(['qid1','qid2'],axis=1,inplace=True)

        test_cp['is_duplicate'] = -1
        test_cp.rename(columns={'test_id':'id'},inplace=True)
        comb = pd.concat([train_cp,test_cp])

        comb['q1_hash'] = comb['question1'].map(questions_dict)
        comb['q2_hash'] = comb['question2'].map(questions_dict)

        q1_vc = comb.q1_hash.value_counts().to_dict()
        q2_vc = comb.q2_hash.value_counts().to_dict()

        def try_apply_dict(x,dict_to_apply):
            try:
                return dict_to_apply[x]
            except KeyError:
                return 0
            #map to frequency space

        comb['q1_freq'] = comb['q1_hash'].map(lambda x: try_apply_dict(x,q1_vc) + try_apply_dict(x,q2_vc))
        comb['q2_freq'] = comb['q2_hash'].map(lambda x: try_apply_dict(x,q1_vc) + try_apply_dict(x,q2_vc))

        train_comb = comb[comb['is_duplicate'] >= 0][['id','q1_hash','q2_hash','q1_freq','q2_freq','is_duplicate']]
        test_comb = comb[comb['is_duplicate'] < 0][['id','q1_hash','q2_hash','q1_freq','q2_freq']]

        print("Done.")

        return (train_comb, test_comb)

    def spell(self):
        if exists(self.MAGIC_TRAIN) and exists(self.MAGIC_TEST):
            print("Magic features for {} and {} have already been computed.".format(self.TRAIN_DATA_FILENAME,
                                                                                    self.TEST_DATA_FILE))
        else:            
            magic_train, magic_test = self._compute_freqs()
            magic_train.to_csv(self.MAGIC_TRAIN,
                               sep=',',
                               encoding='utf-8',
                               index=False)
            print("Saved magic features for the train set {}".format(self.TRAIN_DATA_FILENAME))
            magic_test.to_csv(self.MAGIC_TEST,
                              sep=',',
                              encoding='utf-8',
                              index=False)
            print("Saved magic features for the test set {}".format(self.TEST_DATA_FILE))
            
