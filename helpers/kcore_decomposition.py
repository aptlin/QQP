# Thank you, @tarobxl
# Adapted from https://www.kaggle.com/tarobxl/magic-feature-v2-0-045-gain/notebook
# and https://www.kaggle.com/c/quora-question-pairs/discussion/33371
# * Libraries
import os
from os.path import exists
import time
import re
import csv
import codecs
import numpy as np
import pandas as pd
import networkx as nx

# * Variables
BASE_DIR = 'data/'
TRAIN_DATA_FILENAME = "stopword_clean_train"
TRAIN_DATA_FILE = BASE_DIR + TRAIN_DATA_FILENAME + ".csv"
TEST_DATA_FILE = BASE_DIR + 'test.csv'
TEST_WITH_IDS_FILE = BASE_DIR + 'test_with_ids.csv'


# * Constructor
class KCore_Decomposition:
    def __init__(self,
                 train_data_filename=TRAIN_DATA_FILENAME,
                 test_data_filename=TEST_DATA_FILE,
                 test_with_ids_filename=TEST_WITH_IDS_FILE):

        self.TRAIN_DATA_FILE = BASE_DIR + train_data_filename + ".csv"
        self.TEST_DATA_FILE = test_data_filename
        self.TEST_WITH_IDS_FILE = test_with_ids_filename
        self.KCORE_TRAIN = BASE_DIR + \
                           'kcore/' + \
                           train_data_filename + '-' + \
                           'kcore_train.csv'
        self.KCORE_TEST = BASE_DIR + \
                          'kcore/' + \
                          train_data_filename + '-' \
                          + 'kcore_test.csv'
        self.QUESTION_KCORES = BASE_DIR + \
                               'kcore/' + \
                               train_data_filename + \
                               '-question_kcores.csv'
        self.MAX_QUESTION_KCORES = BASE_DIR + \
                                   'kcore/' + \
                                   train_data_filename + \
                                   '-max_question_kcores.csv'
        
        # containers        
        self.dict_questions = self._generate_dict_questions()        
        self.new_id = 538000 # df_id["qid"].max() ==> 537933
        
        self.df_train = pd.read_csv(self.TRAIN_DATA_FILE, usecols=["qid1", "qid2"])
        if exists(self.TEST_WITH_IDS_FILE):
            self.df_test = pd.read_csv(self.TEST_WITH_IDS_FILE, usecols=["qid1", "qid2"])
        else:
            self._create_test_with_ids()
            self.df_test = pd.read_csv(self.TEST_WITH_IDS_FILE, usecols=["qid1", "qid2"])

    def _generate_dict_questions(self):
        print("Preparing the dictionary of questions.")
        train_orig = pd.read_csv(self.TRAIN_DATA_FILE, header=0)
        test_orig =  pd.read_csv(self.TEST_DATA_FILE, header=0)
        # "id","qid1","qid2","question1","question2","is_duplicate"
        df_id1 = train_orig[["qid1",
                             "question1"]].drop_duplicates(keep="first").copy().reset_index(drop=True)
        df_id2 = train_orig[["qid2",
                             "question2"]].drop_duplicates(keep="first").copy().reset_index(drop=True)

        df_id1.columns = ["qid", "question"]
        df_id2.columns = ["qid", "question"]

        df_id = pd.concat([df_id1, df_id2]).drop_duplicates(keep="first").reset_index(drop=True)
        dict_questions = df_id.set_index('question').to_dict()
        dict_questions = dict_questions["qid"]

        return dict_questions


    def _fetch_id(self, question):
        if question in self.dict_questions:
            return self.dict_questions[question]
        else:
            self.new_id += 1
            self.dict_questions[question] = self.new_id
            return self.new_id

    def _create_test_with_ids(self):
        rows = []        
        if True:
            with open(self.TEST_DATA_FILE, 'r', encoding="utf8") as infile:
                reader = csv.reader(infile, delimiter=",")
                header = next(reader)
                header.append('qid1')
                header.append('qid2')

                if True:
                    pos, max_lines = 0, 10*1000*1000
                    for row in reader:
                        # "test_id","question1","question2"
                        question1 = row[1]
                        question2 = row[2]

                        qid1 = self._fetch_id(question1)
                        qid2 = self._fetch_id(question2)
                        row.append(qid1)
                        row.append(qid2)

                        pos += 1
                        if pos >= max_lines:
                            break
                        rows.append(row)
        with open(self.TEST_WITH_IDS_FILE, "w", encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)


    def _compute_kcore_decomposition(self):
        if exists(self.MAX_QUESTION_KCORES):
            print("Found {}.".format(self.MAX_QUESTION_KCORES))
            cores_dict = pd.read_csv(self.MAX_QUESTION_KCORES,
                                     index_col="qid").to_dict()["max_kcore"]
        else:
            if exists(self.QUESTION_KCORES):
                df_cores = pd.read_csv(self.QUESTION_KCORES, index_col="qid")
                print("Loaded the decomposed graph.")
            else:
                print("Computing the decomposed graph...")
                df = pd.concat([self.df_train, self.df_test])

                g = nx.Graph()
                g.add_nodes_from(df.qid1)
                print("Built the graph.")
                edges = list(df[['qid1', 'qid2']].to_records(index=False))
                print("Saved the edges.")
                g.add_edges_from(edges)
                g.remove_edges_from(g.selfloop_edges())
                df_output = pd.DataFrame(data=g.nodes(), columns=["qid"])

                print("Saved the graph to a dataframe.")

                NB_CORES = 20

                for k in range(2, NB_CORES + 1):
                    fieldname = "kcore{}".format(k)
                    ck = nx.k_core(g, k=k).nodes()
                    df_output[fieldname] = 0
                    df_output.ix[df_output.qid.isin(ck), fieldname] = k
                df_output.to_csv(self.QUESTION_KCORES, index=None)
                print("Decomposed the graph.")                
                df_cores = pd.read_csv(self.QUESTION_KCORES, index_col="qid")
            df_cores.index.names = ["qid"]
            print("Computing the max kcore...")
            df_cores['max_kcore'] = df_cores.apply(lambda row: max(row), axis=1)
            df_cores[['max_kcore']].to_csv(self.MAX_QUESTION_KCORES) # with index
            print("Saved the kcore data.")
            cores_dict = pd.read_csv(self.MAX_QUESTION_KCORES,
                                     index_col="qid").to_dict()["max_kcore"]
            def gen_qid1_max_kcore(row):
                return cores_dict[row["qid1"]]
            def gen_qid2_max_kcore(row):
                return cores_dict[row["qid2"]]
            return (gen_qid1_max_kcore, gen_qid2_max_kcore)
        

    def attach_max_kcore(self):
        if exists(self.KCORE_TRAIN) and exists(self.KCORE_TEST):
            print("Loading kcore decomposition...")
            kcore_train = pd.read_csv(self.KCORE_TRAIN, encoding = 'utf8')
            kcore_test = pd.read_csv(self.KCORE_TEST, encoding = 'utf8')
        else:
            print("Computing kcore decomposition...")
            gen_qid1_max_kcore, gen_qid2_max_kcore = self._compute_kcore_decomposition()
            self.df_train["qid1_max_kcore"] \
                = self.df_train.apply(gen_qid1_max_kcore, axis=1)
            self.df_test["qid1_max_kcore"] \
                = self.df_test.apply(gen_qid1_max_kcore, axis=1)
            
            self.df_train["qid2_max_kcore"] = self.df_train.apply(gen_qid2_max_kcore, axis=1)
            self.df_test["qid2_max_kcore"] = self.df_test.apply(gen_qid2_max_kcore, axis=1)
            
            kcore_train = self.df_train.ix[:, 2:]
            kcore_train.to_csv(self.KCORE_TRAIN, sep=',', encoding='utf-8', index=False)
            
            kcore_test = self.df_test.ix[:, 2:]
            kcore_test.to_csv(self.KCORE_TEST, sep=',', encoding='utf-8', index=False)
        print("Computed the max kcore feature for the data sets.")
        return (kcore_train, kcore_test)            


