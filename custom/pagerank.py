# coding: utf-8
# Based on notebook by https://www.kaggle.com/shubh24 
# https://www.kaggle.com/shubh24/pagerank-on-quora-a-basic-implementation
# See also the work of ZFTurbo: https://kaggle.com/zfturbo
import pandas as pd
import hashlib
import gc 
import time

BASE_DIR = 'data/'
TRAIN_DATA_FILENAME = "vanilla_train"
TRAIN_DATA_FILE = BASE_DIR + TRAIN_DATA_FILENAME + '.csv'
TEST_DATA_FILE = BASE_DIR + 'test.csv'

PAGERANK_TRAIN = 'custom/' + \
                 TRAIN_DATA_FILENAME + \
                 "-pagerank-train.csv"
PAGERANK_TEST = 'custom/' + "pagerank-test.csv"

start_time = time.time()


df_train = pd.read_csv(TRAIN_DATA_FILE).fillna("")
df_test = pd.read_csv(TEST_DATA_FILE).fillna("")


# Generating a graph of Questions and their neighbors
def generate_qid_graph_table(row):
    hash_key1 = hashlib.md5(row["question1"].encode('utf-8')).hexdigest()
    hash_key2 = hashlib.md5(row["question2"].encode('utf-8')).hexdigest()

    qid_graph.setdefault(hash_key1, []).append(hash_key2)
    qid_graph.setdefault(hash_key2, []).append(hash_key1)


qid_graph = {}
print('Generating a qid graph for the train dataframe...')
df_train.apply(generate_qid_graph_table, axis=1)
print('Generating a qid graph for the test dataframe...')
df_test.apply(generate_qid_graph_table, axis=1)


def pagerank():
    MAX_ITER = 20
    d = 0.85

    # Initializing -- every node gets a uniform value!
    pagerank_dict = {i: 1 / len(qid_graph) for i in qid_graph}
    num_nodes = len(pagerank_dict)

    for iter in range(0, MAX_ITER):

        for node in qid_graph:
            local_pr = 0

            for neighbor in qid_graph[node]:
                local_pr += pagerank_dict[neighbor] / len(qid_graph[neighbor])

            pagerank_dict[node] = (1 - d) / num_nodes + d * local_pr

    return pagerank_dict

print('Building the main PR generator...')
pagerank_dict = pagerank()

def get_pagerank_value(row):
    try:
        if row['id'] % 10000 == 0:
            elapsed = time.time() - start_time
            print("Processed {:10.0f} questions in {:10.0f} s ".format(row['id'], elapsed))
    except KeyError:
        if row['test_id'] % 10000 == 0:
            elapsed = time.time() - start_time
            print("Processed {:10.0f} questions in {:10.0f} s ".format(row['test_id'], elapsed))
    q1 = hashlib.md5(row["question1"].encode('utf-8')).hexdigest()
    q2 = hashlib.md5(row["question2"].encode('utf-8')).hexdigest()
    s = pd.Series({
        "q1_pr": pagerank_dict[q1],
        "q2_pr": pagerank_dict[q2]
    })
    return s

print('Computing pageranks for the train dataframe...')
pagerank_feats_train = df_train.apply(get_pagerank_value, axis=1)
print('Writing the pageranks...')
pagerank_feats_train.to_csv(PAGERANK_TRAIN, index=False)
del df_train
gc.collect()
print('Computing pageranks for the test dataframe...')
pagerank_feats_test = df_test.apply(get_pagerank_value, axis=1)
print('Writing the pageranks...')
pagerank_feats_test.to_csv(PAGERANK_TEST, index=False)

