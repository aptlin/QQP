# Code inspired by @lystdo, @bradleypallen and @act444
# See https://www.kaggle.com/lystdo/lstm-with-word2vec-embeddings
# and https://github.com/bradleypallen/keras-quora-question-pairs
# and https://www.kaggle.com/act444/lb-0-158-xgb-handcrafted-leaky
# Thank you, @lystdo, @bradleypallen and @act444.

# * Libraries
# import packages
import os
from os.path import exists
import time
import re
import csv
import codecs
import numpy as np
import pandas as pd
import pickle

import argparse
import functools
from collections import defaultdict

import xgboost as xgb

from collections import Counter
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier

from gensim.models import KeyedVectors

# * Variables

# set directories and parameters

BASE_DIR = 'data/'
EMBEDDING_FILE = BASE_DIR + 'GoogleNews-vectors-negative300.bin'
TRAIN_DATA_FILENAME = "stopword_clean_train"
TRAIN_DATA_FILE = BASE_DIR + TRAIN_DATA_FILENAME + '.csv'
# TRAIN_DATA_FILE = BASE_DIR + 'train_small.csv'
TEST_DATA_FILE = BASE_DIR + 'test.csv'
# TEST_DATA_FILE = BASE_DIR + 'test_small.csv'
WORD_EMBEDDING_MATRIX_FILE = BASE_DIR + \
                             TRAIN_DATA_FILENAME + \
                             '-word_embedding_matrix.npy'

MAX_SEQUENCE_LENGTH = 30
MAX_NB_WORDS = 200000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.1

PREPROCESSED = 'preprocessed/'
PREPROCESSED_TRAIN_DATA_Q1 = BASE_DIR + \
                             PREPROCESSED + \
                             TRAIN_DATA_FILENAME + \
                             '-' + \
                             'train_data_q1.npy'
PREPROCESSED_TRAIN_DATA_Q2 = BASE_DIR + \
                             PREPROCESSED + \
                             TRAIN_DATA_FILENAME + \
                             '-' + \
                             'train_data_q2.npy'
PREPROCESSED_TEST_DATA_Q1  = BASE_DIR + \
                             PREPROCESSED + \
                             TRAIN_DATA_FILENAME + \
                             '-' + \
                             'test_data_q1.npy'
PREPROCESSED_TEST_DATA_Q2  = BASE_DIR + \
                             PREPROCESSED + \
                             TRAIN_DATA_FILENAME + \
                             '-' + \
                             'test_data_q2.npy'
PREPROCESSED_LABELS        = BASE_DIR + \
                             PREPROCESSED + \
                             TRAIN_DATA_FILENAME + \
                             '-' + \
                             'labels.npy'
PREPROCESSED_TEST_IDS      = BASE_DIR + \
                             PREPROCESSED + \
                             TRAIN_DATA_FILENAME + \
                             '-' + \
                             'test_ids.npy'
PREPROCESSED_WORD_INDEX    = BASE_DIR + \
                             PREPROCESSED + \
                             TRAIN_DATA_FILENAME + \
                             '-' + \
                             'word_index.pkl'

NUM_LSTM = np.random.randint(175, 275)
NUM_DENSE = np.random.randint(190, 275)
RATE_DROP_DENSE = (1 + np.random.rand()) * 0.25
RATE_DROP_LSTM = (1 + np.random.rand()) * 0.25
RECTIFIER = 'relu'
REWEIGH = True
NB_FILTER = 64
FILTER_LENGTH = 5

ABHISHEK_TRAIN = 'abhishek/train_features.csv'
ABHISHEK_TEST = 'abhishek/test_features.csv'

MAGIC_TRAIN = 'magic/' + TRAIN_DATA_FILENAME + "-train.csv"
MAGIC_TEST = 'magic/test.csv'

MAGIC_II_TRAIN = 'magic2/train_ic.csv'
MAGIC_II_TEST = 'magic2/test_ic.csv'

CUSTOM_FEATURES_TRAIN = 'custom/' + TRAIN_DATA_FILENAME + "-train.csv"
CUSTOM_FEATURES_TEST = 'custom/test.csv'


# * Constructor

class Tet:
    def __init__(self,
                 train_data_filename=TRAIN_DATA_FILE,
                 test_data_filename=TEST_DATA_FILE,
                 embeddings_filename=EMBEDDING_FILE,
                 max_seq_length=MAX_SEQUENCE_LENGTH,
                 max_nb_words=MAX_NB_WORDS,
                 embeddings_dim=EMBEDDING_DIM,
                 validation_split=VALIDATION_SPLIT,
                 num_dense=NUM_DENSE,
                 num_lstm=NUM_LSTM,
                 rate_drop_dense=RATE_DROP_DENSE,
                 rate_drop_lstm=RATE_DROP_LSTM,
                 rectifier=RECTIFIER,
                 reweigh=REWEIGH):
        self.TRAIN_DATA_FILE = train_data_filename
        self.TEST_DATA_FILE = test_data_filename
        self.EMBEDDING_FILE = embeddings_filename

        ### save texts and data
        ## training
        self.texts_1 = []
        self.texts_2 = []
        self.labels = []
        # with codecs.open(self.TRAIN_DATA_FILE, encoding='utf-8') as f:
        #     reader = csv.reader(f, delimiter=',')
        #     header = next(reader)
        #     for values in reader:
        #         self.texts_1.append(values[3])
        #         self.texts_2.append(values[4])
        #         self.labels.append(int(values[5]))
        # print("Saved {} question pairs.".format(len(self.texts_1)))
        ## testing
        # self.test_texts_1 = []
        # self.test_texts_2 = []
        # self.test_ids = []
        # with codecs.open(self.TEST_DATA_FILE, encoding='utf-8') as f:
        #     reader = csv.reader(f, delimiter=',')
        #     header = next(reader)
        #     for values in reader:
        #         self.test_texts_1.append(values[1])
        #         self.test_texts_2.append(values[2])
        #         self.test_ids.append(values[0])
        ## word2vec indexing
        # self.word_index = {}



        self.MAX_SEQUENCE_LENGTH = max_seq_length
        self.MAX_NB_WORDS = max_nb_words
        self.EMBEDDING_DIM = embeddings_dim
        self.VALIDATION_SPLIT = validation_split

        self.NUM_DENSE = num_dense
        self.NUM_LSTM = num_lstm
        self.RATE_DROP_DENSE = rate_drop_dense
        self.RATE_DROP_LSTM = rate_drop_lstm
        self.RECTIFIER = rectifier
        self.REWEIGH = reweigh
                
        
        self.STAMP = 'tet_xgboost'

        # a container for the computed model
        self.model = {}

        # add custom-engineered features

        self.train_custom_features, self.test_custom_features = self._engineered_features()

# * Preprocessing

    # def _preprocess_data(self):
    #     # process texts in datasets

    #     print('Processing text dataset...')

    #     if exists(PREPROCESSED_TRAIN_DATA_Q1) and \
    #        exists(PREPROCESSED_TRAIN_DATA_Q2) and \
    #        exists(PREPROCESSED_TEST_DATA_Q1) and \
    #        exists(PREPROCESSED_TEST_DATA_Q2) and \
    #        exists(PREPROCESSED_LABELS) and \
    #        exists(PREPROCESSED_TEST_IDS) and \
    #        exists(PREPROCESSED_WORD_INDEX):
    #         data_1 = np.load(open(PREPROCESSED_TRAIN_DATA_Q1, 'rb'))
    #         data_2 = np.load(open(PREPROCESSED_TRAIN_DATA_Q2, 'rb'))
    #         labels = np.load(open(PREPROCESSED_LABELS, 'rb'))
    #         test_data_1 = np.load(open(PREPROCESSED_TEST_DATA_Q1, 'rb'))
    #         test_data_2 = np.load(open(PREPROCESSED_TEST_DATA_Q2, 'rb'))
    #         test_ids = np.load(open(PREPROCESSED_TEST_IDS, 'rb'))
    #         with open(PREPROCESSED_WORD_INDEX, 'rb') as f:
    #             self.word_index = pickle.load(f)

    #     else:
    #         tokenizer = Tokenizer(num_words=self.MAX_NB_WORDS)
    #         tokenizer.fit_on_texts(self.texts_1 +
    #                                self.texts_2 +
    #                                self.test_texts_1 +
    #                                self.test_texts_2)
    #         sequences_1 = tokenizer.texts_to_sequences(self.texts_1)
    #         sequences_2 = tokenizer.texts_to_sequences(self.texts_2)
    #         test_sequences_1 = tokenizer.texts_to_sequences(self.test_texts_1)
    #         test_sequences_2 = tokenizer.texts_to_sequences(self.test_texts_2)
    #         self.word_index = tokenizer.word_index
    #         print('Found %s unique tokens' % len(self.word_index))
    #         data_1 = pad_sequences(sequences_1,
    #                                maxlen=self.MAX_SEQUENCE_LENGTH)
    #         data_2 = pad_sequences(sequences_2,
    #                                maxlen=self.MAX_SEQUENCE_LENGTH)
    #         labels = np.array(self.labels)
    #         print('Shape of data tensor:', data_1.shape)
    #         print('Shape of label tensor:', labels.shape)
    #         test_data_1 = pad_sequences(test_sequences_1,
    #                                     maxlen=self.MAX_SEQUENCE_LENGTH)
    #         test_data_2 = pad_sequences(test_sequences_2,
    #                                     maxlen=self.MAX_SEQUENCE_LENGTH)
    #         test_ids = np.array(self.test_ids)

    #         np.save(open(PREPROCESSED_TRAIN_DATA_Q1, 'wb'), data_1)
    #         np.save(open(PREPROCESSED_TRAIN_DATA_Q2, 'wb'), data_2)
    #         np.save(open(PREPROCESSED_LABELS, 'wb'), labels)
    #         np.save(open(PREPROCESSED_TEST_DATA_Q1, 'wb'), test_data_1)
    #         np.save(open(PREPROCESSED_TEST_DATA_Q2, 'wb'), test_data_2)
    #         np.save(open(PREPROCESSED_TEST_IDS, 'wb'), test_ids)
    #         with open(PREPROCESSED_WORD_INDEX, 'wb') as f:
    #             pickle.dump(self.word_index, f, pickle.HIGHEST_PROTOCOL)
    #         print("Saved the preprocessed data.")

    #     return (data_1,
    #             data_2,
    #             labels,
    #             test_data_1,
    #             test_data_2,
    #             test_ids)

# * Embedding Layer

    # def _create_embedding_layer(self):

    #     # prepare embeddings

    #     print('Preparing embedding matrix')
    #     nb_words = min(self.MAX_NB_WORDS, len(self.word_index)) + 1

    #     # index word vectors

    #     if exists(WORD_EMBEDDING_MATRIX_FILE):
    #         print("Loading a cached embedding matrix.")
    #         embedding_matrix = np.load(open(WORD_EMBEDDING_MATRIX_FILE, 'rb'))
    #     else:
    #         print('Indexing word vectors')
    #         word2vec = KeyedVectors.load_word2vec_format(self.EMBEDDING_FILE,
    #                                                      binary=True)
    #         print('Found %s word vectors of word2vec' % len(word2vec.vocab))
    #         embedding_matrix = np.zeros((nb_words, self.EMBEDDING_DIM))
    #         for word, i in self.word_index.items():
    #             if word in word2vec.vocab:
    #                 embedding_matrix[i] = word2vec.word_vec(word)
    #         print('Null word embeddings: %d' %
    #               np.sum(np.sum(embedding_matrix, axis=1) == 0))
    #         print('Saving the word embeddings.')
    #         np.save(open(WORD_EMBEDDING_MATRIX_FILE, 'wb'), embedding_matrix)


    #     embedding_layer = Embedding(nb_words,
    #                                 self.EMBEDDING_DIM,
    #                                 weights=[embedding_matrix],
    #                                 input_length=self.MAX_SEQUENCE_LENGTH,
    #                                 trainable=False)
    #     return embedding_layer
# * Engineered Features

    def _engineered_features(self):
        # @abhishek's features
        # Thanks to @raddar and @abhishek for the data.
        # See https://www.kaggle.com/c/quora-question-pairs/discussion/31284

        # abhishek_train = pd.read_csv(ABHISHEK_TRAIN, encoding = "ISO-8859-1")
        # abhishek_test = pd.read_csv(ABHISHEK_TEST, encoding = "ISO-8859-1")

        # abhishek_train_features = abhishek_train.ix[:404176, 9:30]\
        #                                         .replace([np.inf,-np.inf],0)\
        #                                         .drop('jaccard_distance', axis=1)\
        #                                         .drop('euclidean_distance', axis=1)

        # abhishek_test_features = abhishek_test.ix[:, 9:30]\
        #                                       .replace([np.inf,-np.inf],0)\
        #                                       .drop('jaccard_distance', axis=1)\
        #                                       .drop('euclidean_distance', axis=1)
        # Krzysztof Dziedzic's magic feature II.
        # Data by @Justfor.
        # See https://www.kaggle.com/justfor/edges/code
        # and https://www.kaggle.com/c/quora-question-pairs/discussion/33287

        magic2_train_features =  pd.read_csv(MAGIC_II_TRAIN,
                                             encoding = "utf-8")


        magic2_test_features =  pd.read_csv(MAGIC_II_TEST,
                                            encoding = "utf-8")

        # @tarobxl kcore feature
        # See https://www.kaggle.com/c/quora-question-pairs/discussion/33371
        from kcore_decomposition import KCore_Decomposition
        kd = KCore_Decomposition()
        kcore_train_features, kcore_test_features = kd.attach_max_kcore()

        # @jturkewitz's magic feature
        # See https://www.kaggle.com/jturkewitz/magic-features-0-03-gain
        magic_train_features =  pd.read_csv(MAGIC_TRAIN,
                                             encoding = "utf-8")
        magic_train_features = magic_train_features.ix[:, 3:5]

        magic_test_features =  pd.read_csv(MAGIC_TEST,
                                            encoding = "utf-8")
        magic_test_features = magic_test_features.ix[:, 3:5]

        custom_train_features = pd.read_csv(CUSTOM_FEATURES_TRAIN,
                                            encoding="utf-8")
        custom_test_features = pd.read_csv(CUSTOM_FEATURES_TEST,
                                           encoding="utf-8")        

        train_features = pd.concat([custom_train_features,
                                    #abhishek_train_features,
                                    magic_train_features,
                                    magic2_train_features,
                                    kcore_train_features], axis=1, join='inner').fillna(0)
        test_features = pd.concat([custom_test_features,
                                   #abhishek_test_features,
                                   magic_test_features,
                                   magic2_test_features,
                                   kcore_test_features], axis=1, join='inner').fillna(0)

        return (train_features, test_features)

# * Model Constructor
    def _model_constructor(self):
        ########################################
        ## sample train/validation data
        ########################################
        print("Loading train data...")        
        X_train = self.train_custom_features

        df_train = pd.read_csv(TRAIN_DATA_FILE, encoding="utf-8")

        y_train = df_train['is_duplicate'].values

        X_train, X_valid, y_train, y_valid = train_test_split(X_train,
                                                              y_train,
                                                              test_size=0.1,
                                                              random_state=4242)

        #UPDownSampling
        pos_train = X_train[y_train == 1]
        neg_train = X_train[y_train == 0]
        X_train = pd.concat((neg_train, pos_train.iloc[:int(0.8*len(pos_train))], neg_train))
        y_train = np.array([0] * neg_train.shape[0] + [1] * pos_train.iloc[:int(0.8*len(pos_train))].shape[0] + [0] * neg_train.shape[0])
        print(np.mean(y_train))
        del pos_train, neg_train
    
        pos_valid = X_valid[y_valid == 1]
        neg_valid = X_valid[y_valid == 0]
        X_valid = pd.concat((neg_valid, pos_valid.iloc[:int(0.8 * len(pos_valid))], neg_valid))
        y_valid = np.array([0] * neg_valid.shape[0] + [1] * pos_valid.iloc[:int(0.8 * len(pos_valid))].shape[0] + [0] * neg_valid.shape[0])
        print(np.mean(y_valid))
        del pos_valid, neg_valid

        
    # data_1, data_2, labels, _, _, _ = self._preprocess_data()
        

        # perm = np.random.permutation(len(data_1))

        # idx_train = perm[:int(len(data_1)*(1-self.VALIDATION_SPLIT))]
        # idx_val = perm[int(len(data_1)*(1-self.VALIDATION_SPLIT)):]

        # # data_1_train = np.vstack((data_1[idx_train], data_2[idx_train]))
        # # data_2_train = np.vstack((data_2[idx_train], data_1[idx_train]))
        # data_1_train = np.vstack((data_1[idx_train], data_2[idx_train]))
        # data_2_train = np.vstack((data_2[idx_train], data_1[idx_train]))

        # self.train_custom_features = self.train_custom_features.as_matrix()

        # features_train = np.vstack((self.train_custom_features[idx_train],
        #                             self.train_custom_features[idx_train]))

        # labels_train = np.concatenate((labels[idx_train], labels[idx_train]))

        # data_1_val = np.vstack((data_1[idx_val], data_2[idx_val]))
        # data_2_val = np.vstack((data_2[idx_val], data_1[idx_val]))

        # features_val = np.vstack((self.train_custom_features[idx_val],
        #                           self.train_custom_features[idx_val]))

        # labels_val = np.concatenate((labels[idx_val], labels[idx_val]))

        # weight_val = np.ones(len(labels_val))
        # if self.REWEIGH:
        #     weight_val *= 0.472001959
        #     weight_val[labels_val==0] = 1.309028344

        ########################################
        ## define the model structure
        ########################################

        params = {}
        params['objective'] = 'binary:logistic'
        params['eval_metric'] = 'logloss'
        params['eta'] = 0.02
        params['max_depth'] = 7
        params['subsample'] = 0.6
        params['base_score'] = 0.2
        
        d_train = xgb.DMatrix(X_train, label=y_train)
        d_valid = xgb.DMatrix(X_valid, label=y_valid)
    
        watchlist = [(d_train, 'train'), (d_valid, 'valid')]
        print("Training the model...")
        bst = xgb.train(params, d_train, 2500, watchlist, early_stopping_rounds=50, verbose_eval=50)
        print(log_loss(y_valid, bst.predict(d_valid)))
        bst_val_score = log_loss(y_valid, bst.predict(d_valid))
        self.STAMP = str(bst_val_score) + "_" + self.STAMP
        
        bst.save_model(self.STAMP + '.mdl')

        # embedding_layer = self._create_embedding_layer()
        # # lstm_layer = LSTM(self.NUM_LSTM,
        # #                   dropout=self.RATE_DROP_LSTM,
        # #                   recurrent_dropout=self.RATE_DROP_LSTM)
        # convolution_layer = Convolution1D(nb_filter=NB_FILTER,
        #                                   filter_length=FILTER_LENGTH,
        #                                   border_mode='valid',
        #                                   subsample_length=1)


        # custom_dim = int(self.train_custom_features.shape[1])
        # custom_input = Input(shape=(custom_dim, ), dtype='float32')
        # #custom_features = Dropout(self.RATE_DROP_DENSE)(custom_input)
        # custom_features = Dense(self.NUM_DENSE,
        #                         kernel_initializer='normal')(custom_input)
        # custom_features = PReLU()(custom_features)
        # custom_features = Dropout(self.RATE_DROP_DENSE)(custom_features)
        # custom_features = BatchNormalization()(custom_features)


        # sequence_1_input = Input(shape=(self.MAX_SEQUENCE_LENGTH,),
        #                          dtype='int32')
        # embedded_sequences_1 = embedding_layer(sequence_1_input)
        # # Q1 = lstm_layer(embedded_sequences_1)
        # Q1 = convolution_layer(embedded_sequences_1)
        # Q1 = PReLU()(Q1)
        # Q1 = GlobalMaxPooling1D()(Q1)
        # Q1 = Dropout(self.RATE_DROP_DENSE)(Q1)

        # sequence_2_input = Input(shape=(self.MAX_SEQUENCE_LENGTH,),
        #                          dtype='int32')
        # embedded_sequences_2 = embedding_layer(sequence_2_input)
        # #Q2 = lstm_layer(embedded_sequences_2)
        # Q2 = convolution_layer(embedded_sequences_2)
        # Q2 = PReLU()(Q2)
        # Q2 = GlobalMaxPooling1D()(Q2)
        # Q2 = Dropout(self.RATE_DROP_DENSE)(Q2)

        # # Q2 = convolution_layer(embedded_sequences_2)
        # # # Q2 = Dropout(self.RATE_DROP_DENSE)(Q2)
        # # # Q2 = convolution_layer(Q2)
        # # Q2 = GlobalMaxPooling1D()(Q2)
        # # Q2 = Dropout(self.RATE_DROP_DENSE)(Q2)


        # merged = concatenate([Q1, Q2, custom_features])
        # merged = BatchNormalization()(merged)

        # merged = Dense(self.NUM_DENSE, kernel_initializer='normal')(merged)
        # merged = PReLU()(merged)
        # merged = Dropout(self.RATE_DROP_DENSE)(merged)
        # merged = BatchNormalization()(merged)

        # merged = Dense(round(0.9 * self.NUM_DENSE), kernel_initializer='normal')(merged)
        # merged = PReLU()(merged)
        # merged = Dropout(self.RATE_DROP_DENSE)(merged)
        # merged = BatchNormalization()(merged)

        # merged = Dense(round(0.7 * self.NUM_DENSE), kernel_initializer='normal')(merged)
        # merged = PReLU()(merged)
        # merged = Dropout(self.RATE_DROP_DENSE)(merged)
        # merged = BatchNormalization()(merged)

        # merged = Dense(round(0.5 * self.NUM_DENSE), kernel_initializer='normal')(merged)
        # merged = PReLU()(merged)
        # merged = Dropout(self.RATE_DROP_DENSE)(merged)
        # merged = BatchNormalization()(merged)

        # preds = Dense(1, activation='sigmoid')(merged)

        # ########################################
        # ## construct the model
        # ########################################
        # model = Model(inputs=[sequence_1_input,
        #                       sequence_2_input,
        #                       custom_input], \
        #               outputs=preds)

        # adam = optimizers.Adam(clipnorm=1.)

        # model.compile(loss='binary_crossentropy',
        #       optimizer=adam,
        #       metrics=['acc'])
        # #model.summary()
        # print("The model {} is built.".format(self.STAMP))

        # ########################################
        # ## add class weight
        # ########################################
        # if self.REWEIGH:
        #     class_weight = {0: 1.309028344, 1: 0.472001959}
        # else:
        #     class_weight = None

        # early_stopping =EarlyStopping(monitor='val_loss',
        #                               patience=3)
        # bst_model_path = self.STAMP + '.h5'
        # model_checkpoint = ModelCheckpoint(bst_model_path,
        #                                    save_best_only=True,
        #                                    save_weights_only=True)

        # hist = model.fit([data_1_train,
        #                   data_2_train,
        #                   features_train],
        #                  labels_train,
        #                  validation_data=([data_1_val,
        #                                    data_2_val,
        #                                    features_val],
        #                                   labels_val, weight_val),
        #                  epochs=10,
        #                  batch_size=512,
        #                  shuffle=True,
        #                  class_weight=class_weight,
        #                  callbacks=[early_stopping,
        #                             model_checkpoint])

        # model.load_weights(bst_model_path)
        # bst_val_score = min(hist.history['val_loss'])
        return (bst, bst_val_score)
# * Prediction
    def predict(self):
        bst, bst_val_score = self._model_constructor()

        self.model = bst
        self.bst_val_score = bst_val_score
        print('Building Test Features')
        
        X_test = self.test_custom_features
        d_test = xgb.DMatrix(X_test)
        print('Start making the submission before fine-tuning...')                
        p_test = bst.predict(d_test)

        df_test = pd.read_csv(TEST_DATA_FILE, encoding="utf-8")
        sub = pd.DataFrame()
        sub['test_id'] = df_test['test_id']
        sub['is_duplicate'] = p_test
        sub.to_csv(self.STAMP+'.csv', index=False)
        print("Plotting feature importance of the model...")
        import seaborn as sns
        sns.set(font_scale = 1.5)
        ax = xgb.plot_importance(bst)
        fig = ax.get_figure()
        fig.set_size_inches(17, 8)
        timestr = time.strftime("%Y-%m-%d-%H%M-")
        fig.savefig(timestr+self.STAMP+"-feature_importance.png")        

        #_, _, _, test_data_1, test_data_2, test_ids = self._preprocess_data()
        ########################################
        ## make the submission
        ########################################


        # preds = model.predict([test_data_1,
        #                        test_data_2,
        #                        self.test_custom_features],
        #                       batch_size=8192,
        #                       verbose=1)
        # preds += model.predict([test_data_2,
        #                         test_data_1,
        #                         self.test_custom_features],
        #                        batch_size=8192,
        #                        verbose=1)
        # preds /= 2

        # acc = pd.DataFrame({'epoch': [ i + 1 for i in self.hist.epoch ],
        #             'training': self.hist.history['acc'],
        #             'validation': self.hist.history['val_acc']})
        # ax = acc.ix[:,:].plot(x='epoch', figsize={5,8}, grid=True)
        # ax.set_ylabel("accuracy")
        # ax.set_ylim([0.0,1.0])
        # print("Plotting accuracy vs epoch dynamics...")
        # fig = ax.get_figure()
        # timestr = time.strftime("%Y-%m-%d-%H%M-")
        # fig.savefig(timestr+self.STAMP+"-accuracy_dynamics.png")
        # print("Saving the submission...")
        # submission = pd.DataFrame({'test_id':test_ids, 'is_duplicate': preds.ravel()})
        # submission.to_csv('%.4f_'%(bst_val_score)+self.STAMP+'.csv', index=False)
        print("Done.")
