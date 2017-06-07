# Code inspired by lystdo and bradleypallen
# See https://www.kaggle.com/lystdo/lstm-with-word2vec-embeddings
# and https://github.com/bradleypallen/keras-quora-question-pairs
# Thank you, lystdo and bradleypallen.

# * Libraries
########################################
## import packages
########################################
import os
from os.path import exists
import time
import re
import csv
import codecs
import numpy as np
import pandas as pd
import pickle

from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Merge, Dense, Input
from keras.layers import Embedding, Dropout, Activation
from keras.layers import TimeDistributed, Lambda
from keras.layers import Convolution1D, GlobalMaxPooling1D
from keras.layers.advanced_activations import PReLU
from keras.layers.merge import concatenate
from keras.models import Model, Sequential
from keras.constraints import maxnorm
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras import optimizers

# * Variables

# set directories and parameters

BASE_DIR = 'data/'
EMBEDDING_FILE = BASE_DIR + 'GoogleNews-vectors-negative300.bin'
TRAIN_DATA_FILENAME = "2017-05-24-1818-shorties_trimmed_train"
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
PREPROCESSED_TRAIN_DATA_Q1 = BASE_DIR + PREPROCESSED + 'train_data_q1.npy'
PREPROCESSED_TRAIN_DATA_Q2 = BASE_DIR + PREPROCESSED + 'train_data_q2.npy'
PREPROCESSED_TEST_DATA_Q1 = BASE_DIR + PREPROCESSED + 'test_data_q1.npy'
PREPROCESSED_TEST_DATA_Q2 = BASE_DIR + PREPROCESSED + 'test_data_q2.npy'
PREPROCESSED_LABELS = BASE_DIR + PREPROCESSED + 'labels.npy'
PREPROCESSED_TEST_IDS = BASE_DIR + PREPROCESSED + 'test_ids.npy'
PREPROCESSED_WORD_INDEX = BASE_DIR + PREPROCESSED + 'word_index.pkl'

NUM_DENSE = np.random.randint(190, 275)
RATE_DROP_DENSE = (1 + np.random.rand()) * 0.25
RECTIFIER = 'relu'
REWEIGH = True
NB_FILTER = 64
FILTER_LENGTH = 5

ABHISHEK_TRAIN = 'abhishek/train_features.csv'
ABHISHEK_TEST = 'abhishek/test_features.csv'

MAGIC_II_TRAIN = 'magic2/train_ic.csv'
MAGIC_II_TEST = 'magic2/test_ic.csv'

TRAIN_CUSTOM_FEATURES = BASE_DIR + 'train_custom_features.csv'
TEST_CUSTOM_FEATURES = BASE_DIR + 'test_custom_features.csv'

# * Constructor


class Gimmel:
    def __init__(self,
                 train_data_filename=TRAIN_DATA_FILE,
                 test_data_filename=TEST_DATA_FILE,
                 embeddings_filename=EMBEDDING_FILE,
                 max_seq_length=MAX_SEQUENCE_LENGTH,
                 max_nb_words=MAX_NB_WORDS,
                 embeddings_dim=EMBEDDING_DIM,
                 validation_split=VALIDATION_SPLIT,
                 num_dense=NUM_DENSE,
                 rate_drop_dense=RATE_DROP_DENSE,
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
        with codecs.open(self.TRAIN_DATA_FILE, encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=',')
            header = next(reader)
            for values in reader:
                self.texts_1.append(values[3])
                self.texts_2.append(values[4])
                self.labels.append(int(values[5]))
        print("Saved {} question pairs.".format(len(self.texts_1)))
        ## testing
        self.test_texts_1 = []
        self.test_texts_2 = []
        self.test_ids = []
        with codecs.open(self.TEST_DATA_FILE, encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=',')
            header = next(reader)
            for values in reader:
                self.test_texts_1.append(values[1])
                self.test_texts_2.append(values[2])
                self.test_ids.append(values[0])
        ## word2vec indexing
        self.word_index = {}



        self.MAX_SEQUENCE_LENGTH = max_seq_length
        self.MAX_NB_WORDS = max_nb_words
        self.EMBEDDING_DIM = embeddings_dim
        self.VALIDATION_SPLIT = validation_split

        self.NUM_DENSE = num_dense
        self.RATE_DROP_DENSE = rate_drop_dense
        self.RECTIFIER = rectifier
        self.REWEIGH = reweigh
        self.STAMP = 'gimmel_%d_%.2f'%(num_dense,
                                   rate_drop_dense)

        # containers for the computed model
        self.model = {}
        self.hist = {}
        self.bst_val_score = {}

        # add custom-engineered features

        self.train_custom_features, self.test_custom_features = self._engineered_features()

# * Preprocessing

    def _preprocess_data(self):
        # process texts in datasets

        print('Processing text dataset...')

        if exists(PREPROCESSED_TRAIN_DATA_Q1) and \
           exists(PREPROCESSED_TRAIN_DATA_Q2) and \
           exists(PREPROCESSED_TEST_DATA_Q1) and \
           exists(PREPROCESSED_TEST_DATA_Q2) and \
           exists(PREPROCESSED_LABELS) and \
           exists(PREPROCESSED_TEST_IDS) and \
           exists(PREPROCESSED_WORD_INDEX):
            data_1 = np.load(open(PREPROCESSED_TRAIN_DATA_Q1, 'rb'))
            data_2 = np.load(open(PREPROCESSED_TRAIN_DATA_Q2, 'rb'))
            labels = np.load(open(PREPROCESSED_LABELS, 'rb'))
            test_data_1 = np.load(open(PREPROCESSED_TEST_DATA_Q1, 'rb'))
            test_data_2 = np.load(open(PREPROCESSED_TEST_DATA_Q2, 'rb'))
            test_ids = np.load(open(PREPROCESSED_TEST_IDS, 'rb'))
            with open(PREPROCESSED_WORD_INDEX, 'rb') as f:
                self.word_index = pickle.load(f)            
            
        else:
            tokenizer = Tokenizer(num_words=self.MAX_NB_WORDS)
            tokenizer.fit_on_texts(self.texts_1 +
                                   self.texts_2 +
                                   self.test_texts_1 +
                                   self.test_texts_2)
            sequences_1 = tokenizer.texts_to_sequences(self.texts_1)
            sequences_2 = tokenizer.texts_to_sequences(self.texts_2)
            test_sequences_1 = tokenizer.texts_to_sequences(self.test_texts_1)
            test_sequences_2 = tokenizer.texts_to_sequences(self.test_texts_2)
            self.word_index = tokenizer.word_index            
            print('Found %s unique tokens' % len(self.word_index))
            data_1 = pad_sequences(sequences_1,
                                   maxlen=self.MAX_SEQUENCE_LENGTH)
            data_2 = pad_sequences(sequences_2,
                                   maxlen=self.MAX_SEQUENCE_LENGTH)
            labels = np.array(self.labels)
            print('Shape of data tensor:', data_1.shape)
            print('Shape of label tensor:', labels.shape)
            test_data_1 = pad_sequences(test_sequences_1,
                                        maxlen=self.MAX_SEQUENCE_LENGTH)
            test_data_2 = pad_sequences(test_sequences_2,
                                        maxlen=self.MAX_SEQUENCE_LENGTH)
            test_ids = np.array(self.test_ids)
            
            np.save(open(PREPROCESSED_TRAIN_DATA_Q1, 'wb'), data_1)
            np.save(open(PREPROCESSED_TRAIN_DATA_Q2, 'wb'), data_2)
            np.save(open(PREPROCESSED_LABELS, 'wb'), labels)
            np.save(open(PREPROCESSED_TEST_DATA_Q1, 'wb'), test_data_1)
            np.save(open(PREPROCESSED_TEST_DATA_Q2, 'wb'), test_data_2)
            np.save(open(PREPROCESSED_TEST_IDS, 'wb'), test_ids)
            with open(PREPROCESSED_WORD_INDEX, 'wb') as f:
                pickle.dump(self.word_index, f, pickle.HIGHEST_PROTOCOL)
            print("Saved the preprocessed data.")
            
        return (data_1,
                data_2,
                labels,
                test_data_1,
                test_data_2,
                test_ids)

# * Embedding Layer

    def _create_embedding_layer(self):

        # prepare embeddings

        print('Preparing embedding matrix')
        nb_words = min(self.MAX_NB_WORDS, len(self.word_index)) + 1

        # index word vectors

        if exists(WORD_EMBEDDING_MATRIX_FILE):
            print("Loading a cached embedding matrix.")
            embedding_matrix = np.load(open(WORD_EMBEDDING_MATRIX_FILE, 'rb'))
        else:
            print('Indexing word vectors')
            word2vec = KeyedVectors.load_word2vec_format(self.EMBEDDING_FILE,
                                                         binary=True)
            print('Found %s word vectors of word2vec' % len(word2vec.vocab))
            embedding_matrix = np.zeros((nb_words, self.EMBEDDING_DIM))
            for word, i in self.word_index.items():
                if word in word2vec.vocab:
                    embedding_matrix[i] = word2vec.word_vec(word)
            print('Null word embeddings: %d' %
                  np.sum(np.sum(embedding_matrix, axis=1) == 0))
            print('Saving the word embeddings.')
            np.save(open(WORD_EMBEDDING_MATRIX_FILE, 'wb'), embedding_matrix)


        embedding_layer = Embedding(nb_words,
                                    self.EMBEDDING_DIM,
                                    weights=[embedding_matrix],
                                    input_length=self.MAX_SEQUENCE_LENGTH,
                                    trainable=False)
        return embedding_layer
# * Engineered Features

    def _engineered_features(self):
        # @abhishek's features
        # Thanks to @raddar and @abhishek for the data.
        # See https://www.kaggle.com/c/quora-question-pairs/discussion/31284

        # abhishek_train = pd.read_csv(ABHISHEK_TRAIN, encoding = "ISO-8859-1")
        # abhishek_test = pd.read_csv(ABHISHEK_TEST, encoding = "ISO-8859-1")
        # abhishek_train_features = abhishek_train.ix[:, 2:30]
        # abhishek_test_features = abhishek_test.ix[:, 2:30]

        # Krzysztof Dziedzic's magic feature II.
        # Data by @Justfor.
        # See https://www.kaggle.com/justfor/edges/code
        # and https://www.kaggle.com/c/quora-question-pairs/discussion/33287

        magic2_train_features =  pd.read_csv(MAGIC_II_TRAIN, encoding = "utf-8")
        magic2_test_features =  pd.read_csv(MAGIC_II_TEST, encoding = "utf-8")

        # @tarobxl kcore feature
        # See https://www.kaggle.com/c/quora-question-pairs/discussion/33371
        from kcore_decomposition import KCore_Decomposition
        kd = KCore_Decomposition()
        kcore_train_features, kcore_test_features = kd.attach_max_kcore()

        train_features = pd.concat([#abhishek_train_features,
                                    magic2_train_features,
                                    kcore_train_features], axis=1, join='inner').fillna(0)
        test_features = pd.concat([#abhishek_test_features,
                                    magic2_test_features,
                                    kcore_test_features], axis=1, join='inner').fillna(0)

        return (train_features, test_features)

# * Model Constructor
    def _model_constructor(self):
        ########################################
        ## sample train/validation data
        ########################################
        data_1, data_2, labels, _, _, _ = self._preprocess_data()

        perm = np.random.permutation(len(data_1))

        idx_train = perm[:int(len(data_1)*(1-self.VALIDATION_SPLIT))]
        idx_val = perm[int(len(data_1)*(1-self.VALIDATION_SPLIT)):]

        # data_1_train = np.vstack((data_1[idx_train], data_2[idx_train]))
        # data_2_train = np.vstack((data_2[idx_train], data_1[idx_train]))
        data_1_train = np.vstack((data_1[idx_train], data_2[idx_train]))
        data_2_train = np.vstack((data_2[idx_train], data_1[idx_train]))

        self.train_custom_features = self.train_custom_features.as_matrix()

        features_train = np.vstack((self.train_custom_features[idx_train],
                                    self.train_custom_features[idx_train]))

        labels_train = np.concatenate((labels[idx_train], labels[idx_train]))

        data_1_val = np.vstack((data_1[idx_val], data_2[idx_val]))
        data_2_val = np.vstack((data_2[idx_val], data_1[idx_val]))

        features_val = np.vstack((self.train_custom_features[idx_val],
                                  self.train_custom_features[idx_val]))

        labels_val = np.concatenate((labels[idx_val], labels[idx_val]))

        weight_val = np.ones(len(labels_val))
        if self.REWEIGH:
            weight_val *= 0.472001959
            weight_val[labels_val==0] = 1.309028344

        ########################################
        ## define the model structure
        ########################################
 
        embedding_layer = self._create_embedding_layer()
        timedist_layer = TimeDistributed(Dense(self.EMBEDDING_DIM,
                                               activation=self.RECTIFIER))
        lambda_layer = Lambda(lambda x: K.max(x, axis=1),
                              output_shape=(self.EMBEDDING_DIM, ))
        
        dense_dim = int(self.train_custom_features.shape[1])
        custom_input = Input(shape=(dense_dim, ), dtype='float32')                        
        custom_features = Dropout(self.RATE_DROP_DENSE)(custom_input)        
        custom_features = Dense(self.NUM_DENSE,
                                kernel_initializer='normal',
                                activation=self.RECTIFIER)(custom_features)
        custom_features = Dropout(self.RATE_DROP_DENSE)(custom_features)
        custom_features = BatchNormalization()(custom_features)

                
        sequence_1_input = Input(shape=(self.MAX_SEQUENCE_LENGTH,),
                                 dtype='int32')
        embedded_sequences_1 = embedding_layer(sequence_1_input)
        timedisted_sequences_1 = timedist_layer(embedded_sequences_1)
        Q1 = lambda_layer(timedisted_sequences_1)

        sequence_2_input = Input(shape=(self.MAX_SEQUENCE_LENGTH,),
                                 dtype='int32')
        embedded_sequences_2 = embedding_layer(sequence_2_input)
        timedisted_sequences_2 = timedist_layer(embedded_sequences_2)
        Q2 = lambda_layer(timedisted_sequences_2)

        merged = concatenate([Q1, Q2, custom_features])
        merged = BatchNormalization()(merged)

        merged = Dense(self.NUM_DENSE, kernel_initializer='normal')(merged)
        merged = PReLU()(merged)
        merged = Dropout(self.RATE_DROP_DENSE)(merged)
        merged = BatchNormalization()(merged)

        merged = Dense(round(0.9 * self.NUM_DENSE), kernel_initializer='normal')(merged)
        merged = PReLU()(merged)
        merged = Dropout(self.RATE_DROP_DENSE)(merged)
        merged = BatchNormalization()(merged)

        merged = Dense(round(0.7 * self.NUM_DENSE), kernel_initializer='normal')(merged)
        merged = PReLU()(merged)
        merged = Dropout(self.RATE_DROP_DENSE)(merged)
        merged = BatchNormalization()(merged)

        merged = Dense(round(0.5 * self.NUM_DENSE), kernel_initializer='normal')(merged)
        merged = PReLU()(merged)
        merged = Dropout(self.RATE_DROP_DENSE)(merged)
        merged = BatchNormalization()(merged)

        preds = Dense(1, activation='sigmoid')(merged)

        ########################################
        ## construct the model
        ########################################
        model = Model(inputs=[sequence_1_input,
                              sequence_2_input,
                              custom_input], \
                      outputs=preds)

        adam = optimizers.Adam(clipnorm=1.)

        model.compile(loss='binary_crossentropy',
              optimizer=adam,
              metrics=['acc'])
        #model.summary()
        print("The model {} is built.".format(self.STAMP))

        ########################################
        ## add class weight
        ########################################
        if self.REWEIGH:
            class_weight = {0: 1.309028344, 1: 0.472001959}
        else:
            class_weight = None

        early_stopping =EarlyStopping(monitor='val_loss',
                                      patience=3)
        bst_model_path = self.STAMP + '.h5'
        model_checkpoint = ModelCheckpoint(bst_model_path,
                                           save_best_only=True,
                                           save_weights_only=True)

        hist = model.fit([data_1_train,                          
                          data_2_train,
                          features_train],                         
                         labels_train,
                         validation_data=([data_1_val,
                                           data_2_val,
                                           features_val],
                                          labels_val, weight_val),
                         epochs=10,
                         batch_size=512,
                         shuffle=True,
                         class_weight=class_weight,
                         callbacks=[early_stopping,
                                    model_checkpoint])

        model.load_weights(bst_model_path)
        bst_val_score = min(hist.history['val_loss'])
        return (model, hist, bst_val_score)
# * Prediction
    def predict(self):
        model, hist, bst_val_score = self._model_constructor()

        self.model = model
        self.hist = hist
        self.bst_val_score = bst_val_score

        _, _, _, test_data_1, test_data_2, test_ids = self._preprocess_data()
        ########################################
        ## make the submission
        ########################################
        print('Start making the submission before fine-tuning')

        self.test_custom_features = self.test_custom_features.as_matrix()

        preds = model.predict([test_data_1,
                               test_data_2,
                               self.test_custom_features],
                              batch_size=8192,
                              verbose=1)
        preds += model.predict([test_data_2,
                                test_data_1,
                                self.test_custom_features],
                               batch_size=8192,
                               verbose=1)
        preds /= 2

        acc = pd.DataFrame({'epoch': [ i + 1 for i in self.hist.epoch ],
                    'training': self.hist.history['acc'],
                    'validation': self.hist.history['val_acc']})
        ax = acc.ix[:,:].plot(x='epoch', figsize={5,8}, grid=True)
        ax.set_ylabel("accuracy")
        ax.set_ylim([0.0,1.0])
        print("Plotting accuracy vs epoch dynamics...")
        fig = ax.get_figure()
        timestr = time.strftime("%Y-%m-%d-%H%M-")
        fig.savefig(timestr+self.STAMP+"-accuracy_dynamics.png")
        print("Saving the submission...")
        submission = pd.DataFrame({'test_id':test_ids, 'is_duplicate': preds.ravel()})
        submission.to_csv('%.4f_'%(bst_val_score)+self.STAMP+'.csv', index=False)
        print("Done.")
