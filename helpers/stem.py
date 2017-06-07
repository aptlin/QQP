# * Libraries
import os
from os.path import exists
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer 
import pandas as pd # data processing, CSV file I/O

# * Variables
BASE_DIR = 'data/'
TRAIN_DATA_FILENAME = "2017-05-24-1818-shorties_trimmed_train"
TRAIN_DATA_FILE = BASE_DIR + TRAIN_DATA_FILENAME + '.csv'
OUTPUT_DATA_FILE = BASE_DIR + "stemmed_clean_train.csv"

STOPWORDS = set(stopwords.words('english'))

# * Constructor
class Stem:
    def __init__(self,
                 train_data_filename=TRAIN_DATA_FILE):
        self.TRAIN_DATA_FILE = train_data_filename
        self.TRAINING_DATA = pd.read_csv(self.TRAIN_DATA_FILE,
                                         encoding = 'utf8')
    def _stemmed(self, text):
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
        return text
    
    def _cleaned_stopwords(self, text):
        text = text.split()
        text = [word for word in text if not word in STOPWORDS]
        text = " ".join(text)
        return text
    
    def stem(self, text):
        text = self._cleaned_stopwords(text)
        text = self._stemmed(text)
        return text

    def stem_questions(self):
        if exists(OUTPUT_DATA_FILE):
            print("The train data set {} have already been stemmed.".format(TRAIN_DATA_FILENAME))
        else:
            self.TRAINING_DATA["question1"] \
                = self.TRAINING_DATA["question1"].apply(self.stem)
            self.TRAINING_DATA["question2"] \
                = self.TRAINING_DATA["question2"].apply(self.stem)
            self.TRAINING_DATA.to_csv(OUTPUT_DATA_FILE,
                                      sep=',',
                                      encoding='utf-8',
                                      index=False)

