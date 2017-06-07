# * Libraries
# ** Utilities
from itertools import chain
from random import sample
import time
# ** Core Processing
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O
# ** Plotting
import matplotlib.pyplot as plt # visualisation
# ** Strings
import re # regex
import string
# ** NLP
# *** SpaCy
import spacy # nlp
import en_core_web_sm as en # en library
nlp = en.load()
# *** NLTK
from nltk.corpus import wordnet
# * Constructor
class Preprocessor:
    def __init__(self, filename):
# ** storing data
        self.filename = filename
        self.training_data = pd.read_csv(self.filename, encoding = 'utf8')
# ** SpaCy processing variables
        self.SPECIAL_TOKENS = {
            'quoted': 'quoted_item',
            'non-ascii': 'non_ascii_word',
            'undefined': 'something'
        }
        self.ENTITY_ENUM = {
            '': '',
            'PERSON': 'person',
            'NORP': 'nationality',
            'FAC': 'facility',
            'ORG': 'organization',
            'GPE': 'country',
            'LOC': 'location',
            'PRODUCT': 'product',
            'EVENT': 'event',
            'WORK_OF_ART': 'artwork',
            'LANGUAGE': 'language',
            'DATE': 'date',
            'TIME': 'time',
            'LAW': 'law',
            
            # We replace the following with just 'number', since 
            # these entities are often misclassified and do not bring
            # significant semantic content to the data set.
            
            'PERCENT': 'number',
            'MONEY': 'number',
            'QUANTITY': 'number',
            'ORDINAL': 'number',
            'CARDINAL': 'number',
            
        }
        
        self.NUMERIC_TYPES = set([
            'DATE',
            'TIME',
            'PERCENT',
            'MONEY',
            'QUANTITY',
            'ORDINAL',
            'CARDINAL',
        ])

                
        
        self.QUESTION_DICT = self._build_question_dict()        
                                
        # create a dictionary with spaCy objects
        self.NLP_DICT = self._build_nlp_dict()        
        ## Determine the types of each word in the data set.
        # Define the cut-off level for the frequency of the secondary type.
        self.secondary_threshold = 2
        self.FREQ_DICT = self._build_freq_dict(self.NLP_DICT) # holds the frequencies of determined types
        self.ENTITIES = {} # holds the primary entities
        self.SECONDARY_ENTITIES = {} # holds the secondary entities               
                
        # Pick the primary and secondary entities.
    
        for key in set(self.FREQ_DICT):                
            # non-type gets a lower priority
            if '' in set(self.FREQ_DICT[key]):
                self.FREQ_DICT[key][''] = self.FREQ_DICT[key][''] - 0.5
                
            entities_list = list(self.FREQ_DICT[key].keys())
            correspondence_map = [
                entities_list,
                [self.FREQ_DICT[key][ent] for ent in entities_list]
            ]
    
            sorted_idx = np.argsort(correspondence_map[1])
                
            if sorted_idx.shape[0]>1:
                best_idx = sorted_idx[-1]
                second_idx = sorted_idx[-2]
                self.ENTITIES[key] = correspondence_map[0][best_idx]
                if correspondence_map[1][second_idx] > self.secondary_threshold:
                    self.SECONDARY_ENTITIES[key] = correspondence_map[0][second_idx]
            else:
                best_idx = sorted_idx[-1]
                self.ENTITIES[key] = correspondence_map[0][best_idx]

        self.short_length = 9
                
    def _build_question_dict(self):
        QUESTION_DICT = {}
        for i,series in self.training_data.iterrows():
            if series['qid1'] not in QUESTION_DICT:
                if series['question1'] is str:
                    QUESTION_DICT[series['qid1']] = series['question1']
                else:
                    QUESTION_DICT[series['qid1']] = ''
            if series['qid2'] not in QUESTION_DICT:
                if series['question2'] is str:
                    QUESTION_DICT[series['qid2']] = series['question2']
                else:
                    QUESTION_DICT[series['qid2']] = ''                                    
        return QUESTION_DICT
                
    def _build_nlp_dict(self):
        NLP_DICT = {}
        for i,k in enumerate(self.QUESTION_DICT):
            if i % 100000 == 0:
                print('   Processed {} out of {} questions with spaCy...'.format(i, len(self.QUESTION_DICT)))        
            NLP_DICT[k] = nlp(self.QUESTION_DICT[k])
        return NLP_DICT
    def _build_freq_dict(self, NLP_DICT):
        FREQ_DICT = {}
        for qid in set(NLP_DICT):
            
            if qid % 100000 == 0:
                print('   Processed frequencies of {} out of {} spaCy objects...'.format(qid, len(NLP_DICT)))        
                
            for token in set(NLP_DICT[qid]):
                
                if token.lower_ not in set(FREQ_DICT):
                    FREQ_DICT[token.lower_] = {}

                if token.ent_type_ not in set(FREQ_DICT[token.lower_]):
                    FREQ_DICT[token.lower_][token.ent_type_] = 0
                    
                FREQ_DICT[token.lower_][token.ent_type_] += 1
        return FREQ_DICT
        

# * Utilities
    def save_to_file(self, postfix):        
        timestr = time.strftime("%Y-%m-%d-%H%M-")
        saved_filename = timestr + postfix + ".csv"
        self.training_data.to_csv("data/"+saved_filename, sep=',', encoding='utf-8', index=False)
# * Primary Cleanup

    def _primary_cleanup(self, string):
        """Cleans up the text.
        
            Inspired by
            https://www.kaggle.com/hubert0527/spacy-name-entity-recognition/notebook
            and
            https://www.kaggle.com/currie32/the-importance-of-cleaning-text

        """
        def pad(text):
            return ' ' + text + ' '
        
        # Deal with empty questions
        
        if type(string) != str or string=='':
            return ''
        
        # We need to deal with capitalisation to train the model better.
        # There are several cases when we can predict there would be capital
        # letters not belonging to proper nouns.
        
        # First, the first letter of a question is likely to be
        # capitalised.
        
        string = string[0].lower() + string[1:]
        
        # Next, we are going to decapitalise the words after ., ?, !, ', ".
        
        def decap_first_character(regex):
            "Lower first character after the regex match."
            match = regex.group(0)
            # save the previous character in the match
            # and lower the next
            return match[:-1] + match[-1].lower()
        
        string = re.sub("(?<=[\.\?\)\!\'\"])[\s]*.", decap_first_character , string)
        
        
        # We make an assumption that the number of proper nouns at the
        # beginning of a sentence is small.
        
        # Inspired by hubert0527 (see the link above), we fix common oddities.
        
        # Replace weird chars in text
        
        string = re.sub("’", "'", string)
        string = re.sub("`", "'", string)
        string = re.sub("“", '"', string)
        string = re.sub("？", "?", string)
        string = re.sub("…", ".", string)
        string = re.sub("é", "e", string)
        
        string = re.sub("\'s", " ", string)
        string = re.sub("\bwhats\b", " what is ", string, flags=re.IGNORECASE)
        string = re.sub("\'ve", " have ", string, flags=re.IGNORECASE)
        string = re.sub("can't", "can not", string, flags=re.IGNORECASE)
        string = re.sub("n't", " not ", string, flags=re.IGNORECASE)
        string = re.sub("i'm", "i am", string, flags=re.IGNORECASE)
        string = re.sub("\'re", " are ", string, flags=re.IGNORECASE)
        string = re.sub("\'d", " would ", string, flags=re.IGNORECASE)
        string = re.sub("\'ll", " will ", string, flags=re.IGNORECASE)
        string = re.sub("e\.g\.", " eg ", string, flags=re.IGNORECASE)
        string = re.sub("b\.g\.", " bg ", string, flags=re.IGNORECASE)
        string = re.sub("i\.e\.", " ie ", string, flags=re.IGNORECASE)
        
        # replace shortcuts for thousands
        string = re.sub(r"(\W|^)([0-9]+)[kK](\W|$)", r"\1\g<2>000\3", string) # better regex provided by @armamut
        
        string = re.sub(" e-mail ", " email ", string, flags=re.IGNORECASE)
        string = re.sub(r" india ", " India ", string)
        string = re.sub(r" switzerland ", " Switzerland ", string)
        string = re.sub(r" china ", " China ", string)
        string = re.sub(r" chinese ", " Chinese ", string) 
        string = re.sub(r" imrovement ", " improvement ", string, flags=re.IGNORECASE)
        string = re.sub(r" intially ", " initially ", string, flags=re.IGNORECASE)
        string = re.sub(r" quora ", " Quora ", string, flags=re.IGNORECASE)
        string = re.sub(r" dms ", " direct messages ", string, flags=re.IGNORECASE)  
        string = re.sub(r" demonitization ", " demonetization ", string, flags=re.IGNORECASE) 
        string = re.sub(r" actived ", " active ", string, flags=re.IGNORECASE)
        string = re.sub(r" kms ", " kilometers ", string, flags=re.IGNORECASE)
        string = re.sub(r" cs ", " computer science ", string, flags=re.IGNORECASE) 
        string = re.sub(r" upvote", " up vote", string, flags=re.IGNORECASE)
        string = re.sub(r" iPhone ", " phone ", string, flags=re.IGNORECASE)
        string = re.sub(r" \0rs ", " rs ", string, flags=re.IGNORECASE)
        string = re.sub(r" calender ", " calendar ", string, flags=re.IGNORECASE)
        string = re.sub(r" ios ", " operating system ", string, flags=re.IGNORECASE)
        string = re.sub(r" gps ", " GPS ", string, flags=re.IGNORECASE)
        string = re.sub(r" gst ", " GST ", string, flags=re.IGNORECASE)
        string = re.sub(r" programing ", " programming ", string, flags=re.IGNORECASE)
        string = re.sub(r" bestfriend ", " best friend ", string, flags=re.IGNORECASE)
        string = re.sub(r" dna ", " DNA ", string, flags=re.IGNORECASE)
        string = re.sub(r" III ", " 3 ", string)
        string = re.sub(r" banglore ", " Bangalore ", string, flags=re.IGNORECASE)
        string = re.sub(r" J K ", " JK ", string, flags=re.IGNORECASE)
        string = re.sub(r" J\.K\. ", " JK ", string, flags=re.IGNORECASE)
        string = re.sub(r" quikly ", " quickly ", string)
        string = re.sub(r" unseccessful ", " unsuccessful ", string)
        string = re.sub(r" demoniti[\S]+ ", " demonetization ", string, flags=re.IGNORECASE)
        string = re.sub(r" demoneti[\S]+ ", " demonetization ", string, flags=re.IGNORECASE)  
        string = re.sub(r" addmision ", " admission ", string)
        string = re.sub(r" childern", " children ", string)
        string = re.sub(r" insititute ", " institute ", string)
        string = re.sub(r" connectionn ", " connection ", string)
        string = re.sub(r" permantley ", " permanently ", string)
        string = re.sub(r" sylabus ", " syllabus ", string)
        string = re.sub(r" sequrity ", " security ", string)
        string = re.sub(r" undergraduation ", " undergraduate ", string) # not typo, but GloVe can't find it
        string = re.sub(r"(?=[a-zA-Z])ig ", "ing ", string)
        string = re.sub(r" latop", " laptop", string)
        string = re.sub(r" programmning ", " programming ", string)  
        string = re.sub(r" begineer ", " beginner ", string)  
        string = re.sub(r" qoura ", " Quora ", string)
        string = re.sub(r" wtiter ", " writer ", string)  
        string = re.sub(r" litrate ", " literate ", string)  
        
        
        string = re.sub("\(s\)", " ", string, flags=re.IGNORECASE)
        string = re.sub("[c-fC-F]\:\/", " disk ", string, flags=re.IGNORECASE)
        
        # Next, we need to deal with numbers.
        
        # First, we remove comma between numbers, i.e. 15,000 -> 15000
        string = re.sub('(?<=[0-9])\,(?=[0-9])', "", string)
        
        # Secondly, float numbers are fixed -- by replacing them with an
        # arbitrary number.
        arbitrary_integer = 42
        string = re.sub('[0-9]+\.[0-9]+', " " + str(arbitrary_integer) + " ", string)
        
        # Handle punctuations and special chars
        
        string = re.sub('\$', " dollar ", string)
        string = re.sub('\%', " percent ", string)
        string = re.sub('\&', " and ", string)
        
        def pad_regex(regex):
            match = regex.group(0)
            return pad(match)
        
        string = re.sub('[\!\?\@\^\+\*\/\,\~\|\`\=\:\;\.\#\\\]', pad_regex, string)
        
        # Handling non-ascii characters
        string = re.sub('[^\x00-\x7F]+', pad(self.SPECIAL_TOKENS['non-ascii']), string)
        
        # Handling quoted and bracketed items
        def quoted_string_parser(pattern):
            text = pattern.group(0)
            parsed = nlp(string[1:-1])
            is_meaningful = False
            for token in parsed:
                # if one of the token is meaningful, we'll take the full string as meaningful
                if len(token.text)>2 and not token.text.isdigit() and token.has_vector:
                    is_meaningful = True
                elif token.text in self.SPECIAL_TOKENS.values():
                    is_meaningful = True
                    
            if is_meaningful:
                return text
            else:
                return pad(text[0]) + self.SPECIAL_TOKENS['quoted'] + pad(text[-1])
                        
        string = re.sub('\".*\"', quoted_string_parser, string)
        string = re.sub("\'.*\'", quoted_string_parser, string)
        string = re.sub("\(.*\)", quoted_string_parser, string)
        string = re.sub("\[.*\]", quoted_string_parser, string)
        string = re.sub("\{.*\}", quoted_string_parser, string)
        string = re.sub("\<.*\>", quoted_string_parser, string)
        
        string = re.sub('[\(\)\[\]\{\}\<\>\'\"]', pad_regex, string) 
        
        # clean stray s's
        string = re.sub(' s ', " ", string)
        
        # clean stray whitespaces
        
        string = re.sub('[\s]+', " ", string)
        string = string.strip()
        return string

    def rough_cleanup(self):
        print("1. Cleaning up the text...")
        self.training_data["question1"] = self.training_data["question1"].apply(self._primary_cleanup)
        self.training_data["question2"] = self.training_data["question2"].apply(self._primary_cleanup)
        print("   Finished the rudimentary clean-up.")
        print("   Saving to the file...")
        self.save_to_file("rudimentary_cleanup_train")
        
# * SpaCy Processing        

    # Define the functions for easy access to the corresponding entities

    def _get_type(self, token):
        
        if type(token) == str:
            token = nlp(token)[0]
            
        key = token.lower_
        
        try:
            return self.ENTITIES[key]
        
        except KeyError:
            return ''
    
    def _has_secondary_type(self, token):
        
        if type(token) == str:
            token = nlp(token)[0]
            
        key = token.lower_
        
        try:
            return key in self.SECONDARY_ENTITIES
        except KeyError:
            return False

    def _get_secondary_type(self, token):
        
        if type(token)==str:
            token = nlp(token)[0]
            
        key = token.lower_
        
        try:
            return self.SECONDARY_ENTITIES[key]
        except KeyError:
            return ''

    def _treat_with_spacy(self, spacy_obj, debug=False, show_failed=False, idx=None):
    
        def not_alpha_or_digit(token):
            text = token.text[0]
            return not (text.isalpha() or text.isdigit())
        
        WORD_LIST = []
        result = ''
        
        # a flag for the deduplication of replacements
        previous_ent_type = None
    
        failed_to_parse = False
        failed_words = []
        
        for token in spacy_obj:
            
            global_ent_type = self._get_type(token)
    
            # deal with special tokens
            if token.text in self.SPECIAL_TOKENS.values():
    
                if token.ent_type_ != '':
                    previous_ent_type = token.ent_type_
                    WORD_LIST.append(token.ent_type_)
                    
                else:
                    previous_ent_type = None
                    WORD_LIST.append(token.text)
                    
                
            # skip junk tokens
            elif not_alpha_or_digit(token) or token.text == ' ' or token.text == 's':
                previous_ent_type = None
                if debug: print(token.text, ' : there is a special string!')
                
            # deduplicate 
            elif global_ent_type == previous_ent_type or token.ent_type_ == previous_ent_type:
                if debug: print('Deduplicated non-numeric entities. ')
            elif global_ent_type in self.NUMERIC_TYPES and previous_ent_type in self.NUMERIC_TYPES:
                if debug: print('Deduplicated numeric entities')
            elif token.ent_type_ in self.NUMERIC_TYPES and previous_ent_type in self.NUMERIC_TYPES:
                if debug: print('Deduplicated numeric entities')
                
            
            # deal with numbers without ent_type_
            elif token.text.isdigit():
                
                if debug: print(token.text, ': replace with a numeric value.')
                
                if previous_ent_type in self.NUMERIC_TYPES:
                    pass
                else:
                    previous_ent_type = 'QUANTITY' 
                    WORD_LIST.append('number')
    
            elif global_ent_type != '':
                
                WORD_LIST.append(self.ENTITY_ENUM[global_ent_type])
                previous_ent_type = global_ent_type
                if debug: print(token.text, ' was replaced with ', self.ENTITY_ENUM[global_ent_type])
                
            # replace proper nouns with the corresponding entities.            
    
            # A proper noun has following special patterns:
            #     1. spaCy returns the lowercase lemma_ attribute
            #     2. if one of the characters but the first is uppercase, it is likely to be proper
    
            elif token.lower_== token.lemma_ and token.text[1:] != token.lemma_[1:] and self._has_secondary_type(token):
                second_type = self._get_secondary_type(token)
                WORD_LIST.append(self.ENTITY_ENUM[second_type])
                if debug: print(token.text, ' : used a secondary ent_type:', self.ENTITY_ENUM[second_type])
                previous_ent_type = second_type
            else:
                if token.lemma_=='-PRON-':
                    WORD_LIST.append(token.lower_)
                    result = token.lower
                    previous_ent_type = None
                    
                # the lemma can be identified
                elif nlp(token.lemma_)[0].has_vector:
                    WORD_LIST.append(token.lemma_)
                    result = token.lemma_
                    previous_ent_type = None
                    
                # the lemma cannot be identified
                elif self._has_secondary_type(token):
                    second_type = self._get_secondary_type(token)
                    WORD_LIST.append(self.ENTITY_ENUM[second_type])
                    result = self.ENTITY_ENUM[second_type]
                    previous_ent_type = second_type
                    if debug: print(token.text, ' : used the secondary ent_type for ', self.ENTITY_ENUM[second_type])
                    
                # neither glove nor spacy identifies the token
                elif nlp(token.lower_)[0].has_vector:
                    WORD_LIST.append(token.lower_)
                    result = token.lower_
                    previous_ent_type = None
                    if debug: print(token.text, ' : t', token.lower_)
                elif token.has_vector:
                    WORD_LIST.append(token.text)
                    result = token.text
                    previous_ent_type = None
                    if debug: print(token.text, ' : identified the alternative ent_type for ', token.text)
                    
    
                else:
                    failed_to_parse = True
                    failed_words.append(token.text)
                    previous_ent_type = None
                    
                    WORD_LIST.append(token.text)
                    if debug: print(token.text, ' : could not identify, left as is')
                    
        
        if show_failed and failed_to_parse:
            print('Failed words: ', failed_words)
            print('Before:', spacy_obj.text)
            print('After: ', ' '.join(WORD_LIST))
            print('====================================================================')
    
        return ' '.join(np.array(WORD_LIST))

    def apply_spacy_filter(self):
        print("2. Applying the spaCy filter...")
        self.training_data["question1"] = self.training_data["question1"].apply(nlp).apply(self._treat_with_spacy)
        self.training_data["question2"] = self.training_data["question2"].apply(nlp).apply(self._treat_with_spacy)
        print("   Finished.")
        print("   Saving to the file...")
        self.save_to_file("spacy_filtered_train")        
# * Short Questions Trimming
    def trim_shorties(self):
        print("3. Trimming short questions...")
    
        short_questions = self.training_data[(self.training_data['question1'].map(len) < self.short_length) | \
                                             (self.training_data['question2'].map(len) < self.short_length) ]
        duplicate_short_questions = short_questions[short_questions['is_duplicate'] == 1]
        print('   There are {} questions less than {} characters long, of which {} are duplicate questions.'\
              .format(len(short_questions), self.short_length, len(duplicate_short_questions)))
    
        self.training_data = self.training_data.drop(short_questions.index)
        print("   Dropped short questions.")
        print("   Saving to the file...")
        self.save_to_file("shorties_trimmed_train")                
# * Oversampling
    def _find_synonym(self, word):
        """Finds a synonym to a non-special word.
        If no synonym is found, returns the word."""
        if word not in set(self.ENTITY_ENUM.values()):
            synonyms = wordnet.synsets(word)
            lemmas = set(chain.from_iterable([token.lemma_names() for token in synonyms]))
            if not lemmas:
                return word
            else:
                return sample(lemmas, 1)[0].replace("_", " ")
        else:
            return word
    
    def _synonymised_sample(self):        
        synonyms_dict = {}
        
        # sample the right number of duplicate questions
        duplicates = self.training_data[self.training_data['is_duplicate'] == 1]
        sample_size = round(0.5*len(self.training_data) - len(duplicates))
        synonymised_duplicates = duplicates.sample(sample_size)
        
        # create a list of unique words in duplicate questions
        
        words = set(list(synonymised_duplicates['question1'].str.split(' ', expand=True).stack().unique()) \
                    + list(synonymised_duplicates['question2'].str.split(' ', expand=True).stack().unique()))
        
        for word in words:
            tokens = nlp(word)
            for token in tokens:
                if token.pos_ == 'NOUN' or token.pos_ == 'ADJ':
                    synonyms_dict[" "+word+" "] = " "+self._find_synonym(word)+" "
        
        # replace words in the duplicates
        synonymised_duplicates = synonymised_duplicates.replace(synonyms_dict, regex=True)
        return synonymised_duplicates
    
    def oversample_positives(self):
        print("4. Oversampling and synonymising duplicate questions...")
        self.training_data = pd.concat([self.training_data, self._synonymised_sample()])
        print("   Finished.")
        print("   The percent composition of duplicate questions",\
              "in the data set with automatically generated duplicates is now {}%."\
              .format(round(self.training_data['is_duplicate'].mean()*100, 2)))
        print("   Saving to the file...")
        self.save_to_file("oversampled_train")        
# * Final Processing
    def processed_questions(self):
        self.rough_cleanup()
        self.apply_spacy_filter()
        self.trim_shorties()
        self.oversample_positives()
        print("Saving to the file...")        
        self.save_to_file("clean_train")
        return self.training_data
