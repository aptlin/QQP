import re
import spacy
import en_core_web_sm as en
nlp = en.load()

SPECIAL_TOKENS = {
    'quoted': 'quoted_item',
    'non-ascii': 'non_ascii_word',
    'undefined': 'something'
}

def cleanup(string):
    """ Cleans up the text.

    Inspired by https://www.kaggle.com/hubert0527/spacy-name-entity-recognition/notebook
    and https://www.kaggle.com/currie32/the-importance-of-cleaning-text
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

    string = re.sub("e-mail", " email ", string, flags=re.IGNORECASE)
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
    string = re.sub('[^\x00-\x7F]+', pad(SPECIAL_TOKENS['non-ascii']), string)

    # Handling quoted and bracketed items
    def quoted_string_parser(pattern):
        text = pattern.group(0)
        parsed = nlp(string[1:-1])
        is_meaningful = False
        for token in parsed:
            # if one of the token is meaningful, we'll take the full string as meaningful
            if len(token.text)>2 and not token.text.isdigit() and token.has_vector:
                is_meaningful = True
            elif token.text in SPECIAL_TOKENS.values():
                is_meaningful = True
                
        if is_meaningful:
            return text
        else:
            return pad(text[0]) + SPECIAL_TOKENS['quoted'] + pad(text[-1])

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

