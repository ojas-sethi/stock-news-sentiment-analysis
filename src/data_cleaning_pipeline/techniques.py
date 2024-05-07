import string, os
import re
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from nltk.corpus import sentiwordnet as swn
from nltk.tokenize import word_tokenize
import spacy, json, nltk

def default(data: str):
    return data

def lower_case(data: str):
    return data.lower()

def remove_punctuation(data: str):
    return  data.translate(str.maketrans('', '', string.punctuation))

def remove_special_chars(data: str):
    # TODO: What do we mean by special characters ? Anything that isn't alphanumeric?
    return " ".join([x for x in data.split(" ") if x.isalnum()])

def remove_urls(data: str):
    # Match each word with a URL regex, and exclude matches
    url_regex= r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    return  " ".join([x for x in data.split(' ') if not bool(re.match(url_regex, x))])

'''
TODO: Stop list removal and expanding abbreviations need more exploring.
      What is going to be included in the dictionary?
'''

def remove_stop_words(data: str):
    try:
        stop_words = set(stopwords.words('english'))
    except:
        os.system("python -m nltk.downloader stopwords")
        stop_words = set(stopwords.words('english'))

    return " ".join([x for x in data.split(" ") if not x in stop_words])

def stemming(data: str):
    porter = PorterStemmer()
    return " ".join([porter.stem(x, to_lowercase=False) for x in data.split(" ")])

def remove_named_entities(data: str):
    try:
        nlp = spacy.load("en_core_web_md")
    except:
        os.system("python -m spacy download en_core_web_md")
        nlp = spacy.load("en_core_web_md")

    proc_string = nlp(data)
    #print(proc_string)
    if not proc_string.ents:
        return data
    res = data[:proc_string.ents[0].start_char]
    for i in range(1, len(proc_string.ents)):
        # The character before each entity should be a space ?
        res += data[proc_string.ents[i-1].end_char:proc_string.ents[i].start_char]
    res+=data[proc_string.ents[len(proc_string.ents)-1].end_char:]

    return res

def lemmatize(data):
    tokens = word_tokenize(data)
    lemmatizer = WordNetLemmatizer()
    lemmatizedTokens = [lemmatizer.lemmatize(token) for token in tokens]

    return ' '.join(lemmatizedTokens)

def nltk_to_wordnet_pos(nltk_pos):
    if nltk_pos.startswith('N'):
        return wordnet.NOUN
    elif nltk_pos.startswith('V'):
        return wordnet.VERB
    elif nltk_pos.startswith('R'):
        return wordnet.ADV
    elif nltk_pos.startswith('J'):
        return wordnet.ADJ
    else:
        return None
    
def find_sentiment_synonym(word, positive=1, part_of_speech='n'):
 
  best_synonym = None
  best_score = float('-inf') if positive else float('inf')  # Initialize scores

  synsets = list(swn.senti_synsets(word, pos=part_of_speech))
  
  if not synsets:
    return word

  # find the best synonym
  for synset in synsets:
    # Get the sentiment score
    score = synset.pos_score() - synset.neg_score()
    # Check if the word is positive
    if positive and score > best_score:
      best_score = score
      best_synonym = synset.synset.name().split('.')[0]
    # Check if the word is negative
    elif not positive and score < best_score:
      best_score = score
      best_synonym = synset.synset.name().split('.')[0]

  return best_synonym
    
def normalize_financial_terms(data: str):
    res = data
    with open('financial_terms.json', 'r') as f:
        financial_term_map = json.load(f)
        ''' 
            We need to use dictionary -> text map because tokenization can get
            rid of some terms which are a collection of words
        '''
        for term in financial_term_map.keys():
            res = res.replace(term, financial_term_map[term])

    # load fin_lexicon.json
    with open('fin_lexicon.json', 'r') as f:
        finTerms = json.load(f)    
    
    # iterate over each sentence
    for sentence in res.split('.'):
        # detect pos tags for each word
        pos_tags = nltk.pos_tag(nltk.word_tokenize(sentence))

        # iterate over each word
        for i in range(len(pos_tags)):
            word = pos_tags[i][0]
            # check if word is in financial terms
            if word in finTerms:
                # convert nltk pos to wordnet pos
                pos = nltk_to_wordnet_pos(pos_tags[i][1])
                # get sentiment of word
                sentiment = finTerms[word]
                # find synonym with max sentiment
                synonym = find_sentiment_synonym(word, sentiment, pos)
                res = res.replace(word, synonym)

    return re


technique_to_function_map = dict({\
        "lower_case":                       lower_case,
        "remove_punctuation":               remove_punctuation,
        "remove_special_chars":             remove_special_chars,
        "remove_urls":                      remove_urls,
        "stemming":                         stemming,
        "remove_named_entities":            remove_named_entities,
        "remove_stop_words":                remove_stop_words,
        "normalize_financial_terms":        normalize_financial_terms,
        "lemmatize":                        lemmatize,
        "default":                          default,
    })



class CleaningTechniqueFactory:
    def __init__(self) -> None:
        pass
    
    def get_all_functions(self):
        return list(technique_to_function_map.keys())

    def generate_cleaning_technique(self, function: str):
        return technique_to_function_map[function] \
               if function in technique_to_function_map else None