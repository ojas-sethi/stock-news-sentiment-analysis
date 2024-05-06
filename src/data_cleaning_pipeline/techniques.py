import string, os
import re
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy, json

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

'''
TODO: Normalization
'''
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

    return res


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