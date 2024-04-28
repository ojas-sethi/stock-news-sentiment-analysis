import string
import re
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import spacy

def lower_case(data: str):
    return data.lower()

def remove_punctuation(data: str):
        return  data.translate(str.maketrans('', '', string.punctuation))

def remove_special_chars(data: str):
        # TODO: What do we mean by special characters ? Anything that isn't alphanumeric?
        return  [x for x in data if x.isalnum()]

def remove_urls(data: str):
        # Match each word with a URL regex, and exclude matches
        url_regex= r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
        return  " ".join([x for x in data.split(' ') if not bool(re.match(url_regex, x))])

'''
TODO: Stop list removal and expanding abbreviations need more exploring.
      What is going to be included in the dictionary?
'''

def remove_stop_words(data: str):
    stop_words = set(stopwords.words('english'))
    return " ".join([x for x in data.split(" ") if not x in stop_words])

def stemming(data: str):
    porter = PorterStemmer()
    return " ".join([porter.stem(x, to_lowercase=False) for x in data.split(" ")])

def remove_named_entities(data: str):
    nlp = spacy.load("en_core_web_sm")
    proc_string = nlp(data)
    res = data[:proc_string.ents[0].start_char]
    for i in range(1, len(proc_string.ents)):
        # The character before each entity should be a space ?
        res += data[proc_string.ents[i-1].end_char:proc_string.ents[i].start_char]
    res+=data[proc_string.ents[len(proc_string.ents)-1].end_char:]

    return res

'''
TODO: Normalization
'''


class CleaningTechniqueFactory:
    def __init__(self) -> None:
          pass
    
    def generate_cleaning_technique(function: str):
        if function == "lower_case":
              return lower_case
        elif function == "remove_punctuation":
              return remove_punctuation
        elif function == "remove_special_chars":
              return remove_special_chars
        elif function == "remove_urls":
              return remove_urls
        elif function == "remove_stop_words":
              return remove_stop_words
        elif function == "stemming":
              return stemming
        elif function == "remove_named_entities":
              return remove_named_entities
        else:
              print("Cleaning Technique not impleneted.")
              return None