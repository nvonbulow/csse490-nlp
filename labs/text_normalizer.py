import nltk
import spacy
import unicodedata
from contractions import CONTRACTION_MAP
import re
from nltk.corpus import wordnet
import collections
#from textblob import Word
from nltk.tokenize.toktok import ToktokTokenizer
from bs4 import BeautifulSoup
import numpy as np

tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words('english')
nlp = spacy.load('en_core_web_sm')
nlp.disable_pipes('parser', 'ner')
# nlp_vec = spacy.load('en_vectors_web_lg', parse=True, tag=True, entity=True)



def strip_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    if bool(soup.find()):
        [s.extract() for s in soup(['iframe', 'script'])]
        stripped_text = soup.get_text()
        stripped_text = re.sub(r'[\r|\n|\r\n]+', '\n', stripped_text)
    else:
        stripped_text = text
    return stripped_text

batch_strip_html_tags = np.vectorize(strip_html_tags)


#def correct_spellings_textblob(tokens):
#	return [Word(token).correct() for token in tokens]  


def simple_porter_stemming(text):
    ps = nltk.porter.PorterStemmer()
    text = ' '.join([ps.stem(word) for word in text.split()])
    return text

batch_simple_porter_stemming = np.vectorize(simple_porter_stemming)

def lemmatize_text(text):
    text = nlp(text)
    text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
    return text

def batch_lemmatize_text(corpus):
    corpus = nlp.pipe(corpus)
    corpus = [' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text]) for text in corpus]
    return corpus

repeat_pattern = re.compile(r'(\w*)(\w)\2(\w*)')
def remove_repeated_characters(tokens):
    match_substitution = r'\1\2\3'
    def replace(old_word):
        if wordnet.synsets(old_word):
            return old_word
        new_word = repeat_pattern.sub(match_substitution, old_word)
        return replace(new_word) if new_word != old_word else new_word
            
    correct_tokens = [replace(word) for word in tokens]
    return correct_tokens

batch_remove_repeated_characters = np.vectorize(remove_repeated_characters)

def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    return batch_expand_contractions([text], contraction_mapping)

def batch_expand_contractions(corpus, contraction_mapping=CONTRACTION_MAP):
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), 
                                      flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)\
                                if contraction_mapping.get(match)\
                                else contraction_mapping.get(match.lower())                       
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction
        
    expanded_corpus = [contractions_pattern.sub(expand_match, text) for text in corpus]
    expanded_corpus = [re.sub("'", "", expanded_text) for expanded_text in expanded_corpus]
    return expanded_corpus


def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text

batch_remove_accented_chars = np.vectorize(remove_accented_chars)

sp_pattern = re.compile(r'[^a-zA-Z0-9\s]|\[|\]')
sprd_pattern = re.compile(r'[^a-zA-Z0-9\s]|\[|\]')
def remove_special_characters(text, remove_digits=False):
    pattern = sp_pattern if not remove_digits else sprd_pattern
    text = re.sub(pattern, '', text)
    return text

batch_remove_special_characters = np.vectorize(remove_special_characters)

def remove_stopwords(text, is_lower_case=False, stopwords=stopword_list):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopwords]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopwords]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text

batch_remove_stopwords = np.vectorize(remove_stopwords)


def normalize_corpus(corpus, html_stripping=True, contraction_expansion=True,
                     accented_char_removal=True, text_lower_case=True, 
                     text_stemming=False, text_lemmatization=True, 
                     special_char_removal=True, remove_digits=True,
                     stopword_removal=True, stopwords=stopword_list):
    
    normalized_corpus = []
    # normalize each document in the corpus
    # strip HTML
    if html_stripping:
        corpus = batch_strip_html_tags(corpus)

    # remove extra newlines
    corpus = [doc.translate(doc.maketrans("\n\t\r", "   ")) for doc in corpus]

    # remove accented characters
    if accented_char_removal:
        corpus = batch_remove_accented_chars(corpus)

    # expand contractions    
    if contraction_expansion:
        corpus = batch_expand_contractions(corpus)

    # lemmatize text
    if text_lemmatization:
        corpus = batch_lemmatize_text(corpus)

    # stem text
    if text_stemming and not text_lemmatization:
        corpus = batch_simple_porter_stemming(corpus)

    # remove special characters and\or digits    
    if special_char_removal:
        # insert spaces between special characters to isolate them    
        special_char_pattern = re.compile(r'([{.(-)!}])')
        corpus = [special_char_pattern.sub(" \\1 ", doc) for doc in corpus]
        corpus = batch_remove_special_characters(corpus, remove_digits=remove_digits)  

    # remove extra whitespace
    corpus = [re.sub(' +', ' ', doc) for doc in corpus]

        # lowercase the text    
    if text_lower_case:
        corpus = [doc.lower() for doc in corpus]

    # remove stopwords
    if stopword_removal:
        corpus = batch_remove_stopwords(corpus, is_lower_case=text_lower_case, stopwords=stopwords)

    # remove extra whitespace
    corpus = [re.sub(' +', ' ', doc) for doc in corpus]
    doc = [doc.strip() for doc in corpus]
        
    return corpus