import os
import nltk
import json
import pickle
import random

from cord_globals import *
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

LEMMATIZER = WordNetLemmatizer()
STOPWORDS = set(stopwords.words('english') + CUSTOM_STOPWORDS)

def pipeline(text):
    '''
    Processes raw text to remove all stopwords and lemmatize
    '''
    t = nltk.word_tokenize(text)
    t = [LEMMATIZER.lemmatize(w) for w in t]
    t = [w for w in t if w not in STOPWORDS]
    t = [w for w in t if len(w) > 1]
    return t


def run(documents, save=True):
    ''' 
    Given several documents, generates 1 dict of all words
    used in the whole corpus, and a dict for each document.
    The dicts map the word to its frequency
    '''
    progress = tqdm(total=NUM_DOCS, desc='JSON Files parsed:')
    corpus = {}

    for document in documents:
        with open(document, 'r') as f:
            schema = json.loads(f.read().lower())
            paragraphs = schema['body_text']
            paper_id = schema['paper_id']

            text = ''
            for p in paragraphs:
                text += p['text']

        text = pipeline(text)
        
        doc_dict = {}
        for word in text:
            # Assume it's already accounted for in corpus
            if word in doc_dict:
                doc_dict[word] += 1
                corpus[word]['count'] += 1

            else:
                doc_dict[word] = 1
                
                # Make sure to add this paper to the corpus to make building
                # the graph eaiser later on
                if word in corpus:
                    corpus[word]['count'] += 1
                    corpus[word]['papers'].add(paper_id)
                else:
                    corpus[word] = {'count': 1, 'papers': {paper_id}}

        if save:
            pickle.dump(
                doc_dict, 
                open(DICTS+F_TO_DICT(document), 'wb+'), 
                protocol=pickle.HIGHEST_PROTOCOL
            )
        
        progress.update()

    if save:
        pickle.dump(
            corpus, 
            open(CORPUS_F, 'wb+'),
            protocol=pickle.HIGHEST_PROTOCOL
        )
    
    return corpus

def runall():
    # I'm sure there's a smarter way to do this but who cares
    run(JSON_FILES)

def test(num_docs=10):
    test_docs = []
    for i in range(num_docs):
        test_docs.append(random.choice(JSON_FILES))

    return run(test_docs, save=False)
    
if __name__ == '__main__':
    runall()