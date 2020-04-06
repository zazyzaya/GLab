import os
import nltk
import json
import pickle
import random
import pandas as pd

from math import inf
from cn_globals import *
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

LEMMATIZER = WordNetLemmatizer()
STOPWORDS = set(stopwords.words('english'))

def get_documents(stop_after=inf):
    with open(PAPERS, 'r') as f:
        paper = f.readline()
        
        i =0
        while(paper and i < stop_after):
            yield json.loads(paper)
            paper = f.readline()
            i += 1

def run(documents, save=True):
    ''' 
    Given several documents, generates 1 dict of all words
    used in the whole corpus, and a dict for each document.
    The dicts map the word to its frequency
    '''
    progress = tqdm(desc='JSON Files parsed:')
    corpus = {}
    num_docs = 0
    data = []
    
    for d in documents(stop_after=100000):
        if 'indexed_abstract' not in d or 'fos' not in d:
            continue
        
        doc_id = d['id']
        title = d['title']
        ia = d['indexed_abstract']['InvertedIndex']
        fos = max(
            [(p['name'], p['w']) for p in d['fos']], 
            key=lambda x : x[1]
        )[0]
        
        doc_dict = {word.lower() : len(ixs) for word, ixs in ia.items()}
        
        # Add by doc index for easier conversion into graph later
        for word in doc_dict.keys():
            if word in corpus:
                corpus[word].add(num_docs)
            else:
                corpus[word] = {num_docs}

        pickle.dump(
            doc_dict, 
            open(DICTS + doc_id, 'wb+'),
            protocol=pickle.HIGHEST_PROTOCOL
        )

        data.append((doc_id, title, fos))
        num_docs += 1
        progress.update()

    df = pd.DataFrame(data, columns=['id', 'title', 'fos'])
    
    if save:
        pickle.dump(
            corpus, 
            open(CORPUS_F, 'wb+'),
            protocol=pickle.HIGHEST_PROTOCOL
        )
        
        df.to_pickle(CSV)
    
    print(num_docs)
    return df

run(get_documents)