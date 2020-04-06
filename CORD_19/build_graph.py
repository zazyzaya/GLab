import os
import json 
import tqdm
import pickle
import random
import numpy as np

from math import log
from cord_globals import *

TF_IDF_THRESHOLD = 10

def tf_idf(tf, doc_count):
    idf = log(NUM_DOCS/doc_count)
    return tf*idf

def build_graph(documents):
    # Undirected, regular old graph
    g = np.zeros((NUM_DOCS, NUM_DOCS))
    corpus = pickle.load(open(CORPUS_F, 'rb'))
    
    progress = tqdm.tqdm(total=NUM_DOCS, desc='Number of nodes added:')
    for node_id in range(NUM_DOCS):
        doc_dict = pickle.load(open(WORD_DATA_FILES[node_id], 'rb'))
        neighbors = g[node_id]
        
        # Link with all papers that share significant words
        for word, count in doc_dict.items():
            thresh = tf_idf(count, len(corpus[word]['papers']))
            
            if thresh > TF_IDF_THRESHOLD:
                for paper in corpus[word]['papers']:
                    neigh_id = HASH_IDX[paper]
                    
                    # Prevent self-loops
                    if neigh_id == node_id:
                        continue
                    
                    # Edge weights are the sum of each tf-idf score of shared words
                    # This is functionally equivilant to using a multi-graph
                    # as later on, we do random walks based on these weights
                    # so P(B|A) is the same in both cases
                    neighbors[neigh_id] += thresh
    
        progress.update()
    
    np.save(GRAPH_FILE, g)
    return g
    
def test(num_nodes=100):
    docs = []
    for i in range(num_nodes):
        docs.append(random.choice(WORD_DATA_FILES))
    
    g = build_graph(docs)
    return g
    
def run():
    return build_graph(WORD_DATA_FILES)

if __name__ == '__main__':
    run()