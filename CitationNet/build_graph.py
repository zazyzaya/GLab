import pickle
import numpy as np
import pandas as pd

from tqdm import tqdm
from cn_globals import *
from math import log
from scipy.sparse import csr_matrix, save_npz

THRESHOLD = 5

def tf_idf(tf, doc_count):
    idf = log(NUM_DOCS/doc_count)
    return tf*idf

def build_graph(df):
    corpus = pickle.load(open(CORPUS_F, 'rb'))
    g = np.zeros((NUM_DOCS, NUM_DOCS))
    
    progress = tqdm(total=NUM_DOCS, desc='Nodes added')
    for i,row in df.iterrows():
        d = pickle.load(open(DICTS + str(row['id']), 'rb'))
        neighbors = g[i]
        
        for word, count in d.items():
            score = tf_idf(count, len(corpus[word]))
            
            # Add weight to edge connecting to neighbor
            if score >= THRESHOLD:
                for idx in corpus[word]:
                    neighbors[idx] += score
                    
        progress.update()
        
    print("Converting to CSR Matrix")
    g = csr_matrix(g)
    save_npz(GRAPH, g)
    
    return g
    
df = pd.read_pickle(CSV)
build_graph(df)