import sys
import random
import numpy as np
import networkx as nx

from joblib import Parallel, delayed
from cord_globals import *
from tqdm import tqdm 
from gensim.models import Word2Vec

# Model parameters
NUM_WALKS = 100
WALK_LEN = 3

# W2V params
NUM_WORKERS = 8
W2V_PARAMS = {
    'size': 256,
    'workers': NUM_WORKERS,
    'sg': 0,
}

all_nodes = list(range(NUM_DOCS))

def generate_walks(num_walks, walk_len, g, starter):
    '''
    Generate random walks on graph for use in skipgram
    '''
    
    # Allow random walks to be generated in parallel given list of nodes
    # for each worker thread to explore
    walks = []
    
    # Can't do much about nodes that have no neighbors
    if sum(g[starter]) == 0:
        return [[str(starter)]]
    
    for _ in range(num_walks):
        walk = [str(starter)]
        n = starter
        
        # Random walk with weights based on tf-idf score
        for __ in range(walk_len):
            # Pick a node weighted randomly from neighbors
            # Stop walk if hit a dead end
            if g[n].max() <= 0:
                break
            
            next_node = random.choices(
                all_nodes,
                weights=g[n]
            )[0]  
            
            walk.append(str(next_node))
            n = next_node 
                
        walks.append(walk)
    
    return walks

def generate_walks_parallel(g, walk_len, num_walks, workers=1):
    '''
    Distributes nodes needing embeddings across all CPUs 
    Because this is just many threads reading one datastructure this
    is an embarrasingly parallel task
    '''
    flatten = lambda l : [item for sublist in l for item in sublist]     
        
    print('Executing tasks')
    # Tell each worker to generate walks on a subset of
    # nodes in the graph
    walk_results = Parallel(n_jobs=workers, prefer='processes', mmap_mode='r')(
        delayed(generate_walks)(
            num_walks, 
            walk_len,
            g,
            node
        ) 
        for node in tqdm(range(NUM_DOCS), desc='Walks generated:')
    )
    
    return flatten(walk_results)


def embed_walks(walks, params, fname):
    '''
    Sends walks to Word2Vec for embeddings
    '''
    print('Embedding walks...')
    model = Word2Vec(walks, **params)
    model.save(fname)
    return model.wv.vectors

def load_embeddings(fname=NODE_EMBEDDINGS):
    return Word2Vec.load(fname).wv.vectors

def run():
    if len(sys.argv) > 1:
        fname = sys.argv[1]
    else:
        fname = NODE_EMBEDDINGS
    
    print('Loading graph')
    g = np.load(GRAPH_FILE, mmap_mode='r')
    
    print('Generating walks')
    walks = generate_walks_parallel(g, WALK_LEN, NUM_WALKS, workers=NUM_WORKERS)
    
    print('Embedding nodes')
    embed_walks(walks, W2V_PARAMS, fname)

if __name__ == '__main__':
    run()    
    