import random
import numpy as np
import networkx as nx

from joblib import Parallel, delayed
from cord_globals import *
from tqdm import tqdm 
from gensim.models import Word2Vec

# Model parameters
NUM_WALKS = 16
WALK_LEN = 8

# W2V params
NUM_WORKERS = 16
W2V_PARAMS = {
    'size': 128,
    'workers': NUM_WORKERS,
    'sg': 0,
}

def generate_walks(num_walks, walk_len, g, node_list, cpu=0):
    '''
    Generate random walks on graph for use in skipgram
    '''
    
    # Allow random walks to be generated in parallel given list of nodes
    # for each worker thread to explore
    progress = tqdm(total=len(node_list), desc='Walks completed on CPU %d' % (cpu))
    all_nodes = list(range(NUM_DOCS))
    
    walks = []
    for starter in node_list:  
        # Skip nodes with no neighbors
        if sum(g[starter]) == 0:
            continue
        
        for _ in range(num_walks):
            walk = [str(starter)]
            n = starter
            
            # Random walk with weights based on tf-idf score
            for __ in range(walk_len):
                next_node = random.choices(
                    all_nodes,
                    weights=g[n]
                )[0]  
                
                walk.append(str(next_node))
                n = next_node  
                
            walks.append(walk)
        progress.update()
    
    return walks

def chunk_list(xs, n):
    chunk_size = int(len(xs)/n)
    
    lists = []
    for i in range(n):
        lists.append(xs[chunk_size*i: chunk_size*(i+1)])
        
    return lists

def generate_walks_parallel(g, walk_len, num_walks, workers=1):
    '''
    Distributes nodes needing embeddings across all CPUs 
    Because this is just many threads reading one datastructure this
    is an embarrasingly parallel task
    '''
    flatten = lambda l : [item for sublist in l for item in sublist]
    
    # The np.array_split method was acting weird on lists of 
    # strings, so I just remade it
    jobs = chunk_list(
        list(range(NUM_DOCS)), 
        workers
    )
    
    if workers <= 1:
        return generate_walks(num_walks, walk_len, g, jobs[0])
        
    print('Executing tasks')
    # Tell each worker to generate walks on a subset of
    # nodes in the graph
    walk_results = Parallel(n_jobs=workers, prefer='processes')(
        delayed(generate_walks)(
            num_walks, 
            walk_len,
            g,
            nl,
            cpu=idx
        ) 
        for idx, nl in enumerate(jobs, 0)    
    )
    
    return flatten(walk_results)


def embed_walks(walks, params):
    '''
    Sends walks to Word2Vec for embeddings
    '''
    print('Embedding walks...')
    model = Word2Vec(walks, **params)
    model.save(NODE_EMBEDDINGS)
    return model.wv.vectors

def load_embeddings():
    return Word2Vec.load(NODE_EMBEDDINGS).wv.vectors


if __name__ == '__main__':
    print('Loading graph')
    g = np.load(GRAPH_FILE, allow_pickle=True)
    
    print('Generating walks')
    walks = generate_walks_parallel(g, WALK_LEN, NUM_WALKS, workers=NUM_WORKERS)
    
    print('Embedding nodes')
    embed_walks(walks, W2V_PARAMS)
    