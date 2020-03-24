import torch
import random
import torch.cuda
import numpy as np
import networkx as nx

from tqdm import tqdm
from gensim.models import Word2Vec
from joblib import Parallel, delayed
from sklearn.metrics.pairwise import cosine_similarity

class Attr2Vec():
    def __init__(self, graph, dimensions=128, walk_length=80, num_walks=10, workers=1):
        self.g = graph
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.workers = workers

        print("Generating walks for " + str(self.g.number_of_nodes) + ' nodes')
        self.walks = self.generate_walks()


    ''' Copied from the original Node2Vec src
    '''
    def fit(self, **skip_gram_params):
        if 'workers' not in skip_gram_params:
            skip_gram_params['workers'] = self.workers

        if 'size' not in skip_gram_params:
            skip_gram_params['size'] = self.dimensions

        return Word2Vec(self.walks, **skip_gram_params)


    ''' Copied from the original Node2Vec src
    '''
    def generate_walks(self):
        flatten = lambda l: [item for sublist in l for item in sublist]

        # Split num_walks for each worker
        num_walks_lists = np.array_split(range(self.num_walks), self.workers)

        walk_results = Parallel(n_jobs=self.workers)(
            delayed(self.generate_walks_task)(len(num_walks), idx) for idx, num_walks
            in enumerate(num_walks_lists, 1))

        walks = flatten(walk_results)

        return walks


    ''' Generalized walk builder to be fed into skip gram
    '''
    def generate_walks_task(self, num_walks, idx):
        pbar = tqdm(total=num_walks, desc='Generating walks on CPU %d' % (idx) )
        walks = list()
        node_list = list(self.g.nodes())

        for w in range(num_walks):
            random.shuffle(node_list)            
            pbar.update(1)

            for n in node_list:
                walk = [n]
                walk_data = self.g.ndata['feat'][n]

                while len(walk) < self.walk_length:
                    next_node = self.select_next(walk[-1], walk_data)
                    walk_data = self.update_walk_data(walk_data, self.g.ndata['feat'][next_node])
                    walk.append(next_node)

                walk = list(map(str, walk))
                walks.append(walk)

        pbar.close()
        return walks


    ''' Placeholder. For now selects similar neighbors; in the future should use (deep?) RL
    '''
    def select_next(self, n, wv):        
        neighbors = self.g.predecessors(n)
        neighbor_attrs = torch.stack([self.g.ndata['feat'] for neigh in neighbors])

        sims = torch.nn.functional.cosine_similarity(neighbor_attrs, wv)
        v = sims.max()
        options = torch.nonzero(sims==v)

        return neighbors[options[random.randint(0, options.size(0)-1)].item()]


    ''' Placeholder. For now just averages two vectors together; in the future should use torch.linear
    '''
    def update_walk_data(self, wv, n):
         return (wv+n)/2