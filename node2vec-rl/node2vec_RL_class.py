import random
import numpy as np
import networkx as nx

from tqdm import tqdm 
from typing import Callable
from node2vec import Node2Vec
from collections import defaultdict
from joblib import Parallel, delayed

default_alpha = lambda x : 1.0

class Node2VecRL(Node2Vec):
    def __init__(self, graph, dimensions=128, walk_length=80, num_walks=10, p=1, q=1, weight_key='weight',
                 workers=1, sampling_strategy=None, quiet=False, opt_init: float = 0.0, epsilon: float = 0.05, 
                 alpha: Callable[[float], float] = default_alpha):

        """ 
        New fields: 
        :param opt_init:    sets optimistic initial probabilities to whatever this value is
        :param epsilon:     probability next node is selected randomly
        :param alpha:       function to update Q values based on time. Returns a value between 0 and 1. 
                            E.g. Q_t = Q_{t-1} + a(t)*(r - Q_{t-1})
        """
        self.opt_init = 0
        self.epsilon = epsilon
        self.alpha = alpha

        super().__init__(graph, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, p=p,
                 q=q, weight_key=weight_key, workers=workers, sampling_strategy=sampling_strategy,
                 quiet=quiet)
                

    # Only place empty dict at each node. Calculate probabilities on the fly during walks
    def _precompute_probabilities(self):
        """
        Precomputes transition probabilities for each node.
        """

        d_graph = defaultdict(dict)

        nodes_generator = self.graph.nodes() if self.quiet \
            else tqdm(self.graph.nodes(), desc='Computing transition probabilities')

        for source in nodes_generator:
            # Init probabilities dict for first travel
            if self.PROBABILITIES_KEY not in d_graph[source]:
                d_graph[source][self.PROBABILITIES_KEY] = dict()

            for current_node in self.graph.neighbors(source):

                # Init probabilities dict
                if self.PROBABILITIES_KEY not in d_graph[current_node]:
                    d_graph[current_node][self.PROBABILITIES_KEY] = dict()

                weights = list()
                d_neighbors = list()

                # Add initial probabilities, and neighbors to dict
                for dest in self.graph.neighbors(current_node):
                    weights.append(self.opt_init)
                    d_neighbors.append(dest)

                # NOTE: this is the only changed part of the method
                # it does no calculations, and just sets the probability to a flat value
                d_graph[current_node][self.PROBABILITIES_KEY] = np.array(weights, dtype='float64')
                d_graph[current_node][self.NEIGHBORS_KEY] = np.array(d_neighbors)

            # Calculate first_travel weights for source
            first_travel_weights = []

            for destination in self.graph.neighbors(source):
                first_travel_weights.append(self.graph[source][destination].get(self.weight_key, 1))

            first_travel_weights = np.array(first_travel_weights)
            d_graph[source][self.FIRST_TRAVEL_KEY] = first_travel_weights / first_travel_weights.sum()

        return d_graph

    # Only changing Parallel algorithm
    def _generate_walks(self) -> list:
        """
        Generates the random walks which will be used as the skip-gram input.
        :return: List of walks. Each walk is a list of nodes.
        """

        flatten = lambda l: [item for sublist in l for item in sublist]

        # Split num_walks for each worker
        num_walks_lists = np.array_split(range(self.num_walks), self.workers)

        walk_results = Parallel(n_jobs=self.workers)(
            delayed(parallel_generate_walks_rl)
                                            (self.d_graph,
                                             self.walk_length,
                                             len(num_walks),
                                             idx,
                                             self.sampling_strategy,
                                             self.NUM_WALKS_KEY,
                                             self.WALK_LENGTH_KEY,
                                             self.NEIGHBORS_KEY,
                                             self.PROBABILITIES_KEY,
                                             self.FIRST_TRAVEL_KEY,
                                             self.quiet,
                                             self.opt_init,
                                             self.epsilon, 
                                             self.alpha) for
            idx, num_walks
            in enumerate(num_walks_lists, 1))

        walks = flatten(walk_results)

        return walks

# Copied from node2vec sourcecode parallel.py
def parallel_generate_walks_rl(d_graph: dict, global_walk_length: int, num_walks: int, cpu_num: int,
                            sampling_strategy: dict = None, num_walks_key: str = None, walk_length_key: str = None,
                            neighbors_key: str = None, probabilities_key: str = None, first_travel_key: str = None,
                            quiet: bool = False, opt_init: float = 0, epsilon: float = 0.05, 
                            alpha: Callable[[float], float] = default_alpha) -> list:
    """
    Generates the random walks which will be used as the skip-gram input.
    :return: List of walks. Each walk is a list of nodes.
    """

    walks = list()

    if not quiet:
        pbar = tqdm(total=num_walks, desc='Generating walks (CPU: {})'.format(cpu_num))

    for n_walk in range(num_walks):

        # Update progress bar
        if not quiet:
            pbar.update(1)

        # Shuffle the nodes
        shuffled_nodes = list(d_graph.keys())
        random.shuffle(shuffled_nodes)

        # Start a random walk from every node
        for source in shuffled_nodes:

            # Skip nodes with specific num_walks
            if source in sampling_strategy and \
                    num_walks_key in sampling_strategy[source] and \
                    sampling_strategy[source][num_walks_key] <= n_walk:
                continue

            # Start walk
            walk = [source]

            # Calculate walk length
            if source in sampling_strategy:
                walk_length = sampling_strategy[source].get(walk_length_key, global_walk_length)
            else:
                walk_length = global_walk_length


            #d_graphc = d_graph.copy()            

            # Perform walk
            while len(walk) < walk_length:
                walk_options = d_graph[walk[-1]].get(neighbors_key, None)
                probabilities = d_graph[walk[-1]][probabilities_key]

                # Skip dead end nodes
                if walk_options.size == 0:
                    break

                if random.random() < epsilon:  # For the first step
                    walk_to = np.random.choice(walk_options, size=1)[0]
                else:
                    walk_to = walk_options[np.random.choice(np.where(probabilities == probabilities.max())[0])]

                update_q(d_graph, walk[-1], walk_to, probabilities_key, neighbors_key, alpha, len(walk))
                walk.append(walk_to)

            walk = list(map(str, walk))  # Convert all to strings
            walks.append(walk)

    if not quiet:
        pbar.close()

    return walks

def update_q(d, source, dest, pk, nk, alpha, t):
    sn = d[source][nk]
    dn = d[dest][nk]

    sidx = np.argwhere(sn == dest)
    # Don't reward visiting the same node over and over
    if d[source][pk][sidx] > 0:
        d[source][pk][sidx] -= 1
    
    # Find the number of neighbors the two share (want to reward exploring nodes closely connected to each other)
    else:
        didx = np.argwhere(dn == source)
        
        reward = np.intersect1d(sn, dn).size - 1
        d[source][pk][sidx] += alpha(t) * (reward - d[source][pk][sidx])
        d[dest][pk][didx] += alpha(t) * (reward - d[dest][pk][didx])