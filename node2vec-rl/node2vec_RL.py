import sys
import dgl
import dgl.data
import numpy as np
import networkx as nx
import multiprocessing
import matplotlib.pyplot as plt

from node2vec import Node2Vec
from node2vec_RL_class import Node2VecRL
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import balanced_accuracy_score

NUM_CPU = 16 if multiprocessing.cpu_count() > 8 else 4

''' Generates networkx graphs from Coauthor dataset in dgl library
'''
def karate_iter():
	graphs = dgl.data.KarateClub()
	
	l = len(graphs)	
	for g in range(l):
		yield graphs[g].to_networkx(node_attrs=['label'])
	
def coauthor_iter():
    graphs = dgl.data.Coauthor('cs')
    for g in range(len(graphs)):
        yield graphs[g].to_networkx(node_attrs=['feat', 'label'])

def embed_karate(g, rl=True):
    # Both have best possible conditions after being tuned; note w/o RL, more dimensions are required and acc is lower
    if rl:
        ntv = Node2VecRL(g, dimensions=4, walk_length=10, num_walks=16, workers=NUM_CPU, quiet=True)
    else:
        ntv = Node2Vec(g, dimensions=16, walk_length=16, num_walks=16, workers=NUM_CPU, quiet=True)

    model = ntv.fit(batch_words=4, window=10, min_count=1)

    return model

def embed_coauthor(g, rl=True):
    if rl:
        ntv = Node2VecRL(g, dimensions=4, walk_length=10, num_walks=16, workers=NUM_CPU)
    else:
        ntv = Node2Vec(g, dimensions=16, walk_length=16, num_walks=16, workers=NUM_CPU)

    print("fitting..")
    model = ntv.fit(batch_words=4, window=10, min_count=1)
    model.save('word_2_vec.dat')

    return model

def katate_club():
    gi = karate_iter()
    g = next(gi)

    ta = 0
    for i in range(10):
        m = embed_karate(g)

        a = AgglomerativeClustering().fit(m.wv.vectors)
        y = np.array([d['label'].item() for _,d in g.nodes.data()])
        y_hat = a.labels_

        acc = balanced_accuracy_score(y, y_hat)
        if acc < 0.5:
            acc = 1-acc
            y_hat = [(i-1)*-1 for i in y_hat]
        
        ta += acc

        print("\tAccuracy: " + str(acc))

        if show:
            plt.scatter(m.wv.vectors[:, 0], m.wv.vectors[:,1], c=y, marker='o', cmap=plt.get_cmap('cool'))
            plt.scatter(m.wv.vectors[:, 0], m.wv.vectors[:,1], c=y_hat, marker='x', cmap=plt.get_cmap('cool'))
            plt.show()

    print('Avg  acc: ' + str(ta/10))	

def coauthor():
    caIter = coauthor_iter()
    g = next(caIter)

    # Make the graph smaller so it's easier to work with
    # g = nx.ego_graph(g, 0, radius=10)

    m = embed_coauthor(g, rl=False)

    print("Clustering...")
    a = AgglomerativeClustering().fit(m.wv.vectors)
    y = np.array([d['label'].item() for _,d in g.nodes.data()])
    y_hat = a.labels_

    acc = balanced_accuracy_score(y, y_hat)
    if acc < 0.5:
        acc = 1-acc
        
        if show:
            y_hat = [(i-1)*-1 for i in y_hat]
    
    print("Accuracy: " + str(acc))
    if show:
        plt.scatter(m.wv.vectors[:, 0], m.wv.vectors[:,1], c=y, marker='o', cmap=plt.get_cmap('cool'))
        plt.scatter(m.wv.vectors[:, 0], m.wv.vectors[:,1], c=y_hat, marker='x', cmap=plt.get_cmap('cool'))
        plt.show()

show = False
if len(sys.argv) > 1 and sys.argv[1] in ['graph', 'plot', 'show']:
    show = True

coauthor()