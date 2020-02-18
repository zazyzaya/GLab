import dgl
import dgl.data
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from sklearn.cluster import AgglomerativeClustering, OPTICS
from sklearn.metrics import balanced_accuracy_score
from node2vec_RL_class import Node2VecRL

''' Generates networkx graphs from Coauthor dataset in dgl library
'''
def graph_iter():
	graphs = dgl.data.KarateClub()
	
	l = len(graphs)	
	for g in range(l):
		yield graphs[g].to_networkx(node_attrs=['label'])
	

def embed(g):
    # Best possible params after being tuned. Hits approx 0.76 Acc normally
    global WL
    ntv = Node2VecRL(g, dimensions=2, walk_length=10, num_walks=17, workers=1, quiet=True)
    model = ntv.fit(batch_words=4, window=10, min_count=1)

    model.save('word_2_vec.dat')
    return model

def cluster(m):
	pass

WL= 0
gi = graph_iter()
g = next(gi)

for wl in range(1,5):
    ta = 0
    WL = 2**wl
    for i in range(10):
        m = embed(g)

        a = AgglomerativeClustering().fit(m.wv.vectors)
        y = np.array([d['label'].item() for _,d in g.nodes.data()])
        y_hat = a.labels_

        acc = balanced_accuracy_score(y, y_hat)
        if acc < 0.5:
            acc = 1-acc
            y_hat = [(i-1)*-1 for i in y_hat]
        
        ta += acc

        print("\tAccuracy: " + str(acc))
        plt.scatter(m.wv.vectors[:, 0], m.wv.vectors[:,1], c=y, marker='o', cmap=plt.get_cmap('cool'))
        plt.scatter(m.wv.vectors[:, 0], m.wv.vectors[:,1], c=y_hat, marker='x', cmap=plt.get_cmap('cool'))
        plt.show()

    print("WL: " + str(WL) + '\tAvg  acc: ' + str(ta/10))	