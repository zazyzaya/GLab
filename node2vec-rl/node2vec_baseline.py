import dgl
import dgl.data
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import balanced_accuracy_score
from node2vec import Node2Vec

''' Generates networkx graphs from Coauthor dataset in dgl library
'''
def graph_iter():
	graphs = dgl.data.KarateClub()
	
	l = len(graphs)	
	for g in range(l):
		yield graphs[g].to_networkx(node_attrs=['label'])
	

def embed(g):
	# Best possible params after being tuned. Hits approx 0.76 Acc normally
	# (Uses 16 dimensions optimally, 2 for display)
	ntv = Node2Vec(g, dimensions=16, walk_length=16, num_walks=16, workers=1, quiet=True)
	model = ntv.fit(batch_words=4, window=10, min_count=1)

	model.save('word_2_vec.dat')
	return model

def cluster(m):
	pass

NUM_DIMS= 0
gi = graph_iter()
g = next(gi)

show = False
if len(sys.argv) > 1 and sys.argv[1] in ['graph', 'plot', 'show']:
    show = True

for wl in range(1, 6):
	ta = 0
	for i in range(10):
		m = embed(g)

		a = AgglomerativeClustering().fit(m.wv.vectors)
		y = np.array([d['label'].item() for _,d in g.nodes.data()])
		y_hat = a.labels_

		acc = balanced_accuracy_score(y, y_hat)
		acc = acc if acc >= 0.5 else 1-acc
		ta += acc

		print("\tAccuracy: " + str(acc))

		if show:
			plt.scatter(m.wv.vectors[:, 0], m.wv.vectors[:,1], c=y, marker='o', cmap=plt.get_cmap('cool'))
			plt.scatter(m.wv.vectors[:, 0], m.wv.vectors[:,1], c=y_hat, marker='x', cmap=plt.get_cmap('cool'))
			plt.show()

	print("WL: " + str(NUM_DIMS) + '\tAvg acc: ' + str(ta/10))