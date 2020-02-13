import dgl
import dgl.data
import networkx as nx

from node2vec import Node2Vec

''' Generates networkx graphs from Coauthor dataset in dgl library
'''
def graph_iter():
	graphs = dgl.data.Coauthor('cs')
	
	l = len(graphs)	
	for g in range(l):
		yield g.to_networkx(node_attrs=['feat', 'label'])


def embed(g):
	ntv = Node2Vec(g, dimensions=64, walk_length=32, num_walks=200, workers=4)
	model = ntv.fit(batch_words=4, window=10, min_count=1)

	return model


def cluster(m):
	
