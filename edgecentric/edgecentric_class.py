import json
import networkx as nx
import numpy as np
from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from abc import ABC, abstractmethod
from math import log, floor

''' Abstract class to create EdgeCentric graph clusters
'''
class EdgeCentricInterface(ABC):
	def __init__(self, nodefeat='nodetype', edgefeat='relation', mc=20, ic=2, ignore_list=[], edge_ct=None):
		self.MAX_CLUSTERS = mc 
		self.INIT_CENTERS = ic
		self.nodefeat = nodefeat
		self.edgefeat = edgefeat
		self.ignore_list = ignore_list
		if edge_ct:	
			self.ignore_list.append(edge_ct)
		
		self.edge_ct = edge_ct

	def run_all(self, flist, fout='results.json', igraph=None, ograph=None):
		g = None	
		if not igraph:	
			print("Building graph")	
			g = nx.DiGraph()	
			for f in flist:
				print("Loading: " + f)
				self.load_edges(f, append_to=g)

			if ograph:
				nx.write_gpickle(g, ograph)
		else:
			print("Loading graph from file")
			g = nx.read_gpickle(igraph)
			

		print("Clustering")
		C, db = self.build_pmfs(g)

		print("Scoring")
		S = self.score_all_nodes(db, C, g)

		print(json.dumps(S, indent=4))
		with open(fout, 'w+') as f:
			f.write(json.dumps(S, indent=4))
			

	def load_edges(self, fname, append_to=None):
		streamer = self.__node_streamer(fname)

		if append_to == None:
			G = nx.DiGraph()
		else:
			G = append_to

		try:
			while(True):
				self.add_edge(G, streamer)

		except(StopIteration):
			pass

		return G


	''' Method to normalize all distributions in an aggregated super-edge 
	'''
	def normalize(self, d):
		for k, v in d.items():
			s = np.sum(v)
			if s:
				d[k] = np.divide(v,s)

		return d


	''' Helper method for build_pmf
	'''
	def __cluster(self, vectors):
		init_c = kmeans_plusplus_initializer(vectors, self.INIT_CENTERS).initialize()

		xm = xmeans(vectors, init_c, self.MAX_CLUSTERS)
		xm.process()

		clusters = xm.get_clusters()
		centers = xm.get_centers()

		return {
			'influence': [len(c)/len(vectors) for c in clusters],
			'centers': centers
		}


	''' Uses X-means clustering to calculate centers and influence for each group of edges 
	'''
	def build_pmfs(self, G):
		db = self.partition_nodes(G)
	
		clusters = {}
		for node_type, relations in db.items():
			clusters[node_type] = {}

			for relation, vals in relations.items():
				clusters[node_type][relation] = {}
			
				val_list = vals.values()
				val_list = list(val_list)
				for k in val_list[0].keys():
					if (k in self.ignore_list):
						continue
					
					print('\t' + k)
					vec = [sample[k] for sample in vals.values()]
					clusters[node_type][relation][k] = self.__cluster(vec) 

		return clusters, db

	''' Strangeness function for statistical divergence. Paper recommends KL divergence
		so that is the default, but this method may be overridden
	'''
	def strangeness(self, v, C):
		ret = 0
		alpha = 1e-06
		for i in range (len(v)):
			ret += v[i] * (log(v[i] + alpha) - log(C[i] + alpha))

		return ret


	def score_node(self, n, pdfs, C, G, scores):
		score = 0.0
		for relation, nodes in pdfs.items():
			if n in nodes:
				if self.edge_ct:	
					f_vr = pdfs[relation][n][self.edge_ct]
				else:
					f_vr = 1

				feat_score = 0.0
				
				for key in C[relation].keys():
					for c_idx in range(len(C[relation][key]['influence'])):
						rho = C[relation][key]['influence'][c_idx]
						ctr = C[relation][key]['centers'][c_idx]
						feat_score += rho * self.strangeness(pdfs[relation][n][key], ctr)

				score += feat_score * f_vr

		return score


	''' Iterates through all nodes in partitioned dict and gives them all a strangeness
		score 
	'''
	def score_all_nodes(self, db, C, G):
		scores = {}	
		for n,dat in G.nodes.data():
			if dat[self.nodefeat] not in scores:
				scores[dat[self.nodefeat]] = [] 

			scores[dat[self.nodefeat]].append(
			(
				n,	
				self.score_node(
					n, 
					db[dat[self.nodefeat]], 
					C[dat[self.nodefeat]], 
					G, 
					scores
				)
			))

		for k, v in scores.items():
			v.sort(key= lambda x: -x[1])
			scores[k] = v[:200]

		return scores


	''' Method which iterates through file containing graph data and
		*** yields *** one edge at a time
	'''
	def __node_streamer(self, fname):
		for line in open(fname, 'r'):
			yield json.loads(line)

	
	''' Method to merge the edges of a specified node n and place into d into the proper
		edge class as { nodetype : { relation: { n: { features: [] } }  }} during final 
		merging stage

		NOTE: By default, method assumes aggregation is done in the add_edge method. 
		Override this method if more aggregation should occur at this stage, otherwise
		it assumes exactly 1 edge of each relation type is associated with a node
	'''
	def partition_nodes(self, G):
		ret = {}
		for u, v, features in G.edges.data():
			ef = features.pop(self.edgefeat)
			nf = G.nodes()[u][self.nodefeat]

			if nf in ret:
				if ef in ret[nf]:
					ret[nf][ef][u] = self.normalize(features)
				else:
					ret[nf][ef] = {u : self.normalize(features)}
			
			else:
				ret[nf] = {ef : {u : self.normalize(features)} }

		return ret
				

	''' Method which extracts graph data from JSON dict version and adds/merges
		the edge to the graph during graph construction stage
	'''
	@abstractmethod
	def add_edge(self, G, streamer):
		pass
