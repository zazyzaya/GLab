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
	def __init__(	self, nodetype='nodetype', edgetype='relation', mc=20, 
					ic=2, ignore_list=[], edge_ct=None, direction_of_interest=0):

		self.MAX_CLUSTERS = mc 
		self.INIT_CENTERS = ic
		self.nodetype = nodetype
		self.edgetype = edgetype
		self.ignore_list = ignore_list + [edgetype]
		self.direction_of_interest = direction_of_interest

		if edge_ct:	
			self.ignore_list.append(edge_ct)
		
		self.edge_ct = edge_ct

	def run_all(self, flist, cout='clusters.json', fout='results.json', 
				igraph=None, ograph=None):

		g = None	
		nb = None
		et = None
		
		if not igraph:	
			print("Building graph")	
			g = nx.MultiDiGraph()	
			nb = dict()
			et = set()
			for f in flist:
				print("Loading: " + f)
				g, nb, et = self.load_edges(f, append_to=(g, nb, et))

			if ograph:
				nx.write_gpickle(g, ograph)
				
				nbf = open('nb.dat', 'w+')
				etf = open('et.dat', 'w+')


				nbf.write(str(nb))
				etf.write(str(et))

				nbf.close()
				etf.close()

		else:
			print("Loading graph from file")
			nbf = open('nb.dat', 'r')
			etf = open('et.dat', 'r')
	
			g = nx.read_gpickle(igraph)
			nb = eval(nbf.read())
			et = eval(etf.read())

			nbf.close()
			etf.close()
			

		print("Clustering")
		C = self.build_pmfs(g, nb, et)
		with open(cout, 'w+') as f:
			f.write(json.dumps(C, indent=4))

		print("Scoring")
		S = self.score_all_nodes(C, g)

		print(json.dumps(S, indent=4))
		with open(fout, 'w+') as f:
			f.write(json.dumps(S, indent=4))
	
	''' Loads a graph from a file
	'''
	def load_graph(self, fname):
		return nx.read_gpickle(fname)

	def build_graph(self, flist):
		g = nx.MultiDiGraph() 
		nb = dict()
		et = set()

		for f in flist:
			g, nb, et = self.load_edges(f, append_to=(g, nb, et))

		return g, nb, et
	def load_edges(self, fname, append_to=None):
		streamer = self.__node_streamer(fname)
		
		if append_to == None:
			G = nx.MultiDiGraph()
			nb = dict() 
			et = set()
		else:
			G, nb, et = append_to

		try:
			while(True):
				self.add_edge(G, streamer, nb, et)

		except(StopIteration):
			pass

		return G, nb, et


	''' Method to normalize all distributions in an aggregated super-edge 
	'''
	def normalize(self, v):
		s = np.sum(v)
		if s:
			v = np.divide(v,s)

		return v


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
	def build_pmfs(self, G, nb, et):
		C = {}
		keys = []

		for nt, ns in nb.items():
			C[nt] = {}

			# Build empty relation map
			relations = {}	
			for e in et:
				relations[e] = []

			out_e = [(a,b) for a,b in G.out_edges(ns)]
		
			# Sort edges into groups by relation type
			# Assumes each edge is one relation without duplicates
			# Ensure this in the add_edge method
			for u,v in out_e:
				edges = [e for e in G[u][v].values()]
				for e in edges:
					relations[e[self.edgetype]].append(e)

			# Run clustering on edge PMFs for each relation type
			# Assumes each relation has the same keys
			for r, edges in relations.items():
				C[nt][r] = {}
				
				for k in edges[0].keys():
					if k in self.ignore_list:
						continue

					print('\t' + k)
					vec = [self.normalize(e[k]) for e in edges]
					C[nt][r][k] = self.__cluster(vec)

		return C


	''' Strangeness function for statistical divergence. Paper recommends KL divergence
		so that is the default, but this method may be overridden
	'''
	def strangeness(self, v, C):
		ret = 0
		alpha = 1e-06
		for i in range (len(v)):
			ret += v[i] * (log(v[i] + alpha) - log(C[i] + alpha))

		return ret


	def score_node(self, n, nd, C, G):
		score = 0.0
		Doi = EdgeCentricEnum()

		if self.direction_of_interest == Doi.in_edges: 
			edges = [(u,v) for u,v in G.in_edges(n)]
		elif self.direction_of_interest == Doi.out_edges:
			edges = [(u,v) for u,v in G.out_edges(n)]
		else:
			edges = [(u,v) for u,v in G.out_edges(n)] + [(u,v) for u,v in G.in_edges(n)]


		relations = {}
		nt = nd[self.nodetype]

		# Build node PDF via simple average (NOTE: could be tweaked?)
		for u,v in edges:
			attrs = [v for v in G[u][v].values()]
			for a in attrs:
				if a[self.edgetype] not in relations:
					relations[a[self.edgetype]] = [] 

				relations[a[self.edgetype]].append(a)
		
		for r, vals in relations.items():
			feat_score = 0.0
			
			if self.edge_ct:
				f_vr = sum([v[self.edge_ct] for v in vals])
			else:
				f_vr = 1

			
			for key in C[nt][r].keys():
				for c_idx in range(len(C[nt][r][key]['influence'])):
					rho = C[nt][r][key]['influence'][c_idx]
					ctr = C[nt][r][key]['centers'][c_idx]
				
					pdf = np.array([v[key] for v in vals])
					pdf = np.average(pdf, axis=0)
					pdf = self.normalize(pdf)

					feat_score += rho * self.strangeness(pdf, ctr)

			score += feat_score * f_vr

		return score


	''' Iterates through all nodes in partitioned dict and gives them all a strangeness
		score 
	'''
	def score_all_nodes(self, C, G):
		scores = {}	
		for n, d in G.nodes.data():
			if d[self.nodetype] not in scores:
				scores[d[self.nodetype]] = [] 
			
			scores[d[self.nodetype]].append((n, self.score_node(n, d, C, G)))

		for nt, v in scores.items():
			v.sort(key=lambda x:  -x[1])	
			v = v[:500]
			scores[nt] = v

		return scores


	''' Method which iterates through file containing graph data and
		*** yields *** one edge at a time
	'''
	def __node_streamer(self, fname):
		for line in open(fname, 'r'):
			yield json.loads(line)
				

	''' Method which extracts graph data from JSON dict version and adds/merges
		the edge to the graph during graph construction stage
	'''
	@abstractmethod
	def add_edge(self, G, streamer, nb, et):
		pass


class EdgeCentricEnum():
	def __init__(self):
		self.out_edges = 0
		self.in_edges = 1
		self.both = 2
