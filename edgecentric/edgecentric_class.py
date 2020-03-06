import json
import networkx as nx
import numpy as np
from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from abc import ABC, abstractmethod
from math import log, floor, inf

''' Abstract class to create EdgeCentric graph clusters
'''
class EdgeCentricInterface(ABC):
	def __init__(	self, nodetype='nodetype', edgetype='relation', mc=128, 
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


	''' Runs all steps of EdgeCentric 
	'''
	def run_all(self, flist, cout='clusters.json', fout='results.json', 
				igraph=None, ograph=None):
		g = None	
		
		if not igraph:	
			print("Building graph")	
			g = nx.MultiDiGraph()	
			
			for f in flist:
				print("Loading: " + f)
				g = self.load_edges(f, append_to=g)

			if ograph:
				nx.write_gpickle(g, ograph)
				
		else:
			print("Loading graph from file")
			g = nx.read_gpickle(igraph)

		print("Clustering")
		C = self.build_pmfs(g)
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

		for f in flist:
			g = self.load_edges(f, append_to=g)

		return g

	def load_edges(self, fname, append_to=None):
		streamer = self.node_streamer(fname)
		
		if append_to == None:
			G = nx.MultiDiGraph()
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
			'influence': [len(vectors)/len(c) for c in clusters],
			'centers': centers
		}


	''' Uses X-means clustering to calculate centers and influence for each group of edges 
	'''	
	def build_pmfs(self, G):
		C = {}
		tmp = {}
		keys = []
		
		# Iterate through all edges to sort them 	
		for u,v,key in G.edges(keys=True):
			e = G[u][v][key]
			rel = e[self.edgetype]
			nt = G.nodes[u][self.nodetype] 

			if nt not in tmp:
				tmp[nt] = {}
				C[nt] = {}
			if rel not in tmp[nt]:
				tmp[nt][rel] = [] 

			tmp[nt][rel].append(e)	

		# Run clustering on edge PMFs for each relation type
		# Assumes each relation has the same keys
		for nt, rels in tmp.items():
			print('\t' + nt)	
			for rel, edges in rels.items():
				C[nt][rel] = {}
				print('\t\t' + rel)
	
				if len(edges) == 0:
					continue 

				for k in edges[0].keys():
					print(k)	
					if k in self.ignore_list:
						print('ignoring')	
						continue

					vec = [self.normalize(e[k]) for e in edges]
					C[nt][rel][k] = self.__cluster(vec)

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
			get_edges = G.in_edges
		elif self.direction_of_interest == Doi.out_edges:
			get_edges = G.out_edges
		else:
			#TODO get_edges = G.out_edges G.in_edges
			get_edges = G.out_edges	

		relations = {}
		nt = nd[self.nodetype]
		tot_edge_ct = 0

		# Build node PDF via simple average (NOTE: could be tweaked?)
		for u,v,k in get_edges(n, keys=True):
			a = G[u][v][k]

			if a[self.edgetype] not in relations:
				relations[a[self.edgetype]] = [] 
			if self.edge_ct:
				tot_edge_ct += a[self.edge_ct]

			relations[a[self.edgetype]].append(a)
		
		for r, vals in relations.items():
			feat_score = 0.0
		
			# We want to draw attention to anomalous kinds of edges (?)
			if self.edge_ct:
				f_vr = tot_edge_ct / sum([v[self.edge_ct] for v in vals])
			else:
				f_vr = 1

			if r not in C[nt]:
				continue

			for key in C[nt][r].keys():
				min_strangeness = (inf, 0)
				for c_idx in range(len(C[nt][r][key]['influence'])):
					rho = C[nt][r][key]['influence'][c_idx]
					ctr = C[nt][r][key]['centers'][c_idx]
				
					pdf = np.array([v[key] for v in vals])
					pdf = np.sum(pdf, axis=0)
					pdf = self.normalize(pdf)

					s = self.strangeness(pdf, ctr)
					
					if s < min_strangeness[0]:
						min_strangeness = (s, rho)
				
				feat_score += min_strangeness[0] * min_strangeness[1]
			score += feat_score #* f_vr

		return score

	''' Can be used to edit node repr on output scores
	'''
	def format_node(self, n):
		return n	


	''' Iterates through all nodes in partitioned dict and gives them all a strangeness
		score 
	'''
	def score_all_nodes(self, C, G):
		scores = {}	

		for n, d in G.nodes.data():
			if d[self.nodetype] not in scores:
				scores[d[self.nodetype]] = [] 
			
			scores[d[self.nodetype]].append((self.format_node(n), self.score_node(n, d, C, G)))

		for nt, v in scores.items():
			v.sort(key=lambda x:  -x[1])	
			v = v[:500]
			scores[nt] = v

		return scores


	''' Method which iterates through file containing graph data and
		*** yields *** one edge at a time
	'''
	@abstractmethod
	def node_streamer(self, fname):
		pass	

	''' Method which extracts graph data from JSON dict version and adds/merges
		the edge to the graph during graph construction stage
	'''
	@abstractmethod
	def add_edge(self, G, streamer):
		pass


class EdgeCentricEnum():
	def __init__(self):
		self.out_edges = 0
		self.in_edges = 1
		self.both = 2
