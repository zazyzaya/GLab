from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from edgecentric_a_b import load_graph_from_file
import json
import numpy as np

# Tunable
MAX_CLUSTERS = 20
INIT_CENTERS = 2
C_DATA = 'cluster_data.json'

''' Runs pyclustering xmeans algorithm on given data
'''
def cluster(vectors):
	init_c = kmeans_plusplus_initializer(vectors, INIT_CENTERS).initialize()

	xm = xmeans(vectors, init_c, MAX_CLUSTERS)
	xm.process()

	clusters = xm.get_clusters()
	centers = xm.get_centers()

	# Don't care as much about what cluster a node is in, so much as 
	# how many nodes are in a given cluster
	return ([len(c)/len(vectors) for c in clusters], centers)


''' Returns saved cluster info from a file 
'''
def load_pmfs():
	ret = None
	with open(C_DATA, 'r') as f:
		ret = json.loads(f.read())

	return ret


''' Builds clusters and saves their center pdf and influence (rho) to a dict
	for each kind of node (local & non-local)
'''
def build_pmfs():
	g = load_graph_from_file()
	
	print("Partitioning nodes")
	ldata, nldata = partition_edges(g)


	names = ['local', 'non-local']
	clusters = {}

	print("Clustering")
	i = 0

	for data in [ldata, nldata]:
		clusters[names[i]] = {}
		big_db = {}
	
		# Need to sort by relation before clustering
		for n in data.keys():
			for k,v in data[n].items():
				if k in big_db:
					big_db[k].append(v)
				else:
					big_db[k] = [v]

		# Then combine all items of same relation and feature into vector arrays
		for relation, vals in big_db.items():
			clusters[names[i]][relation] = {}
		
			for k in vals[0].keys():
			
				vec = [sample[k] for sample in vals]
				info = cluster(vec)
							
				clusters[names[i]][relation][k] = {}
				clusters[names[i]][relation][k]['influence']  = info[0]
				clusters[names[i]][relation][k]['centers'] = info[1]

		i += 1

	with open(C_DATA, 'w+') as f:
		f.write(json.dumps(clusters, indent=4))

	return ldata, nldata 


''' Normalizes a data dict s.t. all array values are between 1 and 0
'''
def normalize(d):
	for k,v in d.items():
		if sum(v) != 0:	
			d[k] = [x/sum(v) for x in v] 


''' Aggregate outbound edges of each node
'''
def aggregate_edges(g, n, d={}, collect_in_edges=True):
	o_edges = [e for e in g.out_edges([n])]

	all_edges = [o_edges]
	if collect_in_edges:
		i_edges = [e for e in g.in_edges([n])]
		all_edges.append(i_edges)
		

	i = 0
	names = ['out', 'in']
	d[n] = {}
	
	for edges in all_edges:
		if len(edges) == 0:
			continue

		# For ease of adding arrays, convert to np arrays 
		x,y = edges.pop()
		ret_dict = g[x][y] 
	
		if 'last_ts' in ret_dict:
			ret_dict.pop('last_ts')

		for k,v in ret_dict.items():
			ret_dict[k] = np.array(v)

		# Then combine all outgoing edges into one
		for x,y in edges:
			nd = g[x][y]
		
			if 'last_ts' in nd:
				nd.pop('last_ts')

			for k,v in nd.items():
				ret_dict[k] += np.array(v)
		
		normalize(ret_dict)
		d[n][names[i]] = ret_dict	
	
		i += 1



''' Builds a list of data vectors for each class of node: local & non-local
'''
def partition_edges(g):
	nlocal = []
	nnotlocal = []

	for n in g.nodes.data():
		if n[1]['isLocal']:
			nlocal.append(n[0])
		else:
			nnotlocal.append(n[0])
	
	# Stores corresponding node data for edges	
	local_nodes = {}
	nlocal_nodes = {}

	for n in nlocal:
		aggregate_edges(g, n, d=local_nodes)	
			
	for n in nnotlocal:
		aggregate_edges(g, n, d=nlocal_nodes)	


	return local_nodes, nlocal_nodes
