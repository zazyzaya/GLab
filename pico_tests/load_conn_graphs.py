import json 
import networkx as nx
import datetime as dt

# Maps conn_state values to integers
CONN_DICT = {
	'S0': 0,
	'S1': 1, 
	'SF': 2,
	'REJ': 3,
	'S2': 4,
	'S3': 5,
	'RSTO': 6,
	'RSTR': 7,
	'RSTOS0': 8,
	'RSTRH': 9,
	'SH': 10,
	'SHR': 11,
	'OTH': 12
}

PROTO_DICT = {
	'unknown_transport': 0,
	'tcp': 1,
	'udp': 2,
	'icmp': 3
}

''' Streams in json objects to load into memory one at a time so not overwhealmed
'''
def node_streamer(fname):
	for line in open(fname, 'r'):
		yield json.loads(line)
	

''' Combines new edge with existing edge if one exists, otherwise adds new edge
	via add_edge function
'''
def add_edge_reduce(G, streamer):
	edge = next(streamer)

	# Just add edge normally if it doesn't yet exist
	if (edge['id.orig_h'], edge['id.resp_h']) not in G.edges():
		add_edge(G, streamer, new_edge=edge)
		return 
	
	# Have to format out the timestamp to just be a float
	d = dt.datetime.strptime(edge['ts'], "%Y-%m-%dT%H:%M:%S.%fZ")

	old_e = G.edges()[(edge['id.orig_h'], edge['id.resp_h'])]

	# Not guarenteed to be in the data	
	dur = 0
	if 'duration' in edge:
		dur += edge['duration']

	# Update vectors
	old_e['conn_state'][CONN_DICT[edge['conn_state']]] += 1
	old_e['proto'][PROTO_DICT[edge['proto']]] += 1

	# Add values to sets
	old_e['orig_p'].add(edge['id.orig_p'])
	old_e['resp_p'].add(edge['id.resp_p'])
	old_e['ts'].append(d.timestamp())

	G.add_edge(	
		edge['id.orig_h'], edge['id.resp_h'],
		orig_p=old_e['orig_p'], 
		resp_p=old_e['resp_p'],
		proto=old_e['proto'],
		norm_proto=[p/sum(old_e['proto']) for p in old_e['proto']],
		orig_pkts=edge['orig_pkts'] + old_e['orig_pkts'],
		resp_pkts=edge['resp_pkts'] + old_e['resp_pkts'],
		duration=dur,
		conn_state=old_e['conn_state'], 
		norm_conn_state=[s/sum(old_e['conn_state']) for s in old_e['conn_state']],
		ts=old_e['ts'] 
	)


''' Parses data from json file into edge in 
	DiGraph G
'''
def add_edge(G, streamer, new_edge=None):
	if new_edge != None:
		edge = new_edge
	else:
		edge = next(streamer)
	
	if edge['local_orig']:	
		G.add_node(edge['id.orig_h'], local=True)
		G.add_node(edge['id.resp_h'], local=False)
	else:
		G.add_node(edge['id.orig_h'], local=False)
		G.add_node(edge['id.resp_h'], local=True)

	# Have to format out the timestamp to just be a float
	d = dt.datetime.strptime(edge['ts'], "%Y-%m-%dT%H:%M:%S.%fZ")

	# Some edges aren't guarenteed to be in the data	
	if 'duration' in edge:
		dur = edge['duration']
	else:
		dur = 0

	# These two have finite options, and very few at that
	# So they are modeled as vectors 
	conn_state_vec = [0] * len(CONN_DICT)
	conn_state_vec[CONN_DICT[edge['conn_state']]] = 1

	proto_vec = [0] * len(PROTO_DICT)
	proto_vec[PROTO_DICT[edge['proto']]] = 1

	G.add_edge(
		edge['id.orig_h'], edge['id.resp_h'],
		orig_p={edge['id.orig_p']}, 
		resp_p={edge['id.resp_p']},
		proto=proto_vec,
		norm_proto=proto_vec.copy(),
		orig_pkts=edge['orig_pkts'],
		resp_pkts=edge['resp_pkts'],
		duration=dur,
		conn_state=conn_state_vec, 
		norm_conn_state=conn_state_vec.copy(),
		ts=[d.timestamp()]
	)


''' Loads n graph edges or whole graph from file. 
	Can also be called multiple times on different files with the append_to option
	Pass a (Multi)DiGraph to append_to and it will continue adding to it
	Use default for regular DiGraph or use add_func=add_edge for MultiDiGraph
'''
def load_edges(fname, n=1000, load_all=False, append_to=None, add_func=add_edge_reduce):
	streamer = node_streamer(fname)

	if append_to == None:
		G = nx.DiGraph()
	else:
		G = append_to

	try:
		if load_all:
			while (True):
				add_func(G, streamer)
	
		else:
			for _ in range(n):
				add_func(G, streamer)
	
	except(StopIteration):
		if not load_all:
			print("Graph contained " + str(_) + "edges. Stopped before loading " + str(n))

	return G
