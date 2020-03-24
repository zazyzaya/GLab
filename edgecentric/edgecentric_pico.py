import datetime as dt
import networkx as nx
import re, os
import numpy as np
import json
from math import log, floor
from edgecentric_class import EdgeCentricInterface, EdgeCentricEnum

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

MAX_PKTS = floor(log(240000)) + 1
PKT_OFFSET = 0 # So when 0 pkts sent, mark that in idx 0

MAX_DUR = floor(log(3000)) + 1
MIN_DUR = 1e-06 
DUR_OFFSET = -1*floor(log(MIN_DUR))

LOG_DIR = os.path.join('/', 'mnt', 'raid0_24TB', 'datasets', 'pico', 'bro')
EC_ENUM = EdgeCentricEnum()

class EdgeCentricPico(EdgeCentricInterface):
	def __init__(self, alpha=5, mc=10, ic=1, direction_of_interest=EC_ENUM.both):
		super().__init__(
			mc=mc, 
			ic=ic, 
			nodetype='isLocal', 
			ignore_list=['last_ts', 'edge_ct'],
			direction_of_interest=direction_of_interest
		)
		self.alpha = alpha

		self.baseline = []
		self.comprimised = []
		self.__init_files()

	def run_all(self, fout='results.json', igraph=None, ograph=None):
		super().run_all(self.baseline + self.comprimised, igraph=igraph, ograph=ograph)

	
	def format_node(self, n):	
		ip, n = n.split('-')
		n = n.split('.')[0]
		n = dt.datetime.fromtimestamp(int(n))

		return ip + ' -- ' + n.strftime('%Y-%m-%d:%H:%M:%S')

	''' Defines each group of conn files to be loaded by the graph builder
	'''
	def __init_files(self):
		r = re.compile(r"conn\..+")
		for folder in os.listdir(LOG_DIR):
			for fname in filter(r.match, os.listdir(os.path.join(LOG_DIR, folder))):
				if folder == '2019-07-19':
					if int(fname[5:7]) < 18:
						self.baseline.append(os.path.join(LOG_DIR, folder, fname))
					else:
						self.comprimised.append(os.path.join(LOG_DIR, folder, fname))
				else:
					self.comprimised.append(os.path.join(LOG_DIR, folder, fname))


	def add_edge(self, G, streamer):
		edge = next(streamer)
		old_e = None

		# Have to format out the timestamp to just be a float
		d = dt.datetime.strptime(edge['ts'], "%Y-%m-%dT%H:%M:%S.%fZ")
		round_ts = d.timestamp()
		round_ts -= round_ts % self.alpha 

		add_ts = lambda x : x + '-' + str(round_ts)
		build_e = lambda x : str(x['id.resp_p'])
	
		relation = build_e(edge)
		src = add_ts(edge['id.orig_h'])
		dst = add_ts(edge['id.resp_h'])

		has_e = G.has_edge(src, dst)
		if has_e:
			has_rel = relation in [v['relation'] for v in G[src][dst].values()]
		else:
			has_rel = False

		# Initialize vectors
		if not (has_e and has_rel):
			oIsLocal = '1' if edge['local_orig'] else '0'
			rIsLocal = '1' if edge['local_resp'] else '0'
			G.add_node(add_ts(edge['id.orig_h']), isLocal=oIsLocal)
			G.add_node(add_ts(edge['id.resp_h']), isLocal=rIsLocal)

			# Set up empty vectors
			proto_vec = np.zeros(len(PROTO_DICT))
			conn_vec = np.zeros(len(CONN_DICT))
			orig_pkt_vec = np.zeros(MAX_PKTS + PKT_OFFSET)
			resp_pkt_vec = np.zeros(MAX_PKTS + PKT_OFFSET)
			dur_vec = np.zeros(MAX_DUR + DUR_OFFSET)
			
			# Keep track of system port traffic
			#orig_p_vec = [0.0] * 1024
			#resp_p_vec = [0.0] * 1024

			# Keep track of time between contact (if it exists)
			td_vec = np.zeros(21)  # Time delta 
			ts_vec = np.zeros(24)  # Hour of occurence
			last_ts = None
			edge_ct = 0

			eid = G.add_edge(
				src, dst,	
				relation=relation
			)

			# Now define old_e as the current dictionary 
			old_e = G[add_ts(edge['id.orig_h'])][add_ts(edge['id.resp_h'])][eid]

		
		# Or get current vectors
		else:
			for e in G[src][dst].values():
				if e['relation'] == relation:
					old_e = e
					break
				
			proto_vec = old_e['proto_vec']
			conn_vec = old_e['conn_vec']
			dur_vec = old_e['dur_vec']
			orig_pkt_vec = old_e['orig_pkt_vec']
			resp_pkt_vec = old_e['resp_pkt_vec']
			#orig_p_vec = old_e['orig_p_vec']
			#resp_p_vec = old_e['resp_p_vec']
			ts_vec = old_e['ts_vec']
			td_vec = old_e['td_vec']
			last_ts = old_e['last_ts']
			edge_ct = old_e['edge_ct']

		# Not guarenteed to be in the data	
		if 'duration' in edge:
			# I don't think duration can be 0 but just in case
			if edge['duration'] != 0:	
				dur_vec[floor(log(edge['duration'])) + DUR_OFFSET] += 1

		# Update vectors
		conn_vec[CONN_DICT[edge['conn_state']]] += 1
		proto_vec[PROTO_DICT[edge['proto']]] += 1
		ts_vec[d.hour] += 1

		'''
		# Only keep track of system ports
		if edge['id.orig_p'] < 1024:
			orig_p_vec[edge['id.orig_p']] += 1
		if edge['id.resp_p'] < 1024:	
			resp_p_vec[edge['id.resp_p']] += 1
		'''

		# Avoid log(0) errors	
		if edge['orig_pkts'] != 0:
			orig_pkt_vec[PKT_OFFSET + floor(log(edge['orig_pkts']))] += 1
		else:
			orig_pkt_vec[PKT_OFFSET] += 1
		
		if edge['resp_pkts'] != 0:
			resp_pkt_vec[PKT_OFFSET + floor(log(edge['resp_pkts']))] += 1
		else:
			resp_pkt_vec[PKT_OFFSET] += 1
			
		# Only care about ts delta
		if last_ts != None:
			idx = 0 if d.timestamp()-last_ts == 0 else floor(log(abs(d.timestamp()-last_ts)))
			td_vec[idx] += 1

		# Finally, replace edge with aggregated version
		#old_e['orig_p_vec'] = orig_p_vec
		#old_e['resp_p_vec'] = resp_p_vec
		old_e['proto_vec'] = proto_vec
		old_e['orig_pkt_vec'] = orig_pkt_vec
		old_e['resp_pkt_vec'] = resp_pkt_vec
		old_e['dur_vec'] = dur_vec
		old_e['conn_vec'] = conn_vec 
		old_e['ts_vec'] = ts_vec
		old_e['td_vec'] = td_vec
		old_e['last_ts'] = d.timestamp()
		old_e['edge_ct'] = edge_ct+1
		
		# Not really necessary but just because
		return G

if __name__ == '__main__':	
	EC = EdgeCentricPico(alpha=1)

	print("Loading graph")
	g = EC.build_graph(EC.baseline)

	print("Clustering")
	C = EC.build_pmfs(g)
	with open('clusters.json', 'w+') as f:
		f.write(json.dumps(C, indent=4))

	print("Loading comprimised graph")
	gp = EC.build_graph(EC.comprimised)

	print("Scoring...")
	S = EC.score_all_nodes(C, gp)

	scores = json.dumps(S, indent=4)
	print(scores)
	with open('results.json', 'w+') as f:
		f.write(scores)


	'''
	print("Loading graph")	
	G = EC.load_graph('graph.pkl')
	with open('clusters.json', 'r') as f:
		C = json.loads(f.read())

	print("scoring nodes")
	s = EC.score_all_nodes(C,G)
	with open('results.json', 'w+') as f:
		f.write(json.dumps(s, indent=4))
	'''
