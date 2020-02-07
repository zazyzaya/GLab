import datetime as dt
import networkx as nx
import re, os
import numpy as np
from math import log, floor
from edgecentric_class import EdgeCentricInterface

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

class EdgeCentricPico(EdgeCentricInterface):
	def __init__(self, alpha=5, mc=20, ic=2):
		super().__init__(mc=mc, ic=ic, nodefeat='isLocal', ignore_list=['last_ts'], edge_ct='edge_ct')
		self.alpha = alpha

		self.baseline = []
		self.comprimised = []
		self.__init_files()

	def run_all(self, fout='results.json', igraph=None, ograph=None):
		super().run_all(self.baseline + self.comprimised, igraph=igraph, ograph=ograph)

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
			
		# Have to format out the timestamp to just be a float
		d = dt.datetime.strptime(edge['ts'], "%Y-%m-%dT%H:%M:%S.%fZ")
		round_ts = d.timestamp()
		round_ts -= round_ts % self.alpha 

		add_ts = lambda x : x + '-' + str(round_ts)

		# Initialize vectors
		if (add_ts(edge['id.orig_h']), add_ts(edge['id.resp_h'])) not in G.edges():
			oIsLocal = edge['local_orig']
			rIsLocal = edge['local_resp']
			G.add_node(add_ts(edge['id.orig_h']), isLocal=oIsLocal)	
			G.add_node(add_ts(edge['id.resp_h']), isLocal=rIsLocal)

			proto_vec = [0.0] * len(PROTO_DICT)
			conn_vec = [0.0] * len(CONN_DICT)
			orig_pkt_vec = [0.0] * (MAX_PKTS + PKT_OFFSET)
			resp_pkt_vec = [0.0] * (MAX_PKTS + PKT_OFFSET)
			dur_vec = [0.0] * (MAX_DUR + DUR_OFFSET)
			
			# Keep track of system port traffic
			orig_p_vec = [0.0] * 1024
			resp_p_vec = [0.0] * 1024

			# Keep track of time between contact (if it exists)
			td_vec = [0.0] * 21 # Time delta 
			ts_vec = [0.0] * 24 # Hour of occurence
			last_ts = None
			edge_ct = 0

		# Or get current vectors
		else:
			old_e = G.edges()[add_ts(edge['id.orig_h']), add_ts(edge['id.resp_h'])]

			proto_vec = old_e['proto_vec']
			conn_vec = old_e['conn_vec']
			dur_vec = old_e['dur_vec']
			orig_pkt_vec = old_e['orig_pkt_vec']
			resp_pkt_vec = old_e['resp_pkt_vec']
			orig_p_vec = old_e['orig_p_vec']
			resp_p_vec = old_e['resp_p_vec']
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

		# Only keep track of system ports
		if edge['id.orig_p'] < 1024:
			orig_p_vec[edge['id.orig_p']] += 1
		if edge['id.resp_p'] < 1024:	
			resp_p_vec[edge['id.resp_p']] += 1

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
		G.add_edge(	
			add_ts(edge['id.orig_h']), add_ts(edge['id.resp_h']),
			relation='relation1',	
			orig_p_vec=orig_p_vec, 
			resp_p_vec=resp_p_vec,
			proto_vec=proto_vec,
			orig_pkt_vec=orig_pkt_vec,
			resp_pkt_vec=resp_pkt_vec,
			dur_vec=dur_vec,
			conn_vec=conn_vec, 
			ts_vec=ts_vec, 
			td_vec=td_vec,
			last_ts=d.timestamp(),
			edge_ct=edge_ct+1
		)

		# Not really necessary but just because
		return G

if __name__ == '__main__':
	EC = EdgeCentricPico(alpha=100)
	EC.run_all()
