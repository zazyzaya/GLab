import numpy as np
import networkx as nx

from math import log, inf, floor
from edgecentric_class import EdgeCentricInterface

FLOWS = '/mnt/raid0_24TB/datasets/LANL_2015/data_files/flows.txt'
EDGEMAP = { 
	'timestamp' : 	0, 
	'duration' : 	1,
	'src_computer': 2,
	'src_port': 	3,
	'dst_computer': 4,
	'dst_port': 	5,
	'protocol': 	6,
	'packet_count': 7,
	'byte_count': 	8
}

# Here because I did EDGEMAP backward and don't feel like
# fixing it right now. 
INDEXMAP = {}
for k in EDGEMAP:
	INDEXMAP[EDGEMAP[k]] = k
	

MAX_DURATION = 10 # ln(10,000) ~= 10 we assume communications aren't longer
MAX_PACKETS  = 15 # See above
MAX_BYTES    = 22 # ln(1e9) ~= 21. Assume packets are less than 1 gig in size

class EdgeCentricLANL(EdgeCentricInterface):
	def __init__(self, delta=1, start_time=None, end_time=60*60*60*24*1):
		self.delta = delta 
		self.start_time = start_time if start_time else 0
		self.end_time = end_time if end_time else inf
		super().__init__(edgetype='protocol', edge_ct='edge_ct')
	

	''' Always going to use FLOWS as filename, but method signature
		has to match up. 
	'''
	def node_streamer(self, fname):
		for line in open(fname, 'r'):
			d = {}
			spl = line.split(',')
			
			for s in range(len(spl)):
				d[INDEXMAP[s]] = spl[s]
			
			ts = int(d['timestamp'])

			# Skip ahead, and stop when past end
			if ts < self.start_time:
				continue
			if ts > self.end_time:
				break

			yield d
	

	def add_edge(self, G, streamer):
		edge = next(streamer)
		old_e = None
		
		ts = int(edge['timestamp'])
		ts -= ts % self.delta
		add_ts = lambda x : x + '-' + str(ts) 

		src = add_ts(edge['src_computer'])
		dst = add_ts(edge['dst_computer'])
		rel = edge['protocol']
		nt = 'default'

		update_edge = False
		if G.has_edge(src,dst):
			update_edge = rel in [e[self.edgetype] for e in G[src][dst].values()]

		if not update_edge:
			G.add_node(src, nodetype=nt)
			G.add_node(src, nodetype=nt)
			
			et = self.edgetype
			eid = G.add_edge(
				src, dst, 
				protocol=rel,
				duration=np.zeros(MAX_DURATION),
				packet_count=np.zeros(MAX_PACKETS),
				byte_count=np.zeros(MAX_BYTES),
				edge_ct=0
			)

			old_e = G[src][dst][eid]

		else:
			for e in G[src][dst].values():
				if e[self.edgetype] == rel:
					old_e = e
					break
		
		# Add 2 so edges with 0 are seperated from all others (and to counter OOB errors)
		dur_idx = floor(log(int(edge['duration']) + 2))	
		packet_idx = floor(log(int(edge['packet_count']) + 2))
		byte_idx = floor(log(int(edge['byte_count']) + 2))

		# Modifies pointer to arrs in G 
		old_e['duration'][dur_idx] += 1
		old_e['packet_count'][packet_idx] += 1
		old_e['byte_count'][byte_idx] += 1
		old_e['edge_ct'] += 1

		return ts 

if __name__ == '__main__':
	EC = EdgeCentricLANL(end_time=60*60)
	EC.run_all([FLOWS])
