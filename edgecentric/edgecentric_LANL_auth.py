import numpy as np

from edgecentric_LANL import EdgeCentricLANL 


AUTH_LOGS = '/mnt/raid0_24TB/datasets/LANL_2015/data_files/auth.txt'

''' Uses edgecentric on the data found in AUTH.txt as it more closely 
	represents the anomalies logged in the RedTeam log
'''
class EdgeCentricLANL_Auth(EdgeCentricLANL):
	INDEX_MAP = {
		0: 'timestamp',
		1: 'src_user', 
		2: 'dst_user', 
		3: 'src_computer',
		4: 'dst_computer',
		5: 'auth_type',
		6: 'logon_type', 
		7: 'auth_orientation',
		8: 'success'
	}

	AUTH_TYPE = {
		'Kerberos': 0,
		'Negotiate': 1,
		'NTLM': 2,
		'?': 3 		# May not want to group all unk's into 1 relation but w/e 
	}

	LOGON_TYPE = {
		'Network': 0,
		'Service': 1,
		'Batch': 2
	}

	AUTH_ORIENTATION = {
		'LogOn': 0,
		'LogOff': 2,
		'TGS': 3,
		'TGT': 4,
		'AuthMap': 5
	}

	def __init__(self, delta=1, start_time=None, end_time=None):
		super().__init__(delta=delta, start_time=start_time, end_time=end_time)
		self.edgetype='auth_type'

	def node_streamer(self, fname):
		skip = True	
		for line in open(fname, 'r'):
			# Ignore first row of headers	
			if skip:
				skip = False
				continue
			
			d = {}
			spl = line.split(',')

			for s in range(len(spl)):
				d[self.INDEX_MAP[s]] = spl[s].strip()
			
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
		print(edge)	

		ts = int(edge['timestamp'])
		ts -= ts % self.delta
		add_ts = lambda x : x.split('@')[0] + '-' + str(ts)

		src = add_ts(edge['src_user'])
		dst = add_ts(edge['dst_user'])
		rel = edge['auth_type']

		update_edge = False
		if G.has_edge(src,dst):
			update_edge = rel in [e[self.edgetype] for e in G[src][dst].values()]
	
		# If we want to add this as a new edge
		if not update_edge:
			G.add_node(src, nodetype='default')
			G.add_node(dst, nodetype='default')

			eid = G.add_edge(
				src, dst,
				auth_type=rel,
				logon_type=np.zeros(len(self.LOGON_TYPE)),
				auth_orientation=np.zeros(len(self.AUTH_ORIENTATION)),
				success=np.zeros(2)
			)

			old_e = G[src][dst][eid]
	
		# Or if we want to update the data on an old edge
		else:
			for e in G[src][dst].values():
				if e[self.edgetype] == rel:	
					old_e = e
					break
	
		# Just ignores missing data in spots where it's missing, doesn't ignore whole row if 
		# one value is missing
		if edge['logon_type'] != '?':
			old_e['logon_type'][self.LOGON_TYPE[edge['logon_type']]] += 1
	
		if edge['auth_orientation'] != '?':
			old_e['auth_orientation'][self.AUTH_ORIENTATION[edge['auth_orientation']]] += 1
		
		if edge['success'] != '?':	
			sidx = 0 if edge['success'] == 'Success' else 0
			old_e['success']


if __name__ == '__main__':
	# Builds the edge clusters
	EC = EdgeCentricLANL_Auth(end_time=150885)

	# Builds baseline cluster centers for data that has not yet been comprimised
	print("Building graph")	
	g = EC.build_graph([AUTH_LOGS])

	print("Clustering...")
	c = EC.build_pmfs(g)

	print("Saving clusters")
	with open('clusters.json', 'w+') as f:
		f.write(json.dumps(c, indent=4))
