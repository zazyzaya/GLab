from edgecentric_a_b import load_graph_from_file
from edgecentric_c import load_pmfs, partition_edges
from math import log

''' Calculates Kullback-Leibler divergence between two prob distros
	____
	\            / v(x) \
	/   v(x) * ln| ---- |
	----         \ C(s) /
  forall x
'''
def KL(v, C):
	ret = 0
	for x in range(len(v)):
		if (v[x] == 0 or C[x] == 0):
			if (v[x] == 0 and C[x] == 0):
				continue
			
			# KL Div is undefined for these situations. Authors didn't specify
			# how to handle them
			else:
				#ret += v[x] * log(v[x]) if C[x] == 0 else v[x] * (log(C[x]))
				continue

		ret += v[x] * (log(v[x]) - log(C[x]))

	return ret


''' Based on definition 4 of Shah et al., 2016
'''
def score_node(g, v, vhat, isLocal, clusters):
	score = 0.0

	r = 'local' if isLocal else 'non-local'	

	# Iter through both outbound and inbound traffic relationships 
	for rel, val in vhat.items():
		f_vr = len(g.out_edges(v)) if rel == 'out' else len(g.in_edges(v))
		if f_vr == 0:
			continue
		
		for feature in clusters[r][rel].keys():	
			feat = clusters[r][rel][feature]

			feat_score = 0
			for distro_idx in range(len(feat['influence'])):
				rho = feat['influence'][distro_idx]
				C = feat['centers'][distro_idx]
				feat_score += rho * KL(val[feature], C)

			score += feat_score * f_vr

	return score


''' Iter through all nodes and give them a rank based on their inbound and outbound traffic
	"surprisingness"
'''
def rank_all(e_partition=None):
	g = load_graph_from_file()	
	c = load_pmfs()

	ln, nln = None, None
	if e_partition==None:
		ln, nln = partition_edges(g, node_list=True)
	else:
		ln, nln = e_partition

	scores = {
		'local': [], 
		'non-local': []
	}

	for k,v in ln.items():
		scores['local'].append((k, score_node(g, k, v, True, c)))

	for k,v in nln.items():
		scores['non-local'].append((k, score_node(g,k,v,False,c)))

	scores['local'].sort(key=lambda x : -x[1])
	scores['non-local'].sort(key=lambda x : -x[1])
	
	return scores
