from edgecentric_a_b import load_graph_from_file
from edgecentric_c import load_pmfs, partition_edges
from math import log
import datetime as dt

''' Given one event v calculate the probability of this occuring
''' 
def probability(v, C):
	ret = 1
	for x in range(len(v)):
		ret *= v[x] * C[x]

	return ret


''' Calculates Kullback-Leibler divergence between two prob distros
	____
	\            / v(x) \
	/   v(x) * ln| ---- |
	----         \ C(s) /
  forall x
'''
KL_ALPHA = 1e-06
def KL(v, C):
	ret = 0
	for x in range(len(v)):
			
		# KL Div is undefined for these situations. Authors didn't specify
		# how to handle them
		ret += v[x] * (log(v[x]+KL_ALPHA) - log(C[x]+KL_ALPHA))

	return ret


''' Based on definition 4 of Shah et al., 2016
'''
def score_node(g, v, vhat, isLocal, clusters, aggregated=True, strangeness=KL):
	score = 0.0

	v = eval(v)[0] if not aggregated else v
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
				feat_score += rho * strangeness(val[feature], C)

			score += feat_score * f_vr

	return score

def format_vector_repr(v):
	x0 = v[0].split('-')[0] 
	x1 = v[1].split('-')[0] 

	ts = v[0].split('-')[1][:-2]
	x2 = dt.datetime.utcfromtimestamp(int(ts)).strftime('%Y-%m-%d %H:%M:%S')
	
	return x0 + ', ' + x1 + ', ' +  x2


''' Iter through all nodes and give them a rank based on their inbound and outbound traffic
	"surprisingness"
'''
def rank_all(e_partition=None, aggregate=True, g=None, strip=False, strangeness=KL):
	if g == None:	
		g = load_graph_from_file()	
	
	c = load_pmfs()

	ln, nln = None, None
	if e_partition==None:
		ln, nln = partition_edges(g, aggregate=aggregate)
	else:
		ln, nln = e_partition

	scores = {
		'local': [], 
		'non-local': []
	}
	
	for k,v in ln.items():
		if strip:
			fk = format_vector_repr(eval(k))
		else:
			fk = k

		scores['local'].append((
			fk, 
			score_node(
				g, k, v, True, c, 
				aggregated=aggregate, 
				strangeness=strangeness
			)
		))

	for k,v in nln.items():
		if strip:
			fk = format_vector_repr(eval(k))
		else:
			fk = k
		
		scores['non-local'].append((
			fk, 
			score_node(
				g,k,v,False,c, 
				aggregated=aggregate, 
				strangeness=strangeness
			)
		))

	scores['local'].sort(key=lambda x : -x[1])
	scores['non-local'].sort(key=lambda x : -x[1])
	
	return scores
