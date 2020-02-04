import edgecentric_a_b as eca
import edgecentric_c as ecc
import edgecentric_d_e as ece

import json


print("Building graph")
eca.load_all_graphs()

print("Clustering..")
ldata, nldata = ecc.build_pmfs()

print("Finding outliers")
ranks = ece.rank_all(e_partition=(ldata, nldata))

print(json.dumps(ranks, indent=4))
with open('results.json', 'w+') as f:
	f.write(json.dumps(ranks, indent=4))
