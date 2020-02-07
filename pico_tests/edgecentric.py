import edgecentric_a_b as eca
import edgecentric_c as ecc
import edgecentric_d_e as ece

import json
import os, sys, re

ALPHA = 60*5

r = re.compile(r"conn\..+")
BASELINE_FILES = []
COMPRIMISED_FILES = []
for folder in os.listdir(eca.LOG_DIR):
	for fname in filter(r.match, os.listdir(os.path.join(eca.LOG_DIR, folder))):
		if folder == '2019-07-19':
			if int(fname[5:7]) < 18:
				BASELINE_FILES.append(os.path.join(eca.LOG_DIR, folder, fname))
			else:
				COMPRIMISED_FILES.append(os.path.join(eca.LOG_DIR, folder, fname))
		else:
			COMPRIMISED_FILES.append(os.path.join(eca.LOG_DIR, folder, fname))

if __name__ == '__main__' and len(sys.argv) == 1:
	print("Building graph")
	g = eca.load_all_graphs(aggregate=False)

	print("Clustering...")
	ecc.build_pmfs(aggregate=False, g=g)

	print("Finding outliers")
	ranks = ece.rank_all(aggregate=False, g=g)

	print(json.dumps(ranks, indent=4))
	with open('results.json', 'w+') as f:
		f.write(json.dumps(ranks, indent=4))


elif __name__ == '__main__' and sys.argv[1] in ['s', 'split']:
	print("Building graph")
	BASE_G = eca.load_list_of_graphs(BASELINE_FILES)
	COMP_G = eca.load_list_of_graphs(COMPRIMISED_FILES)

	print("Clustering..")
	ecc.build_pmfs(aggregate=False, g=BASE_G)

	print("Finding outliers")
	ranks = ece.rank_all(aggregate=False, g=COMP_G)

	print(json.dumps(ranks, indent=4))
	with open('results.json', 'w+') as f:
		f.write(json.dumps(ranks, indent=4))
	

elif __name__ == '__main__' and sys.argv[1] in ['tc', 'timecodes', 't']:
	print("Building graph")

	BASE_G = eca.load_list_of_graphs(BASELINE_FILES, add_func=eca.aggregate_by_timecode) 
	COMP_G = eca.load_list_of_graphs(COMPRIMISED_FILES, add_func=eca.aggregate_by_timecode) 
	
	print("Clustering...")
	ecc.build_pmfs(aggregate=False, g=BASE_G)

	print("Finding outliers")
	ranks = ece.rank_all(aggregate=False, g=COMP_G, strip=True)

	print(json.dumps(ranks, indent=4))
	with open('results.json', 'w+') as f:
		f.write(json.dumps(ranks, indent=4))


# Best one but still has a FPR of like 90%
elif __name__ == '__main__' and sys.argv[1] in ['tcf', 'timecodes-full']:
	print("Building graph")
	g = eca.load_all_graphs(add_func=eca.aggregate_by_timecode)

	print("Clustering")
	ecc.build_pmfs(aggregate=False, g=g)

	print("Finding outliers")
	ranks = ece.rank_all(aggregate=False, g=g, strip=True)
	
	print(json.dumps(ranks, indent=4))
	with open('results.json', 'w+') as f:
		f.write(json.dumps(ranks, indent=4))


elif __name__ == '__main__' and sys.argv[1] in ['tcfh', 'timecodes-full-hellinger']:
	print("Building graph")
	g = eca.load_all_graphs(add_func=eca.aggregate_by_timecode)

	print("Clustering")
	ecc.build_pmfs(aggregate=False, g=g)

	print("Finding outliers")
	ranks = ece.rank_all(aggregate=False, g=g, strip=True, strangeness=ece.hellinger)
	
	print(json.dumps(ranks, indent=4))
	with open('results.json', 'w+') as f:
		f.write(json.dumps(ranks, indent=4))


elif __name__ == '__main__' and sys.argv[1] in ['tcfj', 'tcfjsd', 'timecodes-full-jsd']:
	print("Building graph")
	g = eca.load_all_graphs(add_func=eca.aggregate_by_timecode)

	print("Clustering")
	ecc.build_pmfs(aggregate=False, g=g)

	print("Finding outliers")
	ranks = ece.rank_all(aggregate=False, g=g, strip=True, strangeness=ece.JSD)
	
	print(json.dumps(ranks, indent=4))
	with open('results.json', 'w+') as f:
		f.write(json.dumps(ranks, indent=4))
