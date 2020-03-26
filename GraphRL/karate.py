import time
import networkx as nx

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import balanced_accuracy_score
from test import generate_ntv_walks, generate_rl_walks, test

'''
Tests the walk agent on the KarateClub dataset
'''
def pass_judgement_karate(vectors, y, verbose=False):
    c = AgglomerativeClustering().fit(vectors)
    y_hat = c.labels_
    
    acc = balanced_accuracy_score(y, y_hat)
    acc = acc if acc > 0.5 else 1-acc
    
    if verbose:
        print("Accuracy: " + str(acc))
    
    return acc

def test_karate_club():
    model_conditions = {
        'walk_length': 3,
        'num_walks': 100,
        'quiet': True,
        'novelty_lag': 3
    }

    w2v_conditions = {
        'sg': 0,
        'size': 16,
        'hs': 0, # Best
        'negative': 5, 
    }

    g = nx.karate_club_graph()
    y = [d for n,d in g.nodes(data='club')]
    y = [1 if d=='Mr. Hi' else 0 for d in y]

    print()
    start = time.time()
    test('Default N2V:', generate_ntv_walks, g, model_conditions, 
        w2v_conditions, y, pass_judgement_karate)
    end = time.time()
    print('Elapsed time: ' + str(end-start))

    print()
    start = time.time()
    test('RL N2V:', generate_rl_walks, g, model_conditions, 
        w2v_conditions, y, pass_judgement_karate)
    end = time.time()
    print('Elapsed time: ' + str(end-start))

test_karate_club()