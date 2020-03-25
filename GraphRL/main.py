import networkx as nx

from gensim.models import Word2Vec
from node2vec import Node2Vec
from walk_agent import WalkAgent

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import balanced_accuracy_score

g = nx.karate_club_graph()

''' Generate walks using guided policy
'''
def generate_rl_walks(g, **params):
    wa = WalkAgent(g, params.pop('num_walks'), params.pop('walk_length'))
    walks = []

    for n in g.nodes():
        walks += wa.generate_random_walks(n)

    # Have to correct for different var names
    size = params.pop('dimensions')
    params['size'] = size

    model = Word2Vec(walks, **params)
    return model

''' Generate walks randomly
'''
def generate_ntv_walks(g, **params):
    model = Node2Vec(g, **params)
    return model.fit()

''' Use skipgram generated vectors to predict node classes
'''
def pass_judgement(g, vectors, y):
    c = AgglomerativeClustering().fit(vectors)
    y_hat = c.labels_
    
    acc = balanced_accuracy_score(y, y_hat)
    acc = acc if acc > 0.5 else 1-acc
    print("Accuracy: " + str(acc))
    

test_conditions = {
    'dimensions': 4,
    'walk_length': 3,
    'num_walks': 100,
}

def test(msg, vectors):
    print('\n'+msg)
    y = [d for n,d in g.nodes(data='club')]
    y = [1 if d=='Mr. Hi' else 0 for d in y]
    pass_judgement(g, vectors, y)

print()
test('Default N2V:', generate_ntv_walks(g, **test_conditions).wv.vectors)
print()
test('RL N2V:', generate_rl_walks(g, **test_conditions).wv.vectors)