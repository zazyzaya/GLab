import sys
import dgl
import time
import dgl.data
import numpy as np
import networkx as nx
import multiprocessing
import matplotlib.pyplot as plt

from attr2vec import Attr2Vec
from node2vec import Node2Vec
from sklearn.svm import LinearSVC
from sklearn.utils import shuffle
from node2vec_RL_class import Node2VecRL
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import balanced_accuracy_score, v_measure_score

NUM_CPU = 16 if multiprocessing.cpu_count() > 8 else 4
TRAIN_RATIO = 0.7
VALIDATE_RATIO = 0.15
TEST_RATIO = 0.15

''' Generates networkx graphs from Coauthor dataset in dgl library
'''
def karate_iter():
	graphs = dgl.data.KarateClub()
	
	l = len(graphs)	
	for g in range(l):
		yield graphs[g].to_networkx(node_attrs=['label'])
	
def coauthor_iter():
    graphs = dgl.data.Coauthor('cs')
    for g in range(len(graphs)):
        yield graphs[g].to_networkx(node_attrs=['feat', 'label'])

def ppi_iter():
    graphs = dgl.data.AmazonCoBuy('computers')

    l = len(graphs)
    for g in range(l):
        yield graphs[g].to_networkx(node_attrs=['feat', 'label'])


def embed_karate(g, rl=True):
    # Both have best possible conditions after being tuned; note w/o RL, more dimensions are required and acc is lower
    if rl:
        ntv = Node2Vec(g, dimensions=2, walk_length=10, num_walks=16, workers=NUM_CPU)
    else:
        ntv = Node2Vec(g, dimensions=16, walk_length=16, num_walks=16, workers=NUM_CPU, quiet=True)

    model = ntv.fit(batch_words=4, window=10, min_count=1)

    return model

def embed_coauthor(g, rl=True):
    if rl:
        ntv = Attr2Vec(g, ['feat'], dimensions=80, walk_length=32, num_walks=16, workers=NUM_CPU)
    else:
        ntv = Node2Vec(g, dimensions=80, walk_length=32, num_walks=16, workers=NUM_CPU)

    print("fitting..")
    model = ntv.fit(batch_words=NUM_CPU, window=10, min_count=1)
    model.save('word_2_vec.dat')

    return model

def embed_ppi(g, rl=True):
    if rl:
        ntv = Attr2Vec(g, ['feat'], dimensions=16, walk_length=16, num_walks=16, workers=NUM_CPU)
    else:
        ntv = Node2Vec(g, dimensions=16, walk_length=16, num_walks=16, workers=NUM_CPU)

    print("fitting..")
    model = ntv.fit(batch_words=NUM_CPU, window=10, min_count=1)
    
    if rl:
        model.save('word_2_vec_rl.dat')
    else:
        model.save('word_2_vec.dat')

    return model

def karate_club(rl=True, test=False):
    gi = karate_iter()
    g = next(gi)

    ta = 0
    for i in range(25):
        m = embed_karate(g, rl=rl)

        train = int(len(m.wv.vectors) * 0.4)
        validate = train+int(len(m.wv.vectors) * 0.4) if (not test) else -1

        y = np.array([d['label'].item() for _,d in g.nodes.data()])
        X, y = shuffle(m.wv.vectors, y)

        svm = LinearSVC(C=50, max_iter=1e06, tol=1e-06)
        svm.fit(X[:train], y[:train])

        #a = AgglomerativeClustering().fit(m.wv.vectors)
        #y_hat = a.labels_
        y_hat = svm.predict(X[train:validate])

        acc = balanced_accuracy_score(y[train:validate], y_hat)
        
        '''  No longer necessary 
        if acc < 0.5:
            acc = 1-acc
            y_hat = [(i-1)*-1 for i in y_hat]
        '''

        ta += acc

        if show:
            print("\tAccuracy: " + str(acc))

            plt.scatter(X[:, 0], X[:,1], c=y, marker='o', cmap=plt.get_cmap('cool'))
            plt.scatter(X[train:validate, 0], X[train:validate,1], c=y_hat, marker='x', cmap=plt.get_cmap('cool'))
            plt.show()

    print('Avg  acc: ' + str(ta/25))	

def coauthor(rl=True, test=False):
    caIter = coauthor_iter()
    g = next(caIter)

    # Make the graph smaller so it's easier to work with
    g = nx.ego_graph(g, 0, radius=3)

    m = embed_coauthor(g, rl=rl)
    train = int(len(m.wv.vectors) * TRAIN_RATIO)
    validate = train+int(len(m.wv.vectors) * VALIDATE_RATIO) if (not test) else -1

    # TODO Try this with KNN later
    print("Training RFC...")
    y = np.array([d['label'].item() for _,d in g.nodes.data()])
    X, y = shuffle(m.wv.vectors, y)

    rfc = RandomForestClassifier(class_weight='balanced')
    rfc.fit(X[:train], y[:train])

    y_hat = rfc.predict(X[train:validate])
    acc = balanced_accuracy_score(y[train:validate], y_hat)
    
    print("Accuracy: " + str(acc))
    if show:
        plt.scatter(m.wv.vectors[:, 0], m.wv.vectors[:,1], c=y, marker='o', cmap=plt.get_cmap('cool'))
        plt.scatter(m.wv.vectors[:, 0], m.wv.vectors[:,1], c=y_hat, marker='x', cmap=plt.get_cmap('cool'))
        plt.show()

def ppi(rl=True, test=False):
    i = ppi_iter()
    g = next(i)

    # Make the graph smaller so it's easier to work with
    g = nx.ego_graph(g, 0, radius=2)

    m = embed_ppi(g, rl=rl)
    train = int(len(m.wv.vectors) * TRAIN_RATIO)
    validate = train+int(len(m.wv.vectors) * VALIDATE_RATIO) if (not test) else -1

    # TODO Try this with KNN later
    print("Training RFC...")
    y = np.array([d['label'].item() for _,d in g.nodes.data()])
    X, y = shuffle(m.wv.vectors, y)

    rfc = RandomForestClassifier(class_weight='balanced')
    rfc.fit(X[:train], y[:train])

    y_hat = rfc.predict(X[train:validate])
    acc = balanced_accuracy_score(y[train:validate], y_hat)
    
    print("Accuracy: " + str(acc))
    if show:
        plt.scatter(m.wv.vectors[:, 0], m.wv.vectors[:,1], c=y, marker='o', cmap=plt.get_cmap('cool'))
        plt.scatter(m.wv.vectors[:, 0], m.wv.vectors[:,1], c=y_hat, marker='x', cmap=plt.get_cmap('cool'))
        plt.show()


show = False
if len(sys.argv) > 1 and not set(sys.argv).isdisjoint(set(['graph', 'plot', 'show', 's'])):
    show = True

isRL = False
if len(sys.argv) > 1 and not set(sys.argv).isdisjoint(set(['rl', 'r'])):
    isRL = True

isTest = False
if len(sys.argv) > 1 and not set(sys.argv).isdisjoint(set(['test', 't'])):
    isTest=True    

# Uncomment for time tests
start = time.time()
ppi(rl=isRL, test=isTest)
end = time.time()
print("Time elapsed: " + str(end - start))