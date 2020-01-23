'''
Tests the theories of 
Based on Neural Network-based Graph Embedding for Cross-Platform Binary Code Similarity Detection
Xu et al., 2017

using the AIDS molecule dataset
'''

import os
from graphcollection import GraphCollection
from nn_graph_embedding import GEModel, SiameseNetwork, GraphPair

import torch
import torch.optim as optim
import torch.nn as nn
from random import randint

NUM_EPOCHS = 100
BATCH_SIZE = 1000
WEIGHTS = 'weights.dat'

# Build graph collection
GC = GraphCollection(os.path.join('..', 'data', 'AIDS'))

# Build model
embedding = GEModel()#.cuda()
siamese = SiameseNetwork(embedding)#.cuda()

# Build optimizer. Paper recommends Adam w lr=0.0001
loss_fn = nn.MSELoss()
opt = optim.Adam(siamese.parameters(), lr=0.0001)

train = GC.get_train()
sizes = (len(train[0])-1, len(train[1])-1)

def batch(train, sizes):
    X = []
    y = []
    for i in range(BATCH_SIZE):
        # Randomly select two graphs
        c1 = randint(0,1)
        c2 = randint(0,1)

        g1 = train[c1][randint(0, sizes[c1])]
        g2 = train[c2][randint(0, sizes[c2])]

        X.append(GraphPair(g1,g2))
        y.append(torch.tensor([1 - (c1^c2)]))

    return X, y

# Finally, train
for epoch in range(NUM_EPOCHS):
    X, y = batch(train, sizes)
    siamese.train()

    # Unfortunately, I can't figure out a good way to batch these so we just iterate
    for i in range(len(X)):
        yhat = siamese(X[i])
    
        loss = loss_fn(y[i], yhat)
        loss.backward()
        opt.step()

    print(loss.item())


torch.save(siamese.state_dict(), WEIGHTS)