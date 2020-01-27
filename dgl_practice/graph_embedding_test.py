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

NUM_EPOCHS = 300
BATCH_SIZE = 10
WEIGHTS = 'weights.dat'
DIF_BIAS = 4

# Same as in paper
EMBED_SIZE = 64
CONVS = 10

def batch(train, sizes):
    X = []
    y = []
    for i in range(BATCH_SIZE):
        # Randomly select two graphs
        c1 = randint(0,1)
        c2 = randint(0,1)

        # Bias samples to pick more different than same
        if c1 == c2 and randint(1,DIF_BIAS) == 1:
            c1 = abs(c2-1)

        g1 = train[c1][randint(0, sizes[c1])]
        g2 = train[c2][randint(0, sizes[c2])]

        X.append(GraphPair(g1,g2))
        y.append(torch.tensor([(2*(1 - (c1^c2)))-1]))

    return X, y

if __name__ == '__main__':
    # Build graph collection
    GC = GraphCollection(os.path.join('..', 'data', 'AIDS'))

    # Build model
    embedding = GEModel(embed_size=EMBED_SIZE)#.cuda()
    siamese = SiameseNetwork(embedding, iters=CONVS)#.cuda()

    # Build optimizer. Paper recommends Adam w lr=0.0001
    loss_fn = nn.MSELoss()
    opt = optim.Adam(siamese.parameters(), lr=0.0001)

    train = GC.get_train()
    sizes = (len(train[0])-1, len(train[1])-1)

    # Finally, train
    for epoch in range(NUM_EPOCHS):
        X, y = batch(train, sizes)
        siamese.train()

        # Unfortunately, I can't figure out a good way to batch these so we just iterate
        avg = []
        opt.zero_grad()
        for i in range(len(X)):
            yhat = siamese(X[i])
        
            loss = loss_fn(y[i], yhat)
            loss.backward()
            avg.append(loss.item())

        opt.step()

        avg = sum(avg)/BATCH_SIZE
        print(epoch, avg)

    torch.save(siamese.state_dict(), WEIGHTS)
