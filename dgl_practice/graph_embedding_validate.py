import torch
import os
from nn_graph_embedding import SiameseNetwork, GEModel, GraphPair
from graph_embedding_test import batch, EMBED_SIZE, CONVS, WEIGHTS
from graphcollection import GraphCollection

gc = GraphCollection(os.path.join('..', 'data', 'AIDS'))
v = gc.get_validate()

embedding = GEModel(embed_size=EMBED_SIZE)
siamese = SiameseNetwork(embedding, iters=CONVS)
siamese.load_state_dict(torch.load(WEIGHTS))

sizes = [len(v[0]) - 1, len(v[1]) - 1]

X, y = batch(v, sizes)
for i in range(len(X)):
    yhat = siamese(X[i])
    print(yhat, y[i])