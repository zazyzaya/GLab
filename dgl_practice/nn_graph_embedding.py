'''
Based on Neural Network-based Graph Embedding for Cross-Platform Binary Code Similarity Detection
Xu et al., 2017
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
import dgl.function as fn 

class GEModel(nn.Module):
    def __init__(self, v_params=5, embed_size=10, embedding_depth=4):
        super(GEModel, self).__init__()
        self.v_param_layer = nn.Linear(v_params, embed_size)
        self.embed_size = embed_size
        
        # Tunable number of conv layers. Paper says 4 is best
        self.conv_layers = nn.ModuleList(
            [nn.Linear(embed_size, embed_size) for _ in range(embedding_depth)]
        )

    def forward(self, xv, mu):
        for layer in self.conv_layers:
            mu = F.relu(layer(mu))
        
        xv = self.v_param_layer(xv)
        return torch.tanh(xv + mu)

class SiameseNetwork(nn.Module):
    def __init__(self, Embedder, iters=5):
        super(SiameseNetwork, self).__init__()
        self.embedder = Embedder
        self.iters=iters
        self.linear = nn.Linear(Embedder.embed_size, Embedder.embed_size)

    def forward(self,GP): 
        # Convolutional layer building up 'mu' vector for each node
        for g in [GP.g1, GP.g2]:
            g.ndata['mu'] = torch.zeros((g.number_of_nodes(), self.embedder.embed_size))
            for i in range(self.iters):
                g.update_all(
                    message_func=fn.copy_src(src='mu', out='mail'),
                    reduce_func=fn.sum('mail', 'mu_sum')
                )

                g.ndata['mu'] = self.embedder(g.ndata['x'].float(), g.ndata['mu_sum'])
        
        # Turn sum of mu's into one vector
        v1 = self.linear(sum(GP.g1.ndata['mu'])).view(1,10)
        v2 = self.linear(sum(GP.g2.ndata['mu'])).view(1,10)
        sim = F.cosine_similarity(v1, v2)

        return sim


''' Container class for pairs of graphs. Makes batching possible
'''
class GraphPair():
    def __init__(self,g1,g2):
        self.g1 = g1
        self.g2 = g2