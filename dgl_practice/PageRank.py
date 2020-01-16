import networkx as nx
import matplotlib.pyplot as plt
import torch
import dgl 
import dgl.function as fn

N = 100
DAMP = 0.85
K = 10

# Message function tells all neighbors pv / deg
def pagerank_message_func(edges):
    return {'pv' : edges.src['pv'] / edges.src['deg'] }

# Reduce function combines all of neighbors messages
def pagerank_reduce_func(nodes):
    msgs = torch.sum(nodes.mailbox['pv'], dim=1)
    pv = ((1 - DAMP) / N) + (DAMP * msgs)
    return {'pv': pv}

g = nx.nx.erdos_renyi_graph(N, 0.1)
g = dgl.DGLGraph(g)

nx.draw(g.to_networkx(), node_size=50, node_color=[[.5,.5,.5]])
plt.show()

g.ndata['pv'] = torch.ones(N) / N                   # Initialize each node to be 1/N 
g.ndata['deg'] = g.out_degrees(g.nodes()).float()   # Store each node's degree

g.register_message_func(pagerank_message_func)
g.register_reduce_func(pagerank_reduce_func)


# Bad because does it in serial
def pagerank_naive(g):
    # Send out messages along the edges
    for u, v in zip(*g.edges()):
        g.send((u,v))

    # Recieve messages to compute new PR values
    for v in g.nodes():
        g.recv(v)

# Better because it batches messages all at once (on GPU?)
def pagerank_batch(g):
    g.send(g.edges())
    g.recv(g.nodes())

# Higher-level API that calls send and recv calls 
def pagerank_oneliner(g):
    g.update_all()

# Rather than specifying ourselves, we can let DGL figure it out
# and convert to matrix algeabra for faster execution
def pagerank_builtin(g):
    g.ndata['pv'] = g.ndata['pv'] / g.ndata['deg']
    g.update_all(
        message_func=fn.copy_src(src='pv', out='m'), # Tells it to copy pv value to neighbors
        reduce_func=fn.sum(msg='m', out='m_sum')    # sums more efficiently
    )

    g.ndata['pv'] = (1 - DAMP) / N + DAMP * g.ndata['m_sum'] # Dampens output on node level

for i in range(25):
    pagerank_builtin(g)