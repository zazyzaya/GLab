import networkx as nx
import time

from gensim.models import Word2Vec
from node2vec import Node2Vec
from walk_agent import WalkAgent

''' Generate walks using guided policy
'''
def generate_rl_walks(g, m_args, w2v_args):
    # Have to correct for different var names
    q = m_args.pop('quiet')

    wa = WalkAgent(g, **m_args)
    walks = []

    for n in g.nodes():
        walks += wa.generate_random_walks(n)

    model = Word2Vec(walks, **w2v_args)
    
    # Put all the stuff we removed back in
    m_args['quiet'] = q
    return model

''' Generate walks randomly
'''
def generate_ntv_walks(g, m_args, w2v_args):
    # Ignore args for WalkAgent
    nl = m_args.pop('novelty_lag')

    model = Node2Vec(g, **m_args)
    
    m_args['novelty_lag'] = nl
    return model.fit(**w2v_args)
    

def test(msg, vector_generator, g, m_args, w2v_args, y, judge, num_tests=100):
    accs = []
    for i in range(num_tests):
        vectors = vector_generator(g, m_args, w2v_args).wv.vectors
        accs.append(judge(vectors, y))
    
    print('\n'+msg)
    print('Avg accuracy from %d runs: %f' % (num_tests, sum(accs)/len(accs)))