import random 

class Node():
    def __init__(self, ntype, children=[]):
        self.type = ntype,
        self.children = set()
        self.parents = set()

        for c in children:
            self.add_child(c)

    def add_child(self, child):
        child.parents.add(self)
        self.children.add(child)

    def add_parent(self, p):
        self.parents.add(p)
        p.children.add(self)

class Graph():
    def __init__(self): 
        self.nodes = {}

    def add_node(self, id, n):
        self.nodes[id] = n 

    def add_child(self, p, c):
        if c not in self.nodes:
            self.add_node(c, Node('typeless'))

        self.nodes[p].add_child(c)

    def add_parent(self, p, c):
        self.nodes[c].add_parent(p)

    ''' Adds a flow of n dummy nodes between two nodes 
    '''
    def add_flow(self, p, c, n=10):        
        last = p

        for i in range(n):
            dummy = p + '_child_' + str(i)
            self.add_child(last, dummy)
            last = dummy 

        self.add_child(dummy, c)

    ''' Adds child with random id to p
    '''
    def add_dummy_child(self, p):
        dummy = p + str(random.randint(0,1000))
        p.add_child(dummy)

        return dummy 

''' Builds graph from figure 3 of Poirot paper
'''
def create_poirot_graph():
    g = Graph()
    
    # Add all nodes 
    g.add_node('firefox1', Node('browser'))
    g.add_node('firefox2', Node('browser'))
    
    g.add_node('spoolsv1', Node('spoolsv'))
    g.add_node('spoolsv2', Node('spoolsv'))
    g.add_node('spoolsv3', Node('spoolsv'))
    
    g.add_node('cmd.exe', Node('exe'))
    g.add_node('tmp.exe', Node('exe'))
    g.add_node('Word.exe', Node('exe'))
    
    g.add_node('/access', Node('registry'))
    g.add_node('/Run', Node('registry'))
    g.add_node('/firefox', Node('registry'))
    
    g.add_node('240.1.1.1', Node('ip'))
    g.add_node('240.1.1.2', Node('ip'))
    g.add_node('240.1.1.3', Node('ip'))
    g.add_node('240.2.1.1', Node('ip'))

    g.add_node('launcher1', Node('typeless'))
    g.add_node('java1', Node('typeless'))
    g.add_node('java2', Node('typeless'))
    g.add_node('tmp.doc', Node('typeless'))
    g.add_node('launcher2', Node('typeless'))
    g.add_node('word1', Node('typeless'))
    g.add_node('word2', Node('typeless'))
    g.add_node('launcher3', Node('typeless'))

    # Add edges
    g.add_child('spoolsv1', 'cmd.exe')

    g.add_child('firefox1', 'spoolsv1')
    g.add_flow('firefox1', 'tmp.exe', n=16)
    g.add_flow('firefox1', 'spoolsv3', n=14)

    d = g.add_dummy_child('tmp.exe')
    
    g.add_child('/access', d)
    g.add_child('/access', 'spoolsv2')
    d = g.add_dummy_child('/access')
    
    g.add_dummy_child(d)
    g.add_dummy_child(d)

    g.add_flow('spoolsv3', '/Run', 18)
    g.add_flow('240.2.1.1', 'spoolsv3', 6)

    g.add_child('launcher1', 'firefox1')
    g.add_child('launcher1', 'firefox2')
    g.add_child('launcher1', 'java1')
    g.add_child('launcher1', 'java2')

    g.add_flow('firefox2', 'java1', n=2)
    g.add_flow('firefox2', 'tmp.doc', n=3)
    
    g.add_flow('tmp.doc', 'word1', n=3)
    
    g.add_child('launcher2', 'word1')
    g.add_flow('word1', '/firefox', n=5)
    g.add_flow('word1', 'Word.exe', n=3)

    g.add_flow('Word.exe', 'word1', n=5)
    g.add_child('Word.exe', 'word2')

    g.add_flow('java1', 'java2', n=2)
    g.add_flow('java2', '/firefox', n=2)

    g.add_child('/firefox', 'word2')
    g.add_child('launcher3', 'word2')

    d = g.add_dummy_child('word2')
    g.add_dummy_child(d)

    d = g.add_dummy_child('240.1.1.1')
    g.add_child(d, '240.1.1.1')
    hub = g.add_dummy_child(d) 

    d = g.add_dummy_child('240.1.1.2')
    g.add_child(d, hub)

    d = g.add_dummy_child('240.1.1.3')
    g.add_child(d, hub)

    g.add_child(hub, 'firefox2')