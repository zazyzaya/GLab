'''
Loads graphs from https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets 
into dgl graph objects
'''

import os, sys 
import dgl
from dgl import DGLGraph
import networkx as nx
import numpy as np

class GraphCollection():
    ''' Builds out graph collection given a folder containing
        *_A -- Edge list
        *_edge_labels -- labels for edges that appear on line i of *_A
        *_graph_indicator -- which graph the node at line i belongs to
        *_graph_labels -- class for each graph where graphid = line num NOTE this is binary for some reason
        *_node_attributes -- vector of attributes for the node at line i
        *_node_labels -- labels for node at line i
    '''
    def __init__(self, fpath, train_ratio=0.60):
        self.root = fpath
        self.graphs = {}
        self.__build_graph_collection()
        
        self.train_idx = train_ratio
        self.test_idx = (1-train_ratio)/2 + train_ratio
        
        self.n_graphs = {}
        for k in self.graphs.keys():
            self.n_graphs[k] = len(self.graphs[k])

        self.n_graphs['total'] = sum([v for v in self.n_graphs.values()])


    ''' Return the first train_ratio percent of graphs in each class
    '''
    def get_train(self):
        ret = {}
        for k,v in self.graphs.items():
            ret[k] = v[0:int(len(v)*self.train_idx)]

        return ret


    ''' Return the graphs between train_ratio and 1-(train_ratio/2)
    '''
    def get_test(self):
        ret = {}
        for k,v in self.graphs.items():
            n = len(v)
            ret[k] = v[int(self.train_idx*n):int(self.test_idx*n)]

        return ret


    ''' Return graphs used for neither train nor test
    '''
    def get_validate(self):
        ret = {}
        for k,v in self.graphs.items():
            n = len(v)
            ret[k] = v[int(self.test_idx*n):]

        return ret


    ''' Takes networkx graph and converts it to a dgl graph then 
        adds to the dict of all graphs indexed by label
    '''
    def add_graph(self, nxg, l, edge_labels=[], use_dgl=True):
		g = None	
		if use_dgl:
        	g = DGLGraph()
		
			if len(edge_labels):
				g.from_networkx(nxg, edge_attrs=edge_labels, node_attrs=['x'])
			else:
				g.from_networkx(nxg, node_attrs=['x'])
		
		else:
			g = nxg		

        if l in self.graphs:
            self.graphs[l].append(g)

        else:
            self.graphs[l] = [g]


    ''' Builds this object with graphs in the specified directory 
    '''
    def __build_graph_collection(self):
        prefix = self.root.split(os.path.sep)[-1]
        a = open(os.path.join(self.root, prefix + '_A.txt'), 'r').read().split('\n')
        el = open(os.path.join(self.root, prefix + '_edge_labels.txt'), 'r').read().split('\n')
        gi = open(os.path.join(self.root, prefix + '_graph_indicator.txt'), 'r').read().split('\n')
        na = open(os.path.join(self.root, prefix + '_node_attributes.txt'), 'r').read().split('\n')
        nl = open(os.path.join(self.root, prefix + '_node_labels.txt'), 'r').read().split('\n')
        gl = open(os.path.join(self.root, prefix + '_graph_labels.txt'), 'r').read().split('\n')

        gid = 1
        e = 0
        g = nx.Graph()
        for i in range(len(gi)-1):
            # When graph indicator changes, graph is fully built
            if (int(gi[i]) != gid):
                
                # Now need to add edges (assumes they aren't out of order)
                u, v = [int(x) for x in a[e].split(', ')]
                while(u <= i and v <= i):
                    g.add_edge(u, v, label=int(el[e])) # For now, just assume a single digit
                    e += 1
                    u, v = [int(x) for x in a[e].split(', ')]
                
                self.add_graph(g, int(gl[gid]), edge_labels=['label'])
                g = nx.Graph()
                gid = int(gi[i])

            attrs = np.fromstring(na[i] + ', ' + nl[i], sep=', ')
            g.add_node(i+1, x=attrs)

        # Need to explicitly add the last graph, as this block doesn't run in the above loop
        # I know it's ugly. I'm terribly sorry.
        u, v = [int(x) for x in a[e].split(', ')]
        while(u <= i and v <= i):
            g.add_edge(u, v, label=int(el[e])) # For now, just assume a single digit
            e += 1
            u, v = [int(x) for x in a[e].split(', ')]
