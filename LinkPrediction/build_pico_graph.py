import networkx as nx
import json

def line_loader(files):
    for kerb in files:
        with open(kerb, 'r') as f:
            
            line = f.readline()
            while line:
                line = json.loads(line)
                