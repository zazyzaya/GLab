import sys

# Import CORD-19 methods
sys.path.append('../CORD_19')

HOME = '/mnt/raid0_24TB/isaiah/code/CitationNet/'

PAPERS = '/mnt/raid0_24TB/datasets/DBLP/dblp_papers_v11.txt'
CORPUS_F = HOME + 'corpus.pkl'
DICTS = HOME + 'dicts/'
CSV = HOME + 'papers.pkl'
GRAPH = HOME + 'graph.npz'
NODE_EMBEDDINGS = 'embeddings.model'
NUM_DOCS = 67543