import os

# File locations
DATA_HOME = '/mnt/raid0_24TB/datasets/CORD-19/'
HOME = '/mnt/raid0_24TB/isaiah/code/CORD-19/'

#   JSON data
JSON_DATA_DIR =  DATA_HOME + 'data/'
JSON_DATA_DIRS = ['biorxiv_medrxiv/', 'comm_use_subset/']
DICTS = HOME + 'dictionaries/'

#   Dict data
WORD_DATA_DIR = HOME + 'dictionaries/'
CORPUS_F = WORD_DATA_DIR + 'corpus.data'

#   Graph data
GRAPH_FILE = HOME + 'graph.npy'
NODE_EMBEDDINGS = HOME + 'node_embeddings.model'

# File naming functions
F_TO_JSON = lambda x : (x.split('/')[-1]).split('.')[0] + '.json'
F_TO_DICT = lambda x : (x.split('/')[-1]).split('.')[0] + '.data'
F_TO_HASH = lambda x : F_TO_JSON(x).split('.')[0]

# Lists of all data files
#   JSON data
JSON_FILES = []
for d in JSON_DATA_DIRS:
    JSON_FILES += [JSON_DATA_DIR + d + f for f in os.listdir(JSON_DATA_DIR + d)]
#   Dict data
WORD_DATA_FILES = [WORD_DATA_DIR + F_TO_DICT(f) for f in JSON_FILES]

NUM_DOCS = len(JSON_FILES)

HASH_IDX = {F_TO_HASH(JSON_FILES[i]): i for i in range(len(JSON_FILES))}


# Added some stopwords specific to journal papers
# or that slipped through NLTK's default list    
CUSTOM_STOPWORDS = [
    "n't", 
    "'m", 
    "'re", 
    "'s", 
    "nt", 
    "may",
    "also",
    "fig",
    "http"
]