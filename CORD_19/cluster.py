import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE

NODE_EMBEDDINGS = '/mnt/raid0_24TB/isaiah/code/CORD-19/embeddings/ne_abstract_2_64_high_thresh.model'
print('Loading Embeddings')
model = Word2Vec.load(NODE_EMBEDDINGS)
v = model.wv.vectors
y = AgglomerativeClustering(n_clusters=50).fit(v).labels_

labels = model.vocabulary

# Get to a managable size before using the smart dim-reducer
print('Using TruncatedSVD')
simpler = TruncatedSVD(n_iter=50)
v = simpler.fit_transform(v)

# Nicer visuals but takes longer 
#print('Using t-SNE')
#simplest = TSNE(n_components=2)
#v = simplest.fit_transform(v)

fig, ax = plt.subplots()
sc = plt.scatter(v[:,0], v[:,1], c=y, cmap='Accent')

plt.show()