import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA

NODE_EMBEDDINGS = '/mnt/raid0_24TB/isaiah/code/CORD-19/node_embeddings.model'
model = Word2Vec.load(NODE_EMBEDDINGS)
v = model.wv.vectors
y = AgglomerativeClustering(n_clusters=50).fit(v).labels_

labels = model.vocabulary
simple = PCA(n_components=2)
v = simple.fit_transform(v)

fig, ax = plt.subplots()
sc = plt.scatter(v[:,0], v[:,1], c=y, cmap='Accent')

annot = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"))

# Stolen from SO to make plot show when hovering
def update_annot(ind):
    pos = sc.get_offsets()[ind["ind"][0]]
    annot.xy = pos
    text = "{}, {}".format(" ".join(list(map(str,ind["ind"]))), 
                           " ".join([labels[n] for n in ind["ind"]]))
    annot.set_text(text)
    annot.get_bbox_patch().set_alpha(0.4)


def hover(event):
    vis = annot.get_visible()
    if event.inaxes == ax:
        cont, ind = sc.contains(event)
        if cont:
            update_annot(ind)
            annot.set_visible(True)
            fig.canvas.draw_idle()
        else:
            if vis:
                annot.set_visible(False)
                fig.canvas.draw_idle()

fig.canvas.mpl_connect("motion_notify_event", hover)
plt.show()