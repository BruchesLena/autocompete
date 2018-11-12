import gensim
from sklearn.cluster import KMeans
import numpy as np
from nltk import pos_tag
import nltk

nltk.help.upenn_tagset()

model_w2v = gensim.models.KeyedVectors.load_word2vec_format('D:/Questions/GoogleNews-vectors-negative300.bin', binary=True)

N_CLUSTERS = 30
PATH_TO_WRITE = 'D:\\Typing\\data\\models_hier\\model_6\\clusters_'+str(N_CLUSTERS)+'.txt'
FILTER = ['CC', 'DT', 'EX', 'IN', 'MD', 'PDT', 'POS', 'PRP', 'PRP$', 'RP', 'TO', 'WDT', 'WP', 'WP$', 'WRB']

def load_words(words):
    mapping = {}
    i = 0
    with open('D:\\Typing\\data\\mostCommon.txt', 'r') as f:
        for word in f.read().split('\n'):
            # pos = pos_tag([word])
            # if pos[0][1] in FILTER or '\'' in word :
            #     continue
            mapping[word] = i
            i += 1
            if i == words:
                return mapping


def load_clustered_words():
    mapping = {}
    with open('D:\\Typing\\data\\models_hier\\model_5\\1.txt', 'r') as f:
        for word in f.read().split(','):
            mapping[word] = 1
    return mapping



print 'Loading words...'
mapping = load_words(5000)
# mapping = load_clustered_words()
X = []
words = []
counter = 0
for word in mapping.keys():
    if word in model_w2v:
        X.append(model_w2v[word])
        words.append(word)
        counter += 1
    # else:
    #     X.append(np.zeros(300))

print str(counter), ' words are vectorized'
print 'Clustering...'
X = np.array(X)
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=0, max_iter=1000).fit(X)
labels = kmeans.labels_

print 'Clustering finished'
clusters = {}
i = 0
for word in words:
    cluster = labels[i]
    if cluster in clusters.keys():
        clusters[cluster].append(word)
    else:
        words = [word]
        clusters[cluster] = words
    i += 1

with open(PATH_TO_WRITE, 'w') as f:
    for k, v in clusters.items():
        f.write(str(k)+';')
        for word in v:
            f.write(word+',')
        f.write('\n')