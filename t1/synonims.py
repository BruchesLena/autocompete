from pickle import load, dump

import gensim

word_vectors = gensim.models.KeyedVectors.load_word2vec_format(
    'D:/Questions/GoogleNews-vectors-negative300.bin', binary=True, limit=500000)



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

words = load_words(5000)
syn = {}
for word in words.keys():
    try:
        syns = word_vectors.most_similar(word, topn=3)
        cleaned = []
        for s in syns:
            if '_' in s[0]:
                continue
            cleaned.append(s[0].lower())
        syn[word] = cleaned
    except KeyError:
        continue

dump(syn, open('D:\\Typing\\data\\synonims.pkl', 'wb'))