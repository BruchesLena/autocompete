import gensim
import numpy as np
from pickle import dump


class DataPreparator:
    def __init__(self, batch_size, path_to_dataset, dir):
        self.batch_size = batch_size
        self.model_w2v = gensim.models.KeyedVectors.load_word2vec_format('D:/Questions/GoogleNews-vectors-negative300.bin', binary=True)
        self.path_to_dataset = path_to_dataset
        self.raw_text = self.load_and_clean_text()
        self.chars = sorted(list(set(self.raw_text)))
        self.mapping = dict((c, i) for i, c in enumerate(self.chars))
        dump(self.mapping, open(dir + str(len(self.mapping))+'_mapping.pkl', 'wb'))
        self.vocab_size = len(self.mapping)
        self.length = 20

    def load_and_clean_text(self):
        file = open(self.path_to_dataset, 'r')
        text = file.read()
        file.close()
        raw_text = text.lower()
        tokens = raw_text.split()
        raw_text = ' '.join(tokens)
        return raw_text

    def prepare(self, path):
        num = 0
        encoded_chars = []
        embeddings = []
        outputs = []
        for i in range(self.length, len(self.raw_text)):
            seq = self.raw_text[i-self.length:i+1]
            encoded_seq = [self.mapping[char] for char in seq]
            encoded_chars.append(encoded_seq[:-1])
            outputs.append(encoded_seq[-1])
            for index, char in enumerate(seq[:len(seq) - 1]):
                tokens = seq[0:index].split(' ')
                try:
                    vector = self.model_w2v[tokens[-1]]
                    embeddings.append(vector)
                except KeyError:
                    embeddings.append(np.zeros(300, dtype=np.float32))
            if len(outputs) == self.batch_size:
                with open(path+str(num), 'w') as f:
                    for i in range(len(outputs)):
                        emb = embeddings[20*i:20*i+20]
                        f.write(self.vector_to_string(encoded_chars[i])+';'+self.vectors_to_string(emb)+';'+str(outputs[i])+'\n')
                encoded_chars = []
                embeddings = []
                outputs = []
                num += 1
                print 'Printed', str(num)
                if num == 20000:
                    print 'vocab_size=', str(self.vocab_size)
                    return


    def vector_to_string(self, vector):
        s = ''
        for v in vector:
            s += str(v)+','
        return s[:-1]

    def vectors_to_string(self, vectors):
        s = ''
        for vector in vectors:
            for v in vector:
                s += str(v)+','
        return s[:-1]
