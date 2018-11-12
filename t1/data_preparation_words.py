import gensim
from pickle import dump
import numpy as np


class DataPreparatorWords:

    def __init__(self, batch_size, path_to_dataset, dir):
        self.batch_size = batch_size
        self.model_w2v = gensim.models.KeyedVectors.load_word2vec_format('D:/Questions/GoogleNews-vectors-negative300.bin', binary=True)
        self.path_to_dataset = path_to_dataset
        self.raw_text = self.load_and_clean_text()
        # self.mapping = self.load_words(2000)
        self.mapping = self.load_cluster('3')
        dump(self.mapping, open(dir + str(len(self.mapping)) + '_mappingCluster_3.pkl', 'wb'))
        self.clusters = self.load_clusters()
        self.short_words = self.load_short_words()
        self.length = 5


    def load_and_clean_text(self):
        file = open(self.path_to_dataset, 'r')
        text = file.read()
        file.close()
        raw_text = text.lower()
        raw_text = raw_text.replace('.', '')
        raw_text = raw_text.replace(',', '')
        raw_text = raw_text.replace('?', '')
        raw_text = raw_text.replace('!', '')
        raw_text = raw_text.replace('-.', '')
        tokens = raw_text.split()
        return tokens

    def load_words(self, words):
        mapping = {}
        i = 0
        with open('D:\\Typing\\data\\mostCommon.txt', 'r') as f:
            for word in f.read().split('\n'):
                mapping[word] = i
                i += 1
                if i == words:
                    return mapping

    def load_cluster(self, cluster):
        clusters = self.load_clusters()
        mapping = {}
        i = 0
        for word in clusters[cluster]:
            mapping[word] = i
            i += 1
        return mapping

    def load_clusters(self):
        clusters = {}
        with open('D:\\Typing\\data\\model_clusters_10\\clusters.txt', 'r') as f:
            lines = f.read().split('\n')
            for line in lines:
                parts = line.split(';')
                if len(parts) < 2:
                    continue
                words = parts[1].split(',')
                clusters[parts[0]] = words
        return clusters


    def load_short_words(self):
        mapping = {}
        i = 0
        with open('D:\\Typing\\data\\shortWords.txt', 'r') as f:
            for word in f.read().split('\n'):
                mapping[word] = i
                i += 1
        return mapping

    def prepare(self, path):
        num = 0
        embeddings = []
        short_words = []
        outputs = []
        for i in range(self.length, len(self.raw_text)):
            seq = self.raw_text[i-self.length:i+1]
            if seq[-1] not in self.mapping.keys():
                continue
            for word in seq[:-1]:
                try:
                    embeddings.append(self.model_w2v[word])
                except KeyError:
                    embeddings.append(np.zeros(300))
                if word in self.short_words:
                    vector = np.zeros(len(self.short_words))
                    vector[self.short_words[word]] = 1
                    short_words.append(vector)
                else:
                    short_words.append(np.zeros(len(self.short_words)))
            outputs.append(self.mapping[seq[-1]])
            if len(outputs) == self.batch_size:
                with open(path+str(num), 'w') as f:
                    for x in range(len(outputs)):
                        emb = embeddings[self.length*x:self.length*x+self.length]
                        short = short_words[self.length*x:self.length*x+self.length]
                        f.write(self.vectors_to_string(emb)+';'+self.vectors_to_string(short)+';'+str(outputs[x])+'\n')
                embeddings = []
                outputs = []
                short_words = []
                num += 1
                print 'Printed', str(num)
                if num == 50000:
                    return

    def prepare_clusters(self, path):
        num = 0
        embeddings = []
        short_words = []
        outputs = []
        for i in range(self.length, len(self.raw_text)):
            seq = self.raw_text[i-self.length:i+1]
            if seq[-1] in self.clusters['2']:
                target = 0
            elif seq[-1] in self.clusters['5']:
                target = 1
            elif seq[-1] in self.clusters['8']:
                target = 2
            elif seq[-1] in self.clusters['7']:
                target = 3
            elif seq[-1] in self.clusters['1']:
                target = 4
            elif seq[-1] in self.clusters['3']:
                target = 5
            else:
                continue

            for word in seq[:-1]:
                try:
                    embeddings.append(self.model_w2v[word])
                except KeyError:
                    embeddings.append(np.zeros(300))
                if word in self.short_words:
                    vector = np.zeros(len(self.short_words))
                    vector[self.short_words[word]] = 1
                    short_words.append(vector)
                else:
                    short_words.append(np.zeros(len(self.short_words)))
            outputs.append(target)
            if len(outputs) == self.batch_size:
                with open(path+str(num), 'w') as f:
                    for x in range(len(outputs)):
                        emb = embeddings[self.length*x:self.length*x+self.length]
                        short = short_words[self.length*x:self.length*x+self.length]
                        f.write(self.vectors_to_string(emb)+';'+self.vectors_to_string(short)+';'+str(outputs[x])+'\n')
                embeddings = []
                outputs = []
                short_words = []
                num += 1
                print 'Printed', str(num)
                if num == 50000:
                    return


    def vectors_to_string(self, vectors):
        s = ''
        for vector in vectors:
            for v in vector:
                s += str(v)+','
        return s[:-1]