from collections import OrderedDict

import gensim
from pickle import dump
import numpy as np
import random


class DataPreparatorHier:

    def __init__(self, batch_size, path_to_dataset):
        self.batch_size = batch_size
        self.model_w2v = gensim.models.KeyedVectors.load_word2vec_format('D:/Questions/GoogleNews-vectors-negative300.bin', binary=True)
        self.path_to_dataset = path_to_dataset
        self.raw_text = self.load_and_clean_text()
        self.clusters = self.load_clusters()
        self.reversed_clusters = self.reverse_clusters()
        self.short_words = self.load_short_words()
        self.length = 5
        self.needed_clusters = ['26','3','42','20','0','1','2','4','5']


    def load_and_clean_text(self):
        file = open(self.path_to_dataset, 'r')
        text = file.read()
        file.close()
        lines = text.split('\n')
        random.shuffle(lines)
        text = ' '.join(lines)
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

    def reverse_clusters(self):
        reversed = {}
        for k, v in self.clusters.items():
            for word in v:
                reversed[word] = k
        return reversed

    def load_clusters(self):
        clusters = OrderedDict()
        with open('D:\\Typing\\data\\models_hier\\model_6\\clusters_30.txt', 'r') as f:
            lines = f.read().split('\n')
            for line in lines:
                parts = line.split(';')
                if len(parts) < 2:
                    continue
                words = parts[1].split(',')[:-1]
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
        clusters = []
        outputs = []
        samples_counter = {}
        clusters_counter = {}
        for i in range(self.length, len(self.raw_text)):
            seq = self.raw_text[i-self.length:i+1]
            # if seq[-1] not in self.reversed_clusters.keys() or self.reversed_clusters[seq[-1]] not in self.needed_clusters:
            #     continue
            if seq[-1] not in self.reversed_clusters.keys():
                continue
            target_cluster = self.reversed_clusters[seq[-1]]
            if target_cluster in clusters_counter.keys():
                if clusters_counter[target_cluster] > 30000:
                    continue
                else:
                    clusters_counter[target_cluster] += 1
            else:
                clusters_counter[target_cluster] = 1


            if seq[-1] in samples_counter.keys():
                if samples_counter[seq[-1]] > 300:
                    continue
                else:
                    samples_counter[seq[-1]] += 1
            else:
                samples_counter[seq[-1]] = 1
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
                if word in self.reversed_clusters.keys():
                    vector_clusters = np.zeros(len(self.clusters)+1)
                    cluster = int(self.reversed_clusters[word])
                    vector_clusters[cluster] = 1
                    clusters.append(vector_clusters)
                else:
                    vector_clusters = np.zeros(len(self.clusters) + 1)
                    vector_clusters[-1] = 1
                    clusters.append(vector_clusters)
            target_word = self.clusters[target_cluster].index(seq[-1])
            output = []
            for k, v in self.clusters.items():
                array = np.zeros(len(v)+1)
                if k == target_cluster:
                    array[target_word] = 1.0
                else:
                    array[-1] = 1.0
                output.append(array)
            outputs.append(output)
            if len(outputs) == self.batch_size:
                with open(path+str(num), 'w') as f:
                    for x in range(len(outputs)):
                        emb = embeddings[self.length*x:self.length*x+self.length]
                        short = short_words[self.length*x:self.length*x+self.length]
                        clust = clusters[self.length*x:self.length*x+self.length]
                        f.write(self.vectors_to_string(emb)+';'+self.vectors_to_string(short)+';'+self.vectors_to_string(clust)+';'+self.multi_output_to_string(outputs[x])+'\n')
                embeddings = []
                outputs = []
                short_words = []
                num += 1
                print 'Printed', str(num)
                if num == 50000:
                    return

    def statistics_on_clusters(self):
        samples = {}
        samples_counter = {}
        for i in range(self.length, len(self.raw_text)):
            seq = self.raw_text[i-self.length:i+1]
            # if seq[-1] not in self.reversed_clusters.keys() or self.reversed_clusters[seq[-1]] not in self.needed_clusters:
            #     continue
            if seq[-1] not in self.reversed_clusters.keys():
                continue
            cluster = self.reversed_clusters[seq[-1]]
            if cluster in samples_counter.keys():
                if samples_counter[cluster] > 10000:
                    continue
                else:
                    samples_counter[cluster] += 1
            else:
                samples_counter[cluster] = 1
            if cluster in samples.keys():
                words = samples[cluster]
                if seq[-1] in words:
                    words[seq[-1]] += 1
                else:
                    words[seq[-1]] = 1
            else:
                words = OrderedDict()
                words[seq[-1]] = 1
                samples[cluster] = words
        with open('D:\\Typing\\data\\models_hier\\model_4\\clusters_stat.txt', 'w') as f:
            for clust, w in samples.items():
                w = OrderedDict(sorted(w.items(), key=lambda t: t[1]))
                f.write(str(clust)+'\n')
                sum = 0
                for k, v in w.items()[::-1]:
                    sum += v
                    f.write(k+':'+str(v)+'\n')
                f.write('All words in cluster: ' + str(sum))
                f.write('\n')



    def print_info(self):
        for k, v in self.clusters.items():
            print k, str(len(v))

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
                        f.write(self.vectors_to_string(emb)+';'+self.vectors_to_string(short)+';' + self.multi_output_to_string(outputs[x])+'\n')
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

    def multi_output_to_string(self, vectors):
        s = ''
        for vector in vectors:
            for v in vector:
                s += str(v)+','
            s += ';'
        return s[:-1]