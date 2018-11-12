import heapq
from collections import OrderedDict

import gensim
import numpy as np
from gensim.models import FastText
from keras import Input
from keras.engine import Model
from keras.layers import LSTM, Dense, concatenate
from keras.models import Sequential,load_model
from pickle import load

from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import time

# model_word2vec = gensim.models.KeyedVectors.load_word2vec_format('D:/Questions/GoogleNews-vectors-negative300.bin', binary=True)

class Autocompleter:

    def __init__(self, model_word_vectors):
        self.path = 'D:/Typing/data/'
        # load the model
        self.model = self.load_model()
        # load the mapping
        self.mapping = load(open(self.path + '54_mapping.pkl', 'rb'))
        # self.word_vectors = gensim.models.KeyedVectors.load_word2vec_format('D:/Questions/GoogleNews-vectors-negative300.bin', binary=True)
        self.word_vectors = model_word_vectors

    def sample(self, preds, top_n=3):
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds)
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        return heapq.nlargest(top_n, range(len(preds)), preds.take)

    def load_model(self):
        model = Sequential()
        model.add(LSTM(100, input_shape=(20, 354), return_sequences=True))
        model.add(LSTM(70))
        model.add(Dense(54, activation='softmax'))
        model.load_weights(self.path + 'models/weights_10_4')
        return model


    def predict_completion(self, text):
        original_text = text
        completion = ''
        while True:
            encoded = self.vectorize(text)

            preds = self.model.predict(encoded, verbose=0)[0]
            next_index = self.sample(preds, top_n=3)[0]
            next_char = self.get_char(next_index)
            text = text[1:] + next_char
            completion += next_char
            if len(completion) > 20:
                return completion
            end_chars = [" ", "-",',','.']
            if len(original_text + completion) + 2 > len(original_text) and next_char in end_chars:
                return completion


    def predict_completions(self, text, seq_length, n=3):
        t0 = time.time()
        encoded = self.vectorize(text)
        t1 = time.time()
        t = t1-t0
        print 'single chars vectorization: ' + str(t)
        t0 = time.time()
        preds = self.model.predict(encoded, verbose=0)[0]
        t1 = time.time()
        t = t1-t0
        print 'single chars prediction: ' + str(t)
        next_indices = self.sample(preds, top_n=n)
        return [self.get_char(idx) + self.predict_completion(text[1:] + self.get_char(idx)) for idx in next_indices]
        return result

    def get_words(self, index):
        return self.get_char(index) + self.predict_completion(self.text[1:] + self.get_char(index))

    def get_char(self, index):
        for char, i in self.mapping.items():
            if i == index:
                return char

    # generate a sequence of characters with a language model
    def generate_seq(self, model, mapping, seq_length, seed_text, n_chars):
        in_text = seed_text
        # generate a fixed number of characters
        for _ in range(n_chars):
            encoded = self.vectorize(in_text)
            # predict character
            yhat = self.model.predict_classes(encoded, verbose=0)
            # reverse map integer to character
            out_char = ''
            for char, index in mapping.items():
                if index == yhat:
                    out_char = char
                    break
            # append to input
            in_text += out_char
        return in_text

    def vectorize(self, seq):
        if len(seq) < 20:
            seq = seq.rjust(20, ' ')
        encoded_chars = []
        embeddings = []
        encoded_seq = [self.mapping[char] for char in seq]
        encoded_seq = to_categorical(encoded_seq, num_classes=len(self.mapping))
        encoded_chars.append(encoded_seq)
        for index, char in enumerate(seq[:len(seq)]):
            tokens = seq[0:index].split(' ')
            try:
                vector = self.word_vectors[tokens[-1]]
                embeddings.append(vector)
            except KeyError:
                embeddings.append(np.zeros(300, dtype=np.float32))
        encoded_chars = np.array(encoded_chars)
        embeddings = np.array(embeddings)
        inputs = np.concatenate((encoded_chars.reshape(20,54), embeddings), axis=1)
        inputs = [inputs]
        inputs = np.array(inputs)
        return inputs

    def complete(self, text):
        self.text = text.lower()
        if len(text) > 20:
            text = text[len(text) - 20:]
        completions = self.predict_completions(text, 20)
        return completions

class Autocompleter_words:

    def __init__(self):
        self.path = 'D:/Typing/data/'
        # load the model
        self.model = load_model(self.path + 'models_words/model_words_1.h5')
        # load the mapping
        self.mapping = load(open(self.path + '2000_mappingWords.pkl', 'rb'))
        self.word_vectors = gensim.models.KeyedVectors.load_word2vec_format(
            'D:/Questions/GoogleNews-vectors-negative300.bin', binary=True)
        self.short_words = self.load_short_words()

    def load_short_words(self):
        mapping = {}
        i = 0
        with open('D:\\Typing\\data\\shortWords.txt', 'r') as f:
            for word in f.read().split('\n'):
                mapping[word] = i
                i += 1
        return mapping

    def complete(self, text):
        text = text.lower().split(' ')
        if len(text) > 5:
            text = text[len(text) - 5:]
        completions = self.predict_completions(text)
        return completions

    def predict_completions(self, text):
        vector = self.vectorize(text)
        completions = self.model.predict(vector)[0]
        predictions = (-completions).argsort()[:3]
        predicted_words = []
        for p in predictions:
            for k, v in self.mapping.items():
                if v == p:
                    predicted_words.append(k)
        return predicted_words

    def vectorize(self, text):
        embeddings = []
        short_words = []
        for word in text:
            try:
                embeddings.append(self.word_vectors[word])
            except KeyError:
                embeddings.append(np.zeros(300))
            if word in self.short_words:
                vector = np.zeros(len(self.short_words))
                vector[self.short_words[word]] = 1
                short_words.append(vector)
            else:
                short_words.append(np.zeros(len(self.short_words)))
        embeddings = np.array(embeddings)
        short_words = np.array(short_words)
        vector = np.concatenate((embeddings, short_words), axis=1)
        v = [vector]
        v = np.array(v)
        return v

class Autocompleter_hier:

    def __init__(self, path, model_word_vectors):
        self.path = path
        # self.word_vectors = gensim.models.KeyedVectors.load_word2vec_format(
        #     'D:/Questions/GoogleNews-vectors-negative300.bin', binary=True)
        self.word_vectors = model_word_vectors
        self.short_words = self.load_short_words()
        self.model_hier = load_model(self.path+'model_cluster_hier.h5')
        self.mapping_clusters = {0:2, 1:5, 2:8, 3:7, 4:1, 5:3}
        self.mapping_classes = {2:172, 5:280, 8:81, 7:135, 1:113, 3:103}
        # self.model_2 = load_model(self.path+'model_cluster_2.h5')
        # self.mapping_2 = load(open(self.path + '172_mappingCluster_2.pkl', 'rb'))
        # self.model_5 = load_model(self.path+'model_cluster_5.h5')
        # self.mapping_5 = load(open(self.path + '280_mappingCluster_5.pkl', 'rb'))
        # self.model_8 = load_model(self.path+'model_cluster_8.h5')
        # self.mapping_8 = load(open(self.path + '81_mappingCluster_8.pkl', 'rb'))
        # self.model_7 = load_model(self.path+'model_cluster_7.h5')
        # self.mapping_7 = load(open(self.path + '135_mappingCluster_7.pkl', 'rb'))
        # self.model_1 = load_model(self.path+'model_cluster_1.h5')
        # self.mapping_1 = load(open(self.path + '113_mappingCluster_1.pkl', 'rb'))
        # self.model_3 = load_model(self.path+'model_cluster_3.h5')
        # self.mapping_3 = load(open(self.path + '103_mappingCluster_3.pkl', 'rb'))

    def load_short_words(self):
        mapping = {}
        i = 0
        with open('D:\\Typing\\data\\shortWords.txt', 'r') as f:
            for word in f.read().split('\n'):
                mapping[word] = i
                i += 1
        return mapping

    def complete(self, text):
        text = text.lower().split(' ')
        if len(text) > 5:
            text = text[len(text) - 5:]
        completions = self.predict_completions(text)
        return completions

    def predict_completions(self, text):
        vector = self.vectorize(text)
        completions_hier = self.model_hier.predict(vector)[0]
        predicted_cluster = np.argmax(completions_hier)
        cluster = self.mapping_clusters[predicted_cluster]
        classes = self.mapping_classes[cluster]
        completions = []
        model = load_model(self.path+'model_cluster_'+str(cluster)+'.h5')
        mapping = load(open(self.path + str(classes)+'_mappingCluster_'+str(cluster)+'.pkl', 'rb'))
        predicted_words = model.predict(vector)[0]
        predictions = (-predicted_words).argsort()[:3]
        for p in predictions:
            for k, v in mapping.items():
                if v == p:
                        completions.append(k)
        return completions

    def vectorize(self, text):
        embeddings = []
        short_words = []
        for word in text:
            try:
                embeddings.append(self.word_vectors[word])
            except KeyError:
                embeddings.append(np.zeros(300))
            if word in self.short_words:
                vector = np.zeros(len(self.short_words))
                vector[self.short_words[word]] = 1
                short_words.append(vector)
            else:
                short_words.append(np.zeros(len(self.short_words)))
        embeddings = np.array(embeddings)
        short_words = np.array(short_words)
        vector = np.concatenate((embeddings, short_words), axis=1)
        v = [vector]
        v = np.array(v)
        return v

class AutocomleterSingleHier:
    def __init__(self, path, model_word_vectors, path_to_weights, clusters_name):
        self.path = path
        self.weights = path_to_weights
        if model_word_vectors is None:
            self.word_vectors = gensim.models.KeyedVectors.load_word2vec_format(
                'D:/Questions/GoogleNews-vectors-negative300.bin', binary=True)
        else:
            self.word_vectors = model_word_vectors
        self.clusters = self.load_clusters(self.path+clusters_name)
        self.reversed_clusters = self.reverse_clusters()
        self.short_words = self.load_short_words()
        self.model_hier = self.load_model()
        self.mapping_clusters = {0:26, 1:3, 2:42, 3:20, 4:0, 5:1, 6:2, 7:4, 8:5}
        # self.mapping_clusters = {0:0, 1:1, 2:2,3:3,4:4,5:5,6:6,7:7,8:9,9:10,10:11,11:12,12:13,13:14,14:15,15:16,16:17,17:19,18:20,19:23,20:24}


    def load_model(self):
        classes = []
        for c in self.clusters:
            classes.append(len(c)+1)

        d = 331 + len(classes) + 1
        input = Input(shape=(5, d))
        input_1 = LSTM(150, return_sequences=True)(input)

        merged = concatenate([input, input_1])
        outs = []

        for i in range(len(classes)):
            layer_1 = LSTM(70, return_sequences=True)(merged)
            layer_2 = LSTM(70)(layer_1)
            output = Dense(classes[i], activation='softmax')(layer_2)
            outs.append(output)
        model = Model(inputs=input, outputs=outs)
        model.load_weights(self.path+self.weights)
        return model

    def load_clusters(self, path):
        clusters = {}
        with open(path, 'r') as f:
            lines = f.read().split('\n')
            for line in lines:
                parts = line.split(';')
                if len(parts) < 2:
                    continue
                words = parts[1].split(',')
                clusters[parts[0]] = words
        return clusters

    def reverse_clusters(self):
        reversed = {}
        for k, v in self.clusters.items():
            for word in v:
                reversed[word] = k
        return reversed

    def load_short_words(self):
        mapping = {}
        i = 0
        with open('D:\\Typing\\data\\shortWords.txt', 'r') as f:
            for word in f.read().split('\n'):
                mapping[word] = i
                i += 1
        return mapping

    def complete(self, text):
        text = text.lower().split(' ')
        if len(text) > 5:
            text = text[len(text) - 5:]
        if len(text) < 6:
            text.reverse()
            while len(text) < 6:
                text.append(' ')
            text.reverse()
        completions = self.predict_completions(text)
        return completions

    def predict_completions(self, text):
        beginning = (text[-1] != '')
        vector = self.vectorize(text[:-1])
        completions_hier = self.model_hier.predict(vector)
        completions = {}
        i = 0
        for comp in completions_hier:
            # rate = 1.0 - comp[0][-1]
            predictions = (-comp[0]).argsort()[:4]
            if beginning:
                predictions = (-comp[0]).argsort()
            # cluster = self.mapping_clusters[i]
            cluster = i
            counter = 0
            for prediction in predictions:
                if prediction == len(self.clusters[str(cluster)])-1:
                    continue
                word = self.clusters[str(cluster)][prediction]
                coeff = 1.0
                if word == 'car':
                    coeff = 0.5
                if beginning:
                    if word.startswith(text[-1]):
                        completions[word] = comp[0][prediction]*coeff
                        counter += 1
                        if counter == 3:
                            break
                else:
                    completions[word] = comp[0][prediction]*coeff
            i += 1
        # completions = OrderedDict(sorted(completions.items(), key=lambda t: t[1]))
        # return completions.keys()[::-1][:3]
        return self.normalize(completions)

    def normalize(self, completions):
        normalized = {}
        sum = np.sum(completions.values())
        for completion, value in completions.items():
            normalized[completion] = value/sum
        return normalized

    def vectorize(self, text):
        embeddings = []
        short_words = []
        clusters = []
        for word in text:
            try:
                embeddings.append(self.word_vectors[word])
            except KeyError:
                embeddings.append(np.zeros(300))
            if word in self.short_words:
                vector = np.zeros(len(self.short_words))
                vector[self.short_words[word]] = 1
                short_words.append(vector)
            else:
                short_words.append(np.zeros(len(self.short_words)))
            if word in self.reversed_clusters.keys():
                vector_clusters = np.zeros(len(self.clusters) + 1)
                cluster = int(self.reversed_clusters[word])
                vector_clusters[cluster] = 1
                clusters.append(vector_clusters)
            else:
                vector_clusters = np.zeros(len(self.clusters) + 1)
                vector_clusters[-1] = 1
                clusters.append(vector_clusters)
        embeddings = np.array(embeddings)
        short_words = np.array(short_words)
        vector = np.concatenate((embeddings, short_words, clusters), axis=1)
        v = [vector]
        v = np.array(v)
        return v
