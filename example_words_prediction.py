import heapq

import gensim
import numpy as np
from keras.models import load_model
from pickle import load

from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

# model_word2vec = gensim.models.KeyedVectors.load_word2vec_format('D:/Questions/GoogleNews-vectors-negative300.bin', binary=True)

class Autocompleter:

    def __init__(self):
        self.path = 'D:/Typing/data/'
        # load the model
        self.model = load_model(self.path + 'models/model_10_2.h5')
        # load the mapping
        self.mapping = load(open(self.path + '54_mapping.pkl', 'rb'))

    def sample(self, preds, top_n=3):
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds)
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        return heapq.nlargest(top_n, range(len(preds)), preds.take)


    def predict_completion(self, text):
        original_text = text
        completion = ''
        while True:
            # x = prepare_multi_input(text, last_full_word)
            encoded = [self.mapping[char] for char in text]
            # truncate sequences to a fixed length
            encoded = pad_sequences([encoded], 20, truncating='pre')
            # one hot encode
            encoded = to_categorical(encoded, num_classes=len(self.mapping))
            encoded = encoded.reshape(1, encoded.shape[0], encoded.shape[1])

            # embeddings = list()
            # for index, char in enumerate(text[:len(text)]):
            #     tokens = text[0:index].split(' ')
            #     try:
            #         vector = model_word2vec[tokens[-1]]
            #         embeddings.append(vector)
            #     except KeyError:
            #         embeddings.append(np.zeros(300, dtype=np.float32))
            # embeddings = np.array(embeddings)
            #
            # embeddings = np.array(embeddings).reshape((1, 20, 300))
            # inputs = np.dstack((encoded, embeddings))

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
        # x = prepare_multi_input(text, last_full_word)
        encoded = [self.mapping[char] for char in text]
        # truncate sequences to a fixed length
        encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
        # one hot encode
        encoded = to_categorical(encoded, num_classes=len(self.mapping))
        encoded = encoded.reshape(1, encoded.shape[0], encoded.shape[1])
        # embeddings = list()
        # text = text.rjust(20, ' ')
        # for index, char in enumerate(text[:len(text)]):
        #     tokens = text[0:index].split(' ')
        #     try:
        #         vector = model_word2vec[tokens[-1]]
        #         embeddings.append(vector)
        #     except KeyError:
        #         embeddings.append(np.zeros(300, dtype=np.float32))
        # embeddings = np.array(embeddings)
        #
        # embeddings = np.array(embeddings).reshape((1, 20, 300))
        # inputs = np.dstack((encoded, embeddings))
        preds = self.model.predict(encoded, verbose=0)[0]
        next_indices = self.sample(preds, top_n=n)
        return [self.get_char(idx) + self.predict_completion(text[1:] + self.get_char(idx)) for idx in next_indices]

    def get_char(self, index):
        for char, i in self.mapping.items():
            if i == index:
                return char

    # generate a sequence of characters with a language model
    def generate_seq(self, model, mapping, seq_length, seed_text, n_chars):
        in_text = seed_text
        # generate a fixed number of characters
        for _ in range(n_chars):
            # encode the characters as integers
            encoded = [mapping[char] for char in in_text]
            # truncate sequences to a fixed length
            encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
            # one hot encode
            encoded = to_categorical(encoded, num_classes=len(mapping))
            encoded = encoded.reshape(1, encoded.shape[0], encoded.shape[1])
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

    def complete(self, text):
        text = text.lower()
        if len(text) > 20:
            text = text[len(text) - 20:]
        completions = self.predict_completions(text, 20)
        return completions

# autocompleter = Autocompleter()
# while True:
#     text = raw_input('text:'+'\n')
#     completions = autocompleter.complete(text)
#     for c in completions:
#         print text+c