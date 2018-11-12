#coding=utf-8
from keras import Input
from keras.engine import Model
from keras.layers import LSTM, Dense
from gensim.models import FastText
import numpy as np
import io

from keras.models import Sequential


def myModel():
    model = Sequential()
    model.add(LSTM(100, input_shape=(3,300)))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1000, activation='softmax'))
    model.load_weights('H:/data1/weights1')
    return model

word_vectors = FastText.load("D:\\Typing\\araneum_none_fasttextskipgram_300_5_2018.model")
model = myModel()

def load_freq_words(n):
    words = {}
    counter = 0
    with io.open('D:\\Typing\\freq_words.txt', 'r', encoding='utf-8') as f:
        w = f.read().split('\n')
        for word in w:
            if counter < n:
                words[word] = counter
                counter += 1
            else:
                return words

freq_words = load_freq_words(1000)

while True:
    text = raw_input('text: \n').decode('utf-8')
    words = text.split(' ')
    inputs = (word_vectors[words[0]], word_vectors[words[1]], word_vectors[words[2]])
    inp = [inputs]
    inp = np.array(inp)
    predictions = model.predict([inp])
    max_values = np.argpartition(-predictions, 5)[:5][0]
    # max_values = max_values[np.argsort(predictions[max_values])]
    for v in range(5):
        position = max_values[v]
        print freq_words.keys()[position]

        # (Z[np.argpartition(-Z, n)[:n]])
