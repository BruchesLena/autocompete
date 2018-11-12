#coding=utf-8
import io
import os
from collections import Counter

from gensim.models import FastText
from nltk import WordPunctTokenizer
import xml.etree.ElementTree as ET
import numpy as np


class DataPreparator:
    def __init__(self, input_size, batch_size, path_to_write):
        self.word_vectors = FastText.load("D:\\Typing\\araneum_none_fasttextskipgram_300_5_2018.model")
        self.input_size = input_size
        self.tokenizer = WordPunctTokenizer()
        self.batch_size = batch_size
        self.path = path_to_write
        self.punctuations = ['.', ',', '-', '\'', '\"', '!', '?', '(', ')', ':', ';']

    def define_word_vector(self):
        num = 0
        dir = "D:\\Typing\\texts\\"
        prefix = "{http://www.gribuser.ru/xml/fictionbook/2.0}"
        files = os.listdir(dir)
        inputs = []
        outputs = []
        for file in files:
            tree = ET.parse(dir + file)
            root = tree.getroot()
            for child in root.iter(prefix + 'p'):
                text = child.text
                if text is None:
                    continue
                for line in text.split("."):
                    for char in line:
                        if char in self.punctuations:
                            line = line.replace(char, '')
                    words = self.tokenizer.tokenize(line)
                    for i in range(len(words)-5):
                        try:
                            input = (self.word_vectors[words[i]], self.word_vectors[words[i+1]], self.word_vectors[words[i+2]])
                            output = (self.word_vectors[words[i+3]])
                        except KeyError:
                            continue
                        inputs.append(input)
                        outputs.append(output)
                        if len(outputs) == self.batch_size:
                            with open(self.path + str(num), 'w') as f:
                                for k in range(len(outputs)):
                                    f.write(self.vectors_to_string(inputs[k])+':'+self.vectors_to_string(outputs[k])+'\n')
                                print str(num)
                                num += 1
                                inputs = []
                                outputs = []

    def define_freq_word(self, n=1000):
        num = 0
        self.freq_words = self.load_freq_words(n)
        dir = "D:\\Typing\\texts\\"
        prefix = "{http://www.gribuser.ru/xml/fictionbook/2.0}"
        files = os.listdir(dir)
        inputs = []
        outputs = []
        for file in files:
            tree = ET.parse(dir + file)
            root = tree.getroot()
            for child in root.iter(prefix + 'p'):
                text = child.text
                if text is None:
                    continue
                for line in text.split("."):
                    for char in line:
                        if char in self.punctuations:
                            line = line.replace(char, '')
                    words = self.tokenizer.tokenize(line)
                    for i in range(len(words) - 5):
                        if words[i+3] in self.freq_words.keys():
                            try:
                                input = (self.word_vectors[words[i]], self.word_vectors[words[i + 1]],
                                         self.word_vectors[words[i + 2]])
                            except KeyError:
                                continue
                            output = np.zeros(n)
                            output[self.freq_words[words[i+3]]] = 1
                            inputs.append(input)
                            outputs.append(output)
                            if len(outputs) == self.batch_size:
                                with open(self.path + str(num), 'w') as f:
                                    for k in range(len(outputs)):
                                        f.write(self.vectors_to_string(inputs[k]) + ':' + self.vectors_to_string(
                                            outputs[k]) + '\n')
                                    print str(num)
                                    num += 1
                                    inputs = []
                                    outputs = []
                                    if num == 85000:
                                        return

    def load_freq_words(self, n):
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

    def count_freq_words(self):
        dir = "D:\\Typing\\texts_1\\"
        prefix = "{http://www.gribuser.ru/xml/fictionbook/2.0}"
        files = os.listdir(dir)
        counter = Counter()
        n = 0
        for file in files:
            print str(n)
            n += 1
            tree = ET.parse(dir + file)
            root = tree.getroot()
            for child in root.iter(prefix + 'p'):
                text = child.text
                if text is None:
                    continue
                for line in text.split("."):
                    for char in line:
                        if char in self.punctuations:
                            line = line.replace(char, '')
                    words = self.tokenizer.tokenize(line)
                    for word in words:
                        counter[word.lower()] +=1
        with io.open('D:\\Typing\\freq_words.txt', 'w', encoding='utf-8') as f:
            for w in counter.most_common(len(counter)):
                f.write(w[0]+u'\n')


    def vectors_to_string(self, vectors):
        s = ''
        if isinstance(vectors, tuple):
            for vector in vectors:
                for element in vector:
                    s += str(element) + ','
        else:
            for element in vectors:
                s += str(element) + ','
        return s[:len(s) - 1]

data_prep = DataPreparator(3, 50, 'H:/data1/learnSet')
data_prep.define_freq_word()