from collections import OrderedDict
import os
from pickle import dump, load


class StartingCompletion:

    def __init__(self):
        if os.path.exists('D:\\Typing\\data\\first_words.pkl'):
            self.words = load(open('D:\\Typing\\data\\first_words.pkl', 'rb'))
        else:
            self.words = self.first_words()
            dump(self.words, open('D:\\Typing\\data\\first_words.pkl', 'wb'))
        pass

        if os.path.exists('D:\\Typing\\data\\second_words.pkl'):
            self.seconds = load(open('D:\\Typing\\data\\second_words.pkl', 'rb'))
        else:
            self.seconds = self.second_words()
            dump(self.seconds, open('D:\\Typing\\data\\second_words.pkl', 'wb'))

    def load_corpus(self):
        with open('D:\\Typing\\data\\corpus.txt', 'r') as f:
            return f.read().lower().split('\n')

    def first_words(self):
        corpus = self.load_corpus()
        words = {}
        for sentence in corpus:
            sentence = sentence.replace('.', ' ')
            sentence = sentence.replace(',', ' ')
            sentence = sentence.replace('?', ' ')
            sentence = sentence.replace('!', ' ')
            if ' ' in sentence:
                word = sentence[:sentence.index(' ')]
            else:
                word = sentence
            if word in words:
                words[word] += 1
            else:
                words[word] = 1
        words = OrderedDict(sorted(words.items(), key=lambda t: t[1]))
        return words.keys()[::-1]

    def second_words(self):
        corpus = self.load_corpus()
        second_words = {}
        num = 0
        for sentence in corpus:
            print str(num)
            num += 1
            sentence = sentence.replace('.', ' ')
            sentence = sentence.replace(',', ' ')
            sentence = sentence.replace('?', ' ')
            sentence = sentence.replace('!', ' ')
            tokens = sentence.split(' ')
            if len(tokens) < 2:
                continue
            if tokens[0]!='' and tokens[0] in self.first_words() and tokens[1] != '':
                if tokens[0] in second_words.keys():
                    seconds = second_words[tokens[0]]
                    if tokens[1] in seconds:
                        seconds[tokens[1]] += 1
                    else:
                        seconds[tokens[1]] = 1
                else:
                    seconds = {tokens[1]:1}
                    second_words[tokens[0]] = seconds
        ordered = {}
        for word, seconds in second_words.items():
            seconds = OrderedDict(sorted(seconds.items(), key=lambda t: t[1]))
            ordered[word] = seconds.keys()[::-1]
        return ordered

    def predict(self, text):
        words = text.split(' ')
        if len(words) == 1:
            if text.strip() == '':
                return self.words[:3]
            predictions = []
            for word in self.words:
                if word.startswith(text):
                    predictions.append(word)
            return predictions
        if words[-1] == '':
            try:
                return self.seconds[words[0]][:3]
            except KeyError:
                return []
        predictions = []
        try:
            for word in self.seconds[words[0]]:
                if word.startswith(words[-1]):
                    predictions.append(word)
        except KeyError:
            return predictions
        return predictions

# comleter = StartingCompletion()
# while True:
#     text = raw_input('text:\n')
#     print comleter.predict(text)

