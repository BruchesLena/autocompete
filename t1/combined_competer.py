import gensim
from keras.layers import LSTM, Dense, Bidirectional
from keras.models import load_model, Sequential

from example_words_prediction import Autocompleter, AutocomleterSingleHier
from data_preparation_grammar import DataPreparatorGrammar
from relation_autocomplete import RelationCompletion
from stat_autocomplete import StatAutocomplete
import numpy as np
from nltk import pos_tag
from collections import OrderedDict, Set
from starting_completion import StartingCompletion
from spacy.en import English
import time
from pickle import load


class CombinedAutocompleter:

    def __init__(self):
        self.word_vectors = gensim.models.KeyedVectors.load_word2vec_format(
            'D:/Questions/GoogleNews-vectors-negative300.bin', binary=True, limit=500000)
        print 'vectors loaded'
        self.starting_completer = StartingCompletion()
        # self.chars_completer = Autocompleter(self.word_vectors)
        # print 'chars loaded'
        self.words_completer = AutocomleterSingleHier('D:\\Typing\\data\\models_hier\\model_4\\', self.word_vectors, 'weights_hier_4_2', 'clusters_23.txt')
        print 'words loaded'
        self.word_add_completer = AutocomleterSingleHier('D:\\Typing\\data\\models_hier\\model_4\\', self.word_vectors, 'weights_hier_4_add', 'clusters_add.txt')
        print 'add loaded'
        self.model_stat = StatAutocomplete()
        # self.relation_completer = RelationCompletion()
        self.grammar = DataPreparatorGrammar(0, 'D:\\Typing\\data\\corpus.txt')
        self.model_grammar = self.load_grammar_model()
        print 'grammar loaded'
        self.functional = ['CC', 'DT', 'EX', 'IN', 'MD', 'PDT', 'POS', 'PRP', 'PRP$', 'RP', 'TO', 'WDT', 'WP', 'WP$', 'WRB']
        self.nlp = English()
        self.vocab = self.load_vocab()
        self.generalizing_stat = self.get_words()
        self.synonims = load(open('D:\\Typing\\data\\synonims.pkl', 'rb'))


    def get_words(self):
        words = []
        for word in self.word_vectors.index2word:
            if '_' in word:
                continue
            words.append(word.lower())
        return words


    def load_vocab(self):
        with open('D:\\Typing\\data\\mostCommon.txt', 'r') as f:
            return f.read().split('\n')

    def load_grammar_model(self):
        model = Sequential()
        # model.add(LSTM(100, input_shape=(5, 37)))
        # model.add(Dense(37, activation='softmax'))
        # model.load_weights('D:/Typing/data/models_grammar/weights_grammar_1')

        # model.add(LSTM(100, input_shape=(5, 35)))
        # model.add(Dense(35, activation='softmax'))
        # model.load_weights('D:/Typing/data/models_grammar/weights_grammar_spacy')

        model.add(Bidirectional(LSTM(100, return_sequences=True), input_shape=(5, 335)))
        model.add(Bidirectional(LSTM(100)))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(35, activation='softmax'))
        model.load_weights('D:/Typing/data/models_grammar/weights_grammar_spacy_2.h5')
        return model

    def complete(self, text):
        words_in_text = text.split(' ')
        chars_predictions = self.chars_completer.complete(text)
        predictions = []
        all_functional = True
        for word in chars_predictions:
            word = word.rstrip()
            if len(word) > 4:
                all_functional = False
            else:
                if not word.startswith(' ') and words_in_text[-1] != '':
                    predictions.append(words_in_text[-1]+word)
                else:
                    predictions.append(word)
        if all_functional:
            return predictions
        words_predictions = self.words_completer.complete(text)
        for word in words_predictions:
            predictions.append(word)
        if len(predictions) < 4:
            while len(predictions) < 4:
                predictions.append('')
        return predictions[:3]

    def complete_with_grammar(self, text):
        words_in_text = text.split(' ')
        # if words_in_text[0] == '' and len(words_in_text) == 2:
        if len(words_in_text) < 3:
            predictions = self.starting_completer.predict(text)
            while len(predictions) < 4:
                predictions.append('')
            return predictions[:3]
        if len(words_in_text) > 6:
            words_in_text = words_in_text[len(words_in_text) - 5:]
        if len(words_in_text) < 6:
            words_in_text.reverse()
            while len(words_in_text) < 6:
                words_in_text.append(' ')
            words_in_text.reverse()
        # vector = self.grammar.vectorize(words_in_text[:-1])
        t0 = time.time()
        vector = self.grammar.vectorize_with_spacy(words_in_text[:-1])
        vector = [vector]
        vector = np.array(vector)
        t1 = time.time()
        t = t1 - t0
        print 'vectorization for grammar: ' + str(t)
        t0 = time.time()
        pred = self.model_grammar.predict(vector)
        t1 = time.time()
        t = t1 - t0
        print 'grammar prediction: ' + str(t)
        print 'grammar inited'
        predicted_tags = {}
        for i in range(len(pred[0])):
            # if i == 36:
            if i == 34:
                continue
            # predicted_tags[self.grammar.mapping[i]] = pred[0][i]
            predicted_tags[self.grammar.mapping_spacy[i]] = pred[0][i]


        predictions = {}

        t0 = time.time()
        words_predictions = self.words_completer.complete(text)
        t1 = time.time()
        t = t1-t0
        print 'words prediction: ' + str(t)
        print 'words inited'
        t0 = time.time()
        words_predictions.update(self.word_add_completer.complete(text))
        # self.augment_predictions(words_predictions, words_in_text[-1])
        t1 = time.time()
        t = t1 - t0
        print 'additional words + updating dict: ' + str(t)
        print 'add inited'
        try:
            max_score = np.max(words_predictions.values())
        except ValueError:
            max_score = 1.0
        for word, score in words_predictions.items():
            # word_tag = pos_tag([word])[0][1]
            try:
                word_tag = self.nlp(word.decode('utf-8'))[-1].tag_
            except UnicodeEncodeError:
                print 'error in word ' + word
                continue
            if word_tag not in predicted_tags.keys():
                continue
            predictions[word] = score * predicted_tags[word_tag]
        t0 = time.time()
        chars_predictions = self.chars_completer.complete(text)
        t1 = time.time()
        t = t1 - t0
        print 'chats prediction: ' + str(t)
        print 'chars inited'
        t0 = time.time()
        for word in chars_predictions:
            word = word.replace('.', '')
            word = word.replace(',', '')
            word = word.replace('?', '')
            word = word.replace('!', '')
            word = word.rstrip()
            if not word.startswith(' ') and words_in_text[-1] != '':
                full_word = words_in_text[-1]+word
                # word_tag = pos_tag([full_word])[0][1]
                word_tag = self.nlp(full_word.decode('utf-8'))[0].tag_
                if word_tag in self.functional:
                    coeff = max_score
                else:
                    coeff = max_score*0.1
                try:
                    if full_word in predictions:
                        predictions[full_word] += coeff * predicted_tags[word_tag]
                    else:
                        predictions[full_word] = coeff * predicted_tags[word_tag]
                except KeyError:
                    continue
            else:
                # word_tag = pos_tag([word])[0][1]
                word_tag = self.nlp(word.decode('utf-8'))[0].tag_
                if word_tag in self.functional:
                    coeff = max_score
                else:
                    coeff = max_score*0.1
                try:
                    if word in predictions:
                        predictions[word] += coeff * predicted_tags[word_tag]
                    else:
                        predictions[word] = coeff * predicted_tags[word_tag]
                except KeyError:
                    continue

        predictions = OrderedDict(sorted(predictions.items(), key=lambda t: t[1]))
        result = []
        t1 = time.time()
        t = t1 - t0
        print 'scoring: ' + str(t)
        for w in predictions.keys()[::-1]:
            if w in self.vocab and not w.isdigit():
                result.append(w)
        if len(result) > 3:
            r = self.word_vectors.doesnt_match(result[:4])
            result.remove(r)
            return result[:3]
        else:
            return result
        # if len(predictions) < 4:
        #     i = 0
        #     while len(predictions) < 4:
        #         predictions[str(i)] = 0.0
        #         i += 1
        # return predictions.keys()[::-1][:3]

    def complete_combined(self, text):
        try:
            text.decode('utf-8')
        except UnicodeEncodeError:
            return
        words_in_text = text.split(' ')
        if len(words_in_text) < 3 :
            try:
                predictions = self.starting_completer.predict(text)
                if len(predictions) == 0:
                    return list(set(self.generalize_stat(text).keys()[:3]))
                return predictions[:3]
            except IndexError:
                return list(set(self.generalize_stat(text).keys()[:3]))
        if len(words_in_text) > 6:
            words_in_text = words_in_text[len(words_in_text) - 5:]
        if len(words_in_text) < 6:
            words_in_text.reverse()
            while len(words_in_text) < 6:
                words_in_text.append(' ')
            words_in_text.reverse()
        vector = self.grammar.vectorize_with_spacy(words_in_text[:-1])
        vector = [vector]
        vector = np.array(vector)
        t0 = time.time()
        pred = self.model_grammar.predict(vector)
        t1 = time.time()
        t = t1 - t0
        print 'grammar prediction: ' + str(t)
        predicted_tags = {}
        for i in range(len(pred[0])):
            if i == 34:
                continue
            predicted_tags[self.grammar.mapping_spacy[i]] = pred[0][i]

        predictions = {}

        t0 = time.time()
        words_predictions = self.words_completer.complete(text)
        t1 = time.time()
        t = t1-t0
        print 'words prediction: ' + str(t)
        t0 = time.time()
        self.update_dict(words_predictions, self.word_add_completer.complete(text), 1.0)
        t1 = time.time()
        t = t1 - t0
        print 'additional words + updating dict: ' + str(t)
        t0 = time.time()
        self.update_dict(words_predictions, self.model_stat.complete(text, self.nlp), 3.0)
        t1 = time.time()
        t = t1-t0
        print 'stat predictions: ' + str(t)

        # words_predictions = self.model_stat.complete(text, self.nlp)
        # if len(words_predictions.keys()) == 0 or self.is_functional(text):
        #     t0 = time.time()
        #     # words_predictions = self.words_completer.complete(text)
        #     self.update_dict(words_predictions, self.words_completer.complete(text), 1.0)
        #     t1 = time.time()
        #     t = t1-t0
        #     print 'words prediction: ' + str(t)
        #     t0 = time.time()
        #     self.update_dict(words_predictions, self.word_add_completer.complete(text), 1.0)
        #     t1 = time.time()
        #     t = t1 - t0
        #     print 'additional words + updating dict: ' + str(t)

        if len(words_predictions.keys()) == 0:
            t0 = time.time()
            self.update_dict(words_predictions, self.generalize_stat(words_in_text[-1]), 1.0)
            t1 = time.time()
            t = t1-t0
            print 'generalized stat prediction: ' + str(t)
        # t0 = time.time()
        # self.update_dict(words_predictions, self.relation_completer.predict(text))
        # t1 = time.time()
        # t = t1-t0
        print 'relation prediction: ' + str(t)
        t0 = time.time()
        # self.augment_predictions(words_predictions, words_in_text[-1])
        if len(words_predictions) > 100:
            words_predictions = OrderedDict(sorted(words_predictions.items(), key=lambda t: t[1], reverse=True))
        for word, score in words_predictions.items()[:100]:
            try:
                if text.endswith(' '):
                    full_text = text+word
                    word_tag = self.nlp(full_text.decode('utf-8'))[-1].tag_
                else:
                    last_part = text.split(' ')[-1]
                    full_text = text + word[len(last_part):]
                    word_tag = self.nlp(full_text.decode('utf-8'))[-1].tag_
            except UnicodeEncodeError:
                print 'error in word ' + word
                continue
            except IndexError:
                print 'Index error in word ' + word
                continue
            except UnicodeDecodeError:
                print 'UnicodeDecodeError in word ' + word
                continue
            if word_tag not in predicted_tags.keys():
                continue
            predictions[word] = score * predicted_tags[word_tag]
        predictions = OrderedDict(sorted(predictions.items(), key=lambda t: t[1]))
        t1 = time.time()
        t = t1 - t0
        print 'scoring: ' + str(t)
        if len(predictions.keys()) >= 3:
            return predictions.keys()[::-1][:3]
        else:
            predictions = predictions.keys()[::-1]
            gen_stat = self.generalize_stat(words_in_text[-1])
            g = 0
            while len(predictions) < 3 and g < len(gen_stat.keys()):
                predictions.append(gen_stat.keys()[g])
                g += 1
            return list(set(predictions))


    def update_dict(self, initial_dict, new_dict, coefficient):
        if len(initial_dict.keys()) == 0:
            initial_dict = new_dict
            return
        for key, value in new_dict.items():
            if key in initial_dict.keys():
                initial_dict[key] += value*coefficient
            else:
                initial_dict[key] = value*coefficient


    def complete_stat(self, text):
        words_in_text = text.split(' ')
        if len(words_in_text) < 3:
            try:
                predictions = self.model_stat.complete(text, True)
                if len(predictions) == 0:
                    return list(set(self.generalize_stat(text)[:3]))
                return predictions
            except IndexError:
                return list(set(self.generalize_stat(text)[:3]))
        if len(words_in_text) > 6:
            words_in_text = words_in_text[len(words_in_text) - 5:]
        if len(words_in_text) < 6:
            words_in_text.reverse()
            while len(words_in_text) < 6:
                words_in_text.append('a ')
            words_in_text.reverse()
        words_predictions = self.relation_completer.predict(text)
        if len(words_predictions) == 3:
            return words_predictions
        t0 = time.time()
        vector = self.grammar.vectorize_with_spacy(words_in_text[:-1])
        vector = [vector]
        vector = np.array(vector)
        t1 = time.time()
        t = t1 - t0
        print 'vectorization for grammar: ' + str(t)
        t0 = time.time()
        pred = self.model_grammar.predict(vector)
        t1 = time.time()
        t = t1 - t0
        print 'grammar prediction: ' + str(t)
        print 'grammar inited'
        predicted_tags = {}
        for i in range(len(pred[0])):
            if i == 34:
                continue
            predicted_tags[self.grammar.mapping_spacy[i]] = pred[0][i]

        predictions = {}

        stat_predictions = self.model_stat.complete(text)
        t0 = time.time()
        for word, score in stat_predictions.items():
            try:
                word_tag = self.nlp(word.decode('utf-8'))[-1].tag_
            except UnicodeEncodeError:
                print 'error in word ' + word
                continue
            if word_tag not in predicted_tags.keys():
                continue
            predictions[word] = score * predicted_tags[word_tag]
        predictions = OrderedDict(sorted(predictions.items(), key=lambda t: t[1]))
        t1 = time.time()
        t = t1 - t0
        words_predictions.extend(predictions.keys())
        print 'scoring: ' + str(t)
        if len(words_predictions) == 0:
            return list(set(self.generalize_stat(words_in_text[-1])[:3]))
        if len(words_predictions) > 3:
            return predictions.keys()[::-1][:3]
        else:
            predictions = words_predictions[::-1]
            gen_stat = self.generalize_stat(words_in_text[-1])
            g = 0
            while len(predictions) < 3:
                predictions.append(gen_stat[g])
                g += 1
            return list(set(predictions))

    def augment_predictions(self, word_predictions, predicting_word):
        augmented = {}
        for word in word_predictions.keys():
            try:
                synonims = self.synonims[word]
            except KeyError:
                continue
            for s in synonims:
                if s == word:
                    continue
                if predicting_word != '' and not s.startswith(predicting_word):
                    continue
                if s in augmented.keys():
                    augmented[s] += word_predictions[word]*0.5
                else:
                    augmented[s] = word_predictions[word]*0.5
        word_predictions.update(augmented)

    def generalize_stat(self, starting_text):
        if ' ' in starting_text:
            starting_text = starting_text.split(' ')[-1]
        d = OrderedDict()
        for i in self.generalizing_stat[1:]:
            if i.startswith(starting_text):
                d[i] = 1.0
        return d

    def is_functional(self, text):
        tagged = self.nlp(text.decode('utf-8'))
        if tagged[-1].tag_ in self.functional and tagged[-2].tag_ in self.functional:
            return True
        return False