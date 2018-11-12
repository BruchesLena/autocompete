# -*- coding: utf-8 -*-

import os
from collections import OrderedDict
import io

from cPickle import dump, load


class StatAutocomplete:

    def __init__(self):
        self.texts_dir = '/Users/bruches/Documents/Typing/texts/'
        self.path_bigramms = '/Users/bruches/Documents/Typing/stat_bigramms.pkl'
        self.path_unigramms = '/Users/bruches/Documents/Typing/stat_unigramms.pkl'
        if os.path.exists(self.path_bigramms) and os.path.exists(self.path_unigramms):
            self.stat_bigramms = load(open(self.path_bigramms, 'rb'))
            self.stat_unigramms = load(open(self.path_unigramms, 'rb'))
        else:
            self.stat_bigramms, self.stat_unigramms = self.count_stat()
            dump(self.stat_bigramms, open(self.path_bigramms, 'wb'))
            dump(self.stat_unigramms, open(self.path_unigramms, 'wb'))


    def count_stat(self):
        # corpus = self.load_corpus()
        # words = corpus.split(' ')

        bigramms = {}
        unigramms = {}
        ind = 0
        for file in os.listdir(self.texts_dir):
            print('counting: ', str(ind))
            opened_file = io.open(self.texts_dir+file, 'r', encoding='utf-8')
            for sentence in opened_file.read().split('\n'):
                sentence = sentence.replace('.', ' ')
                sentence = sentence.replace(',', ' ')
                sentence = sentence.replace('?', ' ')
                sentence = sentence.replace('!', ' ')
                sentence = sentence.replace(':', ' ')
                sentence = sentence.replace(';', ' ')
                sentence = sentence.replace('-', ' ')
                tokens = sentence.split(" ")
                for i in range(1, len(tokens)):
                    target_word = tokens[i].lower()
                    if target_word == '' or target_word == '.' or target_word == ',' or target_word == ':' or target_word == ';' or target_word == '?' or target_word == '!' or target_word == "''":
                        continue
                    if (i-2)>=0:
                        bigramm = ' '.join(tokens[i-2:i]).lower()
                        self.fill_stat(bigramm, bigramms, target_word)
                    unigramm = tokens[i-1].lower()
                    self.fill_stat(unigramm, unigramms, target_word)
            if ind == 200:
                break
            ind += 1

        probs_bigramms = self.count(bigramms)
        probs_unigramms = self.count(unigramms)
        return probs_bigramms, probs_unigramms

    def fill_stat(self, ngramm, ngramms, target_word):
        if ngramms.has_key(ngramm):
            s = ngramms[ngramm]
            if s.has_key(target_word):
                s[target_word] += 1
            else:
                s[target_word] = 1
                ngramms[ngramm] = s
        else:
            s = {target_word:1}
            ngramms[ngramm] = s

    def count(self, ngramms):
        counter = {}
        for ngramm, words in ngramms.items():
            total = 0
            for c in words.values():
                total += c
            for context, c in words.items():
                prob = c*1.0 / total*1.0
                if counter.has_key(ngramm):
                    counter[ngramm][context] = prob
                else:
                    probs = {context:prob}
                    counter[ngramm] = probs
        return counter

    def load_corpus(self):
        file = open('D:\\Typing\\data\\corpus.txt', 'r')
        text = file.read()
        file.close()
        raw_text = text.lower()
        tokens = raw_text.split()
        raw_text = ' '.join(tokens)
        return raw_text

    def complete(self, text, nlp, sort=False):
        completions = {}
        words = text.split(' ')
        beginning = words[-1] != ''
        from_bigramms = self.search_completions(False, ' '.join(words[len(words)-3:-1]))
        score_bigramms = 1.0
        if len(from_bigramms) < 3:
            score_bigramms = 0.5
        for k, v in from_bigramms.items():
            if beginning:
                if k.startswith(words[-1]):
                    if k in completions:
                        completions[k] += v*score_bigramms
                    else:
                        completions[k] = v*score_bigramms
            else:
                if k in completions:
                    completions[k] += v*score_bigramms
                else:
                    completions[k] = v*score_bigramms
        if len(completions.keys()) > 2:
            completions = self.normalize(completions)
            if sort:
                ordered_completions = OrderedDict(sorted(completions.items(), key=lambda t: t[1]))
                return ordered_completions.keys()[::-1][:3]
        from_unigramms = self.search_completions(True, text.split(' ')[-2])
        score_unigramms = 1.0
        if len(from_unigramms) < 3:
            score_unigramms = 0.5
        for k, v in from_unigramms.items():
            if beginning:
                if k.startswith(words[-1]):
                    completions[k] = v*score_unigramms
            else:
                completions[k] = v*score_unigramms
        completions = self.normalize(completions)
        # for key, value in completions.items():
        #     tag = nlp(key)[-1].tag_
        #     if tag in ['CC', 'DT', 'EX', 'HVS', 'IN', 'MD', 'TO', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB']:
        #         completions[key] = value*5.0
        if sort:
            ordered_completions = OrderedDict(sorted(completions.items(), key=lambda t: t[1]))
            return ordered_completions.keys()[::-1][:3]
        return completions

    def normalize(self, completions):
        normalized = {}
        summed = sum(completions.values())
        for completion, value in completions.items():
            normalized[completion] = value/summed
        return normalized

    def search_completions(self, unigramms, text):
        completions = {}
        if unigramms:
            stat = self.stat_unigramms
        else:
            stat = self.stat_bigramms
        if stat.has_key(text):
            return stat[text]
        return completions


if __name__ == '__main__':
    autocomlete = StatAutocomplete()
    while True:
        text = raw_input('text:'+'\n').decode('utf-8')
        completions = autocomlete.complete(text, None, True)
        for c in completions:
            print(c)
