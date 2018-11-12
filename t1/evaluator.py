from example_words_prediction import Autocompleter
from combined_competer import CombinedAutocompleter

class Evaluator:

    def __init__(self, autocomleter):
        self.autocompleter = autocomleter
        self.corpus = self.load_corpus()

    def load_corpus(self):
        with open('D:\\Typing\\data\\test_set.txt', 'r') as f:
            return f.read().split('\n')

    def test_ksr(self):
        rk = 0
        for sentence in self.corpus:
            pass

    def test_precision(self):
        # correct = 0
        ends = 0
        for sentence in self.corpus:
            sentence = sentence.replace(',', '')
            print sentence
            words = sentence.lower().split(' ')
            for i in range(len(words)):
                word = words[i]
                word = word.replace('\\','')
                word = word.replace('.', '')
                word = word.replace(',', '')
                word = word.replace('?', '')
                word = word.replace('!', '')
                for c in range(len(word)):
                    previous_text = ' '.join(words[:i])+' '+word[:c]
                    previous_text.replace(',', '')
                    predictions = self.autocompleter.complete_combined(previous_text)
                    # predictions = self.autocompleter.complete(previous_text)
                    # completions = [word[:c]+p[:-1] for p in predictions]
                    # if word in completions:
                    #     correct += 1
                    if word in predictions:
                        # correct += 1
                        ends += (len(word)-c)
                        break
        corpus_size = 0
        for sentence in self.corpus:
            sentence = sentence.replace(' ','')
            corpus_size += len(sentence)
        # precision = correct*1.0 / corpus_size*1.0
        precision = (ends, corpus_size)
        return precision

# autocompleter = Autocompleter(None)
autocompleter = CombinedAutocompleter()
tester = Evaluator(autocompleter)
result = tester.test_precision()
print str(result[0]),'/',str(result[1])