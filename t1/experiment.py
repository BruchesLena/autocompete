from collections import Counter

import gensim
from cPickle import dump

from keras.models import load_model
import numpy as np

from data_preparation import DataPreparator
from data_preparation_words import DataPreparatorWords
from data_preparation_hier import DataPreparatorHier
from example_words_prediction import Autocompleter_words, Autocompleter_hier, AutocomleterSingleHier
from combined_competer import CombinedAutocompleter
from stat_autocomplete import StatAutocomplete
from data_preparation_grammar import DataPreparatorGrammar


# prep = DataPreparator(32, 'D:\\Typing\\data\\corpus.txt', 'D:\\Typing\\data\\')
# prep = DataPreparatorWords(32, 'D:\\Typing\\data\\from_wiki.txt', 'D:\\Typing\\data\\')
# prep.prepare('H:\\balanced\\learnSet')
# prep = DataPreparatorHier(32, 'D:\\Typing\\data\\models_hier\\model_6\\from_wiki.txt')
# # # prep = DataPreparatorHier(32, 'D:\\Typing\\wiki_texts\\wiki_2.txt')
# prep.prepare('H:\\data\\learnSet')
# prep.print_info()
# prep.statistics_on_clusters()

# prep = DataPreparatorGrammar(32, 'D:\\Typing\\data\\corpus.txt')
# prep.prepare('H:\\grammar\\learnSet')

# with open('D:\\Typing\\data\\shortWords.txt', 'w') as short_words:
#     with open('D:\\Typing\\data\\mostCommon.txt', 'r') as common_words:
#         for word in common_words.read().split('\n'):
#             if len(word) < 3:
#                 short_words.write(word+'\n')

# a = Autocompleter_words()
# print a.complete('i want to have dinner in the')
# print a.complete('i want to have breakfast in the')

# a = Autocompleter_hier('D:\\Typing\\data\\model_clusters_10\\')
# while True:
#     text = raw_input('text:\n')
#     print a.complete(text)

# a = AutocomleterSingleHier('D:\\Typing\\data\\models_hier\\model_6\\', None,'weights_1','clusters_30.txt')
# while True:
#     text = raw_input('text:\n')
#     print a.complete(text)


a = CombinedAutocompleter()
while True:

    text = raw_input('text:\n')
    print a.complete_combined(text)

# a = StatAutocomplete()
# while True:
#     text = raw_input('text:\n')
#     print a.complete(text)

