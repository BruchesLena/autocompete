from corpus import Corpus
import os
from collections import OrderedDict
from pickle import dump, load

class Phraser:

    def __init__(self, relations, pos_to_remove, pos_to_add, path_to_corpus):
        self.relations = relations
        self.pos_to_remove = pos_to_remove
        self.pos_to_add = pos_to_add
        self.path_to_corpus = path_to_corpus
        self.data = {}

    def fill_data(self, num_of_docs):
        num = 0
        for f in os.listdir(self.path_to_corpus):
            if num == num_of_docs:
                return
            print str(num), str(len(self.data.keys()))
            num += 1
            corpus = Corpus()
            corpus.fill(self.path_to_corpus+f)
            for sentence in corpus.sentences:
                for wordform in sentence.wordforms:
                    wf_noun = wordform
                    if wf_noun.upostag != 'NOUN' or wf_noun.deprel != 'ROOT':
                        continue
                    wf_verb = sentence.wordforms[int(wf_noun.head)]
                    if wf_verb.upostag != 'VERB' or wf_verb.lemma.lower()=='be':
                        continue
                    main_word = wf_verb.lemma.lower()
                    context = wf_noun.lemma.lower()

                    # needed_pos = sentence.wordforms[int(wordform.head)].upostag == 'VERB' and wordform.upostag == 'ADP'
                    # # conditions = wordform.xpostag not in self.pos_to_remove and wordform.lemma.lower()!='not'
                    # if not needed_pos:
                    #     continue
                    # main_word = sentence.wordforms[int(wordform.head)].lemma.lower()
                    # context = wordform.lemma.lower()
                    if main_word in self.data.keys():
                        if context in self.data[main_word].keys():
                            self.data[main_word][context] += 1
                        else:
                            self.data[main_word][context] = 1
                    else:
                        w = {context : 1}
                        self.data[main_word] = w

    def order_data(self):
        for main_word, contexts in self.data.items():
            ordered = OrderedDict(sorted(contexts.items(), key=lambda t: t[1]))
            self.data[main_word] = ordered

    def dump(self, path_to_file):
        dump(self.data, open(path_to_file + '.pkl', 'wb'))

    def load_data(self, path_to_file):
        self.data = load(open(path_to_file + '.pkl', 'rb'))

    def write_in_txt(self, path_to_file):
        with open(path_to_file, 'w') as f:
            for word, contexts in self.data.items():
                f.write(word+":")
                contexts = OrderedDict(reversed(list(contexts.items())))
                for context, value in contexts.items():
                    f.write(context+'='+str(value)+',')
                f.write('\n')

    def invert_data(self):
        inverted_data = {}
        num = 0
        for word, contexts in self.data.items():
            print str(num)
            num +=1
            for context, value in contexts.items():
                if context in inverted_data.keys():
                    inverted_data[context][word] = value
                else:
                    w = {word : value}
                    inverted_data[context] = w
        self.data = inverted_data


    def count(self, path):
        files = os.listdir(path)
        for f in files:
            if f.endswith('.pkl'):
                print f
                file_name = f[:f.index('.')]
                self.load_data(path+file_name)
                relative = self.count_relative()
                self.data = relative
                self.order_data()
                self.dump(path+'rel_'+file_name)
                self.write_in_txt(path+'rel_'+file_name+'.txt')

    def count_relative(self):
        relative = {}
        for phrase, contexts in self.data.items():
            sum = 0
            for v in contexts.values():
                sum += v
            rr = {}
            for context, value in contexts.items():
                rr[context] = (value*1.0)/(sum*1.0)
            relative[phrase] = rr
        return relative


phraser = Phraser(['advcl'], ['WRB', 'EX'], ['ADV'], 'H:\\parsed_wiki\\')
phraser.count('D:\\Typing\\phrases\\')

# phraser.load_data('H:\\verbs_dir_objects')
# phraser.invert_data()
# phraser.order_data()
# phraser.dump('H:\\dir_objects_verbs')
# phraser.write_in_txt('H:\\dir_objects_verbs.txt')

# phraser.fill_data(200)
# print 'ordering...'
# phraser.order_data()
# phraser.dump('H:\\verbs_dir_objects')


