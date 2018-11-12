import time
from nltk import pos_tag
import numpy as np
from spacy.en import English
from pickle import load

class DataPreparatorRelation:

    def __init__(self, batch_size, path_to_dataset, need_relations):
        self.batch_size = batch_size
        self.path_to_dataset = path_to_dataset
        if path_to_dataset is not None:
            self.raw_text = self.load_and_clean_text()
        self.mapping_spacy = [u'acomp', u'advcl', u'advmod', u'agent', u'amod', u'appos', u'attr', u'aux', u'auxpass',
                              u'cc', u'ccomp', u'complm', u'conj', u'cop', u'csubj', u'csubjpass', u'dep',
                              u'det', u'dobj', u'expl', u'hmod', u'hyph', u'infmod', u'intj', u'iobj', u'mark',
                              u'meta', u'neg', u'nmod', u'nn', u'npadvmod', u'nsubj', u'nsubjpass', u'num',
                              u'number', u'oprd', u'obj', u'obl', u'parataxis', u'partmod', u'pcomp', u'pobj',
                              u'poss', u'possessive', u'preconj', u'prep', u'prt', u'punct', u'quantmod',
                              u'rcmod', u'ROOT', u'xcomp']
        self.mapping_spacy_pos_tags = [u'BES', u'CC', u'DT', u'EX', u'HVS', u'IN', u'JJ', u'JJR', u'JJS', u'MD', u'NN', u'NNP',
                              u'NNPS',
                              u'NNS', u'PDT', u'POS', u'PRP', u'PRP$', u'RB', u'RBR', u'RBS', u'RP', u'TO', u'UH',
                              u'VB', u'VBD',
                              u'VBG', u'VBN', u'VBP', u'VBZ', u'WDT', u'WP', u'WP$', u'WRB']
        self.length = 5
        self.nlp = English()
        if need_relations:
            self.relations = self.load_relations()
        self.relation_types = ['adjs_nouns', 'verbs_adverbs', 'verbs_prepositions', 'verbs_objects', 'verbs_dir_objects', 'subjects_verbs', 'nouns_adjs']

    def load_and_clean_text(self):
        file = open(self.path_to_dataset, 'r')
        text = file.read()
        file.close()
        raw_text = text.lower()
        raw_text = raw_text.replace('?', '.')
        raw_text = raw_text.replace('!', '.')
        raw_text = raw_text.replace(',', '')
        sentences = raw_text.split('.')
        return sentences

    def load_relations(self):
        path = 'D:\\Typing\\phrases\\'
        # files = ['adjs_nouns', 'adverbs_verbs', 'nouns_adjs', 'subjects_verbs', 'verbs_dir_objects', 'verbs_objects', 'verbs_prepositions', 'verbs_adverbs']
        files = ['adjs_nouns', 'verbs_adverbs', 'verbs_prepositions', 'verbs_objects', 'verbs_dir_objects', 'subjects_verbs', 'nouns_adjs']
        relations = {}
        for f in files:
            t0 = time.time()
            rel = load(open(path+'rel_'+f+'.pkl', 'rb'))
            t1 = time.time()
            tt = t1-t0
            print 'loaded '+f+' in '+str(tt)
            relations[f] = rel
        return relations

    def prepare(self, path):
        num = 0
        inputs = []
        outputs = []
        for sentence in self.raw_text:
            tokens = sentence.split()
            for i in range(self.length, len(tokens)):
                seq = tokens[i-self.length:i+1]
                encoded_seq = self.vectorize_with_spacy(seq)
                inputs.append(encoded_seq[:-1])
                outputs.append(self.vectorize_output(seq))
                if len(outputs) == self.batch_size:
                    with open(path+str(num), 'w') as f:
                        for i in range(len(outputs)):
                            f.write(self.vectors_to_string(inputs[i])+';'+self.vectors_to_string(outputs[i])+'\n')
                    inputs = []
                    outputs = []
                    num += 1
                    print 'Printed', str(num)
                    if num == 20000:
                        return

    def vectorize(self, sequence):
        try:
            pos_tagged_seq = pos_tag(sequence)
        except IndexError:
            pos_tagged_seq = []
            while True:
                pos_tagged_seq.append(('', ''))
                if len(pos_tagged_seq) == len(sequence):
                    break
        tagged = []
        for token in pos_tagged_seq:
            if token[1] in self.mapping:
                tag = self.mapping.index(token[1])
                vector = np.zeros(len(self.mapping)+1)
                vector[tag] = 1
                tagged.append(vector)
            else:
                vector = np.zeros(len(self.mapping)+1)
                vector[-1] = 1
                tagged.append(vector)
        return tagged

    def vectorize_with_spacy(self, sequence):
        seq = ' '.join(sequence)
        try:
            pos_tagged_seq = self.nlp(seq.decode('utf-8'))
        except IndexError:
            pos_tagged_seq = []
            while True:
                pos_tagged_seq.append(('', ''))
                if len(pos_tagged_seq) == len(sequence):
                    break
        heads = []
        relations = []
        embeddings = []
        if len(pos_tagged_seq) >6:
            pos_tagged_seq = pos_tagged_seq[:6]
        for token in pos_tagged_seq:
            relations.append(token.dep_)
            heads.append(token.head.i)
            embeddings.append(token.vector)
        if len(heads) < 5:
            inversed = heads[::-1]
            inversed_rel = relations[::-1]
            while len(inversed) < 6:
                inversed.append('a')
                inversed_rel.append('a')
            heads = inversed[::-1]
            relations = inversed_rel[::-1]
        tagged = np.zeros((6, len(self.mapping_spacy)+305))
        for i in range(len(heads)):
            head = heads[i]
            relation = relations[i]
            if relation in self.mapping_spacy:
                tag = self.mapping_spacy.index(relation)
                vector = np.zeros(len(self.mapping_spacy)+305)
                vector[5+tag] = 1
                if head < 5:
                    vector[head] = 1
                vector[len(self.mapping_spacy)+5:] = embeddings[i]
                tagged[i] = vector
            else:
                vector = np.zeros(len(self.mapping_spacy)+305)
                if head < 5:
                    vector[head] = 1
                vector[len(self.mapping_spacy) + 5:] = embeddings[i]
                tagged[i] = vector
        return tagged

    def vectorize_output(self, sequence):
        seq = ' '.join(sequence)
        parsed_seq = self.nlp(seq.decode('utf-8'))
        output_token = list(parsed_seq)[5]
        output_vector = np.zeros((5, len(self.relations)))
        num = 0
        for token in list(parsed_seq)[:5]:
            for relation, contexts in self.relations.items():
                if token.lemma_.lower() in contexts.keys() and output_token.lemma_.lower() in contexts[token.lemma_.lower()]:
                    output_vector[num, self.relation_types.index(relation)] = 1.0
            num +=1
        return output_vector


    def normalize(self, vector):
        norm = np.zeros(vector.shape)
        for v in range(len(vector)):
            sum = np.sum(vector[v])
            if sum == 0.0:
                continue
            for element in range(len(vector[v])):
                norm[v][element] = vector[v][element]*1.0 / sum*1.0
        return norm



    def vector_to_string(self, vector):
        s = ''
        for v in vector:
            s += str(v)+','
        return s[:-1]

    def vectors_to_string(self, vectors):
        s = ''
        for vector in vectors:
            for v in vector:
                s += str(v)+','
        return s[:-1]

# prep = DataPreparatorRelation(32, 'D:\\Typing\\data\\corpus.txt')
# prep.prepare('H:\\data\\learnSet')