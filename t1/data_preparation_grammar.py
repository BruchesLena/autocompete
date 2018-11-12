from nltk import pos_tag
import numpy as np
from spacy.en import English

class DataPreparatorGrammar:

    def __init__(self, batch_size, path_to_dataset):
        self.batch_size = batch_size
        self.path_to_dataset = path_to_dataset
        self.raw_text = self.load_and_clean_text()
        self.mapping = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN',
                        'NNP', 'NNPS', 'NNS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP',
                        'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB']
        self.mapping_spacy = [u'BES', u'CC',u'DT',u'EX',u'HVS',u'IN', u'JJ', u'JJR',u'JJS', u'MD', u'NN', u'NNP', u'NNPS',
                              u'NNS', u'PDT', u'POS', u'PRP', u'PRP$',u'RB',u'RBR', u'RBS',u'RP',u'TO',u'UH',u'VB',u'VBD',
                              u'VBG',u'VBN', u'VBP', u'VBZ',u'WDT',u'WP',u'WP$',u'WRB']
        self.vector_size = len(self.mapping)
        self.length = 5
        self.nlp = English()

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
                outputs.append(encoded_seq[-1][:len(self.mapping_spacy)+1])
                if len(outputs) == self.batch_size:
                    with open(path+str(num), 'w') as f:
                        for i in range(len(outputs)):
                            f.write(self.vectors_to_string(inputs[i])+';'+self.vector_to_string(outputs[i])+'\n')
                    inputs = []
                    outputs = []
                    num += 1
                    print 'Printed', str(num)
                    if num == 20000:
                        print 'vocab_size=', str(len(self.mapping))
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
        tags = []
        vectors = []
        for token in pos_tagged_seq:
            tags.append(token.tag_)
            vectors.append(token.vector)
        if len(tags) < 5:
            inversed = tags[::-1]
            while len(inversed) < 5:
                inversed.append('a')
                vectors.append(np.zeros(300))
            tags = inversed[::-1]
        tagged = np.zeros((5, len(self.mapping_spacy)+301))
        # for token in tags[len(tags)-6:]:
        for i in range(len(tags[len(tags)-5:])):
            token = tags[len(tags)-5:][i]
            if token in self.mapping_spacy:
                tag = self.mapping_spacy.index(token)
                vector = np.zeros(len(self.mapping_spacy)+1)
                vector[tag] = 1
                tagged[i][:len(self.mapping_spacy)+1] = vector
            else:
                vector = np.zeros(len(self.mapping_spacy)+1)
                vector[-1] = 1
                tagged[i][:len(self.mapping_spacy)+1] = vector
            word_vector = vectors[len(vectors)-5:][i]
            tagged[i][len(self.mapping_spacy)+1:] = word_vector
        return tagged


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