from collections import OrderedDict

from keras.layers import LSTM, Dense
from keras.models import Sequential
# from pickle import load
from cPickle import load
from data_preparation_relation import DataPreparatorRelation
from spacy.en import English
import numpy as np
import time

class RelationCompletion:

    def __init__(self):
        self.data_preparator = DataPreparatorRelation(0, None, False)
        self.model = self.load_model()
        self.relations = self.load_relations()
        self.nlp = English()
        self.relation_types = ['adjs_nouns', 'verbs_adverbs', 'verbs_prepositions', 'verbs_objects',
                               'verbs_dir_objects', 'subjects_verbs', 'nouns_adjs']

    def load_model(self):
        model = Sequential()
        model.add(LSTM(100, input_shape=(5, 357)))
        model.add(Dense(52, activation='softmax'))
        model.load_weights('D:\\Typing\\data\\models_relations\\weights_rel.h5')
        return model

    def load_model_1(self):
        model = Sequential()
        model.add(LSTM(100, input_shape=(5, 357), return_sequences=True))
        model.add(LSTM(7, activation='sigmoid', return_sequences=True))
        model.load_weights('D:\\Typing\\phrases\\models\\weights_relations_2.h5')
        return model

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


    def predict_nn(self, text):
        words = text.split(' ')
        if text.count(' ') < 5:
            words = text.split(' ')
            inversed = words[::-1]
            while len(inversed) < 6:
                inversed.append('a')
            words = inversed[::-1]
        last_part = None
        if not text.endswith(' '):
            words = text.split(' ')
            last_part = words[-1]
        vectors = self.data_preparator.vectorize_with_spacy(words)[:-1]
        vectors = [vectors]
        vectors = np.array(vectors)
        predictions = self.model.predict(vectors)[0]
        word = 0
        # words = text.split(' ')
        completions = {}
        for prediction in predictions:
            for p in range(len(prediction)):
                if prediction[p] > 0.2:
                    relation = self.relations[self.relation_types[p]]
                    if words[word] in relation.keys():
                        self.add_to_list(completions, relation[words[word]])
            word += 1
        if last_part is None:
            normalized = self.normalize(completions)
            ordered = OrderedDict(sorted(normalized.items(), key=lambda t: t[1]))
            return ordered.keys()[::-1][:3]
            # return self.normalize(completions)
        with_part = {}
        completions = OrderedDict(sorted(completions.items(), key=lambda t: t[1]))
        for word in completions.keys()[::-1]:
            if word.startswith(last_part.encode('utf-8')):
                with_part[word] = completions[word]
                if len(with_part) == 3:
                    return self.normalize(with_part)
        return self.normalize(with_part)


    def predict(self, text):
        if text.count(' ') < 5:
            words = text.split(' ')
            inversed = words[::-1]
            while len(inversed) < 6:
                inversed.append('a')
            words = inversed[::-1]
            text = ' '.join(words)
        last_part = None
        if not text.endswith(' '):
            words = text.split(' ')
            last_part = words[-1]
            text = ' '.join(words[:-1])
        vectors = self.data_preparator.vectorize_with_spacy(text)[:-1]
        vectors = [vectors]
        vectors = np.array(vectors)
        predicted_relations = self.model.predict(vectors)
        predictions = (-predicted_relations[0]).argsort()[:3]
        words = self.handle_relation_result('dobj', self.nlp(text.decode('utf-8')), last_part, predictions)
        return words

    def handle_relation_result(self, relation, sequence, last_part, predictions):
        if relation == 'det':
            return ['a', 'the', 'some']
        completions = {}
        self.check_adj(completions, relation, sequence)
        self.check_verb(completions, relation, sequence)
        self.check_verb_prep(completions, relation, sequence)
        if 18 in predictions:
            self.check_dobj(completions, relation, sequence)
        self.check_attr(completions, relation, sequence)
        self.check_subject(completions, relation, sequence)
        # ordered_completions = OrderedDict(sorted(completions.items(), key=lambda t: t[1]))
        if last_part is None:
            # return ordered_completions.keys()[::-1][:3]
            # return completions
            return self.normalize(completions)
        with_part = {}
        for word in completions.keys()[::-1]:
            if word.startswith(last_part.encode('utf-8')):
                with_part[word] = completions[word]
                if len(with_part) == 3:
                    # return with_part
                    return self.normalize(with_part)
        # return with_part
        return self.normalize(with_part)

    def normalize(self, completions):
        normalized = {}
        sum = np.sum(completions.values())
        for completion, value in completions.items():
            normalized[completion] = value/sum
        return normalized


    def check_adj(self, completions, relation, sequence):
        token = list(sequence)[-1]
        if token.pos_ == 'ADJ' and token.lemma_.lower() in self.relations['adjs_nouns'].keys():
            self.add_to_list(completions, self.relations['adjs_nouns'][token.lemma_.lower()])

    def check_verb(self, completions, relation, sequence):
        last_token = list(sequence)[-1]
        if last_token.pos_ == 'VERB':
            if last_token.lemma_.lower() in self.relations['verbs_adverbs'].keys():
                self.add_to_list(completions, self.relations['verbs_adverbs'][last_token.lemma_.lower()])
            if last_token.lemma_.lower in self.relations['verbs_prepositions'].keys():
                self.add_to_list(completions, self.relations['verbs_prepositions'][last_token.lemma_.lower()])
        elif last_token.pos_ == 'NOUN' and last_token.head.pos_ == 'VERB':
            verb_token = last_token.head
            if verb_token.lemma_.lower() in self.relations['verbs_adverbs'].keys():
                self.add_to_list(completions, self.relations['verbs_adverbs'][verb_token.lemma_.lower()])
            if verb_token.lemma_.lower in self.relations['verbs_prepositions'].keys():
                self.add_to_list(completions, self.relations['verbs_prepositions'][verb_token.lemma_.lower()])


    def check_verb_prep(self, completions, relation, sequence):
        if list(sequence)[-1].pos_ == 'ADP' and list(sequence)[-2].pos_ == 'VERB':
            phrase = list(sequence)[-2].lemma_.lower()+'_'+list(sequence)[-1].lemma_.lower()
            if phrase in self.relations['verbs_objects'].keys():
                self.add_to_list(completions, self.relations['verbs_objects'][phrase])
        elif list(sequence)[-1].pos_ == 'ADP':
            token_prep = list(sequence)[-1]
            if token_prep.head.pos_ == 'VERB':
                phrase = token_prep.head.lemma_.lower()+'_'+token_prep.lemma_.lower()
                if phrase in self.relations['verbs_objects'].keys():
                    self.add_to_list(completions, self.relations['verbs_objects'][phrase])

    def check_dobj(self, completions, relation, sequence):
        if relation == 'dobj':
            if list(sequence)[-1].pos_ == 'NOUN' and list(sequence)[-1].head.pos_ == 'VERB':
                return
            verb_token = None
            for token in list(sequence)[::-1]:
                if token.pos_ == 'VERB':
                    verb_token = token
                    break
            if verb_token is not None and verb_token.lemma_.lower() in self.relations['verbs_dir_objects'].keys():
                self.add_to_list(completions, self.relations['verbs_dir_objects'][verb_token.lemma_.lower()])

    def check_subject(self, completions, relation, sequence):
        last_token = list(sequence)[-1]
        verb_token = None
        for token in list(sequence)[::-1]:
            if token.pos_ == 'VERB':
                verb_token = token
                break
        if last_token.pos_=='NOUN' and verb_token is None:
            if last_token.lemma_.lower() in self.relations['subjects_verbs'].keys():
                self.add_to_list(completions, self.relations['subjects_verbs'][last_token.lemma_.lower()])

    def check_attr(self, completions, relation, sequence):
        last_token = list(sequence)[-1]
        verb_id = None
        if last_token.lemma_.lower() == 'be':
            verb_id = last_token.i
        if verb_id is None:
            return
        subject_token = None
        for token in list(sequence):
            if token.pos_ == 'NOUN' and token.head.i == verb_id:
                subject_token = token
                break
        if subject_token is not None:
            if subject_token.lemma_.lower() in self.relations['nouns_adjs'].keys():
                self.add_to_list(completions, self.relations['nouns_adjs'][subject_token.lemma_.lower()])



    def add_to_list(self, completions, words_from_relations):
        for word in words_from_relations.keys()[::-1]:
            if word in completions:
                completions[word] += words_from_relations[word]
            else:
                completions[word] = words_from_relations[word]


# comleter = RelationCompletion()
# while True:
#     text = raw_input('text:\n')
#     print comleter.predict_nn(text)

