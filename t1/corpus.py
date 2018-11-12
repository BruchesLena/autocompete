import os

class Wordform:
    def __init__(self, wordform_parts, id='_', form='_', lemma='_', upostag='_', xpostag='_', feats='_', head='_', deprel='_', seps='_', misc='_'):
        if wordform_parts is not None and len(wordform_parts) == 8:
            self.id = wordform_parts[0]
            self.form = wordform_parts[1]
            self.lemma = wordform_parts[2]
            self.upostag = wordform_parts[3]
            self.xpostag = wordform_parts[4]
            self.head = wordform_parts[5]
            self.deprel = wordform_parts[6]
        else:
            self.id = id
            self.form = form
            self.lemma = lemma
            self.upostag = upostag
            self.xpostag = xpostag
            self.feats = feats
            self.head = head
            self.deprel = deprel
            self.seps = seps
            self.misc = misc


class Sentence:
    def __init__(self):
        self.sentence = "" #string representation
        self.wordforms = [] #array of Wordforms

    def add_wordform(self, wordform):
        self.wordforms.append(wordform)

    def set_sentence_string(self, sentence):
        self.sentence = sentence


class Corpus:
    def __init__(self):
        self.sentences = [] #array of sentences

    def fill(self, path):
        with open(path, 'r') as f:
            lines = f.read().split("\n")
        sentence = Sentence()
        for line in lines:
            if line == "":
                if len(sentence.wordforms) > 0:
                    self.sentences.append(sentence)
                    sentence = Sentence()
                continue
            if line.startswith("s:"):
                text = line.split(": ")
                sentence.set_sentence_string(text[1])
                continue
            if line.startswith("# newdoc id") or line.startswith("# sent_id") or line.startswith("# Checktree"):
                continue
            wordform_parts = line.split("\t")
            wordform = Wordform(wordform_parts)
            sentence.add_wordform(wordform)