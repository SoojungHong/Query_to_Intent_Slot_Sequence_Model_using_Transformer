from stanfordcorenlp import StanfordCoreNLP
from nltk.tree import *

import json
import nltk
import re



np_trees = []

# ----------------------------------------------------------------------------------------------------------------------
# Instruction to start StanfordCoreNLP
# Execute the command in terminal
# First, go to C:\Users\shong\Documents\Next_Generation_Location_Service\NLP_tools\stanford-corenlp-full-2018-10-05\
# > cd stanford-corenlp-full-2018-10-05
# > java -mx6g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -timeout 5000
# ----------------------------------------------------------------------------------------------------------------------

class StanfordNLP:

    def __init__(self, host='http://localhost', port=9000):
        self.nlp = StanfordCoreNLP(host, port=port, timeout=30000)
        self.props = {
            'annotators': 'tokenize,ssplit,pos,lemma,ner,parse,depparse,dcoref,relation',
            'pipelineLanguage': 'en',
            'outputFormat': 'json'
        }

    def word_tokenize(self, sentence):
        return self.nlp.word_tokenize(sentence)

    def pos(self, sentence):
        return self.nlp.pos_tag(sentence)

    def ner(self, sentence):
        return self.nlp.ner(sentence)

    def parse(self, sentence):
        return self.nlp.parse(sentence)

    def dependency_parse(self, sentence):
        return self.nlp.dependency_parse(sentence)

    def annotate(self, sentence):
        return json.loads(self.nlp.annotate(sentence, properties=self.props))

    @staticmethod
    def tokens_to_dict(_tokens):
        tokens = nltk.defaultdict(dict)
        for token in _tokens:
            tokens[int(token['index'])] = {
                'word': token['word'],
                'lemma': token['lemma'],
                'pos': token['pos'],
                'ner': token['ner']
            }
        return tokens

    def show_parse_tree(self, query):
        # show it with tree
        from nltk.parse.corenlp import CoreNLPParser
        cnlp = CoreNLPParser('http://localhost:9000')
        next(cnlp.raw_parse(query)).draw()



