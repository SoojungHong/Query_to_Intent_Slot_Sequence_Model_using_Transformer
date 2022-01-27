from src.training_data_preprocessing.NLP_utils import NLP_utils
from src.training_data_preprocessing.standardNLP import StanfordNLP
from nltk.tree import *
import unicodedata
import re


utils = NLP_utils()
sNLP = StanfordNLP()

list_of_phrase = []


# --------------------------------------------------------------------------
# traverse search the syntactic parse tree with depth-first search manner
# store the extracted phrase into phrase_list
# --------------------------------------------------------------------------
def traverse(t, phrase_list):

    try:
        t.label()

    except AttributeError:
        return

    if t.label() == "ROOT":
        phrase_list.append(utils.list_to_string(t.leaves()))

    elif t.label() == "NP":

        if t.__len__() == 1:
            child = t[0]
            if child.label() == "NN" or child.label() == "NNS" or child.label() == "NNP" or child.label() == "NNPS":
                phrase_list.append(utils.list_to_string(child.leaves()))
            elif child.label() == "VBG":  # include present particle
                phrase_list.append(utils.list_to_string(child.leaves()))

        else:  # when it's not only child
            Adj_tokens = []
            NP_tokens = []
            for child in t:
                if child.label() == "NP" or child.label() == "VP" or child.label() == "PP" or child.label() == "ADJP" or child.label() == "ADVP" :
                    traverse(child, phrase_list)

                else:  # it is word level
                    if child.label() == "NN" or child.label() == "NNS" or child.label() == "NNP" or child.label() == "NNPS" or child.label() == "JJ" or child.label() == "CD" or child.label() == "CC":
                        NP_tokens.append(utils.list_to_string(child.leaves()))

                    elif child.label() == "JJS":  # adjective superlative
                        Adj_tokens.append(utils.list_to_string(child.leaves()))


            if Adj_tokens.__len__() != 0:
                phrase_list.append(utils.list_to_string(Adj_tokens))

            if NP_tokens.__len__() != 0:
                phrase_list.append(utils.list_to_string(NP_tokens))

        return  #FIXME : CHECK


    elif t.label() == "VP":
        if t.__len__() == 1:
            if t.label() == "VBG":
                phrase_list.append(utils.list_to_string(t.leaves()))
        else:
            for child in t:
                if child.label() == "VBG":
                    phrase_list.append(utils.list_to_string(child.leaves()))

    elif t.label() == "PP":
        # save IN such as at, in
        if t.__len__() == 1:
            child = t[0]
            if child.label() == "IN":
                phrase_list.append(utils.list_to_string(child.leaves()))
        else:
            for child in t:
                if child.label() == "NP" or child.label() == "VP" or child.label() == "PP" or child.label() == "ADJP" or child.label() == "ADVP":
                    traverse(child, phrase_list)
                else:  # it is word level

                    """
                    # -------------------------------------------------
                    # OPTION 1 : split only token such as in, at, by
                    # -------------------------------------------------

                    Preposition_tokens = []
                    NP_tokens = []
                    for child in t:
                        if child.label() == "IN":
                            Preposition_tokens.append(list_to_string(child.leaves()))
                        if child.label() == "NP" or child.label() == "VP" or child.label() == "PP" or child.label() == "ADJP" or child.label() == "ADVP":
                            traverse(child)

                    if Preposition_tokens.__len__() != 0:
                        print('Prepositional Phrase : ', Preposition_tokens)
                        phrase_list.append(list_to_string(Preposition_tokens))

                    break
                    """

                    """
                    # --------------------------
                    # OPTION 2 : split phrase
                    # --------------------------
                    print("Phrase : ", t.leaves())
                    phrase_list.append(list_to_string(t.leaves()))
                    break
                    """

                    #----------------------------------------------------------------------
                    # OPTION 3 : take all phrase include prepostion and NP phrase inside
                    #----------------------------------------------------------------------
                    Preposition_tokens = []
                    NP_tokens = []
                    if child.label() == "IN":
                        Preposition_tokens.append(utils.list_to_string(t.leaves()))  # add whole prepositional phrase
                    if child.label() == "NP" or child.label() == "VP" or child.label() == "PP" or child.label() == "ADJP" or child.label() == "ADVP":
                        traverse(child, phrase_list)

                    if Preposition_tokens.__len__() != 0:
                        phrase_list.append(utils.list_to_string(Preposition_tokens))

        return

    elif t.label() == "ADJP":  # Adjective Phrase
        if t.__len__() == 1:
            child = t[0]
            if child.label() == "JJ" or child.label() == "JJR" or child.label() == "JJS":
                phrase_list.append(utils.list_to_string(child.leaves()))
        else:
            for child in t:
                if child.label() == "NP" or child.label() == "VP" or child.label() == "PP" or child.label() == "ADJP" or child.label() == "ADVP":
                    traverse(child, phrase_list)
                else:  # it is word level
                    Adj_tokens = []

                    for child in t:
                        if child.label() == "JJ" or child.label() == "NN" or child.label() == "NNS" or child.label() == "NNP":
                            Adj_tokens.append(utils.list_to_string(child.leaves()))  # add whole prepositional phrase
                        if child.label() == "NP" or child.label() == "VP" or child.label() == "PP" or child.label() == "ADJP" or child.label() == "ADVP":
                            traverse(child, phrase_list)

                    if Adj_tokens.__len__() != 0:
                        phrase_list.append(utils.list_to_string(Adj_tokens))

                    break
        return

    elif t.label() == "ADVP":  # Adverb Phrase
        if t.__len__() == 1:
            child = t[0]
            if child.label() == "RB" or child.label() == "RBS":
                phrase_list.append(utils.list_to_string(child.leaves()))

    elif t.label() == "WHNP":  # WH-noun phrase
        Adj_tokens = []

        if t.__len__() > 1:
            for child in t:
                if child.label() == "NP" or child.label() == "VP" or child.label() == "PP" or child.label() == "ADVP":
                    traverse(child, phrase_list)
                elif child.label() == "JJ" or child.label() == "NN" or child.label() == "NNS" or child.label() == "NNP" or child.label() == "ADJP":
                    Adj_tokens.append(utils.list_to_string(child.leaves()))  # add whole prepositional phrase

            if Adj_tokens.__len__() != 0:
                phrase_list.append(utils.list_to_string(Adj_tokens))

            return

    for child in t:
        traverse(child, phrase_list)


def show_parse_tree(text):
    from nltk.parse.corenlp import CoreNLPParser
    stanford = CoreNLPParser('http://localhost:9000')
    next(stanford.raw_parse(text)).draw()


def debug(text):
    debug_phrase_list = []
    print('raw query : ', text)
    parsed_text = sNLP.parse(text)
    print('parsed : \n', parsed_text)
    show_parse_tree(text)
    normalized_parsed_query = re.sub('[\r\n]', '', parsed_text)
    print('parsed query without newline : ', normalized_parsed_query)
    print('')
    tree = ParentedTree.fromstring(normalized_parsed_query)
    traverse(tree, debug_phrase_list)
    print('Sequence of Phrase : ', debug_phrase_list)


def create_input_sequences():
    #queries = "C:/Users/shong/PycharmProjects/ngls_query/Intent-Slot-Tagging-Model/src/data/mvp_queries.txt"
    #queries = "C:/Users/shong/PycharmProjects/ngls_query/Intent-Slot-Tagging-Model/src/data/test_queries.txt"
    queries = "C:/Users/shong/PycharmProjects/ngls_query/Intent-Slot-Tagging-Model/src/data/alexa_queries.txt"

    f = open(queries, "r")
    splitted_phrases = "C:/Users/shong/PycharmProjects/ngls_query/Intent-Slot-Tagging-Model/src/data/alexa_queries_input_sequence.csv"
    splitted_input_file = open(splitted_phrases, "w")

    for line in f:
        print('query : ', line)
        line = normalizeString(unicodeToAscii(line))
        parsed_text = sNLP.parse(line)
        normalized_parsed_query = re.sub('[\r\n]', '', parsed_text)
        list_of_phrase = []
        tree = ParentedTree.fromstring(normalized_parsed_query)
        traverse(tree, list_of_phrase)
        print('Sequence of Phrase : ', utils.list_to_string_with_comma(list_of_phrase))
        print('\n')
        list_of_phrase_with_comma = utils.list_to_string_with_comma(list_of_phrase)
        splitted_input_file.write(list_of_phrase_with_comma)
        splitted_input_file.write('\n')

    f.close()


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z0-9.!?]+", r" ", s)
    return s


# --------------------
#  debug a query
# --------------------
text = "i am cycling what is around me"
# "public parks with a lake or river inside"
# "most important historical sights in London"
debug(normalizeString(unicodeToAscii(text)))


# ---------------------------------------------------------
# run whole queries to generate input phrase sequences
# ---------------------------------------------------------
#create_input_sequences()