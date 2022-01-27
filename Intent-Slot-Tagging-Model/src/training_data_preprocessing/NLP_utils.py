import unicodedata
import random
import re


class NLP_utils:

    def list_to_string(self, curr_list):
        listToStr = ' '.join([str(elem) for elem in curr_list])
        return listToStr

    def list_to_string_with_comma(self, curr_list):
        out = ','.join([str(c) for c in curr_list])
        return out


    # -------------------------------------------------
    # to simplify, turn Unicode characters to ASCII
    # -------------------------------------------------
    def unicodeToAscii(s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )


    # ------------------------------------------------------------
    # make everything lowercase, and trim most punctuation.
    # Lowercase, trim, and remove non-letter characters
    # -----------------------------------------------------------
    def normalizeString(s):
        s = NLP_utils.unicodeToAscii(s.lower().strip())
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        return s


    MAX_LENGTH = 10

    eng_prefixes = (
        "i am ", "i m ",
        "he is", "he s ",
        "she is", "she s ",
        "you are", "you re ",
        "we are", "we re ",
        "they are", "they re "
    )

    def filterPair(p):
        return len(p[0].split(' ')) < NLP_utils.MAX_LENGTH and \
               len(p[1].split(' ')) < NLP_utils.MAX_LENGTH and \
               p[1].startswith(NLP_utils.eng_prefixes)

    def filterPairs(pairs):
        return [pair for pair in pairs if NLP_utils.filterPair(pair)]

