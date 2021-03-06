from bert_serving.client import BertClient
from torch import nn
import torch.nn


# ------------------------------------------------------------------------------------------------------------------
#  Prerequisite : run following command
#
# pip install bert-serving-server  # server
# pip install bert-serving-client  # client, independent of `bert-serving-server`
#  Project_Dir>bert-serving-start -model_dir C:/Users/shong/Downloads/multi_cased_L-12_H-768_A-12/ -num_worker=4
# ------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# if you have error with tensorflow loading,
# 1. unstall tensorflow 2. install latest after download (pip install tensorflow-1.6.0-cp36-cp36m-win_amd64.whl)
# >pip uninstall tensorflow
# >pip install tensorflow-1.10.0-cp36-cp36m-win_amd64.whl (download from https://github.com/fo40225/tensorflow-windows-wheel/blob/master/1.6.0/py36/CPU/sse2/tensorflow-1.6.0-cp36-cp36m-win_amd64.whl)
# ----------------------------------------------------------------------------------------------------------------------


class PhraseEmbedding:

    def __init__(self):
        self.bc = BertClient()
        #print('phrase embedding...')


    def get_embedding(self, phrase):
        phrase_list = []
        phrase_list.append(phrase)
        #print('[debug] phrase_list : ', phrase_list)
        encoded_phrase = self.bc.encode(phrase_list)
        return encoded_phrase


    def get_cosine_similarity(self, vec1, vec2):
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        cos_sim = cos(torch.tensor(vec1), torch.tensor(vec2))
        return cos_sim


    def compare_phrases(self, phrase1, phrase2):
        phrase1_list = []
        phrase1_list.append(phrase1)

        phrase2_list = []
        phrase2_list.append(phrase2)

        phrase1_encode = self.bc.encode([phrase1])
        phrase2_encode = self.bc.encode([phrase2])

        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        output = cos(torch.tensor(phrase1_encode), torch.tensor(phrase2_encode))
        print('comparison score : ', output)
        return output


# ----------------
#  Test
# ----------------
pe = PhraseEmbedding()
#print(pe.get_embedding("restaurant nearby"))
#print(pe.compare_phrases("restaurant nearby", "pizza nearby"))