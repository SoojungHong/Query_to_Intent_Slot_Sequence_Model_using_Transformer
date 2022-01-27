from __future__ import unicode_literals, print_function, division
from torch import optim
from io import open

import random
import os
import torch
import torch.nn.functional as F

from src.phrase_embedding.phrase_embedding import *


# -----------------------------------
#  Loading input & output sequence
# -----------------------------------
#queries = "C:/Users/shong/PycharmProjects/ngls_query/Intent-Slot-Tagging-Model/src/data/mvp_input_sequences.txt"
#output = "C:/Users/shong/PycharmProjects/ngls_query/Intent-Slot-Tagging-Model/src/data/mvp_output_sequences.txt"
queries = "C:/Users/shong/PycharmProjects/ngls_query/Intent-Slot-Tagging-Model/src/data/mvp_queries_with_spec_input_sequence.txt"
output = "C:/Users/shong/PycharmProjects/ngls_query/Intent-Slot-Tagging-Model/src/data/mvp_queries_with_spec_output_sequence.txt"


# debugging
#queries = "C:/Users/shong/PycharmProjects/ngls_query/Intent-Slot-Tagging-Model/src/data/few_queries_input.txt"
#output = "C:/Users/shong/PycharmProjects/ngls_query/Intent-Slot-Tagging-Model/src/data/few_queries_output.txt"


# -------------------
# global param
# -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1

p_embedding = PhraseEmbedding()


# -------------------------------
#  phrase to index dictionary
# -------------------------------
class PhraseSet:

    def __init__(self, name):
        self.name = name
        self.phrase2index = {}
        self.phrase2count = {}
        self.phrase2vector = {}
        self.vector2index = {}
        self.index2vector = {}
        self.index2phrase = {0: "SOS", 1: "EOS"}
        self.n_phrase = 2  # Count SOS and EOS

    def add_list_of_phrase(self, list_of_phrase):
        for phrase in list_of_phrase.split(','):
            self.addPhrase(phrase)


    def get_phrase_vector(self, phrase):
        phrase_vector = p_embedding.get_embedding(phrase)
        return phrase_vector


    def addPhrase(self, phrase):
        for p in phrase:
            if p not in self.phrase2index:
                self.phrase2index[p] = self.n_phrase
                self.phrase2count[p] = 1
                self.index2phrase[self.n_phrase] = p
                self.phrase2vector[p] = self.get_phrase_vector(p)
                self.vector2index[p] = self.n_phrase
                self.index2vector[self.n_phrase] = self.get_phrase_vector(p)
                self.n_phrase += 1
            else:
                self.phrase2count[p] += 1


def readPhrases():
    print("Reading lines...")

    input_lines = []
    output_lines = []
    num_lines = 0
    max_num_phrase = 0

    f = open(queries, "r")
    ff = open(output, "r")

    for line in f:
        line_phrase_list = line.strip().split(',')
        input_lines.append(line_phrase_list)
        num_lines = num_lines + 1

    for line in ff:
        line_phrase_list = line.strip().split(',')
        output_lines.append(line_phrase_list)
        if max_num_phrase < len(line_phrase_list):
            max_num_phrase = len(line_phrase_list)

    pairs_input = []
    pairs_output = []

    # make PhraseSet instances
    for i in range(num_lines):
        pairs_input.append(input_lines[i])
        pairs_output.append(output_lines[i])

    pairs = [pairs_input, pairs_output]
    input_set = PhraseSet('input')
    output_set = PhraseSet('output')

    return input_set, output_set, pairs, num_lines, max_num_phrase


# ------------------------------------
# summary of preparing the data
# ------------------------------------
# Read text file and split into lines, split lines into pairs
# Normalize text, filter by length and content
# Make word lists from sentences in pairs

def prepareData():
    input_set, output_set, pairs, num_lines, max_num_phrase = readPhrases()
    print("Read %s input output pairs" % len(pairs))
    print("Counting phrases...")

    for pair_input in pairs[0]:
        input_set.addPhrase(pair_input)

    for pair_output in pairs[1]:
        output_set.addPhrase(pair_output)

    print("Counted phrases:")
    print(input_set.name, input_set.n_phrase)
    print(output_set.name, output_set.n_phrase)
    #print(input_set.phrase2vector)
    #print(output_set.phrase2vector)

    return input_set, output_set, pairs, num_lines, max_num_phrase


input_set, output_set, pairs, num_lines, max_num_phrase = prepareData()
#print(random.choice(pairs))
print(pairs)

MAX_LENGTH = 22 # ToDo : How many should be max_length? --> maximum length of input sequence?


# -----------------------------------------------------------------------------------------
# encoder :
# With a seq2seq model the encoder creates a single vector
# single vector which, in the ideal case, encodes the “meaning” of the input sequence
# a single vector — a single point in some N dimensional space of sentences.
# ------------------------------------------------------------------------------------------

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)     # ToDo : Find better (?) embedding
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)



# -------------------------------------------------------------------------
# Decoder :
# The decoder is another RNN that takes the encoder output vector(s)
# and outputs a sequence of words to create the translation.
# -------------------------------------------------------------------------

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)



# ----------------------------------------------------------------------------------------------------
# Attention :
# Attention allows the decoder network to “focus” on a different part of the encoder’s outputs
# for every step of the decoder’s own outputs.
# ----------------------------------------------------------------------------------------------------

class ORG_AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(ORG_AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)



class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)




# -------------------------------------------------------------------------------------------------------
# preparing training data
# To train, for each pair we will need an input tensor (indexes of the words in the input sentence)
# and target tensor (indexes of the words in the target sentence).
# While creating these vectors we will append the EOS token to both sequences.
# -------------------------------------------------------------------------------------------------------

def ORG_my_indexesFromSentence(phrase_set, list_of_phrases):
    return [phrase_set.phrase2index[phrase] for phrase in list_of_phrases]


""" # Option 1 : Brute-Force Search using phrase 
def my_indexesFromSentence(phrase_set, list_of_phrases):
    index_list = []
    for phrase in list_of_phrases:
        if phrase in phrase_set.phrase2index:
            print('it is in')
            curr_idx = phrase_set.phrase2index[phrase]
            index_list.append(curr_idx)
        else:
            print('it is not in phrase_set.phrase2index')
            # convert all phrases to vectors and compare with current phrase and get the most similar phrase's index
            # ToDo : prepare the phrase vector to index, index to phrase vector
            #print('phrase : ', phrase)
            #print('test : ', p_embedding.get_embedding(phrase))
            unseen_phrase_vector = p_embedding.get_embedding(phrase)
            current_max_score = 0
            most_similar_phrase = ''
            for i in range(len(phrase_set.index2phrase)):
                curr_phrase = phrase_set.index2phrase[i]
                p_embedding.get_embedding(curr_phrase)
                #print(curr_phrase)
                score = p_embedding.compare_phrases(phrase, curr_phrase)
                #print('similarity score : ', score )
                if score > current_max_score:
                    current_max_score = score
                    most_similar_phrase = curr_phrase
            print('most similar phrase : ', most_similar_phrase)
            print('proposed index : ', phrase_set.phrase2index[most_similar_phrase])
            index_list.append(phrase_set.phrase2index[most_similar_phrase])
    return index_list
"""


def my_indexesFromSentence(phrase_set, list_of_phrases):
    index_list = []
    for phrase in list_of_phrases:
        if phrase in phrase_set.phrase2index:
            curr_idx = phrase_set.phrase2index[phrase]
            index_list.append(curr_idx)
        else:
            unseen_phrase_vector = p_embedding.get_embedding(phrase)
            current_max_score = 0
            most_similar_phrase = ''
            for i in range(len(phrase_set.index2phrase)):  # EOS or SOS will not be considered, index 0 and 1 are not considered
                if i < 2:
                    continue
                else:
                    cos_score = p_embedding.get_cosine_similarity(unseen_phrase_vector, phrase_set.phrase2vector[phrase_set.index2phrase[i]]) #compare_phrases(phrase, curr_phrase)
                    if cos_score > current_max_score:
                        current_max_score = cos_score
                        most_similar_phrase = phrase_set.index2phrase[i]
            print('most similar phrase : ', most_similar_phrase)
            print('proposed index : ', phrase_set.phrase2index[most_similar_phrase])
            index_list.append(phrase_set.phrase2index[most_similar_phrase])
    return index_list


def my_tensorFromSentence(phrase_set, list_of_phrases):
    indexes = my_indexesFromSentence(phrase_set, list_of_phrases)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def my_tensorsFromPair(pair_input, pair_output):
    input_tensor = my_tensorFromSentence(input_set, pair_input)
    target_tensor = my_tensorFromSentence(output_set, pair_output)
    return (input_tensor, target_tensor)


# --------------------------------------------------------------------------------------------------
# train model
#
# comment : “Teacher forcing” is the concept of using the real target outputs as each next input,
# instead of using the decoder’s guess as the next input.
# --------------------------------------------------------------------------------------------------

teacher_forcing_ratio = 0.5


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


# ------------------------------
# time elapse helper function
# ------------------------------

import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


# -----------------------------------------------
# training process :
#
#    Start a timer
#    Initialize optimizers and criterion
#    Create set of training pairs
#    Start empty losses array for plotting
# -----------------------------------------------

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


def my_trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    n_test_queries = 244     # ToDo : How many n test queries should be? & parameterize it
    # ToDo : use for test and training and get prediction score

    #training_pairs = [my_tensorsFromPair(pairs[0][i], pairs[1][i])  # FIXME : apply random choice?
    #                  for i in range(n_test_queries)]
    training_pairs = [my_tensorsFromPair(pairs[0][i], pairs[1][i])
                      for i in range(n_test_queries)]

    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))
            print("[!] saving model...")
            if not os.path.isdir(".save"):
                os.makedirs(".save")
            torch.save(encoder.state_dict(), './.save/encoder_%d.pt' % (iter))
            torch.save(decoder.state_dict(), './.save/decoder_%d.pt' % (iter))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0


    showPlot(plot_losses)



# --------------------------------------------------------------------------------------------------------------------
# evaluation
#
# Evaluation is mostly the same as training.
# However, there are no targets so we simply feed the decoder’s predictions back to
# itself for each step.
# Every time it predicts a word (label) we add it to the output string,
# and if it predicts the EOS token we stop there. We also store the decoder’s attention outputs for display later.
# --------------------------------------------------------------------------------------------------------------------

def ORG_my_evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = my_tensorFromSentence(input_set, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):    #ToDo : max_length --> max num of phrase
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)

            if topi.item() == EOS_token:    # ToDo : when topi.item() is EOS_TOKEN?
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_set.index2phrase[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


def my_evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = my_tensorFromSentence(input_set, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(input_length-1):    #ToDo : max_length --> max num of phrase
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)

            # FIXME by architectural change or generic constrain setting!
            if topi.item() == EOS_token:
                decoder_output.data[0][1] = -100
                topv, topi = decoder_output.data.topk(1)
                decoded_words.append(output_set.index2phrase[topi.item()])
            else:
                decoded_words.append(output_set.index2phrase[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


# -------------------------------
# evaluate random input phrase
# -------------------------------

def evaluateRandomly(encoder, decoder, n=1):    # FIXME : n
    for i in range(n):
        pair_x = pairs[0][0]
        pair_y = pairs[1][0]
        print('>', pair_x)
        print('=', pair_y)
        output_words, attentions = my_evaluate(encoder, decoder, pair_x)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


# ===============================
#  Training and Evaluating
# ===============================

hidden_size = 768 #512 #256 # ToDo : hidden_size matters? draw performance for each hidden_size

encoder1 = EncoderRNN(input_set.n_phrase, hidden_size).to(device)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_set.n_phrase, dropout_p=0.1).to(device)

n_iter = 244 # 30 # ToDo : How many iteration should be?
my_trainIters(encoder1, attn_decoder1, n_iter, print_every=5000)
evaluateRandomly(encoder1, attn_decoder1)


evaluate_input_sequence = ['where can i eat steak 1 hour by car around my hotel','steak','1 hour','1 hour by car around my hotel','around my hotel']
#['find apple store near a bistro','apple store','bistro','near a bistro'] # ['cheapest parking with free places near address','parking','free places','address','near address','with free places near address']

#['where can i have lunch inside a park','lunch','park','within a park']
print('evaluating input sequence : ', evaluate_input_sequence)
output_words, attentions = my_evaluate(encoder1, attn_decoder1, evaluate_input_sequence)
print('output words : ', output_words)
print('attentions : ', attentions)
plt.matshow(attentions.numpy())


# FIXME : Visualize
def showAttention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


# ----------------------------
#  Visualize Attention
# ----------------------------
def my_showAttention(input_sentence, output_words, attentions): # FIXME : Show attention
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    # ORG # ax.set_xticklabels([''] + input_sentence.split(' ') + ['<EOS>'], rotation=90)
    ax.set_xticklabels([''] + input_sentence + ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


def evaluateAndShowAttention(input_sentence):
    output_words, attentions = my_evaluate(encoder1, attn_decoder1, input_sentence)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    my_showAttention(input_sentence, output_words, attentions)

# ToDo : Show Attention Matrix visualization
#evaluateAndShowAttention(evaluate_input_sequence)

