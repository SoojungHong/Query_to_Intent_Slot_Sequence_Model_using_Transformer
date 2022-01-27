from src.input_transformation.EncoderRNN import *
from src.input_transformation.DecoderRNN import *
from src.input_transformation.AttnDecoderRNN import *
from src.input_transformation.prepare_data import *
from src.input_transformation.Lang import *
from src.input_transformation.train_and_evaluate import *


ENCODE_PATH = './.save/encoder_5000.pt'
DECODER_PATH = './.save/decoder_5000.pt'

# ----------------------
#  load trained model
# ----------------------

hidden_size = 256

input_lang, output_lang, pairs = prepareData('eng', 'fra', True)

encoder_model = EncoderRNN(input_lang.n_words, hidden_size).to(device)
att_decoder_model = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)

encoder_model.load_state_dict(torch.load(ENCODE_PATH))
att_decoder_model.load_state_dict(torch.load(DECODER_PATH))

test_sentence = 'merci'
print('test french sentence : ', test_sentence)
output_words, attentions = evaluate(encoder_model, att_decoder_model, test_sentence)
output_sentence = ' '.join(output_words)
print('<', output_sentence)
print('')
