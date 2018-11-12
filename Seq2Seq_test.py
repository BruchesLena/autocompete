#coding=utf-8
from keras import Input
from keras.engine import Model
from keras.layers import LSTM, Dense
from gensim.models import FastText
import numpy as np


def myModel():
    # Define an input sequence and process it.
    encoder_inputs = Input(shape=(None, 300))
    encoder = LSTM(50, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None, 300))
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder_lstm = LSTM(50, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                         initial_state=encoder_states)
    decoder_dense = Dense(300, activation='sigmoid')
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.load_weights('H:/data/weights')
    return model

word_vectors = FastText.load("D:\\Typing\\araneum_none_fasttextskipgram_300_5_2018.model")
model = myModel()

while True:
    text = raw_input('text: \n').decode('utf-8')
    words = text.split(' ')
    inputs = (word_vectors[words[0]], word_vectors[words[1]], word_vectors[words[2]])
    inp = [inputs]
    inp = np.array(inp)
    output = model.predict([inp, inp])
    continuations = word_vectors.similar_by_vector(output.flatten())
    for word in continuations:
        print word[0], word[1]
