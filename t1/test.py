import numpy as  np
from keras import Input
from keras.engine import Model
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.optimizers import Adam, RMSprop, Adagrad
from keras.layers import LSTM, Convolution1D, LeakyReLU, MaxPooling1D, UpSampling1D, Merge, Conv1D, concatenate
from keras.utils.vis_utils import plot_model
from keras.models import load_model
import io
import time

# model = Sequential()
# model.add(LSTM(200, input_shape=(20, 54), return_sequences=True))
# model.add(LSTM(150))
# model.add(Dense(54, activation='softmax'))
#
# plot_model(model, "D:\\Typing\\data\\models\\model_9.png", True)





#
# model_json = model.to_json()
# with open("D:\Morph\StatCombinerChars\model2.json", "w") as f:
#     f.write(model_json)
# print "model is saved"
# model.load_weights("D:/Morph/StatCombinerChars/weightsOut22")
# print "weights are loaded"
# model.save_weights("D:\Morph\StatCombinerChars\weights2.h5")
# print "weights are saved"
#model.save_weights("D:\Morph\StatCombinerChars\weights2.h5")

t0 = time.time()

model = Sequential()
model.add(LSTM(100, input_shape=(5, 37)))
model.add(Dense(37, activation='softmax'))
model.load_weights('D:/Typing/data/models_grammar/weights_grammar_1')

t1 = time.time()

tt = t1 - t0
print 'time: ' + str(tt)

