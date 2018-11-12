# load doc into memory
from array import array
from pickle import dump

import gensim
import keras
from keras.engine import Model
from keras.layers import LSTM, Dense, Convolution1D, Flatten, MaxPooling1D, BatchNormalization, Input, Conv1D, \
    concatenate
from keras.models import Sequential
from keras.utils import to_categorical
import numpy as np

path = 'D:/Typing/data/'

# def load_doc(filename):
#     file = open(filename, 'r')
#     text = file.read()
#     file.close()
#     return text
#
#
# # load text
# raw_text = load_doc(path+'corpus.txt')
# raw_text = raw_text.lower()
# print(raw_text)
#
# # clean
# tokens = raw_text.split()
# raw_text = ' '.join(tokens)

# organize into sequences of characters
length = 20
# sequences = list()
# for i in range(length, len(raw_text)):
#     # select sequence of tokens
#     seq = raw_text[i-length:i+1]
#     # store
#     sequences.append(seq)
#     print('Total Sequences: %d' % len(sequences))

# save tokens to file, one dialog per line
def save_doc(lines, filename):
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()

# save sequences to file
# out_filename = path+'char_sequences.txt'
# save_doc(sequences[:500000], out_filename)

# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text

# load
in_filename = path+'char_sequences.txt'
raw_text = load_doc(in_filename)
lines = raw_text.split('\n')[:100000]

chars = sorted(list(set(raw_text)))
mapping = dict((c, i) for i, c in enumerate(chars))

model_word2vec = gensim.models.KeyedVectors.load_word2vec_format('D:/Questions/GoogleNews-vectors-negative300.bin', binary=True)
sequences = list()
embeddings = list()
for line in lines:
    # integer encode line
    encoded_seq = [mapping[char] for char in line]
    # store
    sequences.append(encoded_seq)

    for index, char in enumerate(line[:len(line)-1]):
        tokens = line[0:index].split(' ')
        try:
            vector = model_word2vec[tokens[-1]]
            embeddings.append(vector)
        except KeyError:
            embeddings.append(np.zeros(300, dtype=np.float32))

# vocabulary size
vocab_size = len(mapping)
print('Vocabulary Size: %d' % vocab_size)

sequences = np.array(sequences)
embeddings = np.array(embeddings).reshape((100000,20,300))
X, y = sequences[:,:-1], sequences[:,-1]



sequences = [to_categorical(x, num_classes=vocab_size) for x in X]
X = np.array(sequences)
inputs = np.dstack((X, embeddings))

y = to_categorical(y, num_classes=vocab_size)

# define model
model = Sequential()

model.add(Convolution1D(100, 10, input_shape=(inputs.shape[1], inputs.shape[2]), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Convolution1D(200, 7, activation='relu', padding='same'))
model.add(MaxPooling1D())
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(vocab_size, activation='softmax'))

print(model.summary())

# compile model
saver=keras.callbacks.ModelCheckpoint(path+'weights_1', monitor='val_loss', verbose=1, save_best_only=True,mode='auto')
stopper=keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['top_k_categorical_accuracy'], )
# fit model
model.fit(inputs, y, epochs=50, validation_split=0.2, callbacks=[saver, stopper])

# save the model to file
model.save(path+'modelLSTM.h5')

# save the mapping
dump(mapping, open(path+'mapping.pkl', 'wb'))