from collections import OrderedDict
from pyexpat import model

import numpy
import os.path
import csv
import seq2seq
from seq2seq.models import AttentionSeq2Seq, Seq2Seq, SimpleSeq2Seq

from IPython.lib.editorhooks import emacs
from bokeh.models.widgets import inputs
from boto.dynamodb.batch import Batch
from docutils.utils.math.latex2mathml import mo
from scipy.linalg._flapack import sgegv
from keras.utils import to_categorical

class DataChunk:
    def __init__(self, path,pathData):
        self.path = path
        v=path.index("/");
        self.pathDaa=pathData+path[v+1:len(path)]

    def getInput(self):
        return self.path+"input.npy"

    def getOutput(self, name):
        return self.path + "output_"+name+".npy"

    def aquireData(self):
        inputPath=self.getInput()
        output_paths = []
        for i in range(0, 19):
            output_paths.append(self.getOutput(str(i)))
        all_paths_exists = True
        for path in output_paths:
            if not os.path.exists(path):
                all_paths_exists = False
        if all_paths_exists:
            all_arrays = []
            for path in output_paths:
                array = numpy.load(path)
                all_arrays.append(array)
            input_array = numpy.load(inputPath)
            return input_array, all_arrays
        if os.path.exists(self.path):
            self.prepareData()
            all_paths_exists = True
            for path in output_paths:
                if not os.path.exists(path):
                    all_paths_exists = False
            if all_paths_exists:
                return self.aquireData()
            raise IOError("Data preparation failed")
        raise IOError("Chunk file does not exists")

    def prepareData(self):
        print "Preparing"
        inputs, outputs=self.internalPrepare();
        inputs = numpy.array(inputs)
        numpy.save(self.getInput(), inputs)
        for i, item in enumerate(outputs.values()):
            array = numpy.array(item)
            numpy.save(self.getOutput(str(i)), array)

    def internalPrepare(self):
        f = file(self.path, "r")
        inputs = []
        outputs = OrderedDict()
        for l in f.readlines():
            parts = l.split(';')
            if len(parts) < 2:
                continue
            embedding = numpy.fromstring(parts[0], dtype=numpy.float32, sep=',')
            embedding = embedding.reshape(5, 300)
            short_words = numpy.fromstring(parts[1], dtype=numpy.float32, sep=',')
            short_words = short_words.reshape(5, 31)
            clusters = numpy.fromstring(parts[2], dtype=numpy.float32, sep=',')
            clusters = clusters.reshape(5,20)
            input1 = numpy.concatenate((embedding, short_words, clusters), axis=1)
            inputs.append(input1)
            for i in range(3, len(parts)):
                if i == len(parts)-1:
                    output_array = numpy.fromstring(parts[i], dtype=numpy.float32, sep=',')[:-1]
                else:
                    output_array = numpy.fromstring(parts[i], dtype=numpy.float32, sep=',')
                if i in outputs.keys():
                    outputs[i].append(output_array)
                else:
                    array = [output_array]
                    outputs[i] = array
        return inputs, outputs

def chunker(path,p1,pref):
    res=[]
    for v in range(0,35000):
        fp=path + pref + str(v);
        if (os.path.exists(fp)):
            res.append(DataChunk(fp,p1))
        else:
            return res
    return res



def GeneratorFromChunks(chunks):
    while True:
        for i in chunks:
            x, all_y = i.aquireData()
            yield x, all_y



from keras.models import Sequential, Model
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.optimizers import Adam, RMSprop, Adagrad
from keras.layers import LSTM, Dense, Input, concatenate, Conv1D
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers import Embedding, Activation, Merge
from keras.layers.wrappers import Bidirectional
import keras.callbacks
def myModel():

    classes = [84,82,203,114,174,60,75,537,101,96,44,261,56,116,60,74,67,72,26]

    input = Input(shape=(5,351))
    input_1 = LSTM(150, return_sequences=True)(input)

    merged = concatenate([input, input_1])
    outs = []

    for i in range(len(classes)):
        layer_1 = LSTM(70, return_sequences=True)(merged)
        layer_2 = LSTM(70)(layer_1)
        output = Dense(classes[i], activation='softmax')(layer_2)
        outs.append(output)
    model = Model(inputs=input, outputs=outs)
    print 'Model built'
    return model;
class StatPrinter(keras.callbacks.Callback):
    def __init__(self, model,batch):
        super(StatPrinter, self).__init__()
        self.model=model
        self.batch=batch


    def on_epoch_end(self, epoch, logs={}):
        sucCount = 0
        totalCount = 0;
        for v in self.batch:
            input,output=v.aquireData();
            #classes = self.model.predict_classes(input);
            classes = self.model.predict(input)
            for i in range(0, len(output)):
                num = numpy.argmax(output[i]);
                # out.write("%s\n"%classes[i])
                if (classes[i] == num):
                    sucCount = sucCount + 1;
                totalCount = totalCount + 1;
        print ""

        print "Samples count:"+str(sucCount)+":"+str(totalCount)



ds=chunker("H:/shuffled/","H:/shuffled/","learnSet")
training=GeneratorFromChunks(ds[0:6500])
test=GeneratorFromChunks(ds[6500:7000])
modelPath="H:/weights_hier"
model1 = myModel();
def train(mp):
    global ds,test,training, model1
    saver=keras.callbacks.ModelCheckpoint(mp, monitor='val_loss', verbose=1, save_best_only=True,mode='auto')
    stopper=keras.callbacks.EarlyStopping(monitor='val_loss', patience=1, verbose=1, mode='auto')

    #st=StatPrinter(model,ds[70:75])
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    adam = Adam(lr=0.01)
    adagrad = Adagrad(lr=0.01)
    rmsprop = RMSprop(lr=0.0005)
    # model1.load_weights(modelPath)
    #model.compile(optimizer=adam, loss='categorical_crossentropy')
    #model.compile(loss='mse', optimizer='rmsprop')
    model1.compile(optimizer=rmsprop, loss='categorical_crossentropy', metrics=['categorical_accuracy'])


    model1.fit_generator(training,steps_per_epoch=6500,epochs=100,max_q_size=3,validation_data=test,validation_steps=500,callbacks=[saver,stopper])

train(modelPath)
model1.save('H:/model_hier.h5')

