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


class DataChunk:
    def __init__(self, path,pathData):
        self.path = path
        v=path.index("/");
        self.pathDaa=pathData+path[v+1:len(path)]

    def getInput(self):
        return self.path+"input.npy"

    def getOutput(self):
        return self.path + "output.npy"

    def aquireData(self):
        inputPath=self.getInput()
        outputPath = self.getOutput()
        if os.path.exists(inputPath)&os.path.exists(outputPath):
            inputArray=numpy.load(inputPath);
            outputArray = numpy.load(outputPath);
            return inputArray,outputArray
        if os.path.exists(self.path):
            self.prepareData()
            if os.path.exists(inputPath) & os.path.exists(outputPath):
                return self.aquireData();
            raise IOError("Data preparation failed")
        raise IOError("Chunk file does not exists")

    def prepareData(self):
        print "Preparing"
        inputs,outputs=self.internalPrepare();
        inputs = numpy.array(inputs)
        outputs = numpy.array(outputs)
        numpy.save(self.getInput(), inputs)
        numpy.save(self.getOutput(), outputs)

    def internalPrepare(self):
        f = file(self.path, "r")
        inputs = []
        outputs = []
        num = 0
        for l in f.readlines():
            last = l.index(':');
            first = l[0: last].strip()
            if first == 'null':
                continue
            second = l[last + 1:len(l)].strip();
            inputArray = numpy.fromstring(second,dtype=numpy.float32,sep=",")
            # output = numpy.fromstring(first, dtype=numpy.float32, sep=",")
            # output = numpy.array(output)
            output = numpy.zeros(38, dtype=numpy.float32)
            output[int(first)] = 1.0
            # output = output.reshape(1, 300)
            input = numpy.array(inputArray);
            input = input.reshape(40, 38)
            inputs.append(input);
            outputs.append(output);
        return inputs,outputs

def chunker(path,p1,pref):
    res=[]
    for v in range(0,15000):
        fp=path + pref + str(v);
        if (os.path.exists(fp)):
            res.append(DataChunk(fp,p1))
        else:
            return res
    return res



def GeneratorFromChunks(chunks):
    while True:
        for i in chunks:
            (x, y) = i.aquireData()
            yield x,y



from keras.models import Sequential, Model
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.optimizers import Adam, RMSprop, Adagrad
from keras.layers import LSTM, Dense, Input
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers import Embedding, Activation, Merge
from keras.layers.wrappers import Bidirectional
import keras.callbacks
def myModel():
    model = Sequential()
    model.add(Convolution1D(100, 7, input_shape=(40, 38), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Convolution1D(150, 7, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Convolution1D(200, 7, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(200, activation='relu'))
    model.add(Dense(38, activation='softmax'))
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



ds=chunker("H:/typing_data/","H:/typing_data/","learnSet")
training=GeneratorFromChunks(ds[0:10000])
test=GeneratorFromChunks(ds[10000:15000])
modelPath="H:/weights1"
def train(mp):
    global ds,test,training
    saver=keras.callbacks.ModelCheckpoint(mp, monitor='val_loss', verbose=1, save_best_only=True,mode='auto')
    stopper=keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')
    model = myModel();
    #st=StatPrinter(model,ds[70:75])
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    adam = Adam(lr=0.01)
    adagrad = Adagrad(lr=0.01)
    rmsprop = RMSprop(lr=0.00001)
    #model.load_weights(modelPath)
    #model.compile(optimizer=adam, loss='categorical_crossentropy')
    #model.compile(loss='mse', optimizer='rmsprop')
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['top_k_categorical_accuracy'])


    model.fit_generator(training,steps_per_epoch=10000,epochs=1500,max_q_size=3,validation_data=test,validation_steps=5000,callbacks=[saver,stopper])

train(modelPath)


