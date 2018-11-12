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
        outputFoodPath = self.getOutput('food')
        outputFamilyPath = self.getOutput('family')
        outputDatePath = self.getOutput('dates')
        outputPronounsPath = self.getOutput('pronouns')
        outputIngPath = self.getOutput('ing')
        outputCognPath = self.getOutput('cogn')
        outputTimePath = self.getOutput('time')
        outputModalPath = self.getOutput('modal')
        outputAdjPath = self.getOutput('adj')
        if os.path.exists(inputPath)&os.path.exists(outputFoodPath)&os.path.exists(outputFamilyPath)&os.path.exists(outputDatePath)&os.path.exists(outputPronounsPath) \
                & os.path.exists(outputIngPath)&os.path.exists(outputCognPath)&os.path.exists(outputTimePath)&os.path.exists(outputModalPath)&os.path.exists(outputAdjPath):
            inputArray=numpy.load(inputPath);
            outputFoodArray = numpy.load(outputFoodPath)
            outputFamilyArray = numpy.load(outputFamilyPath)
            outputDateArray = numpy.load(outputDatePath)
            outputPronounArray = numpy.load(outputPronounsPath)
            outputIngArray = numpy.load(outputIngPath)
            outputCognArray = numpy.load(outputCognPath)
            outputTimeArray = numpy.load(outputTimePath)
            outputModalArray = numpy.load(outputModalPath)
            outputAdjArray = numpy.load(outputAdjPath)
            return inputArray,outputFoodArray, outputFamilyArray, outputDateArray, outputPronounArray,outputIngArray, outputCognArray, outputTimeArray, outputModalArray, outputAdjArray
        if os.path.exists(self.path):
            self.prepareData()
            if os.path.exists(inputPath) &os.path.exists(outputFoodPath)&os.path.exists(outputFamilyPath)&os.path.exists(outputDatePath)&os.path.exists(outputPronounsPath) \
                & os.path.exists(outputIngPath)&os.path.exists(outputCognPath)&os.path.exists(outputTimePath)&os.path.exists(outputModalPath)&os.path.exists(outputAdjPath):
                return self.aquireData();
            raise IOError("Data preparation failed")
        raise IOError("Chunk file does not exists")

    def prepareData(self):
        print "Preparing"
        inputs,outputsFood, outputsFamily, outputsDate, outputsPronoun, outputsIng, outputsCogn, outputsTime, outputsModal, outputsAdj=self.internalPrepare();
        inputs = numpy.array(inputs)
        outputsFood = numpy.array(outputsFood)
        outputsFamily = numpy.array(outputsFamily)
        outputsDate = numpy.array(outputsDate)
        outputsPronoun = numpy.array(outputsPronoun)
        outputsIng = numpy.array(outputsIng)
        outputsCogn = numpy.array(outputsCogn)
        outputsTime = numpy.array(outputsTime)
        outputsModal = numpy.array(outputsModal)
        outputsAdj = numpy.array(outputsAdj)
        numpy.save(self.getInput(), inputs)
        numpy.save(self.getOutput('food'), outputsFood)
        numpy.save(self.getOutput('family'), outputsFamily)
        numpy.save(self.getOutput('dates'), outputsDate)
        numpy.save(self.getOutput('pronouns'), outputsPronoun)
        numpy.save(self.getOutput('ing'), outputsIng)
        numpy.save(self.getOutput('cogn'), outputsCogn)
        numpy.save(self.getOutput('time'), outputsTime)
        numpy.save(self.getOutput('modal'), outputsModal)
        numpy.save(self.getOutput('adj'), outputsAdj)

    def internalPrepare(self):
        f = file(self.path, "r")
        inputs = []
        outputs_food = []
        outputs_family = []
        outputs_dates = []
        outputs_pronouns = []
        outputs_ing = []
        outputs_cogn = []
        outputs_time = []
        outputs_modal = []
        outputs_adj = []
        num = 0
        for l in f.readlines():
            parts = l.split(';')
            if len(parts) < 2:
                continue
            embedding = numpy.fromstring(parts[0], dtype=numpy.float32, sep=',')
            embedding = embedding.reshape(5, 300)
            short_words = numpy.fromstring(parts[1], dtype=numpy.float32, sep=',')
            short_words = short_words.reshape(5, 31)
            input1 = numpy.concatenate((embedding, short_words), axis=1)
            inputs.append(input1)
            output_food = numpy.fromstring(parts[2], dtype=numpy.float32, sep=',')
            outputs_food.append(output_food)
            output_family = numpy.fromstring(parts[3], dtype=numpy.float32, sep=',')
            outputs_family.append(output_family)
            output_dates = numpy.fromstring(parts[4], dtype=numpy.float32, sep=',')
            outputs_dates.append(output_dates)
            output_pronouns = numpy.fromstring(parts[5], dtype=numpy.float32, sep=',')
            outputs_pronouns.append(output_pronouns)
            output_ing = numpy.fromstring(parts[6], dtype=numpy.float32, sep=',')
            outputs_ing.append(output_ing)
            output_cogn = numpy.fromstring(parts[7], dtype=numpy.float32, sep=',')
            outputs_cogn.append(output_cogn)
            output_time = numpy.fromstring(parts[8], dtype=numpy.float32, sep=',')
            outputs_time.append(output_time)
            output_modal = numpy.fromstring(parts[9], dtype=numpy.float32, sep=',')
            outputs_modal.append(output_modal)
            output_adj = numpy.fromstring(parts[10], dtype=numpy.float32, sep=',')[:-1]
            outputs_adj.append(output_adj)
        return inputs, outputs_food, outputs_family, outputs_dates, outputs_pronouns, outputs_ing, outputs_cogn, outputs_time, outputs_modal, outputs_adj

def chunker(path,p1,pref):
    res=[]
    for v in range(0,47000):
        fp=path + pref + str(v);
        if (os.path.exists(fp)):
            res.append(DataChunk(fp,p1))
        else:
            return res
    return res



def GeneratorFromChunks(chunks):
    while True:
        for i in chunks:
            (x, y_food, y_family, y_date, y_pronoun, y_ing, y_cogn, y_time, y_modal, y_adj) = i.aquireData()
            yield x,[y_food, y_family, y_date, y_pronoun, y_ing, y_cogn, y_time, y_modal, y_adj]



from keras.models import Sequential, Model
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.optimizers import Adam, RMSprop, Adagrad
from keras.layers import LSTM, Dense, Input, concatenate
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers import Embedding, Activation, Merge
from keras.layers.wrappers import Bidirectional
import keras.callbacks
def myModel():

    input = Input(shape=(5,331))
    input_1 = LSTM(150, return_sequences=True)(input)
    input_2 = LSTM(150, return_sequences=True)(input_1)

    merged = concatenate([input, input_2])

    food_1 = LSTM(150, return_sequences=True)(merged)
    food_2 = LSTM(150)(food_1)
    food_output = Dense(41, activation='softmax')(food_2)

    family_1 = LSTM(150, return_sequences=True)(merged)
    family_2 = LSTM(150)(family_1)
    family_output = Dense(59, activation='softmax')(family_2)

    dates_1 = LSTM(150, return_sequences=True)(merged)
    dates_2 = LSTM(150)(dates_1)
    dates_output = Dense(9, activation='softmax')(dates_2)

    pronouns_1 = LSTM(150, return_sequences=True)(merged)
    pronouns_2 = LSTM(150)(pronouns_1)
    pronouns_output = Dense(34, activation='softmax')(pronouns_2)

    ing_1 = LSTM(150, return_sequences=True)(merged)
    ing_2 = LSTM(150)(ing_1)
    ing_output = Dense(60, activation='softmax')(ing_2)

    cogn_1 = LSTM(150, return_sequences=True)(merged)
    cogn_2 = LSTM(150)(cogn_1)
    cogn_output = Dense(53, activation='softmax')(cogn_2)

    time_1 = LSTM(150, return_sequences=True)(merged)
    time_2 = LSTM(150)(time_1)
    time_output = Dense(38, activation='softmax')(time_2)

    modal_1 = LSTM(150, return_sequences=True)(merged)
    modal_2 = LSTM(150)(modal_1)
    modal_output = Dense(66, activation='softmax')(modal_2)

    adj_1 = LSTM(150, return_sequences=True)(merged)
    adj_2 = LSTM(150)(adj_1)
    adj_output = Dense(101, activation='softmax')(adj_2)


    model = Model(inputs=input, outputs=[food_output, family_output, dates_output, pronouns_output, ing_output, cogn_output, time_output, modal_output,adj_output])


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



ds=chunker("H:/data/","H:/data/","learnSet")
training=GeneratorFromChunks(ds[0:3000])
test=GeneratorFromChunks(ds[3000:3400])
modelPath="H:/weights_hier"
model1 = myModel();
def train(mp):
    global ds,test,training, model1
    saver=keras.callbacks.ModelCheckpoint(mp, monitor='val_loss', verbose=1, save_best_only=True,mode='auto')
    stopper=keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')

    #st=StatPrinter(model,ds[70:75])
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    adam = Adam(lr=0.01)
    adagrad = Adagrad(lr=0.01)
    rmsprop = RMSprop(lr=0.0005)
    # model1.load_weights(modelPath)
    #model.compile(optimizer=adam, loss='categorical_crossentropy')
    #model.compile(loss='mse', optimizer='rmsprop')
    model1.compile(optimizer=rmsprop, loss='categorical_crossentropy', metrics=['top_k_categorical_accuracy'])


    model1.fit_generator(training,steps_per_epoch=3000,epochs=100,max_q_size=3,validation_data=test,validation_steps=400,callbacks=[saver,stopper])

train(modelPath)
model1.save('H:/model_hier.h5')


