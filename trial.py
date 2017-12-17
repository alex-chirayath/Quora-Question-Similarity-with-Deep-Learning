from time import time
import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
import re
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns

import itertools
import datetime
from keras.models import model_from_json
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Merge, Dense, Concatenate
import keras.backend as K
from keras.optimizers import Adadelta, Adam, SGD
from keras.callbacks import ModelCheckpoint
import pickle
import torch
from keras.models import load_model
import pandas as pd
from keras.callbacks import History 

import tensorflow as tf


def exponent_neg_manhattan_distance(left, right):
    return K.exp(-K.sum(K.abs(left-right), axis=1, keepdims=True))

def loadData():
    with open('trainData_index.pickle', "rb") as input_file:
        train_data = pickle.load(input_file)
    with open('testData_index.pickle', "rb") as input_file:
        test_data = pickle.load(input_file)
    with open('wordQembed.pickle', "rb") as input_file:
        embeddings = pickle.load(input_file)
    print "data loaded"

    return train_data, test_data, embeddings

def trainValidationSplit(questionsCols, isDuplicate, validationDataSize):
    return train_test_split(questionsCols, isDuplicate, test_size=validationDataSize)

def maxSentenceLength(train_data, test_data):
    return max(train_data.question1.map(lambda x: len(x)).max(),
                     train_data.question2.map(lambda x: len(x)).max(),
                     test_data.question1.map(lambda x: len(x)).max(),
                     test_data.question2.map(lambda x: len(x)).max())

def padSequenceTrain(trainX, validX, trainY, maxSentenceLen):
    for dataset, side in itertools.product([trainX, validX], ['q1', 'q2']):
        dataset[side] = pad_sequences(dataset[side], maxlen=maxSentenceLen)

    # Make sure everything is ok
    assert trainX['q1'].shape == trainX['q2'].shape
    assert len(trainX['q1']) == len(trainY)

    print len(validX)
    return trainX, validX

def neuralNetworkModel2(maxSentenceLen, embeddings, n_hidden, gradient_clipping_norm, optimizerUse, distanceFunc):
    
    q1_in = Input(shape=(maxSentenceLen,), dtype='int32')
    q2_in = Input(shape=(maxSentenceLen,), dtype='int32')

    
    embedLayer = Embedding(len(embeddings), 50, weights=[embeddings], input_length=maxSentenceLen, trainable=False)

    q1_embed = embedLayer(q1_in)
    q2_embed = embedLayer(q2_in)

    siameseLstm = LSTM(n_hidden, dropout=0.0, recurrent_dropout=0.00, return_sequences=True)
    siameseLstm2 = LSTM(50, dropout=0.00, recurrent_dropout=0.0, return_sequences=False)

    q1_out = siameseLstm(q1_embed)
    q2_out = siameseLstm(q2_embed)

    q1_out2=siameseLstm2(q1_out)
    q2_out2=siameseLstm2(q2_out)

    if distanceFunc == 'exponent_neg_manhattan_distance':
        distance = Merge(mode=lambda x: exponent_neg_manhattan_distance(x[0], x[1]), output_shape=lambda x: (x[0][0], 1))([q1_out2, q2_out2])
    if distanceFunc == 'cosine_distance':
        distance = Merge(mode=lambda x: cosine_distance(x[0], x[1]), output_shape=lambda x: (x[0][0], 1))([q1_out, q2_out])
    if distanceFunc == 'Dense_layer_softmax':
        distance = Dense(10, activation='softmax')(Concatenate(axis=-1)([q1_out,q2_out]))

    siamese = Model([q1_in, q2_in], [distance])

    if optimizerUse=='Adadelta':
        optimizer = Adadelta(clipnorm=gradient_clipping_norm)
    
    if optimizerUse == 'Adam':
        optimizer = Adam(clipnorm=gradient_clipping_norm)        
    
    if optimizerUse == 'SGD':
        optimizer = SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False, clipnorm = gradient_clipping_norm)

    return siamese, optimizer


def neuralNetworkModel(maxSentenceLen, embeddings, n_hidden, gradient_clipping_norm, optimizerUse, distanceFunc):
    
    q1_in = Input(shape=(maxSentenceLen,), dtype='int32')
    q2_in = Input(shape=(maxSentenceLen,), dtype='int32')

    
    embedLayer = Embedding(len(embeddings), 50, weights=[embeddings], input_length=maxSentenceLen, trainable=False)

    q1_embed = embedLayer(q1_in)
    q2_embed = embedLayer(q2_in)

    siameseLstm = LSTM(n_hidden, dropout=0.0, recurrent_dropout=0.05, return_sequences=False)

    q1_out = siameseLstm(q1_embed)
    q2_out = siameseLstm(q2_embed)

    if distanceFunc == 'exponent_neg_manhattan_distance':
        distance = Merge(mode=lambda x: exponent_neg_manhattan_distance(x[0], x[1]), output_shape=lambda x: (x[0][0], 1))([q1_out, q2_out])
    if distanceFunc == 'cosine_distance':
        distance = Merge(mode=lambda x: cosine_distance(x[0], x[1]), output_shape=lambda x: (x[0][0], 1))([q1_out, q2_out])
    if distanceFunc == 'Dense_layer_softmax':
        distance = Dense(10, activation='softmax')(Concatenate(axis=-1)([q1_out,q2_out]))

    siamese = Model([q1_in, q2_in], [distance])

    if optimizerUse=='Adadelta':
        optimizer = Adadelta(clipnorm=gradient_clipping_norm)
    
    if optimizerUse == 'Adam':
        optimizer = Adam(clipnorm=gradient_clipping_norm)        
    
    if optimizerUse == 'SGD':
        optimizer = SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False, clipnorm = gradient_clipping_norm)

    return siamese, optimizer

def compileModel(siamese, optimizer, loss, metric):
    siamese.compile(loss=loss, optimizer=optimizer, metrics=[metric])
    return siamese

def startTraining(siamese, trainX, trainY, validX, validY, batch_size, n_epoch,history):


    training_start_time = time()

    siamese.fit([trainX['q1'], trainX['q2']], trainY, batch_size=batch_size, nb_epoch=n_epoch,
                                validation_data=([validX['q1'], validX['q2']], validY), callbacks=[history])
    print("Training time finished.\n{} epochs in {}".format(n_epoch, datetime.timedelta(seconds=time()-training_start_time)))

    return siamese 

def saveModel(siamese, weightsFile, modelFile):
    siamese.save_weights(weightsFile)
    # siamese.save('keras_25epochs_malstm_2000batch.h5')
    model_json = siamese.to_json()
    with open(modelFile, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    print("Saved model to disk")

def validateModel(flag, maxSentenceLen, validX, validY, siamese, batch_size):
    if flag:
        print "validation:"
        predictionsEval= siamese.evaluate([validX['q1'], validX['q2']],validY, batch_size=batch_size)
        print("%s: %.2f%%" % (siamese.metrics_names[1], predictionsEval[1]*100))

def trainModel(flag, train_data, test_data, embeddings, questionHeader, weightsFile, modelFile, loss, metric, batch_size, n_epoch, validationDataSize,n_hidden, gradient_clipping_norm, optimizerUse, distanceFunc):
    if flag:

        trainingDataSize = len(train_data) - validationDataSize
        questionsCols = train_data[questionHeader]
        isDuplicate = train_data['is_duplicate']

        maxSentenceLen = maxSentenceLength(train_data, test_data)
        print maxSentenceLen
        trainX, validX, trainY, validY = trainValidationSplit(questionsCols, isDuplicate, validationDataSize)
        print 'train data split'

        trainX = {'q1': trainX.question1, 'q2': trainX.question2}
        validX = {'q1': validX.question1, 'q2': validX.question2}
        testX={'q1': test_data.question1, 'q2': test_data.question2}
        trainY = trainY.values
        validY = validY.values

        trainX, validX = padSequenceTrain(trainX, validX, trainY, maxSentenceLen)

        siamese, optimizer = neuralNetworkModel(maxSentenceLen, embeddings, n_hidden, gradient_clipping_norm, optimizerUse, distanceFunc)

        siamese = compileModel(siamese, optimizer,loss, metric)

        history = History()

        siamese = startTraining(siamese, trainX, trainY, validX, validY, batch_size, n_epoch, history)

        saveModel(siamese, weightsFile, modelFile)

        # summarize history for accuracy
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig('accuracy.png')
        plt.close()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig('loss.png')


        validateModel(True, maxSentenceLen, validX, validY, siamese, batch_size)

        del siamese

def testModel(flag, train_data, test_data, embeddings, questionHeader, weightsFile, loss, metric, batch_size, n_hidden, gradient_clipping_norm, optimizerUse, distanceFunc):
    if flag:
        maxSentenceLen = 245
        siamese, optimizer = neuralNetworkModel(maxSentenceLen, embeddings, n_hidden, gradient_clipping_norm, optimizerUse, distanceFunc)
        siamese.load_weights(weightsFile)
        siamese = compileModel(siamese, optimizer, loss, metric)
        print("Loaded model from disk")
        validationDataSize=10000
        questionsCols = train_data[questionHeader]
        isDuplicate = train_data['is_duplicate']
        trainX, validX, trainY, validY = trainValidationSplit(questionsCols, isDuplicate, validationDataSize)
        print 'train data split'

        trainX = {'q1': trainX.question1, 'q2': trainX.question2}
        validX = {'q1': validX.question1, 'q2': validX.question2}
        testX={'q1': test_data.question1, 'q2': test_data.question2}
        trainY = trainY.values
        validY = validY.values

        trainX, validX = padSequenceTrain(trainX, validX, trainY, maxSentenceLen)

        validateModel(True, maxSentenceLen, validX, validY, siamese, batch_size)





def testModel2(flag, train_data, test_data, embeddings, questionHeader, weightsFile, loss, metric, batch_size, n_hidden, gradient_clipping_norm, optimizerUse, distanceFunc):
    if flag:
        maxSentenceLen = 245
        siamese, optimizer = neuralNetworkModel2(maxSentenceLen, embeddings, n_hidden, gradient_clipping_norm, optimizerUse, distanceFunc)
        siamese.load_weights(weightsFile)
        siamese = compileModel(siamese, optimizer, loss, metric)
        print("Loaded model from disk")
        validationDataSize=10000
        questionsCols = train_data[questionHeader]
        isDuplicate = train_data['is_duplicate']
        trainX, validX, trainY, validY = trainValidationSplit(questionsCols, isDuplicate, validationDataSize)
        print 'train data split'

        trainX = {'q1': trainX.question1, 'q2': trainX.question2}
        validX = {'q1': validX.question1, 'q2': validX.question2}
        testX={'q1': test_data.question1, 'q2': test_data.question2}
        trainY = trainY.values
        validY = validY.values

        trainX, validX = padSequenceTrain(trainX, validX, trainY, maxSentenceLen)

        validateModel(True, maxSentenceLen, validX, validY, siamese, batch_size)



def runModel(runFlag, createEmbedsFlag,trainFlag,testFlag,weightsFile,modelFile,loss,metric,optimizerUse,distanceFunc,batch_size,n_epoch,validationDataSize,n_hidden,gradient_clipping_norm):
    if runFlag:
        questionHeader = ['question1', 'question2']    
        train_data, test_data, embeddings = loadData()
        testModel(testFlag, train_data.tail(40000), test_data, embeddings, questionHeader,weightsFile, loss, metric, batch_size, n_hidden, gradient_clipping_norm, optimizerUse, distanceFunc)
        testModel(testFlag, train_data.tail(40000).head(30000), test_data, embeddings, questionHeader,weightsFile, loss, metric, batch_size, n_hidden, gradient_clipping_norm, optimizerUse, distanceFunc)
        testModel(testFlag, train_data.tail(40000).head(20000), test_data, embeddings, questionHeader,weightsFile, loss, metric, batch_size, n_hidden, gradient_clipping_norm, optimizerUse, distanceFunc)
        testModel(testFlag, train_data.tail(50000).head(20000), test_data, embeddings, questionHeader,weightsFile, loss, metric, batch_size, n_hidden, gradient_clipping_norm, optimizerUse, distanceFunc)

def runModel2(runFlag, createEmbedsFlag,trainFlag,testFlag,weightsFile,modelFile,loss,metric,optimizerUse,distanceFunc,batch_size,n_epoch,validationDataSize,n_hidden,gradient_clipping_norm):
    if runFlag:
        questionHeader = ['question1', 'question2']    
        train_data, test_data, embeddings = loadData()
        testModel2(testFlag, train_data.tail(40000), test_data, embeddings, questionHeader,weightsFile, loss, metric, batch_size, n_hidden, gradient_clipping_norm, optimizerUse, distanceFunc)
        testModel2(testFlag, train_data.tail(40000).head(30000), test_data, embeddings, questionHeader,weightsFile, loss, metric, batch_size, n_hidden, gradient_clipping_norm, optimizerUse, distanceFunc)
        testModel2(testFlag, train_data.tail(40000).head(20000), test_data, embeddings, questionHeader,weightsFile, loss, metric, batch_size, n_hidden, gradient_clipping_norm, optimizerUse, distanceFunc)
        testModel2(testFlag, train_data.tail(50000).head(20000), test_data, embeddings, questionHeader,weightsFile, loss, metric, batch_size, n_hidden, gradient_clipping_norm, optimizerUse, distanceFunc)


def main():

    # runModel(runFlag = True, createEmbedsFlag=False, trainFlag = False, testFlag = True, weightsFile = "keras_150epochs_malstm_1000batchWeights_Adam_F1.h5", modelFile = "keras_150epochs_malstm_1000batch_Adam_F1.json",
    # loss =  'mean_squared_logarithmic_error', metric = 'accuracy', optimizerUse='Adam', distanceFunc = 'exponent_neg_manhattan_distance', batch_size = 1000, n_epoch = 100,
    # validationDataSize = 10000, n_hidden = 150, gradient_clipping_norm = 1.25) #winner model

    # runModel2(runFlag = True, createEmbedsFlag=False, trainFlag = True, testFlag = True, weightsFile = "2siamese.h5", modelFile = "2siamese.json",
    # loss =  'mean_squared_logarithmic_error', metric = 'accuracy', optimizerUse='Adam', distanceFunc = 'exponent_neg_manhattan_distance', batch_size = 1000, n_epoch = 25,
    # validationDataSize = 10000, n_hidden = 100, gradient_clipping_norm = 1.25)

    # runModel2(runFlag = True, createEmbedsFlag=False, trainFlag = True, testFlag = True, weightsFile = "2siamese2.h5", modelFile = "2siamese2.json",
    # loss =  'mean_squared_logarithmic_error', metric = 'accuracy', optimizerUse='Adam', distanceFunc = 'exponent_neg_manhattan_distance', batch_size = 1000, n_epoch = 100,
    # validationDataSize = 10000, n_hidden = 100, gradient_clipping_norm = 1.25)

    runModel(runFlag = True, createEmbedsFlag=False, trainFlag = False, testFlag = True, weightsFile = "keras_150epochs_malstm_1000batchWeights_Adam_F1_200k.h5", modelFile = "keras_150epochs_malstm_1000batch_Adam_F1_200k.json",
    loss =  'mean_squared_logarithmic_error', metric = 'accuracy', optimizerUse='Adam', distanceFunc = 'exponent_neg_manhattan_distance', batch_size = 1000, n_epoch = 50,
    validationDataSize = 10000, n_hidden = 150, gradient_clipping_norm = 1.25) #winner model


if __name__ == '__main__':
    main()
