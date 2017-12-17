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

# F(B)=(1+B^2)*(PR/(B*B*P+R))
def f2_score(y_true, y_pred):
    y_true = tf.cast(y_true, "float64")
    y_pred = tf.cast(tf.round(y_pred), "float64") # implicit 0.5 threshold via tf.round
    y_correct = y_true * y_pred
    sum_true = tf.reduce_sum(y_true, axis=1)
    sum_pred = tf.reduce_sum(y_pred, axis=1)
    sum_correct = tf.reduce_sum(y_correct, axis=1)
    precision = sum_correct / sum_pred
    recall = sum_correct / sum_true
    f_score = 5 * precision * recall / (4 * precision + recall)
    f_score = tf.where(tf.is_nan(f_score), tf.zeros_like(f_score), f_score)
    return tf.reduce_mean(f_score)

def f1_score(y_true, y_pred):
    y_true = tf.cast(y_true, "float64")
    y_pred = tf.cast(tf.round(y_pred), "float64") # implicit 0.5 threshold via tf.round
    y_correct = y_true * y_pred
    sum_true = tf.reduce_sum(y_true, axis=1)
    sum_pred = tf.reduce_sum(y_pred, axis=1)
    sum_correct = tf.reduce_sum(y_correct, axis=1)
    precision = sum_correct / sum_pred
    recall = sum_correct / sum_true
    f_score = 2 * precision * recall / (precision + recall)
    f_score = tf.where(tf.is_nan(f_score), tf.zeros_like(f_score), f_score)
    return tf.reduce_mean(f_score)

def text_to_word_list(text):
    ''' Pre process and convert texts to a list of words '''
    text = str(text)
    text = text.lower()

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    text = text.split()

    return text

def createIdx(train_data,test_data,word2embed):
    stops = set(stopwords.words('english'))
    inverse_vocabulary = ['<unk>']  # '<unk>' will never be used, it is only a placeholder for the [0, 0, ....0] embedding
    vocabulary = dict()
    for dataset in [train_data, test_data]:
        for index, row in dataset.iterrows():

            for question in questionHeader:

                q2n = []  # q2n -> question numbers representation
                for word in text_to_word_list(row[question]):

                    # Check for unwanted words
                    if word in stops and word not in word2embed:
                        continue

                    if word not in vocabulary:
                        vocabulary[word] = len(inverse_vocabulary)
                        q2n.append(len(inverse_vocabulary))
                        inverse_vocabulary.append(word)
                    else:
                        q2n.append(vocabulary[word])

                # Replace questions as word to question as number representation
                dataset.set_value(index, question, q2n)
        f1 = open('testData_index.pickle','w')
        f2 = open('trainData_index.pickle','w')
        pickle.dump(test_data,f1)
        pickle.dump(train_data,f2)
        f1.close()
        f2.close()

        embeddings = 1 * np.random.randn(len(vocabulary) + 1, 50)  # This will be the embedding matrix
        embeddings[0] = 0  # So that the padding will be ignored
    # Build the embedding matrix
        for word, index in vocabulary.items():
            if word in word2embed:
                    embeddings[index] = word2embed[word].numpy()
        f3 = open('wordQembed.pickle','w')
        pickle.dump(embeddings,f3)
        f3.close()

def exponent_neg_manhattan_distance(left, right):
    return K.exp(-K.sum(K.abs(left-right), axis=1, keepdims=True))

def cosine_distance(left,right):
    left = K.l2_normalize(left, axis=-1)
    right = K.l2_normalize(right, axis=-1)
    return -K.mean(left * right, axis=-1, keepdims=True)

def createEmbeds(flag):
    if flag:
        trainFile = 'train.csv'
        testFile = 'test.csv'
        train_data = pd.read_csv(trainFile)
        test_data = pd.read_csv(testFile)

        embeddingWords = 'word2embed.pickle'

        with open(embeddingWords, "rb") as input_file:
            word2embed = pickle.load(input_file)
        print 'embedding loaded'

        print "in createIdx"

        createIdx(train_data,test_data,word2embed)


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

        maxSentenceLen = 245
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
        maxSentenceLen = maxSentenceLength(train_data, test_data)
        siamese, optimizer = neuralNetworkModel(maxSentenceLen, embeddings, n_hidden, gradient_clipping_norm, optimizerUse, distanceFunc)
        siamese.load_weights(weightsFile)
        siamese = compileModel(siamese, optimizer, loss, metric)
        print("Loaded model from disk")
        testX={'q1':test_data.question1, 'q2':test_data.question2}
        for dataset, side in itertools.product([testX], ['q1', 'q2']):
            dataset[side] = pad_sequences(dataset[side], maxlen=maxSentenceLen)
        
        predictions= siamese.predict([testX['q1'], testX['q2']], batch_size=batch_size)
        print predictions
        rounded=[]
        # rounded = [round(x[0]) for x in predictions]
        for x in predictions:
            rounded.append(x)
        del predictions
        print(rounded)
        nos=[]
        for i in range(len(rounded)):
            nos.append((float(rounded[i])))
        del rounded
        df = pd.DataFrame(nos, columns=["is_duplicate"])
        df.to_csv('predictions.csv')

def runModel(runFlag, createEmbedsFlag,trainFlag,testFlag,weightsFile,modelFile,loss,metric,optimizerUse,distanceFunc,batch_size,n_epoch,validationDataSize,n_hidden,gradient_clipping_norm):
    if runFlag:
        questionHeader = ['question1', 'question2']    
        createEmbeds(createEmbedsFlag)
        train_data, test_data, embeddings = loadData()
        trainModel(trainFlag, train_data.head(200000).append(train_data.tail(40000)), test_data, embeddings, questionHeader, weightsFile, modelFile, loss, metric, batch_size, n_epoch, validationDataSize, n_hidden, gradient_clipping_norm, optimizerUse, distanceFunc)
        testModel(testFlag, train_data, test_data, embeddings, questionHeader,weightsFile, loss, metric, batch_size, n_hidden, gradient_clipping_norm, optimizerUse, distanceFunc)

def main():

    runModel(runFlag = False, createEmbedsFlag=False, trainFlag = False, testFlag = False, weightsFile = "keras_25epochs_malstm_1000batchWeights_AdaDelta.h5", modelFile = "keras_25epochs_malstm_1000batch_AdaDelta.json",
    loss =  'mean_squared_logarithmic_error', metric = 'accuracy', optimizerUse='Adadelta', distanceFunc = 'exponent_neg_manhattan_distance', batch_size = 1000, n_epoch = 25,
    validationDataSize = 40000, n_hidden = 50, gradient_clipping_norm = 1.25)
    
    runModel(runFlag = True, createEmbedsFlag=False, trainFlag = True, testFlag = True, weightsFile = "keras_150epochs_malstm_1000batchWeights_Adam_F1_200k.h5", modelFile = "keras_150epochs_malstm_1000batch_Adam_F1_200k.json",
    loss =  'mean_squared_logarithmic_error', metric = 'accuracy', optimizerUse='Adam', distanceFunc = 'exponent_neg_manhattan_distance', batch_size = 1000, n_epoch = 50,
    validationDataSize = 40000, n_hidden = 150, gradient_clipping_norm = 1.25) #winner model

    runModel(runFlag = False, createEmbedsFlag=False, trainFlag = False, testFlag = False, weightsFile = "keras_25epochs_malstm_1000batchWeights_SGD.h5", modelFile = "keras_25epochs_malstm_1000batch_SGD.json",
    loss =  'mean_squared_logarithmic_error', metric = 'accuracy', optimizerUse='SGD', distanceFunc = 'exponent_neg_manhattan_distance', batch_size = 1000, n_epoch = 25,
    validationDataSize = 40000, n_hidden = 50, gradient_clipping_norm = 1.25)

    runModel(runFlag = False, createEmbedsFlag=False, trainFlag = False, testFlag = False, weightsFile = "keras_25epochs_malstm_1000batchWeights_Adam_Cosine.h5", modelFile = "keras_25epochs_malstm_1000batch_Adam_Cosine.json",
    loss =  'mean_squared_logarithmic_error', metric = 'accuracy', optimizerUse='SGD', distanceFunc = 'cosine_distance', batch_size = 1000, n_epoch = 25,
    validationDataSize = 40000, n_hidden = 50, gradient_clipping_norm = 1.25)

if __name__ == '__main__':
    main()
