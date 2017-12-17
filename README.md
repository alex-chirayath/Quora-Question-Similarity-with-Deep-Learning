# Paraphrase-Detection-Quora-Question-Similarity-with-Deep-Learning
Given a pair of questions on Quora, the  NLP task aims to classify  whether two given questions are duplicates or not. The underlying  idea is to identify whether the pair of questions have the same intent  though they might have been structured differently .

#Approach
To classify the questions correctly, it is essential that the learning model can interpret the underlying semantic representation of the 
sentences. Most models achieved this by using word vectors of large dimensions (200 or 300) with dense hidden layers and hand-crafted  features.  We  aimed  at  determining  whether  comparable 
performance could be achieved by using a comparatively simpler neural network that learns from scratch.
For this text classification task, we decided to use LSTM owing to their  effective  memory  retention  capabilities  for  understanding 
word  dependencies. As  a  part  of our  study,  we  tried  several approaches based on LSTM and have listed them in the following 
subsections. GloVe word embeddings of 50 dimensions were used as inputs to the LSTMs in order to create a representation for each 
question.  The  only  pre-processing performed  on  the  dataset involved replacing certain common contraction of words to distinct 
words to match  the  GloVe  vectors.  We  used  random  word embeddings for those words which were not present in the GloVe 
vocabulary.  Zero  padding  was  performed  considering  the maximum question length in dataset in order to have fixed size 
inputs for the network. No other explicit feature engineering was carried out during the course of the project. 

For the project, we decided to choose Multinomial Naive Bayes as our baseline for the project with Term Frequency – Inverse Document Frequency as the feature representations. Also, a prediction of just class 0 (non-duplicate) for all examples is bound to be 63% accurate on training data and our model needs to give a higher accuracy to be considered non-trivial. 
Holdout cross validation strategy was used since the training size was too large to perform k-fold cross validation given the time constraints. We held out 40,000 samples (roughly 10%) from the training data as the validation set. 

The Quora dataset is imbalanced in the sense that there are many samples of class 0 (non-duplicates) as compared to class 1 (duplicates). To solve the issue of class 0 overpowering class 1 during loss minimization in online algorithm, we implemented the batch algorithm and used class weights in optimization so that the model considers this imbalance.

The experiments involved different architectures of neural networks involving LSTMs. Since the task is to identify duplicates and the result remains the same even if we swap question 1 and question 2, it was decided that the same LSTM should be used as

two different LSTMs would lead to different representations based on whether it occurs as question 1 or question 2 for question representation. Here, we use the concept of Siamese network of shared weights between LSTMs for representing the two questions. Also, in addition to using a shared LSTM independently, a variation based on conditional encoding of question 2 on question 1 was part of the experiments as well. We also tried different variants by replacing a distance function with a bilinear layer followed by an output layer with softmax as shown in the diagram for Architecture b), which would generate the probability of class 0 and class 1 identical to classification task. Architecture a) illustrates a Dual-Siamese network where we use the outputs of the LSTM as input to another LSTM model to further extract meaningful representation of the questions, following which distance is taken. Another experiment comprised of determining performance of a model based on a single Siamese layer but with bidirectional LSTMs to capture additional context.

Hyper-parameter tuning involved varying the number of epochs, number of hidden layers, hidden layer size, output vector size, distance function used, optimizer used, dropout rate, batch size. The metric being used is accuracy and log-loss. 
