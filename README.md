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
