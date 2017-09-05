# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 21:33:26 2017

@author: BintangBuntoro
"""

import MainSentiment as ms
import numpy as np
from sklearn import svm
from sklearn.cross_validation import cross_val_score

## Create Tweet Feature & Lexicon Feature
#dataset_LX,Y = ms.tweetFeature('Dataset/dataset-positif.csv','Dataset/dataset-negatif.csv','1','0')

# Create TF-IDF
#dataset_corpus = ms.createCorpus('Dataset/dataset-positif.csv','Dataset/dataset-negatif.csv')
#dataset_TFIDF = ms.createTfidf(dataset_corpus,dataset_corpus,2)

## Extracted Data
#print "Process Concatenate Dataset"
#dataset_vector =  np.concatenate((dataset_TFIDF, dataset_LX), axis=1)
#
## SVM Classifier
#print('Modelling SVM...')
#svmClassifier = svm.LinearSVC()
#scores = cross_val_score(svmClassifier, dataset_vector, Y, cv=5, scoring='f1')
#print(scores)
#print("Perormance: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
#
#np.save("dataset lexicon Pattern & Sentiword.npy",dataset_LX)
#np.save("dataset label.npy",Y)
#np.save("dataset corpus all-preprocessing.npy",dataset_corpus)
#np.save("Model/dataset TF-IDF bigram",dataset_TFIDF)

#train_LX,Y = ms.tweetFeature('Dataset/positif.csv','Dataset/negatif.csv','1','0')
#test_LX,Y_test = ms.tweetFeature('Dataset/Testing/positif.csv','Dataset/Testing/negatif.csv','1','0')

train_corpus = ms.createCorpus('Dataset/positif.csv','Dataset/negatif.csv')
test_corpus = ms.createCorpus('Dataset/Testing/positif.csv','Dataset/Testing/negatif.csv')
train_TFIDF = ms.createTfidf(train_corpus,train_corpus,2)
test_TFIDF = ms.createTfidf(test_corpus,train_corpus,2)

#data_train = np.concatenate((train_TFIDF, train_LX), axis=1)
#data_test = np.concatenate((test_TFIDF, test_LX), axis=1)

#np.save("Model/TT/train_LX lexicon solo pattern.npy",train_LX)
#np.save("Model/TT/test_LX lexicon solo pattern.npy",test_LX)
np.save("Model/TT/bigram train_TFIDF PP negation & lemma.npy",train_TFIDF)
np.save("Model/TT/bigram test_TFIDF PP negation & lemma.npy",test_TFIDF)
#np.save("Model/TT/Y_train.npy",Y)
#np.save("Model/TT/Y_test.npy",Y_test)
