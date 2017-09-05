# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 21:58:24 2017

@author: BintangBuntoro
"""

import numpy as np
from sklearn import svm
import F1ScoreCalculator

def Skenario_1(temp):
    print "=== SKENARIO 1 ==="
    print "Pengujian Preprocessing"
    train_TFIDF = []
    test_TFIDF = []
    train_LX = np.load("Model/train_LX lexicon all.npy")
    test_LX = np.load("Model/test_LX lexicon all.npy")
    train_Y = np.load("Model/Y_train.npy")
    test_Y = np.load("Model/Y_test.npy")
    verbose = True
    Y_pred = 0
    acc = 0
    f1 = 0
    if temp == 1:
        print "All Preprocessing"
        print "Load Dataset"
        train_TFIDF = np.load("Model/train_TFIDF PP all.npy")
        test_TFIDF = np.load("Model/test_TFIDF PP all.npy")
    elif temp == 2:
        print "Stopword & Lemma"
        print "Load Dataset"
        train_TFIDF = np.load("Model/train_TFIDF PP stopword & lemma.npy")
        test_TFIDF = np.load("Model/test_TFIDF PP stopword & lemma.npy")
    elif temp == 3:
        print "Tag Negation & Stopword"
        print "Load Dataset"
        train_TFIDF = np.load("Model/train_TFIDF PP negation & stopword.npy")
        test_TFIDF = np.load("Model/test_TFIDF PP negation & stopword.npy")
    elif temp == 4:
        print "Tag Negation & Lemma"
        print "Load Dataset"
        train_TFIDF = np.load("Model/train_TFIDF PP negation & lemma.npy")
        test_TFIDF = np.load("Model/test_TFIDF PP negation & lemma.npy")
    else:
        print "Invalid Menu"
        verbose = False
    
    if verbose:
        data_train = np.concatenate((train_TFIDF, train_LX), axis=1)
        data_test = np.concatenate((test_TFIDF, test_LX), axis=1)
        print('Modelling SVM...')
        svmClassifier = svm.LinearSVC()
        svmClassifier = svmClassifier.fit(data_train,train_Y)
        Y_pred = svmClassifier.predict(data_test)
        acc, f1 = F1ScoreCalculator.fmeasure(test_Y,Y_pred)
        print "Akurasi = ",acc
        
    return Y_pred,acc,f1

def Skenario_2(temp):
    print "=== SKENARIO 2 ==="
    print "Pengujian Lexicon"
    train_TFIDF = np.load("Model/train_TFIDF PP negation & lemma.npy")
    test_TFIDF = np.load("Model/test_TFIDF PP negation & lemma.npy")
    train_LX = []
    test_LX = []
    train_Y = np.load("Model/Y_train.npy")
    test_Y = np.load("Model/Y_test.npy")
    verbose = True
    Y_pred = 0
    acc = 0
    f1 = 0
    if temp == 1:
        print "Afinn & Pattern"
        print "Load Dataset"
        train_LX = np.load("Model/train_LX lexicon afinn & pattern.npy")
        test_LX = np.load("Model/test_LX lexicon afinn & pattern.npy")
    elif temp == 2:
        print "Afinn & Sentiword"
        print "Load Dataset"
        train_LX = np.load("Model/train_LX lexicon afinn & sentiword.npy")
        test_LX = np.load("Model/test_LX lexicon afinn & sentiword.npy")
    elif temp == 3:
        print "Pattern & Sentiword"
        print "Load Dataset"
        train_LX = np.load("Model/train_LX lexicon pattern & sentiword.npy")
        test_LX = np.load("Model/test_LX lexicon pattern & sentiword.npy")
    elif temp == 4:
        print "All Lexicon"
        print "Load Dataset"
        train_LX = np.load("Model/train_LX lexicon all.npy")
        test_LX = np.load("Model/test_LX lexicon all.npy")
    elif temp == 5:
        print "Afinn"
        print "Load Dataset"
        train_LX = np.load("Model/train_LX lexicon solo afinn.npy")
        test_LX = np.load("Model/test_LX lexicon solo afinn.npy")
    elif temp == 6:
        print "Pattern"
        print "Load Dataset"
        train_LX = np.load("Model/train_LX lexicon solo pattern.npy")
        test_LX = np.load("Model/test_LX lexicon solo pattern.npy")
    elif temp == 7:
        print "Sentiword"
        print "Load Dataset"
        train_LX = np.load("Model/train_LX lexicon solo sentiword.npy")
        test_LX = np.load("Model/test_LX lexicon solo sentiword.npy")
    else:
        print "Invalid Menu"
        verbose = False
    
    if verbose:
        data_train = np.concatenate((train_TFIDF, train_LX), axis=1)
        data_test = np.concatenate((test_TFIDF, test_LX), axis=1)
        print('Modelling SVM...')
        svmClassifier = svm.LinearSVC()
        svmClassifier = svmClassifier.fit(data_train,train_Y)
        Y_pred = svmClassifier.predict(data_test)
        acc, f1 = F1ScoreCalculator.fmeasure(test_Y,Y_pred)
        print "Akurasi = ",acc
        
    return Y_pred,acc,f1

def Skenario_3(temp):
    print "=== SKENARIO 3 ==="
    print "Pengujian N-Gram"
    print "Load Dataset"
    train_TFIDF = np.load("Model/train_TFIDF PP negation & lemma.npy")
    test_TFIDF = np.load("Model/test_TFIDF PP negation & lemma.npy")
    train_LX = np.load("Model/train_LX lexicon all.npy")
    test_LX = np.load("Model/test_LX lexicon all.npy")
    train_Y = np.load("Model/Y_train.npy")
    test_Y = np.load("Model/Y_test.npy")
    data_train = []
    data_test = []
    verbose = True
    Y_pred = 0
    acc = 0
    f1 = 0
    if temp == 1:
        print "Unigram + Lexicon"
        data_train = np.concatenate((train_TFIDF, train_LX), axis=1)
        data_test = np.concatenate((test_TFIDF, test_LX), axis=1)
    elif temp == 2:
        print "Bigram + Lexicon"
        data_train = np.load("Model/bigram train_TFIDF PP negation & lemma.npy")
        data_test = np.load("Model/bigram test_TFIDF PP negation & lemma.npy")
        data_train = np.concatenate((data_train, train_LX), axis=1)
        data_test = np.concatenate((data_test, test_LX), axis=1)
    elif temp == 3:
        print "Unigram"
        data_train = train_TFIDF
        data_test = test_TFIDF
    elif temp == 4:
        print "Bigram"
        data_train = np.load("Model/bigram train_TFIDF PP negation & lemma.npy")
        data_test = np.load("Model/bigram test_TFIDF PP negation & lemma.npy")
    else:
        print "Invalid Menu"
        verbose = False
    
    if verbose:
        print('Modelling SVM...')
        svmClassifier = svm.LinearSVC()
        svmClassifier = svmClassifier.fit(data_train,train_Y)
        Y_pred = svmClassifier.predict(data_test)
        acc, f1 = F1ScoreCalculator.fmeasure(test_Y,Y_pred)
        print "Akurasi = ",acc
        
    return Y_pred,acc,f1

Y_pred,acc,f1 = Skenario_3(4)