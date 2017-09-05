# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 17:26:19 2017

@author: Bintang Buntoro
"""

def createEmoticonDictionary(emotFileName):
    emo_scores = {'Positive': 1.0, 'Extremely-Positive': 1.0, 'Negative':-1.0,'Extremely-Negative': -1.0,'Neutral': 0.0}
    listEmoScores = {}
    fi = open(emotFileName,"r")
    l = fi.readline()
    
    while l:
        l = l.replace("\xc2\xa0"," ")
        li = l.split(" ")
        l2 = li[:-1]
        l2.append(li[len(li)-1].split("\t")[0])
        sentiment = li[len(li)-1].split("\t")[1][:-1]
        score = emo_scores[sentiment]
        l2.append(score)
        for i in range(0,len(l2)-1):
            listEmoScores[l2[i]]=l2[len(l2)-1]
        l = fi.readline()
    return listEmoScores

def emoticonScore(tweet,d):
    #Menghitung keseluruhan score untuk setiap emoticon pada tweet
    s = 0.0;
    l = tweet.split(" ")
    nbr = 0;
    for i in range(0,len(l)):
        if l[i] in d.keys():
            nbr = nbr+1
            s = s+d[l[i]]
    if (nbr!=0):
        s = s/nbr
    return s

def hashtagWords(tweet):
    result = []
    for w in tweet.split():
        if w[0] == '#' :
            result.append(w)
            
    return result

def upperCase(tweet):
    result = 0
    for w in tweet.split():
        if w.isupper():
            result = 1
    return result

def exclamationTest(tweet):
    result = 0
    if("!" in tweet):
        result = 1
    return result

def questionTest(tweet):
    result = 0
    if("?" in tweet):
        result = 1
    return result

def freqCapital(tweet): # ratio of number of capitalized letters to the length of tweet
    count=0
    for c in tweet:
        if (str(c).isupper()):
            count=count+1
    if len(tweet)==0:
        return 0
    else:
        return count/len(tweet)    
    
def scoreUnigram(tweet,posuni,neguni):
    pos=0
    neg=0
    l=len(tweet.split())

    for w in tweet.split():
        if w in posuni:
            pos+=1
        if w in neguni:
            neg+=1
    if (l!=0) :
        pos=pos/l
        neg=neg/l
    return [pos,neg]
