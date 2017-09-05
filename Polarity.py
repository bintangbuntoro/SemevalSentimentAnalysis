# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 17:16:38 2017

@author: Bintang Buntoro
"""
from pattern.en import sentiment

def hashTest(word):
    return word[0]=='#'

def loadAfinn(afinnFileName):
    f = open(afinnFileName, 'r')
    afinn = {}
    line = f.readline()
    nbr = 0
    while line:
        nbr+=1
        l = line[:-1].split('\t')
        afinn[l[0]] = float(l[1])/4 #Normalisasi
        line = f.readline()
        
    return afinn

def afinnPolarity(tweet, afinn):
    p = 0.0
    nbr = 0
    for w in tweet.split():
        if w in afinn.keys():
            nbr+=1
            p+=afinn[w]
    if (nbr != 0):
        return p/nbr
    else:
        return 0.0
    
def loadSentiWordnet(filename): # need fixing , use loadSentiFull instead 
    output={}
    print "Opening SentiWordnet file..."
    fi=open(filename,"r")
    line=fi.readline() # skip the first header line
    line=fi.readline()
    print "Loading..."

    while line:
        l=line.split('\t')
        try:
            sentence=l[4]
            new = [word for word in sentence.split() if (word[-2] == "#" and word[-1].isdigit())]
            pos=abs(float(l[2]))
            neg=abs(float(l[3]))
        except:
#            print line
            line=fi.readline()
            continue

        for w in new:
            output[(w[:-2])]=[pos,neg] # dict(word,tag)=scores
        line=fi.readline()
    fi.close()
    return output

def sentiPolarity(tweet,sentDict): # polarity aggregate of a tweet from sentiWordnet dict
    pos=0.0
    neg=0.0
    n_words=0
    for w in tweet.split():
        if w in sentDict.keys():
            n_words=n_words+1
            pos=pos+sentDict[w][0]
            neg=neg+sentDict[w][1]
            
    if (n_words ==0 ):
        return [pos-neg]
    else:
        return [(pos-neg)/n_words]
                
def patternPolarity(tweet):
    polarity = sentiment(tweet)[0]
    return polarity
        