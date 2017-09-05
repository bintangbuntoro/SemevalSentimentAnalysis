# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 22:01:48 2017

@author: Bintang Buntoro
"""

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import Polarity
import Features
import re
import string
import nltk

def getStopWordList(stopWordFileName):
    #Membaca file stopword dan membuatnya menjadi list
    stopWords = []
    stopWords.append('at_user')
    stopWords.append('url')
    
    fp = open(stopWordFileName, 'r')
    line = fp.readline()
    while line:
        word = line.strip()
        stopWords.append(word)
        line = fp.readline()
    fp.close()
    return stopWords

def loadSlangs(slangsFileName):
    slangs = {}
    fi = open(slangsFileName,'r')
    line = fi.readline()
    while line:
        l = line.split(r',%,')
        if len(l) == 2:
            slangs[l[0]] = l[1][:-1]
        line = fi.readline()
    fi.close()
    return slangs

def slangWord(tweet):
    result = ''
    words = tweet.split()
    for w in words:
        if w in slangs.keys():
            result=result+slangs[w]+" "
        else:
            result=result+w+" "
    return result

def negation_tag(sentence):
    transformed = re.sub(r'\b(?:not|never|no|none|nothing|nowhere|noone|no|nobody|nowhere|neither|no one|nothing|hardly|scarcely|barely)\b[\w\s]+[^\w\s?]', 
       lambda match: re.sub(r'(\s+)(\w+)', r'\1NEG_\2', match.group(0)), 
       sentence,
       flags=re.IGNORECASE)
    
    return transformed


def removeUrl(text):
    link_regex    = re.compile('((https?):((//)|(\\\\))+([\w\d:#@%/;$()~?\+-=\\\.&](#!)?)*)', re.DOTALL)
    links         = re.findall(link_regex, text)
    for link in links:
        text = text.replace(link[0], ', ')
    text = re.sub("\.\.\.", "", text)
    return text

def removePunctuation(text):
    remove = string.punctuation
    remove = remove.replace("_", "") # don't remove hyphens
    pattern = r"[{}]".format(remove) # create the pattern

    remoPunct = re.sub(pattern, "", text)
    return remoPunct

def removeStopword(words):
    processedTweet = ''
    for w in words:
        #Strip punctuation
        if w in stopWords:
            None
        else:
            w = w.replace('''"''', ''' ''')
            processedTweet = processedTweet+w+' '
    return processedTweet

def lemmatize(line):
    lemma = nltk.wordnet.WordNetLemmatizer()
    line = line.split()
    sentence = []
    for word in line:
        temp = lemma.lemmatize(word)
        if temp == word:
            temp = lemma.lemmatize(word,'v')
        sentence.append(temp)
    sentence = ' '.join(sentence)
    return sentence

def preprocessing(line):
    line = slangWord(line)
    line = negation_tag(line)
    line = removeUrl(line)
    line = removePunctuation(line)
    line = line.lower()
    line = line.split()
    line = removeStopword(line)
    line = lemmatize(line)
    
    return line

def createCorpus(posFileName, negFileName):
    print "Process Create Corpus"
    listCorpus = []
    f = open(posFileName, 'r')
    line = f.readline()
    while line:
        try:
            line = preprocessing(line)
            listCorpus.append(line)
        except:
            None
        line = f.readline()
    f.close()

    f = open(negFileName, 'r')
    line = f.readline()

    while line:
        try:
            line = preprocessing(line)
            listCorpus.append(line)
        except:
            None
        line = f.readline()
    f.close()

    return listCorpus

def createTfidf(data,corpus,ngram):
    print "Process TF-IDF"
    count_vectorizer = CountVectorizer(ngram_range=(ngram,ngram),token_pattern=r'\b\w+\b', min_df=1)
    count_vectorizer.fit_transform(corpus)

    # Data Train
    freq_term_matrix_train = count_vectorizer.transform(data)
    TF_train = freq_term_matrix_train.todense()

    tfidf_train = TfidfTransformer(norm="l2")
    tfidf_train.fit(TF_train)

    tf_idf_matrix_train = tfidf_train.transform(freq_term_matrix_train)
    TFIDF_train = tf_idf_matrix_train.todense()
    TFIDF_train = np.asarray(TFIDF_train)
    
    return TFIDF_train

def tweetFeature(posFileName, negFileName, poslabel, neglabel):
    print "Process Tweet Features..."
    lexiScore = []
    labels = []
    f = open(posFileName, 'r')
    line = f.readline()
    while line:
        try:
            y = mapTweetLexicon(line, emoticonDict, slangs)
            lexiScore.append(y)
            labels.append(float(poslabel))
        except:
            None
        line = f.readline()
    f.close()

    f = open(negFileName, 'r')
    line = f.readline()

    while line:
        try:
            y = mapTweetLexicon(line, emoticonDict, slangs)
            lexiScore.append(y)
            labels.append(float(neglabel))
        except:
            None
        line = f.readline()
    f.close()
    print("Loading done...")
    return lexiScore, labels


def mapTweetLexicon(tweet, emoDict, slangs):
    out = []
    line = slangWord(tweet)
    line = removeUrl(line)
    line = removePunctuation(line)
    line = line.lower()
    line = line.split()
    line = removeStopword(line)
    afinnValue = Polarity.afinnPolarity(line, afinn)
    out.append(afinnValue)
    sentiWordValue = Polarity.sentiPolarity(line,sentiWordnet)
    out.extend(sentiWordValue)
    patternValue = Polarity.patternPolarity(line)
    out.append(patternValue)
    return out

print "Inisialisasi Dictionary"
stopWords = getStopWordList('Resources/stopWords.txt')
slangs = loadSlangs('Resources/internetSlangs.txt')
emoticonDict = Features.createEmoticonDictionary("Resources/emoticon.txt")
#Load Lexicon
afinn = Polarity.loadAfinn('Resources/afinn.txt')
sentiWordnet = Polarity.loadSentiWordnet('Resources/36694_SentiWordNet_3.0.0.txt')
