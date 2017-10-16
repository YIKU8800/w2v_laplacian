# -*- coding: utf-8 -*-
"""
Created on Fri May  2 08:27:25 2014

@author: kurita
"""

import numpy as np
#np.set_printoptions(threshold=np.inf)
import matplotlib.pyplot as plt
import MySU3
import csv
from gensim.models import Word2Vec
#from gensim.models import KeyedVectors
import scipy

print "loading word2vec model..."

model = Word2Vec.load('en_1000_no_stem/en.model')
#model = KeyedVectors.load_word2vec_format("wiki_model/wiki.en.text.vector", binary=False)

print "loading completed..."

word='sunset'
dim = 1000

data=np.load('y_train.npy')
print 'data\n', data

label=[]
index={}
#f=open("corel5k_words_modify.txt")
f=open("corel5k_words_modify_1000.txt")
line=f.read()
f.close()
line=line.split('\n')
del line[260]
for i in xrange(len(line)):
    label.append(line[i])
    index[line[i]]=i

beta = 0.03

def calc_similarity(w1, w2):
    dis = (1 - model.similarity(w1, w2)) / 2
    sim = np.exp(-beta * np.power(dis, 1))
    if(w1 == w2):
        sim = 1.0
    #if(sim < 0.000001):
        #sim = 0
    return sim

'''
def calc_similarity(w1, w2):
    #sim = model.similarity(w1,w2)
    #sim = sim * 0.5 + 0.5
    dis = (1 - model.similarity(w1,w2)) / 2
    sim = np.exp(-beta * dis)
    #sim = np.exp(beta * sim)
    return sim
'''

similarity=np.zeros(260)
for i in xrange(260):
    similarity[i] = calc_similarity(word, label[i])
    similarity=np.array(similarity)


count=0

for i in similarity.argsort():
    if count==0:
        print label[i],i
    if count>240:
        print label[i],similarity[i],i
    count+=1

similarity=np.ones([260,260])
for i in xrange(260):
    for j in xrange(260):
	similarity[i][j] = calc_similarity(label[i], label[j])
print similarity

np.save('./../L/WL_%d.npy' % (dim), similarity)
