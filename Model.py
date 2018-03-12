import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import gensim
from gensim.models.doc2vec import TaggedDocument
import sklearn
import os,sys
import numpy as np
import pandas as pd
import string
import re
import json
import multiprocessing

regex = re.compile('[%s]' % re.escape(string.punctuation))
stemming = PorterStemmer()

class TaggedLineDocument(object):
    def __init__(self,file_name):
        self.file_name = file_name
    def __iter__(self):
        with open(self.file_name,'r') as myfile:
            next(myfile)
            for line in myfile:
                s = regex.sub('',line.split('\t')[5][:-8].lower().replace("old navy","")) + " " + regex.sub('',line.split('\t')[6].lower().replace("old navy",""))
                yield TaggedDocument(words = [w for w in nltk.wordpunct_tokenize(s) if w not in [set(stopwords.words('english'))]],tags = [line.split('\t')[7]])
#data_doc = TaggedDocument(data,tag)
l = 0
des = TaggedLineDocument("Old_Navy-Product_Catalog.txt")
for d in des:
    l = l + 1
    if l > 10:
        break
    print(d[0])

description_dict = {}
for d in des:
    description_dict[d[1][0]] = d[0]
doc_model = gensim.models.Doc2Vec(documents=des,vector_size= 200, window_size = 5, min_count = 1, alpha = 0.025,min_alpha = 0.01,sample = 0,workers=multiprocessing.cpu_count(),train_lbls = False,dm=1,seed=100,dm_concat=0,dm_mean=0)
#doc_model.train(des,epochs=10)
doc_model.save('old_navy_similar_items.doc2vec')
result = {}
doc_model.most_similar()
for k in description_dict:
    result[k] = doc_model.most_similar([doc_model.infer_vector(description_dict[k])],topn=len(description_dict))

with open("similar_index.json","w") as fp:
    json.dump(result,fp)

