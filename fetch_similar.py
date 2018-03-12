import re
import string
import os,sys
import json
import io
import gensim

skus = input("Please enter the sku_ids for which you want to find similar items separated by commas ").split(",")
n_items = input("Please enter the number of similar items separated by commas ").split(",")

similar_model = gensim.models.Doc2Vec.load('old_navy_similar_items.doc2vec')
#results = {}

for i in range(len(n_items)):
    try:
        #results[skus[i]] = list(zip(*similar_model.docvecs.most_similar(skus[i],topn=n_items[i])))[0]
        print("The items similar to " + skus[i] + " are :" + '\n' + '\n'.join(list(zip(*similar_model.docvecs.most_similar(skus[i],topn=int(n_items[i]))))[0]))
    except KeyError:
        print(" The given sku is invalid")
    finally:
        pass

