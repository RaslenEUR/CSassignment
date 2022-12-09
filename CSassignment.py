#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 13:55:23 2022

@author: raslen
"""


import json
import pandas as pd
import numpy as np
import re
import random
from math import pi
from numpy import arccos
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import AgglomerativeClustering
from sympy import isprime
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score

data_path = '/Users/raslen/Documents/EUR/21:22/computer science/TVs-all-merged.json'

#%% 1: Importing the data
with open(data_path) as json_file:
    data_orig = json.load(json_file)
values = list(data_orig.values())
data = [values[i][j] for i in range(len(data_orig)) for j in range(len(values[i]))]

# Title and key-values lists
n_prod = len(data)
title_lst = [prod['title'] for prod in data]
webshop_lst = [prod['shop'] for prod in data]
modelID_lst = [prod['modelID'] for prod in data] # 1262 unique ids

modelID_lst_cleaned = modelID_lst.copy() # 1221 unique ids (cleaned 41)
for i in range(len(modelID_lst_cleaned)):
    modelID_lst_cleaned[i] = modelID_lst_cleaned[i].lower().replace('-', '')

TOTAL_N_DUPLICATES = len(modelID_lst) - len(set(modelID_lst))
TOTAL_N_DUPLICATES_CLEANED = len(modelID_lst) - len(set(modelID_lst_cleaned))

key_values_lst = [data[i]['featuresMap'] for i in range(n_prod)]
key_values_lst = [[*zip(*[d.values()])] for d in key_values_lst]
key_values_lst = [" ".join(["".join(i) for i in key_values_lst[j]]) for j in range(n_prod)]


featuresMap_lst = [data[i]['featuresMap'] for i in range(n_prod)]
# keys_set = set()
keys_lst = []
values_lst = []
for features in featuresMap_lst:
    #keys_set |= set(features.keys())
    keys_lst.extend(list(features.keys()))
    values_lst.extend(list(features.values()))

df_features = pd.DataFrame(data=zip(keys_lst, values_lst), columns=['keys','values'])
  
# keys referring to 'brand name'
word = 'brand'  
word_synonyms = [word]
for keys in set(keys_lst):
    if word in keys.lower():
        word_synonyms.append(keys)
df_features['keys'].replace(word_synonyms, word, inplace=True)
df_features['values'] = df_features['values'].str.lower()
brand_set = set(df_features[df_features['keys']==word]['values'].tolist())

brand_lst = []
for prod_idx in range(n_prod):
    brand_known = False
    features = featuresMap_lst[prod_idx]
    for keys in features.keys():
        if keys in word_synonyms:
            brand_lst.append(features[keys].lower())
            brand_known = True
            break # assuming only one brand feature per product
    if not(brand_known):
        for brand in brand_set:
            if re.search(f'{brand}[^a-z]', title_lst[prod_idx]): # assuming brandname to be followed by any non-alphabetical character to prevent matchings of brandname abbreviations in words
                brand_lst.append(brand)
                brand_known = True
                print(f'product {prod_idx} is from {brand}')
                break
        if not(brand_known):
            brand_lst.append(None)
    
# end test
#%% 2: Cleaning the data  
mapping_normalize = {
    'Inch'      :'inch',
    'inches'    :'inch',
    ' inches'   :'inch',
    '"'         :'inch',
    '”'         :'inch',
    "'"         :'inch',
    '-inch'     :'inch',
    ' inch'     :'inch',
    'Hertz'     :'hz',
    'hertz'     :'hz',
    'Hz'        :'hz',
    'HZ'        :'hz',
    ' hz'       :'hz',
    '-hz'       :'hz',
    '-'         :'' # new
    }

# normalize units
rep=mapping_normalize
rep = dict((re.escape(k), v) for k, v in rep.items()) 
pattern = re.compile('|'.join(rep.keys()))
title_lst = [pattern.sub(lambda m: rep[re.escape(m.group(0))], sub) for sub in title_lst]
key_values_lst = [pattern.sub(lambda m: rep[re.escape(m.group(0))], sub) for sub in key_values_lst]

# convert uppercase to lowercase
title_lst = list(map(lambda x: x.lower(), title_lst))
key_values_lst = list(map(lambda x: x.lower(), key_values_lst))

#%% 3: Extracting model words
# model words from title
vectorizer_title = CountVectorizer(token_pattern=r"([a-zA-Z0-9]*(?:(?:[0-9]+[^0-9^,^ ^(^)^/^[^]+)|(?:[^0-9^,^ ^(^)^/^[^]+[0-9]+))[a-zA-Z0-9]*)",
                                   binary=True)
X = vectorizer_title.fit_transform(title_lst)
X = X.toarray().astype('int8') # dim_products x dim_features
model_words_title = vectorizer_title.get_feature_names()
X = pd.DataFrame(X, columns=model_words_title)

# model words from key-value pairs - all decimal numbers
vectorizer_key_values = CountVectorizer(token_pattern=r"(?:[0-9]+\.+[0-9]+)",
                                        binary=True)
Y = vectorizer_key_values.fit_transform(key_values_lst)
Y = Y.toarray().astype('int8') # dim_products x dim_features
model_words_key_values = vectorizer_key_values.get_feature_names()
Y = pd.DataFrame(Y, columns=model_words_key_values)

# model_words_title in key-values
P = vectorizer_title.transform(key_values_lst)
P = P.toarray().astype('int8')
P = pd.DataFrame(P, columns=model_words_title)

# model_words_key_values in title
Q = vectorizer_key_values.transform(title_lst)
Q = Q.toarray().astype('int8')
Q = pd.DataFrame(Q, columns=model_words_key_values)

binary_token_matrix = pd.concat([X,Y,P,Q], axis=0).fillna(0)
binary_token_matrix = binary_token_matrix.groupby(binary_token_matrix.index).sum().astype('int8')
binary_token_matrix = binary_token_matrix.mask(binary_token_matrix>0,1).T
n_mw = binary_token_matrix.shape[0]

#%% 4: Minhash Signatures

# Hash functions [(a+bx)mod(p)]+1 --> +1 for easier allocation to buckets
random.seed(0)

n_hashfunc = int(n_mw/2) # same as Hartveld et al.
while isprime(n_hashfunc):
    n_hashfunc += 1 #otherwise no way to satisfy n=r*b
    
a_lst = [*range(1,n_hashfunc+1)]
b_lst = random.sample(range(1,10*n_hashfunc), n_hashfunc)

primes_lst = [i for i in range(n_hashfunc+1, 2*n_prod) if isprime(i)]
p_lst = [random.choice(primes_lst)]*n_hashfunc

hashfunc_lst = [(a,b,p) for a,b,p in zip(a_lst, b_lst, p_lst)]

largest_int = 1+p_lst[0]

signature_mat = np.empty((0, n_prod), dtype=int)
ones_vec = np.ones((1, n_prod),dtype=int)
selection_mat = binary_token_matrix.copy().astype(int).replace(0, largest_int).to_numpy()
for i in range(n_hashfunc):
    a_i, b_i, p_i = hashfunc_lst[i]
    hashfunc_i_values = np.matrix([((a_i+b_i*x) % p_i)+1 for x in range(1, n_mw+1)])
    mat_i = np.matmul(hashfunc_i_values.T, ones_vec)
    signature_mat = np.vstack((signature_mat, np.multiply(mat_i, selection_mat).min(axis=0)))

#%% 5: LSH
# same hash function as van Dam, Iris et al.
M = pd.DataFrame(signature_mat).astype(str)
r_lst = [r for r in range(5,int(n_hashfunc/2)) if (n_hashfunc%r==0)]
b_lst = [int(n_hashfunc/r) for r in r_lst]
t_lst = [(1/b)**(1/r) for b,r in zip(b_lst, r_lst)]

idx = 17
r, b, t = r_lst[idx], b_lst[idx], t_lst[idx] 

M = M.groupby(M.index // r).agg(''.join) # hash to buckets
prod_series = pd.Series(M.columns)
candidate_lst = []
for band in range(b):
    temp = prod_series.groupby(M.loc[band,:], sort=False).apply(list).tolist() #grouping prods with same value in same bucket (list)
    candidate_lst.extend([set(t) for t in temp if len(t)>1]) # adding buckets of candidate neighbors

# all candidate pairs
candidate_pair_lst = set([item for i in candidate_lst for item in list(combinations(i, 2))])

def calc_norm_polar_distance(token_matrix):
    similarity_matrix = cosine_similarity(token_matrix)
    similarity_matrix[similarity_matrix>1]=1 # fixing potential rounding errors
    similarity_matrix[similarity_matrix<0]=0
    distance_matrix = (1/pi)*arccos(similarity_matrix) # normalized polar distance, used by Wang, motivated using Houle #TODO
    return distance_matrix

d_mat = calc_norm_polar_distance(binary_token_matrix.T)

# set of all product pairs
set_allpairs = set([frozenset(pair) for pair in list(combinations([*range(n_prod)], 2))])

# set of candidate-pairs from LSH
set_candidatepairs = set([frozenset(item) for i in candidate_lst for item in list(combinations(i, 2))])

# set of pairs from (possibly) same brand
brand_unknown_lst = [idx for idx in range(n_prod) if brand_lst[idx] is None]
brand_bucket_lst = prod_series.groupby(brand_lst, sort=False).apply(list).tolist()
for lst in brand_bucket_lst:
    lst.extend(brand_unknown_lst)
    #add unknown brand prods to every brandbucket
set_brandpairs = set([frozenset(pair) for bucket in brand_bucket_lst for pair in list(combinations(bucket, 2))])

# set of pairs from same webshop
webshop_bucket_lst = prod_series.groupby(webshop_lst, sort=False).apply(list).tolist()
set_webshoppairs = set([frozenset(pair) for bucket in webshop_bucket_lst for pair in list(combinations(bucket, 2))])

# set of pairs to consider
set_pairs_to_consider = set_candidatepairs.intersection(set_brandpairs)
#set_pairs_to_consider -= set_webshoppairs

# set distance of disregarded pairs to high value
# set of pairs to disregard
set_disregardedpairs = set_allpairs - set_pairs_to_consider
for i,j in set_disregardedpairs:
    d_mat[i,j] = 999999
    
#%% Performance analysis
    



for i in range(len(set_pairs_to_consider)):
    jaccard_sim=[]
    list1_idx=list(list(set_pairs_to_consider)[i])[0]
    list2_idx=list(list(set_pairs_to_consider)[i])[1]
    nom=len(set(signature_mat[:,list1_idx] & set(signature_mat[:,list2_idx])))
    denom=len(set(signature_mat[:,list1_idx] | set(signature_mat[:,list2_idx])))
    temp=nom/denom
    if temp>0.5:
        jaccard_sim.append(list(set_pairs_to_consider)[i])


