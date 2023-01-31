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
    
#%% Check whether candidate pairs are true duplicates if similarity surpasses threshold


jaccard_similarities = []
    
for ic in candidate_pairs:
    # ic contains indexes of candidate pairs
    product_1 = sparse_matrix[ic[0]]
    product_2 = sparse_matrix[ic[1]]
    
    # Compute jaccard similarity
    union = list(set(product_1) | set(product_2))
    intersection = list(set(product_1) & set(product_2))
    jaccard = len(intersection) / len(union)
        
    # Save candidate pair and corresponding jaccard similarity
    jaccard_similarities.append([ic, jaccard]) 

jaccard_similarities = np.array(jaccard_similarities)

# Hypertune optimal percentage for Jaccard similarity (s.t. errors are minimized, for given condition: Duplicate = TRUE, if Jaccard >= percentage)
required_jaccard = list(range(0, 101))
percentage_tuning = []
for percentage in required_jaccard:
    # Classify corresponding duplicates using percentage threshold
    mask = jaccard_similarities[:, 1] >= (percentage/100)
    duplicates_classified = list(jaccard_similarities[mask, 0])
        
    # Get original indices of candidate pairs (source index in raw data unpacked)
    original_indices = []
    for pair in duplicates_classified:
        original_indices.append((bootstrap_source_indices[pair[0]], bootstrap_source_indices[pair[1]]))        
    duplicates_classified = original_indices      
    
    # Get performance
    correct = len(set(duplicates_true) & set(duplicates_classified))
    false_positives = len(set(duplicates_classified) - set(duplicates_true)) 
    false_negatives = len(set(duplicates_true) - set(duplicates_classified))
        
    performance_score = objective_function_jaccard(correct, false_positives, false_negatives, _weights_jaccard_objective)
    
    f1 = get_f1(correct, false_positives, false_negatives, duplicates_classified)
    
    # Store results
    percentage_tuning.append([percentage, correct, false_positives + false_negatives, false_positives, false_negatives, performance_score, f1])
  
# Format for interpretability
percentage_tuning_df = pd.DataFrame(percentage_tuning, columns = ["Percentage", "Correct", "Total Errors", "False positives", "False negatives", "Score", "F1"])
percentage_tuning_df = pd.DataFrame(percentage_tuning_df).set_index("Percentage", drop = True)

# Classify corresponding duplicates using best percentage (by score)
best_percentage = percentage_tuning_df["F1"].idxmax()
mask = jaccard_similarities[:, 1] >= best_percentage/100
final_duplicates_classified = jaccard_similarities[mask, 0]
    
# Get original indices of candidate pairs (source index in raw data unpacked)
original_indices = []
for pair in final_duplicates_classified:
    original_indices.append((bootstrap_source_indices[pair[0]], bootstrap_source_indices[pair[1]]))        
final_duplicates_classified = original_indices       

#
                
print(f"\n----------------------\nFinal performance:\n     - correct: {correct}\n     - incorrect: {false_positives + false_negatives} ({false_positives} FP & {false_negatives} FN)\n     - F1: {f1}")

# Track required computing time
time_spent = datetime.now() - start_time
computing_times["get_jaccard_similarity()"] = time_spent
return {"percentage_tuning" : percentage_tuning_df, "Duplicate_predictions" : {"Percentage" : best_percentage, "Predicted_duplicates" : list(final_duplicates_classified)}}




#%% 
    

def data_json_unpacked(data_json):
    data_json_unpacked = []
    for key, value in data_json.items():
        for element in value:
            data_json_unpacked.append(element)
    
    return data_json_unpacked    



def get_all_labels(data):
    # intialise empty list
    all_duplicate_pairs = []
    
    # Iterate through all products, if duplicates: assign a non-zero index (i) to the i'th pair
    i = 1
    for key, value in data.items():
        # all elements in i'th duplicate set get assigned same value i
        if len(value) > 1:
            for element in value:
                all_duplicate_pairs.append(i) 
            i += 1
        # Unique products get assigned a 0
        else:
            all_duplicate_pairs.append(0)
    
    # Count duplicate sets & items
    n_duplicate_items = len(all_duplicate_pairs) - all_duplicate_pairs.count(0)
    n_duplicate_sets = max(all_duplicate_pairs)
    
    # Allocate every duplicate set to a list (get indexes of every unique non-zero entry in list)
    all_duplicate_sets = []
    for i in range(1, n_duplicate_sets + 1):
        indices = [j for j, x in enumerate(all_duplicate_pairs) if x == i] 
        all_duplicate_sets.append(indices)
    
    # Get all duplicate pairs:
    all_duplicate_pairs = []
    for duplicate_set in all_duplicate_sets:
        all_combinations = list(itertools.combinations(duplicate_set, 2)) # gets all pairs in bucket
        all_duplicate_pairs.extend(all_combinations)
    n_duplicate_pairs = len(all_duplicate_pairs)
        
    print(f"Total number of dupicates:\n     - Sets: {n_duplicate_sets}\n     - Items: {n_duplicate_items}\n     - Pairs: {n_duplicate_pairs}")    
    return {"duplicate_sets" : all_duplicate_sets, "duplicate_pairs" : all_duplicate_pairs}



def get_bootstrap(data, number_of_samples, labels):
    """
    Get data split for single bootstrap
    """
    # Get all indices (of data_json_unpacked)
    indices = list(range(0, len(data)))

    # Get train and test split for this bootstrap
    indices_train = random.choices(indices, k = number_of_samples)    
    indices_train = list(set(indices_train))
    indices_test = list(set(indices) - set(indices_train))
    
    # Convert Indices to List with corresponding items (note that the new indices now don't directly point to the corresponding item, but only via the index lists above)
    data_train = [data[i] for i in indices_train]
    data_test = [data[i] for i in indices_test]
    
    duplicates_train = []
    duplicates_test = []
    
    for duplicate_pair in labels:
        if duplicate_pair[0] in indices_train and duplicate_pair[1] in indices_train:
            duplicates_train.append(duplicate_pair)
        elif duplicate_pair[0] in indices_test and duplicate_pair[1] in indices_test:
            duplicates_test.append(duplicate_pair)
    
    print(f"     - Number of items (Train): {len(indices_train)} with {len(duplicates_train)} duplicates pairs")    
    print(f"     - Number of items (Test): {len(indices_test)} with {len(duplicates_test)} duplicates pairs")
    return {"data" : data_train, "source_indices" : indices_train, "labels" : duplicates_train}, {"data" : data_test, "source_indices" : indices_test, "labels" : duplicates_test}



def get_all_titles(data):
    """
    Gets all title (doesn't remove duplicates)
    """
    # Loop through dictionary and add all titles to list
    all_titles = []
    for element in data:
            title = element["title"]
            all_titles.append(title)
        
    return all_titles



def get_all_substrings(data, filter_subs):
    """
    Gets all substrings in sub (removes duplicates)
    """
    all_sub = []
    
    # Loop through dictionary and add all substrings to list
    for element in data:
        title = element["title"]
        sub = title.split(" ")  
        all_sub.extend(sub)
    
    # Filters (likely) redundant strings to reduce computation 
    if filter_subs == True:
        # Remove punctuation 
        for punctuation in ["(", ")", "-", "'", '"', "/", ",", "]", "["]:
            all_sub = [sub.replace(punctuation, "") if punctuation in sub else sub for sub in all_sub]
        
        all_sub_filtered = []
        for sub in all_sub:
            # Capitalise all substrings
            sub = sub.upper()
            
            # preserve numbers
            if sub.isdigit() == True:
                all_sub_filtered.append(sub)            
            
            # remove short strings (which are not numbers)    
            elif len(sub) >= 2:
               all_sub_filtered.append(sub)
               
        all_sub = all_sub_filtered
               
    # Remove duplicates
    all_sub = list(set(all_sub))                
    # Sort alphabetically
    all_sub = sorted(all_sub)
        
    return all_sub



def get_feature_key_value(product):   
    """
    returns all feature key/value pairs (in a string) for the single product/function-input, and same output in list form
    """
    # Get (raw) feature key and value
    features = product["featuresMap"]
    keys   = list(features.keys())
    values = list(features.values())
    
    # Remove ":" to merge features that should've been the same category (by assumption)
    keys_filtered = []
    for feature in keys:                
        if ":" in feature:
            feature = feature.replace(":", "")  
        keys_filtered.append(feature)
    
    # Create strings in format: "('key', 'value')" to maintain both info on presence key and corresponding value
    feature_list = list(zip(keys_filtered, values))
    feature_list = [str(pair).upper() for pair in feature_list]
    feature_string = " ".join(feature_list)

    return feature_string, feature_list
    


def get_all_features(data):
    """
    Gets feature key-value pairs of all products and forms lists of all possible entries
    """
    all_features = []
    # Iterate through all products 
    for product in data:
        _, feature_list = get_feature_key_value(product)
        all_features.extend(feature_list)
    
    # Remove duplicates
    all_features = list(set(all_features))
    
    return all_features
  

 

def get_sparse_matrix(data, all_substrings):
    start_time = datetime.now()
    """
    Creates sparse matrix based on the substrings within each items title
    This matrix contains the indices (in list of all substrings) of the substrings that are present in the title
    """
    # Save all product contents
    sparse_matrix = []
    
    # Iterate through all products and check whether substrings are in title
    for product in data:
        sparse_vector = []

        # Get title
        title = product["title"]

        # Add 1 if substring in title, else 0 (save index of substring)
        i = 0
        for x in all_substrings:
            if x in title.upper():
                sparse_vector.append(i)
            i += 1
            
        sparse_matrix.append(sparse_vector)

    # Convert nested lists to DF
    sparse_matrix_df = pd.DataFrame(sparse_matrix).T
    
    # Track required computing time
    time_spent = datetime.now() - start_time
    computing_times["get_sparse_matrix()"] = time_spent
    
    return sparse_matrix, sparse_matrix_df



def get_signature_matrix(sparse_matrix, all_substrings, n_perm):
    """
    Constructs signature matrix, a compressed version of the data that aims to maintain the similarity amongst items
    """    
    start_time = datetime.now()
        
    # Create intial permutation (all_substrings should be the version that's used to form sparse matrix)
    initialisation = list(range(0, len(all_substrings)))
    permutation_vectors = []
    
    # n = number of (unique) permutation vectors
    for i in range(0, n_perm):
        permutation = random.sample(initialisation, len(initialisation))
        permutation_vectors.append(permutation)
    
    signature_matrix = []
    
    for product in sparse_matrix:
        # Initialise empty signature vector for every product
        signature_vector = []
        
       # Repeat n-times, for n distinct shuffled permutation vectors
        for vector in permutation_vectors:
 
            # Iterate through (shuffled) elements in permutation vector 
            for substring_index in vector:
                # Stop when index in permutation vector is found in sparse vector. 
                if substring_index in product:
                    # Add index of current element in permutation vector
                    match = vector.index(substring_index)
                    break
            
            # Store values
            signature_vector.append(match)
        signature_matrix.append(signature_vector)
    
    # Track required computing time
    time_spent = datetime.now() - start_time
    computing_times["get_signature_matrix()"] = time_spent
    
    return {"permutation_vectors" : permutation_vectors, "signature_matrix" : signature_matrix}



def get_lsh(signature_matrix, bootstrap_labels, bootstrap_source_indices, tuned_parameter):
    """
    applies LSH (locality-sensitivity hashing), with a signature matrix as input.
    """
    start_time = datetime.now()
    signature_matrix = np.array(signature_matrix) # easier to grab data from array type
    
    # number of permutations
    n_items = signature_matrix.shape[0]
    n_perm = signature_matrix.shape[1]
    
    bucket_list = []
    candidate_pair_matrix = np.zeros([n_items, n_items])
    
    # for all n permutations:
    for k in range(n_perm):
        # Each column corresponds to a permutation
        permutation = list(signature_matrix[:,k])
    
        # intialise empty buckets
        buckets = []
       
        # For all unique signatures (no need to check for all values, signatures only)
        all_signatures = sorted(set(permutation))
        for signature in all_signatures:
            # If the signature appears more than once, then we search all items of that signature to add to the same bucket
            if permutation.count(signature) >= 2:
                indices = [i for i, x in enumerate(permutation) if x == signature]
                buckets.append(indices)
        
        bucket_list.append(buckets)
        
        # empty array to fill
        candidate_pairs = np.zeros([n_items, n_items])
   
        # Iterate through all buckets and set dummy to 1, for item-pairs which share the same bucket
        for b in buckets:
            all_combinations = list(itertools.combinations(b, 2)) # gets all pairs in bucket
            for pair in all_combinations:
                candidate_pairs[pair[0], pair[1]] += 1 # set dummy to 1
        
        # Store all candidate pair qualifications of all permutations by summing
        candidate_pair_matrix += candidate_pairs
    
    # Tune frequency for Training (consider all values <= number of permutations) 
    all_selection_criteria = []
   
    for freq in range(1, _n_perm + 1):      

        # create list of all candidate pairs (using candidate pair matrix)
        candidate_pairs = np.where(candidate_pair_matrix >= freq)
        candidate_pairs = list(zip(candidate_pairs[0], candidate_pairs[1]))
        
        # Get original indices of candidate pairs (source index in raw data unpacked)
        original_indices = []
        for pair in candidate_pairs:
            original_indices.append((bootstrap_source_indices[pair[0]], bootstrap_source_indices[pair[1]]))        
        candidate_pairs = original_indices       

        # Get total number of candidate pairs (for corresponding freq)
        number_of_candidate_pairs =  len(candidate_pairs)
        
        # Check whether true duplicates are included in candidate pairs
        not_detected_pairs = set(bootstrap_labels) - set(candidate_pairs)
        not_detected_percentage = round(100 * (len(not_detected_pairs)/len(bootstrap_labels)), 2)
        
        number_of_detected = len(bootstrap_labels) - len(not_detected_pairs)
        number_of_not_detected = len(not_detected_pairs)
        
        selection_criteria = [freq, number_of_candidate_pairs, not_detected_percentage, number_of_detected, number_of_not_detected, len(bootstrap_labels), _n_perm]
        all_selection_criteria.append(selection_criteria)
        
        # Percentage of duplicates that we allow to not be included in the candidate pairs (will never be detected), minimizes number candidate pairs while satisfying condition
        if not_detected_percentage <= _acceptable_failure:
            selected_frequency = freq
        # If the lost observations are higher than threshold for every frequency
        elif freq == 1 and not_detected_percentage > _acceptable_failure: 
            selected_frequency = freq
            print("No frequency found for which threshold condition is satisfied")

    # Testing has a frequency that's already tuned
    if tuned_parameter != -1:
        selected_frequency = tuned_parameter

    # Get candidate pairs based on selected frequency
    selected_candidate_pairs = np.where(candidate_pair_matrix >= selected_frequency)
    selected_candidate_pairs = list(zip(selected_candidate_pairs[0], selected_candidate_pairs[1]))

    # Format for interpretability
    all_selection_criteria_df = pd.DataFrame(all_selection_criteria, columns = ["Frequency", "Candidate pairs", "Missed (%)", "Duplicates included", "Duplicates not included", "Total duplicates", "Total permutations"])
    all_selection_criteria_df = pd.DataFrame(all_selection_criteria_df).set_index("Frequency", drop = True)

    # Print selected frequency, raise error if selection was unsuccesful
    if 'selected_frequency' in locals():
        print(f"\nSelected: Frequency = {selected_frequency}\n     - candidate pairs: {all_selection_criteria[selected_frequency - 1][1]}\n     - Failed to detect: {all_selection_criteria[selected_frequency - 1][4]}/{len(bootstrap_labels)} ({all_selection_criteria[selected_frequency - 1][2]}%)")
    else:
        print(f"Tuning results:\n{all_selection_criteria_df}")
        raise  ValueError('Problem: most likely, the acceptable percentage was too strict. To fix: increase acceptable percentage or number of permutations. (note: percentage are not fractions, e.g. 10% = 10, not 0.1)')
    
    # Track required computing time
    time_spent = datetime.now() - start_time
    computing_times["get_lsh()"] = time_spent
    
    return {"buckets" : bucket_list, "candidate_pair_matrix" : candidate_pair_matrix, "frequency_tuning" : all_selection_criteria_df}, selected_candidate_pairs, selected_frequency



def get_sparse_matrix_full(data, all_features, all_substrings):
    """
    Creates sparse matrix based on the substrings within each items title
    This matrix contains the indices (in list of all substrings) of the substrings that are present in the title
    """
    start_time = datetime.now()
    # Save all product contents
    sparse_matrix = []
    
    # Iterate through all products and checks for present features 
    for product in data:
        sparse_vector = []
        
        # Get title
        title = product["title"]
        # Gets string containing all feature key-value pairs
        feature_string, _ = get_feature_key_value(product)

        # Add if substring in title or feature in featureMap (save index of substring)
        i = 0
        for x in all_features:
            if x in feature_string.upper():
                sparse_vector.append(i)
            i += 1
        # (Don't reset i to 0 in between checks)
        for y in all_substrings:
            if y in title.upper():
                sparse_vector.append(i)
            i += 1
            
        sparse_matrix.append(sparse_vector)

    # Convert nested lists to DF
    sparse_matrix_df = pd.DataFrame(sparse_matrix).T
    
    # Track required computing time
    time_spent = datetime.now() - start_time
    computing_times["get_sparse_matrix_full()"] = time_spent
    
    return sparse_matrix, sparse_matrix_df



def objective_function_jaccard(correct, false_positives, false_negatives, weights):
    """
    Objective function to determine optimal jaccard similarity percentage threshold
    -> maximizes a (weighted) score function that counts correct classifications and penalizes incorrect ones
    """
    w_fp = weights["false_positive"]    # weight false positive
    w_fn = 1 - w_fp                     # weight false negative
    w_c = weights["correct"]            # weight correct
    w_e = 1 - w_c                       # weight error (penalty on false classifications)

    # Normalise s.t. w_fp and w_fn sum to w_e
    w_fp_normalised = w_fp * (w_e / (w_fp + w_fn))
    w_fn_normalised = w_fn * (w_e / (w_fp + w_fn))
    
    # Score: (weighted) count of correct classifications, penalized by incorrect classifications)
    score = w_c * correct - ( w_fp_normalised * false_positives + w_fn_normalised * false_negatives )
    
    # Show coefficients for score function
    # print(f"{score} = {w_c}*correct - ({w_fp_normalised}*FP + {w_fn_normalised}*FN)")
    return score



def get_f1(correct, false_positives, false_negatives, candidates_pairs):
    """
    Performance metrics function that returns F1-score
    """
    recall      = correct / (correct + false_negatives) # aka completeness
    precision   = correct / len(candidates_pairs) # aka quality

    # F1 undefined for recall and precision of 0, denoted by 0
    if recall == 0 and precision == 0:
        return 0
    
    f1 = (2 * precision * recall) / (precision + recall) # weighted avg of precision & recall
    return f1
    


def get_jaccard_similarity(sparse_matrix, candidate_pairs, duplicates_true, bootstrap_source_indices):
    """ 
    Computes Jaccard similarity for all candidate pairs
    Classifies all pairs with jaccard similarity above a percentage as duplicates
    """
    start_time = datetime.now()
    jaccard_similarities = []
    
    for cp in candidate_pairs:
        # cp contains indexes of candidate pairs
        product_1 = sparse_matrix[cp[0]]
        product_2 = sparse_matrix[cp[1]]
        
        # Compute jaccard similarity
        union = list(set(product_1) | set(product_2))
        intersection = list(set(product_1) & set(product_2))
        jaccard = len(intersection) / len(union)
        
        # Save candidate pair and corresponding jaccard similarity
        jaccard_similarities.append([cp, jaccard]) 

    jaccard_similarities = np.array(jaccard_similarities)

    # Hypertune optimal percentage for Jaccard similarity (s.t. errors are minimized, for given condition: Duplicate = TRUE, if Jaccard >= percentage)
    required_jaccard = list(range(0, 101))
    percentage_tuning = []
    for percentage in required_jaccard:
        # Classify corresponding duplicates using percentage threshold
        mask = jaccard_similarities[:, 1] >= (percentage/100)
        duplicates_classified = list(jaccard_similarities[mask, 0])
        
        # Get original indices of candidate pairs (source index in raw data unpacked)
        original_indices = []
        for pair in duplicates_classified:
            original_indices.append((bootstrap_source_indices[pair[0]], bootstrap_source_indices[pair[1]]))        
        duplicates_classified = original_indices      
        
        # Get performance
        correct = len(set(duplicates_true) & set(duplicates_classified))
        false_positives = len(set(duplicates_classified) - set(duplicates_true)) 
        false_negatives = len(set(duplicates_true) - set(duplicates_classified))
        
        performance_score = objective_function_jaccard(correct, false_positives, false_negatives, _weights_jaccard_objective)
        
        f1 = get_f1(correct, false_positives, false_negatives, duplicates_classified)
        
        # Store results
        percentage_tuning.append([percentage, correct, false_positives + false_negatives, false_positives, false_negatives, performance_score, f1])
  
    # Format for interpretability
    percentage_tuning_df = pd.DataFrame(percentage_tuning, columns = ["Percentage", "Correct", "Total Errors", "False positives", "False negatives", "Score", "F1"])
    percentage_tuning_df = pd.DataFrame(percentage_tuning_df).set_index("Percentage", drop = True)

    # Classify corresponding duplicates using best percentage (by score)
    best_percentage = percentage_tuning_df["F1"].idxmax()
    mask = jaccard_similarities[:, 1] >= best_percentage/100
    final_duplicates_classified = jaccard_similarities[mask, 0]
    
    # Get original indices of candidate pairs (source index in raw data unpacked)
    original_indices = []
    for pair in final_duplicates_classified:
        original_indices.append((bootstrap_source_indices[pair[0]], bootstrap_source_indices[pair[1]]))        
    final_duplicates_classified = original_indices       
    
    # Get final performance
    correct = len(set(duplicates_true) & set(final_duplicates_classified))
    false_positives = len(set(final_duplicates_classified) - set(duplicates_true)) 
    false_negatives = len(set(duplicates_true) - set(final_duplicates_classified))

    performance_score = objective_function_jaccard(correct, false_positives, false_negatives, _weights_jaccard_objective)
    f1 = get_f1(correct, false_positives, false_negatives, final_duplicates_classified)
                    
    print(f"\n----------------------\nFinal performance:\n     - correct: {correct}\n     - incorrect: {false_positives + false_negatives} ({false_positives} FP & {false_negatives} FN)\n     - F1: {f1}")
    
    # Track required computing time
    time_spent = datetime.now() - start_time
    computing_times["get_jaccard_similarity()"] = time_spent
    return {"percentage_tuning" : percentage_tuning_df, "Duplicate_predictions" : {"Percentage" : best_percentage, "Predicted_duplicates" : list(final_duplicates_classified)}}



def performance_evaluation(output_jaccard, bootstrap, tuned_parameter, candidate_pairs):
    # Get classification performance metrics
    jaccard = output_jaccard["percentage_tuning"]
    
    # Get performance measures (in series)
    classification_score = jaccard["Score"]
    completeness = jaccard["Correct"] / len(bootstrap["labels"]) # recall
    quality = jaccard["Correct"] / len(candidate_pairs) # precision
    
    recall = completeness # Recall is a metric that quantifies the number of correct positive predictions made out of all positive predictions that could have been made.
    precision = quality # Precision evaluates the fraction of correct classified instances among the ones classified as positive
    F1 = (2 * precision * recall) / (precision + recall) # avg(precision, recall)
    
    # For training, get best % for every metric
    if tuned_parameter == -1:
        best_classification_score = [classification_score.idxmax(), classification_score.max()]
        best_completeness =         [completeness.idxmax(), completeness.max()]
        best_quality =              [quality.idxmax(), quality.max()]
        best_recall =               [recall.idxmax(), recall.max()]
        best_precision =            [precision.idxmax(), precision.max()]
        best_F1 =                   [F1.idxmax(), F1.max()]
    else:    
        best_classification_score = [tuned_parameter, classification_score[tuned_parameter]]
        best_completeness =         [tuned_parameter, completeness[tuned_parameter]]
        best_quality =              [tuned_parameter, quality[tuned_parameter]]
        best_recall =               [tuned_parameter, recall[tuned_parameter]]
        best_precision =            [tuned_parameter, precision[tuned_parameter]]
        best_F1 =                   [tuned_parameter, F1[tuned_parameter]]
    
    metrics = pd.Series(["classification_score", "completeness", "quality", "recall", "precision", "F1"], name = "Metric")
    performance_measures = pd.DataFrame([best_classification_score, best_completeness, best_quality, best_recall, best_precision, best_F1], columns = ["Percentage Jaccard", "Metric value"])
    performance_measures = pd.concat([metrics, performance_measures], axis = 1)
    
    print("\n", performance_measures)
    
    return performance_measures, best_F1[0]



def get_jaccard_similarity_duplicates(sparse_matrix, candidate_pairs):
    """ 
    Plots the jaccard similarities of all true duplicates
    """
    jaccard_similarities = []
    
    for cp in candidate_pairs:
        # cp contains indexes of candidate pairs
        product_1 = sparse_matrix[cp[0]]
        product_2 = sparse_matrix[cp[1]]
        
        # Compute jaccard similarity
        union = list(set(product_1) | set(product_2))
        intersection = list(set(product_1) & set(product_2))
        jaccard = len(intersection) / len(union)
        
        # Save candidate pair and corresponding jaccard similarity
        jaccard_similarities.append(jaccard) 

    sns.distplot(jaccard_similarities, hist = False, kde = True, bins = 5, kde_kws = {"shade" : True, "linewidth" : 1})
    plt.xlabel("Jaccard similarity")
    plt.ylabel("Density")
    plt.show()    
    
    
    
def get_jaccard_similarity_duplicates_and_candidates(sparse_matrix, true_duplicates, candidate_pair):
    """ 
    Plots the jaccard similarities of all true duplicates and candidates
    """
    td_jaccard_similarities = []
    
    for td in true_duplicates:
        # td contains indexes of candidate pairs
        product_1 = sparse_matrix[td[0]]
        product_2 = sparse_matrix[td[1]]
        
        # Compute jaccard similarity
        union = list(set(product_1) | set(product_2))
        intersection = list(set(product_1) & set(product_2))
        jaccard = len(intersection) / len(union)
        
        # Save candidate pair and corresponding jaccard similarity
        td_jaccard_similarities.append(jaccard) 
    
    cp_jaccard_similarities = []
    
    for cp in candidate_pairs:
        # cp contains indexes of candidate pairs
        product_1 = sparse_matrix[cp[0]]
        product_2 = sparse_matrix[cp[1]]
        
        # Compute jaccard similarity
        union = list(set(product_1) | set(product_2))
        intersection = list(set(product_1) & set(product_2))
        jaccard = len(intersection) / len(union)
        
        # Save candidate pair and corresponding jaccard similarity
        cp_jaccard_similarities.append(jaccard) 

    sns.distplot(cp_jaccard_similarities, hist = False, kde = True, bins = 5, kde_kws = {"shade" : True, "linewidth" : 1}, label = "Candidate")
    sns.distplot(td_jaccard_similarities, hist = False, kde = True, bins = 5, kde_kws = {"shade" : True, "linewidth" : 1}, label = "True")
    plt.xlabel("Jaccard similarity")
    plt.ylabel("Density")
    plt.show()    


  
def get_plot_lsh(bootstrap, lsh, performance_metrics):
    n_bootstrap =  len(bootstrap["data"])
    total_comparisons =  (n_bootstrap * (n_bootstrap - 1))/2
    
    lsh = lsh["frequency_tuning"]
    
    fraction_comparison = [x / total_comparisons for x in lsh["Candidate pairs"]]

    recall  = [1 - missed/100 for missed in lsh["Missed (%)"]] # pair completeness
    quality = [found / total_candidates for found, total_candidates in zip(lsh["Duplicates included"], lsh["Candidate pairs"])] # pair precision
    f1      = [(2 * r * q)/(r + q) for r, q in zip(recall, quality) if r != 0]
    fraction_comparison_f1 = fraction_comparison[:len(f1)]
       
    # Plot pair recall
    plt.plot(fraction_comparison, recall)
    plt.xlabel("Fraction of comparisons")
    plt.ylabel("Recall")
    plt.show()    
    
    # Plot pair quality
    plt.plot(fraction_comparison, quality)
    plt.xlabel("Fraction of comparisons")
    plt.ylabel("Quality")
    # plt.xlim(-0.01, 0.2)
    plt.show()    
    
    # Plot f1 
    plt.plot(fraction_comparison_f1, f1)
    plt.xlabel("Fraction of comparisons")
    plt.ylabel("F1")
    plt.show()
    
    return fraction_comparison, recall, quality



def get_plot_jaccard(bootstrap, jaccard, performance_metrics, candidate_pairs):
    n_bootstrap =  len(bootstrap["labels"])
    
    jaccard = jaccard["percentage_tuning"]
    thresholds = jaccard.index.values.tolist()

    candidate_pairs = len(candidate_pairs) 

    recall  = [correct / n_bootstrap for correct in jaccard["Correct"]] # pair completeness
    quality = [correct / candidate_pairs for correct in jaccard["Correct"]] # pair precision
    f1      = [(2 * r * q)/(r + q) for r, q in zip(recall, quality) if r != 0]
    thresholds_f1 = thresholds[:len(f1)]
    
    # Plot pair recall
    plt.plot(thresholds, recall)
    plt.xlabel("Jaccard thresholds")
    plt.ylabel("Recall")
    plt.show()    
    
    # Plot pair quality
    plt.plot(thresholds, quality)
    plt.xlabel("Jaccard thresholds")
    plt.ylabel("Quality")
    # plt.xlim(-0.01, 0.2)
    plt.show()    
    
    # Plot f1 
    plt.plot(thresholds_f1, f1)
    plt.xlabel("Jaccard thresholds")
    plt.ylabel("F1")
    plt.show()

#%%

# Data preparation
data_json_unpacked = data_json_unpacked(data_json)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
_all_titles = get_all_titles(data_json_unpacked)
labels = get_all_labels(data_json)

all_performance_metrics = {"Train" : [], "Test" : []}

for bootstrap in range(1, _n_bootstrap + 1):
    print(f"\n==============================\nBOOTSTRAP {bootstrap} \n==============================")
    
    # Set seed
    random.seed(bootstrap)
    
    # Get Bootstrap split        
    bootstrap_train, bootstrap_test = get_bootstrap(data_json_unpacked, number_of_samples = 1624, labels = labels["duplicate_pairs"])    

    """
    TRAINING (Tuning threshold jaccard similarity)
    """ 
    print(f"----------\nTRAINING {bootstrap}\n----------")
    
    # Get attributes for candidate selection
    all_substrings = get_all_substrings(bootstrap_train["data"], filter_subs = True)  
    all_features = get_all_features(bootstrap_train["data"])
    
    # Get candidate pairs
    _sparse_matrix, output_sparse_matrix = get_sparse_matrix(bootstrap_train["data"], all_substrings)
    output_signature_matrix = get_signature_matrix(_sparse_matrix, all_substrings, n_perm = _n_perm)
    output_lsh, candidate_pairs, selected_frequency = get_lsh(output_signature_matrix["signature_matrix"], bootstrap_train["labels"], bootstrap_train["source_indices"], -1)
    
    # Classify dupicates
    _sparse_matrix_full, output_sparse_matrix_full = get_sparse_matrix_full(bootstrap_train["data"], all_features, all_substrings)
    output_jaccard = get_jaccard_similarity(_sparse_matrix_full, candidate_pairs, bootstrap_train["labels"], bootstrap_train["source_indices"])
    
    performance_metrics, tuned_jaccard_threshold = performance_evaluation(output_jaccard, bootstrap_train, -1, candidate_pairs)
    all_performance_metrics["Train"].append(performance_metrics)
    
    """
    TESTING
    """
    print(f"----------\nTESTING {bootstrap}\n----------")    
    # Get attributes for candidate selection
    all_substrings = get_all_substrings(bootstrap_test["data"], filter_subs = True)  
    all_features = get_all_features(bootstrap_test["data"])
    
    # Get candidate pairs
    _sparse_matrix, output_sparse_matrix = get_sparse_matrix(bootstrap_test["data"], all_substrings)
    output_signature_matrix = get_signature_matrix(_sparse_matrix, all_substrings, n_perm = _n_perm)
    output_lsh, candidate_pairs, _ = get_lsh(output_signature_matrix["signature_matrix"], bootstrap_test["labels"], bootstrap_test["source_indices"], selected_frequency)
    
    # Classify dupicates
    _sparse_matrix_full, output_sparse_matrix_full = get_sparse_matrix_full(bootstrap_test["data"], all_features, all_substrings)
    output_jaccard = get_jaccard_similarity(_sparse_matrix_full, candidate_pairs, bootstrap_test["labels"], bootstrap_test["source_indices"])
    
    performance_metrics, _ = performance_evaluation(output_jaccard, bootstrap_test, tuned_jaccard_threshold, candidate_pairs)
    all_performance_metrics["Test"].append(performance_metrics)

    """
    Investigate true duplicates
    """
     # Classify dupicates
    _sparse_matrix_full, output_sparse_matrix_full = get_sparse_matrix_full(data_json_unpacked, all_features, all_substrings)
    jaccard_bootstraps = get_jaccard_similarity_duplicates(_sparse_matrix_full, list(random.choices(candidate_pairs, k = 10000)))
    
    # Classify dupicates
    _sparse_matrix_full, output_sparse_matrix_full = get_sparse_matrix_full(data_json_unpacked, all_features, all_substrings)
    jaccard_duplicates = get_jaccard_similarity_duplicates(_sparse_matrix_full, labels["duplicate_pairs"])
    
    # Classify dupicates
    _sparse_matrix_full, output_sparse_matrix_full = get_sparse_matrix_full(data_json_unpacked, all_features, all_substrings)
    jaccard_duplicates = get_jaccard_similarity_duplicates_and_candidates(_sparse_matrix_full, labels["duplicate_pairs"], list(random.choices(candidate_pairs, k = 10000)))
   
    # Plots LSH & Jaccard
    get_plot_lsh(bootstrap_test, output_lsh, performance_metrics)
    get_plot_jaccard(bootstrap_test, output_jaccard, performance_metrics, candidate_pairs)















