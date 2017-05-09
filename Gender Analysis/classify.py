"""
classify.py
"""
import pickle
import re
from pprint import pprint
import numpy as np
from itertools import chain, combinations
from scipy.sparse import csr_matrix
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
from collections import Counter, defaultdict
from sklearn.neighbors import KNeighborsClassifier

import glob
import matplotlib.pyplot as plt
import os
import string
import tarfile
from scipy.sparse import lil_matrix



def add_gender(followerDetails,male_names,female_names):
    count = 0
    fcount = 0
    mcount = 0
    for username in followerDetails.keys():
        detail = followerDetails[username]
        name = detail['name']
        if name:
            name_parts = re.findall('\w+', name.split()[0].lower())
            if len(name_parts) > 0:
                first = name_parts[0].lower()
                if first in male_names:
                    detail['gender'] = 'male'
                    mcount = mcount + 1
                elif first in female_names:
                    detail['gender'] = 'female'
                    fcount = fcount + 1
                else:
                    detail['gender'] = 'unknown'
                    count = count + 1
        followerDetails[username] = detail
    if(mcount == 0):
        for username in followerDetails.keys():
            if (followerDetails[username]['gender'] == 'unknown'):
                followerDetails[username]['gender'] = 'male'
                mcount =mcount +1
                count =count - 1
                break
    if(fcount == 0):
        for username in followerDetails.keys():
            if (followerDetails[username]['gender'] == 'unknown'):
                followerDetails[username]['gender'] = 'female'
                fcount =fcount +1
                count =count - 1
                break


def tokenize(doc, keep_internal_punct=False):
    token = []
    if(keep_internal_punct == False ):
        token = re.sub('\W+', ' ', doc.lower()).split()
    
    else:
        if(keep_internal_punct == True ):
            token = re.sub(r'(?<!\S)[^\s\w]+|[^\s\w]+(?!\S)',' ', str(doc).lower()).split()
    return np.array(token)

def eval_all_combinations(docs, labels, punct_vals,
                          feature_fns, min_freqs):
                      
    token_punct_vals = {}
    for keep_internal_punct in punct_vals:
        doc_list = []
        for doc in docs:
            narray = tokenize(doc, keep_internal_punct)
            doc_list.append(narray)
        token_punct_vals[keep_internal_punct] = doc_list
    combi = []
    for i in range(1,len(feature_fns)+1):
        combiTemp = [list(com) for com in combinations(feature_fns, i)]
        for c in combiTemp:
            combi.append(c)
    
    unsortedList = []
    for key in token_punct_vals.keys():
        doc_list = token_punct_vals[key]
        for combinatn in combi:
            for freq in min_freqs:
                csr , vocab = vectorize(doc_list, combinatn, freq)
                model = LogisticRegression()
                accuracy = cross_validation_accuracy(model, csr, labels, 5)
                tempDict = {}
                tempDict['punct'] = key
                tempDict['features'] = combinatn
                tempDict['min_freq'] = freq
                tempDict['accuracy'] = accuracy
                tempDict['model'] = 'LogisticRegression'
                unsortedList.append(tempDict)
                for i in range (1,11):
                    model2 = KNeighborsClassifier(n_neighbors=1)
                    accuracy = cross_validation_accuracy(model, csr, labels, 5)
                    tempDict = {}
                    tempDict['punct'] = key
                    tempDict['features'] = combinatn
                    tempDict['min_freq'] = freq
                    tempDict['accuracy'] = accuracy
                    tempDict['model'] = 'KNeighborsClassifier'
                    tempDict['neighbor'] = i
                    unsortedList.append(tempDict)
    sortedList = sorted(unsortedList, key = lambda k:-k['accuracy'])
    return sortedList

def featurize(tokens, feature_fns):
    feats = defaultdict(lambda: 0)
    for feature in feature_fns:
        feature(tokens,feats)
    return sorted(feats.items())

def make_vocabulary(tokens_list):
    vocabulary = defaultdict(lambda: len(vocabulary))
    for tokens in tokens_list:
        for token in tokens:
            vocabulary[token]
    return vocabulary

def make_feature_matrix(tokens_list, vocabulary):
    X = lil_matrix((len(tokens_list), len(vocabulary)))
    for i, tokens in enumerate(tokens_list):
        for token in tokens:
            if token in vocabulary.keys():
                j = vocabulary[token]
                X[i,j] += 1
    return X.tocsr()

def vectorize(tokens_list, feature_fns, min_freq, vocab=None):
    indptr = [0]
    indices = []
    data = []
    feats = []
    count_dict = {}
    for d in tokens_list:
        feature = featurize(d, feature_fns)
        feats.append(feature)
        for f in feature:
            if f[0] in count_dict:
                count_dict[f[0]] = count_dict[f[0]] + 1
            else:
                count_dict[f[0]] = 1
    if(vocab == None):
        vocab = make_vocabulary(tokens_list)
    matrix = make_feature_matrix(tokens_list, vocab)
    return matrix,vocab


def token_features(tokens, feats):
    list = []
    for t in tokens:
        tokenKey = 'token=' + t
        list.append(tokenKey)
    count = Counter(list)
    for key, value in count.items():
        feats[key] = feats[key] + value


def token_pair_features(tokens, feats, k=3):
    tokenList = []
    for i in range(len(tokens)):
        windows = []
        for j in range(i,i+k):
            if(j < len(tokens)):
                windows.append(tokens[j])
        if(len(windows) == k):
            subWindows = list(combinations(windows, 2))
            for subWindow in subWindows:
                value = subWindow[0] + '__' + subWindow[1]
                tokenList.append(value)
    tpList = []
    for tweet in tokenList:
        tokenKey = 'token_pair=' + tweet
        tpList.append(tokenKey)
    count = Counter(tpList)
    for key, value in count.items():
        feats[key] = feats[key] + value

def get_docs(followerDetails):
    count = 0
    docs = []
    truth = []
    for userName in followerDetails.keys():
        if('gender' in followerDetails[userName]):
            if(followerDetails[userName]['gender'] != 'unknown'):
                docs.append(followerDetails[userName]['description'])
                if len(followerDetails[userName]['description']) == 0:
                    count = count +1
                
                if followerDetails[userName]['gender'] != 'male':
                    truth.append(1)
                else:
                    truth.append(0)
    return docs,np.array(truth)

def accuracy_score(truth, predicted):
       return len(np.where(truth==predicted)[0]) / len(truth)


def cross_validation_accuracy(clf, X, labels, k):
    cv = KFold(len(labels), k)
    accuracies = []
    for train_ind, test_ind in cv:
        clf.fit(X[train_ind], labels[train_ind])
        predictions = clf.predict(X[test_ind])
        accuracies.append(accuracy_score(labels[test_ind], predictions))
    return np.mean(accuracies)

def get_unknown_docs(followerDetails):
    docs = []
    count = 0
    userNameList = []
    for userName in followerDetails.keys():
        if('gender' in followerDetails[userName]):
            if(followerDetails[userName]['gender'] == 'unknown'):
                docs.append(followerDetails[userName]['description'])
                if len(followerDetails[userName]['description']) == 0:
                    count = count +1
                userNameList.append(userName)
    return docs,userNameList

def predict_unkown(result,unknown_docs,docs,labels,followerDetails,userNameList):
    internal_punct = result['punct']
    combinatn = result['features']
    freq = result['min_freq']
    accuracy = result['accuracy']
    doc_list = []
    for doc in docs:
        narray = tokenize(doc, internal_punct)
        doc_list.append(narray)
    
    unknown_doc_list = []
    for doc in unknown_docs:
        narray = tokenize(doc, internal_punct)
        unknown_doc_list.append(narray)
    csr , vocab = vectorize(doc_list, combinatn, freq)
    u_csr , u_vocab = vectorize(unknown_doc_list, combinatn, freq,vocab)
    model = LogisticRegression()
    mcount = 0
    fcount = 0
    model.fit(csr, labels)
    predictions = model.predict(u_csr)
    for user,p in zip(userNameList,predictions):
        det = followerDetails[user]
        if p == 0:
            det['gender'] = 'male'
            mcount = mcount+1
        else:
            det['gender'] = 'female'
            fcount = fcount+1
        followerDetails[user] = det


def main():
    male_names = pickle.load(open('maleNames.pkl', 'rb'))
    female_names = pickle.load(open('femaleNames.pkl', 'rb'))
    users = pickle.load(open('users.pkl', 'rb'))
    friendsList = pickle.load(open('friendList.pkl', 'rb'))
    followerDetails = pickle.load(open('followerDetails.pkl', 'rb'))
    add_gender(followerDetails,male_names,female_names)
    docs,labels = get_docs(followerDetails)
    feature_fns = [token_features, token_pair_features]
    results = eval_all_combinations(docs, labels,
                                    [True, False],
                                    feature_fns,
                                    [2,5,10])
    best_result = results[0]
    unknown_docs,userNameList = get_unknown_docs(followerDetails)
    predict_unkown(best_result,unknown_docs,docs,labels,followerDetails,userNameList)
    pickle.dump(followerDetails, open('followerDetails.pkl', 'wb'))




if __name__ == '__main__':
    main()
