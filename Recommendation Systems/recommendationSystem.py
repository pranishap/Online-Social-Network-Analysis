# coding: utf-8

from collections import Counter, defaultdict
import math
import numpy as np
import os
import pandas as pd
import re
from scipy.sparse import csr_matrix
import urllib.request
import zipfile

def download_data():
    url = 'https://www.dropbox.com/s/h9ubx22ftdkyvd5/ml-latest-small.zip?dl=1'
    urllib.request.urlretrieve(url, 'ml-latest-small.zip')
    zfile = zipfile.ZipFile('ml-latest-small.zip')
    zfile.extractall()
    zfile.close()


def tokenize_string(my_string):
    return re.findall('[\w\-]+', my_string.lower())


def tokenize(movies):
    movies['tokens'] = movies['genres'].apply(tokenize_string)
    return movies


def featurize(movies):
    docDictionary = {}
    i = 1
    tokenSet = set()
    df = {}
    maxktf = {}
    
    for docTokens in movies['tokens'].tolist():
        wordcount = {}
        for token in docTokens:
            tokenSet.add(token)
            if(token not in wordcount.keys()):
                if(token in df.keys()):
                    df[token] += 1
                else:
                    df[token] = 1
            if(token in wordcount.keys()):
                wordcount[token] += 1
            else:
                wordcount[token] = 1
        maxktf[i] = max([wordcount[i] for i in wordcount.keys()])
        docDictionary[i] = wordcount
        i += 1
    tokenList = list(tokenSet)
    tokenList = sorted(tokenList)
    vocab = {}
    j = 0
    for token in tokenList:
        vocab[token] = j
        j += 1

    n = i

    matirxList = []
    s,num = 1,j
    for key in docDictionary.keys():
        value = docDictionary[key]
        indptr = [0]
        indices = []
        data = []
        for term in value.keys():
            index = vocab[term]
            indices.append(index)
            tf = value[term]
            maxk = maxktf[key]
            dfTerm = df[term]
            val = (tf/maxk)* math.log10(n/dfTerm)
            data.append(val)
        indptr.append(len(indices))
        matrix = csr_matrix((data, indices, indptr),shape = (s,num))
        matirxList.append(matrix)
    movies['features'] = matirxList
    t = movies,vocab
    return t





def train_test_split(ratings):
    test = set(range(len(ratings))[::1000])
    train = sorted(set(range(len(ratings))) - test)
    test = sorted(test)
    return ratings.iloc[train], ratings.iloc[test]


def cosine_sim(a, b):
    dot_prod = a.multiply(b).sum()
    norma = np.linalg.norm(a.toarray(),ord=None)
    normb = np.linalg.norm(b.toarray(),ord=None)
    cosine = dot_prod/(norma * normb)
    return cosine




def make_predictions(movies, ratings_train, ratings_test):
    predictedList = []
    for index, row in ratings_test.iterrows():
        userId = row['userId']
        movieId = row['movieId']
        trainList = ratings_train.loc[ratings_train['userId'] == userId]
        movie_i_row = movies.loc[movies['movieId'] == movieId]
        csr_movie_i = movie_i_row['features'].iloc[0]
        ratingList =[]
        cosineList = []
        isNeg = True
        val = 0
        for ind,trow in trainList.iterrows():
            if(trow['movieId'] != movieId):
                u_movie_id = trow['movieId']
                movie_u_row = movies.loc[movies['movieId'] == u_movie_id]
                csr_movie_u = movie_u_row['features'].iloc[0]
                cosineVal = cosine_sim(csr_movie_i, csr_movie_u)
                ratingList.append(trow['rating'])
                cosineList.append(cosineVal)
                if(cosineVal > 0):
                    isNeg = False
    
        if(isNeg):
            val = sum(ratingList)/len(ratingList)
        else:
            dot_prod = [a*b for a,b in zip(ratingList,cosineList)]
            total_weight = sum(cosineList)
            val =sum(dot_prod)/total_weight
        predictedList.append(val)
    return np.array(predictedList)


def mean_absolute_error(predictions, ratings_test):
    return np.abs(predictions - np.array(ratings_test.rating)).mean()


def main():
    download_data()
    path = 'ml-latest-small'
    ratings = pd.read_csv(path + os.path.sep + 'ratings.csv')
    movies = pd.read_csv(path + os.path.sep + 'movies.csv')
    movies = tokenize(movies)
    movies, vocab = featurize(movies)
    print('vocab:')
    print(sorted(vocab.items())[:10])
    ratings_train, ratings_test = train_test_split(ratings)
    print('%d training ratings; %d testing ratings' % (len(ratings_train), len(ratings_test)))
    predictions = make_predictions(movies, ratings_train, ratings_test)
    print('error=%f' % mean_absolute_error(predictions, ratings_test))
    print(predictions[:10])


if __name__ == '__main__':
    main()
