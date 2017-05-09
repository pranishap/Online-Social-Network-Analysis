# coding: utf-8

from collections import Counter, defaultdict
from itertools import chain, combinations
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from scipy.sparse import csr_matrix
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
import string
import tarfile
import urllib.request


def download_data():
    url = 'https://www.dropbox.com/s/xk4glpk61q3qrg2/imdb.tgz?dl=1'
    urllib.request.urlretrieve(url, 'imdb.tgz')
    tar = tarfile.open("imdb.tgz")
    tar.extractall()
    tar.close()


def read_data(path):
    fnames = sorted([f for f in glob.glob(os.path.join(path, 'pos', '*.txt'))])
    data = [(1, open(f).readlines()[0]) for f in sorted(fnames)]
    fnames = sorted([f for f in glob.glob(os.path.join(path, 'neg', '*.txt'))])
    data += [(0, open(f).readlines()[0]) for f in sorted(fnames)]
    data = sorted(data, key=lambda x: x[1])
    return np.array([d[1] for d in data]), np.array([d[0] for d in data])


def tokenize(doc, keep_internal_punct=False):
    token = []
    if(keep_internal_punct == False ):
        token = re.sub('\W+', ' ', doc.lower()).split()
    
    else:
        if(keep_internal_punct == True ):
            token = re.sub(r'(?<!\S)[^\s\w]+|[^\s\w]+(?!\S)',' ', str(doc).lower()).split()
    return np.array(token)


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




neg_words = set(['bad', 'hate', 'horrible', 'worst', 'boring'])
pos_words = set(['awesome', 'amazing', 'best', 'good', 'great', 'love', 'wonderful'])

def lexicon_features(tokens, feats):
    pos_counts = 0
    neg_counts = 0
    for token in tokens:
        if(token.lower() in [word.lower() for word in neg_words]):
            neg_counts = neg_counts + 1
        elif(token.lower() in [word.lower() for word in pos_words]):
            pos_counts = pos_counts + 1
    feats['neg_words'] =  neg_counts
    feats['pos_words'] =  pos_counts


def featurize(tokens, feature_fns):
    feats = defaultdict(lambda: 0)
    for feature in feature_fns:
        feature(tokens,feats)
    return sorted(feats.items())

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
        vocab = {}
        for token in sorted(count_dict.keys()):
            if (count_dict[token] >= min_freq):
                vocab.setdefault(token,len(vocab))

    for d in feats:
        for term in d:
            if(term[0] in vocab):
                index = vocab[term[0]]
                indices.append(index)
                data.append(term[1])
        indptr.append(len(indices))
    matrix = csr_matrix((data, indices, indptr), dtype=np.int64)
    return matrix,vocab


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
                unsortedList.append(tempDict)
                         
    sortedList = sorted(unsortedList, key = lambda k:-k['accuracy'])
    return sortedList

def plot_sorted_accuracies(results):
    accu_list = []
    for res in results:
        accu_list.append(res['accuracy'])
    sortedList = sorted(accu_list, key = lambda k:k)
    length = len(sortedList)
    plt.plot(np.arange(length), sortedList)
    plt.xlabel('setting', size=14)
    plt.ylabel('accuracy', size=14)
    plt.savefig('accuracies.png')


def mean_accuracy_per_setting(results):
    frequency_dict = {}
    punct_dict = {}
    feature_dict = {}
    final_list = []
    for dict in results:
        f = dict['min_freq']
        p = dict['punct']
        feat_com = tuple(dict['features'])
        accuracy = dict['accuracy']
        
        if(f in frequency_dict):
            flist = frequency_dict[f]
            flist.append(accuracy)
            frequency_dict[f] = flist
        else:
            newList = []
            newList.append(accuracy)
            frequency_dict[f] = newList
        if(p in punct_dict):
            pList = punct_dict[p]
            pList.append(accuracy)
            punct_dict[p] = pList
        else:
            newList = []
            newList.append(accuracy)
            punct_dict[p] = newList
    
        if(feat_com in feature_dict):
            feList = feature_dict[feat_com]
            feList.append(accuracy)
            feature_dict[feat_com] = feList
        else:
            newList = []
            newList.append(accuracy)
            feature_dict[feat_com] = newList
    
    for f_comb in feature_dict.keys():
            f_value = feature_dict[f_comb]
            length = len(f_value)
            sum_feature = sum(f_value)
            mean = sum_feature / length
            feature_name = ""
            for fe in f_comb:
                if(feature_name == ""):
                    feature_name = 'features=' + fe.__name__
                else:
                    feature_name = feature_name  + " " + fe.__name__
            
            t = (mean , feature_name)
            final_list.append(t)
    

    for freq in frequency_dict.keys():
            freq_value = frequency_dict[freq]
            length = len(freq_value)
            sum_feature = sum(freq_value)
            mean = sum_feature / length
            freq_string = 'min_freq=' + str(freq)
            t = (mean , freq_string)
            final_list.append(t)


    for punct in punct_dict.keys():
            punct_value = punct_dict[punct]
            length = len(punct_value)
            sum_feature = sum(punct_value)
            mean = sum_feature / length
            punct_string = 'punct=' + str(punct)
            t = (mean , punct_string)
            final_list.append(t)


    return sorted(final_list, key = lambda k:-k[0])





def fit_best_classifier(docs, labels, best_result):
    min_freq = best_result['min_freq']
    internal_punct = best_result['punct']
    feature_fns = best_result['features']
    accuracy = best_result['accuracy']
    token_List = []
    for doc in docs:
        t_list = tokenize(doc, internal_punct)
        token_List.append(t_list)
    matrix , vocab = vectorize(token_List, feature_fns, min_freq)
    clf = LogisticRegression()
    clf.fit(matrix,labels)
    return clf , vocab




def top_coefs(clf, label, n, vocab):
    coef = clf.coef_[0]
    if( label == 0):
        top_coef_ind = np.argsort(coef)[::1][:n]
    else:
        top_coef_ind = np.argsort(coef)[::-1][:n]

    top_coef_terms = []
    vocabList = {}
    for key in vocab.keys():
        value = vocab[key]
        vocabList[value] = key

    for index in top_coef_ind:
        top_coef_terms.append(vocabList[index])

    top_coef = coef[top_coef_ind]
    return [x for x in zip(top_coef_terms, abs(top_coef))]


def parse_test_data(best_result, vocab):
    min_freq = best_result['min_freq']
    internal_punct = best_result['punct']
    feature_fns = best_result['features']
    accuracy = best_result['accuracy']
    docs, labels = read_data(os.path.join('data', 'test'))
    token_List = []
    for doc in docs:
        t_list = tokenize(doc, internal_punct)
        token_List.append(t_list)
    matrix , vocab = vectorize(token_List, feature_fns, min_freq,vocab)

    return docs, labels , matrix




def print_top_misclassified(test_docs, test_labels, X_test, clf, n):
    misclassified_List = []
    predicted = clf.predict(X_test)
    probability = clf.predict_proba(X_test)
    for i in range(len(test_docs)):
        if(test_labels[i] != predicted[i]):
            dict = {}
            dict['truth'] = test_labels[i]
            p = predicted[i]
            dict['predicted'] = p
            dict['proba'] = probability[i][p]
            dict['doc'] = test_docs[i]
            misclassified_List.append(dict)

    sortedList = sorted(misclassified_List, key = lambda k:-k['proba'])
    for j in range(n):
        d = sortedList[j]
        print('\n\n truth=%d predicted=%d proba=%.6f \n %s ' % (d['truth'],d['predicted'],d['proba'],d['doc']) )


def main():
    feature_fns = [token_features, token_pair_features, lexicon_features]
    # Download and read data.
    download_data()
    docs, labels = read_data(os.path.join('data', 'train'))
    # Evaluate accuracy of many combinations
    # of tokenization/featurization.
    results = eval_all_combinations(docs, labels,
                                    [True, False],
                                    feature_fns,
                                    [2,5,10])
    # Print information about these results.
    best_result = results[0]
    worst_result = results[-1]
    print('best cross-validation result:\n%s' % str(best_result))
    print('worst cross-validation result:\n%s' % str(worst_result))
    plot_sorted_accuracies(results)
    print('\nMean Accuracies per Setting:')
    print('\n'.join(['%s: %.5f' % (s,v) for v,s in mean_accuracy_per_setting(results)]))
    
    # Fit best classifier.
    clf, vocab = fit_best_classifier(docs, labels, results[0])
                                    
    # Print top coefficients per class.
    print('\nTOP COEFFICIENTS PER CLASS:')
    print('negative words:')
    print('\n'.join(['%s: %.5f' % (t,v) for t,v in top_coefs(clf, 0, 5, vocab)]))
    print('\npositive words:')
    print('\n'.join(['%s: %.5f' % (t,v) for t,v in top_coefs(clf, 1, 5, vocab)]))
    
    # Parse test data
    test_docs, test_labels, X_test = parse_test_data(best_result, vocab)
                                    
    # Evaluate on test set.
    predictions = clf.predict(X_test)
    print('testing accuracy=%f' %accuracy_score(test_labels, predictions))
    
    print('\nTOP MISCLASSIFIED TEST DOCUMENTS:')
    print_top_misclassified(test_docs, test_labels, X_test, clf, 5)


if __name__ == '__main__':
    main()
