import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import collections
import seaborn as sns
from math import *
from sklearn import cluster, datasets, mixture
from sklearn.metrics import accuracy_score
from random import randrange

# np.random.seed(0)
# N_A = 100
# d=2
    
def crossValSplit(X, y, numFolds):
    '''
    Description:
        Function to split the data into number of folds specified
    Input:
        dataset: data that is to be split
        numFolds: integer - number of folds into which the data is to be split
    Output:
        split data
    '''
    XSplit, ySplit = list(), list()
    XCopy, yCopy = list(X), list(y)
    foldSize = int(len(X) / numFolds)
    for _ in range(numFolds):
        Xfold, yfold = list(), list()
        while len(Xfold) < foldSize:
            index = randrange(len(XCopy))
            Xfold.append(XCopy.pop(index))
            yfold.append(yCopy.pop(index))
        XSplit.append(np.vstack(Xfold))
        ySplit.append(np.stack(yfold))
    return XSplit, ySplit    
    
def kFCVEvaluateA(X, y, k, numFolds, *args):
    '''
    Description:
        Driver function for k-Fold cross validation 
    '''
    Xfolds, yfolds = crossValSplit(X, y, numFolds)

    scores = list()
    for i in range(numFolds):
        Xtrain, ytrain = Xfolds.copy(), yfolds.copy()
        X_test, y_test = Xtrain.pop(i), ytrain.pop(i)
        
        X, y = np.vstack(Xtrain), np.concatenate(ytrain)
        predicted = find_best_label(assign_unlabelled(X,y,X_test,k))
            
        accuracy = accuracy_score(y_test, predicted)
        scores.append(accuracy)
    meanScore = sum(scores)/float(len(scores))
    print('Mean Accuracy at k1 =%d: %.3f' % (k, meanScore))
    return meanScore

def kFCVEvaluateBC(X, y, perA, k1, k2, numFolds, coef=0):
    '''
    Description:
        Driver function for k-Fold cross validation 
    '''
    Xfolds, yfolds = crossValSplit(X, y, numFolds)

    scores = list()
    for i in range(numFolds):
        Xtrain, ytrain = Xfolds.copy(), yfolds.copy()
        X_test, y_test = Xtrain.pop(i), ytrain.pop(i)
    
        X, y = np.vstack(Xtrain), np.concatenate(ytrain)
        n = int(len(X)*perA)
        X_A, X_B, y_A = X[:n], X[n:], y[:n]
        k1 = n if n < k1 else k1
        k2 = len(X_B) if len(X_B) < k2 else k2
        if perA==1: predicted = find_best_label(assign_unlabelled(X_A,y_A,X_test,k1))
        else:    
            if coef == 0: predicted = knn_B(k1, k2, X_A, y_A, X_B, X_test)
            else: predicted = knn_C(coef, k1, k2, X_A, y_A, X_B, X_test)
        accuracy = accuracy_score(y_test, predicted)
        scores.append(accuracy)
    meanScore = sum(scores)/float(len(scores))
    # print('Mean Accuracy at (k1, k2) =(%d, %d): %.3f' % (k1, k2, meanScore))
    return meanScore
    
def generate_two_moon(n_samples=1500, noise=.1):
    x, y = datasets.make_moons(n_samples=n_samples, noise=noise)
    return x, y

def plot_result(X, y, title="", legend="", ax=None):
    ax = ax or plt.gca()
    ax.plot(X[y == 0][:, 0], X[y == 0][:, 1], '.',
            alpha=0.5, label=legend+' x0', color='royalblue')
    ax.plot(X[y == 1][:, 0], X[y == 1][:, 1], '.',
            alpha=0.5, label=legend+' x1', color='coral')
    ax.set_title(title)
    ax.legend(fontsize=14)
    return ax

def plot_heatmap(pivot1, title="", ax = None):
    ax = ax or plt.gca()
    sns.heatmap(pivot1, annot=True, ax = ax)
    ax.set_title(title, fontsize = 10)
    return ax

def plot_bar(x, y, xlabel="", ylabel="", title="", xinterval = 1, ax = None):
    ax = ax or plt.gca()
    rects = ax.bar(x, y, color='b')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title, fontsize = 20)
    plt.ylim(0.6, 1)
    plt.xticks(np.arange(min(x), max(x)+1, xinterval))
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., height,
                '%.2f' % float(height),
        ha='center', va='bottom')
    return ax 

# calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1)):
        distance += (row1[i] - row2[i])**2
    return sqrt(distance)

# Locate the most similar neighbors
# output: [[features, labels]]
def get_neighbors(train, train_labels, test_row, num_neighbors):
    distances = list()
    n = len(train)
    for i in range(n):
        dist = euclidean_distance(test_row, train[i])
        # ([features, labels], distance)
        distances.append(([list(train[i]), train_labels[i]], dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()  # [[features, labels]]
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    return neighbors

def count_prob(arr):
    dist = {}
    n = len(arr)
    if n == 0: return dist
    if type(arr[0]) is not collections.OrderedDict:
        for key in arr:
            if key in dist: dist[key] += 1
            else: dist[key] = 1
    # get the weighted average of probability of being in which class
    else:
        for prob_dict in arr:
            for key in prob_dict:
                if key in dist: dist[key] += prob_dict[key]
                else: dist[key] = prob_dict[key]
    for key in dist: dist[key] = dist[key]/n
    od = collections.OrderedDict(sorted(dist.items()))
    return od


# Make a prediction with neighbors
# train_labels: can be prob?
def predict_classification(train, train_labels, test_row, num_neighbors):
    neighbors = get_neighbors(train, train_labels, test_row, num_neighbors)
    output_values = [row[-1] for row in neighbors]  # [labels]
    prediction = count_prob(output_values)
    return prediction

# find num_neighbors of X_unlabelled from X_labelled
# and assign lables distribution/ weighted labels accoding to y_labelled
# output: list of pmf/ single weighted label
def assign_unlabelled(X_labelled, y_labelled, X_unlabelled, num_neighbors):
    
    predictions = list()
    for row in X_unlabelled:
        output = predict_classification(
            X_labelled, y_labelled, row, num_neighbors)
        predictions.append(output)
    return(np.array(predictions))


# output: KNN with weihted labelled and unlabelled
def mixed_knn(y_A_pred, y_B_pred, coef):
    y_pred_perc = []
    for i in range(len(y_A_pred)):
        y_a, y_b = y_A_pred[i], y_B_pred[i]
        y_pred_perc.append({label: (1 + coef) * y_a.get(label, 0) -
                            coef * y_b.get(label, 0) for label in set(y_a) | set(y_b)})
    y_pred = np.array([])
    for dist in y_pred_perc:
        y_pred = np.append(y_pred, max(dist, key=dist.get))
    return y_pred

def knn_B(k1, k2, X_A, y_A, X_B, X_test):
    y_A_pred_B = assign_unlabelled(X_A, y_A, X_B, k1)
    y_B_pred_dtr = assign_unlabelled(X_B, y_A_pred_B, X_test, k2)
    y_B_pred = find_best_label(y_B_pred_dtr)
    return y_B_pred

def knn_C(coef, k1, k2, X_A, y_A, X_B, X_test):
    y_A_pred_dtr = assign_unlabelled(X_A, y_A, X_test, k1) 
    y_A_pred_B = assign_unlabelled(X_A, y_A, X_B, k1)
    y_B_pred_dtr = assign_unlabelled(X_B, y_A_pred_B, X_test, k2)    
    return mixed_knn(y_A_pred_dtr, y_B_pred_dtr, coef)
    
def find_best_label(y_dist):
    y_pred = np.array([])
    for dist in y_dist:
        y_pred = np.append(y_pred, max(dist, key=dist.get))
    return y_pred

def find_best_record(df, feature, cmp1 = None, cmp2 = None, cmp3 = None):
    """find the best record df with feature maximized
        if feature is equal, find in ascending order of cmp1, cmp2, then cmp3

    Args:
        df (DataFrame): DataFrame
        feature (str): feature to maximize
        cmp1, cpm2, cmp3 (str): attributes to compare if feature is equal
    """
    subdf = df[df[feature] ==df[feature].max()]
    if subdf.shape[0] > 1 and cmp1 != None:
        subdf = subdf[subdf[cmp1] ==subdf[cmp1].min()]
    if subdf.shape[0] > 1 and cmp2 != None:
        subdf = subdf[subdf[cmp2] ==subdf[cmp2].min()]
    if subdf.shape[0] > 1 and cmp3 != None:
        subdf = subdf[subdf[cmp3] ==subdf[cmp3].min()]
    
    return subdf
    
def generate_KNNaccuracy(d, N_A, N_B, k1_B, k2_B, k1_AB, k2_AB, X_A, y_A, X_B, X_test, y_test):
    A_dist, B_dist = {}, {} # {k1/(k1, k2): percentage} save the result to calculate mixed accuracy
    A_accuracy, B_accuracy, mixed_accuracy = [], [], []
    for k1 in k1_B:
        # if k1 larger than N_A
        if k1 > N_A: continue
        # avoid duplicate calculation
        if k1 not in A_dist:
            y_A_pred_dtr = assign_unlabelled(X_A, y_A, X_test, k1) 
            y_A_pred = find_best_label(y_A_pred_dtr) 
            A_accuracy.append({'N_A': N_A, 'N_B': N_B, 'k1': k1, 'accuracy': accuracy_score(y_test, y_A_pred)})
            y_A_pred_B = assign_unlabelled(X_A, y_A, X_B, k1)
            A_dist[k1] = (y_A_pred_dtr, y_A_pred_B)
        else: y_A_pred_dtr, y_A_pred_B = A_dist[k1][0], A_dist[k1][1]

        for k2 in k2_B:
            if k2 > N_B: continue
            if (k1, k2) not in B_dist:
                y_B_pred_dtr = assign_unlabelled(X_B, y_A_pred_B, X_test, k2)
                y_B_pred = find_best_label(y_B_pred_dtr)
                B_accuracy.append({'N_A': N_A, 'N_B': N_B,'k1': k1, 'k2': k2, 'accuracy': accuracy_score(y_test, y_B_pred)})
                B_dist[(k1,k2)]= y_B_pred_dtr
            else: y_B_pred_dtr = B_dist[(k1,k2)]
            
    for k1 in k1_AB:
        if k1 > N_A: continue
        if k1 not in A_dist:
            y_A_pred_dtr = assign_unlabelled(X_A, y_A, X_test, k1) 
            y_A_pred_B = assign_unlabelled(X_A, y_A, X_B, k1)
        else: y_A_pred_dtr, y_A_pred_B = A_dist[k1][0], A_dist[k1][1]

        for k2 in k2_AB:
            if k2 > N_B: continue
            if (k1, k2) not in B_dist:
                y_B_pred_dtr = assign_unlabelled(X_B, y_A_pred_B, X_test, k2)
            else: y_B_pred_dtr = B_dist[(k1,k2)]
            coef = ((N_B*k1)/(N_A*k2))**(2/d)
            y_pred = mixed_knn(y_A_pred_dtr, y_B_pred_dtr, coef)
            mixed_accuracy.append({'N_A': N_A, 'N_B': N_B, 'k1': k1, 'k2': k2, 'lambda': coef, 'accuracy': accuracy_score(y_test, y_pred)})
    
    return pd.DataFrame(A_accuracy), pd.DataFrame(B_accuracy), pd.DataFrame(mixed_accuracy)


# define helper functions
def locate_max_accuracy(d, N_A, N_B, k1_B, k2_B, k1_AB, k2_AB, x, y, X_A, y_A, X_test, y_test, acc_table):    
    # get unlabelled training set
    X_B, _ = x[:N_B], y[:N_B]
    
    labelled_accuracy, unlabelled_accuracy, mixed_accuracy = generate_KNNaccuracy(d, N_A, N_B, k1_B, k2_B, k1_AB, k2_AB, X_A, y_A, X_B, X_test, y_test)
    labelled_accuracy = labelled_accuracy[labelled_accuracy["k1"]%2!=0]
    best_A, best_B, best_mixed = find_best_record(labelled_accuracy, "accuracy", 'k1'), \
                                 find_best_record(unlabelled_accuracy, "accuracy", 'k1', 'k2'), \
                                 find_best_record(mixed_accuracy, "accuracy", 'k1', 'k2')
    
    lb_pivot = labelled_accuracy.pivot_table(columns='k1', values='accuracy')
    unlb_pivot = unlabelled_accuracy.pivot_table(index=["k2"], columns='k1', values='accuracy')
    mixed_pivot = mixed_accuracy.pivot_table(index=["k2"], columns='k1', values='accuracy')
    
    acc_table.append({'N_A': N_A, 'N_B': N_B,
                        '0': best_A.accuracy.iloc[0], '-1':best_B.accuracy.iloc[0],
                        'lambda': best_mixed.accuracy.iloc[0]}) #rounding
    return best_A, best_B, best_mixed, acc_table, lb_pivot, unlb_pivot, mixed_pivot

# def iterate(run, N_unlb, k1_B, k2_B, k1_AB, k2_AB):
#     lb_pivotArr, unlb_pivotArr, mixed_pivotArr = [], [], [] # for ploting average heatmap
#     for i in range(run):
#         x, y = generate_two_moon(N, 0.3)
#         X_labelled, y_labelled, x, y, X_test, y_test = \
#             x[:N_A], y[:N_A], \
#             x[N_A:N-N_test], y[N_A:N-N_test], \
#             x[N-N_test:], y[N-N_test:]
#         # store the best accuracy (re k) for different m, n
#         best_A, best_B, best_mixed, acc_table, lb_pivot, unlb_pivot, mixed_pivot = locate_max_accuracy(N_unlb, k1_B, k2_B, k1_AB, k2_AB, x, y, X_labelled, y_labelled, X_test, y_test, [])
#         listA.append(best_A)
#         listB.append(best_B)
#         listMixed.append(best_mixed)
#         listAcc.append(pd.DataFrame(acc_table))
#         lb_pivotArr.append(lb_pivot)
#         unlb_pivotArr.append(unlb_pivot)
#         mixed_pivotArr.append(mixed_pivot)
#     return lb_pivotArr, unlb_pivotArr, mixed_pivotArr

def plot_avr_heatmap(hm1, hm2, N_A, N_B, run):
    ah1 = pd.concat(hm1).groupby(level=0).mean()
    ah2 = pd.concat(hm2).groupby(level=0).mean()
    fig, ax = plt.subplots(1, 2, figsize=(22, 9))
    plot_heatmap(ah1, "KNN with unlabelled data (N_A, N_B) = (%d, %d), average across %d simulations"%(N_A, N_B, run), ax[0])
    plot_heatmap(ah2, "KNN with mixed data (N_A, N_B) = (%d, %d), average across %d simulations"%(N_A, N_B, run), ax[1])
    return ax

def avr_best_score(hm1,hm2, t1):
    avr_acc=[]
    a1, a2, t1 = pd.concat(hm1).groupby(level=0).mean(), pd.concat(hm2).groupby(level=0).mean(), pd.concat(t1).groupby(level=0).mean()
    a1_m, a2_m, t1_m = a1.max(), a2.max(), t1.max(axis=1)
    t1_k1_max = t1.idxmax(axis=1)
    avr_acc.append({'k1': t1_k1_max.accuracy, 'k2': None, 'accuracy': t1_m.max()})
    
    a1_max, k1_max= a1_m.max(), a1_m.idxmax()
    k2_max = a1.idxmax()[k1_max]
    avr_acc.append({'k1': k1_max, 'k2': k2_max, 'accuracy': a1_max})

    a2_max, k1_max= a2_m.max(), a2_m.idxmax()
    k2_max = a2.idxmax()[k1_max]
    avr_acc.append({'k1': k1_max, 'k2': k2_max, 'accuracy': a2_max})
    return pd.DataFrame(avr_acc, index=['0', '-1', 'mixed'])