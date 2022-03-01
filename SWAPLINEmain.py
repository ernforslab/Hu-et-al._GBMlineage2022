import datetime
import seaborn as sns

import pickle as pickle
from scipy.spatial.distance import cdist, pdist, squareform
import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
#from sklearn.model_selection import StratifiedShuffleSplit
from collections import defaultdict
from sklearn import preprocessing
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedShuffleSplit
from collections import defaultdict
from sklearn import preprocessing
import random
import datetime
from sklearn.decomposition import PCA
import scipy
from sklearn.metrics import pairwise_distances
from scipy.sparse import issparse, coo_matrix
import sys

def prediction(mwanted_order, mclasses_names, mprotogruop,
               mdf_train_set,mtrain_index,mreorder_ix,
               mcolor_dict,net,learninggroup="train"):
    #mwanted_order = mwanted_order, mclasses_names = mclasses_names, mprotogruop = dfpfcclus.loc["Cluster"].values,
    #mdf_train_set = mdf_train_set, figsizeV = 18, mtrain_index = mtrain_index, net = net, mreorder_ix = mreorder_ix,
    #mcolor_dict = refcolor_dict, learninggroup = "test"

    if  learninggroup=="train":
        mreorder_ix = [list(mclasses_names).index(i) for i in mwanted_order]
        mbool00 = np.in1d( mclasses_names[mtrain_index],  mwanted_order )
        if sum(mcolor_dict.index.isin(mwanted_order))!=len(mwanted_order):
            mcolor_dict={}
            for item in mwanted_order:
                mcolor_dict[item]=random.sample(range(0, 255), 3)
        mcolor_dict = mcolor_dict.map(lambda x: list(map(lambda y: y/255., x)))
        #mcolor_dict = mcolor_dict.map(lambda x: list(map(lambda y: y/255., x)))
        #rcParams['savefig.dpi'] = 500
        #mnewcolors = array(list(mcolor_dict[mprotogruop].values))
        normalizer = 0.9*mdf_train_set.values.max(1)[:,np.newaxis]
        refdataLR=net.predict_proba((mdf_train_set.values/ normalizer).T)

        todaytime=f"{datetime.datetime.now():%Y%m%d%I%M%p}"

        dataRef= refdataLR[:,mreorder_ix]
        mreordername=[]
        for i in mreorder_ix:
            mreordername.append(list(mclasses_names)[i])
        dfprobCL=pd.DataFrame(dataRef*100, index=mdf_train_set.columns,columns=mreordername)
        #dfnewcl=pd.DataFrame(array([xtest,ytest]).T, index=mdf_train_set.columns)
        return mreordername, dfprobCL, mcolor_dict, refdataLR, mreorder_ix
    elif learninggroup=="test":
        #mreorder_ix = [list(mwanted_order).index(i) for i in mwanted_order]
        if sum(mcolor_dict.index.isin(mwanted_order))!=len(mwanted_order):
            mcolor_dict={}
            for item in mwanted_order:
                mcolor_dict[item]=random.sample(range(0, 255), 3)
        mcolor_dict = mcolor_dict.map(lambda x: list(map(lambda y: y/255., x)))
        #mnewcolors = array(list(mcolor_dict[mprotogruop].values))
        normalizerTest=mdf_train_set.max(1)-mdf_train_set.min(1)
        normalizedValue=(mdf_train_set.sub(mdf_train_set.min(1),0).div(normalizerTest,0).fillna(0).values).T
        dataRef=net.predict_proba( normalizedValue)
        mreordername=[]
        for i in mreorder_ix:
            mreordername.append(list(mclasses_names)[i])
        dfprobCL=pd.DataFrame(dataRef*100, index=mdf_train_set.columns,columns=mreordername)
        #dfnewcl=pd.DataFrame(array([xtest,ytest]).T, index=mdf_train_set.columns)
        return mreordername, dfprobCL,  mcolor_dict, dataRef


def permutationTest(mdf_train_set,net, dfprobRef,mreorder_ix,num):
    test = mdf_train_set.values.reshape((len(mdf_train_set.columns) * len(mdf_train_set.index)))
    test = np.random.permutation(test)
    test = test.reshape((len(mdf_train_set.index), len(mdf_train_set.columns)))
    dftest = pd.DataFrame(test).astype(float)
    xp = dftest.values
    xp -= xp.min()
    xp /= xp.ptp()
    test0 = net.predict_proba((xp).T)[:, mreorder_ix]
    for i in range(0, num):
        test = mdf_train_set.values.reshape((len(mdf_train_set.columns) * len(mdf_train_set.index)))
        test = np.random.permutation(test)
        test = test.reshape((len(mdf_train_set.index), len(mdf_train_set.columns)))
        dftest = pd.DataFrame(test).astype(float)
        xp = dftest.values
        xp -= xp.min()
        xp /= xp.ptp()
        dataRef2 = net.predict_proba((xp).T)[:, mreorder_ix]

        test0 = np.append(test0, dataRef2, axis=0)
        # test0=test0+dataRef2

    thresholdlist = []
    temp = []
    for threshold in np.arange(0.0, 1.0, 0.01):
        thresholdlist.append("Prob_%s%%" % int(threshold * 100))
        temp.append((np.sum(test0 > threshold, axis=0) / test0.shape[0]))

    ratiodf = pd.DataFrame(temp)
    ratiodf.index = thresholdlist
    ratiodf.columns = dfprobRef.columns
    dftest0 = pd.DataFrame(test0 * 100, columns=dfprobRef.columns)
    return dftest0, ratiodf

def indices_distancesDensematrix(D, n_neighbors):
    sample_range = np.arange(D.shape[0])[:, None]
    indices = np.argpartition(D, n_neighbors-1, axis=1)[:, :n_neighbors]
    indices = indices[sample_range, np.argsort(D[sample_range, indices])]
    distances = D[sample_range, indices]
    return indices, distances
def sparse_matrixindicesDistances(indices, distances, n_obs, n_neighbors):
    n_nonzero = n_obs * n_neighbors
    indptr = np.arange(0, n_nonzero + 1, n_neighbors)
    D = scipy.sparse.csr_matrix((distances.copy().ravel(),  # copy the data, otherwise strange behavior here
                                indices.copy().ravel(),
                                indptr),
                                shape=(n_obs, n_obs))
    D.eliminate_zeros()
    return D

def SWAPLINE_dist(dfnn, n_neighbors, dfposi, metric = 'euclidean', n_pcs=30, TopN=30):
    #n_pcs = 30, n_neighbors = len(dfnn.index),  metric = 'euclidean'
    X = dfnn

    pca_ = PCA(n_components=n_pcs, svd_solver='arpack', random_state=0)
    X_pca = pca_.fit_transform(X)
    PariDistances = pairwise_distances(X_pca, metric=metric)
    knn_indices, knn_distances = indices_distancesDensematrix(PariDistances, n_neighbors)
    _distances = sparse_matrixindicesDistances(knn_indices, knn_distances, X_pca.shape[0], n_neighbors)
    dftestdist = pd.DataFrame(knn_distances)
    dftest = 0
    dftestindex = pd.DataFrame(knn_indices)
    # dfnn=df.T
    # dfnn.shape
    dftestindex.index = dfnn.index
    umap1AllCluster = []
    umap2AllCluster = []
    clusternames = list(set(dfposi["Cluster"]))
    sys.stdout.write("[%s]" % "Processing")
    sys.stdout.flush()
    sys.stdout.write("\b" * (50 + 1))  # return to start of line, after '['
    perc = len(clusternames)
    for item in clusternames:
        # toolbar_width = len(clusternames)
        itemindex = clusternames.index(item)
        # setup toolbar

        sys.stdout.write("-%s%%-" % int(itemindex*100 / perc))
        sys.stdout.flush()
        umap1cluster = []
        umap2cluster = []
        clustemp = dfposi.loc[dfposi["Cluster"] == item]["Index"]
        for i in range(len(dftestindex.index)):
            nearestvalue = dftestindex.iloc[i, :].loc[dftestindex.iloc[i, :].isin(clustemp)][:TopN].tolist()
            umap1cluster.append(
                (dfposi.iloc[nearestvalue, 1].astype(float).mean() + dfposi.iloc[nearestvalue[0], 1]) / 2)
            umap2cluster.append(
                (dfposi.iloc[nearestvalue, 0].astype(float).mean() + dfposi.iloc[nearestvalue[0], 0]) / 2)
        umap1AllCluster.append(umap1cluster + np.random.uniform(-0.075, 0.075, size=len(umap1cluster)))
        umap2AllCluster.append(umap2cluster + np.random.uniform(-0.075, 0.075, size=len(umap2cluster)))

    dfcellclusumap1 = pd.DataFrame(umap1AllCluster, index=clusternames, columns=dftestindex.index).T
    dfcellclusumap2 = pd.DataFrame(umap2AllCluster, index=clusternames, columns=dftestindex.index).T
    sys.stdout.write("]\n")
    return dfcellclusumap1, dfcellclusumap2

def SWAPLINE_assign(dfprobCL, negtest, n, dfcellclusumap1,dfcellclusumap2,nodelist):
    #n= len(set(dfposi["Cluster"]))
    #nodelist=[['Neural_crest', 'Neural_tube',  'Ectoderm'],['Neural_crest','Pericyte/SMC',  'VLMC'],['Neural_crest', 'Ectoderm','VLMC'],
#['Rgl','Neural_tube',  'Ectoderm'],['Rgl','Neural_tube',  'Glia'],['Rgl','Neural_tube',  'OPCs'],['Rgl','Neural_tube',  'Neuron'],
#['Rgl','OPCs',  'Neuron'],['Rgl','Glia', 'Neuron'],['Rgl','Glia', 'OPCs']]
    dffinalprob = dfprobCL - negtest
    dffinalprob[dffinalprob < 0] = 0

    dfrank2 = dffinalprob.T
    # dfrank.shape
    sumlist = []
    for testx in range(len(dfrank2.columns)):
        dftempnn = dfprobCL.T.loc[dfrank2.nlargest(n, dfrank2.columns[testx]).iloc[:n, :].index, dfrank2.columns[testx]]
        sumlist.append(np.sum(dftempnn))
    dfsumnew = dfprobCL.T
    dfsumnew.loc["sum_nn"] = sumlist
    indexlist = dfsumnew.T.loc[dfsumnew.loc["sum_nn"] > 1].index
    dfrank = dffinalprob.T
    newumap1 = []
    newumap2 = []
    for testx in dfrank.columns:
        nodeprob = []
        for item in nodelist:
            nodeprob.append(dfrank[testx].loc[item].sum())
        nodename = nodelist[np.array(nodeprob).argmax(axis=0)]
        # dftempnn=dfrank.nlargest(n,testx)[testx][:n]
        dftempnn = dfrank.loc[nodename, testx]
        newumap1.append(np.sum(dfcellclusumap1.loc[testx, dftempnn.index] * (dftempnn / np.sum(dftempnn))))
        newumap2.append(np.sum(dfcellclusumap2.loc[testx, dftempnn.index] * (dftempnn / np.sum(dftempnn))))
    dfnewumap = pd.DataFrame([newumap2, newumap1], columns=dffinalprob.index)
    dfnewumap=dfnewumap.T
    return dfnewumap