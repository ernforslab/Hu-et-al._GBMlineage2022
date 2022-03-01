
import pandas as pd
import numpy as np
import datetime
import seaborn as sns
import pandas as pd
import pickle as pickle
from scipy.spatial.distance import cdist, pdist, squareform
#import backspinpy
import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedShuffleSplit
from collections import defaultdict
from sklearn import preprocessing
import matplotlib.patches as mpatches


def TransSpeciesGeneName(dfm, dictfilename, path):
    # dfm=dfpfc, dictfilename=dictfilename, path=path

    with open('%s%s' % (path, dictfilename), 'rb') as f:
        mouse2human_dict = pickle.load(f, encoding='ASCII')
    # dfm=df_f
    # convert the mouse gene name in the human gene name
    mouse_translable = [i for i in dfm.index if i in mouse2human_dict]
    dfm = dfm.loc[mouse_translable, :]
    dfm.index = [mouse2human_dict[i] for i in dfm.index]
    return dfm


def prefilter(df_f, filename, path):
    # df_f = dfpfc, filename = filename, path = path
    dropgene = open('%s%s' % (path, filename)).read().split('\n')
    other_genes = ['HBG1', 'HBA1', 'HBA2', 'HBE1', 'HBZ', 'BLVRB', 'S100A6', 'SNAR-E', 'SNAR-A13_loc1',
                   'SNAR-C1_loc1', 'SNAR-A1_loc2', 'SNAR-A8_loc1', 'SNAR-C1_loc2', 'SNAR-A2_loc2', 'SNAR-C4',
                   'SNAR-A12_loc1', 'SNAR-C3', 'SNAR-C1_loc3', 'SNAR-G2', 'SNAR-G1', 'SNAR-A11_loc9',
                   'SNAR-A6_loc3', 'SNAR-A14_loc7', 'SNAR-A6_loc5',
                   'SNAR-A10_loc6', 'SNAR-A5_loc9', 'SNAR-A14_loc3', 'SNAR-A9_loc9', 'SNAR-A11_loc7',
                   'SNAR-B1_loc1', 'SNAR-B1_loc2', 'SNAR-D', 'SNAR-F']
    dropgene.extend(other_genes)
    dropgene = list(set(dropgene))
    df_f = df_f.loc[~np.in1d(df_f.index, dropgene)].astype(float)
    df_f = df_f.loc[np.sum(df_f >= 1, 1) >= 5, :]  # is at least 1 in 5 cells
    df_f = df_f.loc[np.sum(df_f >= 2, 1) >= 2, :]  # is at least 2 in 2 cells
    df_f = df_f.loc[np.sum(df_f >= 3, 1) >= 1, :]  # is at least 3 in 1 cells
    return df_f


def MVgenes(df_f):
    # plotShow default= Ture
    mu = df_f.mean(1).values
    sigma = df_f.std(1, ddof=1).values
    cv = sigma / mu
    score, mu_linspace, cv_fit, params = CV_Mean(mu, cv, 'SVR', svr_gamma=0.003)
    mu_sorted = mu[np.argsort(score)[::-1]]
    cv_sorted = cv[np.argsort(score)[::-1]]
    y = cv_fit.tolist()
    x = mu_linspace.tolist()
    pars = thrscount(x, y)

    thrs = 0
    for i in range(len(np.log2(mu_sorted) > 0)):
        if i == 0:
            if func(np.log2(cv_sorted[i]), pars[0], pars[1], pars[2]) < np.log2(mu_sorted[i]):
                thrs = thrs + 1
        else:
            if np.log2(cv_sorted[i]) < np.log2(cv_sorted[i - 1]):
                if func(np.log2(cv_sorted[i]), pars[0], pars[1], pars[2]) < np.log2(mu_sorted[i]):
                    thrs = thrs + 1
    thrs = min(max(thrs * 10, 1000), 5000)
    # thrs=2210
    MVlist = df_f.iloc[np.argsort(score)[::-1], :].iloc[:thrs, :].index

    return mu, cv, sigma, score, mu_linspace, cv_fit, params, mu_sorted, cv_sorted, thrs, MVlist

def CV_Mean(mu, cv, fit_method='SVR', svr_gamma=0.003, x0=[0.5, 0.5], verbose=False):
    ### modified from BackSPIN, (GioeleLa Manno, et al., 2016, PMID: 27716510 )

    log2_m = np.log2(mu)
    log2_cv = np.log2(cv)

    if len(mu) > 1000 and 'bin' in fit_method:
        # histogram with 30 bins
        n, xi = histogram(log2_m, 30)
        med_n = percentile(n, 50)
        for i in range(0, len(n)):
            # index of genes within the ith bin
            ind = where((log2_m >= xi[i]) & (log2_m < xi[i + 1]))[0].astype(int)
            if len(ind) > med_n:
                # Downsample if count is more than median
                ind = ind[random.permutation(len(ind))]
                ind = ind[:len(ind) - int(med_n)]
                mask = ones(len(log2_m), dtype=bool)
                mask[ind] = False
                log2_m = log2_m[mask]
                log2_cv = log2_cv[mask]
            elif (around(med_n / len(ind)) > 1) and (len(ind) > 5):
                # Duplicate if count is less than median
                log2_m = r_[log2_m, tile(log2_m[ind], int(round(med_n / len(ind)) - 1))]
                log2_cv = r_[log2_cv, tile(log2_cv[ind], int(round(med_n / len(ind)) - 1))]
    else:
        if 'bin' in fit_method:
            print('More than 1000 input feature needed for bin correction.')
        pass

    if 'SVR' in fit_method:
        try:
            from sklearn.svm import SVR
            if svr_gamma == 'auto':
                svr_gamma = 1000. / len(mu)
            # Fit the Support Vector Regression
            clf = SVR(gamma=svr_gamma)
            clf.fit(log2_m[:, np.newaxis], log2_cv)
            fitted_fun = clf.predict
            score = np.log2(cv) - fitted_fun(np.log2(mu)[:, np.newaxis])
            params = None
            # The coordinates of the fitted curve
            mu_linspace = np.linspace(min(log2_m), max(log2_m))
            cv_fit = fitted_fun(mu_linspace[:, np.newaxis])
            return score, mu_linspace, cv_fit, params

        except ImportError:
            if verbose:
                print('SVR fit requires scikit-learn python library. Using exponential instead.')
            if 'bin' in fit_method:
                return fit_CV(mu, cv, fit_method='binExp', x0=x0)
            else:
                return fit_CV(mu, cv, fit_method='Exp', x0=x0)
    elif 'Exp' in fit_method:
        from scipy.optimize import minimize
        # Define the objective function to fit (least squares)
        fun = lambda x, log2_m, log2_cv: sum(abs(log2((2. ** log2_m) ** (-x[0]) + x[1]) - log2_cv))
        # Fit using Nelder-Mead algorythm
        optimization = minimize(fun, x0, args=(log2_m, log2_cv), method='Nelder-Mead')
        params = optimization.x
        # The fitted function
        fitted_fun = lambda log_mu: log2((2. ** log_mu) ** (-params[0]) + params[1])
        # Score is the relative position with respect of the fitted curve
        score = np.log2(cv) - fitted_fun(np.log2(mu))
        # The coordinates of the fitted curve
        mu_linspace = np.linspace(min(log2_m), max(log2_m))
        cv_fit = fitted_fun(mu_linspace)
        return score, mu_linspace, cv_fit, params



def thrscount(x, y):
    from scipy.optimize import curve_fit
    from scipy.stats.distributions import t
    initial_guess = [10, 0 - x[-1], y[-1]]
    pars, pcov = curve_fit(func, x, y, p0=initial_guess)
    alpha = 0.05  # 95% confidence interval = 100*(1-alpha)

    n = len(y)  # number of data points
    p = len(pars)  # number of parameters

    dof = max(0, n - p)  # number of degrees of freedom

    # student-t value for the dof and confidence level
    tval = t.ppf(1.0 - alpha / 2., dof)
    for i, j, var in zip(range(n), pars, np.diag(pcov)):
        sigma = var ** 0.5
        print('p{0}: {1} [{2}  {3}]'.format(i, j,
                                            j - sigma * tval,
                                            j + sigma * tval))
    return pars

def func(x, a, b, c):
    'nonlinear function in a and b to fit to data'
    return a * x * x + b * x + c


def enrichmentscoreBETA(dfpfcclus, df_dev, fc=1.25, shortcut=True):
    #dfpfcclus = dfpfcclus, df_dev = df_dev, fc = 1.25, shortcut = True
    dfgrp = dfpfcclus.iloc[-1:, :].T.join(df_dev.T, how="inner")

    dfmean = dfgrp.groupby(['Cluster']).mean()
    dfmedian = dfgrp.groupby(['Cluster']).median().T
    df_means = df_dev.mean(1)
    if shortcut == False:
        TotalNzCount = sum(dfgrp.iloc[:, 1:] > 0)
        grpNzCount = dfgrp.groupby(['Cluster']).agg(lambda x: x.ne(0).sum())
        RestNzCount = TotalNzCount - grpNzCount
        RatioNzCount = (grpNzCount + 0.1) / (RestNzCount + 0.1)
        df_means = df_means.loc[RatioNzCount.columns]
        df_fold = (dfmean + 0.01).div(df_means + 0.01, axis=1) ** 0.5
        # df_fold=dfmean.div(df_means,axis=1)
        EScore = df_fold[RatioNzCount.columns].fillna(0) * RatioNzCount
        EScore = EScore.T
        df_fold = df_fold.T.dropna()
        df_avgpos = df_means
        df_avgpos = df_avgpos.fillna(0.0)
        score00 = df_fold
        score10 = df_fold.multiply(df_avgpos, axis=0)
        ix00 = np.argsort(score00, 0)
        # ix05 = np.argsort( score05 , 0)
        ix10 = np.argsort(score10, 0)
        markers = defaultdict(set)
        N = int(len(df_fold.index) / len(df_fold.columns) * 3)

        for ct in df_fold.columns:
            markers[ct] |= set(df_fold.index[ix00.loc[:, ct][::-1]][:N])
            markers[ct] |= set(df_fold.index[ix10.loc[:, ct][::-1]][:N])

        RatioNzCount = RatioNzCount.T
        mkdict = {}
        for ct in df_fold.columns:
            temp = {}
            for num in range(min(3, int(len(df_fold.columns) / 4) + 1), len(df_fold.columns)):
                temp[num] = []
            for mk in markers[ct]:
                x = 0
                y = 0
                for ct2 in list(set(df_fold.columns) - set([ct])):
                    # if (score10.loc[mk,ct] >= float(score10.loc[mk,ct2])) & (EScore.loc[mk,ct] >= float(EScore.loc[mk,ct2]))&(ratiovalue.loc[mk,ct]>0.9)& (score10.loc[mk,ct] > 1) & (EScore.loc[mk,ct] > 1) :
                    if (score10.loc[mk, ct] >= float(score10.loc[mk, ct2]) * fc) & (
                            EScore.loc[mk, ct] >= float(EScore.loc[mk, ct2]) * fc):
                        x = x + 1
                    if (score10.loc[mk, ct] * fc < float(score10.loc[mk, ct2])) & (
                            EScore.loc[mk, ct] * fc < float(EScore.loc[mk, ct2])):
                        # if (score10.loc[mk,ct] < float(score10.loc[mk,ct2])) & (EScore.loc[mk,ct] < float(EScore.loc[mk,ct2])) &(ratiovalue.loc[mk,ct]<0.1)& (EScore.loc[mk,ct] < 0.1):
                        y = y + 1
                if x in list(range(min(3, int(len(df_fold.columns) / 4) + 1), len(df_fold.columns))):
                    temp[x].append(mk)
                if y in list(range(min(3, int(len(df_fold.columns) / 4) + 1), len(df_fold.columns))):
                    temp[y].append(mk)
                    # markers[ct2] -= set([mk])
            # for num in range(2,len(df_fold.columns)-1):
            mkdict[ct] = temp
            genelist = []
            grouplist = []
            numberlist = []
            for num in range(min(3, int(len(df_fold.columns) / 4) + 2), len(df_fold.columns)):
                for ct in df_fold.columns:
                    genelist.extend(mkdict[ct][num])
                    grouplist.extend([ct] * len(mkdict[ct][num]))
                    numberlist.extend([num] * len(mkdict[ct][num]))
            dfmk = pd.DataFrame([genelist, grouplist, numberlist])
            dfmk.columns = dfmk.iloc[0, :]
            dfmk = dfmk.T
            dfmk.columns = ["Gene", "Group", "Num"]
            dftest = EScore.loc[dfmk.index]
            dftest = dfmk.iloc[:, 1:].T.append(dftest.T)
            dftest = dftest.T.sort_values(by=['Group', 'Num'], ascending=[True, False])
            collist = []
            for item in score10.columns:
                collist.append("Expr_%s" % item)
            score10.columns = collist
            dftestnew = dftest.join(score10, how="inner")
            list_genes = list(set(dftestnew.index))
            return list_genes, dftestnew


    elif shortcut == True:
        # df_fold=(dfmean+0.01).div(df_means+0.01,axis=1)**0.5
        df_fold = dfmean.div(df_means, axis=1)
        # dfmean=dfgrp.groupby(['Cluster']).mean()
        # df_means = df_dev.mean(1)
        # df_fold=dfmean.div(df_means,axis=1)
        df_fold = df_fold.T.dropna()
        df_avgpos = df_means
        df_avgpos = df_avgpos.fillna(0)
        score00 = df_fold
        score10 = df_fold.multiply(df_avgpos, axis=0)

        ix00 = np.argsort(score00, 0)
        ix10 = np.argsort(score10, 0)
        markers = defaultdict(set)
        N = int(len(df_fold.index) / len(df_fold.columns) * 3)
        print(N)
        for ct in df_fold.columns:
            markers[ct] |= set(df_fold.index[ix00.loc[:, ct][::-1]][:N])
            markers[ct] |= set(df_fold.index[ix10.loc[:, ct][::-1]][:N])
        print(len(markers))
        genelist = []
        for ct in df_fold.columns:
            for mk in markers[ct]:
                for ct2 in list(set(df_fold.columns) - set([ct])):
                    if (score10.loc[mk, ct] >= float(score10.loc[mk, ct2])) & (
                            score00.loc[mk, ct] >= float(score00.loc[mk, ct2])) & (dfmedian.loc[mk, ct] > 0):
                        genelist.append(mk)
                    elif (score10.loc[mk, ct] < float(score10.loc[mk, ct2])) & (
                            score00.loc[mk, ct] < float(score00.loc[mk, ct2])) & (dfmedian.loc[mk, ct] <= 0):
                        genelist.append(mk)

        return genelist, df_fold




def MVgene_Scaling(list_genes, dfpfc, dfpfcclus,score, commongene, mprotogruop,tftable,thrs,
                   std_scaling=False, TPTT=10000, sharedMVgenes=None,
                   learninggroup="train"):
    if learninggroup == "train":
        df_dev_rev = dfpfc.iloc[np.argsort(score)[::-1], :].iloc[max(min(-thrs, -2000), -5000):, :]
        tflist = pd.read_table(tftable, index_col=0, header=0, sep="\t").index.tolist()
        sharedMVgenes = list(set(list_genes + tflist) - set(df_dev_rev.index))
        if  np.sum(TPTT) != 0:
            dfpfc = (dfpfc / dfpfc.sum()).multiply(TPTT, axis=0).fillna(0)
        if std_scaling == True:
            scalepfc = dfpfc.div(dfpfc.std(1), axis=0).dropna(0)
        scalepfc = dfpfc.astype(float).dropna(0)
        scalepfc = dfpfc.div(dfpfc.std(1), axis=0)
        scalepfc = scalepfc.dropna(0)
        dfpfc_dev = scalepfc.loc[set(scalepfc.index) & set(sharedMVgenes)].dropna()
        dfpfc_dev_log = np.log2(dfpfc_dev + 1)
        dfpfc_dev_all = dfpfc_dev_log.T.join(dfpfcclus.T, how="inner").dropna()
        bool1 = mprotogruop != "nan"
        mclasses_names, mclasses_index = np.unique(mprotogruop[bool1], return_inverse=True, return_counts=False)
        mtrain_index = mclasses_index
        mdf_train_set = dfpfc_dev_log.loc[:, bool1].copy()
        # mdf_train_set = dfpfc_dev_log.copy()
        mdf_train_set = mdf_train_set.loc[mdf_train_set.sum(1) > 0]
        sharedMVgenes = mdf_train_set.index.tolist()
        return mdf_train_set, mclasses_names, mtrain_index, sharedMVgenes
    elif learninggroup == "test":
        dfpfc = dfpfc.reindex(commongene).fillna(0)
        if np.sum(TPTT) != 0:
            dfpfc = (dfpfc / dfpfc.sum()).multiply(TPTT, axis=0).fillna(0)

        if std_scaling == True:
            scalegbm = dfpfc.div(dfpfc.std(1), axis=0).dropna(0)
        else:
            scalegbm = dfpfc.astype(float).dropna(0)

        dfgbm_dev = scalegbm.reindex(sharedMVgenes).fillna(0)
        dfgbm_dev_log = np.log2(dfgbm_dev + 1).fillna(0)
        # df_dev_gbm = df_dev_gbm.loc[mdf_train_set.index].fillna(0)
        dfgbm_dev_all = dfgbm_dev_log.T.join(dfpfcclus.loc["Cluster"].T, how="inner").T
        # dfgbmcol = dfgbm_dev_all.iloc[-1:, :]
        # dfgbm = dfgbm_dev_all.iloc[:-1, :]
        dfclpncol = dfgbm_dev_all.iloc[-1:, :]
        dfclpn = dfgbm_dev_all.iloc[:-1, :]
        protogruop = dfclpncol.loc["Cluster"].values
        bool1 = protogruop != 'none'
        classes_names, classes_index = np.unique(protogruop[bool1], return_inverse=True, return_counts=False)
        train_index = classes_index
        # dfgbm_train_set = dfgbm_dev_log.loc[:, bool1].copy()
        # train_index = classes_index
        # dfgbm_train_set = dfgbm_train_set.loc[dfgbm_train_set.sum(1) > 0]
        dfclpn = dfclpn.loc[:, bool1].copy()
        df_train_setclpn = dfclpn.loc[dfclpn.sum(1) > 0]
        df_train_setclpn = df_train_setclpn.reindex(sharedMVgenes).fillna(0)

        return df_train_setclpn, dfclpncol, protogruop