
import seaborn as sns

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
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerPatch
#from numpy import in1d
today=f"{datetime.datetime.now():%Y-%m-%d-%I:%M%p}"



def AccuracyPlot( Xhigh, acc, accCutoff=0.95, Xlow=-1,Ylow=0.5, Yhigh=1,):
    fig_args = {'figsize': (6, 3), 'facecolor': 'white', 'edgecolor': 'white'}
    #acc = net.history[:, 'valid_acc'], accCutoff = 0.95,
    #Xlow = -1, Xhigh = len(net.history[:, 'valid_acc']) + 1,
    fig = plt.figure(**fig_args)
    ax = fig.add_subplot(111)
    ax.plot(np.array([abs(i) for i in range(Xhigh-1)]),np.array( acc ), c='k', lw=2 )

    ax.axhline( accCutoff, c='b' )
    #axvline( 35 , c='r')
    plt.ylabel('Accuracy Score', fontsize=15)
    plt.xlabel('Epoches', fontsize=15)
    plt.xlim( Xlow, Xhigh)
    plt.ylim(Ylow, Yhigh)
    plt.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return ax

####NOT READY!!!
####NOT READY!!!
####NOT READY!!!
def MVplot(mu, cv, mu_sorted,cv_sorted, thrs,
           Xlow=-8.5, Xhigh=6.5, Ylow=-2, Yhigh=6.5,alphaValue=0.2, sValue=10,
           fig_args={'figsize': (8, 8), 'facecolor': 'white', 'edgecolor': 'white'}):
    #mu = mu, cv = cv, mu_sorted = mu_sorted, cv_sorted = cv_sorted, thrs = thrs,
    #mu_linspace = mu_linspace, cv_fit = cv_fit,
    #Xlow = -8.5, Xhigh = 6.5, Ylow = -2, Yhigh = 6.5, alphaValue = 0.2, sValue = 10,

    fig = plt.figure(**fig_args)


    ax.scatter(np.log2(mu_sorted[thrs:]), np.log2(cv_sorted[thrs:]), marker='o', edgecolor='none', alpha=alphaValue, s=sValue,
               c='r')
    # x.plot(mu_linspace, cv_fit*1.1,'-k', linewidth=1, label='$FitCurve$')
    # plot(linspace(-9,7), -0.5*linspace(-9,7), '-r', label='$Poisson$')
    plt.ylabel('log2 CV')
    plt.xlabel('log2 mean')
    ax.grid(alpha=0.3)
    plt.xlim(-8.6, 6.5)
    plt.ylim(-2, 6.5)
    ax.legend(loc=1, fontsize=15)
    plt.gca().set_aspect(1.2)
    plt.grid(False)
    return ax
