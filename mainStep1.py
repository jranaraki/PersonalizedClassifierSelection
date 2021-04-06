#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().run_line_magic('config', 'Completer.use_jedi = False')

import warnings
import time
import os
import pandas
import numpy
import tempfile
from joblib import Parallel, delayed
from sys import stdout
from math import log, e
from tqdm import tqdm
from scipy.stats import chisquare, entropy
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier, NearestCentroid
from sklearn.naive_bayes import GaussianNB, BernoulliNB, ComplementNB, MultinomialNB
from sklearn.linear_model import RidgeClassifier, LinearRegression, SGDClassifier, LogisticRegression
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessClassifier


# In[2]:


def kl_divergence(p, q):
    out = numpy.sum(numpy.where(p != 0, p * numpy.log(p / q), 0))
    if numpy.isnan(out) | numpy.isinf(out):
        out = 0
    return out


# In[3]:


def rademacher(loss):
    m = loss.shape[0]
    unq = loss.unique()
    sigma = numpy.random.randint(low = unq.min(), high = unq.max()+1, size = m)
    out = numpy.sum(numpy.multiply(numpy.equal(sigma, loss), 1) * (1/len(unq))) / m
    return out


# In[4]:


def normalize(x):
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    out = pandas.DataFrame(x_scaled)
    return out


# In[5]:


def calFeatures(dfPath):
    row = []
    
    # Reading data file
    df = pandas.read_csv(dfPath, header = None)
    
    # Normalizing dataset
    df.iloc[:,:-1] = normalize(df.iloc[:,:-1].values)

    #Remove zeroed columns
    df = df.loc[:, (df != 0).any(axis=0)]

    #Impute the dataset
    imputer = SimpleImputer(missing_values = numpy.nan, strategy = 'mean')
    imputer = imputer.fit(df) 
    df = pandas.DataFrame(imputer.transform(df))    

    #Add header to dataframe   
    cols = []
    for i in range(df.shape[1]):
        cols.append('f' + str(i))
    df.columns = cols
    
    # Fix index
    data = df
    data = data.reset_index(drop=True)
    
    #Number of samples and features, and division
    m = df.shape[0]
    n = df.shape[1]
    row.append(m)
    row.append(n)
    row.append(m/n)

    #Number of unique classes
    row.append(df.iloc[:,-1].nunique())

    #Mean of all data
    row.append(((df.iloc[:,:-1]).mean()).mean())
    
    #STD of all data
    row.append(((df.iloc[:,:-1]).std()).mean())

    #Entropy of class
    ent = entropy(df.iloc[:,-1])
    
    if numpy.isfinite(ent):
        row.append(entropy(df.iloc[:,-1]))
    else:
        row.append(0)

    #Quantile (25% and 75%), and range
    seventyfive = numpy.percentile(df.iloc[:,-1], 75, interpolation='higher')
    twentyfive = numpy.percentile(df.iloc[:,-1], 25, interpolation='higher')
    row.append(seventyfive)
    row.append(twentyfive)
    row.append(seventyfive - twentyfive)

    #Chisquare of class
    chi, p = chisquare(df)        
    row.append(chi[-1])
    #Average of Chisquare of all  
    row.append(numpy.mean(chi[:-1]))
    #Median of Chisquare of all  
    row.append(numpy.median(chi[:-1]))
    #Standard deviation of Chisquare of all  
    row.append(numpy.std(chi[:-1]))
    #Min of Chisquare of all  
    row.append(numpy.min(chi[:-1]))
    #Max of Chisquare of all  
    row.append(numpy.max(chi[:-1]))

    #Pearson correlation
    corr = df.corr(method='pearson')
    #corr 2 class
    row.append((corr.iloc[-1,:]).mean())
    #corr 2 all
    row.append(((corr.iloc[:-1,:-1]).mean()).mean())

    #Kendall correlation
    corr = df.corr(method='kendall')
    #corr 2 class
    row.append((corr.iloc[-1,:]).mean())
    #corr 2 all
    row.append(((corr.iloc[:-1,:-1]).mean()).mean())

    #Spearman correlation
    corr = df.corr(method='spearman')
    #corr 2 class
    row.append((corr.iloc[-1,:]).mean())
    #corr 2 all
    row.append(((corr.iloc[:-1,:-1]).mean()).mean())

    #Covariance
    cov = df.cov()
    #cov 2 class
    row.append((cov.iloc[-1,:]).mean())
    #cov 2 all
    row.append(((cov.iloc[:-1,:-1]).mean()).mean())
    
    #KL_Divergence
    nd = numpy.random.normal(loc=1, scale=2, size=m)
    ud = numpy.random.uniform(size=m)
    ld = numpy.random.logistic(loc=1, scale=2, size=m)
    ed = numpy.random.exponential(scale=2, size=m)
    cd = numpy.random.chisquare(df=2, size=m)
    rd = numpy.random.rayleigh(scale=2, size=m)
    pd = numpy.random.pareto(a=2, size=m)
    zd = numpy.random.zipf(a=2, size=m)
    distros = ['nd', 'ud', 'ld', 'ed', 'cd', 'rd', 'pd', 'zd']
    
    for distro in distros:   
        val = eval(distro)
        #KL 2 all
        tmp = pandas.DataFrame(df.iloc[:, :-1])       
        res = tmp.apply(lambda x: kl_divergence(x, val), axis = 0)
        row.append(res.mean())
        #KL 2 class
        res = kl_divergence(df.iloc[:, -1], val)
        row.append(res)

    #Rademacher complexity of class
    row.append(rademacher(df.iloc[:, -1]))
    
    #Convert to dataframe
    row = numpy.array(row)
    row = pandas.DataFrame(row.reshape(-1, len(row)))
    
    return row, data


# In[6]:


def calAccuracy(df, cls):
     try:
        out = cross_validate(cls, data.iloc[:, :-1], data.iloc[:, -1], cv = 10)
        result = numpy.mean(out['test_score'])
        return cls, result
     except:
        return numpy.NAN


# In[7]:


# Setting parameters
warnings.filterwarnings("ignore")
resultsPath = os.path.join(os.getcwd(), 'results')
dataPath = os.path.join(os.getcwd(), 'eeg')
finalData = pandas.DataFrame()
featureSelection = False

classifiers = ['AdaBoostClassifier()', 'BernoulliNB()', 'ComplementNB()',
       'DecisionTreeClassifier(criterion="entropy")',
       'DecisionTreeClassifier()', 'ExtraTreesClassifier(n_jobs=-1)',
       'GaussianNB()', 'GradientBoostingClassifier()',
       'GradientBoostingClassifier(criterion="mse")',
       'KNeighborsClassifier(n_jobs=-1, algorithm="ball_tree")',
       'KNeighborsClassifier(n_jobs=-1, algorithm="brute")',
       'KNeighborsClassifier(n_jobs=-1, algorithm="kd_tree")',
       'LinearDiscriminantAnalysis(solver="lsqr")',
       'LinearDiscriminantAnalysis()',
       'LogisticRegression(n_jobs=-1)', 'LogisticRegression(n_jobs=-1, penalty="none")',
       'MLPClassifier(max_iter=500)', 'MultinomialNB()', 'NearestCentroid()',
       'RandomForestClassifier(n_jobs=-1)', 'RidgeClassifier()', 'SGDClassifier(n_jobs=-1)']


features = ['m', 'n', 'm/n', 'nClass', 'meanAll', 'avgSTDAll', 'entropyClass', 'Q75', 'Q25', 'Q75-Q25',
              'ChiClass', 'avgChiAll', 'medChiAll', 'stdChiAll', 'minChiAll', 'maxChiAll', 'avgPearCorrClass',
              'avgPearCorrAll', 'avgKendCorrClass', 'avgKendCorrAll', 'avgSpeaCorrClass', 'avgSpeaCorrAll',
              'avgCovClass', 'avgCovAll', 'avgKLNormAll', 'avgKLUnifAll', 'avgKLLogiAll', 'avgKLExpoAll',
              'avgKLChiAll', 'avgKLRaylAll', 'avgKLParetAll', 'avgKLZipfAll', 'KLNormClass', 'KLUnifClass',
              'KLLogiClass', 'KLExpoClass', 'KLChiClass', 'KLRaylClass', 'KLParetClass', 'KLZipfClass', 'radComClass']


# In[8]:


out = pandas.DataFrame(columns = features)
acc = pandas.DataFrame()

for root, directories, files in os.walk(dataPath, topdown = False):  
    for i,name in enumerate(tqdm(files)):
#         print('* ' + name)
        row, data = calFeatures(os.path.join(dataPath, name))
        row.columns = features
        out = out.append(row, ignore_index=True)
        result = Parallel(n_jobs=-1)(delayed(calAccuracy)(data, eval(cls)) for cls in classifiers)
        
        for j in range(len(classifiers)):
            cls = str(result[j][0]).replace("'", '')
#             print(cls)
            acc.loc[i, cls] = result[j][1]

for k in range(len(classifiers)):
    out['class'] = acc.loc[:, acc.columns[k]]
    out.to_csv(os.path.join(resultsPath, acc.columns[k] + '.csv'), index=False)

