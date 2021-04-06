#!/usr/bin/env python
# coding: utf-8

# In[38]:


# get_ipython().run_line_magic('config', 'Completer.use_jedi = False')

import warnings
import time
import os
import pandas
import numpy
import pickle
import itertools
import matplotlib.pyplot
import scipy.stats
import networkx

from tqdm import tqdm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from math import trunc


# In[39]:


def normalize(x):
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    out = pandas.DataFrame(x_scaled)
    return out


# In[40]:


def prepare(resultsPath, dataPath, classifiers):
    files = []
    for root, directories, file in os.walk(dataPath, topdown = False):
        files.append(file)
        
    acc = pandas.DataFrame(columns = classifiers, index = files[0])
    for cls in classifiers:  
        # Reading data
        dataFile = os.path.join(resultsPath, cls + '.csv')
        data = pandas.read_csv(dataFile)
        
        # Normalizing data
        data.iloc[:, :-1] = pandas.DataFrame(normalize(data.iloc[:, :-1]))

        # Adding index
        try:
            data['files'] = files[0]
        except:
            print(dataFile)
            print(data)
            break
        data = data.set_index('files')

        acc[cls] = data.iloc[:, -1]
    return data, acc, files[0]


# In[ ]:


warnings.filterwarnings("ignore")
resultsPath = os.path.join(os.getcwd(), 'results')
dataPath = os.path.join(os.getcwd(), 'eeg')
modelsPath = os.path.join(os.getcwd(), 'models')
expPath = os.path.join(os.getcwd(), 'experiments')

classifiers = ['AdaBoostClassifier()', 'BernoulliNB()', 'ComplementNB()',
       'DecisionTreeClassifier(criterion=entropy)', 'DecisionTreeClassifier()',
       'ExtraTreesClassifier(n_jobs=-1)', 'GaussianNB()',
       'GradientBoostingClassifier()',
       'GradientBoostingClassifier(criterion=mse)',
       'KNeighborsClassifier(algorithm=ball_tree, n_jobs=-1)',
       'KNeighborsClassifier(algorithm=brute, n_jobs=-1)',
       'KNeighborsClassifier(algorithm=kd_tree, n_jobs=-1)',
       'LinearDiscriminantAnalysis(solver=lsqr)',
       'LinearDiscriminantAnalysis()', 'LogisticRegression(n_jobs=-1)',
       'LogisticRegression(n_jobs=-1, penalty=none)',
       'MLPClassifier(max_iter=500)', 'MultinomialNB()', 'NearestCentroid()',
       'RandomForestClassifier(n_jobs=-1)', 'RidgeClassifier()',
       'SGDClassifier(n_jobs=-1)']

features = ['m', 'n', 'm/n', 'nClass', 'meanAll', 'avgSTDAll', 'entropyClass', 'Q75', 'Q25', 'Q75-Q25',
            'ChiClass', 'avgChiAll', 'medChiAll', 'stdChiAll', 'minChiAll', 'maxChiAll', 'avgPearCorrClass',
            'avgPearCorrAll', 'avgKendCorrClass', 'avgKendCorrAll', 'avgSpeaCorrClass', 'avgSpeaCorrAll',
            'avgCovClass', 'avgCovAll', 'avgKLNormAll', 'avgKLUnifAll', 'avgKLLogiAll', 'avgKLExpoAll',
            'avgKLChiAll', 'avgKLRaylAll', 'avgKLParetAll', 'avgKLZipfAll', 'KLNormClass', 'KLUnifClass',
            'KLLogiClass', 'KLExpoClass', 'KLChiClass', 'KLRaylClass', 'KLParetClass', 'KLZipfClass', 'radComClass', 'class']


classifiers = ['ExtraTreesClassifier(n_jobs=-1)',
               'LinearDiscriminantAnalysis(solver=lsqr)','MLPClassifier(max_iter=500)',
               'RidgeClassifier()','LogisticRegression(n_jobs=-1)', 'RandomForestClassifier(n_jobs=-1)']

rounding = 1
counter = 0
iterations = 50
split = 0.3

# Read in data and all accs
data, allAccs, files = prepare(resultsPath, dataPath, classifiers)
rows, cols = data.shape
avgStd = numpy.average(allAccs.std())
bestOverall = allAccs.idxmax(axis = 1).value_counts().index[0]
out = pandas.DataFrame()

for nFeatures in tqdm(range(2, min(trunc((1 - split) * rows), cols))):
    # Create acc variable containing the same column (best classifiers) x number of classifiers
    acc = pandas.DataFrame()
    acc = acc.append([allAccs.idxmax(axis = 1)] * len(classifiers)).T
    acc.columns = classifiers
    bestAcc = allAccs.max(axis = 1)

    # If the current classifier's accuracy is close enough to the best accuracy (diff accs <= rounding) 
    # we set the label to be the current classifier to improve dataset balance
    for classifier in classifiers:
        clsAcc = allAccs.loc[:, classifier]
        dist = bestAcc - clsAcc
        acc.loc[dist <= rounding/100, classifier] = classifier

    # Create a bucket to store all "good" classifiers for each subject. Note that rounding can affect bucket elements
    bucket = allAccs.idxmax(axis = 1)
    bucketSize = []
    for row in range(acc.shape[0]):
        bucket.iloc[row] = list(acc.iloc[row, :].value_counts().index)
        bucketSize.append(len(bucket.iloc[row]))

    # Set the labels to be actual best classifiers
    data['class'] = allAccs.idxmax(axis = 1)

    # Training & testing
    bestAcc = 0
    bestPred = []
    crntAcc = 0
    crntPred = []
    wrstAcc = numpy.inf
    wrstPred = []
    accs = []
    bestModel = []
    for iteration in range(iterations):
        X_train, X_test, y_train, y_test = train_test_split(
            data.iloc[:, :-1], data.iloc[:, -1], test_size = split, stratify = data.iloc[:, -1], random_state = 0)

        testIndex = X_test.index

        # Feature extraction
        pca = PCA(n_components = nFeatures)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

        # Classification
        model = RandomForestClassifier(class_weight = "balanced").fit(X_train, y_train)
        crntPred = model.predict(X_test)

        # Calculate accuracy
        temp = []
        for idx,item in enumerate(testIndex):
            temp.append(any(elem == crntPred[idx] for elem in bucket.loc[item]))

        # Store the best and the worst results
        crntAcc = numpy.sum(temp)/len(temp) * 100
        if crntAcc > bestAcc:
            bestModel = model
            bestAcc = crntAcc
            bestPred = crntPred
            inBucket = temp
            final = pandas.DataFrame(numpy.array(testIndex).reshape(-1,1))

        if crntAcc < wrstAcc:
            wrstAcc = crntAcc
            wrstPred = crntPred

        accs.append(crntAcc)

    #print('nFeatures: ' + str(nFeatures) + ', Rounding: ' + str(rounding) + '%, Best accuracy: ' +
    #      str(numpy.round(bestAcc, 2)) + ', Average accuracy: ' + str(numpy.round(numpy.average(accs), 2)) +
    #      '±' + str(numpy.round(numpy.std(accs), 2)) + ', Worst accuracy: ' + str(numpy.round(wrstAcc, 2)))

    # Report
    final['Predicted'] = pandas.DataFrame(bestPred.reshape(-1, 1))
    tempCls = pandas.DataFrame(allAccs.idxmax(axis = 1))
    tempAcc = pandas.DataFrame(allAccs.max(axis = 1))
    for row in range(final.shape[0]):
        subject = final.loc[row, 0]
        final.loc[row, 'Bucket'] = ''.join(bucket.loc[subject])
        final.loc[row, 'PA'] = allAccs.loc[subject, final.loc[row, 'Predicted']]
        final.loc[row, 'In bucket'] = inBucket[row]
        final.loc[row, 'Actual'] = tempCls.loc[subject, 0]
        final.loc[row, 'AA'] = tempAcc.loc[subject, 0]
        final.loc[row, bestOverall] = allAccs.loc[subject, bestOverall]

    final['AA - PA'] = final['AA'] - final['PA']
    final['AA - ' + bestOverall] = final['AA'] - final[bestOverall]
    final['PA - ' + bestOverall] = final['PA'] - final[bestOverall]

    if (final['PA'].mean() >= final[bestOverall].mean()):
        out.loc[counter, 'nFeature'] = nFeatures
        out.loc[counter, 'rounding'] = rounding
        out.loc[counter, 'mean(AA)'] = final['AA'].mean()
        out.loc[counter, 'mean(PA)'] = final['PA'].mean()
        out.loc[counter, bestOverall] = final[bestOverall].mean()
        out.loc[counter, 'max(AA - PA)'] = final['AA - PA'].max()
        out.loc[counter, 'max(AA - ' + bestOverall + ')'] = final['AA - ' + bestOverall].max()
        out.loc[counter, 'max(PA - ' + bestOverall + ')'] = final['PA - ' + bestOverall].max()
        out.loc[counter, 'mean(AA - PA)'] = final['AA - PA'].mean()
        out.loc[counter, 'mean(AA - ' + bestOverall + ')'] = final['AA - ' + bestOverall].mean()
        out.loc[counter, 'mean(PA - ' + bestOverall + ')'] = final['PA - ' + bestOverall].mean()
        out.loc[counter, 'min(PA - ' + bestOverall + ')'] = final['PA - ' + bestOverall].min()
        
        # Storing the trained model for future use
        filename = 'model_nF' + str(nFeatures) + '_rounding' + str(rounding) + '.dat'
        file = open(os.path.join(modelsPath, filename), 'wb')
        pickle.dump(bestModel, file)
        file.close()
        
        counter = counter + 1
    
    final.to_csv(os.path.join(resultsPath, 'final_rounding' + str(rounding) + '_features' + str(nFeatures) + '.csv'), index = False)

# Width setting of JupyterNotebook
pandas.options.display.max_columns = None
pandas.options.display.max_colwidth = None

# Displaying and storing the results
print('Best result in ' + str(iteration+1) + ' iterations')
finalBest = pandas.read_csv(os.path.join(resultsPath, 'final_rounding' + str(trunc(out.loc[out.idxmax()['mean(PA)'], 'rounding']))
                        + '_features' + str(trunc(out.loc[out.idxmax()['mean(PA)'], 'nFeature'])) + '.csv'))
#display(finalBest)
finalBest.to_csv(os.path.join(expPath, 'final_rounding' + str(trunc(out.loc[out.idxmax()['mean(PA)'], 'rounding']))
                        + '_features' + str(trunc(out.loc[out.idxmax()['mean(PA)'], 'nFeature'])) + '.csv'), index = False)

print('All Results')
#display(out)
out.to_csv(os.path.join(expPath, 'out_rounding' + str(rounding) + '.csv'), index = False)

# Plot bucket size
fig, ax = matplotlib.pyplot.subplots()
line1, = ax.plot(bucketSize)
ax.hlines(numpy.average(bucketSize), 0, len(bucketSize), colors='r')
ax.set_ylabel('Bucket size')
ax.set_xlabel('Subjects')
fig.suptitle('Bucket size for each subject with rounding ' + str(rounding) + '%, and average of ' 
         + str(numpy.round(numpy.average(bucketSize), 2)))
tmp = fig.draw
fig.savefig(os.path.join(expPath, 'bucket_rounding' + str(rounding) + '.eps'))


# In[ ]:




