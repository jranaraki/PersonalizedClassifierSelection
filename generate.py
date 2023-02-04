"""
Programmed by Javad Rahimipour Anaraki on 2021.04.13
Institute of Biomedical Engineering
University of Toronto
Rosebrugh Building, 164 College Street, Room 407
Toronto, Ontario M5S 3G9 Canada
Email: j [DOT] rahimipour [AT] utoronto [DOT] ca
Website: https://jranaraki.github.io/

This code generates 41 structural features and forms classifier dataset
"""

import os
import pandas
import numpy
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.stats import chisquare, entropy
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier, \
    RandomForestClassifier
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.naive_bayes import GaussianNB, BernoulliNB, ComplementNB, MultinomialNB
from sklearn.linear_model import RidgeClassifier, SGDClassifier, LogisticRegression


def kl_divergence(p, q):
    """
    Calculate KL divergence to two distributions
    :param p: First distribution
    :param q: Second distribution
    :return: KL divergence value
    """
    kl = numpy.sum(numpy.where(p != 0, p * numpy.log(p / q), 0))
    if numpy.isnan(kl) | numpy.isinf(kl):
        kl = 0
    return kl


def rademacher(input_vector):
    """
    Calculate Rademacher complexity
    :param input_vector: The input vector to calculate Rademacher complexity against
    :return: Rademacher complexity value
    """
    m = input_vector.shape[0]
    unq = input_vector.unique()
    sigma = numpy.random.randint(low=unq.min(), high=unq.max() + 1, size=m)
    rad = numpy.sum(numpy.multiply(numpy.equal(sigma, input_vector), 1) * (1 / len(unq))) / m
    return rad


def normalize(x):
    """
    Perform normalization on the input
    :param x: Input dataset
    :return: Normalized dataset
    """
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    normalized = pandas.DataFrame(x_scaled)
    return normalized


def calculate_features(data_path):
    """
    Calculate 41 structural characteristics of each dataset
    :param data_path: Input data path
    :return: A row containing 41 structural characteristics
    """
    out_row = []

    # Reading data file
    df = pandas.read_csv(data_path, header=None)

    # Normalizing dataset
    df.iloc[:, :-1] = normalize(df.iloc[:, :-1].values)

    # Remove zeroed columns
    df = df.loc[:, (df != 0).any(axis=0)]

    # Impute the dataset
    imputer = SimpleImputer(missing_values=numpy.nan, strategy='mean').fit(df)
    df = pandas.DataFrame(imputer.transform(df))

    # Add header to dataframe
    cols = []
    for h in range(df.shape[1]):
        cols.append('f' + str(h))
    df.columns = cols

    # Fix index
    data = df
    data = data.reset_index(drop=True)

    # Number of samples and features, and division
    m = df.shape[0]
    n = df.shape[1]
    out_row.append(m)
    out_row.append(n)
    out_row.append(m / n)

    # Number of unique classes
    out_row.append(df.iloc[:, -1].nunique())

    # Mean of all data
    out_row.append(((df.iloc[:, :-1]).mean()).mean())

    # STD of all data
    out_row.append(((df.iloc[:, :-1]).std()).mean())

    # Entropy of class
    ent = entropy(df.iloc[:, -1])

    if numpy.isfinite(ent):
        out_row.append(entropy(df.iloc[:, -1]))
    else:
        out_row.append(0)

    # Quantile (25% and 75%), and range
    seventy_five = numpy.percentile(df.iloc[:, -1], 75, interpolation='higher')
    twenty_five = numpy.percentile(df.iloc[:, -1], 25, interpolation='higher')
    out_row.append(seventy_five)
    out_row.append(twenty_five)
    out_row.append(seventy_five - twenty_five)

    # Chi-square of class
    chi, p = chisquare(df)
    out_row.append(chi[-1])
    # Average of Chi-square of all
    out_row.append(numpy.mean(chi[:-1]))
    # Median of Chi-square of all
    out_row.append(numpy.median(chi[:-1]))
    # Standard deviation of Chi-square of all
    out_row.append(numpy.std(chi[:-1]))
    # Min of Chi-square of all
    out_row.append(numpy.min(chi[:-1]))
    # Max of Chi-square of all
    out_row.append(numpy.max(chi[:-1]))

    # Pearson correlation
    corr = df.corr(method='pearson')
    # corr 2 class
    out_row.append((corr.iloc[-1, :]).mean())
    # corr 2 all
    out_row.append(((corr.iloc[:-1, :-1]).mean()).mean())

    # Kendall correlation
    corr = df.corr(method='kendall')
    # corr 2 class
    out_row.append((corr.iloc[-1, :]).mean())
    # corr 2 all
    out_row.append(((corr.iloc[:-1, :-1]).mean()).mean())

    # Spearman correlation
    corr = df.corr(method='spearman')
    # corr 2 class
    out_row.append((corr.iloc[-1, :]).mean())
    # corr 2 all
    out_row.append(((corr.iloc[:-1, :-1]).mean()).mean())

    # Covariance
    cov = df.cov()
    # cov 2 class
    out_row.append((cov.iloc[-1, :]).mean())
    # cov 2 all
    out_row.append(((cov.iloc[:-1, :-1]).mean()).mean())

    # KL Divergence
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
        # KL 2 all
        tmp = pandas.DataFrame(df.iloc[:, :-1])
        res = tmp.apply(lambda x: kl_divergence(x, val), axis=0)
        out_row.append(res.mean())
        # KL 2 class
        res = kl_divergence(df.iloc[:, -1], val)
        out_row.append(res)

    # Rademacher's complexity of class
    out_row.append(rademacher(df.iloc[:, -1]))

    # Convert to dataframe
    out_row = numpy.array(out_row)
    out_row = pandas.DataFrame(out_row.reshape(-1, len(out_row)))

    return out_row, data


def calculate_accuracy(df, cls):
    """
    Calculate classification accuracy using 10cv
    :param df: Input dataframe
    :param cls: Classifier's name
    :return: Accuracy
    """
    try:
        cv = cross_validate(cls, df.iloc[:, :-1], df.iloc[:, -1], cv=10)
        accuracy = numpy.mean(cv['test_score'])
        return cls, accuracy
    except:
        return numpy.NAN


def main():
    results_path = os.path.join(os.getcwd(), 'results')
    data_path = os.path.join(os.getcwd(), 'eeg')

    classifiers = [AdaBoostClassifier(), BernoulliNB(), ComplementNB(),
                   DecisionTreeClassifier(criterion="entropy"),
                   DecisionTreeClassifier(), ExtraTreesClassifier(n_jobs=-1),
                   GaussianNB(), GradientBoostingClassifier(),
                   GradientBoostingClassifier(criterion="squared_error"),
                   KNeighborsClassifier(n_jobs=-1, algorithm="ball_tree"),
                   KNeighborsClassifier(n_jobs=-1, algorithm="brute"),
                   KNeighborsClassifier(n_jobs=-1, algorithm="kd_tree"),
                   LinearDiscriminantAnalysis(solver="lsqr"),
                   LinearDiscriminantAnalysis(),
                   LogisticRegression(n_jobs=-1), LogisticRegression(n_jobs=-1, penalty="none"),
                   MLPClassifier(max_iter=500), MultinomialNB(), NearestCentroid(),
                   RandomForestClassifier(n_jobs=-1), RidgeClassifier(), SGDClassifier(n_jobs=-1)]

    features = ['m', 'n', 'm/n', 'nClass', 'meanAll', 'avgSTDAll', 'entropyClass', 'Q75', 'Q25', 'Q75-Q25',
                'ChiClass', 'avgChiAll', 'medChiAll', 'stdChiAll', 'minChiAll', 'maxChiAll', 'avgPearCorrClass',
                'avgPearCorrAll', 'avgKendCorrClass', 'avgKendCorrAll', 'avgSpeaCorrClass', 'avgSpeaCorrAll',
                'avgCovClass', 'avgCovAll', 'avgKLNormAll', 'avgKLUnifAll', 'avgKLLogiAll', 'avgKLExpoAll',
                'avgKLChiAll', 'avgKLRaylAll', 'avgKLParetAll', 'avgKLZipfAll', 'KLNormClass', 'KLUnifClass',
                'KLLogiClass', 'KLExpoClass', 'KLChiClass', 'KLRaylClass', 'KLParetClass', 'KLZipfClass', 'radComClass']

    out = pandas.DataFrame(columns=features)
    acc = pandas.DataFrame()

    for root, directories, files in os.walk(data_path, topdown=False):
        for i, name in enumerate(tqdm(files)):
            row, data = calculate_features(os.path.join(data_path, name))
            row.columns = features
            out = out.append(row, ignore_index=True)
            result = Parallel(n_jobs=-1)(delayed(calculate_accuracy)(data, eval(cls)) for cls in classifiers)

            for j in range(len(classifiers)):
                cls = str(result[j][0]).replace("'", '')
                acc.loc[i, cls] = result[j][1]

    for k in range(len(classifiers)):
        out['class'] = acc.loc[:, acc.columns[k]]
        out.to_csv(os.path.join(results_path, acc.columns[k] + '.csv'), index=False)


if __name__ == "__main__":
    main()
