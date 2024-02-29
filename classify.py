"""
This code uses PCA to extract features from the classifier dataset and classify the reduced dataset using RF
"""

import os
import pandas
import numpy
import pickle
import random

from sklearn import preprocessing
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from math import trunc


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


def prepare(results_path, data_path, classifiers):
    """
    Reading in all the data files and classifiers
    :param results_path: Path to results
    :param data_path: Path to data files
    :param classifiers: List of classifiers
    :return: Data containing structural features, accuracies of input classifiers, and list of input files
    """
    data = None
    files = []
    for root, directories, file in os.walk(data_path, topdown=False):
        files.append(file)

    accuracies = pandas.DataFrame(columns=classifiers, index=files[0])
    for classifier in classifiers:
        data_path = os.path.join(results_path, classifier + '.csv')
        data = pandas.read_csv(data_path)
        data[data.columns[:-1]] = pandas.DataFrame(normalize(data.iloc[:, :-1]))

        # Adding index
        try:
            data['files'] = files[0]
        except:
            print(data_path)
            print(data)
            break
        data = data.set_index('files')

        accuracies[classifier] = data.iloc[:, -1]
    return data, accuracies, files[0]


def generate_bucket(all_accuracies, classifiers, rounding, data):
    """
    Generating bucket for each input, and the best accuracies
    :param all_accuracies: All accuracies for the input data files
    :param classifiers: List of classifiers
    :param rounding: Rounding value
    :param data: Data containing structural features
    :return: Bucket and the best accuracy for each input
    """
    acc = pandas.DataFrame([all_accuracies.idxmax(axis=1)] * len(classifiers)).T
    acc.columns = classifiers
    best_accuracy = all_accuracies.max(axis=1)

    # If the current classifier's accuracy is close enough to the best accuracy (diff accuracies <= rounding)
    # we set the label to be the current classifier to improve dataset balance
    for classifier in classifiers:
        cls_acc = all_accuracies.loc[:, classifier]
        dist = best_accuracy - cls_acc
        acc.loc[dist <= rounding / 100, classifier] = classifier

    # Create a bucket to store all "good" classifiers for each subject.
    # Note that rounding can affect bucket elements
    bucket = all_accuracies.idxmax(axis=1)
    bucket_size = []
    for row in range(acc.shape[0]):
        bucket.iloc[row] = list(acc.iloc[row, :].value_counts().index)
        bucket_size.append(len(bucket.iloc[row]))

    # Set the labels to be actual best classifiers
    data['class'] = all_accuracies.idxmax(axis=1)

    return bucket, best_accuracy


def calculate_accuracy(data, no_features, bucket, split, iterations, rounding):
    """
    Calculating the resulting accuracy of PCA+RF predicting the best classifier for each sample in test dataset
    :param data: Data containing structural features
    :param no_features: Number of features to be extracted
    :param bucket: Bucket for each sample
    :param split: Split size for train and test datasets
    :param iterations: Number of iterations to run classification
    :param rounding: Rounding value
    :return: Resulting accuracies for the number of iterations, best model, the worst accuracy and best prediction values
    """
    best_accuracy = 0
    best_prediction = []
    worst_accuracy = numpy.inf
    accuracies = []
    predictions = []
    for iteration in range(iterations):
        x_train, x_test, y_train, y_test = train_test_split(
            data.iloc[:, :-1], data.iloc[:, -1], test_size=split, stratify=data.iloc[:, -1], random_state=0)

        test_index = x_test.index

        # Feature extraction
        pca = PCA(n_components=no_features)
        x_train = pca.fit_transform(x_train)
        x_test = pca.transform(x_test)

        # Classification
        model = RandomForestClassifier(class_weight='balanced_subsample').fit(x_train, y_train)
        current_prediction = model.predict(x_test)

        # Calculate accuracy
        temp = []
        for idx, item in enumerate(test_index):
            temp.append(any(elem == current_prediction[idx] for elem in bucket.loc[item]))

        # Store the best and the worst results
        current_accuracy = numpy.sum(temp) / len(temp) * 100
        if current_accuracy > best_accuracy:
            best_model = model
            best_accuracy = current_accuracy
            best_prediction = current_prediction
            in_bucket = temp
            final = pandas.DataFrame(numpy.array(test_index).reshape(-1, 1))

        if current_accuracy < worst_accuracy:
            worst_accuracy = current_accuracy
            worst_prediction = current_prediction

        accuracies.append(current_accuracy)
        predictions.append(current_prediction)
    print('no_features: ' + str(no_features) + ', Rounding: ' + str(rounding) + '%, Best accuracy: ' +
          str(numpy.round(best_accuracy, 2)) + ', Average accuracy: ' + str(
        numpy.round(numpy.average(accuracies), 2)) +
          '±' + str(numpy.round(numpy.std(accuracies), 2)) + ', Worst accuracy: ' + str(
        numpy.round(worst_accuracy, 2)))

    return accuracies, best_prediction, worst_accuracy, best_model, final


def generate_report(final, best_prediction, all_accuracies, classifiers, bucket, best_overall, counter, out,
                    no_features, rounding, models_path, best_model, results_path):
    """
    Generating Table V of the paper
    :param final: Final output table
    :param best_prediction: Best predictions
    :param all_accuracies: All accuracies for the input data files
    :param classifiers: List of classifiers
    :param bucket: Bucket for each sample
    :param best_overall: Best overall classifier
    :param counter: Counter to keep track of results
    :param out: Output variable
    :param no_features: Number of features to be extracted
    :param rounding: Rounding value
    :param models_path: Path to models
    :param best_model: Best generated model
    :param results_path: Path to results
    :return: Updated output variable
    """
    random.seed(2)
    numpy.random.seed(2)
    final['P'] = pandas.DataFrame(best_prediction.reshape(-1, 1))
    temp_classifiers = pandas.DataFrame(all_accuracies.idxmax(axis=1))
    temp_accuracies = pandas.DataFrame(all_accuracies.max(axis=1))
    random_classifiers = numpy.random.choice(classifiers, final.shape[0])
    for row in range(final.shape[0]):
        subject = final.loc[row, 0]
        final.loc[row, 'B'] = ', '.join(bucket.loc[subject])
        final.loc[row, 'AP'] = all_accuracies.loc[subject, final.loc[row, 'P']]
        final.loc[row, 'A'] = temp_classifiers.loc[subject, 0]
        final.loc[row, 'AA'] = temp_accuracies.loc[subject, 0]
        final.loc[row, 'R'] = random_classifiers[row]
        final.loc[row, 'AR'] = all_accuracies.loc[subject, random_classifiers[row]]
        final.loc[row, best_overall] = all_accuracies.loc[subject, best_overall]

    final['AA - AP'] = final['AA'] - final['AP']
    final['AA - ' + best_overall] = final['AA'] - final[best_overall]
    final['AP - ' + best_overall] = final['AP'] - final[best_overall]

    if final['AP'].mean() >= final[best_overall].mean():
        out.loc[counter, 'no_features'] = no_features
        out.loc[counter, 'rounding'] = rounding
        out.loc[counter, 'mean(AA)'] = final['AA'].mean()
        out.loc[counter, 'mean(AP)'] = final['AP'].mean()
        out.loc[counter, best_overall] = final[best_overall].mean()
        out.loc[counter, 'max(AA - AP)'] = final['AA - AP'].max()
        out.loc[counter, 'max(AA - ' + best_overall + ')'] = final['AA - ' + best_overall].max()
        out.loc[counter, 'max(AP - ' + best_overall + ')'] = final['AP - ' + best_overall].max()
        out.loc[counter, 'mean(AA - AP)'] = final['AA - AP'].mean()
        out.loc[counter, 'mean(AA - ' + best_overall + ')'] = final['AA - ' + best_overall].mean()
        out.loc[counter, 'mean(AP - ' + best_overall + ')'] = final['AP - ' + best_overall].mean()
        out.loc[counter, 'min(AP - ' + best_overall + ')'] = final['AP - ' + best_overall].min()

        # Storing the trained model for future use
        filename = 'model_nF' + str(no_features) + '_rounding' + str(rounding) + '.dat'
        file = open(os.path.join(models_path, filename), 'wb')
        pickle.dump(best_model, file)
        file.close()

        counter = counter + 1

    final.to_csv(
        os.path.join(results_path, 'final_rounding' + str(rounding) + '_features' + str(no_features) + '.csv'),
        index=False)
    return out, counter


def store_results(out, results_path, rounding, paper_path, best_overall):
    """
    Store the results
    :param out: Output variable
    :param results_path: Path to results
    :param rounding: Rounding value
    :param paper_path: Path to paper
    :param best_overall: Best overall results
    :return: CSV file containing the results
    """
    if out.shape[0] > 0:
        final_best = pandas.read_csv(os.path.join(results_path, 'final_rounding' + str(rounding) + '_features' + str(
            trunc(out.loc[out.idxmax()['mean(AP)'], 'no_features'])) + '.csv'))

        out = out.round(4)
        out = out.drop('rounding', axis=1)
        final_best = final_best.round(4)
        final_best.iloc[:, 0] = final_best.iloc[:, 0].replace('.csv', '', regex=True)
        final_best.to_csv(os.path.join(paper_path, 'final_rounding' + str(rounding) + '_features' + str(
            trunc(out.loc[out.idxmax()['mean(AP)'], 'no_features'])) + '.csv'), index=False, sep=',')

        print('Best number of features: ' + str(trunc(out.loc[out.idxmax()['mean(AP)'], 'no_features'])))
        print('Average improvement:' + str(
            numpy.round(numpy.average(out.loc[:, 'mean(AP)'] - out.loc[:, best_overall]), 4))
              + '±' + str(numpy.round(numpy.std(out.loc[:, 'mean(AP)'] - out.loc[:, best_overall]), 4)))


def main():
    rounding = 1
    counter = 0
    iterations = 10
    split = 0.3

    results_path = os.path.join(os.getcwd(), 'results')
    data_path = os.path.join(os.getcwd(), 'eeg')
    models_path = os.path.join(os.getcwd(), 'models')
    paper_path = os.path.join(os.getcwd(), 'paper')

    classifiers = ['LogisticRegression(n_jobs=-1)',
                   'RidgeClassifier()',
                   'MLPClassifier(max_iter=500)',
                   'RandomForestClassifier(n_jobs=-1)',
                   'ExtraTreesClassifier(n_jobs=-1)',
                   'LinearDiscriminantAnalysis(solver=lsqr)']

    short_name = ['LR', 'RC', 'MLP', 'RF', 'ET', 'LDA']

    # Read in data and all accuracies
    data, all_accuracies, files = prepare(results_path, data_path, classifiers)
    rows, cols = data.shape
    out = pandas.DataFrame()

    all_accuracies.columns = short_name
    classifiers = short_name
    best_overall = all_accuracies.idxmax(axis=1).value_counts().index[0]

    # Create acc variable containing the same column (the best classifiers) x number of classifiers
    for no_features in (range(2, min(trunc((1 - split) * rows), cols))):
        bucket, best_accuracy = generate_bucket(all_accuracies, classifiers, rounding, data)
        accuracies, best_prediction, worst_accuracy, best_model, final = calculate_accuracy(data, no_features, bucket,
                                                                                            split, iterations, rounding)
        out, counter = generate_report(final, best_prediction, all_accuracies, classifiers, bucket, best_overall,
                                       counter, out, no_features, rounding, models_path, best_model, results_path)
        store_results(out, results_path, rounding, paper_path, best_overall)


if __name__ == "__main__":
    main()
