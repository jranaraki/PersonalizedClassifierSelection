---
title: "PersonalizedClassifierSelection"
author: "Javad Rahimipour Anaraki, Antonina Kolokolova, and Tom Chau"
date: '29/02/2024'
---

## Notice
This code is related to the "Personalized Classifier Selection for EEG-based Brain-Computer Interfaces" paper [LINK](https://www.mdpi.com/2073-431X/13/7/158). If you need more details and explanations about the algorithm, please contact Javad Rahimipour Anaraki.

## Use case
A systematic methodology for individual classifier selection, wherein structural characteristics of an EEG dataset are used to predict a classifier that will perform with high accuracy.

## Code
The code has three parts:
  - Convert the data
  - Generate features
  - Classify

### Convert the data
This code reads BCI2000 EDF files, applies ICA and down-sampling (160Hz -> 10Hz), concatenates three sessions of performing Task 2 (i.e. 4, 8, and 12), and finally stores the results to a CSV file for each participant. The original data and the paper for the BCI2000 dataset can be downloaded from https://physionet.org/content/eegmmidb/1.0.0/ and https://pubmed.ncbi.nlm.nih.gov/15188875/, respectively.

### Generate features
This code generates 41 structural features and forms a classifier dataset.

### Classify
This code uses PCA to extract features from the classifier dataset and classify the reduced dataset using RF.

### Statistical results
To generate statistical results, we used [KEEL](http://keel.es/) software to perform nonparametric statistical comparisons.

## Requirements
Install the requirements using: 

```
conda env create -f environment.yml
```

## Run
To run the code, run each step using the following commands:
```
python edf2csv.py
python generate.py
python classifiy.py
```
