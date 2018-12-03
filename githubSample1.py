#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 21:15:13 2018

@author: prithadawn
"""

"""
Compare different models to predict if the income of a user is below or above 50K from the Adult Data Set provided in:
https://archive.ics.uci.edu/ml/datasets/Adult
"""
import pandas
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import preprocessing
import time

DATA_LABELS = {
    "workclass": ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay",
                  "Never-worked"],

    "salary": ["<=50K", ">50K"],
    "education": ["Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm", "Assoc-voc", "9th",
                  "7th-8th", "12th", "Masters", "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool"],
    "marital-status": ["Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed",
                       "Married-spouse-absent", "Married-AF-spouse"],
    "occupation": ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty",
                   "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving",
                   "Priv-house-serv", "Protective-serv", "Armed-Forces"],
    "relationship": ["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"],
    "race": ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"],
    "sex": ["Female", "Male"],
    "native-country": ["United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany",
                       "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece", "South", "China", "Cuba", "Iran",
                       "Honduras", "Philippines", "Italy", "Poland", "Jamaica", "Vietnam", "Mexico", "Portugal",
                       "Ireland", "France", "Dominican-Republic", "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia",
                       "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand", "Yugoslavia", "El-Salvador",
                       "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands"]
    }

NAMES = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship",
         "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "salary"]


def substitute_unknown_values(data):
    """
    Substitute ? values in 'occupation, 'workclass' and 'native_country' with the most common value in those columns
    :param data:
    :return:
    """
    for klass in ["occupation", "native-country", "workclass"]:
        median_klass_value = int(data[klass].loc[data[klass] != "?"].median())
        data.loc[data[klass] == "?", klass] = median_klass_value
    return data


def process_data(data):
    """
    Convert data into numeric values. Returns data and labels
    :param data: DataFrame with the data
    :return: x: processed data, y: labels
    """

    for key, labels in DATA_LABELS.items():
        for index, label in enumerate(labels):
            # change label to numeric value
            data.loc[data[key] == label, key] = index

    data = substitute_unknown_values(data)

    # select data and labels
    x = data.drop("salary", axis=1)
    y = data["salary"]
    return x, y


def train_adult_data_set():
    # train data
    print("Processing training data")
    train_data = pandas.read_csv(open('adult.data'),  header=None, delimiter=' *, *', engine='python', names=NAMES)
    x_train, y_train = process_data(train_data)

    # test data
    print("Processing test data")
    test_data = pandas.read_csv(open('adult.test'),  header=None, delimiter=' *, *', engine='python', names=NAMES)
    # process for test data:
    # remove first line
    test_data = test_data.drop([0])
    # replace unnecessary '.' character in test data
    test_data['salary'] = test_data['salary'].str.replace('.', '')
    x_test, y_test = process_data(test_data)

    # train random forest classifier
    random_forest = RandomForestClassifier(n_estimators=100)
    print("\nTraining", random_forest)
    start_time = time.time()
    random_forest.fit(x_train, list(y_train))
    end_time = time.time()
    score = random_forest.score(x_test, list(y_test))
    print("\nAccuracy: ", score)
    print("Training time: ", end_time - start_time)

    # train KNN classifier
    nearest_neighbors = KNeighborsClassifier()
    print("\nTraining", nearest_neighbors)
    start_time = time.time()
    nearest_neighbors.fit(x_train, list(y_train))
    end_time = time.time()
    score = nearest_neighbors.score(x_test, list(y_test))
    print("\nAccuracy: ", score)
    print("Training time: ", end_time - start_time)

    # normalise data
    standard_scaler = preprocessing.StandardScaler()
    x_train = pandas.DataFrame(standard_scaler.fit_transform(x_train))
    x_test = pandas.DataFrame(standard_scaler.fit_transform(x_test))

    # train KNN with normalised data
    nearest_neighbors = KNeighborsClassifier()
    print("\nTraining", nearest_neighbors)
    start_time = time.time()
    nearest_neighbors.fit(x_train, list(y_train))
    end_time = time.time()
    score = nearest_neighbors.score(x_test, list(y_test))
    print("\nAccuracy: ", score)
    print("Training time: ", end_time - start_time)

    # train SVM classifier
    svc = SVC()
    print("\nTraining", svc)
    start_time = time.time()
    svc.fit(x_train, list(y_train))
    end_time = time.time()
    score = svc.score(x_test, list(y_test))
    print("\nAccuracy: ", score)
    print("Training time: ", end_time - start_time)

if __name__ == "__main__":
    train_adult_data_set()