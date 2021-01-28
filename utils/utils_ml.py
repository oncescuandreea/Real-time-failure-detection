# -*- coding: utf-8 -*-

import numpy as np  # maths equations
import random

from sklearn.model_selection import train_test_split
from tensorflow.python.keras import utils

from collections import Counter
import mysql.connector
import pandas


def tf_idf_retrieval(mycursor: mysql.connector.cursor):
    """
    This function retrieves tfidf scores for all documents
    inputs:
        sql database
    outputs:
        listTFIDF - list of TFIDF scores dictionaries associated with each doc
        indexname - list of document names in order corresponding to listTFIDF
        length - length of indexname list; number of documents available
    """
    # retrieve columns from dataframe, containing the key words corresponding
    # to tfidf scores
    sql = "SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA='final' AND TABLE_NAME='tfidfpd2'";
    mycursor.execute(sql)
    NameOfColumns = mycursor.fetchall()

    # retrieve tfidf table from sql
    sql = "Select * from tfidfpd2"
    mycursor.execute(sql)
    dictionaries = mycursor.fetchall()

    # lengthOfwordvec - Number of words in the TFIDF vector of each document
    lengthOfwordvec = len(dictionaries[0])

    # List containing all tfidf dictionaries for all reports
    listTFIDF = []

    # indexname retains the name of the reports being analysed in the order
    # they are retrieved in
    indexname = []

    for wordvec in dictionaries:
        tfidfdict = {}  # dictionary of one report
        indexname.append(wordvec[0])
        for i in range(1, lengthOfwordvec):
            tfidfdict.update({NameOfColumns[i - 1][0][1:]: wordvec[i]})
        listTFIDF.append(tfidfdict)
    length = len(dictionaries)  # number of documents
    return listTFIDF, indexname, length


def train_val_split(X_train_tot: np.ndarray, y_train_tot: np.ndarray, randomState: int):
    """
    Function which returns a split of the data into validation and test
    Random state is given such that the split of the actual labels and the NLP
    predicted ones is always consistent
    Inputs:
        X_train_tot - features extracted from all documents
        y_train_tot - labels corresponding to each document, either actual or
                    predicted
        randomState - to control the split and allow for reproducibility
    Outputs:
        X_train - features corresponding to the train documents
        X_val - features corresponding to the validation documents
        y_train - labels corresponding to the train documents
        y_val - labels corresponding to the validation documents
    """
    # randomState=np.random.random()

    X_train, X_val, y_train, y_val = train_test_split(X_train_tot, y_train_tot, test_size=0.25, random_state=randomState)

    return [X_train, X_val, y_train, y_val]


def categoric(y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray):
    """
    Function that transforms string labels into numbers for classification
    Inputs:
        y_train - labels corresponding to the train documents
        y_val - labels corresponding to the validation documents
        y_test - labels corresponding to the test documents
    Outputs:
        y_trainNN - numerical labels corresponding to the train documents
        y_valNN - numerical labels corresponding to the validation documents
        y_testNN - numerical labels corresponding to the test documents
    """
    y_trainNN = utils.to_categorical(y_train)
    y_valNN = utils.to_categorical(y_val)
    y_testNN = utils.to_categorical(y_test)

    return y_trainNN, y_valNN, y_testNN


def get_data_from_different_labels_for_cluster_initialisation(no_labeled_sets: int,
                                                              id2name: dict,
                                                              final: pandas.DataFrame,
                                                              indexname: list,
                                                              cnx: mysql.connector):
    """
    Function randomly chooses documents and the associated data. There will be
    no_labeled_sets of each type of document.
    Inputs:
        no_labeled_sets: how many documents of each type to be selected
        corrdict [dictionary] - dictionary relating the report ID and the
            report name
        final [117 x 268] - the dataframe containing the names of the reports
            and the corresponding tfidf vectors
        indexname - list relating the report index to the report name, the
            index being the order in which reports are accessed in sql
        cnx - to access the database
    Outputs:
        index_labels - index of documents to be used as train labels
        labels_tot - labels corresponding to the index of documents used for
            labeling
        cluster_center_average [noclasses x no_features] - TFIDF cluster vector
            with values given by averaging each vector location from each
            document from the corresponding labels. Used to initialise the
            Kmeans clustering
    """
    # extract all data form table an. it contains the id of the file and the
    # labels for each sensor
    mycursor = cnx.cursor()
    sqldataID = "select * from an"
    mycursor.execute(sqldataID)
    recorddataID = mycursor.fetchall()  # all labels and the ID for each doc
    number_of_reports = len(recorddataID) - 1  # number of reports

    no_labels = 7 * no_labeled_sets  # each type of document receives a  label
    labels_tot = []  # all labels associated with the randomly extracted documents
    no_labels_so_far = 0  # how many documents have been labeled
    index_labels = []  # the indexes of the randomly labelled documents, these
    # indexes correspond to the order of the documents in the dataframe
    label_only_once = []  # a list containing only one of each lable

    while no_labels_so_far < no_labels:

        random_sample_report = random.randint(0, number_of_reports)
        ID = recorddataID[random_sample_report][0]  # get ID of each recorded failure
        # create a string corresponding to the failure by adding the failure 
        # type and working words together
        label = recorddataID[random_sample_report][3] + \
                recorddataID[random_sample_report][1] + \
                recorddataID[random_sample_report][4] + \
                recorddataID[random_sample_report][2]

        report_name = id2name[ID]  # get the corresponding report name from ID

        # if the numbre of required labeled documents of each type has not been
        # reached, then add label to label_only_once if it has not been met 
        # before and add it to the list with all labels so far
        if labels_tot.count(label) < no_labeled_sets:
            if label not in label_only_once:
                label_only_once.append(label)
            labels_tot.append(label)
            no_labels_so_far += 1

            # append dataframe index of labeled document
            index_labels.append(indexname.index(report_name))

    # initialise list containing the cluster center values
    cluster_center_average = []

    # for each type of failure/document find all corresponding indexes and
    # add up the corresponding tfidf vectors
    for label in label_only_once:
        location_of_labels = [i for i, labell in enumerate(labels_tot) if
                              labell == label]
        sum_tfidf_scores = 0
        for location in location_of_labels:
            sum_tfidf_scores = sum_tfidf_scores + np.asarray(final.loc[index_labels[location], :])
        cluster_center_average.append(sum_tfidf_scores / no_labeled_sets)

    return [index_labels, labels_tot, cluster_center_average]


def features_list(mycursor: mysql.connector.cursor, ID: str):
    """
    Function created list of features containing all features from all sensors
    and the ID of that specific dataset.
    Inputs:
        mycursor - for accessing sql database
        ID - for accessing the corresponding features in the database
    Outputs:
        features - list of combined features from GSR, Acc, Hum, Temp sensors
    """
    dict_sensors = {'GSR': 'gsr3', 'Acc': 'FEAT3', 'Hum': 'hum4', 'Temp': 'tempd6'}

    features = []
    features.append(ID)
    for sensor in ['GSR', 'Acc', 'Hum', 'Temp']:
        sql = f"select * from {dict_sensors[sensor]} where ID='" + ID + "'"
        mycursor.execute(sql)
        data = mycursor.fetchall()
        for el in data[0][1:]:
            features.append(el)
    return features


def normalise(Xa: np.ndarray, length: int):
    """
    Function returns a normalised version of Xa. Features are normalsied
    Inputs:
        Xa [lengt x No_features] - matrix of feature vectors except the ID
        lengt - number of documents
    Outputs:
        XSVM [No_documents x No_features] - normalised matrix of features
    """
    Xa = np.asarray(Xa)
    Xa = Xa.astype(np.float64)
    Xnew = np.transpose(Xa)
    XSVM = np.empty([length, 73])
    for i in range(0, 73):
        mini = min(Xnew[i])
        maxi = max(Xnew[i])
        for j in range(0, length):
            XSVM[j][i] = (Xa[j][i] - mini) / (maxi - mini)
    return XSVM


def id_to_name(mycursor: mysql.connector.cursor):
    """
    Create dictionary relating the name and the id of the reports
    """
    id2name = {}
    sql = "select * from corr"
    mycursor.execute(sql)
    correlationdata = mycursor.fetchall()

    for row in correlationdata:
        id2name.update({row[0]: row[1]})
    return id2name


def labels_and_features(mycursor: mysql.connector.cursor, id2name: dict, reportName2cluster: dict,
                        length: int, cluster_to_labels: dict):
    """
    Extract from database features that will then be used to train machine learning algorithms. For the
    always working case, extract features form feat_working2 database and keep only the first 15 of them.
    Associate the workingworkingworkingworking label to them
    @param mycursor: mysql.connector.cursor
    @param id2name: dictionary with id name and the corresponding report name
    @param reportName2cluster: dictionary containing the report name and the corresponding kmeans cluster
    @param length: number of sets of features
    @param cluster_to_labels: dictionary of associations from kmeans clusters to best label match
    @return: arrays of true labels, nlp predicted labels and corresponding features
    """
    Xa = []
    sqldata_id = "select * from an"
    mycursor.execute(sqldata_id)
    recorddata_id = mycursor.fetchall()  # all features from all documents
    y_nlp = []  # Kmeans labels
    y = []  # manual labels

    for results in recorddata_id:
        ID = results[0]  # get ID of each recorded failure

        label_nlp = cluster_to_labels[reportName2cluster[id2name[ID]]]  # get cluster number from K means
        # create a string corresponding to the failure by adding the failure type and working words together
        label = results[3] + results[1] + results[4] + results[2]

        y_nlp.append(label_nlp)
        y.append(label)

        features = features_list(mycursor, ID)
        Xa.append(features[1:])

    # add to features and labels the ones for the case when everything works
    # choose only 15 from the table
    sql_data_working = "select * from feat_working2"
    mycursor.execute(sql_data_working)
    working_data = mycursor.fetchall()
    for i in range(0, 15):
        Xa.append(list(working_data[0][1:]))
        y_nlp.append(7)
        y.append('workingworkingworkingworking')

    X_svm = normalise(Xa, length + 15)
    return y, y_nlp, X_svm


def train_val_split_stratify(counter: int, inc: int, X_train_tot: np.ndarray, y_train_tot: np.ndarray,
                             X_train_tot_nlp: np.ndarray, y_train_tot_nlp: np.ndarray,
                             X_test: np.ndarray, y_test: np.ndarray, y_test_nlp: np.ndarray):
    """
    Function verifies if both the NLP and the actual labels split contains
    examples of each class. It increases inc until the random state selected
    splits the data correctly. The function then outputs the split data
    Inputs:
        counter - how many times the script was run and the split was performed
        inc - increment varies such that the split is correct
        X_traintot/NlP - train and validation features
        y_traintot/NLP - train and validation labels/predicted labels
        X_test/NLP - test features
        y_test/NLP - test labels/predicted labels
    Outputs:
        dictXy - containing the train/val/test features split and labels split
            for both actual labels and NLP infered ones and the categorical
            number associated with the labels
        counter - does not change
        inc - increment used for the random split

    """
    # set random state to be the same for both predicted and actual labels
    randomState = counter + inc

    dictXy = {}

    # split remaining data in train and validation data for actual labels
    [X_train, X_val, y_train, y_val] = train_val_split(X_train_tot, y_train_tot, randomState)

    # split remaining data in train and validation data for predicted labels
    [_, _, y_train_nlp, y_val_nlp] = train_val_split(X_train_tot_nlp, y_train_tot_nlp, randomState)

    # in case the selected random state does not distribute data evenly (some labels are not represented in the
    # train/validation set, increase randomState until the split is adequate)
    while len(Counter(y_train)) != len(Counter(y_val)) or len(Counter(y_train_nlp)) != len(Counter(y_val_nlp)):
        inc += 1
        randomState = counter + inc

        # split remaining data in train and validation data for actual labels
        [X_train, X_val, y_train, y_val] = train_val_split(X_train_tot, y_train_tot, randomState)

        # split remaining data in train and validation data for predicted labels
        [_, _, y_train_nlp, y_val_nlp] = train_val_split(X_train_tot_nlp, y_train_tot_nlp, randomState)

    # transform the actual labels into numbers
    [y_train_nn, y_val_nn, y_test_nn] = categoric(y_train, y_val, y_test)

    # do the same for NLP although not necessary in this case
    [y_train_nn_nlp, y_val_nn_nlp, y_test_nn_nlp] = categoric(y_train_nlp, y_val_nlp, y_test_nlp)

    dictXy['X_train'] = X_train
    dictXy['X_val'] = X_val
    dictXy['X_test'] = X_test

    dictXy['y_train'] = y_train
    dictXy['y_val'] = y_val
    dictXy['y_test'] = y_test

    dictXy['y_train_cat'] = y_train_nn
    dictXy['y_val_cat'] = y_val_nn
    dictXy['y_test_cat'] = y_test_nn

    dictXy['y_train_NLP'] = y_train_nlp
    dictXy['y_val_NLP'] = y_val_nlp
    dictXy['y_test_NLP'] = y_test_nlp

    dictXy['y_train_NLP_cat'] = y_train_nn_nlp
    dictXy['y_val_NLP_cat'] = y_val_nn_nlp
    dictXy['y_test_NLP_cat'] = y_test_nn_nlp

    return dictXy, counter, inc
