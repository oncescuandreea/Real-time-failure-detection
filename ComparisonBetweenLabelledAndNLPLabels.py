# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 11:05:54 2020

@author: Andreea
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 19:00:20 2020

@author: Andreea
"""

# In this script the results obtained by correctly labelling all reports
# and using clustering for labelling are compared

# same file as the previous version but use already extracted info to get the vectors for clustering
# input info needed:
# an sql table with tfidf scores and words
# folder with datasets   .csv
# correlation table sql already created  
# ============================================================================= 
# output results:
# tf-idf for each word within a document  
# ranking based on bow for each document 
# retain only 1st 7 words most informative in wordrep4
# directly added to sql as a table
# =============================================================================
# improved lemmatizer with tf-idf but manually implemented then select 15 words 
# after disregarding tf-idf scores less than 0.0019
# importing useful libraries

import random
import argparse
from collections import Counter

import matplotlib.pyplot as plt
import mysql.connector  # connect to database
import numpy as np  # maths equations
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import optimizers
from pathlib import Path

from utils import TFIDFretrieval
from utils import get_data_from_different_labels_for_cluster_initialisation
from utils import get_parameter_sets, get_parameters, train_nn, train_svm
from utils import kmeans_clustering, NLP_labels_analysis
from utils import id_to_name, labels_and_features, train_val_split_stratify
from utils import print_file_test, print_file_val, train_NB


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--provided_labels",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--sql_password",
        type=str,
    )
    parser.add_argument(
        "--results_folder",
        type=Path,
        default="C:/Users/oncescu/data/4yp",
    )
    args = parser.parse_args()
    # connect to database
    cnx = mysql.connector.connect(user='root', password=args.sql_password,
                                  host='127.0.0.1',
                                  database='final')

    # retrieve from the tfidf table the name of the words for which the scores were calculated
    # the names are stored in NameOfColumns
    mycursor = cnx.cursor()

    # retreive from the satme table the actual values corresponding to the words and for each
    # document create a dictionary with the word and the values. then create a dataframe to allow for visualisation of the tfidf vectors
    [listTFIDF, indexname, lengt] = TFIDFretrieval(mycursor)
    # final is the dataframe containing the names of the reports and the corresponding tfidf vectors. if add indexname to command below then index is named as the report pd.DataFrame(listTFIDF,indexname)
    final = pd.DataFrame(listTFIDF)
    id2name = id_to_name(mycursor)

    num_clusters = 7
    number_of_tests = args.provided_labels

    avg = 0

    list_of_params, list_of_params_svm = get_parameter_sets(number_of_tests)

    no_labeled_sets = number_of_tests
    tf.compat.v1.reset_default_graph()
    tf.random.set_seed(0)
    np.random.seed(0)
    random.seed(0)
    [test, _, arrayn] = \
        get_data_from_different_labels_for_cluster_initialisation(no_labeled_sets,
                                                                  id2name,
                                                                  final,
                                                                  indexname,
                                                                  cnx)

    [clusters, number_labels_provided] = kmeans_clustering(num_clusters,
                                                           no_labeled_sets,
                                                           arrayn,
                                                           final)

    # create dictionary of name of reports and cluster associated
    report_name_to_cluster = {}
    cluster_to_report_name = {}

    for count, el in enumerate(indexname):
        report_name_to_cluster.update({el: clusters[count]})

    f, newdir, cluster_to_labels = NLP_labels_analysis(mycursor,
                                                       num_clusters=num_clusters,
                                                       length=lengt,
                                                       clusters=clusters,
                                                       number_of_labels_provided=number_labels_provided,
                                                       test=test,
                                                       id2name=id2name,
                                                       results_folder=args.results_folder)

    mycursor = cnx.cursor()

    y, yNLP, X_normalised = labels_and_features(mycursor,
                                                id2name,
                                                report_name_to_cluster,
                                                lengt,
                                                cluster_to_labels)

    classnames, indices = np.unique(y, return_inverse=True)
    classnamesNLP, indicesNLP = np.unique(yNLP, return_inverse=True)

    counter = 0
    maxG = 0
    minG = 1

    X_traintot_nlp, X_test_nlp, y_traintot_nlp, y_test_nlp = train_test_split(X_normalised,
                                                                              indicesNLP,
                                                                              test_size=0.2,
                                                                              random_state=32)
    print(f'y_test_nlp is {y_test_nlp}', file=f)
    print(f'y_test_nlp count is {Counter(y_test_nlp)}', file=f)

    X_traintot, X_test, y_traintot, y_test = train_test_split(X_normalised,
                                                              indices,
                                                              test_size=0.2,
                                                              random_state=32)
    print(f'y_test_real is {y_test}', file=f)
    print(f'y_test_real count is {Counter(y_test)}', file=f)
    accuracy_nn_test_list = []  # list of NN test accuracies
    accuracy_nn_val_list = []  # list of NN validation accuracies
    accuracy_nn_test_list_nlp = []
    accuracy_nn_val_list_nlp = []
    accuracy_svm_test_list = []  # list of SVM accuracy on test data
    accuracy_svm_val_list = []  # list of SVM accuracy values on validation data
    accuracy_svm_test_list_nlp = []
    accuracy_svm_val_list_nlp = []

    minmax_nn = {}
    conf_matrix_nn = {}
    minmax_nn['max'] = 0
    minmax_nn['min'] = 1
    minmax_nn['maxv'] = 0
    minmax_nn['minv'] = 1

    minmax_nn_nlp = {}
    conf_matrix_nn_nlp = {}
    minmax_nn_nlp['max'] = 0
    minmax_nn_nlp['min'] = 1
    minmax_nn_nlp['maxv'] = 0
    minmax_nn_nlp['minv'] = 1

    minmax_svm = {}
    conf_matrix_svm = {}
    minmax_svm['max'] = 0
    minmax_svm['min'] = 1
    minmax_svm['maxv'] = 0
    minmax_svm['minv'] = 1

    minmax_svm_nlp = {}
    conf_matrix_svm_nlp = {}
    minmax_svm_nlp['max'] = 0
    minmax_svm_nlp['min'] = 1
    minmax_svm_nlp['maxv'] = 0
    minmax_svm_nlp['minv'] = 1

    dict_nn, dict_svm = get_parameters(random='False',
                                       list_of_params=list_of_params,
                                       list_of_params_svm=list_of_params_svm,
                                       number_of_tests=number_of_tests,
                                       file=f)

    inc = 1  # increment for setting random seed value and for making sure the split is correct
    noclasses = len(classnames)

    print(number_of_tests)

    while counter < 100:
        newfile = f"C:/Users/oncescu/coding/4yp/{counter}_{number_of_tests}.txt"
        fi = open(newfile, 'w')
        fi.close()
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=8,
                                                    restore_best_weights=True)
        [dictXy, counter, inc] = train_val_split_stratify(counter, inc,
                                                          X_traintot,
                                                          y_traintot,
                                                          X_traintot_nlp,
                                                          y_traintot_nlp,
                                                          X_test, X_test_nlp,
                                                          y_test, y_test_nlp)

        sgd = optimizers.SGD(lr=dict_nn['learning_rate'],
                             decay=dict_nn['decay_set'],
                             nesterov=False)  # used to be lr 0.01

        minmax_nn, conf_matrix_nn, history = train_nn(noclasses, dict_nn, sgd,
                                                      dictXy,
                                                      accuracy_nn_test_list,
                                                      callback,
                                                      accuracy_nn_val_list,
                                                      minmax_nn,
                                                      conf_matrix_nn,
                                                      label_type='NN')
        if minmax_nn == conf_matrix_nn == history:
            continue

        minmax_nn_nlp, conf_matrix_nn_nlp, history_nlp = train_nn(noclasses,
                                                                  dict_nn, sgd,
                                                                  dictXy,
                                                                  accuracy_nn_test_list_nlp,
                                                                  callback,
                                                                  accuracy_nn_val_list_nlp,
                                                                  minmax_nn_nlp,
                                                                  conf_matrix_nn_nlp,
                                                                  label_type='NLP')

        # =============================================================================

        minmax_svm, conf_matrix_svm = train_svm(dict_svm, dictXy,
                                                accuracy_svm_test_list,
                                                accuracy_svm_val_list,
                                                minmax_svm,
                                                conf_matrix_svm,
                                                label_type='NN')

        minmax_svm_nlp, conf_matrix_svm_nlp = train_svm(dict_svm, dictXy,
                                                        accuracy_svm_test_list_nlp,
                                                        accuracy_svm_val_list_nlp,
                                                        minmax_svm_nlp,
                                                        conf_matrix_svm_nlp,
                                                        label_type='NLP')

        # =============================================================================

        counter += 1

    # =============================================================================
    train_NB(X_traintot, y_traintot, X_traintot_nlp, y_traintot_nlp,
             X_test, y_test, X_test_nlp, y_test_nlp, f)
    # =============================================================================

    print_file_test('NN', 'real', f, minmax_nn, conf_matrix_nn,
                    accuracy_nn_test_list)
    print_file_test('NN', 'NLP', f, minmax_nn_nlp, conf_matrix_nn_nlp,
                    accuracy_nn_test_list_nlp)

    print(".............................", file=f)

    print_file_val('NN', 'real', f, minmax_nn, conf_matrix_nn,
                   accuracy_nn_val_list)
    print_file_val('NN', 'NLP', f, minmax_nn_nlp, conf_matrix_nn_nlp,
                   accuracy_nn_val_list_nlp)

    print(".............................", file=f)

    # =============================================================================

    print_file_test('SVM', 'real', f, minmax_svm, conf_matrix_svm,
                    accuracy_svm_test_list)
    print_file_test('SVM', 'NLP', f, minmax_svm_nlp, conf_matrix_svm_nlp,
                    accuracy_svm_test_list_nlp)

    print(".............................", file=f)

    print_file_val('SVM', 'real', f, minmax_svm, conf_matrix_svm,
                   accuracy_svm_val_list)
    print_file_val('SVM', 'NLP', f, minmax_svm_nlp, conf_matrix_svm_nlp,
                   accuracy_svm_val_list_nlp)

    f.close()
    # =============================================================================

    # plot accuracies for train and validation
    # =============================================================================
    plt.figure(1)  # added line compared to previous laptop
    # =============================================================================
    plt.plot(history.history['accuracy'])  # before it was accuracy/acc in between quotes
    plt.plot(history.history['val_accuracy'])  # before it was val_accuracy in between quotes)
    plt.title('Accuracy on train and validation data')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(newdir + "/NN_Accuracy_train_val.png", dpi=1200)
    # plt.show()

    # =============================================================================
    plt.figure(2)  # added line compared to previous laptop
    # =============================================================================
    plt.plot(history_nlp.history['accuracy'])  # accuracy in between quotes
    plt.plot(history_nlp.history['val_accuracy'])  # before it was val_accuracy in between quotes)
    plt.title('NLP Accuracy on train and validation data')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(newdir + "/NN_NLP_Accuracy_train_val.png", dpi=1200)
    # plt.show()

    # plot loss for train and validation
    # =============================================================================
    plt.figure(3)  # added line compared to previous laptop
    # =============================================================================
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss function value for train and validation data')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.savefig(newdir + "/NN_Loss_train_val.png", dpi=1200)
    # plt.show()

    # =============================================================================
    plt.figure(4)  # added line compared to previous laptop
    # =============================================================================
    plt.plot(history_nlp.history['loss'])
    plt.plot(history_nlp.history['val_loss'])
    plt.title('NLPLoss function value for train and validation data')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.savefig(newdir + "/NN_NLP_Loss_train_val.png", dpi=1200)
    # plt.show()

    # =============================================================================
    plt.figure(5)  # added line compared to previous laptop
    # =============================================================================

    plt.plot(accuracy_nn_val_list)
    plt.plot(accuracy_nn_test_list)
    plt.title('NN accuracy on validation and test data given real labels')
    plt.ylabel('Accuracy')
    plt.xlabel('Number of runs')
    plt.legend(['validation', 'test'], loc='upper left')
    plt.savefig(newdir + "/NN_Validation_vs_test.png", dpi=1200)
    # plt.show()

    plt.plot(accuracy_svm_val_list)
    plt.plot(accuracy_svm_test_list)
    plt.title('SVM accuracy on validation and test data')
    plt.ylabel('Accuracy')
    plt.xlabel('Number of runs')
    plt.legend(['validation', 'test'], loc='upper left')
    plt.savefig(newdir + "/SVM_Validation_vs_test.png", dpi=1200)
    # plt.show()

    # =============================================================================
    # print(f"Average accuracy on validation is: {avg}")
    # print(f"Directory is: {name}")
    # =============================================================================
    # =============================================================================
    plt.figure(6)  # added line compared to previous laptop
    # =============================================================================
    plt.plot(accuracy_svm_val_list_nlp)
    plt.plot(accuracy_svm_test_list_nlp)
    plt.title('NLP SVM accuracy on validation and test data')
    plt.ylabel('Accuracy')
    plt.xlabel('Number of runs')
    plt.legend(['validation', 'test'], loc='upper left')
    plt.savefig(newdir + "/SVM_NLP_Validation_vs_test.png", dpi=1200)
    # plt.show()
    # number_of_tests -= 1


if __name__ == "__main__":
    main()
