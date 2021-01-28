# -*- coding: utf-8 -*-

# In this script the results obtained by correctly labelling all reports
# and using clustering for labelling are compared

# input info needed:
# an sql table with tfidf scores and words
# sql tables with extracted features for sensors
# correlation table sql containing id, name of report and name of dataset file
# ============================================================================= 
# output results:
# .txt files with how good the nlp clustering is whether supervised, semi-supervised or unsupervised
# best and worst accuracies and corresponding confusion matrices
# plots of accuracy and loss functions during training
# =============================================================================
# importing useful libraries

import random
import argparse
from collections import Counter

import mysql.connector  # connect to database
import numpy as np  # maths equations
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import optimizers
from pathlib import Path

from utils.utils_ml import tf_idf_retrieval
from utils.utils_ml import get_data_from_different_labels_for_cluster_initialisation
from utils.utils_ml import id_to_name, labels_and_features, train_val_split_stratify
from utils.utils_hyperparameters import get_parameter_sets, get_parameters
from utils.utils_nlp import kmeans_clustering, nlp_labels_analysis
from utils.utils_train import train_nn, train_svm, train_nb
from utils.utils_print import plot_ml_results, print_summary_of_results


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
                                  database='final_bare')

    # retrieve from the tfidf table the name of the words for which the scores were calculated
    # the names are stored in NameOfColumns
    mycursor = cnx.cursor()

    # retreive from the satme table the actual values corresponding to the words and for each document create a
    # dictionary with the word and the values. then create a dataframe to allow for visualisation of the tfidf vectors
    [listTFIDF, indexname, lengt] = tf_idf_retrieval(mycursor)
    # final is the dataframe containing the names of the reports and the corresponding tfidf vectors. if add
    # indexname to command below then index is named as the report pd.DataFrame(listTFIDF, indexname)
    final = pd.DataFrame(listTFIDF)
    id2name = id_to_name(mycursor)

    num_clusters = 7
    number_of_tests = args.provided_labels

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

    for count, el in enumerate(indexname):
        report_name_to_cluster.update({el: clusters[count]})

    f, newdir, cluster_to_labels = nlp_labels_analysis(mycursor,
                                                       num_clusters=num_clusters,
                                                       length=lengt,
                                                       clusters=clusters,
                                                       number_of_labels_provided=number_labels_provided,
                                                       test=test,
                                                       id2name=id2name,
                                                       results_folder=args.results_folder)

    mycursor = cnx.cursor()

    y, y_nlp, X_normalised = labels_and_features(mycursor,
                                                 id2name,
                                                 report_name_to_cluster,
                                                 lengt,
                                                 cluster_to_labels)

    classnames, indices = np.unique(y, return_inverse=True)
    classnames_nlp, indices_nlp = np.unique(y_nlp, return_inverse=True)

    counter = 0

    X_train_tot_nlp, X_test_nlp, y_train_tot_nlp, y_test_nlp = train_test_split(X_normalised,
                                                                                indices_nlp,
                                                                                test_size=0.2,
                                                                                random_state=32)
    print(f'y_test_nlp is {y_test_nlp}', file=f)
    print(f'y_test_nlp count is {Counter(y_test_nlp)}', file=f)

    X_train_tot, X_test, y_train_tot, y_test = train_test_split(X_normalised,
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

    minmax_nn = {'max': 0, 'min': 1, 'maxv': 0, 'minv': 0}
    conf_matrix_nn = {}

    minmax_nn_nlp = {'max': 0, 'min': 1, 'maxv': 0, 'minv': 0}
    conf_matrix_nn_nlp = {}

    minmax_svm = {'max': 0, 'min': 1, 'maxv': 0, 'minv': 0}
    conf_matrix_svm = {}

    minmax_svm_nlp = {'max': 0, 'min': 1, 'maxv': 0, 'minv': 0}
    conf_matrix_svm_nlp = {}

    dict_nn, dict_svm = get_parameters(random_bool=False,
                                       list_of_params=list_of_params,
                                       list_of_params_svm=list_of_params_svm,
                                       number_of_tests=number_of_tests,
                                       file=f)

    inc = 1  # increment for setting random seed value and for making sure the split is correct
    noclasses = len(classnames)

    print(number_of_tests)

    while counter < 100:
        newfile = str(args.results_folder / f"{counter}_{number_of_tests}.txt")
        fi = open(newfile, 'w')
        fi.close()
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=8,
                                                    restore_best_weights=True)
        [dictXy, counter, inc] = train_val_split_stratify(counter, inc,
                                                          X_train_tot,
                                                          y_train_tot,
                                                          X_train_tot_nlp,
                                                          y_train_tot_nlp,
                                                          X_test,
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
    train_nb(X_train_tot, y_train_tot, X_train_tot_nlp, y_train_tot_nlp,
             X_test, y_test, X_test_nlp, f)
    # =============================================================================

    print_summary_of_results(f, minmax_nn, conf_matrix_nn, accuracy_nn_test_list,
                             minmax_nn_nlp, conf_matrix_nn_nlp, accuracy_nn_test_list_nlp,
                             accuracy_nn_val_list, accuracy_nn_val_list_nlp, minmax_svm,
                             conf_matrix_svm, accuracy_svm_test_list, minmax_svm_nlp,
                             conf_matrix_svm_nlp, accuracy_svm_test_list_nlp,
                             accuracy_svm_val_list, accuracy_svm_val_list_nlp)
    plot_ml_results(history_nlp, history, accuracy_nn_val_list, accuracy_nn_test_list,
                    accuracy_svm_val_list, accuracy_svm_test_list,
                    accuracy_svm_val_list_nlp, accuracy_svm_test_list_nlp, newdir)


if __name__ == "__main__":
    main()
