from utils.utils_ml import features_list, normalise
import mysql.connector
import argparse
from pathlib import Path
import tensorflow as tf
import random
import numpy as np
from utils.utils_hyperparameters import get_parameter_sets, get_parameters
from collections import Counter
from datetime import datetime
import os
from utils.utils_train import train_nn, train_svm, train_nb
from sklearn.model_selection import train_test_split
from utils.utils_ml import train_val_split, categoric
from tensorflow.keras import optimizers
import _io
from utils.utils_print import print_file_test, print_file_val
import matplotlib.pyplot as plt


def labels_and_features(mycursor: mysql.connector.cursor, length: int):
    """
    Extract from database features that will then be used to train machine learning algorithms. For the
    always working case, extract features form feat_working2 database and keep only the first 15 of them.
    Associate the workingworkingworkingworking label to them
    @param mycursor: mysql.connector.cursor
    @param length: number of sets of features
    @return: arrays of true labels, nlp predicted labels and corresponding features
    """
    Xa = []
    sqldata_id = "select * from an"
    mycursor.execute(sqldata_id)
    recorddata_id = mycursor.fetchall()  # all features from all documents
    y = []  # manual labels

    for results in recorddata_id:
        ID = results[0]  # get ID of each recorded failure

        # create a string corresponding to the failure by adding the failure type and working words together
        label = results[3] + results[1] + results[4] + results[2]

        y.append(label)

        features = features_list(mycursor, ID)
        Xa.append(features[1:])

    X_svm = normalise(Xa, length)
    return y, X_svm


def train_val_split_stratify(counter: int, inc: int, X_train_tot: np.ndarray, y_train_tot: np.ndarray,
                             X_test: np.ndarray, y_test: np.ndarray):
    """
    Function verifies if both the NLP and the actual labels split contains
    examples of each class. It increases inc until the random state selected
    splits the data correctly. The function then outputs the split data
    Inputs:
        counter - how many times the script was run and the split was performed
        inc - increment varies such that the split is correct
        X_traintot - train and validation features
        y_traintot - train and validation labels/predicted labels
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

    # in case the selected random state does not distribute data evenly (some labels are not represented in the
    # train/validation set, increase randomState until the split is adequate)
    while len(Counter(y_train)) != len(Counter(y_val)):
        inc += 1
        randomState = counter + inc

        # split remaining data in train and validation data for actual labels
        [X_train, X_val, y_train, y_val] = train_val_split(X_train_tot, y_train_tot, randomState)

    # transform the actual labels into numbers
    [y_train_nn, y_val_nn, y_test_nn] = categoric(y_train, y_val, y_test)

    dictXy['X_train'] = X_train
    dictXy['X_val'] = X_val
    dictXy['X_test'] = X_test

    dictXy['y_train'] = y_train
    dictXy['y_val'] = y_val
    dictXy['y_test'] = y_test

    dictXy['y_train_cat'] = y_train_nn
    dictXy['y_val_cat'] = y_val_nn
    dictXy['y_test_cat'] = y_test_nn

    return dictXy, counter, inc


def get_parameters_2(random_bool: bool, list_of_params: list, list_of_params_svm: list, number_of_tests: int,
                     file: _io.TextIOWrapper):
    """
    Function returning two dictionaries with the chosen hyperparameters for NN
    and for the SVM. It also prints to the file the values
    Inputs:
        random - True if random search hyperparameter values wanted
                   False to use best hyperparameters found so far
        list_of_params [number_of_tests x 6] - list of lists containing NN
            hyperparameters
        list_of_params_svm [number_of_tests x 4] - list of lists containing SVM
            hyperparameters
        number_of_tests - number of sets of randomly chosen parameters
            corresponding to the number of times the scrips will be run
        file - file to which hyperparameters are written
    Outputs:
        dict_nn - dictionary with hyperparameters for NN
        dict_svm - dictionary with hyperparameters for SVM
    """
    dict_nn = {}
    dict_svm = {}
    if random_bool is False:
        dict_nn['no_hidden'] = 180
        dict_nn['no_layers'] = 2
        dict_nn['activation_fct'] = 'relu'
        dict_nn['regularizer'] = 0.01
        dict_nn['learning_rate'] = 0.01
        dict_nn['number_of_epochs'] = 130

        dict_svm['kernel'] = 'linear'
        dict_svm['C'] = 1
        dict_svm['gamma'] = 1
        dict_svm['decision_function'] = 'ovo'  # as in report
    else:
        dict_nn['no_hidden'] = list_of_params[number_of_tests][0]
        dict_nn['no_layers'] = list_of_params[number_of_tests][1]
        dict_nn['activation_fct'] = list_of_params[number_of_tests][2]
        dict_nn['regularizer'] = list_of_params[number_of_tests][3]
        dict_nn['learning_rate'] = list_of_params[number_of_tests][4]
        dict_nn['number_of_epochs'] = list_of_params[number_of_tests][5]

        dict_svm['kernel'] = list_of_params_svm[number_of_tests][0]
        dict_svm['C'] = list_of_params_svm[number_of_tests][1]
        dict_svm['gamma'] = list_of_params_svm[number_of_tests][2]
        dict_svm['decision_function'] = list_of_params_svm[number_of_tests][3]

    dict_nn['loss_fct'] = 'categorical_crossentropy'  # used to be categorical crossentropy
    dict_nn['decay_set'] = 1e-2 / dict_nn['number_of_epochs']  # used to be 1e-2/number_of_epochs

    print(file=file)
    print("No of hidden nodes: " + str(dict_nn['no_hidden']), file=file)
    print(file=file)
    print("Value of regularizer term " + str(dict_nn['regularizer']), file=file)
    print(file=file)
    print("Activation function is " + dict_nn['activation_fct'], file=file)
    print(file=file)
    print("Learning rate is " + str(dict_nn['learning_rate']), file=file)
    print(file=file)
    print("Momentum not used", file=file)
    print(file=file)
    print("Number of epochs " + str(dict_nn['number_of_epochs']), file=file)
    print(file=file)
    print("Number of hidden layers" + str(dict_nn['no_layers']), file=file)
    print(file=file)
    print("Loss function used is " + dict_nn['loss_fct'], file=file)
    print(file=file)
    print("Decay is " + str(dict_nn['decay_set']), file=file)

    print("------------------------------", file=file)
    print("SVM C is " + str(dict_svm['C']), file=file)
    print(file=file)
    print("SVM kernel is " + dict_svm['kernel'], file=file)
    print(file=file)
    print("SVM gamma is " + str(dict_svm['gamma']), file=file)
    print(file=file)
    print("SVM decision function is " + dict_svm['decision_function'], file=file)

    return dict_nn, dict_svm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--provided_labels",
        type=int,
        default=0,
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
    parser.add_argument(
        "--database_name",
        type=str,
        default='final_bare',
    )
    args = parser.parse_args()
    # connect to database
    cnx = mysql.connector.connect(user='root', password=args.sql_password,
                                  host='127.0.0.1',
                                  database=args.database_name)

    # retrieve from the tfidf table the name of the words for which the scores were calculated
    # the names are stored in NameOfColumns
    length = 117

    list_of_params, list_of_params_svm = get_parameter_sets(0)

    tf.compat.v1.reset_default_graph()
    tf.random.set_seed(0)
    np.random.seed(0)
    random.seed(0)

    mycursor = cnx.cursor()

    y, X_normalised = labels_and_features(mycursor,
                                          length)

    classnames, indices = np.unique(y, return_inverse=True)

    counter = 0

    X_train_tot, X_test, y_train_tot, y_test = train_test_split(X_normalised,
                                                                indices,
                                                                test_size=0.2,
                                                                random_state=32)

    newdir = str(args.results_folder) + "/Results_only_classification_4"
    os.mkdir(newdir)
    f = open(newdir + "/results.txt", 'w')
    print(f'y_test_real is {y_test}', file=f)
    print(f'y_test_real count is {Counter(y_test)}', file=f)
    print(f'y_trainval_real count is {Counter(y_train_tot)}', file=f)
    accuracy_nn_test_list = []  # list of NN test accuracies
    accuracy_nn_val_list = []  # list of NN validation accuracies
    accuracy_svm_test_list = []  # list of SVM accuracy on test data
    accuracy_svm_val_list = []  # list of SVM accuracy values on validation data

    minmax_nn = {'max': 0, 'min': 1, 'maxv': 0, 'minv': 1}
    conf_matrix_nn = {}

    minmax_svm = {'max': 0, 'min': 1, 'maxv': 0, 'minv': 1}
    conf_matrix_svm = {}

    dict_nn, dict_svm = get_parameters_2(random_bool=False,
                                         list_of_params=list_of_params,
                                         list_of_params_svm=list_of_params_svm,
                                         number_of_tests=0,
                                         file=f)

    inc = 1  # increment for setting random seed value and for making sure the split is correct
    noclasses = len(classnames)

    while counter < 100:
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=8,
                                                    restore_best_weights=True)
        [dictXy, counter, inc] = train_val_split_stratify(counter, inc,
                                                          X_train_tot,
                                                          y_train_tot,
                                                          X_test,
                                                          y_test)

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

        # =============================================================================

        minmax_svm, conf_matrix_svm = train_svm(dict_svm, dictXy,
                                                accuracy_svm_test_list,
                                                accuracy_svm_val_list,
                                                minmax_svm,
                                                conf_matrix_svm,
                                                label_type='NN')

        # =============================================================================

        counter += 1

    # =============================================================================
    train_nb(X_train_tot, y_train_tot, X_test, y_test, f)
    print(".............................", file=f)
    # =============================================================================
    print_file_test('NN', 'real', f, minmax_nn, conf_matrix_nn,
                    accuracy_nn_test_list)

    print(".............................", file=f)

    print_file_val('NN', 'real', f, minmax_nn, conf_matrix_nn,
                   accuracy_nn_val_list)

    print(".............................", file=f)

    # =============================================================================

    print_file_test('SVM', 'real', f, minmax_svm, conf_matrix_svm,
                    accuracy_svm_test_list)

    print(".............................", file=f)

    print_file_val('SVM', 'real', f, minmax_svm, conf_matrix_svm,
                   accuracy_svm_val_list)

    f.close()

    # =============================================================================
    # plot accuracies for train and validation
    # =============================================================================
    plt.figure(1)  # added line compared to previous laptop
    # =============================================================================
    plt.plot(history.history['accuracy'])  # before it was accuracy/acc in between quotes
    plt.plot(history.history['val_accuracy'])  # before it was val_accuracy in between quotes)
    plt.title('Accuracy on train and validation data')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(newdir + "/NN_Accuracy_train_val.png", dpi=1200)
    # plt.show()

    # plot loss for train and validation
    # =============================================================================
    plt.figure(2)  # added line compared to previous laptop
    # =============================================================================
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss function value for train and validation data')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.savefig(newdir + "/NN_Loss_train_val.png", dpi=1200)
    # plt.show()

    # =============================================================================
    plt.figure(3)  # added line compared to previous laptop
    # =============================================================================

    plt.plot(accuracy_nn_val_list)
    plt.plot(accuracy_nn_test_list)
    plt.title('NN accuracy on validation and test data given real labels')
    plt.ylabel('Accuracy')
    plt.xlabel('Number of runs')
    plt.legend(['Validation', 'Test'], loc='upper left')
    plt.savefig(newdir + "/NN_Validation_vs_test.png", dpi=1200)
    # plt.show()

    plt.plot(accuracy_svm_val_list)
    plt.plot(accuracy_svm_test_list)
    plt.title('SVM accuracy on validation and test data')
    plt.ylabel('Accuracy')
    plt.xlabel('Number of runs')
    plt.legend(['Validation', 'Test'], loc='upper left')
    plt.savefig(newdir + "/SVM_Validation_vs_test.png", dpi=1200)
    # plt.show()


if __name__ == "__main__":
    main()
