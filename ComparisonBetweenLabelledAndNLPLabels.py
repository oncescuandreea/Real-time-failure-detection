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

#same file as the previous version but use already extracted info to get the vectors for clustering
#input info needed:
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
#after disregarding tf-idf scores less than 0.0019
#importing useful libraries

import mysql.connector #connect to database

import pandas as pd

from collections import Counter  #create dictionary for bow


import numpy as np #maths equations
import random
import matplotlib.pyplot as plt
import tensorflow as tf

from numpy import mean

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from tensorflow.keras import optimizers
from tensorflow.keras.models import model_from_json

from utils import TFIDFretrieval, train_val_split, categoric, modelf
from utils import get_data_from_different_labels_for_cluster_initialisation
from utils import kmeans_clustering, NLP_labels_analysis, scoreCNNs
from utils import get_parameter_sets, get_parameters, train_NN, train_SVM
from utils import name_to_id, labels_and_features, train_val_split_stratify
from utils import print_file_test, print_file_val, train_NB


#connect to database
cnx = mysql.connector.connect(user='root', password='sqlAmonouaparola213',
                          host='127.0.0.1',
                          database='final')


# retrieve from the tfidf table the name of the words for which the scores were calculated
# the names are stored in NameOfColumns
mycursor=cnx.cursor()


#retreive from the satme table the actual values corresponding to the words and for each 
# document create a dictionary with the word and the values. then create a dataframe to allow for visualisation of the tfidf vectors
[listTFIDF, indexname, lengt] = TFIDFretrieval(mycursor)
# final is the dataframe containing the names of the reports and the corresponding tfidf vectors. if add indexname to command below then index is named as the report pd.DataFrame(listTFIDF,indexname)
final = pd.DataFrame(listTFIDF)
name2id = name_to_id(mycursor)

num_clusters = 7
number_of_tests = 1

avg = 0

list_of_params, list_of_paramsSVM = get_parameter_sets(number_of_tests)

while number_of_tests >= 0:
    
    no_labeled_sets = number_of_tests
    tf.compat.v1.reset_default_graph()
    tf.random.set_random_seed(0)
    np.random.seed(0)
    random.seed(0)
    [test, _, _] =\
            get_data_from_different_labels_for_cluster_initialisation(no_labeled_sets,
                                                                      name2id,
                                                                      final,
                                                                      indexname,
                                                                      cnx)
    
    [clusters, numberOfLabelsProvided] = kmeans_clustering(num_clusters,
                                                           no_labeled_sets,
                                                           name2id,
                                                           final,
                                                           indexname,
                                                           cnx)
    
    #create dictionary of name of reports and cluster associated
    reportName2cluster={}
    
    counter = 0
    for el in indexname:
        reportName2cluster.update({el:clusters[counter]})
        counter += 1
                
    f, newdir = NLP_labels_analysis(num_clusters=num_clusters,
                                    lengt=lengt,
                                    clusters=clusters,
                                    numberOfLabelsProvided=numberOfLabelsProvided,
                                    test=test)

    mycursor = cnx.cursor()
    
    y, yNLP, X_normalised = labels_and_features(mycursor,
                                        name2id,
                                        reportName2cluster,
                                        lengt)


    classnames, indices = np.unique(y, return_inverse=True)
    classnamesNLP, indicesNLP = np.unique(yNLP, return_inverse=True)

    counter = 0
    maxG = 0
    minG = 1

    X_traintotNLP, X_testNLP, y_traintotNLP, y_testNLP = train_test_split(X_normalised,
                                                                          indicesNLP,
                                                                          test_size=0.2,
                                                                          random_state=32)
    
    X_traintot, X_test, y_traintot, y_test = train_test_split(X_normalised,
                                                              indices,
                                                              test_size=0.2,
                                                              random_state=32)

    
    accuracy_NN_test_list = [] #list of NN test accuracies
    accuracy_NN_val_list = [] #list of NN validation accuracies
    accuracy_NN_test_list_NLP = []
    accuracy_NN_val_list_NLP = [] 
    accuracy_SVM_test_list = [] #list of SVM accuracy on test data
    accuracy_SVM_val_list = [] #list of SVM accuracy values on validation data
    accuracy_SVM_test_list_NLP = []
    accuracy_SVM_val_list_NLP = []
    
    
    minmax_NN = {}
    conf_matrix_NN = {}
    minmax_NN['max'] = 0
    minmax_NN['min'] = 1
    minmax_NN['maxv'] = 0
    minmax_NN['minv'] = 1
    
    minmax_NN_NLP = {}
    conf_matrix_NN_NLP = {}
    minmax_NN_NLP['max'] = 0
    minmax_NN_NLP['min'] = 1
    minmax_NN_NLP['maxv'] = 0
    minmax_NN_NLP['minv'] = 1
    
    minmax_SVM = {}
    conf_matrix_SVM = {}
    minmax_SVM['max'] = 0
    minmax_SVM['min'] = 1
    minmax_SVM['maxv'] = 0
    minmax_SVM['minv'] = 1
    
    minmax_SVM_NLP = {}
    conf_matrix_SVM_NLP = {}
    minmax_SVM_NLP['max'] = 0
    minmax_SVM_NLP['min'] = 1
    minmax_SVM_NLP['maxv'] = 0
    minmax_SVM_NLP['minv'] = 1

    
    dictNN, dictSVM = get_parameters(random='False',
                                     list_of_params=list_of_params,
                                     list_of_paramsSVM=list_of_paramsSVM,
                                     number_of_tests=number_of_tests,
                                     file=f)
    
    inc = 1 #increment for setting random seed value and for making sure the split is correct
    noclasses = len(classnames)
    
    print(number_of_tests)
    
    while counter < 1:
        newfile = f"C:/Users/oncescu/coding/4yp/{counter}_{number_of_tests}.txt"
        fi = open(newfile, 'w')
        fi.close() 
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=8,
                                                    restore_best_weights=True)
        [dictXy, counter, inc] = train_val_split_stratify(counter, inc,
                                                          X_traintot,
                                                          y_traintot,
                                                          X_traintotNLP,
                                                          y_traintotNLP,
                                                          X_test, X_testNLP,
                                                          y_test, y_testNLP)


        sgd = optimizers.SGD(lr=dictNN['learning_rate'],
                             decay=dictNN['decay_set'],
                             nesterov=False) #used to be lr 0.01

        
        # minmax_NN, conf_matrix_NN, history = train_NN(noclasses, dictNN, sgd,
        #                                               dictXy,
        #                                               accuracy_NN_test_list,
        #                                               callback,
        #                                               accuracy_NN_val_list,
        #                                               minmax_NN,
        #                                               conf_matrix_NN,
        #                                               label_type='NN')
        # if minmax_NN == conf_matrix_NN == history:
        #     continue
        
        minmax_NN_NLP, conf_matrix_NN_NLP, history_NLP = train_NN(noclasses,
                                                                  dictNN, sgd,
                                                                  dictXy,
                                                                  accuracy_NN_test_list_NLP,
                                                                  callback,
                                                                  accuracy_NN_val_list_NLP,
                                                                  minmax_NN_NLP,
                                                                  conf_matrix_NN_NLP, 
                                                                  label_type='NLP')
        
    # =============================================================================
        
        # minmax_SVM, conf_matrix_SVM = train_SVM(noclasses, dictSVM, dictXy,
        #                                         accuracy_SVM_test_list,
        #                                         accuracy_SVM_val_list,
        #                                         minmax_SVM,
        #                                         conf_matrix_SVM,
        #                                         label_type='NN')

        minmax_SVM_NLP, conf_matrix_SVM_NLP = train_SVM(noclasses, dictSVM, dictXy,
                                                        accuracy_SVM_test_list_NLP,
                                                        accuracy_SVM_val_list_NLP,
                                                        minmax_SVM_NLP,
                                                        conf_matrix_SVM_NLP,
                                                        label_type='NLP')
       
    # =============================================================================

        counter += 1

    # =============================================================================
    train_NB(X_traintot, y_traintot, X_traintotNLP, y_traintotNLP,
             X_test, y_test, X_testNLP, y_testNLP, f)
    # =============================================================================

    
    # print_file_test('NN', 'real', f, minmax_NN, conf_matrix_NN,
    #                 accuracy_NN_test_list)
    print_file_test('NN', 'NLP', f, minmax_NN_NLP, conf_matrix_NN_NLP,
                    accuracy_NN_test_list_NLP)
    
    print(".............................", file=f)

    # print_file_val('NN', 'real', f, minmax_NN, conf_matrix_NN,
    #                 accuracy_NN_val_list)
    print_file_val('NN', 'NLP', f, minmax_NN_NLP, conf_matrix_NN_NLP,
                    accuracy_NN_val_list_NLP)

    print(".............................", file=f)

    # =============================================================================
    
    # print_file_test('SVM', 'real', f, minmax_SVM, conf_matrix_SVM,
    #                 accuracy_SVM_test_list)
    print_file_test('SVM', 'NLP', f, minmax_SVM_NLP, conf_matrix_SVM_NLP,
                    accuracy_SVM_test_list_NLP)
    
    print(".............................", file=f)

    # print_file_val('SVM', 'real', f, minmax_SVM, conf_matrix_SVM,
    #                 accuracy_SVM_val_list)
    print_file_val('SVM', 'NLP', f, minmax_SVM_NLP, conf_matrix_SVM_NLP,
                    accuracy_SVM_val_list_NLP)

    f.close()
    # =============================================================================

    #plot accuracies for train and validation
# =============================================================================
#     plt.figure(1) # added line compared to previous laptop
# =============================================================================
#     plt.plot(history.history['accuracy']) # before it was accuracy/acc in between quotes
#     plt.plot(history.history['val_accuracy']) # before it was val_accuracy in between quotes)
#     plt.title('Accuracy on train and validation data')
#     plt.ylabel('accuracy')
#     plt.xlabel('epoch')
#     plt.legend(['train', 'validation'], loc='upper left')
#     plt.savefig(newdir+"/NN_Accuracy_train_val.png", dpi=1200)
#     plt.show()

# =============================================================================
#     plt.figure(2) # added line compared to previous laptop
# =============================================================================
    plt.plot(history_NLP.history['accuracy']) # accuracy in between quotes
    plt.plot(history_NLP.history['val_accuracy'])# before it was val_accuracy in between quotes)
    plt.title('NLP Accuracy on train and validation data')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(newdir+"/NN_NLP_Accuracy_train_val.png", dpi=1200)
    plt.show()

    #plot loss for train and validation
# =============================================================================
#     plt.figure(3) # added line compared to previous laptop
# =============================================================================
#     plt.plot(history.history['loss'])
#     plt.plot(history.history['val_loss'])
#     plt.title('Loss function value for train and validation data')
#     plt.ylabel('loss')
#     plt.xlabel('epoch')
#     plt.legend(['train', 'validation'], loc='upper right')
#     plt.savefig(newdir+"/NN_Loss_train_val.png", dpi=1200)
#     plt.show()

# =============================================================================
#     plt.figure(4) # added line compared to previous laptop
# =============================================================================
    plt.plot(history_NLP.history['loss'])
    plt.plot(history_NLP.history['val_loss'])
    plt.title('NLPLoss function value for train and validation data')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.savefig(newdir+"/NN_NLP_Loss_train_val.png", dpi=1200)
    plt.show()

# =============================================================================
#     plt.figure(5)  # added line compared to previous laptop
# =============================================================================
    
    # plt.plot(accuracy_NN_val_list)
    # plt.plot(accuracy_NN_test_list)
    # plt.title('NN accuracy on validation and test data given real labels')
    # plt.ylabel('Accuracy')
    # plt.xlabel('Number of runs')
    # plt.legend(['validation', 'test'], loc='upper left')
    # plt.savefig(newdir+"/NN_Validation_vs_test.png", dpi=1200)
    # plt.show()
    
    # plt.plot(accuracy_SVM_val_list)
    # plt.plot(accuracy_SVM_test_list)
    # plt.title('SVM accuracy on validation and test data')
    # plt.ylabel('Accuracy')
    # plt.xlabel('Number of runs')
    # plt.legend(['validation', 'test'], loc='upper left')
    # plt.savefig(newdir+"/SVM_Validation_vs_test.png", dpi=1200)
    # plt.show()
    
# =============================================================================
# print(f"Average accuracy on validation is: {avg}")
# print(f"Directory is: {name}")
# =============================================================================
# =============================================================================
#     plt.figure(6)  # added line compared to previous laptop
# =============================================================================
    plt.plot(accuracy_SVM_val_list_NLP)
    plt.plot(accuracy_SVM_test_list_NLP)
    plt.title('NLP SVM accuracy on validation and test data')
    plt.ylabel('Accuracy')
    plt.xlabel('Number of runs')
    plt.legend(['validation', 'test'], loc='upper left')
    plt.savefig(newdir+"/SVM_NLP_Validation_vs_test.png", dpi=1200)
    plt.show()
    number_of_tests -= 1
