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

from utils import TFIDFretrieval, train_val_split, categoric, modelf
from utils import get_data_from_different_labels_for_cluster_initialisation
from utils import kmeans_clustering, NLP_labels_analysis, scoreCNNs
from utils import get_parameter_sets, get_parameters
from utils import name_to_id, labels_and_features


#connect to database
cnx = mysql.connector.connect(user='root', password='Amonouaparola213',
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
number_of_tests = 2

avg = 0

list_of_params, list_of_paramsSVM = get_parameter_sets(number_of_tests)

while number_of_tests >= 1:
    
    no_labeled_sets = number_of_tests
    tf.compat.v1.reset_default_graph()
    tf.random.set_seed(0)
    np.random.seed(0)
    random.seed(0)
    [test, testlabels, _] =\
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
    
    y, yNLP, XSVM = labels_and_features(mycursor,
                                        name2id,
                                        reportName2cluster,
                                        lengt)


    classnames, indices = np.unique(y, return_inverse=True)
    classnamesNLP, indicesNLP = np.unique(yNLP, return_inverse=True)

    counter = 0
    miniTNNs = []
    minivTNNs = [] #list of validation accuracies
    miniTNNsNLP = []
    minivTNNsNLP = [] #list of validation accuracies
    maxG = 0
    minG = 1

    X_traintotNLP, X_testNLP, y_traintotNLP, y_testNLP = train_test_split(XSVM, indicesNLP, test_size=0.2, random_state=32)
    X_traintot, X_test, y_traintot, y_test = train_test_split(XSVM, indices, test_size=0.2, random_state=32)


    minSVMs = [] #list of SVM accuracy on test data
    svmval = [] #list of SVM accuracy values on validation data
    minSVMsNLP = [] #list of SVM accuracy on test data
    svmvalNLP =[]
    minSVM = 1
    maxSVM = 0
    minvSVM = 1
    maxvSVM = 0
    maxNN = 0
    minNN = 1
    maxvNN = 0
    minvNN = 1



    minSVMNLP = 1
    maxSVMNLP = 0
    minvSVMNLP = 1
    maxvSVMNLP = 0
    maxNNNLP = 0
    minNNNLP = 1
    maxvNNNLP = 0
    minvNNNLP = 1
    
    dictNN, dictSVM = get_parameters(random='False',
                                     list_of_params=list_of_params,
                                     list_of_paramsSVM=list_of_paramsSVM,
                                     number_of_tests=number_of_tests,
                                     file=f)
    
    inc=1 #increment for setting random seed value and for making sure the split is correct
    noclasses = len(classnames)
    
    print(number_of_tests)
    
    while counter < 3:
        sgd = optimizers.SGD(lr=dictNN['learning_rate'],
                             decay=dictNN['decay_set'],
                             nesterov=False) #used to be lr 0.01

        # create NN model for actual labels
        modelCNN1 = modelf(noclasses=noclasses,
                           no_hidden=dictNN['no_hidden'],
                           regularizer=dictNN['regularizer'],
                           activation_fct=dictNN['activation_fct'],
                           no_layers=dictNN['no_layers'])
        modelCNN1.compile(loss=dictNN['loss_fct'],
                          optimizer=sgd,
                          metrics=['accuracy'])


        # create NN model for clustered labels
        modelCNN1NLP = modelf(noclasses=noclasses,
                              no_hidden=dictNN['no_hidden'],
                              regularizer=dictNN['regularizer'],
                              activation_fct=dictNN['activation_fct'],
                              no_layers=dictNN['no_layers'])
        modelCNN1NLP.compile(loss=dictNN['loss_fct'],
                             optimizer=sgd,
                             metrics=['accuracy'])

        # set random state to be the same for both predicted and actual labels
        randomState = counter+inc

        # split remaining data in train and validation data for actual labels
        [X_train, X_val, y_train, y_val] = train_val_split (X_traintot,y_traintot,randomState)
# =============================================================================
#         print("------------ X_train -------------", file=f)
#         print(X_train, file=f)
#         print("------------ y_val ---------------", file=f)
#         print(y_val, file=f)
#         print("------------ initial weights --------", file=f)
#         print(modelCNN1.get_weights(), file=f)
# =============================================================================

        # split remaining data in train and validation data for predicted labels
        [X_trainNLP, X_valNLP, y_trainNLP, y_valNLP] = train_val_split (X_traintotNLP,y_traintotNLP,randomState)

        # in case the selected random state does not distribute data evenly (soome labels are not represented in the train/validation set, increase randomState until the split is adequate)
        while len(Counter(y_train))!=len(Counter(y_val)) or len(Counter(y_trainNLP))!=len(Counter(y_valNLP)):
            inc += 1
            randomState = counter + inc

            # split remaining data in train and validation data for actual labels
            [X_train, X_val, y_train, y_val] = train_val_split (X_traintot,y_traintot,randomState)

            # split remaining data in train and validation data for predicted labels
            [X_trainNLP, X_valNLP, y_trainNLP, y_valNLP] = train_val_split (X_traintotNLP,y_traintotNLP,randomState)

        # transform the actual labels into numbers
        [y_trainNN, y_valNN, y_testNN]=categoric(y_train, y_val, y_test)

        # do the same for NLP although not neccesary in this case
        [y_trainNNNLP, y_valNNNLP, y_testNNNLP]=categoric(y_trainNLP, y_valNLP, y_testNLP)


        #fit models to training data actual labels
        history = modelCNN1.fit(X_train, y_trainNN, validation_data=(X_val,y_valNN),
                              epochs=dictNN['number_of_epochs'], batch_size=1)
        #used to save accuracy and loss values for train and validation sets

        #fit models to training data clustered labels
        historyNLP = modelCNN1NLP.fit(X_trainNLP, y_trainNNNLP,
                                    validation_data=(X_valNLP,y_valNNNLP),
                                    epochs=dictNN['number_of_epochs'],
                                    batch_size=1) 
        #used to save accuracy and loss values for train and validation sets

        #test set actual labels
        [scoreNNtest,predictedNNtest] = scoreCNNs(X_test, y_testNN, modelCNN1, noclasses)
        miniTNNs.append(scoreNNtest) #save accuracy score of Neural Networks on test data

        #test set clustered labels
        [scoreNNtestNLP, predictedNNtestNLP] = scoreCNNs (X_testNLP, y_testNNNLP, modelCNN1NLP, noclasses)
        miniTNNsNLP.append(scoreNNtestNLP) #save accuracy score of Neural Networks on test data clustered labels

        #validation set actual labels
        [scoreNNval, predictedNNval] = scoreCNNs (X_val, y_valNN, modelCNN1, noclasses)
        minivTNNs.append(scoreNNval) #save accuracy score of Neural Networks on validation data

        #validation set clustered labels
        [scoreNNvalNLP, predictedNNvalNLP] = scoreCNNs (X_valNLP, y_valNNNLP, modelCNN1NLP, noclasses)
        minivTNNsNLP.append(scoreNNvalNLP) #save accuracy score of Neural Networks on validation data clustered labels

    # =============================================================================
        modelSVM = svm.SVC(kernel=dictSVM['kernel'],
                           C=dictSVM['C'],
                           gamma=dictSVM['gamma'],
                           decision_function_shape=dictSVM['decision_function'],
                           class_weight='balanced')
        modelSVM.fit(X_train, y_train)


        modelSVMNLP = svm.SVC(kernel=dictSVM['kernel'],
                              C=dictSVM['C'],
                              gamma=dictSVM['gamma'],
                              decision_function_shape=dictSVM['decision_function'],
                              class_weight='balanced')
        modelSVMNLP.fit(X_trainNLP, y_trainNLP)

        #find accuracy on validation set and append it to list
        validation = modelSVM.predict(X_val)
        scoreSVMval = accuracy_score(y_val, validation, normalize=True) #calculate accuracy on validation set
        svmval.append(scoreSVMval) #apped accuracy on validation set to svmval

        #find accuracy on test set and append it to list
        predictedSVM = modelSVM.predict(X_test)
        scoreSVM = accuracy_score(y_test, predictedSVM, normalize=True)
        minSVMs.append(scoreSVM) #append SVM accuracy score to minSVMs



        #find accuracy on validation set and append it to list predicted labels
        validationNLP = modelSVMNLP.predict(X_valNLP)
        scoreSVMvalNLP = accuracy_score(y_valNLP, validationNLP, normalize=True) #calculate accuracy on validation set
        svmvalNLP.append(scoreSVMvalNLP) #apped accuracy on validation set to svmval

        #find accuracy on test set and append it to list predicted labels
        predictedSVMNLP = modelSVMNLP.predict(X_testNLP)
        scoreSVMNLP = accuracy_score(y_testNLP, predictedSVMNLP, normalize=True)
        minSVMsNLP.append(scoreSVMNLP) #append SVM accuracy score to minSVMs

        #save for maximum accuracy of SVM on test data the confusion matrices as well
        if scoreSVM > maxSVM:
            maxSVM = scoreSVM
            confmax = confusion_matrix(y_test, predictedSVM)
        if scoreSVM < minSVM:
            minSVM = scoreSVM
            confmin = confusion_matrix(y_test, predictedSVM)
        if scoreSVMval > maxvSVM:
            maxvSVM = scoreSVMval
            confvmax = confusion_matrix(y_val, validation)
        if scoreSVMval < minvSVM:
            minvSVM = scoreSVMval
            confvmin = confusion_matrix(y_val, validation)


        #save for maximum accuracy of SVM on test/val data the confusion matrices predicted labels
        if scoreSVMNLP > maxSVMNLP:
            maxSVMNLP = scoreSVMNLP
            confmaxNLP = confusion_matrix(y_testNLP, predictedSVMNLP)
        if scoreSVMNLP < minSVMNLP:
            minSVMNLP = scoreSVMNLP
            confminNLP = confusion_matrix(y_testNLP, predictedSVMNLP)
        if scoreSVMvalNLP > maxvSVMNLP:
            maxvSVMNLP = scoreSVMvalNLP
            confvmaxNLP = confusion_matrix(y_valNLP, validationNLP)
        if scoreSVMvalNLP < minvSVMNLP:
            minvSVMNLP = scoreSVMvalNLP
            confvminNLP = confusion_matrix(y_valNLP, validationNLP)
    # =============================================================================

        if scoreNNtest >= maxNN:
            maxNN = scoreNNtest
            confNNtestmax = confusion_matrix(np.argmax(y_testNN, axis=-1),
                                             np.argmax(predictedNNtest,
                                                       axis=-1),
                                             labels=list(range(0, noclasses)))
        if scoreNNtest <= minNN:
            minNN = scoreNNtest
            confNNtestmin = confusion_matrix(np.argmax(y_testNN, axis=-1),
                                             np.argmax(predictedNNtest,
                                                       axis=-1),
                                             labels=list(range(0, noclasses)))
        if scoreNNval >= maxvNN:
            maxvNN = scoreNNval
            confNNvalmax = confusion_matrix(np.argmax(y_valNN, axis=-1),
                                            np.argmax(predictedNNval, axis=-1),
                                            labels=list(range(0, noclasses)))
        if scoreNNval <= minvNN:
            minvNN = scoreNNval
            confNNvalmin = confusion_matrix(np.argmax(y_valNN, axis=-1),
                                            np.argmax(predictedNNval, axis=-1),
                                            labels=list(range(0, noclasses)))


        if scoreNNtestNLP >= maxNNNLP:
            maxNNNLP = scoreNNtestNLP
            confNNtestmaxNLP = confusion_matrix(np.argmax(y_testNNNLP, axis=-1),
                                                np.argmax(predictedNNtestNLP,
                                                          axis=-1),
                                                labels=list(range(0,
                                                                  noclasses)))
        if scoreNNtestNLP <= minNNNLP:
            minNNNLP = scoreNNtestNLP
            confNNtestminNLP = confusion_matrix(np.argmax(y_testNNNLP, axis=-1),
                                                np.argmax(predictedNNtestNLP,
                                                          axis=-1),
                                                labels=list(range(0, noclasses)))
        if scoreNNvalNLP >= maxvNNNLP:
            maxvNNNLP = scoreNNvalNLP
            confNNvalmaxNLP = confusion_matrix(np.argmax(y_valNNNLP, axis=-1),
                                               np.argmax(predictedNNvalNLP,
                                                         axis=-1),
                                               labels=list(range(0, noclasses)))
        if scoreNNvalNLP <= minvNNNLP:
            minvNNNLP = scoreNNvalNLP
            confNNvalminNLP = confusion_matrix(np.argmax(y_valNNNLP, axis=-1),
                                               np.argmax(predictedNNvalNLP,
                                                         axis=-1),
                                               labels=list(range(0, noclasses)))


        counter += 1

    # =============================================================================
    modelG = GaussianNB()
    modelG.fit(X_traintot, y_traintot)
    predictedG = modelG.predict(X_test)
    scoreG = accuracy_score(y_test, predictedG, normalize=True)

    modelGNLP = GaussianNB()
    modelGNLP.fit(X_traintotNLP, y_traintotNLP)
    predictedGNLP = modelGNLP.predict(X_testNLP)
    scoreGNLP = accuracy_score(y_testNLP, predictedGNLP, normalize=True)

    print("Accuracy NB is:",file=f)
    print(scoreG,file=f)
    print("Confusion matrix for Naive Bayes:",file=f)
    print(confusion_matrix(y_test, predictedG),file=f)

    print("Accuracy NB is:",file=f)
    print(scoreGNLP,file=f)
    print("Confusion matrix for Naive Bayes:",file=f)
    print(confusion_matrix(y_testNLP, predictedGNLP),file=f)

    print(".............................",file=f)
    # =============================================================================

    print("Test accuracy NN max is:",file=f)
    print(maxNN,file=f)
    print("Test Confusion matrix for Neural Networks max:",file=f)
    print(confNNtestmax,file=f)
    print("Test Accuracy NN min is:",file=f)
    print(minNN,file=f)
    print("Test Confusion matrix for Neural Networks min:",file=f)
    print(confNNtestmin,file=f)
    print("Mean test accuracy for 100 runs",file=f)
    print(mean(miniTNNs),file=f)


    print("NLP Test accuracy NN max is:",file=f)
    print(maxNNNLP,file=f)
    print("NLP Test Confusion matrix for Neural Networks max:",file=f)
    print(confNNtestmaxNLP,file=f)
    print("NLP Test Accuracy NN min is:",file=f)
    print(minNNNLP,file=f)
    print("NLP Test Confusion matrix for Neural Networks min:",file=f)
    print(confNNtestminNLP,file=f)
    print("NLP Mean test accuracy for 100 runs",file=f)
    print(mean(miniTNNsNLP),file=f)

    print(".............................",file=f)

    print("Validation accuracy NN max is:",file=f)
    print(maxvNN,file=f)
    print("Validation Confusion matrix for Neural Networks max:",file=f)
    print(confNNvalmax,file=f)
    print("Validation Accuracy NN min is:",file=f)
    print(minvNN,file=f)
    print("Validation Confusion matrix for Neural Networks min:",file=f)
    print(confNNvalmin,file=f)
    print("Mean Validation accuracy for 100 runs",file=f)
    print(mean(minivTNNs),file=f)
# =============================================================================
#     if mean(minivTNNs)>avg:
#         avg = mean(minivTNNs)
#         name = newdir
# =============================================================================


    print("NLP Validation accuracy NN max is:",file=f)
    print(maxvNNNLP,file=f)
    print("NLP Validation Confusion matrix for Neural Networks max:",file=f)
    print(confNNvalmaxNLP,file=f)
    print("NLP Validation Accuracy NN min is:",file=f)
    print(minvNNNLP,file=f)
    print("NLP Validation Confusion matrix for Neural Networks min:",file=f)
    print(confNNvalminNLP,file=f)
    print("NLP Mean Validation accuracy for 100 runs",file=f)
    print(mean(minivTNNsNLP),file=f)

    print(".............................",file=f)

    # =============================================================================
    print("Max test accuracy SVM value:",file=f)
    print(maxSVM,file=f)
    print("test confusion matrix for maximum SVM accuracy:",file=f)
    print(confmax,file=f)
    print("Min test accuracy SVM value:",file=f)
    print(minSVM,file=f)
    print("test confusion matrix for minimum SVM accuracy:",file=f)
    print(confmin,file=f)
    print("Mean test accuracy value for 100 runs:",file=f)
    print(mean(minSVMs),file=f)

    print("NLP Max test accuracy SVM value:",file=f)
    print(maxSVMNLP,file=f)
    print("NLP test confusion matrix for maximum SVM accuracy:",file=f)
    print(confmaxNLP,file=f)
    print("NLP Min test accuracy SVM value:",file=f)
    print(minSVMNLP,file=f)
    print("NLP test confusion matrix for minimum SVM accuracy:",file=f)
    print(confminNLP,file=f)
    print("NLP Mean test accuracy value for 100 runs:",file=f)
    print(mean(minSVMsNLP),file=f)

    print(".............................",file=f)

    print("val Max accuracy SVM value:",file=f)
    print(maxvSVM,file=f)
    print("val confusion matrix for maximum SVM accuracy:",file=f)
    print(confvmax,file=f)
    print("val Min accuracy SVM value:",file=f)
    print(minvSVM,file=f)
    print("val confusion matrix for minimum SVM accuracy:",file=f)
    print(confvmin,file=f)
    print("val Mean accuracy value for 100 runs:",file=f)
    print(mean(svmval),file=f)
# =============================================================================
#     if mean(svmval)>avg:
#         avg = mean(svmval)
#         name = newdir
# =============================================================================

    print("NLP val Max accuracy SVM value:", file=f)
    print(maxvSVMNLP, file=f)
    print("NLP val confusion matrix for maximum SVM accuracy:", file=f)
    print(confvmaxNLP, file=f)
    print("NLP val Min accuracy SVM value:", file=f)
    print(minvSVMNLP, file=f)
    print("NLP val confusion matrix for minimum SVM accuracy:", file=f)
    print(confvminNLP, file=f)
    print("NLP val Mean accuracy value for 100 runs:", file=f)
    print(mean(svmvalNLP), file=f)

    f.close()
    # =============================================================================

    #plot accuracies for train and validation
# =============================================================================
#     plt.figure(1) # added line compared to previous laptop
# =============================================================================
    plt.plot(history.history['accuracy']) # before it was accuracy/acc in between quotes
    plt.plot(history.history['val_accuracy']) # before it was val_accuracy in between quotes)
    plt.title('Accuracy on train and validation data')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(newdir+"/AccuracytrainvalACC.png",dpi=1200)
    plt.show()

# =============================================================================
#     plt.figure(2) # added line compared to previous laptop
# =============================================================================
    plt.plot(historyNLP.history['accuracy']) # accuracy in between quotes
    plt.plot(historyNLP.history['val_accuracy'])# before it was val_accuracy in between quotes)
    plt.title('NLP Accuracy on train and validation data')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(newdir+"/NLPAccuracytrainvalACC.png",dpi=1200)
    plt.show()

    #plot loss for train and validation
# =============================================================================
#     plt.figure(3) # added line compared to previous laptop
# =============================================================================
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss function value for train and validation data')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.savefig(newdir+"/LosstrainvalALL.png",dpi=1200)
    plt.show()

# =============================================================================
#     plt.figure(4) # added line compared to previous laptop
# =============================================================================
    plt.plot(historyNLP.history['loss'])
    plt.plot(historyNLP.history['val_loss'])
    plt.title('NLPLoss function value for train and validation data')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.savefig(newdir+"/NLPLosstrainvalALL.png",dpi=1200)
    plt.show()

# =============================================================================
#     plt.figure(5)  # added line compared to previous laptop
# =============================================================================
    
    plt.plot(minivTNNs)
    plt.plot(miniTNNs)
    plt.title('NN accuracy on validation and test data given real labels')
    plt.ylabel('Accuracy')
    plt.xlabel('Number of runs')
    plt.legend(['validation', 'test'], loc='upper left')
    plt.savefig(newdir+"/validationvstestNNALL.png",dpi=1200)
    plt.show()
    
    plt.plot(svmval)
    plt.plot(minSVMs)
    plt.title('SVM accuracy on validation and test data')
    plt.ylabel('Accuracy')
    plt.xlabel('Number of runs')
    plt.legend(['validation', 'test'], loc='upper left')
    plt.savefig(newdir+"/validationvstestSVMALL.png",dpi=1200)
    plt.show()
    
# =============================================================================
# print(f"Average accuracy on validation is: {avg}")
# print(f"Directory is: {name}")
# =============================================================================
# =============================================================================
#     plt.figure(6)  # added line compared to previous laptop
# =============================================================================
    plt.plot(svmvalNLP)
    plt.plot(minSVMsNLP)
    plt.title('NLP SVM accuracy on validation and test data')
    plt.ylabel('Accuracy')
    plt.xlabel('Number of runs')
    plt.legend(['validation', 'test'], loc='upper left')
    plt.savefig(newdir+"/NLPvalidationvstestSVMALL.png", dpi=1200)
    plt.show()
    number_of_tests -= 1
