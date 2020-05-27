# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 21:02:19 2020

@author: Andreea
"""

import os
import numpy as np #maths equations
import random

from numpy import mean
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from tensorflow.python.keras import utils
#from tensorflow.python.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import regularizers
from sklearn import svm
from sklearn.metrics import confusion_matrix

from datetime import datetime
from collections import Counter


def TFIDFretrieval(mycursor):
    '''
    This function retrieves tfidf scores for all documents
    inputs: 
        sql database
    outputs: 
        listTFIDF - list of TFIDF scores dictionaries associated with each doc
        indexname - list of document names in order corresponding to listTFIDF
        lengt - length of indexname list; number of documents available
    '''
    # retrieve columns from dataframe, containing the key words corresponding
    # to tfidf scores
    sql="SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA='final' AND TABLE_NAME='tfidfpd'";
    mycursor.execute(sql)
    NameOfColumns=mycursor.fetchall()
    
    #retrieve tfidf table from sql
    sql="Select * from tfidfpd"
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
        tfidfdict = {} # dictionary of one report
        indexname.append(wordvec[0])
        for i in range(1,lengthOfwordvec):
            tfidfdict.update({NameOfColumns[i - 1][0][1:]:wordvec[i]})
        listTFIDF.append(tfidfdict)
    lengt=len(dictionaries) #number of documents
    return listTFIDF, indexname, lengt


def train_val_split (X_traintot, y_traintot, randomState):
    '''
    Function which returns a split of the data into validation and test
    Random state is given such that the split of the actual labels and the NLP
    predicted ones is always consistent
    Inputs:
        X_traintot - features extracted from all documents
        y_traintot - labels corresponding to each document, either actual or 
                    predicted
        randomState - to control the split and allow for reproductibility
    Outputs:
        X_train - features corresponding to the train documents
        X_val - features corresponding to the validation documents
        y_train - labels corresponding to the train documents
        y_val - labels corresponding to the validation documents
    '''
    #randomState=np.random.random()
    
    X_train, X_val, y_train, y_val = train_test_split(X_traintot, y_traintot, test_size=0.25, random_state=randomState)
    
    return [X_train, X_val, y_train, y_val]
    
def categoric(y_train, y_val, y_test):
    '''
    Function that transforms string labels into numbers for classification
    Inputs:
        y_train - labels corresponding to the train documents
        y_val - labels corresponding to the validation documents
        y_test - labels corresponding to the test documents
    Outputs:
        y_trainNN - numerical labels corresponding to the train documents
        y_valNN - numerical labels corresponding to the validation documents
        y_testNN - numerical labels corresponding to the test documents
    '''
    y_trainNN = utils.to_categorical(y_train)
    y_valNN = utils.to_categorical(y_val)
    y_testNN = utils.to_categorical(y_test)
    return y_trainNN, y_valNN, y_testNN
    
def scoreCNNs(X, y, model, noclasses):
    '''
    Function returning the accuracy score for the artificial neural networks
    Inputs:
        X - features
        y - numerical labels
        model - NN model
        noclasses - number of classes
    Outputs:
        scoreNN - accuracy score
        predictedCNN1 - label prediction
    '''
    p1=model.predict(X)
    predictedCNN1=[]
    for row in p1:
        pCNN1=[]
        for i in range(0,noclasses):
            pCNN1.append(0)
        pCNN1[np.argmax(row)] = 1
        predictedCNN1.append(pCNN1)
    predictedCNN1=np.asarray(predictedCNN1)
    predictedCNN1=predictedCNN1.astype(np.float32)

    scoreNN = accuracy_score(y, predictedCNN1, normalize=True) #calculate accuracy score on test data
    return scoreNN, predictedCNN1

def modelf(noclasses, no_hidden, regularizer, activation_fct, no_layers):
    '''
    Function creates a ANN model based on the variable given by the user
    Inputs:
        noclasses - number of classes corresponding to number of output nodes
        no_hidden - number of nodes in the hidden layers
        regularizer - value of the regularizer to be used
        activation_fct - activation function for the hidden layers
        no_layers - number of hidden layers
    Outputs:
        modelNN - Neural network model
    '''
    modelNN = Sequential()
    np.random.seed(0)
    modelNN.add(Dense(no_hidden, input_dim=73, activation=activation_fct, kernel_regularizer=regularizers.l2(regularizer)))
    if no_layers>1:
        no_layers -= 1
        for i in range(0, no_layers):
            modelNN.add(Dense(no_hidden, activation=activation_fct, kernel_regularizer=regularizers.l2(regularizer)))

    modelNN.add(Dense(noclasses, activation='softmax'))
    return modelNN


def get_data_from_different_labels_for_cluster_initialisation(no_labeled_sets,
                                                              corrdict,
                                                              final,
                                                              indexname,
                                                              cnx):
    '''
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
    '''
    #extract all data form table an. it contains the id of the file and the 
    # labels for each sensor
    mycursor = cnx.cursor()
    sqldataID = "select * from an"
    mycursor.execute(sqldataID)
    recorddataID = mycursor.fetchall()  # all labels and the ID for each doc
    label = None
    number_of_reports = len(recorddataID) - 1 #number of reports

    no_labels = 7 * no_labeled_sets # each type of document receives a  label
    labels_tot = [] # all labels associated with the randomly extracted documents
    no_labels_so_far = 0 # how many documents have been labeled
    index_labels = [] # the indexes of the randomly labelled documents, these 
    #indexes correspond to the order of the documents in the dataframe
    label_only_once = [] # a list containing only one of each lable
    
    while no_labels_so_far < no_labels:
        
        random_sample_report = random.randint(0, number_of_reports)
        ID = recorddataID[random_sample_report][0]  # get ID of each recorded failure
        # create a string corresponding to the failure by adding the failure 
        # type and working words together
        label = recorddataID[random_sample_report][3] + \
                recorddataID[random_sample_report][1] + \
                recorddataID[random_sample_report][4] + \
                recorddataID[random_sample_report][2]
        
        report_name = corrdict[ID] # get the corresponding report name from ID
        
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
            
    # initialise list containing the cluer center values
    cluster_center_average = []
    
    # for each type of failure/document find all corresponding indexes and
    # add up the corresponding tfidf vectors
    for label in label_only_once:
        location_of_labels = [i for i, labell in enumerate(labels_tot) if
                              labell == label]
        sum_tfidf_scores = 0
        for location in location_of_labels:
            sum_tfidf_scores=sum_tfidf_scores+np.asarray(final.loc[index_labels[location],:])
        cluster_center_average.append(sum_tfidf_scores/no_labeled_sets)
            
    return [index_labels, labels_tot, cluster_center_average]

def kmeans_clustering(num_clusters,
                      no_labeled_sets,
                      corrdict,
                      final,
                      indexname,
                      cnx,
):
    '''
    Function returning the predicted labeled for each indexed document
    Inputs:
        num_clusters - number of classes
        no_labeled_sets - how many sets of labels are manually provided
        corrdict [dictionary] - dictionary relating the report ID and the 
            report name
        final [117 x 268] - the dataframe containing the names of the reports
            and the corresponding tfidf vectors
        indexname - list relating the report index to the report name, the
            index being the order in which reports are accessed in sql
        cnx - to access the database
    Outputs:
        clusters [No_documents x 1] - column vector with predicted cluster for
            each document index
        numberOfLabelsProvided - text to be output giving the number of
            manually labeled sets
    '''
    if no_labeled_sets == 0:
        numberOfLabelsProvided = "Unsupervised learning"
        km = KMeans(n_clusters=num_clusters, max_iter=32)
    else:
        if no_labeled_sets == 1:
            numberOfLabelsProvided = "1 set of labels"
        else:
            numberOfLabelsProvided = str(no_labeled_sets) + " sets of labels"
        [_, _, arrayn] =\
        get_data_from_different_labels_for_cluster_initialisation(no_labeled_sets,
                                                                  corrdict,
                                                                  final,
                                                                  indexname,
                                                                  cnx)
        arrayn=np.asarray(arrayn)
        km = KMeans(n_clusters=num_clusters, init=arrayn, max_iter=32, n_init=1)

    km.fit(final)
    
    clusters = km.labels_.tolist()
    return [clusters, numberOfLabelsProvided]

def parametersNN():
    '''
    Function used to perform random search for the neural netowrk hyperparameters
    Outputs:
        randomly chosen hyperparameters
    '''
    no_hidden_val = range(1, 4)
    regularizer_val = np.linspace(0.001, 0.1, 1000)
    functions = ['tanh', 'relu', 'sigmoid', 'exponential']
    no_hidden_nodes = range(100,300)
    learning_rates = np.linspace(0.0001, 0.02, 1000)
    epochs = range(130, 220)
    
    no_hidden_layers = random.sample(no_hidden_val, 1)[0]
    no_hidden = random.sample(no_hidden_nodes, 1)[0]
    activation_fct = functions[random.randint(0, len(functions) - 1)]
    regularizer = random.sample(list(regularizer_val), 1)[0]
    learning_rate = random.sample(list(learning_rates), 1)[0]
    number_of_epochs = random.sample(epochs, 1)[0]
    return no_hidden, no_hidden_layers, activation_fct, regularizer, learning_rate, number_of_epochs

def parametersSVM():
    '''
    Function used to perform random search for the SVM hyperparameters
    Outputs:
        randomly chosen hyperparameters
    '''
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    kernel = kernels[random.randint(0, len(kernels) - 1)]
    
    range_powers_C = range(-5,5)
    power_C = random.sample(range_powers_C, 1)[0]
    C = 10 ** power_C
    
    decision_function_shapes = ['ovr', 'ovo']
    decision_function = decision_function_shapes[random.randint(0, len(decision_function_shapes) - 1)]
    
    values_gamma = np.linspace(0, 1, 50)
    gamma_float = random.sample(list(values_gamma), 1)[0]
    gammas = ['auto', 'scale', gamma_float]
    gamma = gammas[random.randint(0, len(gammas) - 1)]
    return kernel, C, gamma, decision_function

def get_parameter_sets(number_of_tests):
    '''
    Function used to retrieve {number_of_tests} lists of sets of randomly chosen
    parameters for NN and SVM before fixing the random seeds. Used to later 
    choose the best parameters
    Inputs:
        number_of_tests - number of sets of hyperparameters to be chosen before
            deciding the parameters
    Outputs:
        list_of_paramsNN [number_of_tests x 6] - list of randomly chosen 
            hyperparameters for the neural network
        list_of_paramsSVM [number_of_tests x 4] - list of randomly chosen 
            hyperparameters for the SVM
    '''
    # NN randomly chosen parameters
    list_of_paramsNN = []
    for i in range(0, number_of_tests + 1):
        list_of_paramsNN.append(parametersNN())
    
    # SVM randomly chosen parameters
    list_of_paramsSVM = []
    for i in range(0, number_of_tests + 1):
        list_of_paramsSVM.append(parametersSVM())
    
    return list_of_paramsNN, list_of_paramsSVM

def get_parameters(random, list_of_params, list_of_paramsSVM, number_of_tests, file):
    '''
    Function returning two dictionaries with the chosen hyperparameters for NN
    and for the SVM. It also prints to the file the values
    Inputs:
        random - True if random search hyperparameter values wanted
                   False to use best hyperparameters found so far
        list_of_params [number_of_tests x 6] - list of lists containing NN 
            hyperparameters
        list_of_paramsSVM [number_of_tests x 4] - list of lists containing SVM
            hyperparameters
        number_of_tests - number of sets of randomly chosen parameters 
            corresponding to the number of times the scrips will be run
        file - file to which hyperparameters are written
    Outputs:
        dictNN - dictionary with hyperparameters for NN
        dictSVM - dictionary with hyperparameters for SVM
    '''
    dictNN = {}
    dictSVM = {}
    if random == 'False':
        dictNN['no_hidden'] = 189 #used to be 140 best 180
        dictNN['no_layers'] = 2 #used to be 2
        dictNN['activation_fct'] = 'relu' #used to be relu
        dictNN['regularizer'] = 0.0019909909909909913 # used to be 0.01
        dictNN['learning_rate'] = 0.01703193193193193 #used to be 0.01
        dictNN['number_of_epochs'] = 159 #used to be 130 # 154
        
        dictSVM['kernel'] = 'sigmoid' #used to be linear
        dictSVM['C'] = 1000 # used to be 1
        dictSVM['gamma'] = 'auto' # used to be 1
        dictSVM['decision_function'] = 'ovo' #as in report
    else:
        dictNN['no_hidden'] = list_of_params[number_of_tests][0]
        dictNN['no_layers'] = list_of_params[number_of_tests][1]
        dictNN['activation_fct'] = list_of_params[number_of_tests][2]
        dictNN['regularizer'] = list_of_params[number_of_tests][3]
        dictNN['learning_rate'] = list_of_params[number_of_tests][4]
        dictNN['number_of_epochs'] = list_of_params[number_of_tests][5]

        dictSVM['kernel'] = list_of_paramsSVM[number_of_tests][0]
        dictSVM['C'] = list_of_paramsSVM[number_of_tests][1]
        dictSVM['gamma'] = list_of_paramsSVM[number_of_tests][2]
        dictSVM['decision_function'] = list_of_paramsSVM[number_of_tests][3]

    dictNN['loss_fct'] = 'categorical_crossentropy' #used to be categorical crossentropy
    dictNN['decay_set'] = 1e-2/dictNN['number_of_epochs'] # used to be 1e-2/number_of_epochs
    
    print(file=file)
    print("No of hidden nodes: " + str(dictNN['no_hidden']), file=file)
    print(file=file)
    print("Value of regularizer term " + str(dictNN['regularizer']), file=file)
    print(file=file)
    print("Activation function is " + dictNN['activation_fct'], file=file)
    print(file=file)
    print("Learning rate is " + str(dictNN['learning_rate']), file=file)
    print(file=file)
    print("Momentum not used", file=file)
    print(file=file)
    print("Number of epochs " + str(dictNN['number_of_epochs']), file=file)
    print(file=file)
    print("Number of hidden layers" + str(dictNN['no_layers']), file=file)
    print(file=file)
    print("Loss function used is " + dictNN['loss_fct'], file=file)
    print(file=file)
    print("Decay is " + str(dictNN['decay_set']), file=file)
    
    print("------------------------------", file=file)
    print("SVM C is " + str(dictSVM['C']), file=file)
    print(file=file)
    print("SVM kernel is "+ dictSVM['kernel'], file=file)
    print(file=file)
    print("SVM gamma is " + str(dictSVM['gamma']), file=file)
    print(file=file)
    print("SVM decision function is " + dictSVM['decision_function'], file=file)
    
    return dictNN, dictSVM

def NLP_labels_analysis(num_clusters,
                        lengt,
                        clusters,
                        numberOfLabelsProvided,
                        test,
):
    '''
    Function which returns the confusion matrix for the predicted labels using
    Kmeans
    Inputs:
        num_clusters - number of classes/clusters expected
        lengt - number of reports/documents available
        clusters [No_documents x 1] - column vector with predicted cluster for
            each document index
        numberOfLabelsProvided - text to be output giving the number of
            manually labeled sets
        test - index of documents to be used as train labels
    Outputs:
        file - .txt file containing the indexes of files used as given labels 
            and accuracy matrix
        newdir - name of directory containing the file
    '''
    # create confusion matrix for the labels
    
    countl={}
    countvec=[]
    for i in range(0,num_clusters):
        countl[i]=0
    for i in range(0,num_clusters):
        countvec.append(dict(countl))
    
    ok=0
    label1=[]
    
    tot1=0
    label2=[]
    
    tot2=0
    label3=[]
    
    tot3=0
    label4=[]
    
    tot4=0
    label5=[]
    
    tot5=0
    label6=[]
    
    tot6=0
    label7=[]
    
    tot7=0
    maxim=dict(countl) #used to detect the main diagonal by finding the maximum entry for each manually set label
    
    labels={}
    # create confusion matrices by finding coresponding clusters
    # at every run the computer changes the ids of the clusters and we want to find the
    # correlation between our counting sysetem
    for i in range(0, lengt):
        if (i>=0 and i<=11) or (41<=i and i<=46) or (59<=i and i<=64): #this is label 1
            countvec[0][clusters[i]]+=1 #unordered confusion matrix 
            if countvec[0][clusters[i]] >maxim[0]:
                maxim[0]=countvec[0][clusters[i]]
                l1=clusters[i]
            tot1+=1
            label1='gsr ground pin'
        else:
            if (i>=12 and i<=29) or (53<=i and i<=58): #this is label 2
                countvec[1][clusters[i]]+=1
                if countvec[1][clusters[i]] >maxim[1]:
                    maxim[1]=countvec[1][clusters[i]]
                    l2=clusters[i]
                tot2+=1
                label2='gsr analog pin'
            else:
                if (i>=30 and i<=40) or (47<=i and i<=52): #this is label 3
                    countvec[2][clusters[i]]+=1
                    if countvec[2][clusters[i]] >maxim[2]:
                        maxim[2]=countvec[2][clusters[i]]
                        l3=clusters[i]
                    tot3+=1
                    label3='gsr resistor burnt'
                else:
                    if i==65 or (i>=74 and i<=79) or (i>=100 and i<=105): #this is label 4
                        countvec[3][clusters[i]]+=1
                        if countvec[3][clusters[i]] >maxim[3]:
                            maxim[3]=countvec[3][clusters[i]]
                            l4=clusters[i]
                        tot4+=1
                        label4='temperature ground pin'
                    else:
                        if (i>=66 and i<=70) or (i>=106 and i<=111): #this is label 5
                            countvec[4][clusters[i]]+=1
                            if countvec[4][clusters[i]] >maxim[4]:
                                maxim[4]=countvec[4][clusters[i]]
                                l5=clusters[i]
                            tot5+=1
                            label5='acceleration power pin'
                        else:
                            if (i>=71 and i<=73) or (i>=80 and i<=81) or (i>=112 and i<=117): #this is label 6
                                countvec[5][clusters[i]]+=1
                                if countvec[5][clusters[i]] >maxim[5]:
                                   maxim[5]=countvec[5][clusters[i]]
                                   l6=clusters[i]
                                tot6+=1
                                label6='acceleration ground pin'
                            else:
                                countvec[6][clusters[i]]+=1 
                                if countvec[6][clusters[i]] >maxim[6]:
                                   maxim[6]=countvec[6][clusters[i]]
                                   l7=clusters[i]
                                tot7+=1
                                label7='humidity power pin'
    
    
    val=0.75 #accuracy value; used when looping through to find the number of times needed to run the code to get above 75% accuracy
    if l5==l6:        #sometimes accelerometer ground/power pin get clustered in the same cluster because of the reports similarity       
        labels[0] = l1
        labels[1] = l2
        labels[2] = l3
        labels[3] = l4
        labels[4] = l5
        labels[5] = 21 - l1 - l2 - l3 - l4 - l5 - l7
        labels[6] = l7
        ok=0
    else:
        labels[0] = l1
        labels[1] = l2
        labels[2] = l3
        labels[3] = l4
        labels[4] = l5
        labels[5] = l6
        labels[6] = l7
        if maxim[0]/tot1>val and maxim[1]/tot2>val and maxim[2]/tot3>val and maxim[3]/tot4>val and maxim[4]/tot5>val and maxim[5]/tot6>val and maxim[6]/tot7>val:
            ok = 1 #used to stop the loop when if conditions are met
    
    now = datetime.now()
    current_time = now.strftime("%H_%M_%S")
    current_date = now.date()
    newdir = "C:/Users/Andreea/Results/Results_"+str(current_date) + "_"+current_time
    os.mkdir(newdir)
    f = open(newdir+"/results.txt", 'w')
    for i in range(0,7):
        for j in range(0,7):
            print(countvec[i][labels[j]], end=" ", file=f)
        print(file=f)
    print(numberOfLabelsProvided, file=f)
    print(file=f)
    print("Indices of the files used:",file=f)
    print(test, file=f)
    return f, newdir

def features_list(mycursor, ID):
    '''
    Function created list of features containing all features from all sensors
    and the ID of that specific dataset.
    Inputs:
        mycursor - for accessing sql database
        ID - for accessing the corresponding features in the database
    Outputs:
        features - list of combined features from GSR, Acc, Hum, Temp sensors
    '''
    dict_sensors={}
    dict_sensors['GSR'] = 'gsr2'
    dict_sensors['Acc'] = 'FEAT2'
    dict_sensors['Hum'] = 'hum3'
    dict_sensors['Temp'] = 'tempd5'
    
    features = []
    features.append(ID)
    for sensor in ['GSR', 'Acc', 'Hum', 'Temp']:
        sql = f"select * from {dict_sensors[sensor]} where ID='" + ID + "'"
        mycursor.execute(sql)
        data = mycursor.fetchall()
        for el in data[0][1:]:
            features.append(el)
    return features

def normalise(Xa, lengt):
    '''
    Function returns a normalised version of Xa. Features are normalsied
    Inputs:
        Xa [lengt x No_features] - matrix of feature vectors except the ID
        lengt - number of documents
    Outputs:
        XSVM [No_documents x No_features] - normalised matrix of features
    '''
    Xa = np.asarray(Xa)
    Xa = Xa.astype(np.float64)
    Xnew = np.transpose(Xa)
    XSVM = np.empty([lengt, 73])
    for i in range(0, 73):
        mini = min(Xnew[i])
        maxi = max(Xnew[i])
        for j in range(0, lengt):
            XSVM[j][i] = (Xa[j][i] - mini) / (maxi - mini)
    return XSVM

def name_to_id(mycursor):
    '''
    Create dictionary relating the name and the id of the reports
    '''
    name2id = {}
    sql = "select * from corr"
    mycursor.execute(sql)
    correlationdata=mycursor.fetchall()
    
    for row in correlationdata:
        name2id.update({row[0]:row[1]})
    return name2id

def labels_and_features(mycursor, name2id, reportName2cluster, lengt):
    Xa=[]
    sqldataID="select * from an"
    mycursor.execute(sqldataID)
    recorddataID = mycursor.fetchall() #all features from all documents
    label = None
    labelNLP = None
    yNLP = [] # Kmeans labels
    y = [] #manual labels

    for results in recorddataID:
        ID = results[0] # get ID of each recorded failure
        
        labelNLP = reportName2cluster[name2id[ID]] # get cluster number from K means
        label=results[3]+results[1]+results[4]+results[2] # create a string corresponding to the failure by adding the failure type and working words together

        yNLP.append(labelNLP)
        y.append(label)

        features = features_list(mycursor, ID)
        Xa.append(features[1:])
    
    #add to features and labels the ones for the case when everything works
    # choose only 15 from the table
    sql_data_working = "select * from feat_working"
    mycursor.execute(sql_data_working)
    working_data = mycursor.fetchall()
    for i in range(0, 15):
        Xa.append(list(working_data[0][1:]))
        yNLP.append(7)
        y.append('workingworkingworkingworking')
    
    XSVM = normalise(Xa, lengt+15)
    return y, yNLP, XSVM

def train_val_split_stratify(counter, inc, X_traintot, y_traintot,
                             X_traintotNLP, y_traintotNLP,
                             X_test, X_testNLP, y_test, y_testNLP):
    '''
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
        
    '''
    # set random state to be the same for both predicted and actual labels
    randomState = counter+inc
    
    dictXy = {}

    # split remaining data in train and validation data for actual labels
    [X_train, X_val, y_train, y_val] = train_val_split (X_traintot,y_traintot,randomState)


    # split remaining data in train and validation data for predicted labels
    [X_trainNLP, X_valNLP, y_trainNLP, y_valNLP] = train_val_split (X_traintotNLP,y_traintotNLP,randomState)

    # in case the selected random state does not distribute data evenly (soome labels are not represented in the train/validation set, increase randomState until the split is adequate)
    while len(Counter(y_train)) != len(Counter(y_val)) or len(Counter(y_trainNLP)) != len(Counter(y_valNLP)):
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
    
    dictXy['X_train'] = X_train
    dictXy['X_val'] = X_val
    dictXy['X_test'] = X_test
    
    dictXy['X_train_NLP'] = X_trainNLP
    dictXy['X_val_NLP'] = X_valNLP
    dictXy['X_test_NLP'] = X_testNLP
    
    dictXy['y_train'] = y_train
    dictXy['y_val'] = y_val
    dictXy['y_test'] = y_test
    
    dictXy['y_train_cat'] = y_trainNN
    dictXy['y_val_cat'] = y_valNN
    dictXy['y_test_cat'] = y_testNN
    
    dictXy['y_train_NLP'] = y_trainNLP
    dictXy['y_val_NLP'] = y_valNLP
    dictXy['y_test_NLP'] = y_testNLP
    
    dictXy['y_train_NLP_cat'] = y_trainNNNLP
    dictXy['y_val_NLP_cat'] = y_valNNNLP
    dictXy['y_test_NLP_cat'] = y_testNNNLP
    
    return dictXy, counter, inc

def train_NN(noclasses, dictNN, sgd, dictXy, accuracy_NN_test_list, callback,
             accuracy_NN_val_list, minmax, conf_matrix, label_type):
    
    NN_model = modelf(noclasses=noclasses,
                           no_hidden=dictNN['no_hidden'],
                           regularizer=dictNN['regularizer'],
                           activation_fct=dictNN['activation_fct'],
                           no_layers=dictNN['no_layers'])
    NN_model.compile(loss=dictNN['loss_fct'],
                      optimizer=sgd,
                      metrics=['accuracy'])
    
    if label_type == 'NLP':
        label = '_NLP'
    else:
        label = ''

    #fit models to training data actual labels
    try:
        history = NN_model.fit(dictXy[f"X_train{label}"],
                               dictXy[f"y_train{label}_cat"],
                               validation_data=(dictXy[f"X_val{label}"],
                                                dictXy[f"y_val{label}_cat"]),
                               epochs=dictNN['number_of_epochs'],
                               callbacks=[callback],
                               batch_size=1)
    except TypeError:
        return 0, 0 ,0
    [scoreNNtest, predictedNNtest] = scoreCNNs(dictXy[f"X_test{label}"],
                                               dictXy[f"y_test{label}_cat"],
                                               NN_model, noclasses)
    accuracy_NN_test_list.append(scoreNNtest) 
    
    [scoreNNval, predictedNNval] = scoreCNNs (dictXy[f"X_val{label}"],
                                              dictXy[f"y_val{label}_cat"],
                                              NN_model, noclasses)
    accuracy_NN_val_list.append(scoreNNval)
    
    if scoreNNtest >= minmax['max']:
        minmax['max'] = scoreNNtest
        conf_matrix['conftestmax'] =\
            confusion_matrix(np.argmax(dictXy[f"y_test{label}_cat"], axis=-1),
                             np.argmax(predictedNNtest, axis=-1),
                             labels=list(range(0, noclasses)))

    if scoreNNtest <= minmax['min']:
        minmax['min'] = scoreNNtest
        conf_matrix['conftestmin'] =\
            confusion_matrix(np.argmax(dictXy[f"y_test{label}_cat"], axis=-1),
                             np.argmax(predictedNNtest, axis=-1),
                             labels=list(range(0, noclasses)))
            
    if scoreNNval >= minmax['maxv']:
        minmax['maxv'] = scoreNNval
        conf_matrix['confvalmax'] =\
            confusion_matrix(np.argmax(dictXy[f"y_val{label}_cat"], axis=-1),
                             np.argmax(predictedNNval, axis=-1),
                             labels=list(range(0, noclasses)))
            
    if scoreNNval <= minmax['minv']:
        minmax['minv'] = scoreNNval
        conf_matrix['confvalmin'] =\
            confusion_matrix(np.argmax(dictXy[f"y_val{label}_cat"], axis=-1),
                                       np.argmax(predictedNNval, axis=-1),
                                       labels=list(range(0, noclasses)))

    return minmax, conf_matrix, history

def train_SVM(noclasses, dictSVM, dictXy, accuracy_SVM_test_list,
              accuracy_SVM_val_list, minmax, conf_matrix, label_type):
    
    if label_type == 'NLP':
        label = '_NLP'
    else:
        label = ''
        
    modelSVM = svm.SVC(kernel=dictSVM['kernel'],
                           C=dictSVM['C'],
                           gamma=dictSVM['gamma'],
                           decision_function_shape=dictSVM['decision_function'],
                           class_weight='balanced')
    
    modelSVM.fit(dictXy[f"X_train{label}"], dictXy[f"y_train{label}"])
    
    validation = modelSVM.predict(dictXy[f"X_val{label}"])
    scoreSVMval = accuracy_score(dictXy[f"y_val{label}"], validation, normalize=True)
    accuracy_SVM_val_list.append(scoreSVMval) 
    
    predictedSVM = modelSVM.predict(dictXy[f"X_test{label}"])
    scoreSVM = accuracy_score(dictXy[f"y_test{label}"], predictedSVM, normalize=True)
    accuracy_SVM_test_list.append(scoreSVM)
    
    if scoreSVM >= minmax['max']:
        minmax['max'] = scoreSVM
        conf_matrix['conftestmax'] = confusion_matrix(dictXy[f"y_test{label}"],
                                                         predictedSVM)
    if scoreSVM <= minmax['min']:
        minmax['min'] = scoreSVM
        conf_matrix['conftestmin'] = confusion_matrix(dictXy[f"y_test{label}"],
                                                         predictedSVM)
    if scoreSVMval >= minmax['maxv']:
        minmax['maxv'] = scoreSVMval
        conf_matrix['confvalmax'] = confusion_matrix(dictXy[f"y_val{label}"],
                                                        validation)
    if scoreSVMval <= minmax['minv']:
        minmax['minv'] = scoreSVMval
        conf_matrix['confvalmin'] = confusion_matrix(dictXy[f"y_val{label}"],
                                                        validation)
    return minmax, conf_matrix

def train_NB(X_traintot, y_traintot, X_traintotNLP, y_traintotNLP,
             X_test, y_test, X_testNLP, y_testNLP, f):
    
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
    
def print_file_test(type_NN_SVM, type_true_NLP, f,
               minmax, conf_matrix, accuracy_list):
    type_dict = {}
    type_dict ['NN'] = 'Neural Networks'
    type_dict ['SVM'] = 'SVM'
    
    print(f"Test accuracy for {type_dict[type_NN_SVM]} with {type_true_NLP} labels is max:", file=f)
    print(minmax['max'], file=f)
    print(f"Test confusion matrix for {type_dict[type_NN_SVM]} with {type_true_NLP} labels is max:", file=f)
    print(conf_matrix['conftestmax'], file=f)
    print(f"Test accuracy for {type_dict[type_NN_SVM]} with {type_true_NLP} labels is min:", file=f)
    print(minmax['min'], file=f)
    print(f"Test confusion matrix for {type_dict[type_NN_SVM]} with {type_true_NLP} labels is min:", file=f)
    print(conf_matrix['conftestmin'], file=f)
    print(f"Mean test accuracy for {type_dict[type_NN_SVM]} with {type_true_NLP} for 100 runs", file=f)
    print(mean(accuracy_list), file=f)
    
def print_file_val(type_NN_SVM, type_true_NLP, f,
                   minmax, conf_matrix, accuracy_list):
    type_dict = {}
    type_dict ['NN'] = 'Neural Networks'
    type_dict ['SVM'] = 'SVM'
    
    print(f"Validation accuracy for {type_dict[type_NN_SVM]} with {type_true_NLP} labels is max:",
          file=f)
    print(minmax['maxv'], file=f)
    print(f"Validation confusion matrix for {type_dict[type_NN_SVM]} with {type_true_NLP} labels is max:",
          file=f)
    print(conf_matrix['confvalmax'], file=f)
    print(f"Validation accuracy for {type_dict[type_NN_SVM]} with {type_true_NLP} labels is min:",
          file=f)
    print(minmax['minv'], file=f)
    print(f"Test confusion matrix for {type_dict[type_NN_SVM]} with {type_true_NLP} labels is min:",
          file=f)
    print(conf_matrix['confvalmin'], file=f)
    print(f"Mean validation accuracy for {type_dict[type_NN_SVM]} with {type_true_NLP} for 100 runs", file=f)
    print(mean(accuracy_list),file=f)
    
    