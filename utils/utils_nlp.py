import os
import numpy as np
from sklearn.cluster import KMeans
from itertools import permutations
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import mysql.connector
import pandas


def kmeans_clustering(num_clusters: int,
                      no_labeled_sets: int,
                      arrayn: list,
                      final: pandas.DataFrame,
                      ):
    '''
    Function returning the predicted label for each indexed document
    Inputs:
        num_clusters - number of classes
        no_labeled_sets - how many sets of labels are manually provided
        arrayn[noclasses x no_features] - TFIDF cluster vector
            with values given by averaging each vector location from each
            document from the corresponding user given labels. Used to
            initialise the Kmeans clustering
        final [117 x 268] - the dataframe containing the names of the reports
            and the corresponding tfidf vectors
    Outputs:
        clusters [No_documents x 1] - column vector with predicted cluster for
            each document index
        number_of_labels_provided - text to be output giving the number of
            manually labeled sets
    '''
    if no_labeled_sets == 0:
        number_of_labels_provided = "Unsupervised learning"
        km = KMeans(n_clusters=num_clusters, max_iter=32)
    else:
        if no_labeled_sets == 1:
            number_of_labels_provided = "1 set of labels"
        else:
            number_of_labels_provided = str(no_labeled_sets) + " sets of labels"
        arrayn = np.asarray(arrayn)
        km = KMeans(n_clusters=num_clusters, init=arrayn, max_iter=32, n_init=1)

    km.fit(final)

    clusters = km.labels_.tolist()
    return [clusters, number_of_labels_provided]


def nlp_labels_analysis(mycursor: mysql.connector.cursor,
                        num_clusters: int,
                        length: int,
                        clusters: list,
                        number_of_labels_provided: int,
                        test: list,
                        id2name: dict,
                        results_folder: Path,
                        ):
    """
    Function which returns the confusion matrix for the predicted labels using
    Kmeans. This is done by attempting to assign the best cluster number to
    the "corresponding" label number such that the entries on the main
    diagonal are maximized.
    Inputs:
        num_clusters - number of classes/clusters expected
        length - number of reports/documents available
        clusters [No_documents x 1] - column vector with predicted cluster for
            each document index
        number_of_labels_provided - text to be output giving the number of
            manually labeled sets
        test - index of documents to be used as train labels
    Outputs:
        file - .txt file containing the indexes of files used as given labels
            and accuracy matrix
        newdir - name of directory containing the file
    """
    # create confusion matrix for the labels

    countvec = []
    for i in range(0, num_clusters):
        countvec.append(dict.fromkeys(range(0, num_clusters), 0))

    labels = {}
    # create confusion matrices by finding corresponding clusters
    # at every run the computer changes the ids of the clusters and we want to find the
    # correlation between our counting system
    fr_names = list(id2name.values())
    name2id = {v: k for k, v in id2name.items()}
    sqldata_id = "select * from an"
    mycursor.execute(sqldata_id)
    recorddata_id = mycursor.fetchall()
    label_dict = defaultdict()
    labels_words = []
    for results in recorddata_id:
        ID = results[0]  # get ID of each recorded failure

        label = results[3] + results[1] + results[4] + results[2]
        labels_words.append(label)
        label_dict[ID] = label
    classnames, indices = np.unique(labels_words, return_inverse=True)
    for i in range(0, length):
        fr_name = fr_names[i]
        fr_id = name2id[fr_name]
        id_to_label = label_dict[fr_id]
        label_to_class_idx = list(classnames).index(id_to_label)
        countvec[label_to_class_idx][clusters[i]] += 1
    bestSoFar = 0
    for p in permutations(list(range(0, 7))):
        cost = 0
        for i in range(0, 7):
            cost += countvec[i][p[i]]
        if cost > bestSoFar:
            bestSoFar = cost
            labels = p
    labels_to_clusters = defaultdict()
    counter = 0
    for label in labels:
        labels_to_clusters[counter] = label
        counter += 1

    now = datetime.now()
    current_time = now.strftime("%H_%M_%S")
    current_date = now.date()
    newdir = str(results_folder) + "/Results_" + str(current_date) + "_" + current_time
    os.mkdir(newdir)
    f = open(newdir + "/results.txt", 'w')
    f1 = open(newdir + "/nlp_cluster_to_label_association.txt", 'w')
    for i in range(0, 7):
        for j in range(0, 7):
            print(countvec[i][labels[j]], end=" ", file=f)
            print(countvec[i][labels[j]], end=" ", file=f1)
        print(file=f)
        print(file=f1)
    print(number_of_labels_provided, file=f)
    print(number_of_labels_provided, file=f1)
    print(file=f)
    print(file=f1)
    print("Indices of the files used:", file=f)
    print("Indices of the files used:", file=f1)
    print(test, file=f)
    print(test, file=f1)
    f1.close()
    cluster_to_labels = {v: k for k, v in labels_to_clusters.items()}
    return f, newdir, cluster_to_labels
