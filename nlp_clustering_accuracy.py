import argparse
from pathlib import Path
import mysql.connector
from utils.utils_ml import tf_idf_retrieval, id_to_name, get_data_from_different_labels_for_cluster_initialisation
from utils.utils_nlp import kmeans_clustering, nlp_labels_analysis
import pandas as pd
import os
import time
import random

random.seed(0)


def run_random_seed_exp(no_labeled_sets: int, id2name: dict, final: pd.DataFrame, indexname: list,
                        cnx: mysql.connector, num_clusters: int, mycursor: mysql.connector.cursor,
                        length: int, results_folder: Path, refresh: bool):
    counter = 0
    if os.path.exists(results_folder / f"nlp_clustering_{no_labeled_sets}") and refresh is False:
        print(f"Experiments have already been run, proceeding to summary")
    else:
        while counter < 100:

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
            (results_folder / f"nlp_clustering_{no_labeled_sets}").mkdir(exist_ok=True, parents=True)
            f, _, _ = nlp_labels_analysis(mycursor,
                                          num_clusters=num_clusters,
                                          length=length,
                                          clusters=clusters,
                                          number_of_labels_provided=number_labels_provided,
                                          test=test,
                                          id2name=id2name,
                                          results_folder=results_folder / f"nlp_clustering_{no_labeled_sets}")
            time.sleep(2)
            f.close()
            counter += 1


def summarise_results(results_folder, length, no_labeled_sets):
    generated_folders = os.listdir(results_folder / f"nlp_clustering_{no_labeled_sets}")
    max_accuracy = 0
    min_accuracy = 1
    print("Going through generated data and finding maximum and minimum accuracy and location")
    for folder in generated_folders:
        with open(
                results_folder / f"nlp_clustering_{no_labeled_sets}" / folder / "nlp_cluster_to_label_association.txt",
                'r') as f:
            matrix = f.read().splitlines()
        sum_main_diagonal = 0
        for idx, row in enumerate(matrix[0:7]):
            sum_main_diagonal += int(row.split()[idx])
        accuracy = sum_main_diagonal / length
        if accuracy >= max_accuracy:
            max_accuracy = accuracy
            max_folder = folder
        if accuracy <= min_accuracy:
            min_accuracy = accuracy
            min_folder = folder
    return max_accuracy, max_folder, min_accuracy, min_folder


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
        default='final',
    )
    parser.add_argument(
        "--refresh",
        type=bool,
        default=False,
    )
    args = parser.parse_args()
    # connect to database
    cnx = mysql.connector.connect(user='root', password=args.sql_password,
                                  host='127.0.0.1',
                                  database=args.database_name)

    mycursor = cnx.cursor()

    [listTFIDF, indexname, length] = tf_idf_retrieval(mycursor, args.database_name)
    # final is the dataframe containing the names of the reports and the corresponding tfidf vectors. if add
    # indexname to command below then index is named as the report pd.DataFrame(listTFIDF, indexname)
    final = pd.DataFrame(listTFIDF)
    id2name = id_to_name(mycursor)

    num_clusters = 7
    number_of_tests = args.provided_labels

    no_labeled_sets = number_of_tests

    run_random_seed_exp(no_labeled_sets, id2name, final, indexname,
                        cnx, num_clusters, mycursor, length, args.results_folder, args.refresh)

    list_accuracies = summarise_results(args.results_folder, length, no_labeled_sets)
    print(list_accuracies)


if __name__ == "__main__":
    main()
