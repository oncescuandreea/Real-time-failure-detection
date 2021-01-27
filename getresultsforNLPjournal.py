# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 15:45:13 2019

@author: Andreea
"""

# code used to provide tables in the needed format for LaTex file

import mysql.connector  # connect to database
import numpy as np
from collections import defaultdict


def get_mean_variance_latex(cnx: mysql.connector, sensor: str):
    """
    Function generates latex strings to be replaced in tables containing
    the mean and variacne of the sensors' features for each working or
    failure case. For the failure cases the lines are in bold.
    Inputs:
        cnx: connector to sql database "final"
        sensor: name of the sensor for which the table is wanted
            can be tempd3, hum3, gsr2 or feat2
    Outputs:
        Prints the latex code in the order first the lines corresponding
        to the means and then the ones corresponding to the variance
    """
    mycursor = cnx.cursor()
    sql = "select * from an"
    label_to_ID = defaultdict(list)
    mycursor.execute(sql)
    result_labels = mycursor.fetchall()
    label_to_features = defaultdict(list)
    for label in result_labels:
        label_to_ID[label[2]].append(label[0])
    mean_latex_lines = []
    variance_latex_lines = []
    for label in label_to_ID.keys():
        for ID in label_to_ID[label]:
            sql = f"select * from {sensor} where ID='{ID}'"
            mycursor.execute(sql)
            result_features = mycursor.fetchall()
            label_to_features[label].append(list(result_features[0][1:]))
        label_to_features[label] = np.transpose(label_to_features[label])
        number_features = len(label_to_features[label]) - 1
        if label == 'working':
            textbf_before = ''
            textbf_after = ''
        else:
            textbf_before = '\\textbf{'
            textbf_after = '}'
        str_to_print = textbf_before + "mean" + textbf_after + " & "
        for i in range(0, number_features):
            mean_value = round(np.mean(label_to_features[label][i]), 2)
            str_to_print += textbf_before + str(mean_value) + textbf_after + " & "
        mean_value = round(np.mean(label_to_features[label][number_features]), 2)
        str_to_print += textbf_before + str(mean_value) + textbf_after + " \\\\"
        mean_latex_lines.append(str_to_print)

        str_to_print = textbf_before + "variance" + textbf_after + " & "
        for i in range(0, number_features):
            var_value = round(np.var(label_to_features[label][i]), 2)
            str_to_print += textbf_before + str(var_value) + textbf_after + " & "
        var_value = round(np.var(label_to_features[label][number_features]), 2)
        str_to_print += textbf_before + str(var_value) + textbf_after + " \\\\"
        variance_latex_lines.append(str_to_print)
    for mean_string in mean_latex_lines:
        print('\\hline')
        print(mean_string)
    for var_string in variance_latex_lines:
        print('\\hline')
        print(var_string)


def get_top_14_words(cnx: mysql.connector, ID: str):
    """
    Function prints the contents of wordrep100 table for the given ID.
    This table contains top 14 words in descending order given the
    number of appearances. The format is ID, Word, NoApp, TFIDF, count.
    This function only prints the word, the number of appearances and
    the TFIDF score of each word for that report. This print is in
    latex format for the report.
    Inputs:
        ID: id of report of interest (eg. 080320191037030137)
    """
    mycursor = cnx.cursor()
    sql = f"select * from wordrep100 where MeasID={ID}"
    mycursor.execute(sql)
    results = mycursor.fetchall()
    for result in results:
        print('\hline')
        print(result[1] + ' & ' + str(result[2]) + ' & ' + str(float("{0:.4f}".format(result[3]))) + '\\' + '\\')


def get_features_example_tables_temp(cnx: mysql.connector):
    """
    Function prints 6 examples for the features of the
    temperature sensor. This is done by accessing the
    tempd3 table which has the form ID, Feat0-6
    Inputs:
        cnx: sql connector
    """
    mycursor = cnx.cursor()
    list_of_id_reports = ['23022019201803027', '20022019171303018', '25012019144103005', '03022019214503011',
                          '29102018104303002', '080320191024030127']
    for id_report in list_of_id_reports:
        sql = "select * from tempd3 where ID='" + id_report + "'"
        mycursor.execute(sql)
        id_feature_list = mycursor.fetchall()
        for feature_list in id_feature_list:
            sql = "select tempD from an where ID=" + id_report
            mycursor.execute(sql)
            id_label = mycursor.fetchall()

            print('\\hline')
            if id_label[0][0] == 'notworking':
                print('\\textbf{' + feature_list[0] + '} & ' + '\\textbf{' + str(
                    feature_list[1]) + '} & ' + '\\textbf{' + str(
                    feature_list[2]) + '} & ' + '\\textbf{' + str(feature_list[3]) + '} & ' + '\\textbf{' + str(
                    feature_list[4]) + '} & ' + '\\textbf{' + str(feature_list[5]) + '} & ' + '\\textbf{' + str(
                    feature_list[6]) + '} & ' + '\\textbf{' + str(feature_list[7]) + '}\\' + '\\')
            else:
                print(
                    feature_list[0] + ' & ' + str(feature_list[1]) + ' & ' + str(feature_list[2]) + ' & ' + str(
                        feature_list[3]) + ' & ' + str(
                        feature_list[4]) + ' & ' + str(feature_list[5]) + ' & ' + str(feature_list[6]) + ' & ' + str(
                        feature_list[7]) + '\\' + '\\')


def get_features_example_tables_hum(cnx: mysql.connector):
    """
    Function prints 6 examples for the features of the
    temperature sensor. This is done by accessing the
    hum3 table which has the form ID, Feat0-6
    Inputs:
        cnx: sql connector
    """
    mycursor = cnx.cursor()
    list_of_id_reports = ['060320191159030106', '04032019165503090', '060320191207030114', '060320191216030120',
                          '070320191827030126', '080320191036030136']
    for id_report in list_of_id_reports:
        sql = "select * from hum3 where ID='" + id_report + "'"
        mycursor.execute(sql)
        id_feature_list = mycursor.fetchall()
        for feature_list in id_feature_list:
            sql = "select hum from an where ID=" + id_report
            mycursor.execute(sql)
            id_label = mycursor.fetchall()

            print('\\hline')
            if id_label[0][0] == 'pin':
                print('\\textbf{' + feature_list[0] + '} & ' + '\\textbf{' + str(
                    feature_list[1]) + '} & ' + '\\textbf{' + str(
                    feature_list[2]) + '} & ' + '\\textbf{' + str(feature_list[3]) + '} & ' + '\\textbf{' + str(
                    feature_list[4]) + '} & ' + '\\textbf{' + str(feature_list[5]) + '} & ' + '\\textbf{' + str(
                    feature_list[6]) + '} & ' + '\\textbf{' + str(feature_list[7]) + '}\\' + '\\')
            else:
                print(
                    feature_list[0] + ' & ' + str(feature_list[1]) + ' & ' + str(feature_list[2]) + ' & ' + str(
                        feature_list[3]) + ' & ' + str(
                        feature_list[4]) + ' & ' + str(feature_list[5]) + ' & ' + str(feature_list[6]) + ' & ' + str(
                        feature_list[7]) + '\\' + '\\')


def get_features_example_tables_gsr(cnx: mysql.connector):
    """
    Function prints 6 examples for the features of the
    temperature sensor. This is done by accessing the
    gsr2 table which has the form ID, Feat0-10
    Inputs:
        cnx: sql connector
    """
    mycursor = cnx.cursor()
    list_of_id_reports = ['25022019115103034', '04032019113703056', '26022019265103041', '04032019160703078',
                          '19022019174503015', '080320191024030127', '23022019202403028', '04032019165803093']

    for id_report in list_of_id_reports:
        sql = "select * from gsr2 where ID='" + id_report + "'"
        mycursor.execute(sql)
        id_feature_list = mycursor.fetchall()
        for feature_list in id_feature_list:
            sql = "select gsr from an where ID=" + id_report
            mycursor.execute(sql)
            id_label = mycursor.fetchall()

            print('\hline')
            if id_label[0][0] == 'ground' or id_label[0][0] == 'analog' or id_label[0][0] == 'resistor':
                print('\\textbf{' + feature_list[0] + '} & ' + '\\textbf{' + str(
                    feature_list[1]) + '} & ' + '\\textbf{' + str(
                    feature_list[2]) + '} & ' + '\\textbf{' + str(feature_list[3]) + '} & ' + '\\textbf{' + str(
                    feature_list[4]) + '} & ' + '\\textbf{' + str(feature_list[5]) + '} & ' + '\\textbf{' + str(
                    feature_list[6]) + '} & ' + '\\textbf{' + str(feature_list[7]) + '} & ' + '\\textbf{' + str(
                    feature_list[8]) + '} & ' + '\\textbf{' + str(feature_list[9]) + '} & ' + '\\textbf{' + str(
                    feature_list[10]) + '} & ' + '\\textbf{' + str(feature_list[11]) + '}\\' + '\\')
            else:
                print(
                    feature_list[0] + ' & ' + str(feature_list[1]) + ' & ' + str(feature_list[2]) + ' & ' + str(
                        feature_list[3]) + ' & ' + str(
                        feature_list[4]) + ' & ' + str(feature_list[5]) + ' & ' + str(feature_list[6]) + ' & ' + str(
                        feature_list[7]) + ' & ' + str(feature_list[8]) + ' & ' + str(feature_list[9]) + ' & ' + str(
                        feature_list[10]) + ' & ' + str(feature_list[11]) + '\\' + '\\')


def get_features_example_tables_acc(cnx: mysql.connector):
    """
        Function prints 12 examples for the features of the
        accelerometer. This is done by accessing the
        feat2 table which has the form ID, Feat0-47
        Inputs:
            cnx: sql connector
        """
    mycursor = cnx.cursor()
    list_of_id_reports = ['03022019211803009', '03022019214503011', '080320191035030134', '080320191038030138',
                          '25022019121703037', '060320191215030116', '060320191207030112', '23022019203603031',
                          '19022019174503015', '19022019175103017', '080320191026030130', '080320191027030132']
    labels = ['ground'] * 4 + ['working'] * 4 + ['power'] * 4
    for id_report in list_of_id_reports:
        sql = "select * from FEAT2 where ID='" + id_report + "'"
        mycursor.execute(sql)
        id_feature_list = mycursor.fetchall()
        for feature_list in id_feature_list:
            sql = "select acc from an where ID=" + id_report
            mycursor.execute(sql)
            id_label = mycursor.fetchall()
            result = np.array(feature_list[1:])
            result = np.around(result, decimals=2)
            i = 2
            print('\hline')
            if id_label[0][0] == 'power' or id_label[0][0] == 'ground':
                print('\\textbf{' + feature_list[0] + '} & ' + '\\textbf{' + str(
                    result[0 + i]) + '} & ' + '\\textbf{' + str(
                    result[3 + i]) + '} & ' + '\\textbf{' + str(result[6 + i]) + '} & ' + '\\textbf{' + str(
                    result[9 + i]) + '} & ' + '\\textbf{' + str(result[12 + i]) + '} & ' + '\\textbf{' + str(
                    result[15 + 2 * i]) + '} & ' + '\\textbf{' + str(result[16 + 2 * i]) + '} & ' + '\\textbf{' + str(
                    result[21 + 2 * i]) + '} & ' + '\\textbf{' + str(result[22 + 2 * i]) + '} & ' + '\\textbf{' + str(
                    result[45 + i]) + '}\\' + '\\')
            else:
                print(feature_list[0] + ' & ' + str(result[0 + i]) + ' & ' + str(result[3 + i]) + ' & ' + str(
                    result[6 + i]) + ' & ' + str(result[9 + i]) + ' & ' + str(result[12 + i]) + ' & ' + str(
                    result[15 + 2 * i]) + ' & ' + str(result[16 + 2 * i]) + ' & ' + str(
                    result[21 + 2 * i]) + ' & ' + str(result[22 + 2 * i]) + ' & ' + str(result[45 + i]) + '\\' + '\\')


def find_missing_id_and_remove(cnx: mysql.connector):
    """
    Function searches to see if there is any mismatch in the number
    and contents of table gsr2 and the table containing all labels
    for all report IDs. If such documents are found then they are
    deleted from all tables in the database
    """
    mycursor = cnx.cursor()
    sql = "select * from gsr2"
    mycursor.execute(sql)
    gsr2_contents = mycursor.fetchall()
    extra_ids = []
    for result in gsr2_contents:
        id: str = result[0]
        # search for that id in the big table with all labels
        sql = "select * from an where ID='" + id + "'"
        mycursor.execute(sql)
        results = mycursor.fetchall()
        if len(results) == 0:
            print(id)
            extra_ids.append(id)
    for id in extra_ids:
        # this part deletes an extra report from all tables
        sql = "show tables"
        mycursor.execute(sql)
        table_names = mycursor.fetchall()
        for name in table_names:
            sql = "describe " + name[0]
            mycursor.execute(sql)
            id_field_name = mycursor.fetchall()  # each table can have a different ID column name

            sql = f"delete from {name[0]} + where {id_field_name[0][0]}={id}"
            mycursor.execute(sql)
            cnx.commit()


def main():
    cnx = mysql.connector.connect(user='root', password='sqlAmonouaparola213',
                                  host='127.0.0.1',
                                  database='final')


if __name__ == "__main__":
    main()
