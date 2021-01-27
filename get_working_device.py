# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 23:46:57 2020
File used to generate an sql table containing random sets of
working conditions data. It does so by randomly picking failure
datasets and taking the numbers corresponding to features of working
sensors and then concatenating them.
@author: Andreea
"""

import mysql.connector
import random
from feature_extraction_utils import create_sql_table, generate_add_sql_command


def main():
    cnx = mysql.connector.connect(user='root', password='sqlAmonouaparola213',
                                  host='127.0.0.1',
                                  database='final')
    mycursor = cnx.cursor()
    sqldataID = "select * from an"
    mycursor.execute(sqldataID)
    list_of_labels = mycursor.fetchall()

    dict_sensors = {'GSR': 'gsr2', 'Acc': 'FEAT2', 'Hum': 'hum3', 'Temp': 'tempd5'}
    no_features = 73
    no_decimals = 10
    try:
        create_sql_table(mycursor, 'FEAT_working', no_features, no_decimals)
    except mysql.connector.errors.ProgrammingError:
        print("Table already created")
    add_to_sql_command = generate_add_sql_command('FEAT_working', no_features)
    count = 1
    while count <= 117:
        features = []
        ID_list = []
        features.append(count)
        for sensor in ['GSR', 'Acc', 'Hum', 'Temp']:
            found = False
            while found is False:
                random_choice_ID = random.randint(0, 116)
                ID = list_of_labels[random_choice_ID][0]
                if sensor == 'GSR':
                    if list_of_labels[random_choice_ID][3] == 'working':
                        found = True
                elif sensor == 'Acc':
                    if list_of_labels[random_choice_ID][1] == 'working':
                        found = True
                elif sensor == 'Hum':
                    if list_of_labels[random_choice_ID][4] == 'working':
                        found = True
                else:
                    if list_of_labels[random_choice_ID][2] == 'working':
                        found = True
            ID_list.append(ID)
            sql = f"select * from {dict_sensors[sensor]} where ID='" + ID + "'"
            mycursor.execute(sql)
            list_of_features = mycursor.fetchall()
            for el in list_of_features[0][1:]:
                features.append(el)

        mycursor = cnx.cursor()
        mycursor.execute(add_to_sql_command, features)
        cnx.commit()
        count += 1


if __name__ == "__main__":
    main()
