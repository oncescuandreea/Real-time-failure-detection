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
import argparse
from utils.feature_extraction_utils import delete_table

random.seed(0)


def create_table_sql(mycursor: mysql.connector.cursor, table_name: str, number_features: int, no_decimals: int):
    string = f"CREATE TABLE {table_name} (Counter INT"
    for i in range(0, number_features):
        string = string + ",Feat" + str(i) + f" Float(15,{no_decimals})"
    string = string + ")"
    mycursor.execute(string)


def generate_add_sql_command(table_name: str, number_entries: int):
    add_to_sql_command = f"INSERT INTO {table_name} (Counter"
    for i in range(0, number_entries):
        add_to_sql_command = add_to_sql_command + ",Feat" + str(i)
    add_to_sql_command = add_to_sql_command + ") VALUES (%s"
    for i in range(0, number_entries):
        add_to_sql_command = add_to_sql_command + ",'%s'"
    add_to_sql_command = add_to_sql_command + ")"
    return add_to_sql_command


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sql_password",
        type=str,
    )
    args = parser.parse_args()
    cnx = mysql.connector.connect(user='root', password=args.sql_password,
                                  host='127.0.0.1',
                                  database='final_bare')
    mycursor = cnx.cursor()
    sqldataID = "select * from an"
    mycursor.execute(sqldataID)
    list_of_labels = mycursor.fetchall()

    dict_sensors = {'GSR': 'gsr3', 'Acc': 'FEAT3', 'Hum': 'hum4', 'Temp': 'tempd6'}
    no_features = 73
    no_decimals = 4
    table_name = 'FEAT_working2'
    try:
        create_table_sql(mycursor, table_name, no_features, no_decimals)
    except mysql.connector.errors.ProgrammingError:
        print("Table already created. Replacing values")
        delete_table(table_name, cnx)
        create_table_sql(mycursor, table_name, no_features, no_decimals)
    add_to_sql_command = generate_add_sql_command(table_name, no_features)
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
