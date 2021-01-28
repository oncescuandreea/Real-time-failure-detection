import csv
import mysql.connector
from pathlib import Path


def generate_add_sql_command(table_name: str, number_entries: int):
    add_to_sql_command = f"INSERT INTO {table_name} (ID"
    for i in range(0, number_entries):
        add_to_sql_command = add_to_sql_command + ",Feat" + str(i)
    add_to_sql_command = add_to_sql_command + ") VALUES (%s"
    for i in range(0, number_entries):
        add_to_sql_command = add_to_sql_command + ",'%s'"
    add_to_sql_command = add_to_sql_command + ")"
    return add_to_sql_command


def get_id_data(mycursor: mysql.connector.cursor, measurement_id: str,
                data_folder_location: Path):
    sql_get_data_name = "Select nameData from corr where ID='" + measurement_id + "'"
    mycursor.execute(sql_get_data_name)
    data_file_name = mycursor.fetchall()
    data = csv.reader(open(data_folder_location / data_file_name[0][0], 'rt'),
                      delimiter=",", quotechar='|')
    return data


def create_sql_table(mycursor: mysql.connector.cursor, table_name: str, number_features: int, no_decimals: int):
    """
    Function creates sql tables. Existent tables have the following properties:
    accelerometer - table_name: feat2, number_features: 48 (42?), no_decimals: 10
    gsr - table_name: gsr2, number_features: 11, no_decimals: 2
    temperature - table_name: tempd5, number_features: 7, no_decimals: 4
    humidity - table_name: hum3, number_features: 7, no_decimals: 4
    @param mycursor: mysql.connector.cursor
    @param table_name: name of table
    @param number_features: number of feature columns
    @param no_decimals: number of digits after decimal point
    """
    string = f"CREATE TABLE {table_name} (ID VARCHAR(255)"
    for i in range(0, number_features):
        string = string + ",Feat" + str(i) + f" Float(15,{no_decimals})"
    string = string + ")"
    mycursor.execute(string)


def delete_table(table_name: str, cnx: mysql.connector):
    mycursor = cnx.cursor()
    string = f"drop table {table_name}"
    mycursor.execute(string)
    cnx.commit()
