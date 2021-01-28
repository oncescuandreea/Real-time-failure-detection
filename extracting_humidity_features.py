# -*- coding: utf-8 -*-
# file used to extract humidity features and save them to sql
import mysql.connector
import argparse
import numpy as np
from scipy.stats import kurtosis
from scipy.stats import skew
from pathlib import Path
from utils.feature_extraction_utils import generate_add_sql_command, get_id_data, create_sql_table, delete_table


def extract_humidity_features(list_of_ids: list,
                              mycursor: mysql.connector.cursor,
                              add_to_sql_command: str,
                              cnx: mysql.connector,
                              data_folder_location: Path):
    for one_id in list_of_ids:
        measurement_id = one_id[0]
        data = get_id_data(mycursor, measurement_id, data_folder_location)
        max_difference = 0  # maximum difference obtained between two consecutive measurements in absolute value
        val = 0  # most frequent value achieved when there is a sudden change between two consecutive measurements
        val_max_counter = 0  # number of appearances of the most frequent value achieved when there is a sudden
        # change between two consecutive measurements
        humidity_raw_measurements = []  # list containing humidity sensor values in each file
        app = [0] * 500  # no of apparition of each humidity raw value
        noNeg = 0  # no of negative humidity values
        max_change = 2  # checks if change in values is too large for what humidity sensor can record in that timestep
        raw_data_counter = 0
        no_sudden_jumps = 0  # how many sudden jumps
        for row in data:
            if len(row) > 4 and row[4] != '':
                humidity_raw_measurements.append(float(row[6]))
                if float(row[6]) < 0:
                    noNeg += 1
                if raw_data_counter != 0:
                    diff = humidity_raw_measurements[raw_data_counter] - humidity_raw_measurements[raw_data_counter - 1]
                    if abs(diff) > max_change:
                        if abs(humidity_raw_measurements[raw_data_counter]) > abs(humidity_raw_measurements[raw_data_counter - 1]):
                            maxval = humidity_raw_measurements[raw_data_counter]
                        else:
                            maxval = humidity_raw_measurements[raw_data_counter - 1]
                        app[int(maxval) + 300] = app[int(maxval) + 300] + 1  # when doing this, cannot have negative
                        # indexes so shift everything by -300 since that's min val

                        no_sudden_jumps += 1
                        if abs(diff) > max_difference:
                            max_difference = abs(diff)
                raw_data_counter = raw_data_counter + 1
        if no_sudden_jumps != 0:
            for i in range(0, 500):
                if app[i] > val_max_counter:
                    val = i - 300
                    val_max_counter = app[i]
        else:
            val = 0
        hum_feature_vector = [str(measurement_id), np.mean(humidity_raw_measurements),
                              np.var(humidity_raw_measurements), skew(humidity_raw_measurements),
                              kurtosis(humidity_raw_measurements), max_difference, val, val_max_counter]
        hum_feature_vector = [hum_feature_vector[0]] + list(map(float, hum_feature_vector[1:]))
        mycursor.execute(add_to_sql_command, hum_feature_vector)
        cnx.commit()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sql_password",
        type=str,
    )
    parser.add_argument(
        "--data_folder_location",
        type=Path,
        default='C:/Users/oncescu/data/4yp-data',
    )
    args = parser.parse_args()
    cnx = mysql.connector.connect(user='root', password=args.sql_password,
                                  host='127.0.0.1',
                                  database='final_bare')
    mycursor = cnx.cursor()

    sql = "Select ID from an"
    mycursor.execute(sql)
    list_of_ids = list(mycursor.fetchall())
    no_features = 7
    no_decimals = 2
    table_name = 'hum4'
    try:
        create_sql_table(mycursor, table_name, no_features, no_decimals)
    except mysql.connector.errors.ProgrammingError:
        print("Table already created. Replacing values")
        delete_table(table_name, cnx)
        create_sql_table(mycursor, table_name, no_features, no_decimals)
    add_to_sql_command = generate_add_sql_command(table_name, no_features)
    print("Extracting humidity features")
    extract_humidity_features(list_of_ids, mycursor,
                              add_to_sql_command, cnx,
                              args.data_folder_location)
    print("Finished extracting and adding features to sql")


if __name__ == "__main__":
    main()
