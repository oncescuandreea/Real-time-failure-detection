# -*- coding: utf-8 -*-

# feature extraction for gsr data
import mysql.connector
import argparse
import numpy as np
from scipy.stats import kurtosis
from scipy.stats import skew
from pathlib import Path
from utils.feature_extraction_utils import generate_add_sql_command, get_id_data, create_sql_table, delete_table


def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[result.size // 2:]


def extract_gsr_features(list_of_ids: list,
                         mycursor: mysql.connector.cursor,
                         add_to_sql_command: str,
                         cnx: mysql.connector,
                         data_folder_location: Path):
    for one_id in list_of_ids:
        measurement_id = one_id[0]
        data = get_id_data(mycursor, measurement_id, data_folder_location)
        column2 = []
        # find gradient of temperature values and maximum number of increments and final +- equilibirum
        grad = []  # change between consecutive points
        signch = []  # changes in sign of gradient
        count = 0
        maximum_between_consecutive_points = 10  # check if there is a more sudden change between consecutive values than 10
        no_change_counter = 0  # how many consecutive data points are not changing
        x = 0
        maximum_settle = 0
        rem2 = 0
        gsr_feature_vector = []
        for row in data:
            if len(row) > 4 and row[4] != '':
                column2.append(row[4])
                if count != 0:
                    diff = float(column2[count]) - float(column2[count - 1])
                    if abs(diff) > maximum_between_consecutive_points:
                        maximum_between_consecutive_points = abs(diff)
                        remMax = max(float(column2[count]), float(column2[count - 1]))
                        remMin = min(float(column2[count]), float(column2[count - 1]))
                    grad.append(diff)
                    if diff > 0:
                        x = x + 1
                        no_change_counter = 0
                    elif diff < 0:
                        x = x - 1
                    else:
                        no_change_counter = no_change_counter + 1
                    signch.append(x)
                    if no_change_counter > maximum_settle:
                        maximum_settle = no_change_counter
                        rem2 = row[4]  # remember value for which you have the maximum interval of no changing values
                count = count + 1
        c2 = np.array(column2)
        c2 = c2.astype(np.float64)
        gsr_feature_vector.append(str(measurement_id))
        gsr_feature_vector.append(np.mean(c2))
        gsr_feature_vector.append(np.var(c2))
        gsr_feature_vector.append(skew(c2))
        gsr_feature_vector.append(kurtosis(c2))
        gsr_feature_vector.append(max(autocorr(c2)))
        gsr_feature_vector.append(min(autocorr(c2)))
        gsr_feature_vector.append(maximum_between_consecutive_points)
        gsr_feature_vector.append(remMax)
        gsr_feature_vector.append(remMin)
        gsr_feature_vector.append(rem2)
        gsr_feature_vector.append(maximum_settle)
        gsr_feature_vector = [gsr_feature_vector[0]] + list(map(float, gsr_feature_vector[1:]))
        mycursor.execute(add_to_sql_command, gsr_feature_vector)
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
        default='C:/Users/oncescu/OneDrive - Nexus365/Data',
    )
    args = parser.parse_args()
    cnx = mysql.connector.connect(user='root', password=args.sql_password,
                                  host='127.0.0.1',
                                  database='final_bare')
    mycursor = cnx.cursor()

    sql = "Select ID from an"
    mycursor.execute(sql)
    list_of_ids = list(mycursor.fetchall())
    no_features = 11
    no_decimals = 2
    table_name = 'gsr3'
    try:
        create_sql_table(mycursor, table_name, no_features, no_decimals)
    except mysql.connector.errors.ProgrammingError:
        print("Table already created. Replacing values")
        delete_table(table_name, cnx)
        create_sql_table(mycursor, table_name, no_features, no_decimals)
    add_to_sql_command = generate_add_sql_command(table_name, no_features)
    print("Extracting GSR features")
    extract_gsr_features(list_of_ids, mycursor,
                         add_to_sql_command, cnx,
                         args.data_folder_location)
    print("Finished extracting and adding features to sql")


if __name__ == "__main__":
    main()
