import os

os.environ['KERAS_BACKEND'] = 'theano'
import argparse
import mysql.connector
import numpy as np
from scipy.stats import kurtosis
from scipy.stats import skew
from pathlib import Path
from utils.feature_extraction_utils import generate_add_sql_command, get_id_data, create_sql_table, delete_table


def extract_temperature_features(list_of_ids: list,
                                 mycursor: mysql.connector.cursor,
                                 add_to_sql_command: str,
                                 cnx: mysql.connector,
                                 data_folder_location: Path):
    for one_id in list_of_ids:
        measurement_id = one_id[0]

        data = get_id_data(mycursor, measurement_id, data_folder_location)
        temperature_raw_measurements = []  # list containing temperature sensor values for given document id
        app = [0] * 255  # no of apparition of each temperature
        no_neg = 0  # no of negative temperatures
        maximum_difference = 2  # checks if the change in values is too large for what the temperature sensor can record in that timestep; change of 0.5deg in a specific time?
        maximum_gradient = 0  # maximum difference obtained between two consecutive measurements in absolute value
        val = 0  # most frequent value achieved when there is a sudden change between two consecutive measurements
        maxapp = 0  # number of appearances of the most frequent value achieved when there is a sudden change between
        # two consecutive measurements
        countextra = 0  # tells when 2nd recorded value was reached such that gradient can be calculated
        no_sudden_changes = 0  # how many sudden jumps
        for row in data:
            if len(row) > 4 and row[4] != '':
                temperature_raw_measurements.append(float(row[5]))
                if float(row[5]) < 0:
                    no_neg += 1
                if countextra != 0:
                    diff = temperature_raw_measurements[countextra] - temperature_raw_measurements[countextra - 1]
                    if abs(diff) > maximum_difference:
                        if abs(temperature_raw_measurements[countextra]) > abs(
                                temperature_raw_measurements[countextra - 1]):
                            maxval = temperature_raw_measurements[countextra]
                        else:
                            maxval = temperature_raw_measurements[countextra - 1]
                        app[int(maxval) + 127] = app[int(maxval) + 127] + 1  # when doing this, cannot have negative
                        # indexes so shift everything by -127 since that's min val

                        no_sudden_changes += 1
                        if abs(diff) > maximum_gradient:
                            maximum_gradient = abs(diff)
                countextra = countextra + 1
        if no_sudden_changes != 0:
            for i in range(0, 255):
                if app[i] > maxapp:
                    val = i - 127
                    maxapp = app[i]
        else:
            val = 0

        sql_temperature_features = [str(measurement_id), np.mean(temperature_raw_measurements),
                                    np.var(temperature_raw_measurements), skew(temperature_raw_measurements),
                                    kurtosis(temperature_raw_measurements),
                                    maximum_gradient, val, maxapp]
        sql_temperature_features = [sql_temperature_features[0]] + list(map(float, sql_temperature_features[1:]))
        mycursor.execute(add_to_sql_command, sql_temperature_features)
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
    parser.add_argument(
        "--database_name",
        type=str,
        default='final',
    )
    args = parser.parse_args()
    cnx = mysql.connector.connect(user='root', password=args.sql_password,
                                  host='127.0.0.1',
                                  database=args.database_name)
    mycursor = cnx.cursor()

    sql = "Select ID from an"
    mycursor.execute(sql)
    list_of_ids = list(mycursor.fetchall())
    no_features = 7
    no_decimals = 4
    table_name = 'tempd6'
    try:
        create_sql_table(mycursor, table_name, no_features, no_decimals)
    except mysql.connector.errors.ProgrammingError:
        print("Table already created. Replacing values")
        delete_table(table_name, cnx)
        create_sql_table(mycursor, table_name, no_features, no_decimals)
    add_to_sql_command = generate_add_sql_command(table_name, no_features)
    print("Extracting temperature features")
    extract_temperature_features(list_of_ids, mycursor,
                                 add_to_sql_command, cnx,
                                 args.data_folder_location)
    print("Finished extracting and adding features to sql")


if __name__ == "__main__":
    main()
