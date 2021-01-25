import os
os.environ['KERAS_BACKEND'] = 'theano'
import csv
import mysql.connector
import numpy as np
from scipy.stats import kurtosis
from scipy.stats import skew


def get_id_data(mycursor, measurement_id):
    sql_get_data_name = "Select nameData from corr where ID='" + str(measurement_id) + "'"
    mycursor.execute(sql_get_data_name)
    data_file_name = mycursor.fetchall()
    data = csv.reader(open('C:/Users/oncescu/OneDrive - Nexus365/Data/' + data_file_name[0][0], 'rt'),
                      delimiter=",", quotechar='|')
    return data


def extract_temperature_features(list_of_ids, mycursor,
                                 add_to_sql_command, cnx):
    for one_id in list_of_ids:
        val = 0
        measurement_id = one_id[0]

        data = get_id_data(mycursor, measurement_id)
        temperature_raw_measurements = []  # list containing temperature sensor values for given document id
        # find gradient of temperature values and maximum number of increments and final +- equilibirum

        app = [0] * 255  # no of apparition of each temperature
        maxim = -200
        no_neg = 0  # no of negative temperatures
        maximum_difference = 2  # checks if the change in values is too large for what the temperature sensor can record in that timestep; change of 0.5deg in a specific time?
        maximum_gradient = 0
        no_of_0s = 0  # how many consecutive data points are not changing
        x = 0
        maxapp = 0
        countextra = 0  # tells when 2nd recorded value was reached such that gradient can be calculated
        countapp = 0  # how many sudden jumps
        for row in data:
            if len(row) > 4 and row[4] != '':
                temperature_raw_measurements.append(float(row[5]))
                if float(row[5]) < 0:
                    no_neg += 1
                if countextra != 0:
                    diff = temperature_raw_measurements[countextra] - temperature_raw_measurements[countextra - 1]
                    if abs(diff) > maximum_difference:
                        if abs(temperature_raw_measurements[countextra]) > abs(temperature_raw_measurements[countextra - 1]):
                            maxval = temperature_raw_measurements[countextra]
                        else:
                            maxval = temperature_raw_measurements[countextra - 1]
                        app[int(maxval) + 127] = app[int(
                            maxval) + 127] + 1  # wgeb doing this, cannot have negative indexes so shift everything by -300 since that's min val

                        countapp += 1
                        if abs(diff) > maximum_gradient:
                            maximum_gradient = abs(diff)
                    if diff > 0:
                        x = x + 1
                    elif diff < 0:
                        x = x - 1
                    else:
                        no_of_0s = no_of_0s + 1
                    if x > maxim:
                        maxim = x
                countextra = countextra + 1
        if countapp != 0:
            for i in range(0, 255):
                if app[i] > maxapp:
                    val = i - 127
                    maxapp = app[i]
        else:
            val = 0

        sql_temperature_features = [str(measurement_id), np.mean(temperature_raw_measurements), np.var(temperature_raw_measurements), skew(temperature_raw_measurements), kurtosis(temperature_raw_measurements),
                                    maximum_gradient, val, maxapp]
        sql_temperature_features = [sql_temperature_features[0]] + list(map(float, sql_temperature_features[1:]))
        mycursor.execute(add_to_sql_command, sql_temperature_features)
        cnx.commit()


def create_temperature_sql_table(mycursor):
    string = "CREATE TABLE tempd6 (ID VARCHAR(255)"
    for i in range(0, 7):
        string = string + ",Feat" + str(i) + " Float(15,4)"
    string = string + ")"
    mycursor.execute(string)


def delete_table(table_name: str, mycursor, cnx):
    string = f"drop table {table_name}"
    mycursor.execute(string)
    cnx.commit()
    

def generate_add_sql_command(table_name: str, number_entries: int):
    add_to_sql_command = f"INSERT INTO {table_name} (ID"
    for i in range(0, number_entries):
        add_to_sql_command = add_to_sql_command + ",Feat" + str(i)
    add_to_sql_command = add_to_sql_command + ") VALUES (%s"
    for i in range(0, number_entries):
        add_to_sql_command = add_to_sql_command + ",'%s'"
    add_to_sql_command = add_to_sql_command + ")"
    return add_to_sql_command

def main():
    cnx = mysql.connector.connect(user='root', password='sqlAmonouaparola213',  # change back for the other computer
                                  host='127.0.0.1',
                                  database='final')
    # get data from temperature sensor
    # useless part on just getting IDs because i created a wrong table of features for temperature sensor

    mycursor = cnx.cursor()

    sql = "Select ID from an"
    mycursor.execute(sql)
    list_of_ids = list(mycursor.fetchall())

    try:
        create_temperature_sql_table(mycursor)
    except mysql.connector.errors.ProgrammingError:
        print("Table already created")
    add_to_sql_command = generate_add_sql_command('tempd6', 7)
    print("Extracting temperature features")
    extract_temperature_features(list_of_ids, mycursor,
                                 add_to_sql_command, cnx)
    print("Finished extracting and adding features to sql")


if __name__ == "__main__":
    main()
