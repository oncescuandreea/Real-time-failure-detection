# -*- coding: utf-8 -*-
# extracting accelerometer features

# testing fft and spectogram on actual processed data for just 50 points
# now processing the files and creating a signal
# rms added

import numpy as np
import mysql.connector
from scipy import signal
import argparse
from scipy import fftpack
from scipy.stats import kurtosis
from scipy.stats import skew
from operator import add
from pathlib import Path
from feature_extraction_utils import generate_add_sql_command, get_id_data, create_sql_table, delete_table


def takeFirst(elem: list):
    return elem[0]


def extract_accelerometer_features(list_of_ids: list,
                                   mycursor: mysql.connector.cursor,
                                   add_to_sql_command: str,
                                   cnx: mysql.connector,
                                   data_folder_location: Path):
    window = signal.hamming(1000, sym=False)  # a 1000 points window
    listTotal = []  # keep top 5 frequencies for x,y,z

    for one_id in list_of_ids:
        measurement_id = one_id[0]
        data = get_id_data(mycursor, measurement_id, data_folder_location)

        ##################################acc bits
        columnx = []  # containing windowed signal
        columny = []
        columnz = []

        accx = []  # will contain raw signal and 0s for spectogram
        accy = []
        accz = []

        count = 0  # counts the order of the recorded data in the set of 1000 for the accelerometer. always resets
        # when gets to 1000
        raw_data_counter = 3  # start from 3 to ignore the 3rd measurement from the set of 1000 which is always
        # delayed. To get continuous set of 1000 data poits, start from 3rd point
        noTest = 0  # number of test sets
        number_of_continuous_0_x = 0
        number_of_continuous_0_y = 0
        number_of_continuous_0_z = 0
        number_of_continuous_0_x_max = 0
        number_of_continuous_0_y_max = 0
        number_of_continuous_0_z_max = 0
        countercounter = 0

        # create a list for each direction of acceleration which has 2000 points of 0 padding to the left and to the right
        # next padding to the left
        for k in range(0, 2000):
            columnx.append(0)
            columny.append(0)
            columnz.append(0)
            accx.append(0)
            accy.append(0)
            accz.append(0)
        for row in data:
            # apply the window to the accelerometer data and append value to the lists
            if (raw_data_counter % 1000 > 2 or raw_data_counter % 1000 == 0 or raw_data_counter % 1000 == 1) and len(
                    row) >= 4:  # skip every 3rd measurement from each set of accelerometer data. To get continuity of 1000 points, just ignore first 3 measurements from the document
                columnx.append(float(row[1]) * window[count])
                columny.append(float(row[2]) * window[count])
                columnz.append(float(row[3]) * window[count])
                accx.append(float(row[1]))
                accy.append(float(row[2]))
                accz.append(float(row[3]))
                countercounter += 1
                if count != 0:
                    if valx == 0 and float(row[1]) == 0:
                        number_of_continuous_0_x += 1
                    else:
                        if number_of_continuous_0_x > number_of_continuous_0_x_max:
                            number_of_continuous_0_x_max = number_of_continuous_0_x
                        number_of_continuous_0_x = 0
                    if valy == 0 and float(row[2]) == 0:
                        number_of_continuous_0_y += 1
                    else:
                        if number_of_continuous_0_y > number_of_continuous_0_y_max:
                            number_of_continuous_0_y_max = number_of_continuous_0_y
                        number_of_continuous_0_y = 0
                    if valz == 0 and float(row[3]) == 0:
                        number_of_continuous_0_z += 1
                    else:
                        if number_of_continuous_0_z > number_of_continuous_0_z_max:
                            number_of_continuous_0_z_max = number_of_continuous_0_z
                        number_of_continuous_0_z = 0
                valx = float(row[1])
                valy = float(row[2])
                valz = float(row[3])
                count += 1  # increase count from 0 to 1000
            else:
                if raw_data_counter % 1000 == 2:
                    count = 0  # used to apply window to the next recorded signal, reset counter
                    noTest += 1  # increase number of datasets recorded
                    # this adds padding to the right
                    for k in range(0, 2000):
                        columnx.append(0)
                        columny.append(0)
                        columnz.append(0)
                        accx.append(0)
                        accy.append(0)
                        accz.append(0)
            raw_data_counter += 1
        noTest += 1  # counts number of documents, should be legth, come back and check
        a = 0
        delta = 4999  # length of each list columnx/y/z and ax/y/z

        # transform list into float array
        ax = np.array(accx)
        ay = np.array(accy)
        az = np.array(accz)
        ax = ax.astype(np.float64)
        ay = ay.astype(np.float64)
        az = az.astype(np.float64)

        listmean = []  # list calculating the average of the values in feature_list for all data sets recorded in that csv
        # document

        for i in range(1, noTest):
            b = a + delta  # end of interval corresponding to each dataset
            # extract from each windowed column vector the corresponding 4999 data points corresponding to each data
            # document and turn them into arrays
            cx = np.array(columnx[a:b])
            cy = np.array(columny[a:b])
            cz = np.array(columnz[a:b])

            # avoid errors by defining type of values
            cx = cx.astype(np.float64)
            cy = cy.astype(np.float64)
            cz = cz.astype(np.float64)

            # process the raw data stored in ax/y/z arrays
            ast = a + 2000  # set starting point
            afin = a + 2999  # set end point to avoid the zero padding
            # calculate skew
            skew_x = skew(ax[ast:afin])
            skew_y = skew(ay[ast:afin])
            skew_z = skew(az[ast:afin])

            # calculate kurtosis
            kur_x = kurtosis(ax[ast:afin])
            kur_y = kurtosis(ay[ast:afin])
            kur_z = kurtosis(az[ast:afin])

            # calculate mean
            mean_x = np.mean(ax[ast:afin])
            mean_y = np.mean(ay[ast:afin])
            mean_z = np.mean(az[ast:afin])

            # calculate variance
            var_x = np.var(ax[ast:afin])
            var_y = np.var(ay[ast:afin])
            var_z = np.var(az[ast:afin])

            # fft for the zero padded windowed set of data
            spx = fftpack.fft(cx)
            spy = fftpack.fft(cy)
            spz = fftpack.fft(cz)
            # take only abs values of amplitudes
            power_x = np.abs(spx)
            power_y = np.abs(spy)
            power_z = np.abs(spz)

            # create list from amplitudes and frequencies and sort them in descending order; pick top 5 values
            list_final_x = []
            list_final_y = []
            list_final_z = []
            feature_list = []  # contains all amplitudes and frequencies for x y and z

            # add first 4 moments to the list
            feature_list.append(mean_x)
            feature_list.append(mean_y)
            feature_list.append(mean_z)
            feature_list.append(var_x)
            feature_list.append(var_y)
            feature_list.append(var_z)
            feature_list.append(skew_x)
            feature_list.append(skew_y)
            feature_list.append(skew_z)
            feature_list.append(kur_x)
            feature_list.append(kur_y)
            feature_list.append(kur_z)
            # append length of longest 0 interval
            feature_list.append(number_of_continuous_0_x_max)
            feature_list.append(number_of_continuous_0_y_max)
            feature_list.append(number_of_continuous_0_z_max)

            lengthspin = int(len(spx) / 2) + 1  # take half od the values since rest are mirrored
            lentghspfin = len(spx)  # actual length of the list of values obtained after fft
            f1 = fftpack.fftfreq(
                len(spx)) * 333  # because sampling for actual data is s <=> 0.7Hz; f1 represents the x axis
            for j in range(0, lengthspin):
                if f1[j] > 1:
                    list_final_x.append([power_x[j], f1[j]])
                    list_final_y.append([power_y[j], f1[j]])
                    list_final_z.append([power_z[j], f1[j]])
            list_final_x.sort(reverse=True, key=takeFirst)
            list_final_y.sort(reverse=True, key=takeFirst)
            list_final_z.sort(reverse=True, key=takeFirst)
            for j in range(0, 5):
                feature_list.append(list_final_x[j][0])
                feature_list.append(list_final_x[j][1])
                feature_list.append(list_final_y[j][0])
                feature_list.append(list_final_y[j][1])
                feature_list.append(list_final_z[j][0])
                feature_list.append(list_final_z[j][1])

            rmsx = np.sqrt(np.sum(ax ** 2))
            rmsy = np.sqrt(np.sum(ay ** 2))
            rmsz = np.sqrt(np.sum(az ** 2))

            nomeasval = np.sqrt(noTest * 1000)  # number of measured elements in ax/y/z

            # finalise the rms calculation
            rmsx = rmsx / nomeasval
            rmsy = rmsy / nomeasval
            rmsz = rmsz / nomeasval

            # add to list rms values for x y and z
            feature_list.append(rmsx)
            feature_list.append(rmsy)
            feature_list.append(rmsz)

            if i == 1:
                listmean = feature_list  # initialise the mean value with the initial value of the list of fft features
            else:
                listmean = list(
                    map(add, listmean, feature_list))  # add new fft values for next signal to the previous ones
                # element by element
            a = a + 2999
        no_features = len(feature_list)  # find dimensions of number of features
        for k in range(0, no_features):
            listmean[k] = listmean[k] / noTest  # calculate the average of all features
        listmean = [float(feature) for feature in listmean]
        listmean.insert(0, measurement_id)
        listTotal.append(listmean)
        mycursor.execute(add_to_sql_command, listmean)
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
                                  database='final')
    mycursor = cnx.cursor()

    sql = "Select ID from an"
    mycursor.execute(sql)
    list_of_ids = list(mycursor.fetchall())
    no_features = 48
    no_decimals = 10
    table_name = 'FEAT3'
    try:
        create_sql_table(mycursor, table_name, no_features, no_decimals)
    except mysql.connector.errors.ProgrammingError:
        print("Table already created. Replacing values")
        delete_table(table_name, cnx)
        create_sql_table(mycursor, table_name, no_features, no_decimals)
    add_to_sql_command = generate_add_sql_command(table_name, no_features)
    print("Extracting accelerometer features")
    extract_accelerometer_features(list_of_ids, mycursor,
                                   add_to_sql_command, cnx,
                                   args.data_folder_location)
    print("Finished extracting and adding features to sql")


if __name__ == "__main__":
    main()
