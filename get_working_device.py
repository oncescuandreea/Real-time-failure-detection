# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 23:46:57 2020
File used to generate an sql table containing random sets of
working conditions data. It does so by randomly picking failure
datasets and taking the numbers correspodning to features of working
sensors and then concatenating them.
@author: Andreea
"""

import mysql.connector
import random

cnx = mysql.connector.connect(user='root', password='Amonouaparola213',
                          host='127.0.0.1',
                          database='final')
mycursor = cnx.cursor()
sqldataID = "select * from an"
mycursor.execute(sqldataID)
results = mycursor.fetchall()

dict_sensors={}
dict_sensors['GSR'] = 'gsr2'
dict_sensors['Acc'] = 'FEAT2'
dict_sensors['Hum'] = 'hum3'
dict_sensors['Temp'] = 'tempd5'



add ="INSERT INTO FEAT_working (Counter"
for i in range(0, 73):
    add=add+",Feat"+str(i)
add=add+") VALUES (%s"
for i in range(0, 73):
    add=add+",'%s'"
add=add+")"
count = 1
while count <= 117:
    features = []
    ID_list = []
    features.append(count)
    for sensor in ['GSR', 'Acc', 'Hum', 'Temp']:
        found = False
        while found == False:
            random_choice_ID = random.randint(0, 116)
            ID = results[random_choice_ID][0]
            if sensor == 'GSR':
                if results[random_choice_ID][3] == 'working':
                    found = True
            elif sensor == 'Acc':
                if results[random_choice_ID][1] == 'working':
                    found = True
            elif sensor == 'Hum':
                if results[random_choice_ID][4] == 'working':
                    found = True
            else:
                if results[random_choice_ID][2] == 'working':
                    found = True
        ID_list.append(ID)
        sql = f"select * from {dict_sensors[sensor]} where ID='" + ID + "'"
        mycursor.execute(sql)
        data = mycursor.fetchall()
        for el in data[0][1:]:
            features.append(el)
                    
    # =============================================================================
    # string ="CREATE TABLE FEAT_working (Counter INT"
    # for i in range(0, 73):
    #     string=string + ",Feat" + str(i) + " Float(15,10)"
    # string=string + ")"
    # mycursor.execute(string)
    # =============================================================================
    mycursor = cnx.cursor()
    mycursor.execute(add, features)
    cnx.commit()
    count += 1