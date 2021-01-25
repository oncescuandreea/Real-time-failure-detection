import csv


def generate_add_sql_command(table_name: str, number_entries: int):
    add_to_sql_command = f"INSERT INTO {table_name} (ID"
    for i in range(0, number_entries):
        add_to_sql_command = add_to_sql_command + ",Feat" + str(i)
    add_to_sql_command = add_to_sql_command + ") VALUES (%s"
    for i in range(0, number_entries):
        add_to_sql_command = add_to_sql_command + ",'%s'"
    add_to_sql_command = add_to_sql_command + ")"
    return add_to_sql_command


def get_id_data(mycursor, measurement_id):
    sql_get_data_name = "Select nameData from corr where ID='" + str(measurement_id) + "'"
    mycursor.execute(sql_get_data_name)
    data_file_name = mycursor.fetchall()
    data = csv.reader(open('C:/Users/oncescu/OneDrive - Nexus365/Data/' + data_file_name[0][0], 'rt'),
                      delimiter=",", quotechar='|')
    return data


def create_sql_table(mycursor, table_name, number_features, no_decimals):
    string = f"CREATE TABLE {table_name} (ID VARCHAR(255)"
    for i in range(0, number_features):
        string = string + ",Feat" + str(i) + f" Float(15,{no_decimals})"
    string = string + ")"
    mycursor.execute(string)