#Name: Shlok Gopalbhai Gondalia
#Email: shlok@rams.colostate.edu
#Date: Saturday 18, April

import sys
import pandas as pd
from pandas import DataFrame
import csv
import sqlite3

#Error Check for incorrect input
if len(sys.argv) != 5:
    print("Incorrect Input. The correct input is script.py filename PGN SA columns")
    sys.exit()

#Reading Command line Arguments
file_Name = sys.argv[1]
PGN = sys.argv[2]
SA = sys.argv[3]
columns = sys.argv[4]

#Declaring and populating the Bytes columns list
columns_list = []

for i in range (0, len(columns), 2):
    columns_list.append(int(columns[i+1:i+2]))

#Reading the Input CSV File and converting it to SQL Table for better reading
db=sqlite3.connect("preprocess.sqlite")
cursor = db.cursor()
cursor.execute("DROP TABLE IF EXISTS preprocess_sq")
entire_data = pd.read_csv(file_Name).to_sql('preprocess_sq', con=db);

data = pd.read_sql(sql="select * from preprocess_sq where PGN like '" + PGN + "' and SA like '" + SA + "'", con=db)

#Make a list of the attributes which goes into first row
output_first_row = ["id", "time"]

for i in columns_list:
    for j in range (0,8):
        output_first_row.append("b" + str(i) + str(j))

#Open a csv file for writing
output_file_name = file_Name[:-4] + '_' + PGN + '_' + SA + '_' + columns + '.csv'
output_file = open(output_file_name, 'w', newline = '')
writer = csv.writer(output_file)

#Writes row to the CSV file
def writeCSV(row_to_write):
    writer.writerow(row_to_write)

writeCSV(output_first_row)

#Convert Hex from Bytes Column to Binary
def convert(columns_list, row):
    id = data["index"][row]
    time = data["Rel. Time"][row]
    
    output_row = [id, time]

    bytes = data["Bytes"][row]
    separated_bytes = []

    for i in columns_list:
        index = 2*i
        separated_bytes.append(bytes[index:index+2])

    for i in separated_bytes:
        bits = format(int(i, 16), "08b")
        for j in range (0,8):
            bit = bits[j:j+1]
            output_row.append(bit)
    
    writeCSV(output_row)

#Runs the whole script
def run():
    cursor.execute("SELECT COUNT(*) FROM preprocess_sq WHERE PGN LIKE '" + PGN + "' AND SA lIKE '" + SA + "'")
    counter = cursor.fetchone()[0]

    for i in range(0, counter):
        convert(columns_list, i)

run()

output_file.close()
db.commit()
db.close()