#Name: Shlok Gopalbhai Gondalia
#Email: shlok@rams.colostate.edu
#Date: Friday 24, April

import sys
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import csv
import sqlite3

#Hard Code Value
resolution =0.00390625#0.125#(1/256)#0.03125
offset=0#-273
unit="MPa"#"oil temprature (C)"

#Error Check for incorrect input
if len(sys.argv) != 4:
    print("Incorrect Input. The correct input is script.py filename Columns Value")
    sys.exit()

#Reading Command line Arguments
file_Name = sys.argv[1]
columns = sys.argv[2]
value = sys.argv[3]

#Graph Title
graph_title = input("Give the Title for the Plot:\n")

#Method to sort the reverse list of columns
def column_sort(input_list):
    def letter_sort(s):
        return s[1]
    return sorted(input_list, key=letter_sort, reverse=True)

#Grabing all the columns from the input CSV File
csv_column_list = (pd.read_csv(file_Name, nrows=0))
csv_column_list = list(csv_column_list.columns)
csv_column_list = csv_column_list[2:]
"""csv_column_list = column_sort(csv_column_list)"""

#Making a list of columns to denoise
columns_list__to_modify = []

"""for i in range (0, len(columns), 3):
    columns_list__to_modify.append(columns[i:i+3])"""

columns_index__to_modify = columns.split(",")

for i in range(0, len(csv_column_list)):
    if i in columns_index__to_modify:
        columns_list__to_modify.append(csv_column_list[i])

#Sort the csv_column_list in proper order (e.g. B3B4 to B4B3)
csv_column_list = column_sort(csv_column_list)

#Reading the Input CSV File and converting it to SQL Table for better reading
db=sqlite3.connect("evaluate.sqlite")
cursor = db.cursor()
cursor.execute("DROP TABLE IF EXISTS evaluate_sq")
entire_data = pd.read_csv(file_Name).to_sql('evaluate_sq', con=db);

data = pd.read_sql(sql="select * from evaluate_sq", con=db)

#Open a csv file for writing
output_file_name = file_Name[:-4] + '_' + columns + '_' + value + '.csv'
output_file = open(output_file_name, 'w', newline = '')
writer = csv.writer(output_file)

#Writes row to the CSV file
def writeCSV(row_to_write):
    writer.writerow(row_to_write)

output_first_row = ["id", "time", "Actual_value", "Denoised_value"]
writeCSV(output_first_row)

#Lists For Plot
time_list = []
actual_value_list = []      #avl
denoised_value_list = []    #dvl
avl_dvl_list = []

#Convert one row from input to one row of output
def convert(index):
    id = data["id"][index]
    time = data["time"][index]

    time_list.append(time)

    output_row = [id, time]
 
    actual_bits_string = ""
    denoised_bits_string = ""
    for col in csv_column_list:
        if(col in columns_list__to_modify):
            actual_bits_string = actual_bits_string + str(data[col][index])
            denoised_bits_string = denoised_bits_string + str(value)
            continue

        actual_bits_string = actual_bits_string + str(data[col][index])
        denoised_bits_string = denoised_bits_string + str(data[col][index])
    
    actual_value = int(actual_bits_string,2) * resolution+offset
    denoised_value = int(denoised_bits_string,2) * resolution+offset

    actual_value_list.append(actual_value)
    denoised_value_list.append(denoised_value)
    avl_dvl_list.append(abs(actual_value-denoised_value))

    output_row.append(actual_value)
    output_row.append(denoised_value)
    return output_row

def plot():
    mean_error = sum(avl_dvl_list)/len(avl_dvl_list)

    fig, ax = plt.subplots(figsize=(20,12))
    plt.xlabel("Time", fontsize=30)
    plt.ylabel(unit, fontsize=30)

    plt.title(graph_title, fontsize=30)

    plt.plot(time_list, actual_value_list)
    plt.plot(time_list, denoised_value_list)
    plt.grid(True)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    
    textstr = "Mean Error: " + str(round(mean_error,2))
    ax.text(0.02, 0.98, textstr, fontsize=30, verticalalignment='top', horizontalalignment='left', transform=ax.transAxes)

    plt.plot([], c='#5b9bd5', label='Actual Value')
    plt.plot([], c='#ed7c31', label='Denoised value')
    plt.legend()

    plt.legend(loc="upper center", prop={"size":30}, bbox_to_anchor=(0.5, -0.1), ncol=2)
    plt.tight_layout()
    fig.savefig(graph_title + ".png", dpi=fig.dpi)

def run():
    cursor.execute("SELECT COUNT(*) FROM evaluate_sq")
    counter = cursor.fetchone()[0]

    for i in range(0, counter):
        row = convert(i)
        writeCSV(row)

run()
plot()
output_file.close()
db.commit()
db.close()
