import os
from preprocessing.data_loader import load_data

file_path = "data/CMSC170-Data.csv"

if os.path.exists(file_path):
    print("CSV file found!")
    rows = load_data(file_path)
    print("Total records: ", len(rows))

    print("Columns:")  # display column names
    print("Total columns:", len(rows[0]))
    print(rows[0].keys())

    print("\nFirst record:") # display first record - can be removed
    print(rows[0])

else:
    print("CSV file not found!")