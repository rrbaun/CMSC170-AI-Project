import os
from preprocessing.data_loader import load_data
from preprocessing.data_preprocessing import clean_data, prep_features
from algorithms.k_means import run_kmeans 

file_path = "data/CMSC170-Data.csv"

if os.path.exists(file_path):
    print("CSV file found!")
    rows = load_data(file_path)
    # print("Total records: ", len(rows))

    # print("Columns:")  # display column names
    # print("Total columns:", len(rows[0]))
    # print(rows[0].keys())

    # print("\nFirst record:") # display first record - can be removed
    # print(rows[0])

    cleaned_rows = clean_data(rows)
    X, regions, causes, feature_columns = prep_features(cleaned_rows)
    
    # print("\nAll feature columns:", feature_columns) 

    run_kmeans(X, feature_columns, regions, causes)
else:
    print("CSV file not found!")