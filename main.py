import os
from preprocessing.data_loader import load_data
from preprocessing.data_preprocessing import clean_data, prep_features
from algorithms.k_means import run_kmeans
from algorithms.k_nearest_neighbor import run_knn, prepare_supervised_dataset

file_path = "data/CMSC170-Data.csv"

if os.path.exists(file_path):
    print("CSV file found!")
    rows = load_data(file_path)

    cleaned_rows = clean_data(rows)

    print("\nSelect an algorithm to run:")
    print("[1] K-Means Clustering")
    print("[2] K-Nearest Neighbor")
    choice = input("Enter choice (1 or 2): ").strip()

    if choice == "1":
        X, regions, causes, feature_columns = prep_features(cleaned_rows)
        run_kmeans(X, feature_columns, regions, causes)

    elif choice == "2":
        X, y, metadata, feature_columns, threshold = prepare_supervised_dataset(cleaned_rows)
        print(f"\nOriginal observations: {len(rows)}")
        print(f"Cleaned observations: {len(cleaned_rows)}")
        print(f"Number of observations: {len(X)}")
        print(f"Number of features: {X.shape[1]}")
        print(f"Median mortality threshold: {threshold}")
        print("\nClass distribution:")
        for label in ["High", "Low"]:
            import numpy as np
            count = int((y == label).sum())
            print(f"{label} Mortality Impact: {count}")
        run_knn(X, y, feature_columns, metadata)

    elif choice == "3":
        

    else:
        print("Invalid choice.")

else:
    print("CSV file not found!")