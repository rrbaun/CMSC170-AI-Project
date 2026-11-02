import numpy as np
from pathlib import Path
from preprocessing import (
    load_data,
    clean_data,
    stratified_split
)

class KNearestNeighbor:
    def __init__(self, k=5):
        self.k = k             
        self.X_train = None     
        self.y_train = None    

    def fit(self, X_train, y_train):
        if self.k > len(X_train):
            raise ValueError("K cannot exceed the training sample count.")
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        # classify each sample in the test set
        predictions = []
        for sample in X_test:
            prediction = self._predict_one(sample)
            predictions.append(prediction)
        return np.array(predictions)

    def _predict_one(self, sample):
        # compute distance from this sample to every training point
        distances = []
        for i in range(len(self.X_train)):
            distance = self._euclidean_distance(sample, self.X_train[i])
            distances.append((distance, self.y_train[i]))

        # sort by ascending distance and keep the k closest neighbors
        distances.sort(key=lambda item: item[0])
        nearest_neighbors = distances[:self.k]

        # extract just the class labels of the k neighbors
        neighbor_labels = [label for distance, label in nearest_neighbors]

        # return the label that appears most among the neighbors (majority vote)
        labels, counts = np.unique(neighbor_labels, return_counts=True)
        return labels[np.argmax(counts)]

    def _euclidean_distance(self, point1, point2):
        return np.sqrt(np.sum((point1 - point2) ** 2))

    def macro_f1_score(self, y_true, y_pred):
        labels = ["High", "Low"]
        f1_scores = []

        for label in labels:
            # true positives: correctly predicted as this label
            true_positive = np.sum(
                (y_true == label) & (y_pred == label)
            )

            # false positives: wrongly predicted as this label
            false_positive = np.sum(
                (y_true != label) & (y_pred == label)
            )

            # false negatives: this label missed by the classifier
            false_negative = np.sum(
                (y_true == label) & (y_pred != label)
            )

            # precision: of all predicted positives, how many are correct
            precision = (
                true_positive / (true_positive + false_positive)
                if (true_positive + false_positive) > 0
                else 0
            )

            # recall: of all actual positives, how many were found
            recall = (
                true_positive / (true_positive + false_negative)
                if (true_positive + false_negative) > 0
                else 0
            )

            # F1: harmonic mean of precision and recall
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0
            )
            f1_scores.append(f1)

        return np.mean(f1_scores)

    def calculate_confusion_matrix(self, y_true, y_pred):
        labels = ["High", "Low"]
        matrix = np.zeros((2, 2), dtype=int)

        for actual, predicted in zip(y_true, y_pred):
            actual_index = labels.index(actual)
            predicted_index = labels.index(predicted)
            matrix[actual_index][predicted_index] += 1
        return matrix

    def display_confusion_matrix(self, matrix):
        print("\nConfusion Matrix")
        print("Rows: Actual | Columns: Predicted")
        print("                 High    Low")
        print(f"Actual High      {matrix[0][0]:<7} {matrix[0][1]}")
        print(f"Actual Low       {matrix[1][0]:<7} {matrix[1][1]}")


def prepare_supervised_dataset(cleaned_rows):
    # select only age-sex columns (ending in " Male" or " Female") as features
    feature_columns = [
        column
        for column in cleaned_rows[0].keys()
        if column.endswith(" Male") or column.endswith(" Female")
    ]

    # build the feature matrix X from selected columns
    X = np.array(
        [
            [row[column] for column in feature_columns]
            for row in cleaned_rows
        ],
        dtype=float
    )

    # compute total mortality per row and use median as threshold
    totals = X.sum(axis=1)
    threshold = float(np.median(totals))

    # label each row: "High" if total mortality >= median, else "Low"
    y = np.array(
        ["High" if total >= threshold else "Low" for total in totals]
    )

    # keep region and cause of death as metadata (not used as features)
    metadata = [
        {"Region": row["Region"], "Cause of Death": row["Cause of Death"]}
        for row in cleaned_rows
    ]
    return X, y, metadata, feature_columns, threshold


def run_knn(X, y, feature_columns, metadata):
    """
    Handles user interaction, execution, evaluation, and reporting for KNN.
    """
    print("\n" + "="*40)
    print("RUNNING METHOD: K-NEAREST NEIGHBOR")
    print("="*40)

    # 1. Ask user for k
    k_input = input("Enter the number of neighbors, k: ")
    try:
        k = int(k_input)
    except ValueError:
        k = 5
        print(f"Invalid input. Defaulting to k={k}")

    # 2. Split dataset using stratified 80:20 ratio
    print("\nSplitting dataset using 80:20 ratio")
    X_train, X_test, y_train, y_test = stratified_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"Training observations: {len(X_train)}")
    print(f"Testing observations: {len(X_test)}")

    # 3. Run KNN
    knn = KNearestNeighbor(k=k)
    knn.fit(X_train, y_train)
    print(f"\nRunning KNN with K = {k}...")
    y_pred = knn.predict(X_test)

    # 4. Compute evaluation metrics
    score = knn.macro_f1_score(y_test, y_pred)
    matrix = knn.calculate_confusion_matrix(y_test, y_pred)

    print("\nKNN Result-----------------")
    print(f"Macro F1-Score: {score:.4f}")
    knn.display_confusion_matrix(matrix)


def main():
    file_path = (
        Path(__file__).parent.parent
        / "data"
        / "CMSC170-Data.csv"
    )
    print("\nLoading dataset...")
    raw_data = load_data(file_path)

    print(f"Original observations: {len(raw_data)}")
    cleaned_data = clean_data(raw_data)     # remove invalid / total rows

    print(f"Cleaned observations: {len(cleaned_data)}")
    print("\nPreparing dataset for KNN...")

    # build feature matrix, labels, and threshold
    X, y, metadata, feature_names, threshold = (
        prepare_supervised_dataset(cleaned_data)
    )

    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

    print(f"Number of observations: {len(X)}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Median mortality threshold: {threshold}")

    print("\nClass distribution:")
    for label in ["High", "Low"]:
        count = np.sum(y == label)
        print(f"{label} Mortality Impact: {count}")

    run_knn(X, y, feature_names, metadata)

    print("\nKNN Analysis complete")

if __name__ == "__main__":
    main()