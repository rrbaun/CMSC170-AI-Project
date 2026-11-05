import numpy as np
from preprocessing import stratified_split

class Perceptron:
    def __init__(self, learning_rate=0.1, max_epochs=100):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.weight = None
        self.bias = 0.0
        self.converged = False
        self.errors = []
        self.best_errors = None

    def fit(self, X_train, y_train):
        # each row of training data is [x1, ..., xn, d]
        # d is the desired output: +1 for "High", -1 for "Low"
        training_data = []
        for i in range(len(X_train)):
            d = 1 if y_train[i] == "High" else -1
            training_data.append(list(X_train[i]) + [d])

        # one weight per input, the last number of each row is d so it has no weight
        weight = [0.0 for item in training_data[0][:-1]]
        bias = 0.0
        best_weight = weight[:]
        best_bias = bias
        best_errors = len(training_data) + 1
        do_epoch = True
        epoch_count = 0
        self.errors = []

        while do_epoch:
            epoch_count += 1
            do_epoch, weight, bias = self._epoch_loop(training_data, weight, bias)

            # count the mistakes made by the updated weights
            errors = self._count_errors(training_data, weight, bias)
            self.errors.append(errors)

            # keeps best epoch for final results
            if errors < best_errors:
                best_weight = weight[:]
                best_bias = bias
                best_errors = errors

            # epoch limit reached, data is likely not linearly separable
            if epoch_count == self.max_epochs:
                break
        
        self.weight = best_weight
        self.bias = best_bias
        self.best_errors = best_errors
        self.converged = not do_epoch
        return self.weight, self.bias, epoch_count
    
    def _epoch_loop(self, training_data, weight, bias):
        updated = False
        for data in training_data:
            d = data[-1]
            y = self._compute_output(data, weight, bias)
            # update the weights and bias after every misclassified record
            if y != d:
                updated = True
                for j in range(len(weight)):
                    weight[j] = weight[j] + self.learning_rate * (d - y) * data[j]
                bias = bias + self.learning_rate * (d - y)

        return updated, weight, bias

    def _compute_output(self, data, weight, bias):
        v = bias * 1
        for j in range(len(weight)):
            v += data[j] * weight[j]

        if v >= 0:
            return 1
        return -1

    def _count_errors(self, training_data, weight, bias):
        errors = 0
        for data in training_data:
            if self._compute_output(data, weight, bias) != data[-1]:
                errors += 1
        return errors

    def predict(self, X_test):
        # classify each sample in the test set
        predictions = []
        for sample in X_test:
            prediction = self._predict_one(sample)
            predictions.append(prediction)
        return np.array(predictions)

    def _predict_one(self, sample):
        if self._compute_output(sample, self.weight, self.bias) == 1:
            return "High"
        return "Low"

    def accuracy_score(self, y_true, y_pred):
        return np.mean(y_true == y_pred)

    def precision_score(self, y_true, y_pred, label):
        true_positive = np.sum(
            (y_true == label) & (y_pred == label)
        )
        false_positive = np.sum(
            (y_true != label) & (y_pred == label)
        )

        # precision: of all predicted positives, how many are correct
        return (
            true_positive / (true_positive + false_positive)
            if (true_positive + false_positive) > 0
            else 0
        )

    def recall_score(self, y_true, y_pred, label):
        true_positive = np.sum(
            (y_true == label) & (y_pred == label)
        )
        false_negative = np.sum(
            (y_true == label) & (y_pred != label)
        )

        # recall: of all actual positives, how many were found
        return (
            true_positive / (true_positive + false_negative)
            if (true_positive + false_negative) > 0
            else 0
        )

    def macro_f1_score(self, y_true, y_pred):
        labels = ["High", "Low"]
        f1_scores = []

        for label in labels:
            precision = self.precision_score(y_true, y_pred, label)
            recall = self.recall_score(y_true, y_pred, label)

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

def prepare_ratio_features(X, y, feature_columns):
    # use ratios instead of raw count
    age_sex_indices = [
        i for i, column in enumerate(feature_columns)
        if not column.startswith("Not Stated")
    ]
    names = [feature_columns[i] for i in age_sex_indices]
    X_stated = X[:, age_sex_indices]
   
    # filters records with no deaths
    totals = X_stated.sum(axis=1)
    keep = totals > 0
   
    X_ratio = X_stated[keep] / totals[keep][:, None]
    return X_ratio, y[keep], keep, names

def run_perceptron(X, y, feature_columns):
    print("\n" + "="*40)
    print("RUNNING METHOD: PERCEPTRON")
    print("="*40)
    
    # 1. Ask user for the learning rate and the epoch limit
    rate_input = input("Enter the learning rate: ")
    try:
        learning_rate = float(rate_input)
    except ValueError:
        learning_rate = 0.1
        print(f"Invalid input. Defaulting to learning rate={learning_rate}")
    
    epoch_input = input("Enter the maximum number of epochs: ")
    try:
        max_epochs = int(epoch_input)
    except ValueError:
        max_epochs = 100
        print(f"Invalid input. Defaulting to max epochs={max_epochs}")

    # 2. Convert the counts to age/sex ratios
    X_ratio, y_ratio, keep, names = prepare_ratio_features(X, y, feature_columns)
    n_high = int(np.sum(y_ratio == "High"))
    n_low = len(y_ratio) - n_high
    print(f"\nDropped {int((~keep).sum())} records with no deaths")
    print(f"Number of observations: {len(X_ratio)}")
    print(f"High Mortality Impact: {n_high}")
    print(f"Low Mortality Impact: {n_low}")
    print(f"Majority-class accuracy: {max(n_high, n_low) / len(y_ratio):.4f}")
    
    # 3. Linear separability test on all records
    mean = X_ratio.mean(axis=0)
    std = X_ratio.std(axis=0)
    std[std == 0] = 1
    X_all = (X_ratio - mean) / std
    
    print(f"\nRunning Perceptron with learning rate = {learning_rate} across {len(X_all)} records...")
    full = Perceptron(learning_rate=learning_rate, max_epochs=max_epochs)
    weight, bias, epochs = full.fit(X_all, y_ratio)

    if full.converged:
        print(f"\n[Perceptron] Converged successfully at epoch {epochs}!")
        print("Data is linearly separable")
    else:
        print("\nData may not be linearly separable")
        # not relevant information ? can maybe remove
        print(f"Fewest misclassified records with one hyperplane: {full.best_errors} / {len(X_all)}")

    # 4. Split dataset using stratified 80:20 ratio
    print("\nSplitting dataset using 80:20 ratio")
    X_train, X_test, y_train, y_test = stratified_split(
        X_ratio, y_ratio, test_size=0.20, random_state=42
    )
    print(f"Training observations: {len(X_train)}")
    print(f"Testing observations: {len(X_test)}")
    
    # scale using the training data only, so the test data does not leak in
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    # 5. Run Perceptron
    perceptron = Perceptron(learning_rate=learning_rate, max_epochs=max_epochs)
    perceptron.fit(X_train, y_train)
    y_pred = perceptron.predict(X_test)

    # 6. Compute evaluation metrics
    train_accuracy = perceptron.accuracy_score(y_train, perceptron.predict(X_train))
    accuracy = perceptron.accuracy_score(y_test, y_pred)
    score = perceptron.macro_f1_score(y_test, y_pred)
    matrix = perceptron.calculate_confusion_matrix(y_test, y_pred)

    print("\nPerceptron Result-----------------")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Testing Accuracy: {accuracy:.4f}")
    print(f"Macro F1-Score: {score:.4f}")

    perceptron.display_confusion_matrix(matrix)
    print(f"\nFalse positives (Low-risk predicted as High): {matrix[1][0]}")
    print(f"False negatives (High-risk predicted as Low): {matrix[0][1]}")



