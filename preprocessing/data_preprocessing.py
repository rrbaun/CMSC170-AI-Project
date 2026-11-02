import numpy as np 

def clean_data(rows):
    cleaned_rows = []

    for row in rows:
        region = row["Geolocation"].strip().lstrip(".").strip()
        cause = row["Cause of Death"].strip()

        if cause == "Total":
            continue
        cause = cause.lstrip(".").strip()

        cleaned_row = {
            "Region": region,
            "Cause of Death" : cause
        }

        for column, value in row.items():
            if column in ["Geolocation", "Cause of Death"]:
                continue

            if "Total" in column:
                continue
            
            if value:
                value = value.strip()
            else:
                value = ""

            if value == "":
                raise ValueError(f"Missing value in columns: {column}")
            try:
                number = int(value)
            except ValueError:
                raise ValueError(f"Invalid numerical value in {column}: {value}")

            if number < 0:
                raise ValueError(f"Negative mortality count in {column}: {number}")
            
            cleaned_row[column] = number
        cleaned_rows.append(cleaned_row)
    return cleaned_rows

def prep_features(cleaned_rows):
    feature_columns = [
        column
        for column in cleaned_rows[0].keys()
        if column.endswith(" Male") or column.endswith(" Female")
    ]
    X = np.array(
        [
            [row[column] for column in feature_columns]
            for row in cleaned_rows
        ],
        dtype = float 

    )
    regions = [row["Region"] for row in cleaned_rows]
    causes = [row["Cause of Death"] for row in cleaned_rows]

    return X, regions, causes, feature_columns

def stratified_split(X, y, test_size=0.20, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)

    classes, counts = np.unique(y, return_counts=True)
    train_indices = []
    test_indices  = []

    for cls in classes:
        cls_indices = np.where(y == cls)[0]
        np.random.shuffle(cls_indices)

        n_test = max(1, int(len(cls_indices) * test_size))
        test_indices.extend(cls_indices[:n_test].tolist())
        train_indices.extend(cls_indices[n_test:].tolist())

    train_indices = np.array(train_indices)
    test_indices  = np.array(test_indices)

    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]