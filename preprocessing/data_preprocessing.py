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