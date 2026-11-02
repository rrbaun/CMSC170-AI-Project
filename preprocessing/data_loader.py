import csv

def load_data(filepath):
    rows = []

    with open(filepath, "r", encoding="latin-1", newline ="") as file:
        next(file)  # skips the tile and blank lines
        next(file)

        reader = csv.DictReader(file, delimiter=";")    # stores each record as a directory
        for row in reader:
            rows.append(row)
    return rows