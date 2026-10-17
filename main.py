import pandas as pd

subjects = ["math", "programming", "english"]


def load_students(filename):
    data = pd.read_csv(filename)
    return data


def clean_students(data):
    data = data.copy()

    # remove extra spaces
    data["name"] = data["name"].str.strip()
    data["major"] = data["major"].str.strip()

    # majors are written in different ways (cs, CS ...)
    data["major"] = data["major"].str.lower()
    data["major"] = data["major"].replace({"cs": "CS", "it": "IT"})
    data["major"] = data["major"].replace({"math": "Math", "physics": "Physics", "electrical": "Electrical"})

    # same student twice
    data = data.drop_duplicates()

    # grades must be between 0 and 20, the rest is a mistake
    for s in subjects:
        bad = (data[s] < 0) | (data[s] > 20)
        print("Wrong", s, "grades:", bad.sum())
        data.loc[bad, s] = None

    # fill missing grades with the average of that subject
    for s in subjects:
        avg = round(data[s].mean(), 1)
        data[s] = data[s].fillna(avg)

    # age: use the middle value
    data["age"] = data["age"].fillna(data["age"].median())
    data["age"] = data["age"].astype(int)

    data = data.reset_index(drop=True)
    return data


def add_averages(data):
    # average of the 3 subjects for every student
    data["average"] = data[subjects].mean(axis=1).round(2)
    return data


students = load_students("students.csv")

print("Student Performance Analyzer")
print("----------------------------")
print("Rows:", students.shape[0])
print("Columns:", students.shape[1])
print("Column names:", list(students.columns))

print()
print("First 5 students:")
print(students.head())

# check if some values are missing
print()
print("Missing values:")
print(students.isnull().sum())

print()
print("Cleaning data...")
rows_before = len(students)
students = clean_students(students)
print("Rows before:", rows_before)
print("Rows after:", len(students))

print()
print("Missing values after cleaning:")
print(students.isnull().sum().sum())
print("Majors:", list(students["major"].unique()))

students = add_averages(students)
print()
print(students[["name", "major", "average"]].head(10))

class_average = students["average"].mean()
print()
print("Class average:", round(class_average, 2))

# best and worst student
best = students.loc[students["average"].idxmax()]
worst = students.loc[students["average"].idxmin()]
print("Highest:", best["name"], best["average"])
print("Lowest:", worst["name"], worst["average"])

# students above the class average
above = students[students["average"] > class_average]
print("Above average:", len(above), "out of", len(students))

print()
print("Data types:")
print(students.dtypes)
