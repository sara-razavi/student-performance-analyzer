import pandas as pd


def load_students(filename):
    data = pd.read_csv(filename)
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

print()
print("Data types:")
print(students.dtypes)

# check if some values are missing
print()
print("Missing values:")
print(students.isnull().sum())
