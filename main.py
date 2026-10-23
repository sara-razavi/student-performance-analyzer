import pandas as pd
import numpy as np

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


def analyze_subjects(data):
    # one row for every subject
    result = pd.DataFrame()
    result["average"] = data[subjects].mean().round(2)
    result["highest"] = data[subjects].max()
    result["lowest"] = data[subjects].min()
    result["failed"] = (data[subjects] < 10).sum()
    return result


def filter_students(data, major=None, min_average=None, max_age=None):
    result = data
    if major != None:
        result = result[result["major"] == major]
    if min_average != None:
        result = result[result["average"] >= min_average]
    if max_age != None:
        result = result[result["age"] <= max_age]
    return result


def top_students(data, count=5):
    sorted_data = data.sort_values("average", ascending=False)
    return sorted_data.head(count)


def group_by_major(data):
    groups = data.groupby("major")
    result = groups[subjects + ["average"]].mean().round(2)
    result["students"] = groups["name"].count()
    return result


def get_statistics(data):
    stats = pd.DataFrame()
    for s in subjects + ["average"]:
        col = data[s]
        stats[s] = [col.mean(), col.median(), col.std(), col.var(),
                    np.percentile(col, 25), np.percentile(col, 75)]
    stats.index = ["mean", "median", "std", "variance", "25%", "75%"]
    return stats.round(2)


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
print("Subject analysis:")
subject_table = analyze_subjects(students)
print(subject_table)

best_subject = subject_table["average"].idxmax()
hardest_subject = subject_table["average"].idxmin()
print("Best subject:", best_subject)
print("Hardest subject:", hardest_subject)

# who is the best in each subject
print()
for s in subjects:
    top = students.loc[students[s].idxmax()]
    print("Top in", s + ":", top["name"], "(" + str(top[s]) + ")")

print()
print("Filtering")
cs_students = filter_students(students, major="CS")
print("CS students:", len(cs_students))

good_it = filter_students(students, major="IT", min_average=14)
print("IT students with average 14 or more:")
print(good_it[["name", "age", "average"]])

young = filter_students(students, max_age=21, min_average=15)
print("Students 21 or younger with average 15+:", list(young["name"]))

# students who failed at least one subject
failed_any = students[(students[subjects] < 10).any(axis=1)]
print("Failed at least one subject:", len(failed_any))
print(failed_any[["name"] + subjects])

print()
print("Top 5 students:")
print(top_students(students)[["name", "major", "average"]])

# sort by programming, best first
print()
print("Best in programming:")
by_prog = students.sort_values(["programming", "math"], ascending=False)
print(by_prog[["name", "programming", "math"]].head(5))

print()
print("Group analysis")
major_table = group_by_major(students)
print(major_table)

best_major = major_table["average"].idxmax()
print("Best major:", best_major, "with", major_table.loc[best_major, "average"])
print("Biggest major:", major_table["students"].idxmax())

# which major is best at each subject
print()
for s in subjects:
    print(s, "->", major_table[s].idxmax())

# compare ages
print()
print("Average by age:")
by_age = students.groupby("age")["average"].agg(["mean", "count"])
by_age["mean"] = by_age["mean"].round(2)
print(by_age)

# how many students failed in each major
failed_major = students[(students[subjects] < 10).any(axis=1)].groupby("major")["name"].count()
print()
print("Students who failed something, by major:")
print(failed_major)

print()
print("Basic statistics")
print(get_statistics(students))

# pandas describe for a quick check
print()
print(students[subjects].describe().round(2))

# correlation between subjects
print()
print("Correlation between subjects:")
corr = students[subjects].corr().round(2)
print(corr)

pairs = [("math", "programming"), ("math", "english"), ("programming", "english")]
best_pair = pairs[0]
for p in pairs:
    value = corr.loc[p[0], p[1]]
    print(p[0], "and", p[1], ":", value)
    if abs(value) > abs(corr.loc[best_pair[0], best_pair[1]]):
        best_pair = p
print("Strongest relation:", best_pair[0], "and", best_pair[1])

# same thing with numpy
r = np.corrcoef(students["math"], students["programming"])[0, 1]
print("numpy check (math, programming):", round(r, 2))

print("Age and average:", round(students["age"].corr(students["average"]), 2))

# how many students are far from the class average
avg = students["average"].mean()
std = students["average"].std()
far = students[abs(students["average"] - avg) > std]
print()
print("Students more than 1 std away from average:", list(far["name"]))

print()
print("Data types:")
print(students.dtypes)
