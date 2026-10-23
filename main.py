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


def show_title(text):
    print()
    print("=" * 45)
    print(text)
    print("=" * 45)


def get_statistics(data):
    stats = pd.DataFrame()
    for s in subjects + ["average"]:
        col = data[s]
        stats[s] = [col.mean(), col.median(), col.std(), col.var(),
                    np.percentile(col, 25), np.percentile(col, 75)]
    stats.index = ["mean", "median", "std", "variance", "25%", "75%"]
    return stats.round(2)


students = load_students("students.csv")
rows_before = len(students)

print("Student Performance Analyzer")
print("Loaded", rows_before, "rows from students.csv")

print()
print("Cleaning data...")
students = clean_students(students)
print("Rows after cleaning:", len(students))
students = add_averages(students)

# overview
show_title("Overview")
class_average = students["average"].mean()
best = students.loc[students["average"].idxmax()]
worst = students.loc[students["average"].idxmin()]
above = students[students["average"] > class_average]

print("Students:", len(students))
print("Class average:", round(class_average, 2))
print("Highest:", best["name"], "-", best["average"])
print("Lowest:", worst["name"], "-", worst["average"])
print("Above average:", len(above), "out of", len(students))

# subjects
show_title("Subjects")
subject_table = analyze_subjects(students)
print(subject_table)
print()
print("Best subject:", subject_table["average"].idxmax())
print("Hardest subject:", subject_table["average"].idxmin())

print()
for s in subjects:
    # more than one student can have the top grade
    top = students[students[s] == students[s].max()]
    print("Top in", s + ":", ", ".join(top["name"]), "(" + str(students[s].max()) + ")")

# filtering and sorting
show_title("Filtering and sorting")
cs_students = filter_students(students, major="CS")
print("CS students:", len(cs_students))

good_it = filter_students(students, major="IT", min_average=14)
print()
print("IT students with average 14 or more:")
print(good_it[["name", "age", "average"]].to_string(index=False))

young = filter_students(students, max_age=21, min_average=15)
print()
print("Students 21 or younger with average 15+:", ", ".join(young["name"]))

failed_any = students[(students[subjects] < 10).any(axis=1)]
print()
print("Failed at least one subject:", len(failed_any))
print(failed_any[["name"] + subjects].to_string(index=False))

print()
print("Top 5 students:")
print(top_students(students)[["name", "major", "average"]].to_string(index=False))

print()
print("Best in programming:")
by_prog = students.sort_values(["programming", "math"], ascending=False)
print(by_prog[["name", "programming", "math"]].head(5).to_string(index=False))

# groups
show_title("Groups")
major_table = group_by_major(students)
print(major_table)

best_major = major_table["average"].idxmax()
print()
print("Best major:", best_major, "with", major_table.loc[best_major, "average"])
print("Biggest major:", major_table["students"].idxmax())

print()
for s in subjects:
    print("Best major in", s + ":", major_table[s].idxmax())

print()
print("Average by age:")
by_age = students.groupby("age")["average"].agg(["mean", "count"])
by_age["mean"] = by_age["mean"].round(2)
print(by_age)

# statistics
show_title("Statistics")
print(get_statistics(students))

print()
print("Correlation between subjects:")
corr = students[subjects].corr().round(2)
print(corr)

pairs = [("math", "programming"), ("math", "english"), ("programming", "english")]
best_pair = pairs[0]
for p in pairs:
    value = corr.loc[p[0], p[1]]
    if abs(value) > abs(corr.loc[best_pair[0], best_pair[1]]):
        best_pair = p
print()
print("Strongest relation:", best_pair[0], "and", best_pair[1])

# same thing with numpy
r = np.corrcoef(students["math"], students["programming"])[0, 1]
print("numpy check (math, programming):", round(r, 2))
print("Age and average:", round(students["age"].corr(students["average"]), 2))

avg = students["average"].mean()
std = students["average"].std()
far = students[abs(students["average"] - avg) > std]
print()
print("Students more than 1 std away from the class average:")
print(", ".join(far["name"]))

# save the results
students.to_csv("students_clean.csv", index=False)

with open("report.txt", "w") as file:
    file.write("Student Performance Report\n")
    file.write("Students: " + str(len(students)) + "\n")
    file.write("Class average: " + str(round(class_average, 2)) + "\n")
    file.write("Highest: " + best["name"] + " (" + str(best["average"]) + ")\n")
    file.write("Lowest: " + worst["name"] + " (" + str(worst["average"]) + ")\n")
    file.write("Best subject: " + subject_table["average"].idxmax() + "\n")
    file.write("Hardest subject: " + subject_table["average"].idxmin() + "\n")
    file.write("Best major: " + best_major + "\n")
    file.write("Failed at least one subject: " + str(len(failed_any)) + "\n")

print()
print("Saved students_clean.csv and report.txt")
