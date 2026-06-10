import pandas as pd
import numpy as np
import kagglehub
import os

# Download latest version
path = kagglehub.dataset_download("spscientist/students-performance-in-exams")
df = pd.read_csv(os.path.join(path,"StudentsPerformance.csv"))
print("available columns:", df.columns.tolist())

column = 'math score'
data = df[column].dropna().tolist()

mean = np.mean(data)
median = np.median(data)
mode_val = pd.Series(data).mode().tolist()  # mode can have multiple values
variance = np.var(data, ddof=0)  # ddof=0 for population variance
std_dev = np.std(data, ddof=0)

print(f"\nAnalysis for column: {column}")
print("mean:", mean)
print("median:", median)
print("mode:", mode_val)
print("variance:", variance)
print("standard deviation:", std_dev)