# Import pandas library
import pandas as pd

# Load data into pandas object, and print first 5 rows
diabetes = pd.read_csv('pima-indians-diabetes.csv')
print(diabetes.head())

# Normalise the columns with float values
cols_to_norm = ['Number_pregnant', 'Glucose_concentration', 'Blood_pressure', 'Triceps', 'Insulin', 'BMI', 'Pedigree']
