import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load data into pandas object, and print first 5 rows
diabetes = pd.read_csv('pima-indians-diabetes.csv')
print(diabetes.head())

# Normalise the columns with float values
cols_to_norm = ['Number_pregnant', 'Glucose_concentration', 'Blood_pressure', 'Triceps', 'Insulin', 'BMI', 'Pedigree']
diabetes[cols_to_norm] = diabetes[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

# Map the columns in tensorflow
num_preg = tf.feature_column.numeric_column('Number_pregnant')
Gluc_conc = tf.feature_column.numeric_column('Glucose_concentration')
bld_press = tf.feature_column.numeric_column('Blood_pressure')
tricep = tf.feature_column.numeric_column('Triceps')
insulin = tf.feature_column.numeric_column('Insulin')
bmi = tf.feature_column.numeric_column('BMI')
pedigree = tf.feature_column.numeric_column('Pedigree')
age = tf.feature_column.numeric_column('Age')

# plot a histogram of the ages of the participents in the data
diabetes['Age'].hist(bins=20)
plt.show()

# Put age into buckets so its easier to work with
age_buckets = tf.feature_column.bucketized_column(age, boundaries=[20,30,40,50,60,80])
# Create a feature columns list
feat_cols = [num_preg, Gluc_conc, bld_press, tricep, insulin, bmi, pedigree, age_buckets]

# Drop the Class from the x_data as that is the answer
x_data = diabetes.drop('Class', axis=1)
labels = diabetes['Class']
X_train, X_test, y_train, y_test = train_test_split(x_data, labels, test_size=0.35, random_state=101)

# Create an input function
input_func = tf.compat.v1.estimator.inputs.pandas_input_fn(x=X_train, y=y_train, batch_size=10, num_epochs=1000, shuffle=True)

# Create a model to train
model = tf.estimator.LinearClassifier(feature_columns=feat_cols, n_classes=2)
model.train(input_fn=input_func, steps=500)

# Now can use model to predict chance of diabetes for 10 of the patients
pred_input_func = tf.compat.v1.estimator.inputs.pandas_input_fn( x=X_test, batch_size=10, num_epochs=1, shuffle=False)
predictions = model.predict(pred_input_func)
list(predictions)

# Compare the 10 peoples predicted results with their actual results
eval_input_func = tf.compat.v1.estimator.inputs.pandas_input_fn(x=X_test, y=y_test, batch_size=10, num_epochs=1, shuffle=False)
results = model.evaluate(eval_input_func)
print(results)