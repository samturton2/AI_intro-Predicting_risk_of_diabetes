# Simplilearn – Intro To AI

### What is it?
AI is a branch of computer science dedicated to creating intelligent machines that work and react like humans.

### Types of AI
-	Reactive machines – don’t have memories, or past experiences. Specific Jobs
-	Limited Memory – Use past experience and present data to make decisions 
	- _E.g. Self driving cars_
-	Theory of Mind – Can Socialise and understand human emotions
    - _yet to be built_
-	Self Awareness – Superintelligent, sentient and concious

### Achieving AI
-	Machine Learning
-	Deep Learning
	-	Input layer, hidden layer, Output layer
	-	Uses old data to predict new data

### Applications of AI
-	Google Home

## Predicting risk of diabetes
#### Prerequisites
- The code used to create the model, and predict other patients risk is in the main.py file.
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) was downloaded to make use of the [virtual environments](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf), and packages available, as we will be needing tensorflow, sklearn and more.
- Note that Tensorflow only works with python versions up to 3.8
- The ```tf.estimator.inputs``` function seen in main.py is depreciated and only works with versions of tensorflow before 2. I used the ```tf.compat.v1.estimator.inputs``` to get around this bug.

#### main.py
- First step was to import all the packages we will be needing in this programme.
```python
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
```
- Load data into pandas object, and print first 5 rows to double check the data was converted correctly
```py
diabetes = pd.read_csv('pima-indians-diabetes.csv')
print(diabetes.head())
```
