import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load data safely
data = pd.read_csv(os.path.join(os.path.dirname(__file__), "data.csv"))
data.columns = data.columns.str.strip()

# Features & target
X = data[['cgpa', 'study_hours', 'internship']]
y = data['placed']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Proper prediction (NO warning)
sample = pd.DataFrame([[8.2, 5, 1]], columns=['cgpa', 'study_hours', 'internship'])
prediction = model.predict(sample)

print("Prediction for sample student:", prediction)