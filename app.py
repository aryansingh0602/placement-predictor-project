import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
import os

data = pd.read_csv(os.path.join(os.path.dirname(__file__), "data.csv"))
data.columns = data.columns.str.strip()

X = data[['cgpa', 'study_hours', 'internship']]
y = data['placed']

model = LogisticRegression()
model.fit(X, y)

st.title("Placement Predictor")

cgpa = st.number_input("CGPA")
study_hours = st.number_input("Study Hours")
internship = st.selectbox("Internship", [0, 1])

if st.button("Predict"):
    input_data = pd.DataFrame([[cgpa, study_hours, internship]], columns=['cgpa', 'study_hours', 'internship'])
    result = model.predict(input_data)
    
    if result[0] == 1:
        st.success("Placed")
    else:
        st.error("Not Placed")