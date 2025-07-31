import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

# Load dataset
df = pd.read_csv("Salary_Data.csv")

# Prepare data
X = df[['YearsExperience']]
y = df['Salary']

# Train models
lr_model = LinearRegression()
dt_model = DecisionTreeRegressor(random_state=42)
lr_model.fit(X, y)
dt_model.fit(X, y)

# Streamlit UI
st.title("📊 Salary Prediction App")
st.write("Predict salary based on years of experience using ML models.")

# Input from user
exp_input = st.number_input("Enter Years of Experience:", min_value=0.0, max_value=50.0, step=0.1)

# Predict
if st.button("Predict Salary"):
    exp_val = np.array([[exp_input]])
    lr_pred = lr_model.predict(exp_val)[0]
    dt_pred = dt_model.predict(exp_val)[0]
    
    st.success(f"🔹 Linear Regression Prediction: ₹{lr_pred:,.2f}")
    st.success(f"🔸 Decision Tree Prediction: ₹{dt_pred:,.2f}")

    # Plot
    fig, ax = plt.subplots()
    ax.scatter(X, y, label="Data", color="blue")
    ax.plot(X, lr_model.predict(X), color="red", label="Linear Regression")
    ax.set_xlabel("Years of Experience")
    ax.set_ylabel("Salary")
    ax.set_title("Linear Regression Fit")
    ax.legend()
    st.pyplot(fig)
