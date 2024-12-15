import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load Dataset
@st.cache
def load_data():
    file_path = '/mnt/data/synthetic_final_mapping (1).csv'
    return pd.read_csv(file_path)

# Preprocess Data
def preprocess_data(data):
    # Select relevant columns for the model
    relevant_columns = [
        "Role Status", "Region", "Project Type", "Track", "Location Shore", 
        "Primary Skill (Must have)", "Grade", "Competency", "Skills", "Employment ID"
    ]
    data = data[relevant_columns]
    le = LabelEncoder()
    for column in data.select_dtypes(include=['object']).columns:
        data[column] = le.fit_transform(data[column].fillna("Unknown"))
    return data

# Train Model
def train_model(data):
    X = data.drop("Employment ID", axis=1)
    y = data["Employment ID"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return model, accuracy

# Recommend Employees
def recommend_employees(model, input_data, data):
    predictions = model.predict_proba([input_data])[0]
    employee_indices = predictions.argsort()[-3:][::-1]
    employee_ids = data["Employment ID"].unique()
    top_employees = [employee_ids[i] for i in employee_indices]
    return top_employees

# Streamlit App
st.title("Employee Recommendation System")

data = load_data()
st.write("Dataset Preview:", data.head())

if st.button("Train Model"):
    st.write("Preprocessing data...")
    processed_data = preprocess_data(data)

    st.write("Training the model...")
    model, accuracy = train_model(processed_data)
    st.success(f"Model trained with an accuracy of {accuracy * 100:.2f}%")

    st.subheader("Enter Demand Attributes")
    user_input = []
    for column in processed_data.columns.drop("Employment ID"):
        value = st.text_input(f"{column}:")
        user_input.append(value)

    if st.button("Get Recommendations"):
        user_input = [int(val) for val in user_input]
        recommendations = recommend_employees(model, user_input, processed_data)
        st.subheader("Top 3 Recommended Employees:")
        for i, employee in enumerate(recommendations, 1):
            st.write(f"{i}. Employee ID: {employee}")
