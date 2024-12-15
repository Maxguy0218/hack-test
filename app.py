import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load Dataset
@st.cache_resource
def load_and_train_model():
    file_path = 'synthetic_final_mapping (1).csv'
    data = pd.read_csv(file_path)

    # Select relevant columns for the model
    relevant_columns = [
        "Role Status", "Region", "Project Type", "Track", "Location Shore", 
        "Primary Skill (Must have)", "Grade", "Employment ID"
    ]
    data = data[relevant_columns]

    # Preprocess data
    label_encoders = {}
    for column in data.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column].fillna("Unknown"))
        label_encoders[column] = le

    # Train the model
    X = data.drop("Employment ID", axis=1)
    y = data["Employment ID"]
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    return model, data, label_encoders

# Recommend Employees
def recommend_employees(model, input_data, data):
    predictions = model.predict_proba([input_data])[0]
    employee_indices = predictions.argsort()[-3:][::-1]
    employee_ids = data["Employment ID"].unique()
    top_employees = [employee_ids[i] for i in employee_indices]
    return top_employees

# Streamlit App
st.title("Employee Recommendation System")

# Load and train model
st.write("Initializing model...")
model, data, label_encoders = load_and_train_model()
st.success("Model is ready!")

# Collect Demand Attributes
st.subheader("Enter Demand Attributes")
user_input = []
for column in data.columns.drop("Employment ID"):
    if column in label_encoders:
        options = label_encoders[column].classes_
        value = st.selectbox(f"{column}:", options)
        user_input.append(label_encoders[column].transform([value])[0])
    else:
        value = st.number_input(f"{column}:", min_value=0, step=1)
        user_input.append(value)

if st.button("Get Recommendations"):
    try:
        recommendations = recommend_employees(model, user_input, data)
        st.subheader("Top 3 Recommended Employees:")
        for i, employee in enumerate(recommendations, 1):
            st.write(f"{i}. Employee ID: {employee}")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
