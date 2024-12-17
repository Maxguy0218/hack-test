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
st.title("Demand To Talent")

# Load and train model
model, data, label_encoders = load_and_train_model()

# Load the user-provided CSV file
uploaded_file = 'selected_demand.csv'
demand_data = pd.read_csv(uploaded_file)

# User selects ID from dropdown
st.subheader("Select Demand ID")
demand_id = st.selectbox("Demand ID:", demand_data['Demand ID'].unique())

# Auto-populate fields based on selected ID
selected_row = demand_data[demand_data['Demand ID'] == demand_id].iloc[0]
user_input = []

columns = data.columns.drop("Employment ID")
st.subheader("Auto-Populated Demand Attributes")
col1, col2 = st.columns(2)

for idx, column in enumerate(columns):
    with col1 if idx % 2 == 0 else col2:
        if column in selected_row.index:
            value = selected_row[column]
            st.text_input(f"{column}:", value, key=column, disabled=True)
            if column in label_encoders:
                user_input.append(label_encoders[column].transform([value])[0])
            else:
                user_input.append(value)

if st.button("Get Suitable Employees"):
    try:
        recommendations = recommend_employees(model, user_input, data)
        st.subheader("Top 3 Employees:")
        for i, employee in enumerate(recommendations, 1):
            st.write(f"{i}. Employee ID: {employee}")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
