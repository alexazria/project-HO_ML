import streamlit as st
import pandas as pd
import joblib

# Function to load data and model
def load_data_model():
    data = pd.read_csv('Salary.csv')
    grid_search = joblib.load('grid_search.joblib')  # Replace with your actual model file
    return data, grid_search

# Function to prepare user input data
def prepare_user_input(age, gender, job_title, country, race, years_of_experience, senior, education_level, X_train):
    user_data = {
        'Age': age,
        'Gender': 1 if gender == 'Male' else 0,
        'Years of Experience': years_of_experience,
        'Senior': senior,
        'Education Level': education_level
    }
    for column in X_train:
        if column.startswith('Job Title_'):
            user_data[column] = 1 if column == f'Job Title_{job_title}' else 0
        elif column.startswith('Country_'):
            user_data[column] = 1 if column == f'Country_{country}' else 0
        elif column.startswith('Race_'):
            user_data[column] = 1 if column == f'Race_{race}' else 0
        else:
            if column not in user_data:
                user_data[column] = 0
    return pd.DataFrame([user_data], columns=X_train)

# Main function for Streamlit app
def main():
    st.title("Salary Prediction")

    # Load data and model
    data, grid_search = load_data_model()
    X_train_columns = joblib.load('x_train_columns.joblib')


    # User inputs
    age = st.number_input("Enter your age", min_value=18, max_value=100, step=1)
    gender = st.selectbox("Enter your gender", options=['Male', 'Female'])
    job_title = st.selectbox("Enter your job title", options=sorted(data['Job Title'].unique()))
    country = st.selectbox("Enter your nationality", options=sorted(data['Country'].unique()))
    race = st.selectbox("Enter your race", options=sorted(data['Race'].unique()))
    years_of_experience = st.number_input("Enter your years of experience", min_value=0, max_value=50, step=1)
    senior = st.radio("Are you a senior?", options=["yes", "no"])
    if senior == "yes":
        senior = 1
    else:
        senior = 0
    education_level = st.slider("Enter your education level", min_value=1, max_value=10, step=1)

    # Prediction
    if st.button('Predict Salary'):
        user_input_df = prepare_user_input(age, gender, job_title, country, race, years_of_experience, senior, education_level, X_train_columns)
        predicted_salary = grid_search.best_estimator_.predict(user_input_df)
        st.success(f"The predicted salary is: ${predicted_salary[0]:,.2f}")

if __name__ == "__main__":
    main()
