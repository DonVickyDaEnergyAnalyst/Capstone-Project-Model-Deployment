import streamlit as st
import pandas as pd
import numpy as np
import pickle

ALL_FEATURES = ['Age', 'DailyRate', 'DistanceFromHome', 'Education',
       'EnvironmentSatisfaction', 'Gender', 'HourlyRate', 'JobInvolvement',
       'JobLevel', 'JobRole', 'JobSatisfaction', 'MonthlyIncome',
       'MonthlyRate', 'NumCompaniesWorked', 'OverTime', 'PercentSalaryHike',
       'PerformanceRating', 'RelationshipSatisfaction', 'StockOptionLevel',
       'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',
       'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
       'YearsWithCurrManager', 'BusinessTravel_Travel_Frequently',
       'BusinessTravel_Travel_Rarely', 'Department_Research & Development',
       'Department_Sales', 'EducationField_Life Sciences',
       'EducationField_Marketing', 'EducationField_Medical',
       'EducationField_Other', 'EducationField_Technical Degree',
       'MaritalStatus_Married', 'MaritalStatus_Single']

CATEGORICAL_COLS = ['BusinessTravel',
 'Department',
 'EducationField',
 'Gender',
 'JobRole',
 'MaritalStatus',
 'OverTime']

#Load the model
@st.cache_resource #tells Streamlit to load the model once and save into cache memory for future reuse
def load_assets():
    model = pickle.load(open("Employee_attrition_model.pkl", "rb"))
    encoder = pickle.load(open("Employee_attrition_encoder.pkl", "rb"))
    scaler = pickle.load(open("Employee_attrition_scaler.pkl", "rb"))
    return model, encoder, scaler
model, encoder, scaler = load_assets()

def main():
    st.markdown("# 👩‍💼 Employee Attrition Detection App")
    st.write("This app predicts whether an employee is likely to leave the company based on their details such as personal, employment history, compensation and other factors.")

    # Inject small CSS to improve look
    st.markdown(
        """
        <style>
        .stApp { background: linear-gradient(180deg, #f7f9fc 0%, #ffffff 100%); }
        .card { background: white; padding: 12px; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.08); }
        .section-title { color: #0f172a; font-weight:600; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar with instructions and presets
    with st.sidebar:
        st.markdown("## How to use")
        st.write("Adjust the employee fields on the main page and click Predict. The model returns a prediction of attrition.")
        st.markdown("---")
        st.markdown("Built by Team Sentinel (Group 13) of TechCrush Cohort 3.")

    # Create input fields for each feature
    st.header("Employee Information")

    # Organize inputs into columns
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.subheader("Personal Details")
        age = st.slider("Age", min_value=18, max_value=60, value=30)
        gender = st.selectbox("Gender", ["Female", "Male"]) # Use original labels for user
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"]) # Use original labels
        distance_from_home = st.slider("Distance From Home (miles)", 1, 29, 5)
        education = st.selectbox("Education Level", [1, 2, 3, 4, 5], format_func=lambda x: {1: 'Below College', 
            2: 'College', 3: 'Bachelor', 4: 'Master', 5: 'Doctor'}.get(x)) # Use original labels and encoded values

    with col2:
        st.subheader("Employment Details")
        department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"]) # Use original labels
        job_role = st.selectbox("Job Role", ['Sales Executive', 'Research Scientist', 'Laboratory Technician',
            'Manufacturing Director', 'Healthcare Representative', 'Manager',
            'Sales Representative', 'Research Director', 'Human Resources']) # Use original labels
        job_level = st.slider("Job Level", 1, 5, 1)
        job_involvement = st.selectbox("Job Involvement", [1, 2, 3, 4], format_func=lambda x: {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'}.get(x)) # Use original labels and encoded values
        job_satisfaction = st.selectbox("Job Satisfaction", [1, 2, 3, 4], format_func=lambda x: {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'}.get(x)) # Use original labels and encoded values
        environment_satisfaction = st.selectbox("Environment Satisfaction", [1, 2, 3, 4], format_func=lambda x: {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'}.get(x)) # Use original labels and encoded values
        relationship_satisfaction = st.selectbox("Relationship Satisfaction", [1, 2, 3, 4], format_func=lambda x: {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'}.get(x)) # Use original labels and encoded values
        work_life_balance = st.selectbox("Work Life Balance", [1, 2, 3, 4], format_func=lambda x: {1: 'Bad', 2: 'Good', 3: 'Better', 4: 'Best'}.get(x)) # Use original labels and encoded values

    with col3:
        st.subheader("Compensation")
        daily_rate = st.number_input("Daily Rate", min_value=100, max_value=1500, value=800)
        hourly_rate = st.number_input("Hourly Rate", min_value=30, max_value=100, value=65)
        monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=20000, value=6500)
        monthly_rate = st.number_input("Monthly Rate", min_value=2000, max_value=27000, value=14000)
        percent_salary_hike = st.slider("Percent Salary Hike", 11, 25, 15)
        stock_option_level = st.slider("Stock Option Level", 0, 3, 0)
       
    with col4:
        st.subheader("Work History")
        total_working_years = st.slider("Total Working Years", 0, 40, 10)
        training_times_last_year = st.slider("Training Times Last Year", 0, 6, 3)
        years_at_company = st.slider("Years at Company", 0, 40, 5)
        years_in_current_role = st.slider("Years in Current Role", 0, 18, 3)
        years_since_last_promotion = st.slider("Years Since Last Promotion", 0, 15, 1)
        years_with_curr_manager = st.slider("Years with Current Manager", 0, 17, 3)
        num_companies_worked = st.slider("Number of Companies Worked", 0, 9, 2)
        business_travel = st.selectbox("Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"]) # Use original labels
        overtime = st.selectbox("OverTime", ["No", "Yes"]) # Use original labels
        education_field = st.selectbox("Education Field", ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"]) # Use original labels
        performance_rating = st.selectbox("Performance Rating", [3, 4], format_func=lambda x: {3: 'Excellent', 4: 'Outstanding'}.get(x)) # Use original labels and encoded values

    # Creating a button to trigger prediction
    if st.button("Predict Employee Attrition"):
        # Collect input data into a dictionary
        input_dict = {
            'Age': age,
            'DailyRate': daily_rate,
            'DistanceFromHome': distance_from_home,
            'Education': education,
            'EnvironmentSatisfaction': environment_satisfaction,
            'Gender': gender,
            'HourlyRate': hourly_rate,
            'JobInvolvement': job_involvement,
            'JobLevel': job_level,
            'JobRole': job_role,
            'JobSatisfaction': job_satisfaction,
            'MaritalStatus': marital_status,
            'MonthlyIncome': monthly_income,
            'MonthlyRate': monthly_rate,
            'NumCompaniesWorked': num_companies_worked,
            'OverTime': overtime,
            'PercentSalaryHike': percent_salary_hike,
            'PerformanceRating': performance_rating,
            'RelationshipSatisfaction': relationship_satisfaction,
            'StockOptionLevel': stock_option_level,
            'TotalWorkingYears': total_working_years,
            'TrainingTimesLastYear': training_times_last_year,
            'WorkLifeBalance': work_life_balance,
            'YearsAtCompany': years_at_company,
            'YearsInCurrentRole': years_in_current_role,
            'YearsSinceLastPromotion': years_since_last_promotion,
            'YearsWithCurrManager': years_with_curr_manager,
            'BusinessTravel': business_travel,
            'Department': department,
            'EducationField': education_field,
        }

        # Convert input data to a pandas DataFrame
        input_df = pd.DataFrame([input_dict])

        # --- Preprocessing steps ---
        # Drop columns that were dropped during training of the model
        cols_to_drop_app = ["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"]
        input_df = input_df.drop(columns=[col for col in cols_to_drop_app if col in input_df.columns])

        # Apply label encoding for binary columns (Gender and OverTime)
        if 'Gender' in input_df.columns:
            input_df['Gender'] = input_df['Gender'].map({'Female': 0, 'Male': 1})
        if 'OverTime' in input_df.columns:
            input_df['OverTime'] = input_df['OverTime'].map({'No': 0, 'Yes': 1})

        # Safe access to encoder structure
        onehot_cols = []
        high_card_cols = []
        if isinstance(encoder, dict):
            onehot_cols = encoder.get('onehot_columns', []) or []
            high_card_cols = encoder.get('high_card_cols', []) or []
        else:
            # If encoder isn't a dict, attempt to infer categorical columns from CATEGORICAL_COLS
            onehot_cols = [c for c in CATEGORICAL_COLS if c in input_df.columns]

        # Identify which of those columns are actually present and of object dtype
        cols_to_onehot = [col for col in onehot_cols if col in input_df.columns and input_df[col].dtype == 'object']

        processed_input = input_df.copy()
        if cols_to_onehot:
            processed_input = pd.get_dummies(processed_input, columns=cols_to_onehot, drop_first=True)

        # Warn if high cardinality columns need special handling
        for col in high_card_cols:
            if col in processed_input.columns:
                st.warning(f"Encoding for high cardinality column '{col}' not explicitly handled. Using existing values.")

        # Ensure boolean columns are converted to int after encoding
        bool_cols_processed = processed_input.select_dtypes(include=["bool"]).columns
        for col in bool_cols_processed:
            processed_input[col] = processed_input[col].astype(int)

        # Ensure all expected features exist; fill missing with 0
        for feat in ALL_FEATURES:
            if feat not in processed_input.columns:
                processed_input[feat] = 0

        # Reorder columns to ALL_FEATURES
        all_input = processed_input[ALL_FEATURES]

        # Apply scaling
        try:
            all_input_array = all_input.values
            scaled_input_array = scaler.transform(all_input_array)
            prediction = model.predict(scaled_input_array)

            st.subheader("Prediction Result")
            if prediction[0] == 0:
                st.success("The employee is predicted to STAY")
            else:
                st.error("The employee is predicted to LEAVE")
        except Exception as e:
            st.error(f"Error during preprocessing/prediction: {e}")


if __name__ == "__main__":
    main()