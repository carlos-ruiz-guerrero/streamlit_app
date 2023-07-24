from sklearn.models.regression import LinearRegression
import streamlit as st
import pandas as pd

dataset = pd.read_csv("¨name_of_file.csv")
X = dataset[['experience_level', 'employment_type', 'job_title', 'remote_ratio', 'company_size']]
y = dataset['salary_in_usd']
model = LinearRegression()
model.fit(X, y)

def main():
    st.title("Salary Estimator")
    
    experience_level = st.selectbox("Select Experience Level", ['SE' 'MI' 'EN' 'EX'])
    employment_type = st.selectbox("Select Employment Type", ['FT' 'CT' 'FL' 'PT'])
    job_title = st.text_input("Enter Job Title", "")
    remote_ratio = st.slider("Remote Work Ratio (%)", min_value=0, max_value=100, value=50, step=1)
    company_size = st.slider("Company Size", min_value=1, max_value=10000, value=100, step=1)

    user_input = pd.DataFrame({
        'experience_level': [experience_level],
        'employment_type': [employment_type],
        'job_title': [job_title],
        'remote_ratio': [remote_ratio],
        'company_size': [company_size]
    })    

    predicted_salary = model.predict(user_input)[0]
    
    st.write(f"Estimated Average Salary: ${predicted_salary:.2f}")

if __name__ == '__main__':
    main()