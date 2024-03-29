from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

import streamlit as st
import pandas as pd

init_dataset = pd.read_csv('ds_salaries (1).csv')
init_dataset.head()

columns_to_drop = ['work_year', 'salary', 'salary_currency', 'employee_residence', 'company_size']
dataset = init_dataset.drop(columns=columns_to_drop)


X = dataset[['experience_level', 'employment_type', 'job_title', 'company_location', 'remote_ratio']]
y = dataset['salary_in_usd']

X_encoded = pd.get_dummies(X, columns=['experience_level', 'employment_type', 'job_title', 'company_location','remote_ratio'])

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

encoder = OneHotEncoder()

X_encoded = encoder.fit_transform(X)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

def main():
    st.title("Salary Estimator")
    
    experience_level = st.selectbox("Select Experience Level", ['SE','MI','EN','EX'])

 
    employment_type = st.selectbox("Select Employment Type", ['FT','CT','FL','PT'])


    job_title = st.selectbox("Enter Job Title", ['Principal Data Scientist', 'ML Engineer', 'Data Scientist', 'Applied Scientist', 
                                                 'Data Analyst', 'Data Modeler', 'Research Engineer', 'Analytics Engineer', 
                                                 'Business Intelligence Engineer', 'Machine Learning Engineer', 'Data Strategist', 
                                                 'Data Engineer', 'Computer Vision Engineer', 'Data Quality Analyst', 'Compliance Data Analyst', 
                                                 'Data Architect', 'Applied Machine Learning Engineer', 'AI Developer', 'Research Scientist', 
                                                 'Data Analytics Manager', 'Business Data Analyst', 'Applied Data Scientist', 'Staff Data Analyst', 
                                                 'ETL Engineer', 'Data DevOps Engineer', 'Head of Data', 'Data Science Manager', 'Data Manager', 
                                                 'Machine Learning Researcher', 'Big Data Engineer', 'Data Specialist', 'Lead Data Analyst', 
                                                 'BI Data Engineer', 'Director of Data Science', 'Machine Learning Scientist', 'MLOps Engineer', 
                                                 'AI Scientist', 'Autonomous Vehicle Technician', 'Applied Machine Learning Scientist',
                                                 'Lead Data Scientist', 'Cloud Database Engineer', 'Financial Data Analyst', 'Data Infrastructure Engineer',
                                                 'Software Data Engineer', 'AI Programmer', 'Data Operations Engineer', 'BI Developer', 'Data Science Lead', 
                                                 'Deep Learning Researcher', 'BI Analyst', 'Data Science Consultant', 'Data Analytics Specialist', 
                                                 'Machine Learning Infrastructure Engineer', 'BI Data Analyst', 'Head of Data Science', 'Insight Analyst', 
                                                 'Deep Learning Engineer', 'Machine Learning Software Engineer', 'Big Data Architect', 'Product Data Analyst', 
                                                 'Computer Vision Software Engineer', 'Azure Data Engineer', 'Marketing Data Engineer', 'Data Analytics Lead', 
                                                 'Data Lead', 'Data Science Engineer', 'Machine Learning Research Engineer', 'NLP Engineer', 'Manager Data Management', 
                                                 'Machine Learning Developer', '3D Computer Vision Researcher', 'Principal Machine Learning Engineer', 
                                                 'Data Analytics Engineer', 'Data Analytics Consultant', 'Data Management Specialist', 'Data Science Tech Lead', 
                                                 'Data Scientist Lead', 'Cloud Data Engineer', 'Data Operations Analyst', 'Marketing Data Analyst', 'Power BI Developer', 
                                                 'Product Data Scientist', 'Principal Data Architect', 'Machine Learning Manager', 'Lead Machine Learning Engineer', 
                                                 'ETL Developer', 'Cloud Data Architect', 'Lead Data Engineer', 'Head of Machine Learning', 'Principal Data Analyst', 
                                                 'Principal Data Engineer', 'Staff Data Scientist', 'Finance Data Analyst'])
    
    company_location = st.selectbox("Company Location", ['ES', 'US', 'CA', 'DE', 'GB', 'NG', 'IN', 'HK', 'NL', 'CH', 'CF',
    'FR', 'FI', 'UA', 'IE', 'IL', 'GH', 'CO', 'SG', 'AU', 'SE', 'SI',
    'MX', 'BR', 'PT', 'RU', 'TH', 'HR', 'VN', 'EE', 'AM', 'BA', 'KE',
    'GR', 'MK', 'LV', 'RO', 'PK', 'IT', 'MA', 'PL', 'AL', 'AR', 'LT',
    'AS', 'CR', 'IR', 'BS', 'HU', 'AT', 'SK', 'CZ', 'TR', 'PR', 'DK',
    'BO', 'PH', 'BE', 'ID', 'EG', 'AE', 'LU', 'MY', 'HN', 'JP', 'DZ',
    'IQ', 'CN', 'NZ', 'CL', 'MD', 'MT'])
   
    remote_ratio = st.slider("Remote Work Ratio (%)", min_value=0, max_value=100, value=50, step=50)

    user_input = pd.DataFrame.from_dict({
        'experience_level': experience_level, 
        'employment_type': employment_type,
        'job_title': job_title,
        'company_location': company_location,
        'remote_ratio': remote_ratio
   
    }, orient='index').T 
    
    # user_input_encoded = encoder.transform(user_input)
    
    # user_input_encoded = pd.DataFrame(encoder.transform(user_input).toarray(), columns=encoder.get_feature_names(input_features=user_input.columns))
    
    user_input_encoded = encoder.transform(user_input)
    feature_names_encoded = encoder.get_feature_names_out(input_features=user_input.columns)
    user_input_encoded_df = pd.DataFrame(user_input_encoded.toarray(), columns=feature_names_encoded)
    
    predicted_salary = model.predict(user_input_encoded)[0]  
  
    st.write(f"Estimated Average Salary: ${predicted_salary:.2f}")

if __name__ == '__main__':
    main()