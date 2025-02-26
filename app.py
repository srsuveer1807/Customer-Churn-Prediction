import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
import pickle
import tensorflow as tf
import numpy as np
import streamlit as st

#load trained model
model=tf.keras.models.load_model('model.h5')

#load the encoders and scalers
with open('gender_encoder.pkl','rb') as file:
    label_encoder_tuple=pickle.load(file)
label_encoder=label_encoder_tuple[0]
with open('ohe_geo.pkl','rb') as file:
    ohe_tuple = pickle.load(file)
ohe = ohe_tuple[0]

with open('scaler.pkl','rb') as file:
    scaler_tuple=pickle.load(file)
scaler=scaler_tuple[0]

## streamlit app
st.title('Customer Churn Prediction')

# User input
geography = st.selectbox('Geography', ohe.categories_[0])
gender = st.selectbox('Gender', label_encoder.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

## OHE for Geography
geo_encoded=ohe.transform([[geography]]).toarray()
geo_encoded_df=pd.DataFrame(geo_encoded,columns=ohe.get_feature_names_out(['Geography']))

#combine OHE data and input data
input_data=pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)

#scale the input data
input_data_scaled=scaler.transform(input_data)

#predict the input data
prediction=model.predict(input_data_scaled)
prediction_proba=prediction[0][0]

st.write('The churn probability is:%.2f',prediction_proba)
if prediction_proba>0.5:
    st.write('The ustomer is likely to churn')
else:
    st.write('The customer is not likely to churn')