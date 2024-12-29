import pandas as pd
import pickle
import streamlit as st

# Load the model
pickle_file = 'boston_model.sav'
with open(pickle_file, 'rb') as f:
    model = pickle.load(f)

# Title
st.title('Boston Housing Price Predictor')

# Input features
st.sidebar.header('User Input Parameters')

def user_input_features():
    CRIM = st.sidebar.number_input('CRIM (per capita crime rate)', value=0.1)
    ZN = st.sidebar.number_input('ZN (proportion of residential land)', value=0.0)
    INDUS = st.sidebar.number_input('INDUS (proportion of non-retail business acres)', value=0.0)
    CHAS = st.sidebar.selectbox('CHAS (Charles River dummy variable)', [0, 1], index=0)
    NOX = st.sidebar.number_input('NOX (nitric oxides concentration)', value=0.1)
    RM = st.sidebar.number_input('RM (average number of rooms per dwelling)', value=6.0)
    AGE = st.sidebar.number_input('AGE (proportion of owner-occupied units)', value=50.0)
    DIS = st.sidebar.number_input('DIS (distance to employment centers)', value=1.0)
    RAD = st.sidebar.number_input('RAD (index of accessibility to highways)', value=1.0)
    TAX = st.sidebar.number_input('TAX (property tax rate)', value=300.0)
    PTRATIO = st.sidebar.number_input('PTRATIO (pupil-teacher ratio)', value=15.0)
    B = st.sidebar.number_input('B (proportion of Black population)', value=400.0)
    LSTAT = st.sidebar.number_input('LSTAT (lower status of the population)', value=12.0)
    
    # Create a dictionary of user input
    data = {
        'CRIM': CRIM,
        'ZN': ZN,
        'INDUS': INDUS,
        'CHAS': CHAS,
        'NOX': NOX,
        'RM': RM,
        'AGE': AGE,
        'DIS': DIS,
        'RAD': RAD,
        'TAX': TAX,
        'PTRATIO': PTRATIO,
        'B': B,
        'LSTAT': LSTAT
    }
    return pd.DataFrame(data, index=[0])

# Get user input
input_df = user_input_features()

# Main panel
st.subheader('User Input Parameters')
st.write(input_df)

# Predict and display the results
if st.button('Predict'):
    prediction = model.predict(input_df)
    st.subheader('Prediction')
    st.write(f"The predicted median house price is: ${prediction[0]:,.2f}")
