# Improt dependencies
import streamlit as st
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


# Page set-up
st.set_page_config( page_title="Virufy Dashboard", 
                    layout='centered',
                    page_icon = 'Virufy_Icon.png'
                    )

st.image('Virufy_Icon.png')

                    
st.title("How well do our models predict Covid?") # Mian title

# Data prep
results_df = pd.read_csv("results.csv") # load data
country_list = list(results_df.country.unique()) # get the list name of country
model_list = ["Model_1", "Model_2", "Model_3", "Model_4", "Model_5"] # get the list name of model
 

# drill-down

    ## By country
country = st.sidebar.selectbox("Country",options=country_list)
df_country = results_df[results_df['country']==country] # By country


   ## Then by age
age = st.sidebar.slider("Age Range (5 increment)", 18,80,(18,80),5) #df_country.age.min(),df_country.age.max(),(df_country.age.min(), df_country.age.max()), 5) # return as int/float/date/time/datetime or tuple of int/float/date/time/datetime
if age[0]==age[1]:
    df_country_age = df_country[df_country['age']==age[0]] # If the range is limit to one number
else:
    df_country_age = df_country[(df_country['age']>age[0]) & (df_country['age']<age[1])] # else slicing the df by range of the age 
    

    ## Then by model
model = st.sidebar.selectbox("Model",options=model_list)
pcr_test_result_pred = df_country_age[model] # Series of result 



## Confusion Matrix
pcr_test_result = df_country_age['pcr_test_result']


confusion = confusion_matrix(pcr_test_result, pcr_test_result_pred)
FN = confusion[1][0]
TN = confusion[0][0]
TP = confusion[1][1]
FP = confusion[0][1]

# Pie Chart
fig1, ax1 = plt.subplots()
ax1.pie([FN,TN,TP,FP], labels= ['False Negative' , 'True Negative' , 'True Positive' , 'False Positive'], autopct='%1.1f%%')
st.pyplot(fig1)

# Bar Chart
fig2, ax2 = plt.subplots()
ax2.bar(['False Negative' , 'True Negative' , 'True Positive' , 'False Positive'],[FN,TN,TP,FP])
st.pyplot(fig2)





# DataFrames
df_index = ["Sensitivity", "Specificity", "PPV", "NPV"]
st.subheader(f"Results for: {country}")
df = pd.DataFrame()

pcr_test_result = df_country_age['pcr_test_result']

for i in model_list:
    # prediction
    pred = df_country_age[i]
    # confusion matrix
    Matrix = confusion_matrix(pcr_test_result,pred)
    # FN,TN.TP,FP
    F_N = Matrix[1][0]
    T_N = Matrix[0][0]
    T_P = Matrix[1][1]
    F_P = Matrix[0][1]
    #Sensitivity, Specificity, PPV, NPV
    Sens = (T_P/(T_P+F_N))
    Spec = (T_N/(T_N+F_P))
    Ppv = (T_P/(T_P+F_P))
    Npv = (T_N/(F_N+T_N))

    # Each model assign the Sensitivity, Specificity, PPV, NPV
    df[i]= pd.Series([Sens,Spec,Ppv,Npv])
df['indicator'] = df_index
df = df.set_index('indicator') # Set index

st.dataframe(df.style.highlight_max(axis=1)) # Display the dataframe
