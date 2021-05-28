# Improt dependencies
import streamlit as st
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


# Page set-up
st.set_page_config(page_title="Virufy Dashboard", layout='centered') # page_icon
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
age = st.sidebar.slider("Age Range (5 increment)", 18,80,(df_country.age.min(), df_country.age.max()), 5) # return as int/float/date/time/datetime or tuple of int/float/date/time/datetime
if type(age) is int:
    df_country_age = df_country[age] # If the range is limit to one number
elif type(age) is tuple:
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


fig, ax = plt.subplots()
ax.pie([FN,TN,TP,FP], labels= ['False Negative' , 'True Negative' , 'True Positive' , 'False Positive'], autopct='%1.1f%%')

st.pyplot(fig)
'''

# if chart == 'Bar Chart':
# country = st.sidebar.selectbox("Country",options=country_list)
# model = st.sidebar.selectbox("Model",options=model_list)
# # age = st.sidebar.slider("Age Range", 18,80,(18, 80), 1)
# results_df_country = results_df[results_df['country']==country]

# pcr_test_result = results_df_country['pcr_test_result']
# pcr_test_result_pred = results_df_country[model]

# confusion = confusion_matrix(pcr_test_result, pcr_test_result_pred)
# FN = confusion[1][0]
# TN = confusion[0][0]
# TP = confusion[1][1]
# FP = confusion[0][1]
fig1, ax = plt.subplots()
ax.bar(['False Negative' , 'True Negative' , 'True Positive' , 'False Positive'],[FN,TN,TP,FP])
st.pyplot(fig1)

# sens = (round(TP/(TP+FN)))
# spec = (round(TN/(TN+FP)))
# ppv = (round(TP/(TP+FP)))
# npv = (round(TN/(FN+TN)))

df_index = ["Sensitivity", "Specificity", "PPV", "NPV"]

# def df_acc (ctry):
    # st.subheader("Results for: {}".format(ctry))
    # df = pd.DataFrame(index=df_index,
    #    data =  np.random.randn(4, 5),
    #     columns=(model_list))
    # st.dataframe(df.style.highlight_max(axis=0))

# if chart == 'Accuracy':
# age = st.sidebar.slider("Age Range", 18,80,(18, 80), 1)

st.subheader("Results for: {}".format(country_list[0]))
df = pd.DataFrame(index=df_index,
    data =  np.random.randn(4, 5),
    columns=(model_list))
st.dataframe(df.style.highlight_max(axis=0))

st.subheader("Results for: {}".format(country_list[1]))
df = pd.DataFrame(index=df_index,
    data =  np.random.randn(4, 5),
    columns=(model_list))
st.dataframe(df.style.highlight_max(axis=0))

st.subheader("Results for: {}".format(country_list[2]))
df = pd.DataFrame(index= df_index,
    data =  np.random.randn(4, 5),
    columns=(model_list))
st.dataframe(df.style.highlight_max(axis=0))

st.subheader("Results for: {}".format(country_list[3]))
df = pd.DataFrame(index= df_index,
    data =  np.random.randn(4, 5),
    columns=(model_list))
st.dataframe(df.style.highlight_max(axis=0))
##################################################
df = pd.DataFrame(index= df_index,
    data =  np.random.randn(4, 5),
    columns=(model_list))
st.dataframe(df.style.highlight_max(axis=0))

'''