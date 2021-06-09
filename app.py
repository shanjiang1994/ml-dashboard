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
country_list.append('All Countries') # Add 'All' option
model_list = ["Model_1", "Model_2", "Model_3", "Model_4", "Model_5"] # get the list name of model
 

# drill-down

## By country
country = st.sidebar.selectbox("Country",options=country_list)

if country == 'All Countries':
    df_country = results_df # All result
else:
    df_country = results_df[results_df['country']==country] # By country


## Then by age
# age_min = float(df_country.age.min())
# age_max = float(df_country.age.max())

age_min = int(df_country.age.min())
age_max = int(df_country.age.max())

age = st.sidebar.slider("Age Range (5 increment)", age_min,age_max,value =(age_min,age_max),step = 5) # return as int/float/date/time/datetime or tuple of int/float/date/time/datetime
if age[0]==age[1]:
    df_country_age = df_country[df_country['age']==age[0]] # If the range is limit to one number
else:
    df_country_age = df_country[(df_country['age']>age[0]) & (df_country['age']<age[1])] # else slicing the df by range of the age 
    


## If displaying all countires result
if country == 'All Countries' :
    ### Create dataframes for each country
    dict_countries={i:pd.DataFrame() for i in country_list[:-1]} 

    ### for each country, cache the pcr_test_result 
    for country_name in dict_countries: 
        df_allcountry_age = df_country_age[df_country_age['country']==country_name]
        result = df_allcountry_age.pcr_test_result
        #### for each model within each country: compute the confusion matrix and indicators  
        for model_name in model_list:
           
            pred = df_allcountry_age[model_name]

            Matrix = confusion_matrix(result,pred)

            F_N = Matrix[1][0]
            T_N = Matrix[0][0]
            T_P = Matrix[1][1]
            F_P = Matrix[0][1]
            
            
            Sens = (T_P/(T_P+F_N))
            Spec = (T_N/(T_N+F_P))
            Ppv = (T_P/(T_P+F_P))
            Npv = (T_N/(F_N+T_N))
            
            dict_countries[country_name][model_name]= pd.Series([Sens,Spec,Ppv,Npv])
            
            
            
        ### display the dataframe for each country
        df_index = ["Sensitivity", "Specificity", "PPV", "NPV"]
        dict_countries[country_name]['Indicator'] = df_index
        dict_countries[country_name] = dict_countries[country_name].set_index('Indicator') # Set index
        st.subheader(f"Model robustness: {country_name}")
        st.dataframe(dict_countries[country_name].style.highlight_max(axis=1))

else:
    ## Then select by model
    model = st.sidebar.selectbox("Model",options=model_list)

    pcr_test_result_pred = df_country_age[model]

    # DataFrames

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

    df_index = ["Sensitivity", "Specificity", "PPV", "NPV"]
    df['indicator'] = df_index
    df = df.set_index('indicator') # Set index

    st.dataframe(df.style.highlight_max(axis=1)) # Display the dataframe




    ## Confusion Matrix
    pcr_test_result = df_country_age['pcr_test_result']

    confusion = confusion_matrix(pcr_test_result, pcr_test_result_pred)
    FN = confusion[1][0]
    TN = confusion[0][0]
    TP = confusion[1][1]
    FP = confusion[0][1]


    # Title
    st.subheader(f"Distribution for: {country}")


    
    # Pie Chart
    fig, ax = plt.subplots(1,2,figsize=(10,5))
    ax[0].pie([FN,TN,TP,FP], labels= ['False Negative' , 'True Negative' , 'True Positive' , 'False Positive'], autopct='%1.1f%%')




    # Bar Chart
    ax[1].bar(['False Negative' , 'True Negative' , 'True Positive' , 'False Positive'],[FN,TN,TP,FP])
    ax[1].tick_params(labelsize=8)
    st.pyplot(fig)
