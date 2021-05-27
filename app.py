#############version used on heroku###################
import streamlit as st
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Virufy Dashboard", layout='centered')

results_df = pd.read_csv("results.csv")

st.title("How well do the models predict Covid?")

chart = st.sidebar.radio('Chart Type',('Accuracy','Bar Chart','Pie Chart'))

country_list = list(results_df.country.unique())
model_list = ["Model_1", "Model_2", "Model_3", "Model_4", "Model_5"]



if chart == 'Pie Chart':
    country = st.sidebar.selectbox("Country",options=country_list)
    model = st.sidebar.selectbox("Model",options=model_list)
    age = st.sidebar.slider("Age Range", 18,80,(18, 80), 1)
    results_df_country = results_df[results_df['country']==country]

    pcr_test_result = results_df_country['pcr_test_result']
    pcr_test_result_pred = results_df_country[model]

    confusion = confusion_matrix(pcr_test_result, pcr_test_result_pred)
    FN = confusion[1][0]
    TN = confusion[0][0]
    TP = confusion[1][1]
    FP = confusion[0][1]
    fig, ax = plt.subplots()
    ax.pie([FN,TN,TP,FP], labels= ['False Negative' , 'True Negative' , 'True Positive' , 'False Positive'], autopct='%1.1f%%')
    st.pyplot(fig)

if chart == 'Bar Chart':
    country = st.sidebar.selectbox("Country",options=country_list)
    model = st.sidebar.selectbox("Model",options=model_list)
    age = st.sidebar.slider("Age Range", 18,80,(18, 80), 1)
    results_df_country = results_df[results_df['country']==country]

    pcr_test_result = results_df_country['pcr_test_result']
    pcr_test_result_pred = results_df_country[model]

    confusion = confusion_matrix(pcr_test_result, pcr_test_result_pred)
    FN = confusion[1][0]
    TN = confusion[0][0]
    TP = confusion[1][1]
    FP = confusion[0][1]
    fig, ax = plt.subplots()
    ax.bar(['False Negative' , 'True Negative' , 'True Positive' , 'False Positive'],[FN,TN,TP,FP])
    st.pyplot(fig)

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

if chart == 'Accuracy':
    age = st.sidebar.slider("Age Range", 18,80,(18, 80), 1)
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

