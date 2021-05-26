#############version used on heroku###################
import streamlit as st
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
st.set_page_config(page_title="Virufy Dashboard", layout='centered')

results_df = pd.read_csv("results.csv")

st.title("How well does Model A predict Covid?")

country_list = list(results_df.country.unique())
country = st.sidebar.selectbox("Country",options=country_list)

results_df_country = results_df[results_df['country']==country]

pcr_test_result = results_df_country['pcr_test_result']
pcr_test_result_pred = results_df_country['pcr_test_result_pred']

confusion = confusion_matrix(pcr_test_result, pcr_test_result_pred)
FN = confusion[1][0]
TN = confusion[0][0]
TP = confusion[1][1]
FP = confusion[0][1]

chart = st.sidebar.radio('Chart Type',('Bar Chart','Pie Chart'))

if chart == 'Pie Chart':
    fig, ax = plt.subplots()
    ax.pie([FN,TN,TP,FP], labels= ['False Negative' , 'True Negative' , 'True Positive' , 'False Positive'], autopct='%1.1f%%')
    st.pyplot(fig)

if chart == 'Bar Chart':
    fig, ax = plt.subplots()
    ax.bar(['False Negative' , 'True Negative' , 'True Positive' , 'False Positive'],[FN,TN,TP,FP])
    st.pyplot(fig)

# st.subheader("False Negative: {}%".format (round(FN/len(pcr_test_result)*100)))
# st.subheader("True Negative: {}%".format (round(TN/len(pcr_test_result)*100)))
# st.subheader("True Positive: {}%".format (round(TP/len(pcr_test_result)*100)))
# st.subheader("False Positive: {}%".format (round(FP/len(pcr_test_result)*100)))

