#############version used on heroku###################
import streamlit as st
from PIL import Image
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
st.set_page_config(page_title="Virufy Dashboard", layout='centered')

results_df = pd.read_csv("results.csv")

st.title("How well does Model A predict Covid?")

country_list = list(results_df.country.unique())
country = st.selectbox("Country",options=country_list)

results_df_country = results_df[results_df['country']==country]

pcr_test_result = results_df_country['pcr_test_result']
pcr_test_result_pred = results_df_country['pcr_test_result_pred']

confusion = confusion_matrix(pcr_test_result, pcr_test_result_pred)
FN = confusion[1][0]
TN = confusion[0][0]
TP = confusion[1][1]
FP = confusion[0][1]

fig, ax = plt.subplots()
ax.bar(['False Negative' , 'True Negative' , 'True Positive' , 'False Positive'],[FN,TN,TP,FP])
st.pyplot(fig)
