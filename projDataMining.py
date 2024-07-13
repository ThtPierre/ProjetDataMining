import streamlit as st
import pandas as pd

def detect_delimiter(file):
    sample = file.read(1024).decode('utf-8')
    file.seek(0)
    if sample.count(',') > sample.count(';'):
        return ','
    else:
        return ';'


st.title("Projet Data Mining")

with st.sidebar:
    st.title("Pierre Taithe, Benjamin Rousseau, Madjid Zehani")
    st.header("Data Mining 2024")

uploaded_file = None
uploaded_file = st.file_uploader("Choose a file", type="csv")
if uploaded_file is None:
    st.write("Please add a CSV File")
else :  
    delimiter = detect_delimiter(uploaded_file)
    uploaded_file = pd.read_csv(uploaded_file, delimiter=delimiter)
    st.write("Beginning of your file:")
    st.write(uploaded_file.head(3))
    st.write("End of your file :")
    st.write(uploaded_file.tail(3))

    st.subheader("Information about the selected file :")
    st.write("Number of lines : ", uploaded_file.shape[0])
    st.write("Number of columns : ", uploaded_file.shape[1])
    missing_percentage = uploaded_file.isnull().mean().round(4) * 100
    data_types = uploaded_file.dtypes
    df_info = pd.DataFrame({
        'Data types': data_types,
        '% of missing data': missing_percentage
    })
    st.write("Name of columns, their type and % of missing data:")
    st.write(df_info)

    st.header("Data Cleaning:")
    st.write("Choose the method you want to use to clean your data:")
    method = st.selectbox("Method", ["Supress lines with missing data", "Supress columns with too much missing data", "Fill missing data with the mean of the column"])


