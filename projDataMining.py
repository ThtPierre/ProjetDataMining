import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from mpl_toolkits.mplot3d import Axes3D
from fonction import *

def detect_delimiter(file):
    sample = file.read(1024).decode('utf-8')
    file.seek(0)
    if sample.count(',') > sample.count(';'):
        return ','
    else:
        return ';'

st.title("Projet Data Mining")

with st.sidebar:
    st.title("Pierre TAITHE, Benjamin ROUSSEAU, Madjid ZEHANI")
    st.header("Data Mining 2024")

st.header("Part I: Initial Data Exploration")
uploaded_file = st.file_uploader("Choose a file", type="csv")
if uploaded_file is None:
    st.write("Please add a CSV File")
else:
    delimiter = detect_delimiter(uploaded_file)
    data = pd.read_csv(uploaded_file, delimiter=delimiter)
    st.write("Beginning of your file:")
    st.write(data.head(3))
    st.write("End of your file :")
    st.write(data.tail(3))

    st.subheader("Information about the selected file :")
    st.write("Number of lines : ", data.shape[0])
    st.write("Number of columns : ", data.shape[1])
    missing_percentage = data.isnull().mean().round(4) * 100

    data_types = data.dtypes
    df_info = pd.DataFrame({
        "Data types": data_types,
        "% of missing data": missing_percentage,
    })
    st.write("Name of columns, their type and % of missing data:")
    st.write(df_info)
    
    st.header("Part II: data pre-processing and cleaning:")
    st.subheader("1. Data cleaning : ")
    st.write("Choose the method you want to use to clean your data :")
    method = st.selectbox(" Please Select", ["No Cleaning",
                                     "1/ Delete lines with missing data",
                                     "2/ Delete columns with too much missing data",
                                     "3/ Fill missing data with the mean of the column",
                                     "4/ Fill missing data with the median of the column",
                                     "5/ Fill missing data with the mode of the column",
                                     "6/ Fill missing data with KNN"])
    
    if method == "No Cleaning":
        data_propre = data

    elif method == "1/ Delete lines with missing data":
        data_propre = data.dropna()

    elif method == "2/ Delete columns with too much missing data":
        threshold = st.slider("Threshold (%)", 0, 100, 50)
        data_propre = data.dropna(thresh=int((threshold / 100) * data.shape[0]), axis=1)

    elif method == "3/ Fill missing data with the mean of the column":
        data_propre = clean_mean(data)

    elif method == "4/ Fill missing data with the median of the column":
        data_propre = clean_median(data)

    elif method == "5/ Fill missing data with the mode of the column":
        imputer = SimpleImputer(strategy="most_frequent")
        data_propre = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

    elif method == "6/ Fill missing data with KNN":
        data_propre = clean_knn(data)

    st.write("Cleaned data : ")
    st.write(data_propre.head(3))
    st.write(data_propre.tail(3))
