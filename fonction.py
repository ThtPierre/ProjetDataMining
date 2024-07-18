import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from mpl_toolkits.mplot3d import Axes3D

def detect_delimiter(file):
    sample = file.read(1024).decode('utf-8')
    file.seek(0)
    if sample.count(',') > sample.count(';'):
        return ','
    else:
        return ';'

def clean_mean (data): 
    col_num = data.select_dtypes(include=["float64", "int64"]).columns
    imputer = SimpleImputer(strategy="mean")
    data_propre = data.copy()
    data_propre[col_num] = imputer.fit_transform(data[col_num])
    return data_propre

def clean_median (data): 
    col_num = data.select_dtypes(include=["float64", "int64"]).columns
    imputer = SimpleImputer(strategy="median")
    data_propre = data.copy()
    data_propre[col_num] = imputer.fit_transform(data[col_num])
    return data_propre

def clean_knn(data): 
    col_num = data.select_dtypes(include=["float64", "int64"]).columns
    imputer = KNNImputer(n_neighbors=2)
    data_propre = data.copy()
    data_propre[col_num] = imputer.fit_transform(data[col_num])
    return data_propre

def normalisation_minmax(data):
    col_num = data.select_dtypes(include=["float64", "int64"]).columns
    scaler = MinMaxScaler()
    data_normalise = data.copy()
    data_normalise[col_num] = scaler.fit_transform(data[col_num])
    return data_normalise

def normalisation_Zscore(data):
    col_num = data.select_dtypes(include=["float64", "int64"]).columns
    scaler = StandardScaler()
    data_normalise = data.copy()
    data_normalise[col_num] = scaler.fit_transform(data[col_num])
    return data_normalise

def normalisation_robust(data):
    col_num = data.select_dtypes(include=["float64", "int64"]).columns
    scaler = RobustScaler()
    data_normalise = data.copy()
    data_normalise[col_num] = scaler.fit_transform(data[col_num])
    return data_normalise

def histograme(data,feature):
    
    plt.figure(figsize=(10, 6))
    sns.histplot(data[feature], kde=True, bins=25, color='blue', edgecolor='black')
    plt.title(f'Distribution of {feature}', fontsize=15)
    plt.xlabel(feature, fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True)
    st.pyplot(plt)

def boxplot(data,column):

    plt.figure(figsize=(12, 8))
    sns.boxplot(y=data[column])  
    plt.title(f'Box plot of {column}')  
    plt.ylabel(column)  
    plt.tight_layout()  

    st.pyplot(plt)  
