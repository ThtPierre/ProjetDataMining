import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from mpl_toolkits.mplot3d import Axes3D


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

def lineareg(X, y):
    test_size = st.slider("Test size (percentage)", 10, 50, 20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    st.write("Mean Squared Error:", mse)
    st.write("R-squared:", r2)
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
    ax.set_xlabel("Measured")
    ax.set_ylabel("Predicted")
    st.pyplot(fig)

def forestreg(X,y):
    n_tree = st.slider("Number of trees", 10, 100, 50)
    test_size = st.slider("Test size (percentage)", 10, 50, 20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)
    model = RandomForestRegressor(n_estimators=n_tree, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    st.write("Mean Squared Error:", mse)
    st.write("R-squared:", r2)
    
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
    ax.set_xlabel("Measured")
    ax.set_ylabel("Predicted")
    st.pyplot(fig)

def logisticreg(X, y):
    test_size = st.slider("Test size (percentage)", 10, 50, 20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write("Accuracy:", accuracy)
    correct_predictions = np.sum(y_pred == y_test)
    incorrect_predictions = len(y_test) - correct_predictions
    st.write("Number of correct predictions:", correct_predictions)
    st.write("Number of incorrect predictions:", incorrect_predictions)

def forest(X,y): 
    n_estimators = st.slider("Number of trees", 10, 100, 50)
    test_size = st.slider("Test size (percentage)", 10, 50, 20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)                
    st.write("Accuracy:", accuracy)
    correct_predictions = np.sum(y_pred == y_test)
    incorrect_predictions = len(y_test) - correct_predictions
    
    st.write("Number of correct predictions:", correct_predictions)
    st.write("Number of incorrect predictions:", incorrect_predictions)

def knn(X,y):
    k = st.slider("Number of neighbors (K)", 1, 20, 5)
    test_size = st.slider("Test size (percentage)", 10, 50, 20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    st.write("Accuracy:", accuracy)
    correct_predictions = np.sum(y_pred == y_test)
    incorrect_predictions = len(y_test) - correct_predictions
    
    st.write("Number of correct predictions:", correct_predictions)
    st.write("Number of incorrect predictions:", incorrect_predictions)
