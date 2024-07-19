import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
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
    cm = confusion_matrix(y_test, y_pred)
 
    st.subheader('Confusion Matrix')
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    st.pyplot(plt)

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
    cm = confusion_matrix(y_test, y_pred)
 
    st.subheader('Confusion Matrix')
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    st.pyplot(plt)

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
    cm = confusion_matrix(y_test, y_pred)
 
    st.subheader('Confusion Matrix')
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    st.pyplot(plt)

def kmeans_clustering(data, col1, col2, n_clusters):
    X = data[[col1, col2]]
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X)
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    data['cluster'] = labels
    plt.figure(figsize=(10, 8))
    for cluster in range(n_clusters):
        cluster_data = data[data['cluster'] == cluster]
        plt.scatter(cluster_data[col1], cluster_data[col2], label=f'Cluster {cluster + 1}')
    
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', color='k', label='Centroids')
    
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.title('K-Means Clustering')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)
    return centroids

def dbscan_clustering(data, col1, col2, eps, min_samples):
    
    X = data[[col1, col2]]
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(X)
    labels = dbscan.labels_
    data['cluster'] = labels
    

    plt.figure(figsize=(10, 8))
    unique_clusters = np.unique(labels[labels != -1])
    
    for cluster in unique_clusters:
        cluster_data = data[data['cluster'] == cluster]
        plt.scatter(cluster_data[col1], cluster_data[col2], label=f'Cluster {cluster}')
    
    noise_data = data[data['cluster'] == -1]
    plt.scatter(noise_data[col1], noise_data[col2], color='gray', marker='x', label='Noise')
    
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.title('DBSCAN Clustering')
    plt.legend()
    plt.grid(True)
    
    st.pyplot(plt)

def calculate_kmeans_stats(data, centroids):
    cluster_stats = pd.DataFrame(columns=['Cluster', 'Number of Points', 'Center'])
    unique_clusters = np.unique(data['cluster'])
   
    for cluster in unique_clusters:
        cluster_data = data[data['cluster'] == cluster]
        center = centroids[cluster]
        num_points = cluster_data.shape[0]
        new_row = pd.DataFrame({
            'Cluster': [cluster + 1],
            'Number of Points': [num_points],
            'Center': [center]
        })
        cluster_stats = pd.concat([cluster_stats, new_row], ignore_index=True)
   
    st.subheader('Cluster Statistics (K-Means)')
    st.write(cluster_stats)
 
 
def calculate_dbscan_stats(data, col1, col2):
    cluster_stats = pd.DataFrame(columns=['Cluster', 'Number of Points', 'Density'])
    unique_clusters, counts = np.unique(data['cluster'], return_counts=True)
   
    for cluster, count in zip(unique_clusters, counts):
        if cluster == -1:
            new_row = pd.DataFrame({
                'Cluster': ['Noise (DBSCAN)'],
                'Number of Points': [count],
                'Density': [np.nan]
            })
        else:
            cluster_data = data[data['cluster'] == cluster]
            num_points = count
            area = (cluster_data[col1].max() - cluster_data[col1].min()) * (cluster_data[col2].max() - cluster_data[col2].min())
            density = num_points / area if area > 0 else np.nan
            new_row = pd.DataFrame({
                'Cluster': [cluster],
                'Number of Points': [num_points],
                'Density': [density]
            })
        cluster_stats = pd.concat([cluster_stats, new_row], ignore_index=True)
   
    st.subheader('Cluster Statistics (DBSCAN)')
    st.write(cluster_stats)
 
 
def dbscan_clustering_3d(data, col1, col2, col3, eps, min_samples):
    X = data[[col1, col2, col3]]
   
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
   
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X_scaled)
   
    data['cluster'] = labels
 
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
   
    unique_clusters = np.unique(labels)
    for cluster in unique_clusters:
        if cluster == -1:
            cluster_label = 'Noise'
        else:
            cluster_label = f'Cluster {cluster + 1}'
       
        cluster_data = data[data['cluster'] == cluster]
        ax.scatter(cluster_data[col1], cluster_data[col2], cluster_data[col3], label=cluster_label, s=50)
   
    ax.set_xlabel(col1)
    ax.set_ylabel(col2)
    ax.set_zlabel(col3)
    ax.set_title('DBSCAN Clustering')
    ax.legend()
    ax.grid(True)
   
    st.pyplot(fig)
 
    return data
 
def kmeans_clustering_3d(data, col1, col2, col3, n_clusters):
    X = data[[col1, col2, col3]]
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X)
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    data['cluster'] = labels
 
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    for cluster in range(n_clusters):
        cluster_data = data[data['cluster'] == cluster]
        ax.scatter(cluster_data[col1], cluster_data[col2], cluster_data[col3], label=f'Cluster {cluster + 1}')
   
    ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], marker='x', color='k', label='Centroids')
   
    ax.set_xlabel(col1)
    ax.set_ylabel(col2)
    ax.set_zlabel(col3)
    ax.set_title('K-Means Clustering')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
   
    return centroids, labels