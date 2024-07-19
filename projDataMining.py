import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from mpl_toolkits.mplot3d import Axes3D
from fonction import *

st.title("Projet Data Mining")

with st.sidebar:
    st.title("Pierre TAITHE, Benjamin ROUSSEAU, Madjid ZEHANI")
    st.header("Data Mining 2024")
    st.title ("DataMining Project")
    st.write("[Part I: Initial Data Exploration](#part-i-initial-data-exploration)")
    st.write("[Part II: Data Pre-processing and Cleaning](#part-ii-data-pre-processing-and-cleaning)")
    st.write("[Part III: Data Visualization](#part-iii-data-visualization)")
    st.write("[Part IV: Clustering or Prediction](#part-iv-clustering-or-prediction)")


st.header("Part I: Initial Data Exploration")
uploaded_file = st.file_uploader("Choose a file",)
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
    
    st.header("Part II: Data pre-processing and cleaning:")
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

    st.subheader("2. Data Normalization : ")
    st.write("Choose the method you want to use to normalize your data :")
    method2 = st.selectbox(" Please Select", ["No Normalization",
                                     "1/ Min-Max Normalization",
                                     "2/ Z-Score Normalization",
                                     "3/ Robust Normalization"])
    
    if method2 == "No Normalization":
        data_normalise = data_propre
    elif method2 == "1/ Min-Max Normalization":
        data_normalise = normalisation_minmax(data_propre)
    elif method2 == "2/ Z-Score Normalization":
        data_normalise = normalisation_Zscore(data_propre)
    elif method2 == "3/ Robust Normalization":
        data_normalise = normalisation_robust(data_propre)
    
    st.write("Normalized data : ")
    st.write(data_normalise.head(3))
    st.write(data_normalise.tail(3))


    st.header("Part III: Data Vizualisation")
    st.subheader("Histograms and Boxplots")
    numeric_df = data_normalise.select_dtypes(include=["float64", "int64"])
    column_names = numeric_df.columns.tolist()
    selected_column = st.selectbox("Veuillez sélectionner une colonne", column_names)
    vrai=st.checkbox("Normalized data", value=True)
    if vrai:
        st.write("Histogram of your parameter:")
        histograme(numeric_df,selected_column)
        st.write("Boxplot of your parameter:")
        boxplot(numeric_df,selected_column)
    else:
        st.write("Histogram of your parameter:")
        histograme(data_propre,selected_column)
        st.write("Boxplot of your parameter:")
        boxplot(data_propre,selected_column)
    
    st.header("Part IV : Clustering or Prediction")
    choice = st.selectbox("What do you want to do?", ["Clustering", "Prediction"])
    if choice == "Clustering":
        # Bouton pour exécuter le clustering
        method3 = st.selectbox(" Please Select", ["Kmeans",
                                        "DBSCAN"])
        
        if method3 == "Kmeans":
            n_clusters = st.number_input("nombre de cluster:", min_value=2, max_value=20, value=3)
            col1 = st.selectbox("selectionner l'axe X", numeric_df.columns, key="kmeans x")
            col2 = st.selectbox("selectionner l'axe Y", numeric_df.columns, key="kmeans y")
            centroids = kmeans_clustering(numeric_df,col1,col2,n_clusters)
            calculate_kmeans_stats(numeric_df, centroids)
            
        elif method3 == "DBSCAN":
            col1 = st.selectbox("selectionner l'axe X", numeric_df.columns, key="kmeans x")
            col2 = st.selectbox("selectionner l'axe Y", numeric_df.columns, key="kmeans y")
            eps = st.slider('EPS (epsilon)', min_value=0.1, max_value=1.0, value=0.5, step=0.1)
            min_samples = st.slider('Min Samples', min_value=2, max_value=20, value=5)
            dbscan_clustering(numeric_df,col1, col2, eps, min_samples)
            st.write("Clustering Algorithms")
            calculate_dbscan_stats(numeric_df, col1, col2)
    else : 
        st.subheader("Prediction Algorithms")
        column_names = data_normalise.columns.tolist()
        target_column = st.selectbox("Select the target column", column_names)

        algo_type = st.selectbox("Choose the type of prediction", ["Regression", "Classification"])

        if algo_type == "Regression":
            X = numeric_df.drop(columns=[target_column])
            y = numeric_df[target_column]
            algo_choice = st.selectbox("Choose the algorithm", ["Linear Regression", "Random Forest Regressor"])
            selected_columns = st.multiselect("Select the columns you want to use as features", column_names)
            X = numeric_df[selected_columns]
            if algo_choice == "Linear Regression":
                lineareg(X, y)

            elif algo_choice == "Random Forest Regressor":
                forestreg(X, y)
        
        elif algo_type == "Classification":
            X = numeric_df.drop(columns=[target_column])
            y = data_normalise[target_column]
            correlation_matrix = numeric_df.corr()
            st.write("Correlation Matrix")
            st.write(correlation_matrix)
            selected_columns = st.multiselect("Select the columns you want to use as features", column_names)
            X = numeric_df[selected_columns]
            correlation_matrix = numeric_df.corr()
            st.write("Correlation Matrix")
            algo_choice = st.selectbox("Choose the algorithm", ["Logistic Regression", "Random Forest Classifier", "K-Nearest Neighbors"])

            if algo_choice == "Logistic Regression":
                logisticreg(X, y)

            elif algo_choice == "Random Forest Classifier":
                forest(X,y)
            elif algo_choice == "K-Nearest Neighbors":
                knn(X, y)

    st.title("Part V: ")
    st.write("Choose the method between Kmeans 3D and DBSCAN 3D :")
    method4 = st.selectbox("Please Select", ["Kmeans3D", "DBSCAN3D"], key="method4")
 
    if method4 == "Kmeans3D":
        col1 = st.selectbox("Select X axis", numeric_df.columns, key="kmeans3D x")
        col2 = st.selectbox("Select Y axis", numeric_df.columns, key="kmeans3D y")
        col3 = st.selectbox("Select Z axis", numeric_df.columns, key="kmeans3D z")
        n_clusters = st.number_input("Number of clusters:", min_value=2, max_value=20, value=3, key="n_clusters")
        kmeans_clustering_3d(numeric_df, col1, col2, col3, n_clusters)
        #calculate_kmeans_stats(numeric_df, centroids2)
       
 
    elif method4 == "DBSCAN3D":
        col1 = st.selectbox("Select X axis", numeric_df.columns, key="dbscan3D x")
        col2 = st.selectbox("Select Y axis", numeric_df.columns, key="dbscan3D y")
        col3 = st.selectbox("Select Z axis", numeric_df.columns, key="dbscan3D z")
        eps = st.slider('EPS (epsilon)', min_value=0.1, max_value=1.0, value=0.5, step=0.1, key="eps")
        min_samples = st.slider('Min Samples', min_value=2, max_value=20, value=5, key="min_samples")
        dbscan_clustering_3d(numeric_df, col1, col2, col3, eps, min_samples)
        #calculate_dbscan_stats(numeric_df, col1, col2, col3)
