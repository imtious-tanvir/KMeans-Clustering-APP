import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Streamlit title and subtitle
st.title("KMeans Clustering App")
st.subheader("Data Science Project")

# Sidebar for file upload or example data
st.sidebar.header("Upload CSV file or Use Sample")
user_example = st.sidebar.checkbox("Use Example Data")

if user_example:
    df = sns.load_dataset("iris")
    df = df.dropna()
    st.success("Loaded sample dataset: 'IRIS'")
else:
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("CSV file uploaded successfully!")
    else:
        st.warning("Please upload a CSV file or use the example dataset.")
        st.stop()

# Show dataset preview
st.subheader("Dataset Preview")
st.write(df.head())





