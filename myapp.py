import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

st.title("KMeans Clustering App")
st.subheader("Data Science")

st.sidebar.header("Upload CSV file or Use Sample")
user_exmaple = st.sidebar.checkbox("Use Example Data")
if user_exmaple:
  df = sns.load_dataset("iris")
  df = df.dropna()
  st.success("Load sample dataset: 'IRIS'")
else:
  uploaded_file = st.sidebar.file_uploader("Upload your CVS file", type=["csv"])
  if uploaded_file:
    df = pd.read_csv("uploaded_file")
  else:
    st.warning("Please upload a CSV file or Use the example dataset.")
    st.stop()

st.subheader("Dataset Preview")
st.write(df.head())




















