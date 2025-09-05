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

st.subheader("Data Preprocessing")
numeric_col = df.select_dtypes(include = np.number).columns.tolist()
if len(numeric_col) < 2:
    st.error("Need at least two numeric columns for clustering.")
    st.stop()

features = st.multiselect("Select feature columns for clustering", numeric_col, default = numeric_col)
if len(features) == 0:
    st.write("Please select at least one feature.")
    st.stop()

#drop missing values
df = df[features].dropna()

#elbow method
st.subheader("Find Optimal Number of Clusters (Elbow Method)")
max_k = st.slider("Maximum number of clusters to test", min_value = 2, max_value = 10, step = 1, value = 10)
wcss = []
for k in range(1, max_k + 1):
    kmeans = KMeans(n_clusters = k, random_state = 42)
    kmeans.fit(df)
    wcss.append(kmeans.inertia_)
fig_elbow, ax_elbow = plt.subplots()
ax_elbow.plot(range(1, max_k + 1), wcss, marker = 'o')
ax_elbow.set_xlabel("Number of Clusters (K)")
ax_elbow.set_ylabel("Elbow Method For Optimal K")
st.pyplot(fig_elbow)














