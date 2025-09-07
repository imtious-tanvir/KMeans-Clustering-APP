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

#kMean model training
st.subheader("KMeans Model Training")
n_clusters = st.slider("Select number of clusters (k)", min_value = 2, max_value = 10, step = 1, value = 3)
model = KMeans(n_clusters = n_clusters, random_state = 42)
model.fit(df)
labels = model.labels_
df_clustered = df.copy()
df_clustered['Cluster'] = labels
st.success("KMeans clustering complete")

st.subheader("After Training Dataset")
st.write(df_clustered.head())

st.subheader("Cluster Centers (Orginal scale)")
st.write(pd.DataFrame(model.cluster_centers_, columns = features))

#ploting
if len(features) >= 2:
    st.subheader("Cluster Visualisation")
    plt.figure()
    scatter = plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=labels, cmap='viridis')
    plt.xlabel(features[0])
    plt.ylabel(features[1])
    elements = scatter.legend_elements()
    handles = elements[0]
    labels_list = elements[1]
    plt.legend(handles, labels_list, title = 'Clusters')
    st.pyplot(plt)
else:
    st.info("Select at least two features to view scatter plot.")

#predict
st.subtitle("Predict Cluster Nor New Input")
input_data = {}
valid_input = True
for feature in features:
    user_input = st.text_input(f"Enter {feature} (numeric value)")
    try:
        if user_input.strip() == "":
            valid_input = False
        else:
            input_data[feature] = float(user_input)
    except ValueError:
        valid_input = False

#button
if.button("Predict Cluster"):
    if valid_input:
        input_df = pd.DataFrame([input_data])[features]
        cluster_pred = model.predict(input_df)
        st.success(f"The new input belongs to Cluster: {cluster_pred}")
    else:
        st.error("Please enter valid numeric values for all features before predicting.")









