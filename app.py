import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

st.title("Credit Card Clustering App")

uploaded_file = st.file_uploader("Upload your CSV", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    num_cols = df.select_dtypes(include=['float64','int64']).columns
    X_scaled = StandardScaler().fit_transform(df[num_cols])
    
    model = pickle.load(open("best_creditcard_cluster.pkl", "rb"))
    labels = model.predict(X_scaled)
    
    df["Cluster"] = labels
    st.write("Clustered Data:")
    st.dataframe(df)
    
    st.write("Cluster Counts:")
    st.bar_chart(df["Cluster"].value_counts())