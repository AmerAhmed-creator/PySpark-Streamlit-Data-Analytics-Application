import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

st.set_page_config(page_title="Universal Data Tool", layout="wide")

st.title("Dataset Preprocessor & EDA Tool")

# Session state
if "raw_df" not in st.session_state:
    st.session_state.raw_df = None

if "processed_df" not in st.session_state:
    st.session_state.processed_df = None



# PAGE 1 — UPLOAD
page = st.sidebar.selectbox("Navigation", ["Upload Dataset", "Preprocessing", "EDA"])

if page == "Upload Dataset":
    st.header("Upload CSV Dataset")
    uploaded = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        st.session_state.raw_df = df.copy()
        st.session_state.processed_df = df.copy()

        st.success("Dataset loaded!")

        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        st.write(df.describe(include="all"))



# PAGE 2 — PREPROCESSING 
if page == "Preprocessing":
    st.header("Interactive Data Preprocessing (Undo/Redo Enabled)")

    if st.session_state.raw_df is None:
        st.warning("Upload dataset first.")
        st.stop()

    # Always start from original → apply selected operations → preview
    df = st.session_state.raw_df.copy()

    st.subheader("Choose processing steps")

    # Checkboxes for operations
    convert_to_numeric = st.checkbox("Convert selected columns to numeric")
    convert_to_categorical = st.checkbox("Convert selected columns to categorical")
    fill_missing = st.checkbox("Fill missing values")
    normalize_cols = st.checkbox("Normalize selected numeric columns")
    label_encode_cols = st.checkbox("Label encode selected categorical columns")
    drop_cols = st.checkbox("Drop selected columns")

    # Convert to Numeric
    if convert_to_numeric:
        col = st.selectbox("Select column to convert to numeric", df.columns)
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Convert to Categorical
    if convert_to_categorical:
        col = st.selectbox("Select column to convert to categorical", df.columns)
        df[col] = df[col].astype("category")

    # Fill Missing
    if fill_missing:
        col = st.selectbox("Column to fill", df.columns, key="fillcol")
        method = st.radio("Method", ["Mean", "Median", "Mode", "Custom"], key="fillmethod")

        if method == "Mean":
            df[col] = df[col].fillna(df[col].mean())
        elif method == "Median":
            df[col] = df[col].fillna(df[col].median())
        elif method == "Mode":
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            custom = st.text_input("Enter custom value", "0", key="customfill")
            df[col] = df[col].fillna(custom)

    # Normalize
    if normalize_cols:
        numeric = df.select_dtypes(include=np.number).columns
        cols = st.multiselect("Select numeric columns to normalize", numeric, key="normcols")

        if cols:
            scaler = MinMaxScaler()
            df[cols] = scaler.fit_transform(df[cols])

    # Label Encode
    if label_encode_cols:
        cats = df.select_dtypes(include="object").columns
        cols = st.multiselect("Select categorical columns to encode", cats, key="labenc")

        if cols:
            enc = LabelEncoder()
            for c in cols:
                df[c] = enc.fit_transform(df[c].astype(str))

    # Drop Columns
    if drop_cols:
        cols = st.multiselect("Select columns to drop", df.columns, key="dropcols")
        if cols:
            df.drop(columns=cols, inplace=True)

    # Update processed_df
    st.session_state.processed_df = df.copy()

    st.subheader("Preview")
    st.dataframe(df.head())



# PAGE 3 — EDA 
if page == "EDA":
    st.header("Exploratory Data Analysis")

    if st.session_state.processed_df is None:
        st.warning("Upload and preprocess dataset first.")
        st.stop()

    df = st.session_state.processed_df.copy()

    plot_type = st.selectbox(
        "Choose plot type",
        ["Histogram", "Scatter Plot", "Bar Plot", "Box Plot", "Heatmap", "Pairplot"]
    )

    numeric = df.select_dtypes(include=np.number).columns.tolist()
    categoricals = df.select_dtypes(include="object").columns.tolist()
    all_cols = df.columns.tolist()

    # figure size
    fig_size = (12, 8)
    
    # Histogram
    if plot_type == "Histogram":
        col = st.selectbox("Numeric column", numeric)
        fig, ax = plt.subplots(figsize=fig_size)
        sns.histplot(df[col], kde=True, ax=ax)
        plt.xticks(rotation=90)
        st.pyplot(fig)

    # Scatter Plot
    elif plot_type == "Scatter Plot":
        x = st.selectbox("X-axis", all_cols)
        y = st.selectbox("Y-axis (numeric)", numeric)
        fig, ax = plt.subplots(figsize=fig_size)
        ax.scatter(df[x], df[y], alpha=0.5)
        plt.xticks(rotation=90)
        st.pyplot(fig)

    # Bar Plot
    elif plot_type == "Bar Plot":
        cat = st.selectbox("Categorical column", categoricals)
        num = st.selectbox("Numeric column", numeric)
        fig, ax = plt.subplots(figsize=fig_size)
        sns.barplot(x=df[cat], y=df[num], ax=ax)
        plt.xticks(rotation=90)
        st.pyplot(fig)

    # Box Plot
    elif plot_type == "Box Plot":
        col = st.selectbox("Numeric column", numeric)
        fig, ax = plt.subplots(figsize=fig_size)
        sns.boxplot(y=df[col], ax=ax)
        plt.xticks(rotation=90)
        st.pyplot(fig)

    # Heatmap
    elif plot_type == "Heatmap":
        fig, ax = plt.subplots(figsize=(6, 4))  
        sns.heatmap(df[numeric].corr(), annot=True, cmap="coolwarm", ax=ax)
        plt.xticks(rotation=90)
        st.pyplot(fig)

    # Pairplot
    elif plot_type == "Pairplot":
        cols = st.multiselect("Select numeric columns", numeric)
        if cols:
            fig = sns.pairplot(df[cols], height=1.5)
            plt.xticks(rotation=90)
            st.pyplot(fig)
