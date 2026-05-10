import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.set_page_config(page_title="Data Explorer", layout="wide")

st.title("📊 Data Explorer Tool")

# ---------- File Upload ----------
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📌 Dataset Overview")
    st.write("Shape:", df.shape)
    st.dataframe(df.head())

    # ---------- Data Info ----------
    st.subheader("🔍 Data Info")
    st.write(df.describe())
    st.write("Missing Values:")
    st.write(df.isnull().sum())

    # ---------- Select Numeric Columns ----------
    numerical_df = df.select_dtypes(include=[np.number])

    # ---------- Correlation ----------
    st.subheader("📈 Correlation Matrix")
    if not numerical_df.empty:
        corr = numerical_df.corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, ax=ax)
        st.pyplot(fig)
    else:
        st.warning("No numeric columns found.")

    # ---------- Column Selection ----------
    st.sidebar.header("🎛️ Controls")

    numeric_cols = numerical_df.columns.tolist()

    if len(numeric_cols) > 0:
        col = st.sidebar.selectbox("Select column for histogram", numeric_cols)

        # Histogram
        st.subheader(f"📊 Histogram of {col}")
        fig, ax = plt.subplots()
        df[col].hist(ax=ax, bins=20)
        st.pyplot(fig)

        # Boxplot
        st.subheader(f"📦 Boxplot of {col}")
        fig, ax = plt.subplots()
        df.boxplot(column=col, ax=ax)
        st.pyplot(fig)

    # Scatter plot
    if len(numeric_cols) >= 2:
        col1 = st.sidebar.selectbox("X-axis", numeric_cols, index=0)
        col2 = st.sidebar.selectbox("Y-axis", numeric_cols, index=1)

        st.subheader(f"🔵 Scatter Plot: {col1} vs {col2}")
        fig, ax = plt.subplots()
        ax.scatter(df[col1], df[col2])
        ax.set_xlabel(col1)
        ax.set_ylabel(col2)
        st.pyplot(fig)

    # ---------- Insights ----------
    st.subheader("🧠 Auto Insights")

    missing = df.isnull().sum()
    for col in missing.index:
        if missing[col] > 0:
            st.warning(f"{col} has {missing[col]} missing values")

    if not numerical_df.empty:
        corr_matrix = numerical_df.corr()
        for i in corr_matrix.columns:
            for j in corr_matrix.columns:
                if i != j and abs(corr_matrix.loc[i, j]) > 0.8:
                    st.success(f"High correlation between {i} and {j}: {corr_matrix.loc[i, j]:.2f}")

    # ---------- Download Report ----------
    st.subheader("📄 Download Report")
    csv = df.describe().to_csv().encode('utf-8')
    st.download_button("Download Summary CSV", csv, "report.csv", "text/csv")

else:
    st.info("Please upload a CSV file to begin.")