import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------
# CONFIG
# --------------------
st.set_page_config(page_title="SCB Weak String Analyzer", layout="wide")

# --------------------
# FILE PROCESSING
# --------------------
def process_file(df):
    df = df.copy()

    # Drop completely empty columns
    df = df.dropna(axis=1, how='all')

    # Detect irradiation column (assume first column with values > 100 is irradiation)
    irr_col = None
    for col in df.columns:
        if df[col].dtype != 'object' and df[col].max() > 100:
            irr_col = col
            break

    if irr_col is None:
        st.error("Irradiation column not found. Please check file.")
        return None

    # Assume all numeric columns except irradiation are string currents
    current_cols = [col for col in df.columns if col != irr_col and pd.api.types.is_numeric_dtype(df[col])]

    # Calculate expected current based on irradiation (scaling factor adjustable)
    df["Expected_Current"] = df[irr_col] / 100  # Assume 10 A at 1000 W/m2

    # Compute Current Ratio for each string
    for col in current_cols:
        df[f"{col}_CR"] = df[col] / df["Expected_Current"]

    return df, irr_col, current_cols

# --------------------
# WEAK SCB DETECTION
# --------------------
def detect_weak_scb(df, current_cols, threshold=0.9):
    weak_summary = {}
    weak_flags = pd.DataFrame(index=df.index)

    for col in current_cols:
        cr_col = f"{col}_CR"
        weak_flags[col] = df[cr_col] < threshold
        weak_summary[col] = weak_flags[col].any()  # Weak if weak at least once

    return weak_summary, weak_flags

# --------------------
# HEATMAP
# --------------------
def plot_heatmap(df, current_cols):
    cr_cols = [f"{col}_CR" for col in current_cols]
    cr_data = df[cr_cols]
    cr_data.columns = current_cols  # Simplify labels

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(cr_data.T, cmap="coolwarm", center=1.0, cbar_kws={'label': 'Current Ratio'}, ax=ax)
    ax.set_xlabel("Time/Index")
    ax.set_ylabel("String")
    ax.set_title("Current Ratio Heatmap")
    st.pyplot(fig)

# --------------------
# MAIN APP
# --------------------
def main():
    st.title("🔍 SCB Weak String Analyzer")

    uploaded_file = st.file_uploader("Upload your SCB Excel file", type=["xlsx", "xls", "csv"])
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".csv"):
            raw_df = pd.read_csv(uploaded_file)
        else:
            raw_df = pd.read_excel(uploaded_file)

        st.success("File uploaded successfully!")

        result = process_file(raw_df)
        if result is None:
            return

        df, irr_col, current_cols = result

        st.subheader("Data Preview")
        st.dataframe(df.head())

        threshold = st.slider("Weakness Threshold (CR)", 0.5, 1.0, 0.9, 0.01)

        weak_summary, weak_flags = detect_weak_scb(df, current_cols, threshold)

        st.subheader("Weak SCB Summary")
        weak_df = pd.DataFrame.from_dict(weak_summary, orient='index', columns=["Weak"])
        st.dataframe(weak_df)

        st.subheader("Heatmap")
        plot_heatmap(df, current_cols)

        st.download_button(
            "Download Weak SCB Report",
            weak_df.to_csv().encode("utf-8"),
            file_name="weak_scb_summary.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
