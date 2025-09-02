import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.title("SCB Current Analysis Dashboard")

# File Upload
uploaded_file = st.file_uploader("Upload your SCB data Excel", type=["xlsx", "csv"])

if uploaded_file:
    # Load data
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    st.write("### Raw Data")
    st.dataframe(df.head())

    # Extract SCB columns
    timestamp_col = df.columns[0]
    current_cols = df.columns[1:19]  # Assuming B to S (18 strings)
    irr_col = df.columns[24]         # Assuming Y is irradiation

    # Temperature correction (optional)
    temp_coeff = st.number_input("Enter temperature coefficient (%/°C)", value=-0.29) / 100
    module_temp = st.number_input("Enter module temperature (°C)", value=25.0)
    
    expected_current = (df[irr_col] / 1000) * 13  # adjust based on Isc at STC
    temp_factor = 1 + temp_coeff * (module_temp - 25)
    expected_current = expected_current * temp_factor
    
    df_expected = df.copy()
    df_expected["Expected_Current"] = expected_current
    
    # Compute Performance Ratio
    for col in current_cols:
        df_expected[col + "_PR"] = df_expected[col] / df_expected["Expected_Current"]
    
    # Weak String Detection
    weak_threshold = st.slider("Weak String PR Threshold", 0.5, 1.0, 0.9)
    weak_strings = {}
    for col in current_cols:
        weak_count = (df_expected[col + "_PR"] < weak_threshold).sum()
        weak_strings[col] = weak_count
    weak_df = pd.DataFrame.from_dict(weak_strings, orient='index', columns=['Weak Count'])
    
    st.write("### Weak Strings Summary")
    st.dataframe(weak_df.sort_values("Weak Count", ascending=False))

    # Heatmap
    st.write("### Current Heatmap")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(df[current_cols].T, cmap="coolwarm", cbar_kws={'label': 'Current (A)'})
    plt.ylabel("Strings")
    plt.xlabel("Sample Index")
    st.pyplot(fig)

    # Scatter Plot
    st.write("### Irradiance vs Current Scatter")
    selected_string = st.selectbox("Select String for Scatter Plot", current_cols)
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.scatter(df[irr_col], df[selected_string], alpha=0.5)
    ax2.set_xlabel("Irradiance (W/m²)")
    ax2.set_ylabel(f"{selected_string} Current (A)")
    st.pyplot(fig2)
