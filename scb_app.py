import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Constants ---
MODULE_VOC = 49.91   # V
MODULE_PMP = 545     # W
MODULE_VMP = MODULE_VOC * 0.82
MODULE_IMP = MODULE_PMP / MODULE_VMP  # Module current at STC
STRINGS_PER_SCB = 18

# --- Streamlit App ---
st.title("SCB Current Ratio Analyzer (Minute-wise)")

uploaded_file = st.file_uploader("Upload Excel file with SCB current & irradiance", type=["xlsx", "xls"])

if uploaded_file is not None:
    # Read Excel
    df = pd.read_excel(uploaded_file)
    
    # Expect columns: Timestamp, Irradiance, SCB Current
    df.columns = [col.strip() for col in df.columns]  # Clean column names
    
    # Ensure timestamp is datetime
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.sort_values(by='Timestamp')
    df.set_index('Timestamp', inplace=True)
    
    # Compute Expected SCB Current
    df['Expected_SCB_Current'] = MODULE_IMP * (df['Irradiance'] / 1000) * STRINGS_PER_SCB
    
    # Compute Current Ratio
    df['CR'] = df['SCB_Current'] / df['Expected_SCB_Current']
    df['CR'] = df['CR'].replace([np.inf, -np.inf], np.nan)  # Clean invalids
    
    st.success("✅ Processed data with minute-wise CR")
    st.dataframe(df.head(20))
    
    # Plot CR over time
    st.subheader("Current Ratio over Time")
    plt.figure(figsize=(12, 5))
    plt.plot(df.index, df['CR'], label='Current Ratio', marker='o', markersize=2)
    plt.axhline(1.0, color='red', linestyle='--', label='Ideal (1.0)')
    plt.xlabel("Time")
    plt.ylabel("CR")
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)
    
    # Download processed CSV
    csv = df.to_csv()
    st.download_button("Download Processed Data (CSV)", data=csv, file_name="processed_scb_data.csv", mime="text/csv")

else:
    st.info("Please upload an Excel file to start analysis.")
