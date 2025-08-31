import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------- Constants -----------------
MODULE_POWER_WP = 545.0        # Module Watt-peak
MODULE_VOC = 49.91             # Voc at STC
VMP_VOC_RATIO = 0.82           # Typical Vmp/Voc ratio
NUM_STRINGS = 18               # Strings per SCB

VMP = MODULE_VOC * VMP_VOC_RATIO
I_MODULE_STC = MODULE_POWER_WP / VMP  # ≈13.31 A per module
CR_LOW_THRESHOLD = 0.90


# ----------------- Processing Function -----------------
def process_file(df):
    """
    Process raw SCB data (1-min resolution) and calculate:
    - Expected current per string
    - Current Ratio (CR) per string
    - Expected & measured SCB total current
    """
    df = df.copy()
    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])  # First column is timestamp
    df = df.set_index(df.columns[0])               # Set datetime index

    measured_cols = df.columns[0:NUM_STRINGS]      # String currents (B..S)
    irr_col = df.columns[23]                       # Irradiance (Y)

    measured = df[measured_cols].astype(float)
    irr = df[irr_col].astype(float)

    expected_str_current = I_MODULE_STC * (irr / 1000.0)

    result = measured.copy()
    for i, col in enumerate(measured_cols, start=1):
        result[f"Expected_String_{i}"] = expected_str_current
        result[f"CR_String_{i}"] = np.where(expected_str_current > 0,
                                            measured[col] / expected_str_current,
                                            np.nan)

    result["Expected_SCB_Current"] = expected_str_current * NUM_STRINGS
    result["Measured_SCB_Current"] = measured.sum(axis=1)
    result["Irradiance_Wm2"] = irr

    return result


# ----------------- Streamlit App -----------------
def main():
    st.title("SCB String Current Analyzer (1-min data)")
    st.write("Upload your SCB Excel file (1-minute data) to analyze weak strings.")

    uploaded_file = st.file_uploader("Upload Excel/CSV file", type=["xlsx", "xls", "csv"])
    if uploaded_file is not None:
        # Read data
        if uploaded_file.name.endswith(".csv"):
            raw_df = pd.read_csv(uploaded_file)
        else:
            raw_df = pd.read_excel(uploaded_file)

        st.success("File uploaded successfully!")
        st.write("### Raw Data Preview:")
        st.dataframe(raw_df.head())

        # Process data
        df_processed = process_file(raw_df)

        # Time Range Filter
        min_time = df_processed.index.min()
        max_time = df_processed.index.max()
        start_time, end_time = st.slider(
            "Select Time Range",
            min_value=min_time.to_pydatetime(),
            max_value=max_time.to_pydatetime(),
            value=(min_time.to_pydatetime(), max_time.to_pydatetime()),
            format="HH:mm"
        )
        df_filtered = df_processed.loc[start_time:end_time]

        st.write(f"### Processed Data ({len(df_filtered)} rows):")
        st.dataframe(df_filtered.head())

        # Heatmap for Current Ratio
        st.subheader("Current Ratio Heatmap (CR per string)")
        cr_cols = [col for col in df_filtered.columns if col.startswith("CR_String_")]
        cr_data = df_filtered[cr_cols]

        plt.figure(figsize=(15, 6))
        sns.heatmap(cr_data.T, cmap="coolwarm", cbar=True, vmin=0.7, vmax=1.1)
        plt.title("Current Ratio (CR) Heatmap per String")
        plt.xlabel("Time")
        plt.ylabel("Strings")
        st.pyplot(plt)

        # Download button
        csv = df_filtered.to_csv().encode('utf-8')
        st.download_button(
            label="Download Processed CSV",
            data=csv,
            file_name="processed_scb_data.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
