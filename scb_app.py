import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------------- Constants -----------------
MODULE_POWER_WP = 545.0        # Module Watt-peak
MODULE_VOC = 49.91             # Voc at STC
VMP_VOC_RATIO = 0.82           # Typical Vmp/Voc ratio
STRINGS_PER_SCB = 18           # Correct number of strings per SCB

VMP = MODULE_VOC * VMP_VOC_RATIO
I_MODULE_STC = MODULE_POWER_WP / VMP  # ≈13.31 A per module


# ----------------- Processing Function -----------------
def process_file(df):
    """
    Process SCB-level current data (1-min resolution):
    - Calculate expected SCB current from irradiance
    - Calculate CR for each SCB column
    """
    df = df.copy()
    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])  # First column is timestamp
    df = df.set_index(df.columns[0])               # Set datetime index

    irr_col = df.columns[0]                        # Irradiance column (after setting index)
    scb_cols = df.columns[1:]                      # All SCB current columns (B to S)

    irr = df[irr_col].astype(float)

    # Expected SCB current (same for all SCBs at each time)
    expected_str_current = I_MODULE_STC * (irr / 1000.0)
    expected_scb_current = expected_str_current * STRINGS_PER_SCB
    df["Expected_SCB_Current"] = expected_scb_current

    # Calculate CR for each SCB column
    for col in scb_cols:
        df[f"CR_{col}"] = np.where(expected_scb_current > 0,
                                   df[col].astype(float) / expected_scb_current,
                                   np.nan)

    return df, scb_cols


# ----------------- Styling Function -----------------
def highlight_low_cr(val):
    """
    Highlight Current Ratio (CR) below 0.9 in red, between 0.9-1 in orange.
    """
    if pd.isna(val):
        return ''
    if val < 0.9:
        return 'color: white; background-color: red;'
    elif val < 1.0:
        return 'color: black; background-color: orange;'
    return ''


# ----------------- Streamlit App -----------------
def main():
    st.title("🔍 SCB Current Analyzer (1-min data)")
    st.write("Upload SCB-level current data with irradiance for detailed analysis.")

    uploaded_file = st.file_uploader("Upload Excel/CSV file", type=["xlsx", "xls", "csv"])
    if uploaded_file is not None:
        # Read file
        if uploaded_file.name.endswith(".csv"):
            raw_df = pd.read_csv(uploaded_file)
        else:
            raw_df = pd.read_excel(uploaded_file)

        st.success("✅ File uploaded successfully!")
        st.write("### 📜 Raw Data Preview:")
        st.dataframe(raw_df.head())

        # Process data
        df_processed, scb_cols = process_file(raw_df)

        # Time Filter
        min_time = df_processed.index.min()
        max_time = df_processed.index.max()
        start_time, end_time = st.slider(
            "⏱️ Select Time Range",
            min_value=min_time.to_pydatetime(),
            max_value=max_time.to_pydatetime(),
            value=(min_time.to_pydatetime(), max_time.to_pydatetime()),
            format="HH:mm"
        )
        df_filtered = df_processed.loc[start_time:end_time]

        # Show Processed Data
        cr_cols = [f"CR_{col}" for col in scb_cols]
        styled_df = df_filtered[["Expected_SCB_Current"] + cr_cols].style.applymap(highlight_low_cr, subset=cr_cols)
        st.write(f"### 🔍 Processed CR Data ({len(df_filtered)} rows):")
        st.dataframe(styled_df)

        # Plot one SCB CR over time (choose SCB)
        st.subheader("📈 Current Ratio over Time")
        selected_scb = st.selectbox("Select SCB to Plot", scb_cols)
        plt.figure(figsize=(12, 5))
        plt.plot(df_filtered.index, df_filtered[f"CR_{selected_scb}"], label=f"CR_{selected_scb}", color="blue")
        plt.axhline(1.0, color="green", linestyle="--", label="Ideal")
        plt.axhline(0.9, color="red", linestyle="--", label="90% Threshold")
        plt.xlabel("Time")
        plt.ylabel("Current Ratio")
        plt.legend()
        plt.grid(True)
        st.pyplot(plt)

        # Download CSV
        csv = df_filtered.to_csv().encode('utf-8')
        st.download_button(
            label="📥 Download Processed CSV",
            data=csv,
            file_name="processed_scb_data.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
