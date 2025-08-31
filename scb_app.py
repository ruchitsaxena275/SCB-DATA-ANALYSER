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
    - Expected SCB current
    - CR for all SCBs
    """
    df = df.copy()
    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])  # First column is timestamp
    df = df.set_index(df.columns[0])               # Set datetime index

    irr_col = df.columns[-1]  # Last column is irradiance
    irr = df[irr_col].astype(float)

    expected_str_current = I_MODULE_STC * (irr / 1000.0)
    expected_scb_current = expected_str_current * STRINGS_PER_SCB
    df["Expected_SCB_Current"] = expected_scb_current

    # Calculate CR for each SCB current column (B to S)
    scb_cols = df.columns[:-2]  # All but last (irradiance) and Expected
    for col in scb_cols:
        df[f"CR_{col}"] = np.where(expected_scb_current > 0,
                                   df[col].astype(float) / expected_scb_current,
                                   np.nan)

    return df


# ----------------- Weak String Color Tagging -----------------
def tag_weak_strings(df):
    """
    Add a Weakness Tag column for CSV export.
    """
    tag_df = df.copy()
    cr_cols = [col for col in tag_df.columns if col.startswith("CR_")]

    for col in cr_cols:
        tag_df[col] = np.select(
            [
                tag_df[col] < 0.9,
                tag_df[col] < 1.0
            ],
            ["RED", "ORANGE"],
            default="OK"
        )
    return tag_df


# ----------------- Streamlit App -----------------
def main():
    st.title("🔍 SCB Current Analyzer with Weak String Tagging")
    st.write("Upload SCB-level current data with irradiance. Generates CR and highlights weak strings in CSV.")

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

        # Process
        df_processed = process_file(raw_df)

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

        st.write(f"### 🔍 Processed Data ({len(df_filtered)} rows):")
        st.dataframe(df_filtered.head())

        # Plot
        st.subheader("📈 Current Ratio Trends")
        cr_cols = [c for c in df_filtered.columns if c.startswith("CR_")]
        plt.figure(figsize=(12, 5))
        for col in cr_cols:
            plt.plot(df_filtered.index, df_filtered[col], alpha=0.5, label=col)
        plt.axhline(1.0, color="green", linestyle="--")
        plt.axhline(0.9, color="red", linestyle="--")
        plt.xlabel("Time")
        plt.ylabel("CR")
        plt.legend()
        plt.grid(True)
        st.pyplot(plt)

        # Add Weak Tagging
        tagged_df = tag_weak_strings(df_filtered)

        # Download CSV
        csv = tagged_df.to_csv(index=True).encode('utf-8')
        st.download_button(
            label="📥 Download Weak-Tagged CSV",
            data=csv,
            file_name="weak_strings_report.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
