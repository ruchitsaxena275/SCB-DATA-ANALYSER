import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------------- Constants -----------------
MODULE_POWER_WP = 545.0        # Module Watt-peak
MODULE_VOC = 49.91             # Voc at STC
VMP_VOC_RATIO = 0.82           # Typical Vmp/Voc ratio
STRINGS_PER_SCB = 18           # Correct number of strings per SCB
IRRADIANCE_THRESHOLD = 500.0   # W/m² threshold for filtering
TEMP_COEFF = -0.05              # 0.5%/°C (adjustable)

VMP = MODULE_VOC * VMP_VOC_RATIO
I_MODULE_STC = MODULE_POWER_WP / VMP  # ≈13.02 A per module


# ----------------- Processing Function -----------------
def process_file(df):
    """Calculate Temperature-Corrected Expected SCB Current & CR for each SCB."""
    df = df.copy()
    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])  # First column is timestamp
    df = df.set_index(df.columns[0])               # Set datetime index

    # ✅ Explicitly select irradiance and temperature columns by NAME
    required_cols = ["Irradiation", "Temperature"]
    for col in required_cols:
        if col not in df.columns:
            st.error(f"❌ Missing required column: {col}. Please check your Excel file!")
            st.stop()

    irr = df["Irradiation"].astype(float)
    temp = df["Temperature"].astype(float)

    # 🔍 Filter out rows with irradiance <500 W/m²
    df = df[irr >= IRRADIANCE_THRESHOLD]
    irr = irr.loc[df.index]
    temp = temp.loc[df.index]

    # 🌡 Temperature correction factor
    temp_factor = 1 + TEMP_COEFF * (temp - 25)

    # 📐 Calculate corrected expected current per string
    expected_str_current = I_MODULE_STC * (irr / 1000.0) * temp_factor
    expected_scb_current = expected_str_current * STRINGS_PER_SCB
    df["Expected_SCB_Current"] = expected_scb_current

    # 🔢 Calculate CR for each SCB current column
    scb_cols = [c for c in df.columns if c not in required_cols]  # Exclude Irradiation & Temperature
    for col in scb_cols:
        df[f"CR_{col}"] = np.where(
            expected_scb_current > 0,
            df[col].astype(float) / expected_scb_current,
            np.nan
        )

    return df


# ----------------- Weak SCB Identification -----------------
def find_weak_scbs(df, threshold=0.90, min_fraction=0.7):
    """Return SCBs weak for at least 70% of time."""
    cr_cols = [col for col in df.columns if col.startswith("CR_")]
    weak_counts = {}

    for col in cr_cols:
        total = len(df)
        weak = (df[col] < threshold).sum()
        if total > 0 and (weak / total) >= min_fraction:
            weak_counts[col] = round((weak / total) * 100, 2)

    weak_df = pd.DataFrame(list(weak_counts.items()), columns=["SCB", "Weak_%"])
    return weak_df


# ----------------- Weak String Color Tagging -----------------
def tag_weak_strings(df):
    """Add a Weakness Tag column for CSV export."""
    tag_df = df.copy()
    cr_cols = [col for col in tag_df.columns if col.startswith("CR_")]

    for col in cr_cols:
        tag_df[col] = np.select(
            [tag_df[col] < 0.9, tag_df[col] < 1.0],
            ["RED", "ORANGE"],
            default="OK"
        )
    return tag_df


# ----------------- Streamlit App -----------------
def main():
    st.title("🔍 SCB Current Analyzer with Temperature-Corrected Expected Current")
    st.write("Upload SCB-level current data with columns named *Irradiation* & *Temperature*. Rows with irradiance <500 W/m² are filtered out for accuracy.")

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

        if df_processed.empty:
            st.warning("⚠ No data available after filtering irradiance <500 W/m²!")
            return

        # Time Filter
        min_time = df_processed.index.min()
        max_time = df_processed.index.max()
        start_time, end_time = st.slider(
            "⏱ Select Time Range",
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

        # Weak SCBs
        weak_df = find_weak_scbs(df_filtered)

        st.write("### 🚨 Weak SCBs (CR < 0.90 for ≥70% of time):")
        st.dataframe(weak_df)

        # Download Processed Data
        csv_processed = df_filtered.to_csv(index=True).encode('utf-8')
        st.download_button(
            label="📥 Download Processed Data (Full Calculations)",
            data=csv_processed,
            file_name="processed_scb_data.csv",
            mime="text/csv",
        )

        # Download Weak-Tagged CSV
        csv_tagged = tagged_df.to_csv(index=True).encode('utf-8')
        st.download_button(
            label="📥 Download Weak-Tagged CSV",
            data=csv_tagged,
            file_name="weak_strings_report.csv",
            mime="text/csv",
        )

        # Download Weak SCB List
        csv_weak = weak_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Weak SCB List (CR<0.90, ≥70% Time)",
            data=csv_weak,
            file_name="weak_scb_summary.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()






