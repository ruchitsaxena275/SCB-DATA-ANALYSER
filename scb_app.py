import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------------- Constants -----------------
MODULE_POWER_WP = 545.0        # Module watt-peak
MODULE_VOC = 49.91             # Voc at STC
VMP_VOC_RATIO = 0.82
STRINGS_PER_SCB = 18

VMP = MODULE_VOC * VMP_VOC_RATIO
I_MODULE_STC = MODULE_POWER_WP / VMP  # ≈13.31 A per module


# ----------------- Processing -----------------
def process_file(df):
    """
    Process SCB-level current data, calculate Expected current and CRs
    """
    df = df.copy()
    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
    df = df.set_index(df.columns[0])

    irr_col = df.columns[-1]
    irr = df[irr_col].astype(float)

    expected_str_current = I_MODULE_STC * (irr / 1000.0)
    expected_scb_current = expected_str_current * STRINGS_PER_SCB
    df["Expected_SCB_Current"] = expected_scb_current

    scb_cols = df.columns[:-2]  # All SCB current columns
    for col in scb_cols:
        df[f"CR_{col}"] = np.where(expected_scb_current > 0,
                                   df[col].astype(float) / expected_scb_current,
                                   np.nan)

    return df


# ----------------- Weak String Identification -----------------
def find_weak_strings(df, threshold=0.9, time_fraction=0.3):
    """
    Identify weak strings:
    - CR < threshold for >= time_fraction of the selected period
    """
    cr_cols = [c for c in df.columns if c.startswith("CR_")]
    weak_summary = []

    for col in cr_cols:
        below_thresh = df[col] < threshold
        frac_below = below_thresh.mean()  # Fraction of rows below threshold
        if frac_below >= time_fraction:
            weak_summary.append({
                "SCB_String": col.replace("CR_", ""),
                "Weak_Fraction": round(frac_below * 100, 2),
                "Avg_CR": round(df[col].mean(), 3)
            })

    weak_df = pd.DataFrame(weak_summary)
    return weak_df


# ----------------- Streamlit App -----------------
def main():
    st.title("🔍 Weak String Finder (≥30% of Time Below CR 0.9)")
    st.write("Upload your SCB current data, and get **only weak strings** in a separate downloadable file.")

    uploaded_file = st.file_uploader("Upload Excel/CSV file", type=["xlsx", "xls", "csv"])
    if uploaded_file is not None:
        # Load data
        if uploaded_file.name.endswith(".csv"):
            raw_df = pd.read_csv(uploaded_file)
        else:
            raw_df = pd.read_excel(uploaded_file)

        st.success("✅ File uploaded successfully!")
        st.write("### Raw Data Preview:")
        st.dataframe(raw_df.head())

        # Process
        df_processed = process_file(raw_df)

        # Time filter
        min_time, max_time = df_processed.index.min(), df_processed.index.max()
        start_time, end_time = st.slider(
            "⏱️ Select Time Range",
            min_value=min_time.to_pydatetime(),
            max_value=max_time.to_pydatetime(),
            value=(min_time.to_pydatetime(), max_time.to_pydatetime()),
            format="HH:mm"
        )
        df_filtered = df_processed.loc[start_time:end_time]

        # Weak strings
        weak_df = find_weak_strings(df_filtered, threshold=0.9, time_fraction=0.3)

        st.subheader("📉 Weak Strings (≥30% below CR 0.9)")
        if weak_df.empty:
            st.success("🎉 No weak strings detected!")
        else:
            st.dataframe(weak_df)

            csv = weak_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Weak Strings CSV",
                data=csv,
                file_name="weak_strings_summary.csv",
                mime="text/csv"
            )

        # Optional CR trend plot
        cr_cols = [c for c in df_filtered.columns if c.startswith("CR_")]
        st.subheader("📈 CR Trend Plot")
        plt.figure(figsize=(12, 5))
        for col in cr_cols:
            plt.plot(df_filtered.index, df_filtered[col], alpha=0.5, label=col)
        plt.axhline(1.0, color="green", linestyle="--")
        plt.axhline(0.9, color="red", linestyle="--")
        plt.xlabel("Time")
        plt.ylabel("CR")
        plt.legend(loc='upper right', ncol=4)
        plt.grid(True)
        st.pyplot(plt)


if __name__ == "__main__":
    main()
