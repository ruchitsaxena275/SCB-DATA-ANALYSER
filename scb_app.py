import pandas as pd
import numpy as np
import streamlit as st

# ----------------- Core Functions -----------------

def calculate_expected_current(irradiance, factor=0.1):
    """Calculate expected SCB current from irradiance (adjust factor as needed)."""
    return irradiance * factor


def process_scb_data(df):
    """
    Process SCB data:
    - Calculates Expected Current
    - Computes CR (Current Ratio)
    - Identifies Weak SCBs (low CR for >30% of time)
    - Colors weak strings for CSV
    """

    irr_col = "Irradiance"
    scb_cols = [col for col in df.columns if col.startswith("SCB")]

    if irr_col not in df.columns:
        raise ValueError(f"Missing required column: '{irr_col}'")

    # Calculate Expected Current
    df["Expected_Current"] = df[irr_col].apply(calculate_expected_current)

    # Compute CR for each SCB
    for col in scb_cols:
        cr_col = f"CR_{col}"
        df[cr_col] = df[col] / df["Expected_Current"]

    # Determine Weak SCBs
    weak_data = []
    threshold = 0.9  # CR threshold
    weak_cols = []

    for col in scb_cols:
        cr_col = f"CR_{col}"
        low_ratio = (df[cr_col] < threshold).mean()
        if low_ratio > 0.3:  # Weak if >30% of time
            weak_data.append({"SCB": col, "Low_Percentage": round(low_ratio * 100, 2)})
            weak_cols.append(col)

    weak_df = pd.DataFrame(weak_data)

    # Create tag columns
    def tag_value(value, is_weak):
        return "🔴 Weak" if is_weak else "🟢 OK"

    tag_cols = {}
    for col in scb_cols:
        is_weak = col in weak_cols
        tag_cols[f"Tag_{col}"] = [tag_value(v, is_weak) for v in df[col]]

    tag_df = pd.DataFrame(tag_cols)
    df = pd.concat([df, tag_df], axis=1)

    return df, weak_df


# ----------------- Streamlit App -----------------

def main():
    st.title("🔍 SCB Current Analyzer with Weak SCB Identification")
    st.write("Upload your **CSV or Excel** file and analyze SCB currents below.")

    uploaded_file = st.file_uploader("📂 Upload CSV or Excel", type=["csv", "xlsx", "xls"])

    if uploaded_file is not None:
        try:
            # Read file dynamically
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.success("✅ File uploaded successfully!")

            st.subheader("Raw Data Preview")
            st.dataframe(df.head())

            # Process data
            processed_df, weak_scbs = process_scb_data(df)

            st.subheader("📊 Processed Results")
            st.dataframe(processed_df.head())

            # Download Processed CSV
            csv = processed_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Download Processed CSV (with tags)",
                data=csv,
                file_name="processed_scb_data.csv",
                mime="text/csv"
            )

            # Weak SCB Download
            if weak_scbs is not None and not weak_scbs.empty:
                st.subheader("⚠️ Weak SCBs Detected")
                st.dataframe(weak_scbs)

                weak_csv = weak_scbs.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Download Weak SCBs CSV",
                    data=weak_csv,
                    file_name="weak_scbs.csv",
                    mime="text/csv"
                )
            else:
                st.info("✅ No Weak SCBs Found.")

        except Exception as e:
            st.error(f"❌ An error occurred: {e}")

    else:
        st.info("⬆️ Please upload a CSV or Excel file to start.")


if __name__ == "__main__":
    main()
