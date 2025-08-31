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
    """Calculate Expected SCB Current & CR for each SCB."""
    df = df.copy()
    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])  # First column is timestamp
    df = df.set_index(df.columns[0])               # Set datetime index

    irr_col = df.columns[-1]  # Last column is irradiance
    irr = df[irr_col].astype(float)

    expected_str_current = I_MODULE_STC * (irr / 1000.0)
    expected_scb_current = expected_str_current * STRINGS_PER_SCB
    df["Expected_SCB_Current"] = expected_scb_current

    # Calculate CR for each SCB current column
    scb_cols = df.columns[:-2]  # All but last (irradiance) and Expected
    for col in scb_cols:
        df[f"CR_{col}"] = np.where(
            expected_scb_current > 0,
            df[col].astype(float) / expected_scb_current,
            np.nan
        )

    return df

# ----------------- Weak SCB Identification -----------------
def find_weak_scbs(df, threshold=0.94, min_fraction=0.3):
    """Return SCBs weak for at least 30% of time."""
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
    st.title("🔍 SCB Current Analyzer with Weak SCB Identification")
   st.write("Upload your file and analyze SCB currents below.")

