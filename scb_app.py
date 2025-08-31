import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------- Constants --------
MODULE_POWER_WP = 545.0     # Module watt-peak rating
MODULE_VOC = 49.91          # Module Voc
VMP_VOC_RATIO = 0.82        # Vmp/Voc ratio
VMP = MODULE_VOC * VMP_VOC_RATIO
I_MODULE_STC = MODULE_POWER_WP / VMP  # ≈ 13.33 A
CR_LOW_THRESHOLD = 0.95     # Weak string threshold
IRRADIANCE_MIN = 200        # Ignore low irradiance (W/m²)

# -------- Functions --------
def process_file(df):
    """
    Takes uploaded raw data, calculates expected currents,
    current ratios, and prepares for plotting/summary.
    """
    df = df.copy()
    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])  # First col = timestamp
    df = df.set_index(df.columns[0])

    # Identify measured current columns (all numeric until irradiance col)
    measured_cols = df.columns[df.columns.str.contains("_ME_A")]
    irr_col = df.columns[-1]  # Assuming last col is irradiance

    measured = df[measured_cols].astype(float)
    irr = df[irr_col].astype(float)

    # Calculate expected current per string
    expected_str_current = I_MODULE_STC * (irr / 1000.0)

    result = measured.copy()
    for i, col in enumerate(measured_cols, start=1):
        result[f"Expected_String_{i}"] = expected_str_current
        result[f"CR_String_{i}"] = np.where(
            expected_str_current > 0,
            result[col] / expected_str_current,
            np.nan
        )

    result["Expected_SCB_Current"] = expected_str_current * len(measured_cols)
    result["Measured_SCB_Current"] = measured.sum(axis=1)
    result["Irradiance_Wm2"] = irr

    # Filter out rows with very low irradiance
    result = result[irr > IRRADIANCE_MIN]

    return result


def plot_heatmap(df):
    """
    Creates a heatmap of CR values for all strings over time.
    """
    cr_cols = [c for c in df.columns if c.startswith("CR_String_")]
    cr_matrix = df[cr_cols]

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(cr_matrix.T, aspect='auto', origin='lower', vmin=0.0, vmax=1.4)

    ax.set_yticks(np.arange(len(cr_cols)))
    ax.set_yticklabels([f"String {i+1}" for i in range(len(cr_cols))])

    xticks = np.linspace(0, len(cr_matrix)-1, min(12, len(cr_matrix))).astype(int)
    ax.set_xticks(xticks)
    ax.set_xticklabels([df.index[i].strftime("%m-%d %H:%M") for i in xticks], rotation=45, ha='right')

    ax.set_title("String Current Ratio (CR) Heatmap")
    fig.colorbar(im, ax=ax, label="CR (Measured / Expected)")
    st.pyplot(fig)


def daily_summary(df):
    """
    Summarizes weak strings per day based on CR threshold.
    """
    cr_cols = [c for c in df.columns if c.startswith("CR_String_")]
    df2 = df[cr_cols].copy()
    df2["date"] = df.index.date

    grouped = df2.groupby("date")
    rows = []

    for date, grp in grouped:
        weak = []
        for i, c in enumerate(cr_cols, start=1):
            frac = (grp[c] < CR_LOW_THRESHOLD).mean()
            if frac > 0.2:  # Weak for >20% of the time
                weak.append(f"String {i}")
        rows.append({"date": date, "weak_strings": ", ".join(weak)})

    return pd.DataFrame(rows)


# -------- Streamlit UI --------
st.title("SCB String Current Analysis Tool")

file = st.file_uploader("Upload Excel file (Timestamp, String Currents, Irradiance)", type=["xlsx"])
if file:
    df = pd.read_excel(file, engine="openpyxl")
    result = process_file(df)

    st.subheader("Preview of Processed Data")
    st.dataframe(result.head(20))

    st.subheader("Heatmap of Current Ratio (CR)")
    plot_heatmap(result)

    st.subheader("Daily Summary of Weak Strings")
    summary = daily_summary(result)
    st.dataframe(summary)

    # Download buttons
    csv = result.reset_index().to_csv(index=False).encode("utf-8")
    st.download_button("Download Processed Data", csv, "processed_data.csv", "text/csv")

    csv2 = summary.to_csv(index=False).encode("utf-8")
    st.download_button("Download Daily Summary", csv2, "daily_summary.csv", "text/csv")
