import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ================== Constants (edit if needed) ==================
MODULE_POWER_WP = 545.0        # Module Pmp (W)
MODULE_VOC = 49.91             # Module Voc (V)
VMP_VOC_RATIO = 0.82           # Typical Vmp/Voc
STRINGS_PER_SCB = 18           # Strings in one SCB
IRRADIANCE_MASK_MIN = 200.0    # Ignore CR when Irr < this (W/m^2)

VMP = MODULE_VOC * VMP_VOC_RATIO
I_MODULE_STC = MODULE_POWER_WP / VMP  # ≈ 13.31 A per module @ STC

# ================== Helpers ==================
def _guess_timestamp(cols):
    lc = [c.lower() for c in cols]
    for key in ["timestamp", "time", "date", "datetime"]:
        if key in lc:
            return cols[lc.index(key)]
    return cols[0]

def _guess_irradiance(cols):
    lc = [c.lower() for c in cols]
    # try common irradiance names
    keys = ["irradiance", "irr", "poa", "ghi", "w/m2", "w_m2", "wm2"]
    for i, name in enumerate(lc):
        if any(k in name for k in keys):
            return cols[i]
    # fallback: last column
    return cols[-1]

def compute_expected_and_cr(df, ts_col, irr_col, scb_cols):
    out = df.copy()

    # Parse timestamp and sort
    out[ts_col] = pd.to_datetime(out[ts_col], errors="coerce")
    out = out.dropna(subset=[ts_col]).sort_values(ts_col).set_index(ts_col)

    # Numeric conversions
    out[irr_col] = pd.to_numeric(out[irr_col], errors="coerce")
    out[scb_cols] = out[scb_cols].apply(pd.to_numeric, errors="coerce")

    # Expected SCB current per row (like your Excel col Z)
    expected_str_current = I_MODULE_STC * (out[irr_col] / 1000.0)
    out["Expected_SCB_Current"] = expected_str_current * STRINGS_PER_SCB

    # CR for each SCB column (like Excel AA for SCB-1, but for ALL SCBs)
    # Mask very low irradiance to avoid meaningless CR explosions
    valid = (out["Expected_SCB_Current"] > 0) & (out[irr_col] >= IRRADIANCE_MASK_MIN)
    for c in scb_cols:
        cr_col = f"CR_{c}"
        out[cr_col] = np.where(valid, out[c] / out["Expected_SCB_Current"], np.nan)

    return out

def style_cr(df, scb_cols):
    cr_cols = [f"CR_{c}" for c in scb_cols]
    def _fmt(val):
        if pd.isna(val): return ''
        if val < 0.90:   return 'color: white; background-color: red;'
        if val < 0.95:   return 'color: black; background-color: orange;'
        return ''
    return df.style.applymap(_fmt, subset=cr_cols)

def plot_cr_heatmap(df, scb_cols):
    cr_cols = [f"CR_{c}" for c in scb_cols]
    if len(df) == 0 or not any(col in df for col in cr_cols):
        st.info("Nothing to plot.")
        return
    mat = df[cr_cols].to_numpy().T  # shape: (n_scbs, n_times)

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(mat, aspect='auto', origin='lower', vmin=0.7, vmax=1.1)
    ax.set_yticks(np.arange(len(cr_cols)))
    ax.set_yticklabels([col.replace("CR_", "") for col in cr_cols])
    # Sparse x tick labels for readability
    xt = np.linspace(0, max(len(df)-1, 1), min(12, max(len(df), 1))).astype(int)
    ax.set_xticks(xt)
    ax.set_xticklabels([df.index[i].strftime("%H:%M") for i in xt], rotation=45, ha='right')
    ax.set_title("CR Heatmap (each SCB vs Expected_SCB_Current)")
    fig.colorbar(im, ax=ax, label="CR (Measured / Expected)")
    st.pyplot(fig)

# ================== Streamlit App ==================
st.title("SCB Expected Current & CR (Minute-wise)")

uploaded = st.file_uploader("Upload Excel/CSV with: Timestamp + Irradiance + 18 SCB currents (B…S)",
                            type=["xlsx", "xls", "csv"])

if uploaded is not None:
    # Read file
    if uploaded.name.lower().endswith(".csv"):
        raw = pd.read_csv(uploaded)
    else:
        raw = pd.read_excel(uploaded, engine="openpyxl")

    st.subheader("Preview of uploaded data")
    st.dataframe(raw.head())

    # Column selection (robust, so no KeyErrors)
    cols = list(raw.columns)

    ts_default = _guess_timestamp(cols)
    irr_default = _guess_irradiance(cols)

    c1, c2 = st.columns(2)
    with c1:
        ts_col = st.selectbox("Timestamp column", options=cols, index=cols.index(ts_default))
    with c2:
        irr_col = st.selectbox("Irradiance column (W/m²)", options=cols, index=cols.index(irr_default))

    # SCB columns: pick exactly 18 (B…S in your Excel). Default = next 18 after timestamp if possible.
    numeric_candidates = [c for c in cols if c not in [ts_col, irr_col]]
    default_scb_cols = numeric_candidates[:18] if len(numeric_candidates) >= 18 else numeric_candidates
    scb_cols = st.multiselect("Select the 18 SCB current columns (B…S)", options=numeric_candidates,
                              default=default_scb_cols)

    if len(scb_cols) != 18:
        st.warning(f"Please select exactly 18 SCB columns (currently selected: {len(scb_cols)}).")
    else:
        # Compute expected & CR
        processed = compute_expected_and_cr(raw, ts_col, irr_col, scb_cols)

        # Optional time range filter
        tmin, tmax = processed.index.min(), processed.index.max()
        start_t, end_t = st.slider("Time range", min_value=tmin.to_pydatetime(),
                                   max_value=tmax.to_pydatetime(),
                                   value=(tmin.to_pydatetime(), tmax.to_pydatetime()),
                                   format="HH:mm")
        processed = processed.loc[start_t:end_t]

        # Show processed data: Expected + all CR columns
        show_cols = ["Expected_SCB_Current"] + [f"CR_{c}" for c in scb_cols]
        st.subheader(f"Expected SCB Current & CR (rows: {len(processed)})")
        st.dataframe(style_cr(processed[show_cols], scb_cols))

        # Heatmap (time vs SCBs)
        st.subheader("CR Heatmap")
        plot_cr_heatmap(processed, scb_cols)

        # Download
        st.download_button(
            "Download processed CSV",
            data=processed.to_csv().encode("utf-8"),
            file_name="processed_scb_cr.csv",
            mime="text/csv"
        )
else:
    st.info("Upload a file to begin.")
