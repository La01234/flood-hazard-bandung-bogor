import streamlit as st
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
import rasterio
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="Flood Hazard Explorer",
    page_icon="🌊",
    layout="wide"
)

FEATURE_COLS = [
    'elevation', 'slope', 'aspect', 'TWI', 'HAND',
    'NDVI', 'MNDWI', 'NDBI', 'SAR_VV_baseline', 'dist_river'
]

# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_models():
    # Random Forest Bandung
    rf_data = joblib.load('models/rf_bandung.joblib')
    rf_model     = rf_data['model']
    rf_threshold = rf_data['best_threshold']

    # XGBoost Bogor
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model('models/xgb_bogor.json')
    import json
    with open('models/xgb_bogor_meta.json') as f:
        xgb_meta = json.load(f)
    xgb_threshold = xgb_meta.get('best_threshold', 0.5)

    return rf_model, rf_threshold, xgb_model, xgb_threshold

rf_model, rf_threshold, xgb_model, xgb_threshold = load_models()

# ============================================================
# HEADER
# ============================================================
st.title("🌊 Flood Hazard Explorer")
st.markdown("**Analisis Kerentanan Banjir — Kota Bandung & Kota Bogor**")
st.markdown("Urban Analytics Project | Machine Learning + Remote Sensing")
st.divider()

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3 = st.tabs([
    "🗺️ Peta Hazard",
    "📊 Komparasi Model",
    "🔍 Prediksi Titik"
])

# ============================================================
# TAB 1 — PETA HAZARD
# ============================================================
with tab1:
    st.subheader("Peta Flood Susceptibility")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Kota Bandung — Random Forest")
        if os.path.exists('outputs/flood_susceptibility_map_bandung.png'):
            st.image('outputs/flood_susceptibility_map_bandung.png',
                     caption='Flood Susceptibility Map — Bandung',
                     use_column_width=True)
        else:
            st.warning("Peta Bandung belum tersedia")

        st.metric("AUC-ROC", "0.70")
        st.metric("Study Area", "180.375 piksel")
        st.metric("Flood Pixels", "23.275 (12.9%)")

    with col2:
        st.markdown("#### Kota Bogor — XGBoost")
        if os.path.exists('outputs/flood_susceptibility_map_bogor.png'):
            st.image('outputs/flood_susceptibility_map_bogor.png',
                     caption='Flood Susceptibility Map — Bogor',
                     use_column_width=True)
        else:
            st.warning("Peta Bogor belum tersedia")

        st.metric("AUC-ROC", "0.71")
        st.metric("Study Area", "117.120 piksel")
        st.metric("Flood Pixels", "23.994 (20.5%)")

# ============================================================
# TAB 2 — KOMPARASI MODEL
# ============================================================
with tab2:
    st.subheader("Komparasi Performa Model")

    # Tabel perbandingan
    comparison_data = {
        "Metrik"           : ["Algoritma", "AUC-ROC", "Threshold Optimal",
                              "Study Area", "Flood Pixels", "Fitur Utama"],
        "Bandung (RF)"     : ["Random Forest", "0.70", str(round(rf_threshold, 2)),
                              "180.375", "23.275 (12.9%)", "SAR_VV_baseline, NDVI"],
        "Bogor (XGBoost)"  : ["XGBoost", "0.71", str(round(xgb_threshold, 2)),
                              "117.120", "23.994 (20.5%)", "SAR_VV_baseline, elevation"]
    }
    df_compare = pd.DataFrame(comparison_data)
    st.dataframe(df_compare, use_container_width=True, hide_index=True)

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Evaluasi Model — Bandung")
        if os.path.exists('outputs/evaluasi_rf_bandung.png'):
            st.image('outputs/evaluasi_rf_bandung.png',
                     use_column_width=True)

    with col2:
        st.markdown("#### Evaluasi Model — Bogor")
        if os.path.exists('outputs/evaluasi_xgb_bogor.png'):
            st.image('outputs/evaluasi_xgb_bogor.png',
                     use_column_width=True)

# ============================================================
# TAB 3 — PREDIKSI TITIK
# ============================================================
with tab3:
    st.subheader("Prediksi Flood Hazard di Titik Koordinat")
    st.markdown("Masukkan nilai fitur untuk memprediksi tingkat kerentanan banjir.")

    col_kota, col_info = st.columns([1, 2])

    with col_kota:
        kota = st.selectbox(
            "Pilih Kota",
            ["Bandung (Random Forest)", "Bogor (XGBoost)"]
        )

    st.divider()

    # Input fitur
    col1, col2, col3 = st.columns(3)

    with col1:
        elevation = st.number_input("Elevation (m)", value=700.0, step=1.0)
        slope     = st.number_input("Slope (°)", value=3.0, step=0.1)
        aspect    = st.number_input("Aspect (°)", value=180.0, step=1.0)
        TWI       = st.number_input("TWI", value=4.5, step=0.1)

    with col2:
        HAND          = st.number_input("HAND (m)", value=5.0, step=0.1)
        NDVI          = st.number_input("NDVI", value=0.3, step=0.01,
                                         min_value=-1.0, max_value=1.0)
        MNDWI         = st.number_input("MNDWI", value=-0.3, step=0.01,
                                         min_value=-1.0, max_value=1.0)

    with col3:
        NDBI          = st.number_input("NDBI", value=0.1, step=0.01,
                                         min_value=-1.0, max_value=1.0)
        SAR_baseline  = st.number_input("SAR VV Baseline (dB)", value=-8.0, step=0.1)
        dist_river    = st.number_input("Jarak ke Sungai (m)", value=500.0, step=10.0)

    st.divider()

    if st.button("🔍 Prediksi", type="primary"):
        features = np.array([[
            elevation, slope, aspect, TWI, HAND,
            NDVI, MNDWI, NDBI, SAR_baseline, dist_river
        ]])

        if "Bandung" in kota:
            proba     = rf_model.predict_proba(features)[0][1]
            threshold = rf_threshold
            model_name = "Random Forest"
        else:
            proba     = xgb_model.predict_proba(features)[0][1]
            threshold = xgb_threshold
            model_name = "XGBoost"

        pred = 1 if proba >= threshold else 0

        col_result, col_gauge = st.columns([1, 1])

        with col_result:
            if pred == 1:
                st.error(f"⚠️ **BAHAYA BANJIR TERDETEKSI**")
            else:
                st.success(f"✅ **Risiko Banjir Rendah**")

            st.metric("Probabilitas Banjir", f"{proba*100:.1f}%")
            st.metric("Model", model_name)
            st.metric("Threshold", f"{threshold:.2f}")

        with col_gauge:
            fig, ax = plt.subplots(figsize=(4, 3))
            colors = ['green', 'yellow', 'orange', 'red']
            cmap   = mcolors.LinearSegmentedColormap.from_list("hazard", colors)
            ax.barh(["Probabilitas"], [proba], color=cmap(proba), height=0.4)
            ax.barh(["Probabilitas"], [1-proba], left=[proba],
                    color='lightgray', height=0.4)
            ax.axvline(x=threshold, color='black', linestyle='--',
                       linewidth=1.5, label=f'Threshold ({threshold:.2f})')
            ax.set_xlim(0, 1)
            ax.set_xlabel("Probabilitas")
            ax.legend(fontsize=8)
            ax.set_title("Flood Risk Gauge")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; font-size: 12px;'>
Urban Analytics Project | Flood Hazard Analysis Bandung & Bogor<br>
Random Forest (Bandung) | XGBoost (Bogor) | Sentinel-1/2 + DEM
</div>
""", unsafe_allow_html=True)
