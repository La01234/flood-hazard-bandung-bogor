import streamlit as st
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.graph_objects as go
import os
import json

st.set_page_config(page_title='Flood Hazard Explorer', page_icon='🌊', layout='wide')

FEATURE_COLS = ['elevation','slope','aspect','TWI','HAND','NDVI','MNDWI','NDBI','SAR_VV_baseline','dist_river']

DEFAULTS = {
    'Bandung': {'elevation':722.0,'slope':3.4,'aspect':165.0,'TWI':4.5,'HAND':5.0,'NDVI':0.28,'MNDWI':-0.39,'NDBI':0.07,'SAR_baseline':-5.1,'dist_river':570.0},
    'Bogor'  : {'elevation':255.0,'slope':3.9,'aspect':164.0,'TWI':4.5,'HAND':4.0,'NDVI':0.38,'MNDWI':-0.38,'NDBI':0.0,'SAR_baseline':-5.7,'dist_river':445.0}
}

CITY_CENTER = {'Bandung':[-6.9175,107.6191],'Bogor':[-6.5971,106.8060]}

@st.cache_resource
def load_models():
    rf_data      = joblib.load('models/rf_bandung.joblib')
    rf_model     = rf_data['model']
    rf_threshold = rf_data['best_threshold']
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model('models/xgb_bogor.json')
    with open('models/xgb_bogor_meta.json') as f:
        xgb_meta = json.load(f)
    xgb_threshold = xgb_meta.get('best_threshold', 0.5)
    return rf_model, rf_threshold, xgb_model, xgb_threshold

rf_model, rf_threshold, xgb_model, xgb_threshold = load_models()

st.title('🌊 Flood Hazard Explorer')
st.markdown('**Analisis Kerentanan Banjir — Kota Bandung & Kota Bogor**')
st.markdown('Urban Analytics Project | Machine Learning + Remote Sensing')
st.divider()

tab1, tab2, tab3 = st.tabs(['🗺️ Peta Hazard','📊 Komparasi Model','🔍 Prediksi Titik'])

# TAB 1
with tab1:
    st.subheader('Peta Flood Susceptibility')
    kota_peta = st.radio('Pilih tampilan:', ['Bandung','Bogor','Keduanya'], horizontal=True)
    if kota_peta == 'Keduanya':
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('#### Kota Bandung — Random Forest')
            if os.path.exists('outputs/flood_susceptibility_map_bandung.png'):
                st.image('outputs/flood_susceptibility_map_bandung.png', use_column_width=True)
            else:
                st.warning('Peta Bandung belum tersedia')
            st.metric('AUC-ROC','0.70')
            st.metric('Flood Area','12.9%')
        with col2:
            st.markdown('#### Kota Bogor — XGBoost')
            if os.path.exists('outputs/flood_susceptibility_map_bogor.png'):
                st.image('outputs/flood_susceptibility_map_bogor.png', use_column_width=True)
            else:
                st.warning('Peta Bogor belum tersedia')
            st.metric('AUC-ROC','0.71')
            st.metric('Flood Area','20.5%')
    else:
        center = CITY_CENTER[kota_peta]
        m = folium.Map(location=center, zoom_start=13, tiles='CartoDB positron')
        folium.TileLayer('OpenStreetMap', name='OpenStreetMap').add_to(m)
        folium.TileLayer(tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}', attr='Google Satellite', name='Satellite').add_to(m)
        folium.Marker(location=center, popup=f'Pusat Kota {kota_peta}', icon=folium.Icon(color='blue', icon='info-sign')).add_to(m)
        folium.LayerControl().add_to(m)
        img_path = f'outputs/flood_susceptibility_map_{kota_peta.lower()}.png'
        if os.path.exists(img_path):
            col_map, col_img = st.columns([1,1])
            with col_map:
                st.markdown(f'#### Peta Interaktif — {kota_peta}')
                st_folium(m, width=500, height=450)
            with col_img:
                st.markdown('#### Flood Susceptibility Map')
                st.image(img_path, use_column_width=True)
        else:
            st_folium(m, width=700, height=500)
    st.divider()
    st.markdown('**Legenda Tingkat Hazard:**')
    col_l1,col_l2,col_l3,col_l4 = st.columns(4)
    col_l1.markdown('🟢 **Rendah** < 25%')
    col_l2.markdown('🟡 **Sedang** 25-50%')
    col_l3.markdown('🟠 **Tinggi** 50-75%')
    col_l4.markdown('🔴 **Sangat Tinggi** > 75%')

# TAB 2
with tab2:
    st.subheader('Komparasi Performa Model')
    col1,col2,col3,col4 = st.columns(4)
    col1.metric('AUC Bandung (RF)','0.70')
    col2.metric('AUC Bogor (XGBoost)','0.71','+0.01')
    col3.metric('Study Area Bandung','180K piksel')
    col4.metric('Study Area Bogor','117K piksel')
    st.divider()
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        st.markdown('#### Perbandingan Metrik Model')
        metrics = ['AUC-ROC','Precision (Banjir)','Recall (Banjir)','F1 (Banjir)']
        bandung_vals = [0.70, 0.45, 0.62, 0.52]
        bogor_vals   = [0.71, 0.42, 0.65, 0.51]
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(name='Bandung (RF)', x=metrics, y=bandung_vals, marker_color='#185FA5', text=[f'{v:.2f}' for v in bandung_vals], textposition='outside'))
        fig_bar.add_trace(go.Bar(name='Bogor (XGBoost)', x=metrics, y=bogor_vals, marker_color='#993C1D', text=[f'{v:.2f}' for v in bogor_vals], textposition='outside'))
        fig_bar.update_layout(barmode='group', yaxis=dict(range=[0,1], title='Nilai'), legend=dict(orientation='h', y=-0.2), height=350, margin=dict(t=20,b=20))
        st.plotly_chart(fig_bar, use_container_width=True)
    with col_chart2:
        st.markdown('#### Distribusi Kelas Label')
        fig_pie = go.Figure()
        fig_pie.add_trace(go.Bar(name='Non-Banjir', x=['Bandung','Bogor'], y=[87.1,79.5], marker_color='#4CAF50', text=['87.1%','79.5%'], textposition='inside'))
        fig_pie.add_trace(go.Bar(name='Banjir', x=['Bandung','Bogor'], y=[12.9,20.5], marker_color='#E24B4A', text=['12.9%','20.5%'], textposition='inside'))
        fig_pie.update_layout(barmode='stack', yaxis=dict(title='Persentase (%)'), legend=dict(orientation='h', y=-0.2), height=350, margin=dict(t=20,b=20))
        st.plotly_chart(fig_pie, use_container_width=True)
    st.divider()
    st.markdown('#### Eksplorasi Threshold')
    col_sl1, col_sl2 = st.columns(2)
    with col_sl1:
        st.markdown('**Bandung — Random Forest**')
        threshold_bdg = st.slider('Threshold Bandung', min_value=0.1, max_value=0.9, value=float(round(rf_threshold,2)), step=0.01, key='slider_bdg')
        st.caption(f'Default optimal: {rf_threshold:.2f}')
        if threshold_bdg < rf_threshold:
            st.info('Threshold lebih rendah: recall naik, precision turun')
        elif threshold_bdg > rf_threshold:
            st.info('Threshold lebih tinggi: precision naik, recall turun')
        else:
            st.success('Menggunakan threshold optimal')
    with col_sl2:
        st.markdown('**Bogor — XGBoost**')
        threshold_bgr = st.slider('Threshold Bogor', min_value=0.1, max_value=0.9, value=float(round(xgb_threshold,2)), step=0.01, key='slider_bgr')
        st.caption(f'Default optimal: {xgb_threshold:.2f}')
        if threshold_bgr < xgb_threshold:
            st.info('Threshold lebih rendah: recall naik, precision turun')
        elif threshold_bgr > xgb_threshold:
            st.info('Threshold lebih tinggi: precision naik, recall turun')
        else:
            st.success('Menggunakan threshold optimal')
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('#### Evaluasi Model — Bandung')
        if os.path.exists('outputs/evaluasi_rf_bandung.png'):
            st.image('outputs/evaluasi_rf_bandung.png', use_column_width=True)
    with col2:
        st.markdown('#### Evaluasi Model — Bogor')
        if os.path.exists('outputs/evaluasi_xgb_bogor.png'):
            st.image('outputs/evaluasi_xgb_bogor.png', use_column_width=True)

# TAB 3
with tab3:
    st.subheader('Prediksi Flood Hazard di Titik Koordinat')
    col_kota, col_default = st.columns([1,2])
    with col_kota:
        kota = st.selectbox('Pilih Kota', ['Bandung (Random Forest)','Bogor (XGBoost)'])
    kota_key = 'Bandung' if 'Bandung' in kota else 'Bogor'
    defaults = DEFAULTS[kota_key]
    with col_default:
        st.markdown(' ')
        st.markdown(' ')
        use_default = st.button(f'Gunakan nilai rata-rata {kota_key}')
    if use_default:
        for key, val in defaults.items():
            st.session_state[key] = val
    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        elevation    = st.number_input('Elevation (m)', value=float(st.session_state.get('elevation', defaults['elevation'])), step=1.0)
        slope        = st.number_input('Slope (gradian)', value=float(st.session_state.get('slope', defaults['slope'])), step=0.1)
        aspect       = st.number_input('Aspect (gradian)', value=float(st.session_state.get('aspect', defaults['aspect'])), step=1.0)
        TWI          = st.number_input('TWI', value=float(st.session_state.get('TWI', defaults['TWI'])), step=0.1)
    with col2:
        HAND         = st.number_input('HAND (m)', value=float(st.session_state.get('HAND', defaults['HAND'])), step=0.1)
        NDVI         = st.number_input('NDVI', value=float(st.session_state.get('NDVI', defaults['NDVI'])), step=0.01, min_value=-1.0, max_value=1.0)
        MNDWI        = st.number_input('MNDWI', value=float(st.session_state.get('MNDWI', defaults['MNDWI'])), step=0.01, min_value=-1.0, max_value=1.0)
    with col3:
        NDBI         = st.number_input('NDBI', value=float(st.session_state.get('NDBI', defaults['NDBI'])), step=0.01, min_value=-1.0, max_value=1.0)
        SAR_baseline = st.number_input('SAR VV Baseline (dB)', value=float(st.session_state.get('SAR_baseline', defaults['SAR_baseline'])), step=0.1)
        dist_river   = st.number_input('Jarak ke Sungai (m)', value=float(st.session_state.get('dist_river', defaults['dist_river'])), step=10.0)
    st.divider()
    col_pred, col_map = st.columns([1,1])
    with col_pred:
        threshold_used = st.session_state.get('slider_bdg', rf_threshold) if 'Bandung' in kota else st.session_state.get('slider_bgr', xgb_threshold)
        st.caption(f'Threshold aktif: {threshold_used:.2f}')
        if st.button('Prediksi Sekarang', type='primary'):
            features = np.array([[elevation, slope, aspect, TWI, HAND, NDVI, MNDWI, NDBI, SAR_baseline, dist_river]])
            if 'Bandung' in kota:
                proba = rf_model.predict_proba(features)[0][1]
                model_name = 'Random Forest'
            else:
                proba = xgb_model.predict_proba(features)[0][1]
                model_name = 'XGBoost'
            pred = 1 if proba >= threshold_used else 0
            if pred == 1:
                st.error('BAHAYA BANJIR TERDETEKSI')
            else:
                st.success('Risiko Banjir Rendah')
            st.metric('Probabilitas Banjir', f'{proba*100:.1f}%')
            st.metric('Model', model_name)
            st.metric('Threshold', f'{threshold_used:.2f}')
            fig, ax = plt.subplots(figsize=(5,2.5))
            colors = ['green','yellow','orange','red']
            cmap = mcolors.LinearSegmentedColormap.from_list('hazard', colors)
            ax.barh(['Risk'], [proba], color=cmap(proba), height=0.4)
            ax.barh(['Risk'], [1-proba], left=[proba], color='#f0f0f0', height=0.4)
            ax.axvline(x=threshold_used, color='black', linestyle='--', linewidth=2, label=f'Threshold ({threshold_used:.2f})')
            ax.set_xlim(0,1)
            ax.set_xlabel('Probabilitas Banjir')
            ax.legend(fontsize=9)
            ax.set_title(f'Flood Risk: {proba*100:.1f}%', fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
    with col_map:
        st.markdown('#### Lokasi pada Peta')
        center = CITY_CENTER[kota_key]
        m_pred = folium.Map(location=center, zoom_start=13, tiles='CartoDB positron')
        folium.Marker(location=center, popup=f'{kota_key} — Elevation: {elevation:.0f}m', icon=folium.Icon(color='red', icon='info-sign')).add_to(m_pred)
        folium.Circle(location=center, radius=500, color='red', fill=True, fill_opacity=0.3).add_to(m_pred)
        st_folium(m_pred, width=450, height=400)
        st.caption('Marker menunjukkan pusat kota yang dipilih')

st.divider()
st.markdown('<div style="text-align:center;color:gray;font-size:12px;">Flood Hazard Explorer | Urban Analytics UAS 2025 | Bandung (RF) + Bogor (XGBoost)</div>', unsafe_allow_html=True)
