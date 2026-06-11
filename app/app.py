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
        import base64, rasterio
        from rasterio.warp import transform_bounds

        center = CITY_CENTER[kota_peta]
        img_path = f'outputs/flood_susceptibility_map_{kota_peta.lower()}.png'
        tif_path = f'data/raw/flood_features_{kota_peta.lower()}_v2.tif'

        # Hitung bounds WGS84 dari raster
        raster_bounds = None
        if os.path.exists(tif_path):
            with rasterio.open(tif_path) as src:
                b = transform_bounds(src.crs, 'EPSG:4326',
                                     src.bounds.left, src.bounds.bottom,
                                     src.bounds.right, src.bounds.top)
                raster_bounds = [[b[1], b[0]], [b[3], b[2]]]  # [[lat_min,lon_min],[lat_max,lon_max]]

        m = folium.Map(location=center, zoom_start=13, tiles='CartoDB positron')
        folium.TileLayer('OpenStreetMap', name='OpenStreetMap').add_to(m)
        folium.TileLayer(tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
                         attr='Google Satellite', name='Satellite').add_to(m)

        # Overlay PNG susceptibility map
        if os.path.exists(img_path) and raster_bounds:
            with open(img_path, 'rb') as f_img:
                img_b64 = base64.b64encode(f_img.read()).decode()
            folium.raster_layers.ImageOverlay(
                image=f'data:image/png;base64,{img_b64}',
                bounds=raster_bounds,
                opacity=0.6,
                name='Flood Susceptibility',
                interactive=True,
                cross_origin=False,
            ).add_to(m)

        # Batas wilayah studi
        if raster_bounds:
            lat_min, lon_min = raster_bounds[0]
            lat_max, lon_max = raster_bounds[1]
            folium.Rectangle(
                bounds=raster_bounds,
                color='#185FA5', weight=2, fill=False,
                tooltip='Batas Area Studi'
            ).add_to(m)

        folium.Marker(location=center, popup=f'Pusat Kota {kota_peta}',
                      icon=folium.Icon(color='blue', icon='info-sign')).add_to(m)
        folium.LayerControl().add_to(m)

        if os.path.exists(img_path):
            col_map, col_img = st.columns([1,1])
            with col_map:
                st.markdown(f'#### Peta Interaktif — {kota_peta}')
                st.caption('Layer susceptibility map bisa di-toggle di pojok kanan atas peta')
                st_folium(m, width='100%', height=480)
            with col_img:
                st.markdown('#### Flood Susceptibility Map')
                st.image(img_path, use_column_width=True)
                if kota_peta == 'Bandung':
                    st.metric('AUC-ROC','0.70')
                    st.metric('Flood Area','12.9%')
                else:
                    st.metric('AUC-ROC','0.71')
                    st.metric('Flood Area','20.5%')
        else:
            st.info('Susceptibility map PNG belum tersedia — menampilkan peta dasar')
            st_folium(m, width='100%', height=500)

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
    import rasterio
    from rasterio.transform import rowcol
    from rasterio.warp import transform_bounds
    from pyproj import Transformer

    RASTER = {
        'Bandung': 'data/raw/flood_features_bandung_v2.tif',
        'Bogor'  : 'data/raw/flood_features_bogor_v2.tif',
    }
    BAND_MAP = {
        'elevation':1,'slope':2,'aspect':3,'TWI':4,'HAND':5,
        'NDVI':6,'MNDWI':7,'NDBI':8,'SAR_VV_baseline':9,'dist_river':11
    }
    FEATURE_LABELS = {
        'elevation'      : ('Elevation','m','Ketinggian permukaan tanah'),
        'slope'          : ('Slope','°','Kemiringan lereng'),
        'aspect'         : ('Aspect','°','Arah hadap lereng'),
        'TWI'            : ('TWI','','Topographic Wetness Index'),
        'HAND'           : ('HAND','m','Height Above Nearest Drainage'),
        'NDVI'           : ('NDVI','','Indeks vegetasi (-1 s/d 1)'),
        'MNDWI'          : ('MNDWI','','Indeks air (-1 s/d 1)'),
        'NDBI'           : ('NDBI','','Indeks bangunan (-1 s/d 1)'),
        'SAR_VV_baseline': ('SAR VV','dB','Backscatter radar baseline'),
        'dist_river'     : ('Jarak Sungai','m','Jarak ke sungai terdekat'),
    }

    @st.cache_data
    def get_raster_bounds_wgs84(kota_key):
        with rasterio.open(RASTER[kota_key]) as src:
            b = transform_bounds(src.crs, 'EPSG:4326',
                                 src.bounds.left, src.bounds.bottom,
                                 src.bounds.right, src.bounds.top)
            return {'lon_min': b[0], 'lat_min': b[1], 'lon_max': b[2], 'lat_max': b[3]}

    def extract_features_from_raster(kota_key, lat_wgs, lon_wgs):
        path = RASTER[kota_key]
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:32748", always_xy=True)
        x_utm, y_utm = transformer.transform(lon_wgs, lat_wgs)
        result = {}
        with rasterio.open(path) as src:
            b = src.bounds
            if not (b.left <= x_utm <= b.right and b.bottom <= y_utm <= b.top):
                return None, f"Koordinat di luar area studi ({x_utm:.0f}, {y_utm:.0f} UTM)"
            row, col = rowcol(src.transform, x_utm, y_utm)
            row = max(0, min(row, src.height-1))
            col = max(0, min(col, src.width-1))
            for feat, band in BAND_MAP.items():
                data = src.read(band)
                val  = float(data[row, col])
                result[feat] = round(val, 4)
        return result, None

    st.subheader('Prediksi Flood Hazard di Titik Koordinat')
    st.caption('Klik peta atau masukkan koordinat — nilai fitur terisi otomatis dari raster')

    kota = st.selectbox('Pilih Kota', ['Bandung (Random Forest)', 'Bogor (XGBoost)'], key='kota_sel')
    kota_key = 'Bandung' if 'Bandung' in kota else 'Bogor'

    # Jika kota berubah, reset koordinat ke pusat kota baru
    if st.session_state.get('active_kota') != kota_key:
        st.session_state['active_kota'] = kota_key
        st.session_state['lat'] = CITY_CENTER[kota_key][0]
        st.session_state['lon'] = CITY_CENTER[kota_key][1]
        st.session_state['has_prediction'] = False
        extract_features_from_raster.clear()
        st.rerun()

    center_map = CITY_CENTER[kota_key]
    bounds_wgs = get_raster_bounds_wgs84(kota_key)

    st.divider()
    col_map, col_form = st.columns([1.1, 0.9])

    with col_map:
        st.markdown('#### Pilih Lokasi')
        m_click = folium.Map(location=center_map, zoom_start=13, tiles='CartoDB positron')
        folium.TileLayer('OpenStreetMap', name='OpenStreetMap').add_to(m_click)
        folium.TileLayer(
            tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
            attr='Google Satellite', name='Satellite'
        ).add_to(m_click)

        # Batas wilayah studi
        folium.Rectangle(
            bounds=[[bounds_wgs['lat_min'], bounds_wgs['lon_min']],
                    [bounds_wgs['lat_max'], bounds_wgs['lon_max']]],
            color='#185FA5', weight=2.5, fill=True,
            fill_color='#185FA5', fill_opacity=0.05,
            tooltip=f'Area Studi {kota_key} — klik di dalam area ini'
        ).add_to(m_click)

        # Marker lokasi terpilih
        cur_lat = st.session_state.get('lat', center_map[0])
        cur_lon = st.session_state.get('lon', center_map[1])
        has_pred = st.session_state.get('has_prediction') and st.session_state.get('pred_kota') == kota_key
        marker_col = 'red'   if (has_pred and st.session_state.get('last_pred')==1) else                      'green' if (has_pred and st.session_state.get('last_pred')==0) else 'blue'
        folium.Marker(
            location=[cur_lat, cur_lon],
            popup=f'Lat: {cur_lat:.5f}<br>Lon: {cur_lon:.5f}',
            icon=folium.Icon(color=marker_col, icon='map-marker')
        ).add_to(m_click)
        folium.LayerControl().add_to(m_click)

        map_data = st_folium(m_click, width='100%', height=420,
                             returned_objects=['last_clicked'], key=f'map_{kota_key}')

        if map_data and map_data.get('last_clicked'):
            clicked  = map_data['last_clicked']
            new_lat  = round(clicked['lat'], 6)
            new_lon  = round(clicked['lng'], 6)
            if (new_lat, new_lon) != (st.session_state.get('lat'), st.session_state.get('lon')):
                st.session_state['lat'] = new_lat
                st.session_state['lon'] = new_lon
                st.session_state['has_prediction'] = False
                st.rerun()

        st.caption(f'Area studi (kotak biru): Lat {bounds_wgs["lat_min"]:.3f}~{bounds_wgs["lat_max"]:.3f} | Lon {bounds_wgs["lon_min"]:.3f}~{bounds_wgs["lon_max"]:.3f}')

    with col_form:
        st.markdown('#### Koordinat & Fitur')

        col_lat, col_lon = st.columns(2)
        with col_lat:
            lat = st.number_input('Latitude', value=float(st.session_state.get('lat', center_map[0])),
                                  step=0.0001, format='%.6f', key='lat_input')
        with col_lon:
            lon = st.number_input('Longitude', value=float(st.session_state.get('lon', center_map[1])),
                                  step=0.0001, format='%.6f', key='lon_input')

        if (lat, lon) != (st.session_state.get('lat'), st.session_state.get('lon')):
            st.session_state['lat'] = lat
            st.session_state['lon'] = lon
            st.session_state['has_prediction'] = False

        feats, err = extract_features_from_raster(kota_key, lat, lon)

        if err:
            st.warning(f'⚠️ {err}')
            st.caption(f'Pastikan koordinat berada di dalam kotak biru pada peta')
        elif feats:
            st.success('✅ Nilai fitur diekstrak dari raster')
            rows = []
            for feat, val in feats.items():
                label, unit, desc = FEATURE_LABELS[feat]
                rows.append({'Fitur': f'{label} ({unit})' if unit else label,
                             'Nilai': val, 'Keterangan': desc})
            df_feats = pd.DataFrame(rows)
            st.dataframe(df_feats, use_container_width=True, hide_index=True,
                         column_config={'Nilai': st.column_config.NumberColumn(format='%.4f')})

        st.divider()
        threshold_used = st.session_state.get('slider_bdg', rf_threshold) if 'Bandung' in kota                          else st.session_state.get('slider_bgr', xgb_threshold)
        st.caption(f'Model: {"Random Forest" if "Bandung" in kota else "XGBoost"} | Threshold: {threshold_used:.2f}')

        btn_predict = st.button('🔍 Prediksi Sekarang', type='primary',
                                disabled=(feats is None), use_container_width=True)

        if btn_predict and feats:
            features = np.array([[
                feats['elevation'], feats['slope'], feats['aspect'], feats['TWI'],
                feats['HAND'], feats['NDVI'], feats['MNDWI'], feats['NDBI'],
                feats['SAR_VV_baseline'], feats['dist_river']
            ]])
            if 'Bandung' in kota:
                proba = rf_model.predict_proba(features)[0][1]
                model_name = 'Random Forest'
            else:
                proba = xgb_model.predict_proba(features)[0][1]
                model_name = 'XGBoost'
            pred = 1 if proba >= threshold_used else 0
            st.session_state.update({
                'last_pred': pred, 'last_proba': proba,
                'last_model': model_name, 'last_threshold': threshold_used,
                'last_lat': lat, 'last_lon': lon,
                'has_prediction': True, 'pred_kota': kota_key
            })
            st.rerun()

    # ── HASIL ────────────────────────────────────────────────
    if st.session_state.get('has_prediction') and st.session_state.get('pred_kota') == kota_key:
        pred       = st.session_state['last_pred']
        proba      = st.session_state['last_proba']
        model_name = st.session_state['last_model']
        t_used     = st.session_state['last_threshold']
        res_lat    = st.session_state['last_lat']
        res_lon    = st.session_state['last_lon']

        st.divider()
        st.markdown('#### Hasil Prediksi')

        c1, c2, c3, c4 = st.columns(4)
        c1.metric('Status', '🔴 BAHAYA' if pred==1 else '🟢 AMAN')
        c2.metric('Probabilitas', f'{proba*100:.1f}%')
        c3.metric('Model', model_name)
        c4.metric('Threshold', f'{t_used:.2f}')

        if pred == 1:
            st.error('Lokasi ini terindikasi RAWAN BANJIR berdasarkan model ML')
        else:
            st.success('Lokasi ini terindikasi AMAN dari risiko banjir berdasarkan model ML')

        fig = go.Figure(go.Indicator(
            mode='gauge+number+delta',
            value=proba*100,
            number={'suffix':'%', 'font':{'size':36}},
            delta={'reference': t_used*100, 'suffix':'% (threshold)'},
            gauge={
                'axis': {'range':[0,100]},
                'bar': {'color': '#E24B4A' if pred==1 else '#4CAF50'},
                'steps': [
                    {'range':[0,25],  'color':'#EAF3DE'},
                    {'range':[25,50], 'color':'#FFF9C4'},
                    {'range':[50,75], 'color':'#FFE0B2'},
                    {'range':[75,100],'color':'#FFCDD2'},
                ],
                'threshold': {
                    'line': {'color':'black','width':3},
                    'thickness': 0.75,
                    'value': t_used*100
                }
            },
            title={'text': f'Flood Risk Score — {res_lat:.4f}, {res_lon:.4f}'}
        ))
        fig.update_layout(height=280, margin=dict(t=40,b=10,l=20,r=20))
        st.plotly_chart(fig, use_container_width=True)


st.divider()
st.markdown('<div style="text-align:center;color:gray;font-size:12px;">Flood Hazard Explorer | Urban Analytics UAS 2025 | Bandung (RF) + Bogor (XGBoost)</div>', unsafe_allow_html=True)
