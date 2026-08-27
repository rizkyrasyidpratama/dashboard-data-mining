import base64
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    calinski_harabasz_score,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    precision_score,
    recall_score,
    silhouette_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler

# =========================================================
# 1. KONFIGURASI HALAMAN & THEME STYLING GLOBAL
# =========================================================
st.set_page_config(
    page_title="Analytics System - Putus Sekolah",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        background-color: #0f172a !important; 
        color: #f8fafc !important; 
    }

    [data-testid="stHeader"] { background-color: #0f172a !important; }
    [data-testid="stSidebar"] { background-color: #1e293b !important; border-right: 1px solid #334155 !important; }

    /* Typography Colors */
    [data-testid="stAppViewContainer"] h1, [data-testid="stAppViewContainer"] h2, 
    [data-testid="stAppViewContainer"] h3, [data-testid="stAppViewContainer"] h4, 
    [data-testid="stAppViewContainer"] p, [data-testid="stAppViewContainer"] span, 
    [data-testid="stAppViewContainer"] label { color: #f8fafc !important; }

    /* Selectbox & Input Customization */
    div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {
        background-color: #0f172a !important;
        border-color: #334155 !important;
        color: #ffffff !important;
    }
    div[data-baseweb="select"] *, div[data-baseweb="input"] * { color: #ffffff !important; }

    /* CUSTOM SIDEBAR NAVIGATION */
    [data-testid="stSidebar"] div[role="radiogroup"] {
        gap: 8px !important;
    }

    [data-testid="stSidebar"] div[role="radiogroup"] > label {
        background-color: #0f172a !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
        padding: 10px 14px !important;
        margin: 0 !important;
        width: 100% !important;
        transition: all 0.2s ease-in-out !important;
        cursor: pointer !important;
    }

    [data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
        border-color: #38bdf8 !important;
        background-color: #1e293b !important;
        transform: translateX(4px);
    }

    [data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"] {
        background-color: rgba(56, 189, 248, 0.12) !important;
        border-color: #38bdf8 !important;
        box-shadow: 0 0 10px rgba(56, 189, 248, 0.15) !important;
    }

    [data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child {
        display: none !important;
    }

    [data-testid="stSidebar"] div[role="radiogroup"] label p {
        font-size: 0.88rem !important;
        font-weight: 600 !important;
        color: #cbd5e1 !important;
        margin: 0 !important;
    }

    [data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"] p {
        color: #38bdf8 !important;
        font-weight: 700 !important;
    }

    /* Metric Cards Styling */
    .metric-box {
        background: #1e293b;
        border: 1px solid #334155;
        border-left: 4px solid #38bdf8;
        border-radius: 10px;
        padding: 1.1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .metric-label { color: #94a3b8 !important; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; }
    .metric-num { color: #ffffff !important; font-size: 1.8rem; font-weight: 800; margin-top: 0.2rem; }
    .metric-sub { color: #38bdf8 !important; font-size: 0.8rem; font-weight: 600; }

    /* Prediction Result Badges */
    .pred-card-rendah {
        background: rgba(34, 197, 94, 0.15); border: 2px solid #22c55e;
        border-radius: 14px; padding: 2rem; text-align: center; color: #4ade80 !important;
    }
    .pred-card-sedang {
        background: rgba(234, 179, 8, 0.15); border: 2px solid #eab308;
        border-radius: 14px; padding: 2rem; text-align: center; color: #fde047 !important;
    }
    .pred-card-tinggi {
        background: rgba(239, 68, 68, 0.15); border: 2px solid #ef4444;
        border-radius: 14px; padding: 2rem; text-align: center; color: #fca5a5 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# 2. DATA LOAD & PREPROCESSING PRESISI TINGGI (LOCKED)
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(
    BASE_DIR,
    "Dataset - jumlah-peserta-didik-putus-sekolah-menurut-tingkat-tiap-provinsi-2025-semua-wilayah-sd-mi-sederajat-1 - ASC.csv",
)
LOGO_PATH = os.path.join(BASE_DIR, "logo.png")
ROMAWI_LIST = ['I', 'II', 'III', 'IV', 'V', 'VI']
LEVEL_COLS = [f"Tingkat - {lvl}" for lvl in ROMAWI_LIST]

def get_image_base64(path):
    if os.path.exists(path):
        with open(path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    return None

@st.cache_data
def load_and_prep_data():
    if not os.path.exists(DATA_PATH):
        return None, None, None, None, None, None, None, None
    
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip() for c in df.columns]

    for col in LEVEL_COLS + ["Jumlah"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # ---------------------------------------------------------
    # K-MEANS ENGINE
    # ---------------------------------------------------------
    df_kmeans_tahunan = df.groupby(['Periode', 'Kode Kota/Kab', 'Kota/Kab'], as_index=False).agg({
        'Tingkat - I': 'sum', 'Tingkat - II': 'sum', 'Tingkat - III': 'sum',
        'Tingkat - IV': 'sum', 'Tingkat - V': 'sum', 'Tingkat - VI': 'sum',
        'Jumlah': 'sum'
    }).sort_values(['Kode Kota/Kab', 'Periode']).reset_index(drop=True)

    for romawi in ROMAWI_LIST:
        df_kmeans_tahunan[f'Proporsi_{romawi}'] = np.where(
            df_kmeans_tahunan['Jumlah'] > 0,
            df_kmeans_tahunan[f'Tingkat - {romawi}'] / df_kmeans_tahunan['Jumlah'],
            0
        )

    kmeans_data = df_kmeans_tahunan.groupby('Kode Kota/Kab', as_index=False).agg({
        'Proporsi_I': 'mean', 'Proporsi_II': 'mean', 'Proporsi_III': 'mean',
        'Proporsi_IV': 'mean', 'Proporsi_V': 'mean', 'Proporsi_VI': 'mean',
        'Jumlah': 'mean',
        'Kota/Kab': 'first'
    }).sort_values('Kode Kota/Kab').reset_index(drop=True)

    fitur_kmeans = ['Proporsi_I', 'Proporsi_II', 'Proporsi_III', 'Proporsi_IV', 'Proporsi_V', 'Proporsi_VI', 'Jumlah']
    
    scaler_kmeans = StandardScaler()
    X_kmeans_scaled = scaler_kmeans.fit_transform(kmeans_data[fitur_kmeans])
    
    kmeans_model = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans_data["Cluster_Id"] = kmeans_model.fit_predict(X_kmeans_scaled)

    cluster_means = kmeans_data.groupby("Cluster_Id")["Jumlah"].mean().sort_values()
    label_map = {
        cluster_means.index[0]: "Rendah",
        cluster_means.index[1]: "Sedang",
        cluster_means.index[2]: "Tinggi"
    }
    kmeans_data["Cluster"] = kmeans_data["Cluster_Id"].map(label_map)

    # ---------------------------------------------------------
    # RANDOM FOREST ENGINE (LOCKED ALIGNMENT)
    # ---------------------------------------------------------
    df_agregat = df.groupby(['Periode', 'Wilayah', 'Kota/Kab', 'Kode Kota/Kab', 'Status Sekolah'], as_index=False).agg({
        'Tingkat - I': 'sum', 'Tingkat - II': 'sum', 'Tingkat - III': 'sum',
        'Tingkat - IV': 'sum', 'Tingkat - V': 'sum', 'Tingkat - VI': 'sum',
        'Jumlah': 'sum'
    })

    for romawi in ROMAWI_LIST:
        df_agregat[f'Proporsi_{romawi}'] = np.where(
            df_agregat['Jumlah'] > 0,
            df_agregat[f'Tingkat - {romawi}'] / df_agregat['Jumlah'],
            0
        )

    # KUNCI UTAMA: Urutkan label encoder secara konsisten
    le_status = LabelEncoder()
    unique_status = np.sort(df_agregat['Status Sekolah'].astype(str).unique())
    le_status.fit(unique_status)
    df_agregat['Status_Encoded'] = le_status.transform(df_agregat['Status Sekolah'])

    # KUNCI UTAMA: Urutan baris presisi sebelum shift
    df_agregat = df_agregat.sort_values(['Kode Kota/Kab', 'Status Sekolah', 'Periode']).reset_index(drop=True)
    group_cols = ['Kode Kota/Kab', 'Status Sekolah']

    df_agregat['Jumlah_Lag1'] = df_agregat.groupby(group_cols)['Jumlah'].shift(1)
    for romawi in ROMAWI_LIST:
        df_agregat[f'Proporsi_{romawi}_Lag1'] = df_agregat.groupby(group_cols)[f'Proporsi_{romawi}'].shift(1)

    df_agregat['Selisih_Tahun'] = df_agregat['Periode'] - df_agregat.groupby(group_cols)['Periode'].shift(1)
    
    df_rf = df_agregat[df_agregat['Selisih_Tahun'] == 1].copy()

    all_years = sorted(df_rf['Periode'].unique())
    tahun_test = all_years[-1] # 2024
    
    train_mask = df_rf['Periode'] < tahun_test
    test_mask = df_rf['Periode'] == tahun_test
    
    Q1 = df_rf.loc[train_mask, 'Jumlah'].quantile(0.25)
    Q3 = df_rf.loc[train_mask, 'Jumlah'].quantile(0.75)
    
    def assign_target(x):
        if x <= Q1:
            return "Rendah"
        elif x <= Q3:
            return "Sedang"
        else:
            return "Tinggi"
            
    df_rf['Target'] = df_rf['Jumlah'].apply(assign_target)

    fitur_rf = ['Jumlah_Lag1', 'Proporsi_I_Lag1', 'Proporsi_II_Lag1', 'Proporsi_III_Lag1', 
                'Proporsi_IV_Lag1', 'Proporsi_V_Lag1', 'Proporsi_VI_Lag1', 'Status_Encoded', 'Periode']

    X_train = df_rf.loc[train_mask, fitur_rf]
    y_train = df_rf.loc[train_mask, 'Target']
    X_test = df_rf.loc[test_mask, fitur_rf]
    y_test = df_rf.loc[test_mask, 'Target']

    # KUNCI UTAMA: Parameter Random Forest Criterion Gini
    rf_model = RandomForestClassifier(
        n_estimators=100, 
        criterion='gini',
        random_state=42
    )
    rf_model.fit(X_train, y_train)

    y_pred = rf_model.predict(X_test)

    rf_metrics = {
        "acc": accuracy_score(y_test, y_pred),
        "prec": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "rec": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "baseline_acc": y_train.value_counts(normalize=True).max(),
        "cm": confusion_matrix(y_test, y_pred, labels=["Rendah", "Sedang", "Tinggi"]),
        "feature_importances": pd.Series(rf_model.feature_importances_, index=fitur_rf).sort_values(ascending=True),
        "y_test_count": len(y_test),
        "y_train_count": len(y_train)
    }

    return df, kmeans_data, scaler_kmeans, kmeans_model, rf_model, rf_metrics, le_status, df_rf

df_raw, df_kmeans, scaler_kmeans_obj, kmeans_obj, rf_model_obj, rf_results, le_status_obj, df_rf_debug = load_and_prep_data()

if df_raw is None:
    st.error(f"Dataset tidak ditemukan di path: `{DATA_PATH}`. Pastikan file CSV tersedia di folder utama.")
    st.stop()

# =========================================================
# 3. SIDEBAR NAVIGATION & LOGO HEADERS
# =========================================================
logo_b64 = get_image_base64(LOGO_PATH)

logo_img_html = (
    f'<div style="background-color: #ffffff; padding: 8px; border-radius: 10px; display: flex; align-items: center; justify-content: center; width: 64px; height: 64px; flex-shrink: 0; margin-right: 14px;">'
    f'<img src="data:image/png;base64,{logo_b64}" style="max-width: 100%; max-height: 100%; object-fit: contain;" />'
    f'</div>'
) if logo_b64 else ''

with st.sidebar:
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); padding: 1.2rem; border-radius: 12px; border: 1px solid #334155; margin-bottom: 1.4rem; display: flex; align-items: center;">
            {logo_img_html}
            <div>
                <div style="font-size: 0.68rem; font-weight: 800; color: #38bdf8; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.15rem;">
                    DATA MINING SYSTEM
                </div>
                <div style="font-size: 1.05rem; font-weight: 800; color: #ffffff; line-height: 1.2;">
                    Analisis Putus Sekolah
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<p style='font-size: 0.7rem; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.5rem;'>MAIN MENU</p>", unsafe_allow_html=True)

    menu = st.radio(
        label="Navigasi Utama",
        options=[
            "🏠 Overview Dashboard",
            "📊 Exploratory Analysis",
            "🧩 K-Means Clustering",
            "🌲 Random Forest Model",
            "🔮 Predictive Simulation",
            "🔍 Debug & Colab Alignment"
        ],
        label_visibility="collapsed"
    )

    st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="background: rgba(15, 23, 42, 0.6); padding: 0.9rem; border-radius: 8px; border: 1px dashed #334155;">
            <div style="font-size: 0.7rem; color: #64748b; font-weight: 600;">SYSTEM STATUS</div>
            <div style="font-size: 0.78rem; color: #38bdf8; font-weight: 700; margin-top: 0.1rem;">Fully Synchronized</div>
        </div>
        """,
        unsafe_allow_html=True
    )

NO_ZOOM = {'displayModeBar': False, 'scrollZoom': False}

# =========================================================
# HALAMAN 1: OVERVIEW DASHBOARD
# =========================================================
if menu == "🏠 Overview Dashboard":
    st.title("Executive Summary Dashboard")
    st.markdown("Ringkasan data utama tingkat nasional angka putus sekolah peserta didik SD/MI.")
    st.write("")
    
    c1, c2, c3, c4, c5 = st.columns(5)
    total_records = len(df_raw)
    total_prov = df_raw["Wilayah"].nunique() if "Wilayah" in df_raw.columns else 0
    periode_range = f"{df_raw['Periode'].min()} - {df_raw['Periode'].max()}" if "Periode" in df_raw.columns else "-"
    total_cases = int(df_raw["Jumlah"].sum())
    
    with c1:
        st.markdown(f"<div class='metric-box'><div class='metric-label'>TOTAL REKORD DATA</div><div class='metric-num'>{total_records:,}</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='metric-box'><div class='metric-label'>JUMLAH WILAYAH</div><div class='metric-num'>{total_prov}</div></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='metric-box'><div class='metric-label'>PERIODE DATA</div><div class='metric-num'>{periode_range}</div></div>", unsafe_allow_html=True)
    with c4:
        st.markdown(f"<div class='metric-box'><div class='metric-label'>TOTAL KASUS</div><div class='metric-num'>{total_cases:,}</div></div>", unsafe_allow_html=True)
    with c5:
        st.markdown(f"<div class='metric-box'><div class='metric-label'>RF TEST EVALUASI</div><div class='metric-num'>Tahun 2024</div></div>", unsafe_allow_html=True)

    st.write("")
    st.write("")

    col_chart1, col_chart2 = st.columns([1.2, 1], gap="large")

    with col_chart1:
        st.subheader("Tren Kasus Putus Sekolah Berdasarkan Tahun")
        if "Periode" in df_raw.columns:
            trend_df = df_raw.groupby("Periode", as_index=False)["Jumlah"].sum()
            fig_trend = px.line(trend_df, x="Periode", y="Jumlah", markers=True)
            fig_trend.update_traces(line_color="#38bdf8", line_width=3, marker_size=8)
            fig_trend.update_layout(
                height=380, font=dict(color="#f8fafc"),
                xaxis=dict(gridcolor="#334155", dtick=1, fixedrange=True),
                yaxis=dict(gridcolor="#334155", title="Jumlah Siswa", fixedrange=True),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=10, r=10, t=10, b=10)
            )
            st.plotly_chart(fig_trend, use_container_width=True, config=NO_ZOOM)

    with col_chart2:
        st.subheader("Top 10 Wilayah Kasus Tertinggi")
        top10_df = df_raw.groupby("Wilayah", as_index=False)["Jumlah"].sum().sort_values("Jumlah", ascending=False).head(10)
        fig_top = px.bar(top10_df, x="Jumlah", y="Wilayah", orientation="h", text_auto=",d")
        fig_top.update_traces(marker_color="#38bdf8")
        fig_top.update_layout(
            height=380, font=dict(color="#f8fafc"),
            yaxis=dict(autorange="reversed", gridcolor="#334155", title=None, fixedrange=True),
            xaxis=dict(gridcolor="#334155", title="Total Siswa", fixedrange=True),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig_top, use_container_width=True, config=NO_ZOOM)

# =========================================================
# HALAMAN 2: EXPLORATORY ANALYSIS
# =========================================================
elif menu == "📊 Exploratory Analysis":
    st.title("Exploratory Data Analysis (EDA)")
    st.markdown("Eksplorasi rinci karakteristik data berdasarkan jenjang kelas, status sekolah, dan korelasi antar fitur.")
    
    c_f1, c_f2 = st.columns(2)
    with c_f1:
        years = ["Semua Tahun"] + sorted(df_raw["Periode"].unique().tolist()) if "Periode" in df_raw.columns else ["Semua"]
        sel_year = st.selectbox("Filter Tahun Periode:", years)
    with c_f2:
        regions = ["Semua Wilayah"] + sorted(df_raw["Wilayah"].unique().tolist()) if "Wilayah" in df_raw.columns else ["Semua"]
        sel_region = st.selectbox("Filter Wilayah / Provinsi:", regions)

    filtered_eda = df_raw.copy()
    if sel_year != "Semua Tahun":
        filtered_eda = filtered_eda[filtered_eda["Periode"] == sel_year]
    if sel_region != "Semua Wilayah":
        filtered_eda = filtered_eda[filtered_eda["Wilayah"] == sel_region]

    st.write("")
    
    col_e1, col_e2 = st.columns(2, gap="large")

    with col_e1:
        st.subheader("Distribusi Putus Sekolah per Kelas (I - VI)")
        class_sums = filtered_eda[LEVEL_COLS].sum().reset_index()
        class_sums.columns = ["Tingkat Kelas", "Jumlah Siswa"]
        class_sums["Tingkat Kelas"] = class_sums["Tingkat Kelas"].str.replace("Tingkat - ", "Kelas ")
        
        fig_class = px.bar(class_sums, x="Tingkat Kelas", y="Jumlah Siswa", text_auto=",d", color_discrete_sequence=["#38bdf8"])
        fig_class.update_layout(
            height=360, font=dict(color="#f8fafc"),
            xaxis=dict(gridcolor="#334155", fixedrange=True),
            yaxis=dict(gridcolor="#334155", fixedrange=True),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig_class, use_container_width=True, config=NO_ZOOM)

    with col_e2:
        st.subheader("Perbandingan Status Sekolah (Negeri vs Swasta)")
        if "Status Sekolah" in filtered_eda.columns:
            status_df = filtered_eda.groupby("Status Sekolah", as_index=False)["Jumlah"].sum()
            fig_status = px.pie(status_df, names="Status Sekolah", values="Jumlah", hole=0.4, color_discrete_sequence=["#38bdf8", "#818cf8"])
            fig_status.update_layout(
                height=360, font=dict(color="#f8fafc"),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=10, r=10, t=10, b=10)
            )
            st.plotly_chart(fig_status, use_container_width=True, config=NO_ZOOM)

    st.markdown("---")
    st.subheader("Heatmap Korelasi Fitur")
    corr_matrix = filtered_eda[LEVEL_COLS + ["Jumlah"]].corr()
    fig_corr = px.imshow(
        corr_matrix, text_auto=".2f", aspect="auto",
        color_continuous_scale="Blues"
    )
    fig_corr.update_layout(
        height=400, font=dict(color="#f8fafc"),
        xaxis=dict(fixedrange=True), yaxis=dict(fixedrange=True),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_corr, use_
