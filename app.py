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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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
# 2. DATA LOAD & PREPROCESSING CACHE
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(
    BASE_DIR,
    "Dataset - jumlah-peserta-didik-putus-sekolah-menurut-tingkat-tiap-provinsi-2025-semua-wilayah-sd-mi-sederajat-1 - ASC.csv",
)
LOGO_PATH = os.path.join(BASE_DIR, "logo.png")
LEVEL_COLS = [f"Tingkat - {lvl}" for lvl in ["I", "II", "III", "IV", "V", "VI"]]

def get_image_base64(path):
    if os.path.exists(path):
        with open(path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    return None

@st.cache_data
def load_and_prep_data():
    if not os.path.exists(DATA_PATH):
        return None, None, None, None, None, None
    
    df = pd.read_csv(DATA_PATH)
    
    for col in LEVEL_COLS + ["Jumlah"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            
    feature_cols = [c for c in LEVEL_COLS if c in df.columns]
    
    # 1. K-MEANS CLUSTERING (K=3)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols])
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df["Cluster_Id"] = kmeans.fit_predict(X_scaled)
    
    cluster_means = df.groupby("Cluster_Id")["Jumlah"].mean().sort_values()
    label_map = {
        cluster_means.index[0]: "Rendah",
        cluster_means.index[1]: "Sedang",
        cluster_means.index[2]: "Tinggi"
    }
    df["Cluster"] = df["Cluster_Id"].map(label_map)
    
    # 2. RANDOM FOREST CLASSIFICATION
    X = df[feature_cols]
    y = df["Cluster"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    y_pred = rf_model.predict(X_test)
    
    rf_metrics = {
        "acc": accuracy_score(y_test, y_pred),
        "prec": precision_score(y_test, y_pred, average="weighted"),
        "rec": recall_score(y_test, y_pred, average="weighted"),
        "f1": f1_score(y_test, y_pred, average="weighted"),
        "baseline_acc": y_train.value_counts(normalize=True).max(),
        "cm": confusion_matrix(y_test, y_pred, labels=["Rendah", "Sedang", "Tinggi"]),
        "feature_importances": pd.Series(rf_model.feature_importances_, index=feature_cols).sort_values(ascending=True)
    }
    
    return df, feature_cols, scaler, kmeans, rf_model, rf_metrics

df_data, FEATURE_COLS, scaler_obj, kmeans_obj, rf_model_obj, rf_results = load_and_prep_data()

if df_data is None:
    st.error(f"Dataset tidak ditemukan di path: `{DATA_PATH}`. Pastikan file CSV tersedia di folder utama.")
    st.stop()

# =========================================================
# 3. SIDEBAR CUSTOM NAVIGATION & LOGO HEADERS
# =========================================================
logo_b64 = get_image_base64(LOGO_PATH)

# Logo dalam container putih ukuran besar (64px x 64px)
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
            "🔮 Predictive Simulation"
        ],
        label_visibility="collapsed"
    )

    st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
    
    st.markdown(
        """
        <div style="background: rgba(15, 23, 42, 0.6); padding: 0.9rem; border-radius: 8px; border: 1px dashed #334155;">
            <div style="font-size: 0.7rem; color: #64748b; font-weight: 600;">SYSTEM STATUS</div>
            <div style="font-size: 0.78rem; color: #38bdf8; font-weight: 700; margin-top: 0.1rem;">⚡ Fully Operational</div>
            <div style="font-size: 0.68rem; color: #94a3b8; margin-top: 0.3rem;">Engine: Scikit-Learn v1.2+</div>
        </div>
        """,
        unsafe_allow_html=True
    )

NO_ZOOM = {
    'displayModeBar': False,
    'scrollZoom': False
}

# =========================================================
# HALAMAN 1: OVERVIEW DASHBOARD
# =========================================================
if menu == "🏠 Overview Dashboard":
    st.title("🏠 Executive Summary Dashboard")
    st.markdown("Ringkasan data utama tingkat nasional angka putus sekolah peserta didik SD/MI.")
    st.write("")
    
    c1, c2, c3, c4, c5 = st.columns(5)
    total_records = len(df_data)
    total_prov = df_data["Wilayah"].nunique() if "Wilayah" in df_data.columns else 0
    periode_range = f"{df_data['Periode'].min()} - {df_data['Periode'].max()}" if "Periode" in df_data.columns else "-"
    total_cases = int(df_data["Jumlah"].sum())
    
    with c1:
        st.markdown(f"<div class='metric-box'><div class='metric-label'>TOTAL REKORD DATA</div><div class='metric-num'>{total_records:,}</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='metric-box'><div class='metric-label'>JUMLAH WILAYAH</div><div class='metric-num'>{total_prov}</div></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='metric-box'><div class='metric-label'>PERIODE DATA</div><div class='metric-num'>{periode_range}</div></div>", unsafe_allow_html=True)
    with c4:
        st.markdown(f"<div class='metric-box'><div class='metric-label'>TOTAL KASUS</div><div class='metric-num'>{total_cases:,}</div></div>", unsafe_allow_html=True)
    with c5:
        st.markdown(f"<div class='metric-box'><div class='metric-label'>RF TEST EVALUASI</div><div class='metric-num'>20% Data</div></div>", unsafe_allow_html=True)

    st.write("")
    st.write("")

    col_chart1, col_chart2 = st.columns([1.2, 1], gap="large")

    with col_chart1:
        st.subheader("📈 Tren Kasus Putus Sekolah Berdasarkan Tahun")
        if "Periode" in df_data.columns:
            trend_df = df_data.groupby("Periode", as_index=False)["Jumlah"].sum()
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
        st.subheader("🏆 Top 10 Wilayah Kasus Tertinggi")
        top10_df = df_data.groupby("Wilayah", as_index=False)["Jumlah"].sum().sort_values("Jumlah", ascending=False).head(10)
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
    st.title("📊 Exploratory Data Analysis (EDA)")
    st.markdown("Eksplorasi rinci karakteristik data berdasarkan jenjang kelas, status sekolah, dan korelasi antar fitur.")
    
    c_f1, c_f2 = st.columns(2)
    with c_f1:
        years = ["Semua Tahun"] + sorted(df_data["Periode"].unique().tolist()) if "Periode" in df_data.columns else ["Semua"]
        sel_year = st.selectbox("Filter Tahun Periode:", years)
    with c_f2:
        regions = ["Semua Wilayah"] + sorted(df_data["Wilayah"].unique().tolist()) if "Wilayah" in df_data.columns else ["Semua"]
        sel_region = st.selectbox("Filter Wilayah / Provinsi:", regions)

    filtered_eda = df_data.copy()
    if sel_year != "Semua Tahun":
        filtered_eda = filtered_eda[filtered_eda["Periode"] == sel_year]
    if sel_region != "Semua Wilayah":
        filtered_eda = filtered_eda[filtered_eda["Wilayah"] == sel_region]

    st.write("")
    
    col_e1, col_e2 = st.columns(2, gap="large")

    with col_e1:
        st.subheader("🏫 Distribusi Putus Sekolah per Kelas (I - VI)")
        class_sums = filtered_eda[FEATURE_COLS].sum().reset_index()
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
        st.subheader("🏫 Perbandingan Status Sekolah (Negeri vs Swasta)")
        if "Status Sekolah" in filtered_eda.columns:
            status_df = filtered_eda.groupby("Status Sekolah", as_index=False)["Jumlah"].sum()
            fig_status = px.pie(status_df, names="Status Sekolah", values="Jumlah", hole=0.4, color_discrete_sequence=["#38bdf8", "#818cf8"])
            fig_status.update_layout(
                height=360, font=dict(color="#f8fafc"),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=10, r=10, t=10, b=10)
            )
            st.plotly_chart(fig_status, use_container_width=True, config=NO_ZOOM)

    st.markdown("---")
    st.subheader("🔥 Heatmap Korelasi Fitur")
    corr_matrix = filtered_eda[FEATURE_COLS + ["Jumlah"]].corr()
    fig_corr = px.imshow(
        corr_matrix, text_auto=".2f", aspect="auto",
        color_continuous_scale="Blues"
    )
    fig_corr.update_layout(
        height=400, font=dict(color="#f8fafc"),
        xaxis=dict(fixedrange=True), yaxis=dict(fixedrange=True),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_corr, use_container_width=True, config=NO_ZOOM)

# =========================================================
# HALAMAN 3: K-MEANS CLUSTERING
# =========================================================
elif menu == "🧩 K-Means Clustering":
    st.title("🧩 K-Means Clustering Analysis")
    st.markdown("Pengelompokan pola angka putus sekolah wilayah berdasarkan analisis Unsupervised Learning.")
    
    scaler_tmp = StandardScaler()
    X_sc = scaler_tmp.fit_transform(df_data[FEATURE_COLS])
    
    sil = silhouette_score(X_sc, df_data["Cluster_Id"])
    ch = calinski_harabasz_score(X_sc, df_data["Cluster_Id"])
    db = davies_bouldin_score(X_sc, df_data["Cluster_Id"])

    c_m1, c_m2, c_m3, c_m4 = st.columns(4)
    with c_m1:
        st.markdown("<div class='metric-box'><div class='metric-label'>OPTIMAL K</div><div class='metric-num'>K = 3</div><div class='metric-sub'>Rendah, Sedang, Tinggi</div></div>", unsafe_allow_html=True)
    with c_m2:
        st.markdown(f"<div class='metric-box'><div class='metric-label'>SILHOUETTE SCORE</div><div class='metric-num'>{sil:.3f}</div><div class='metric-sub'>Struktur Cluster Sangat Baik</div></div>", unsafe_allow_html=True)
    with c_m3:
        st.markdown(f"<div class='metric-box'><div class='metric-label'>CALINSKI-HARABASZ</div><div class='metric-num'>{ch:,.1f}</div><div class='metric-sub'>Separasi Maksimal</div></div>", unsafe_allow_html=True)
    with c_m4:
        st.markdown(f"<div class='metric-box'><div class='metric-label'>DAVIES-BOULDIN</div><div class='metric-num'>{db:.3f}</div><div class='metric-sub'>Similaritas Antar Cluster</div></div>", unsafe_allow_html=True)

    st.write("")
    
    tab_c1, tab_c2 = st.tabs(["📉 Elbow Method", "🗺️ PCA 2D Visualizer"])
    
    with tab_c1:
        col_el, col_dist = st.columns([1.2, 1], gap="large")
        with col_el:
            st.subheader("Elbow Method (Inertia vs K)")
            inertias = []
            K_range = range(2, 7)
            for k in K_range:
                km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_sc)
                inertias.append(km.inertia_)
                
            fig_elbow = px.line(x=list(K_range), y=inertias, markers=True, labels={"x": "Jumlah Cluster (K)", "y": "Inertia"})
            fig_elbow.update_traces(line_color="#38bdf8", marker_size=10)
            fig_elbow.add_vline(x=3, line_dash="dash", line_color="#ef4444", annotation_text="K Optimal = 3")
            fig_elbow.update_layout(
                height=350, font=dict(color="#f8fafc"),
                xaxis=dict(gridcolor="#334155", fixedrange=True),
                yaxis=dict(gridcolor="#334155", fixedrange=True),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_elbow, use_container_width=True, config=NO_ZOOM)
            
        with col_dist:
            st.subheader("Distribusi Anggota Cluster")
            dist_df = df_data["Cluster"].value_counts().reset_index()
            dist_df.columns = ["Cluster", "Jumlah Wilayah"]
            fig_dist = px.pie(dist_df, names="Cluster", values="Jumlah Wilayah", hole=0.4,
                              color="Cluster", color_discrete_map={"Rendah": "#22c55e", "Sedang": "#eab308", "Tinggi": "#ef4444"})
            fig_dist.update_layout(
                height=350, font=dict(color="#f8fafc"),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_dist, use_container_width=True, config=NO_ZOOM)

    with tab_c2:
        st.subheader("Visualisasi Proyeksi 2D PCA")
        st.caption("Catatan: PCA digunakan murni untuk memproyeksikan data ke dalam 2 dimensi visualisasi, bukan untuk proses clustering.")
        
        pca = PCA(n_components=2)
        pca_coords = pca.fit_transform(X_sc)
        df_pca = df_data.copy()
        df_pca["PC1"] = pca_coords[:, 0]
        df_pca["PC2"] = pca_coords[:, 1]
        
        fig_pca = px.scatter(
            df_pca, x="PC1", y="PC2", color="Cluster",
            hover_data=["Wilayah", "Jumlah"] if "Wilayah" in df_pca.columns else ["Jumlah"],
            color_discrete_map={"Rendah": "#22c55e", "Sedang": "#eab308", "Tinggi": "#ef4444"}
        )
        fig_pca.update_traces(marker=dict(size=9, opacity=0.8))
        fig_pca.update_layout(
            height=420, font=dict(color="#f8fafc"),
            xaxis=dict(gridcolor="#334155", fixedrange=True),
            yaxis=dict(gridcolor="#334155", fixedrange=True),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_pca, use_container_width=True, config=NO_ZOOM)

    st.markdown("---")
    st.subheader("📊 Rata-Rata Karakteristik Cluster")
    cluster_profile = df_data.groupby("Cluster")[FEATURE_COLS + ["Jumlah"]].mean().reindex(["Rendah", "Sedang", "Tinggi"])
    st.dataframe(cluster_profile.style.format("{:,.1f}"), use_container_width=True)

# =========================================================
# HALAMAN 4: RANDOM FOREST MODEL
# =========================================================
elif menu == "🌲 Random Forest Model":
    st.title("🌲 Random Forest Classification")
    st.markdown("Evaluasi performa model Supervised Learning untuk mengklasifikasikan kategori tingkat kerawanan wilayah.")
    
    m = rf_results
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"<div class='metric-box'><div class='metric-label'>ACCURACY</div><div class='metric-num'>{m['acc']*100:.1f}%</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='metric-box'><div class='metric-label'>PRECISION</div><div class='metric-num'>{m['prec']*100:.1f}%</div></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='metric-box'><div class='metric-label'>RECALL</div><div class='metric-num'>{m['rec']*100:.1f}%</div></div>", unsafe_allow_html=True)
    with c4:
        st.markdown(f"<div class='metric-box'><div class='metric-label'>F1-SCORE</div><div class='metric-num'>{m['f1']*100:.1f}%</div></div>", unsafe_allow_html=True)
    with c5:
        imp = (m['acc'] - m['baseline_acc']) * 100
        st.markdown(f"<div class='metric-box'><div class='metric-label'>AKURASI VS BASELINE</div><div class='metric-num'>+{imp:.1f}%</div></div>", unsafe_allow_html=True)

    st.write("")
    
    col_rf1, col_rf2 = st.columns([1, 1.2], gap="large")
    
    with col_rf1:
        st.subheader("📌 Confusion Matrix")
        labels = ["Rendah", "Sedang", "Tinggi"]
        fig_cm = px.imshow(
            m["cm"], x=labels, y=labels, text_auto=True,
            color_continuous_scale="Blues", labels=dict(x="Prediksi Model", y="Kelas Aktual")
        )
        fig_cm.update_layout(
            height=360, font=dict(color="#f8fafc"),
            xaxis=dict(fixedrange=True), yaxis=dict(fixedrange=True),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_cm, use_container_width=True, config=NO_ZOOM)

    with col_rf2:
        st.subheader("⭐ Feature Importance")
        fi_df = m["feature_importances"].reset_index()
        fi_df.columns = ["Fitur", "Importance"]
        fi_df["Fitur"] = fi_df["Fitur"].str.replace("Tingkat - ", "Kelas ")
        
        fig_fi = px.bar(fi_df, x="Importance", y="Fitur", orientation="h", text_auto=".3f", color_discrete_sequence=["#38bdf8"])
        fig_fi.update_layout(
            height=360, font=dict(color="#f8fafc"),
            xaxis=dict(gridcolor="#334155", fixedrange=True),
            yaxis=dict(gridcolor="#334155", title=None, fixedrange=True),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_fi, use_container_width=True, config=NO_ZOOM)
        st.caption("ℹ️ Feature importance menunjukkan kontribusi relatif fitur terhadap keputusan klasifikasi model, bukan hubungan sebab-akibat langsung.")

# =========================================================
# HALAMAN 5: PREDICTIVE SIMULATION
# =========================================================
elif menu == "🔮 Predictive Simulation":
    st.title("🔮 Predictive Simulation")
    st.markdown("Simulasi prediksi kategori tingkat kerawanan wilayah menggunakan model Random Forest yang telah dilatih.")
    
    st.write("")
    
    with st.form("pred_form"):
        st.subheader("📝 Input Jumlah Siswa Putus Sekolah per Tingkat Kelas")
        
        c_i1, c_i2, c_i3 = st.columns(3)
        with c_i1:
            val_k1 = st.number_input("Jumlah Siswa Kelas I", min_value=0, value=150, step=10)
            val_k2 = st.number_input("Jumlah Siswa Kelas II", min_value=0, value=120, step=10)
        with c_i2:
            val_k3 = st.number_input("Jumlah Siswa Kelas III", min_value=0, value=180, step=10)
            val_k4 = st.number_input("Jumlah Siswa Kelas IV", min_value=0, value=200, step=10)
        with c_i3:
            val_k5 = st.number_input("Jumlah Siswa Kelas V", min_value=0, value=210, step=10)
            val_k6 = st.number_input("Jumlah Siswa Kelas VI", min_value=0, value=90, step=10)
            
        st.write("")
        submit_btn = st.form_submit_button("🔮 PREDIKSI KATEGORI WILAYAH", use_container_width=True)

    if submit_btn:
        input_data = pd.DataFrame([[val_k1, val_k2, val_k3, val_k4, val_k5, val_k6]], columns=FEATURE_COLS)
        pred_label = rf_model_obj.predict(input_data)[0]
        pred_proba = rf_model_obj.predict_proba(input_data)[0]
        max_proba = max(pred_proba) * 100
        
        st.markdown("---")
        st.subheader("🎯 Hasil Klasifikasi Prediksi")
        
        card_class = f"pred-card-{pred_label.lower()}"
        
        st.markdown(
            f"""
            <div class='{card_class}'>
                <h3 style='margin:0; font-size:1.1rem; opacity:0.9;'>KATEGORI TINGKAT KERAWANAN</h3>
                <h1 style='font-size: 3.5rem; font-weight: 800; margin: 0.5rem 0;'>{pred_label.upper()}</h1>
                <p style='margin:0; font-size:1rem; font-weight:600;'>Tingkat Keyakinan Model (Confidence): {max_proba:.1f}%</p>
            </div>
            """,
            unsafe_allow_html=True
        )
