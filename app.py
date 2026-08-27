import base64
import os
import pandas as pd
import plotly.express as px
import streamlit as st

# =========================================================
# 1. SETUP PATH & LOGO BASE64
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(
    BASE_DIR,
    "Dataset - jumlah-peserta-didik-putus-sekolah-menurut-tingkat-tiap-provinsi-2025-semua-wilayah-sd-mi-sederajat-1 - ASC.csv",
)
LOGO_PATH = os.path.join(BASE_DIR, "logo.png")
LEVEL_COLUMNS = [f"Tingkat - {level}" for level in ["I", "II", "III", "IV", "V", "VI"]]

def get_image_base64(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""

LOGO_BASE64 = get_image_base64(LOGO_PATH)

st.set_page_config(
    page_title="Dashboard Data Mining - Putus Sekolah",
    page_icon="🎓",
    layout="wide",
)

# =========================================================
# 2. STYLING CSS SLATE DARK (EMPUK DI MATA & KONTRASTING)
# =========================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Background Utama Soft Dark Slate */
    .stApp { 
        background-color: #0f172a !important; 
        color: #f8fafc !important; 
    }

    /* Sidebar Dark Slate */
    [data-testid="stSidebar"] {
        background-color: #1e293b !important;
        border-right: 1px solid #334155 !important;
    }

    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3,
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown span {
        color: #f8fafc !important;
    }

    /* Selectbox Style Adjustment */
    div[data-baseweb="select"] > div {
        background-color: #0f172a !important;
        border-color: #334155 !important;
        color: #ffffff !important;
    }

    /* Brand Card Sidebar */
    .sidebar-brand-card {
        background: #0f172a;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.8rem;
    }
    .sidebar-brand-title {
        color: #38bdf8;
        font-size: 0.68rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .sidebar-brand-subtitle {
        color: #ffffff;
        font-size: 0.95rem;
        font-weight: 700;
        line-height: 1.2;
    }

    /* Tab Header Custom Fix */
    .stTabs [data-baseweb="tab"] p {
        color: #94a3b8 !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
    }
    .stTabs [aria-selected="true"][data-baseweb="tab"] p {
        color: #38bdf8 !important;
    }

    /* Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        padding: 2rem 2.2rem;
        border-radius: 14px;
        color: #ffffff;
        margin-bottom: 1.5rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .hero-title {
        color: #ffffff !important;
        font-size: 2.1rem !important;
        line-height: 1.2 !important;
        margin: 0.2rem 0 0.4rem 0;
        font-weight: 800 !important;
    }

    /* Kartu Metrik Dark Slate */
    .metric-box {
        background: #1e293b;
        border: 1px solid #334155;
        border-left: 4px solid #38bdf8;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.2);
    }
    .metric-label {
        color: #94a3b8;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-num {
        color: #ffffff;
        font-size: 1.75rem;
        font-weight: 800;
        margin-top: 0.2rem;
    }

    /* Ranking Card Dark Custom */
    .rank-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 0.5rem 1rem;
    }
    .rank-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.75rem 0;
        border-bottom: 1px solid #334155;
        font-size: 0.85rem;
    }
    .rank-item:last-child {
        border-bottom: none;
    }
    .rank-name {
        font-weight: 700;
        color: #f1f5f9;
    }
    .rank-val {
        font-weight: 800;
        color: #38bdf8;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# 3. LOAD DATASET
# =========================================================
@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        return pd.DataFrame()
    try:
        return pd.read_csv(DATA_PATH)
    except Exception:
        return pd.DataFrame()

df = load_data()

if df.empty:
    uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        st.stop()

logo_html_hero = f'<img src="data:image/png;base64,{LOGO_BASE64}" style="max-height: 85px; max-width: 180px; object-fit: contain; filter: brightness(0) invert(1);"/>' if LOGO_BASE64 else '<div style="font-size: 3rem;">🎓</div>'
logo_html_sidebar = f'<img src="data:image/png;base64,{LOGO_BASE64}" style="max-height: 45px; max-width: 45px; object-fit: contain; filter: brightness(0) invert(1);"/>' if LOGO_BASE64 else '<div style="font-size: 1.8rem;">🎓</div>'

# =========================================================
# 4. SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown(
        f"""
        <div class='sidebar-brand-card'>
            {logo_html_sidebar}
            <div>
                <div class='sidebar-brand-title'>PROYEK MATA KULIAH</div>
                <div class='sidebar-brand-subtitle'>Data Mining & Eksplorasi</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("⚙️ Control Panel Filter")

    years = ["Semua Tahun"] + (sorted(df["Periode"].unique().tolist()) if "Periode" in df.columns else [])
    selected_year = st.selectbox("Periode Tahun", years)

    statuses = ["Semua Status"] + (sorted(df["Status Sekolah"].dropna().unique().tolist()) if "Status Sekolah" in df.columns else [])
    selected_status = st.selectbox("Status Sekolah", statuses)

    regions = ["Semua Wilayah"] + (sorted(df["Wilayah"].dropna().unique().tolist()) if "Wilayah" in df.columns else [])
    selected_region = st.selectbox("Cakupan Wilayah", regions)

    st.markdown("---")
    st.caption("Visualisasi Data Mining • SD/MI Sederajat")

# Filter Dataframe
filtered_df = df.copy()
if selected_year != "Semua Tahun":
    filtered_df = filtered_df[filtered_df["Periode"] == selected_year]
if selected_status != "Semua Status":
    filtered_df = filtered_df[filtered_df["Status Sekolah"] == selected_status]
if selected_region != "Semua Wilayah":
    filtered_df = filtered_df[filtered_df["Wilayah"] == selected_region]

# =========================================================
# 5. HEADER BANNER & METRICS
# =========================================================
st.markdown(
    f"""
    <div class='hero-banner'>
        <div>
            <span style="color: #38bdf8; font-weight: 800; text-transform: uppercase; font-size: 0.75rem;">Hasil Analisis Data Mining</span>
            <div class='hero-title'>Visualisasi Peserta Didik Putus Sekolah</div>
            <p style="color: #94a3b8; margin: 0; font-size: 0.9rem;">Dashboard eksplorasi pola angka putus sekolah tingkat SD/MI berdasarkan wilayah, status sekolah, dan jenjang kelas.</p>
        </div>
        <div style="padding-left: 1.5rem;">
            {logo_html_hero}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.info(f"**FILTER AKTIF:** Tahun ({selected_year}) • Status ({selected_status}) • Wilayah ({selected_region})")

c1, c2, c3, c4 = st.columns(4)
total_cases = int(filtered_df["Jumlah"].sum()) if "Jumlah" in filtered_df.columns else 0
avg_cases = filtered_df["Jumlah"].mean() if "Jumlah" in filtered_df.columns and not filtered_df.empty else 0

with c1:
    st.markdown(f"<div class='metric-box'><div class='metric-label'>TOTAL KASUS</div><div class='metric-num'>{total_cases:,}</div></div>", unsafe_allow_html=True)
with c2:
    st.markdown(f"<div class='metric-box'><div class='metric-label'>BARIS TERPILIH</div><div class='metric-num'>{len(filtered_df):,}</div></div>", unsafe_allow_html=True)
with c3:
    st.markdown(f"<div class='metric-box'><div class='metric-label'>RATA-RATA PER AREA</div><div class='metric-num'>{avg_cases:,.1f}</div></div>", unsafe_allow_html=True)
with c4:
    st.markdown(f"<div class='metric-box'><div class='metric-label'>WILAYAH TERCAKUP</div><div class='metric-num'>{filtered_df['Wilayah'].nunique() if 'Wilayah' in filtered_df.columns else 0}</div></div>", unsafe_allow_html=True)

st.write("")

NO_ZOOM = {'displayModeBar': False, 'scrollZoom': False}

# =========================================================
# 6. TAB CONTENT
# =========================================================
tab1, tab2, tab3 = st.tabs(["📊 Gambaran Utama", "🏫 Profil Tingkat Kelas", "📑 Eksplorasi Data"])

with tab1:
    col_graph, col_rank = st.columns([2, 1], gap="large")

    with col_graph:
        st.subheader("Tren Pergerakan Kasus")
        if not filtered_df.empty:
            t_data = filtered_df.groupby("Periode", as_index=False)["Jumlah"].sum()
            fig_trend = px.line(t_data, x="Periode", y="Jumlah", markers=True)
            fig_trend.update_traces(line_color="#38bdf8", line_width=3)
            fig_trend.update_layout(
                height=340,
                font=dict(color="#94a3b8"),
                xaxis=dict(fixedrange=True, title=None, dtick=1, gridcolor="#334155"),
                yaxis=dict(fixedrange=True, title="Jumlah Siswa", gridcolor="#334155"),
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_trend, use_container_width=True, config=NO_ZOOM)

    with col_rank:
        st.subheader("Ranking Wilayah Tertinggi")
        if not filtered_df.empty:
            rank_df = filtered_df.groupby("Wilayah", as_index=False)["Jumlah"].sum().sort_values("Jumlah", ascending=False).head(5)
            
            # HTML di-string rapat tanpa tab/spasi depan agar tidak dirender sebagai code block oleh Markdown
            rank_items = "".join(
                f"<div class='rank-item'><span class='rank-name'>{row['Wilayah']}</span><span class='rank-val'>{int(row['Jumlah']):,}</span></div>"
                for _, row in rank_df.iterrows()
            )
            rank_html = f"<div class='rank-card'>{rank_items}</div>"
            st.markdown(rank_html, unsafe_allow_html=True)

with tab2:
    st.subheader("Distribusi Berdasarkan Kelas (I - VI)")
    avail_levels = [c for c in LEVEL_COLUMNS if c in filtered_df.columns]
    if avail_levels and not filtered_df.empty:
        l_data = filtered_df[avail_levels].sum().reset_index()
        l_data.columns = ["Kelas", "Jumlah"]
        l_data["Kelas"] = l_data["Kelas"].str.replace("Tingkat - ", "Kelas ", regex=False)
        
        fig_bar = px.bar(l_data, x="Kelas", y="Jumlah", text_auto=True)
        fig_bar.update_traces(marker_color="#38bdf8")
        fig_bar.update_layout(
            height=380,
            font=dict(color="#94a3b8"),
            xaxis=dict(fixedrange=True, gridcolor="#334155"),
            yaxis=dict(fixedrange=True, gridcolor="#334155"),
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_bar, use_container_width=True, config=NO_ZOOM)

with tab3:
    st.subheader("Tabel Raw Data")
    st.dataframe(filtered_df, use_container_width=True, hide_index=True)