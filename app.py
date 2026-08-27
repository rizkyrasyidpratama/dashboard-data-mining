import streamlit as st
import pandas as pd
import plotly.express as px

# Konfigurasi Halaman
import os

# Memastikan path logo relatif terhadap lokasi file app.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = os.path.join(BASE_DIR, "logo.png")
HAS_LOGO = os.path.exists(LOGO_PATH)

st.set_page_config(
    page_title="Dashboard Putus Sekolah",
    page_icon=LOGO_PATH if HAS_LOGO else "🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk tampilan premium
st.markdown("""
    <style>
    /* Global Styling */
    .main {
        background: #0f172a;
        font-family: 'Inter', sans-serif;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #f8fafc !important;
        font-weight: 800 !important;
    }
    p, span, div {
        color: #e2e8f0;
    }
    
    /* Metric Cards dengan efek Glassmorphism & Hover */
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        padding: 25px;
        border-radius: 16px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3), 0 4px 6px -2px rgba(0, 0, 0, 0.2);
        text-align: center;
        margin-bottom: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 25px -5px rgba(0, 242, 254, 0.15), 0 10px 10px -5px rgba(0, 242, 254, 0.1);
        border: 1px solid rgba(0, 242, 254, 0.3);
    }
    
    /* Typography di dalam Metric Card */
    .metric-title {
        color: #94a3b8;
        font-size: 14px;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 1px;
    }
    .metric-value {
        font-size: 38px;
        font-weight: 900;
        margin-top: 12px;
        background: -webkit-linear-gradient(45deg, #00f2fe, #4facfe, #a18cd1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Penyesuaian Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1e293b;
        border-right: 1px solid #334155;
    }
    
    /* Tabs & Streamlit Elements Styling Overrides */
    div[data-testid="stExpander"] {
        background-color: #1e293b !important;
        border-color: #334155 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Load Data
@st.cache_data
def load_data():
    file_path = "Dataset - jumlah-peserta-didik-putus-sekolah-menurut-tingkat-tiap-provinsi-2025-semua-wilayah-sd-mi-sederajat-1 - ASC.csv"
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        st.error(f"Gagal memuat data: {e}")
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.stop()

# Tampilkan Logo di Sidebar
if HAS_LOGO:
    st.sidebar.image(LOGO_PATH, use_container_width=True)
    st.sidebar.markdown("---")

# Judul Dashboard Utama
col_logo, col_title = st.columns([1, 11])
with col_logo:
    if HAS_LOGO:
        st.image(LOGO_PATH, width=70)
    else:
        st.markdown("<h1>🎓</h1>", unsafe_allow_html=True)

with col_title:
    st.title("Dashboard Analisis Putus Sekolah")
st.markdown("<p style='font-size: 18px; color: #94a3b8;'>Visualisasi dan eksplorasi data peserta didik putus sekolah tingkat SD/MI Sederajat berdasarkan dataset yang ada.</p>", unsafe_allow_html=True)

# Sidebar Filters
st.sidebar.header("🔍 Filter Data")

# Filter Tahun
tahun_list = sorted(df['Periode'].unique().tolist())
selected_tahun = st.sidebar.multiselect("Pilih Tahun (Periode)", tahun_list, default=tahun_list)

# Filter Status Sekolah
status_list = df['Status Sekolah'].unique().tolist()
selected_status = st.sidebar.multiselect("Pilih Status Sekolah", status_list, default=status_list)

# Filter Wilayah
wilayah_list = ["Semua Wilayah"] + sorted(df['Wilayah'].unique().tolist())
selected_wilayah = st.sidebar.selectbox("Pilih Wilayah", wilayah_list)

# Aplikasikan Filter
filtered_df = df[df['Periode'].isin(selected_tahun)]
filtered_df = filtered_df[filtered_df['Status Sekolah'].isin(selected_status)]
if selected_wilayah != "Semua Wilayah":
    filtered_df = filtered_df[filtered_df['Wilayah'] == selected_wilayah]

# KPI Cards
total_data = len(filtered_df)
total_putus_sekolah = filtered_df['Jumlah'].sum()
rata_putus_sekolah = filtered_df['Jumlah'].mean()

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Baris Data</div>
            <div class="metric-value">{total_data:,}</div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Siswa Putus Sekolah</div>
            <div class="metric-value">{total_putus_sekolah:,}</div>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Rata-rata Putus Sekolah per Area</div>
            <div class="metric-value">{rata_putus_sekolah:,.2f}</div>
        </div>
    """, unsafe_allow_html=True)


# Membuat Tab Layout
st.markdown("---")
tab1, tab2, tab3 = st.tabs(["📊 Ringkasan & Tren", "🏫 Analisis Kelas", "📋 Detail Data"])

# ================= TAB 1: RINGKASAN & TREN =================
with tab1:
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.subheader("📈 Distribusi Kasus per Tahun")
        if not filtered_df.empty:
            trend_data = filtered_df.groupby('Periode')['Jumlah'].sum().reset_index()
            fig_trend = px.line(trend_data, x='Periode', y='Jumlah', markers=True, 
                                title="Total Siswa Putus Sekolah per Tahun",
                                color_discrete_sequence=['#00f2fe'],
                                template='plotly_dark')
            fig_trend.update_xaxes(type='category')
            fig_trend.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("Tidak ada data untuk ditampilkan.")
    
    with col_chart2:
        st.subheader("🏫 Kasus Berdasarkan Status Sekolah")
        if not filtered_df.empty:
            status_data = filtered_df.groupby('Status Sekolah')['Jumlah'].sum().reset_index()
            fig_pie = px.pie(status_data, values='Jumlah', names='Status Sekolah', 
                             title="Persentase Kasus: Negeri vs Swasta",
                             color_discrete_sequence=['#4facfe', '#a18cd1'],
                             hole=0.4,
                             template='plotly_dark')
            fig_pie.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("Tidak ada data untuk ditampilkan.")

# ================= TAB 2: ANALISIS KELAS =================
with tab2:
    st.subheader("📊 Analisis Berdasarkan Tingkat Kelas")
    tingkat_cols = ['Tingkat - I', 'Tingkat - II', 'Tingkat - III', 'Tingkat - IV', 'Tingkat - V', 'Tingkat - VI']
    
    if not filtered_df.empty:
        tingkat_data = filtered_df[tingkat_cols].sum().reset_index()
        tingkat_data.columns = ['Tingkat Kelas', 'Total Putus Sekolah']
        
        fig_bar = px.bar(tingkat_data, x='Tingkat Kelas', y='Total Putus Sekolah',
                         title="Total Siswa Putus Sekolah di Setiap Tingkat Kelas",
                         color='Total Putus Sekolah', color_continuous_scale='PuBu',
                         template='plotly_dark')
        fig_bar.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("Tidak ada data untuk ditampilkan.")

# ================= TAB 3: DETAIL DATA =================
with tab3:
    st.subheader("📋 Detail Tabel Data")
    if not filtered_df.empty:
        # Tombol Download Data
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Unduh Data (CSV)",
            data=csv,
            file_name='data_putus_sekolah_filtered.csv',
            mime='text/csv',
        )
        
        st.dataframe(filtered_df, use_container_width=True, height=500)
    else:
        st.info("Tidak ada data untuk ditampilkan.")

