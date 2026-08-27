import streamlit as st
import pandas as pd
import plotly.express as px

# Konfigurasi Halaman
st.set_page_config(
    page_title="Dashboard Putus Sekolah",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk tampilan premium
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        margin-bottom: 20px;
    }
    .metric-title {
        color: #6c757d;
        font-size: 14px;
        text-transform: uppercase;
        font-weight: bold;
    }
    .metric-value {
        color: #2c3e50;
        font-size: 32px;
        font-weight: bold;
        margin-top: 10px;
    }
    h1, h2, h3 {
        color: #2c3e50;
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

# Judul Dashboard
st.title("🎓 Dashboard Analisis Putus Sekolah")
st.markdown("Visualisasi dan eksplorasi data peserta didik putus sekolah tingkat SD/MI Sederajat berdasarkan dataset yang ada.")

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


# Visualisasi Utama
st.markdown("---")

col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    st.subheader("📈 Distribusi Kasus per Tahun")
    if not filtered_df.empty:
        trend_data = filtered_df.groupby('Periode')['Jumlah'].sum().reset_index()
        fig_trend = px.line(trend_data, x='Periode', y='Jumlah', markers=True, 
                            title="Total Siswa Putus Sekolah per Tahun",
                            color_discrete_sequence=['#3498DB'])
        fig_trend.update_xaxes(type='category')
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.info("Tidak ada data untuk ditampilkan.")

with col_chart2:
    st.subheader("🏫 Kasus Berdasarkan Status Sekolah")
    if not filtered_df.empty:
        status_data = filtered_df.groupby('Status Sekolah')['Jumlah'].sum().reset_index()
        fig_pie = px.pie(status_data, values='Jumlah', names='Status Sekolah', 
                         title="Persentase Kasus: Negeri vs Swasta",
                         color_discrete_sequence=['#2ECC71', '#E74C3C'],
                         hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("Tidak ada data untuk ditampilkan.")

# Analisis Tingkat Kelas
st.markdown("---")
st.subheader("📊 Analisis Berdasarkan Tingkat Kelas")

tingkat_cols = ['Tingkat - I', 'Tingkat - II', 'Tingkat - III', 'Tingkat - IV', 'Tingkat - V', 'Tingkat - VI']
if not filtered_df.empty:
    tingkat_data = filtered_df[tingkat_cols].sum().reset_index()
    tingkat_data.columns = ['Tingkat Kelas', 'Total Putus Sekolah']
    
    fig_bar = px.bar(tingkat_data, x='Tingkat Kelas', y='Total Putus Sekolah',
                     title="Total Siswa Putus Sekolah di Setiap Tingkat Kelas",
                     color='Total Putus Sekolah', color_continuous_scale='Blues')
    st.plotly_chart(fig_bar, use_container_width=True)
else:
    st.info("Tidak ada data untuk ditampilkan.")

# Data Tabel
st.markdown("---")
st.subheader("📋 Detail Data")
st.dataframe(filtered_df, use_container_width=True)
