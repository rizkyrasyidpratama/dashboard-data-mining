import base64
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    calinski_harabasz_score,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    precision_score,
    recall_score,
    silhouette_score,
    classification_report
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from collections import Counter
import itertools

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

    [data-testid="stAppViewContainer"] h1, [data-testid="stAppViewContainer"] h2, 
    [data-testid="stAppViewContainer"] h3, [data-testid="stAppViewContainer"] h4, 
    [data-testid="stAppViewContainer"] p, [data-testid="stAppViewContainer"] span, 
    [data-testid="stAppViewContainer"] label { color: #f8fafc !important; }

    div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {
        background-color: #0f172a !important;
        border-color: #334155 !important;
        color: #ffffff !important;
    }
    div[data-baseweb="select"] *, div[data-baseweb="input"] * { color: #ffffff !important; }

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
# 2. DATA LOAD & PREPROCESSING (SAMA PERSIS NOTEBOOK)
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

COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'success': '#2ECC71',
    'warning': '#F1C40F',
    'danger': '#E74C3C',
    'info': '#3498DB',
    'dark': '#2C3E50',
    'light': '#ECF0F1'
}

@st.cache_data
def load_and_prep_data():
    if not os.path.exists(DATA_PATH):
        return None
    
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip() for c in df.columns]

    for col in LEVEL_COLS + ["Jumlah"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # =========================================================
    # PREPROCESSING (SAMA PERSIS NOTEBOOK)
    # =========================================================
    
    # 3.1 AGREGASI DATA
    df_agregat = df.groupby(['Periode', 'Wilayah', 'Kota/Kab', 'Kode Kota/Kab', 'Status Sekolah']).agg({
        'Tingkat - I': 'sum',
        'Tingkat - II': 'sum',
        'Tingkat - III': 'sum',
        'Tingkat - IV': 'sum',
        'Tingkat - V': 'sum',
        'Tingkat - VI': 'sum',
        'Jumlah': 'sum'
    }).reset_index()
    
    # 3.2 PEMBUATAN FITUR PROPORSIONAL
    for romawi in ROMAWI_LIST:
        tingkat_col = f'Tingkat - {romawi}'
        proporsi_col = f'Proporsi_{romawi}'
        df_agregat[proporsi_col] = np.where(
            df_agregat['Jumlah'] > 0,
            df_agregat[tingkat_col] / df_agregat['Jumlah'],
            0
        )
    
    # 3.3 ENCODING STATUS SEKOLAH
    le_status = LabelEncoder()
    df_agregat['Status_Encoded'] = le_status.fit_transform(df_agregat['Status Sekolah'])
    
    # 3.4 PEMBUATAN FITUR LAG
    df_agregat = df_agregat.sort_values(['Kode Kota/Kab', 'Status Sekolah', 'Periode'])
    group_cols = ['Kode Kota/Kab', 'Status Sekolah']
    
    df_agregat['Jumlah_Lag1'] = df_agregat.groupby(group_cols)['Jumlah'].shift(1)
    for romawi in ROMAWI_LIST:
        df_agregat[f'Proporsi_{romawi}_Lag1'] = df_agregat.groupby(group_cols)[f'Proporsi_{romawi}'].shift(1)
    
    df_agregat['Selisih_Tahun'] = df_agregat['Periode'] - df_agregat.groupby(group_cols)['Periode'].shift(1)
    df_rf = df_agregat[df_agregat['Selisih_Tahun'] == 1].copy()
    
    all_years = sorted(df_rf['Periode'].unique())
    tahun_test = all_years[-1] if len(all_years) > 0 else 2024
    tahun_train = all_years[:-1] if len(all_years) > 1 else []
    
    fitur_rf = [
        'Jumlah_Lag1', 'Proporsi_I_Lag1', 'Proporsi_II_Lag1',
        'Proporsi_III_Lag1', 'Proporsi_IV_Lag1', 'Proporsi_V_Lag1',
        'Proporsi_VI_Lag1', 'Status_Encoded', 'Periode'
    ]
    
    # 3.9 PERSIAPAN DATA K-MEANS
    df_kmeans_tahunan = df.groupby(
        ['Periode', 'Kode Kota/Kab', 'Kota/Kab']
    ).agg({
        'Tingkat - I': 'sum',
        'Tingkat - II': 'sum',
        'Tingkat - III': 'sum',
        'Tingkat - IV': 'sum',
        'Tingkat - V': 'sum',
        'Tingkat - VI': 'sum',
        'Jumlah': 'sum'
    }).reset_index()
    
    for romawi in ROMAWI_LIST:
        tingkat_col = f'Tingkat - {romawi}'
        proporsi_col = f'Proporsi_{romawi}'
        df_kmeans_tahunan[proporsi_col] = np.where(
            df_kmeans_tahunan['Jumlah'] > 0,
            df_kmeans_tahunan[tingkat_col] / df_kmeans_tahunan['Jumlah'],
            0
        )
    
    kmeans_data = df_kmeans_tahunan.groupby(
        'Kode Kota/Kab'
    ).agg({
        'Proporsi_I': 'mean',
        'Proporsi_II': 'mean',
        'Proporsi_III': 'mean',
        'Proporsi_IV': 'mean',
        'Proporsi_V': 'mean',
        'Proporsi_VI': 'mean',
        'Jumlah': 'mean',
        'Kota/Kab': 'first'
    }).reset_index()
    
    fitur_kmeans = [
        'Proporsi_I', 'Proporsi_II', 'Proporsi_III',
        'Proporsi_IV', 'Proporsi_V', 'Proporsi_VI', 'Jumlah'
    ]
    
    # =========================================================
    # K-MEANS CLUSTERING (SAMA PERSIS NOTEBOOK)
    # =========================================================
    
    scaler_kmeans = StandardScaler()
    X_kmeans_scaled = scaler_kmeans.fit_transform(kmeans_data[fitur_kmeans])
    
    K_range = range(2, 11)
    inertia_scores = []
    silhouette_scores = []
    calinski_scores = []
    davies_scores = []
    
    for k in K_range:
        kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans_temp.fit_predict(X_kmeans_scaled)
        inertia_scores.append(kmeans_temp.inertia_)
        silhouette_scores.append(silhouette_score(X_kmeans_scaled, labels))
        calinski_scores.append(calinski_harabasz_score(X_kmeans_scaled, labels))
        davies_scores.append(davies_bouldin_score(X_kmeans_scaled, labels))
    
    best_k_sil = K_range[np.argmax(silhouette_scores)]
    best_k_cal = K_range[np.argmax(calinski_scores)]
    best_k_dav = K_range[np.argmin(davies_scores)]
    
    k_votes = [best_k_sil, best_k_cal, best_k_dav]
    vote_counts = Counter(k_votes)
    most_common_k = vote_counts.most_common(1)
    
    if len(most_common_k) > 0 and most_common_k[0][1] >= 2:
        optimal_k = most_common_k[0][0]
    else:
        optimal_k = best_k_sil
    
    kmeans_model = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    kmeans_data['Cluster'] = kmeans_model.fit_predict(X_kmeans_scaled)
    
    cluster_analysis = kmeans_data.groupby('Cluster').agg({
        'Proporsi_I': 'mean',
        'Proporsi_II': 'mean',
        'Proporsi_III': 'mean',
        'Proporsi_IV': 'mean',
        'Proporsi_V': 'mean',
        'Proporsi_VI': 'mean',
        'Jumlah': 'mean'
    })
    
    cluster_order = cluster_analysis.sort_values('Jumlah').index.tolist()
    cluster_names = {}
    for i, cluster_id in enumerate(cluster_order):
        if i == 0:
            cluster_names[cluster_id] = f"Cluster {cluster_id} (Rendah)"
        elif i == 1:
            cluster_names[cluster_id] = f"Cluster {cluster_id} (Sedang)"
        else:
            cluster_names[cluster_id] = f"Cluster {cluster_id} (Tinggi)"
    
    kmeans_data['Cluster_Label'] = kmeans_data['Cluster'].map(cluster_names)
    
    sil_score = silhouette_score(X_kmeans_scaled, kmeans_data['Cluster'])
    ch_score = calinski_harabasz_score(X_kmeans_scaled, kmeans_data['Cluster'])
    db_score = davies_bouldin_score(X_kmeans_scaled, kmeans_data['Cluster'])
    
    # =========================================================
    # RANDOM FOREST (SAMA PERSIS NOTEBOOK)
    # =========================================================
    
    def create_labels_for_fold(data, q1, q3):
        def categorize(value):
            if value <= q1:
                return 0
            elif value <= q3:
                return 1
            else:
                return 2
        return data['Jumlah'].apply(categorize)
    
    tahun_validasi = tahun_train[1:] if len(tahun_train) > 1 else []
    
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [5, 10, 15],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    best_params = {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 2, 'min_samples_leaf': 1}
    fold_accuracies = []
    mean_acc = 0
    std_acc = 0
    best_fold_details = []
    best_avg_score = -1
    
    if len(tahun_validasi) > 0:
        param_combinations = list(itertools.product(
            param_grid['n_estimators'],
            param_grid['max_depth'],
            param_grid['min_samples_split'],
            param_grid['min_samples_leaf']
        ))
        
        for params in param_combinations:
            n_estimators, max_depth, min_samples_split, min_samples_leaf = params
            fold_scores = []
            fold_details = []
            
            for i, tahun_val in enumerate(tahun_validasi):
                train_years = tahun_train[:i+1]
                train_fold = df_rf[df_rf['Periode'].isin(train_years)].copy()
                val_fold = df_rf[df_rf['Periode'] == tahun_val].copy()
                
                Q1_fold = train_fold['Jumlah'].quantile(0.25)
                Q3_fold = train_fold['Jumlah'].quantile(0.75)
                
                y_train_fold = create_labels_for_fold(train_fold, Q1_fold, Q3_fold)
                y_val_fold = create_labels_for_fold(val_fold, Q1_fold, Q3_fold)
                
                X_train_fold = train_fold[fitur_rf]
                X_val_fold = val_fold[fitur_rf]
                
                rf_temp = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    random_state=42,
                    n_jobs=-1
                )
                rf_temp.fit(X_train_fold, y_train_fold)
                y_pred_val = rf_temp.predict(X_val_fold)
                score = accuracy_score(y_val_fold, y_pred_val)
                fold_scores.append(score)
                fold_details.append({
                    'fold': i+1,
                    'train_years': f"{train_years[0]}-{train_years[-1]}",
                    'val_year': tahun_val,
                    'accuracy': score,
                    'q1': Q1_fold,
                    'q3': Q3_fold
                })
            
            avg_score = np.mean(fold_scores)
            if avg_score > best_avg_score:
                best_avg_score = avg_score
                best_params = {
                    'n_estimators': n_estimators,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf
                }
                best_fold_details = fold_details
                fold_accuracies = fold_scores
                mean_acc = avg_score
                std_acc = np.std(fold_scores)
    
    # 6.6 TRAINING FINAL
    train_final = df_rf[df_rf['Periode'] < tahun_test].copy()
    test_final = df_rf[df_rf['Periode'] == tahun_test].copy()
    
    Q1_final = train_final['Jumlah'].quantile(0.25)
    Q3_final = train_final['Jumlah'].quantile(0.75)
    
    y_train_final = create_labels_for_fold(train_final, Q1_final, Q3_final)
    y_test_final = create_labels_for_fold(test_final, Q1_final, Q3_final)
    
    X_train_final = train_final[fitur_rf]
    X_test_final = test_final[fitur_rf]
    
    best_rf = RandomForestClassifier(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        min_samples_split=best_params['min_samples_split'],
        min_samples_leaf=best_params['min_samples_leaf'],
        random_state=42,
        n_jobs=-1
    )
    best_rf.fit(X_train_final, y_train_final)
    
    baseline = DummyClassifier(strategy='most_frequent')
    baseline.fit(X_train_final, y_train_final)
    baseline_pred = baseline.predict(X_test_final)
    baseline_accuracy = accuracy_score(y_test_final, baseline_pred)
    
    y_pred_final = best_rf.predict(X_test_final)
    
    accuracy_final = accuracy_score(y_test_final, y_pred_final)
    precision_final = precision_score(y_test_final, y_pred_final, average='weighted', zero_division=0)
    recall_final = recall_score(y_test_final, y_pred_final, average='weighted', zero_division=0)
    f1_final = f1_score(y_test_final, y_pred_final, average='weighted', zero_division=0)
    
    macro_precision = precision_score(y_test_final, y_pred_final, average='macro', zero_division=0)
    macro_recall = recall_score(y_test_final, y_pred_final, average='macro', zero_division=0)
    macro_f1 = f1_score(y_test_final, y_pred_final, average='macro', zero_division=0)
    
    cm_final = confusion_matrix(y_test_final, y_pred_final, labels=[0, 1, 2])
    
    feature_importance = pd.DataFrame({
        'Fitur': fitur_rf,
        'Importance': best_rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    per_class_metrics = []
    class_names = ['Rendah', 'Sedang', 'Tinggi']
    for i, class_name in enumerate(class_names):
        tp = cm_final[i, i]
        fp = cm_final[:, i].sum() - tp
        fn = cm_final[i, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        support = cm_final[i, :].sum()
        per_class_metrics.append({
            'Kelas': class_name,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Support': support
        })
    per_class_df = pd.DataFrame(per_class_metrics)
    
    # =========================================================
    # EDA (SAMA PERSIS NOTEBOOK UNTUK ANGKA)
    # =========================================================
    tingkat_stats = df[LEVEL_COLS].agg(['mean', 'median', 'std', 'min', 'max', 'sum'])
    tingkat_stats.index = ['Rata-rata', 'Median', 'Std Dev', 'Min', 'Max', 'Total']
    
    total_per_provinsi = df.groupby('Wilayah')['Jumlah'].sum().sort_values(ascending=False)
    total_per_tahun = df.groupby('Periode')['Jumlah'].sum()
    
    status_analysis = df.groupby('Status Sekolah')['Jumlah'].agg(['sum', 'mean', 'median', 'std', 'count'])
    status_analysis.columns = ['Total', 'Rata-rata', 'Median', 'Std Dev', 'Jumlah Data']
    
    corr_matrix = df[LEVEL_COLS + ['Jumlah']].corr()
    
    return {
        'df': df,
        'df_rf': df_rf,
        'kmeans_data': kmeans_data,
        'scaler_kmeans': scaler_kmeans,
        'kmeans_model': kmeans_model,
        'rf_model': best_rf,
        'le_status': le_status,
        'fitur_rf': fitur_rf,
        'fitur_kmeans': fitur_kmeans,
        'tahun_test': tahun_test,
        'tahun_train': tahun_train,
        'tahun_validasi': tahun_validasi,
        'Q1_final': Q1_final,
        'Q3_final': Q3_final,
        'y_test_final': y_test_final,
        'y_pred_final': y_pred_final,
        'accuracy_final': accuracy_final,
        'precision_final': precision_final,
        'recall_final': recall_final,
        'f1_final': f1_final,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'baseline_accuracy': baseline_accuracy,
        'cm_final': cm_final,
        'feature_importance': feature_importance,
        'per_class_df': per_class_df,
        'sil_score': sil_score,
        'ch_score': ch_score,
        'db_score': db_score,
        'cluster_analysis': cluster_analysis,
        'cluster_names': cluster_names,
        'optimal_k': optimal_k,
        'K_range': K_range,
        'inertia_scores': inertia_scores,
        'silhouette_scores': silhouette_scores,
        'calinski_scores': calinski_scores,
        'davies_scores': davies_scores,
        'best_k_sil': best_k_sil,
        'best_k_cal': best_k_cal,
        'best_k_dav': best_k_dav,
        'fold_accuracies': fold_accuracies,
        'mean_acc': mean_acc,
        'std_acc': std_acc,
        'best_fold_details': best_fold_details,
        'tingkat_stats': tingkat_stats,
        'total_per_provinsi': total_per_provinsi,
        'total_per_tahun': total_per_tahun,
        'status_analysis': status_analysis,
        'corr_matrix': corr_matrix,
        'best_params': best_params,
    }

results = load_and_prep_data()

if results is None:
    st.error(f"Dataset tidak ditemukan di path: `{DATA_PATH}`")
    st.stop()

# Extract data
df = results['df']
df_rf = results['df_rf']
kmeans_data = results['kmeans_data']
scaler_kmeans = results['scaler_kmeans']
kmeans_model = results['kmeans_model']
rf_model = results['rf_model']
le_status = results['le_status']
fitur_rf = results['fitur_rf']
fitur_kmeans = results['fitur_kmeans']
tahun_test = results['tahun_test']
tahun_train = results['tahun_train']
tahun_validasi = results['tahun_validasi']
Q1_final = results['Q1_final']
Q3_final = results['Q3_final']
accuracy_final = results['accuracy_final']
precision_final = results['precision_final']
recall_final = results['recall_final']
f1_final = results['f1_final']
macro_precision = results['macro_precision']
macro_recall = results['macro_recall']
macro_f1 = results['macro_f1']
baseline_accuracy = results['baseline_accuracy']
cm_final = results['cm_final']
feature_importance = results['feature_importance']
per_class_df = results['per_class_df']
sil_score = results['sil_score']
ch_score = results['ch_score']
db_score = results['db_score']
cluster_analysis = results['cluster_analysis']
cluster_names = results['cluster_names']
optimal_k = results['optimal_k']
K_range = results['K_range']
inertia_scores = results['inertia_scores']
silhouette_scores = results['silhouette_scores']
calinski_scores = results['calinski_scores']
davies_scores = results['davies_scores']
best_k_sil = results['best_k_sil']
best_k_cal = results['best_k_cal']
best_k_dav = results['best_k_dav']
fold_accuracies = results['fold_accuracies']
mean_acc = results['mean_acc']
std_acc = results['std_acc']
best_fold_details = results['best_fold_details']
tingkat_stats = results['tingkat_stats']
total_per_provinsi = results['total_per_provinsi']
total_per_tahun = results['total_per_tahun']
status_analysis = results['status_analysis']
corr_matrix = results['corr_matrix']
best_params = results['best_params']

# =========================================================
# 3. SIDEBAR NAVIGATION
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

    menu = st.radio(
        label="Navigasi Utama",
        options=[
            "🏠 Overview Dashboard",
            "📊 Exploratory Analysis",
            "🧩 K-Means Clustering",
            "🌲 Random Forest Model",
            "🔮 Predictive Simulation"
        ],
        key="menu_nav",
        label_visibility="collapsed"
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
    total_records = len(df)
    total_prov = df["Wilayah"].nunique()
    total_cities = df["Kota/Kab"].nunique()
    periode_range = f"{df['Periode'].min()} - {df['Periode'].max()}"
    total_cases = int(df["Jumlah"].sum())
    
    with c1:
        st.markdown(f"<div class='metric-box'><div class='metric-label'>TOTAL REKORD DATA</div><div class='metric-num'>{total_records:,}</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='metric-box'><div class='metric-label'>PROVINSI</div><div class='metric-num'>{total_prov}</div></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='metric-box'><div class='metric-label'>KABUPATEN/KOTA</div><div class='metric-num'>{total_cities}</div></div>", unsafe_allow_html=True)
    with c4:
        st.markdown(f"<div class='metric-box'><div class='metric-label'>PERIODE DATA</div><div class='metric-num'>{periode_range}</div></div>", unsafe_allow_html=True)
    with c5:
        st.markdown(f"<div class='metric-box'><div class='metric-label'>TOTAL KASUS</div><div class='metric-num'>{total_cases:,}</div></div>", unsafe_allow_html=True)

    st.write("")
    
    # =========================================================
    # BAGIAN 2: DISTRIBUSI PER TAHUN (Fig 2.1)
    # =========================================================
    st.subheader("Distribusi Data per Tahun")
    
    tahun_dist = df['Periode'].value_counts().sort_index()
    
    col_tahun1, col_tahun2 = st.columns(2)
    
    # GRAFIK 1: Bar chart per Tahun
    with col_tahun1:
        fig_bar_tahun = px.bar(
            x=tahun_dist.index.astype(str), 
            y=tahun_dist.values,
            text=tahun_dist.values,
            color_discrete_sequence=[COLORS['primary']]
        )
        fig_bar_tahun.update_traces(textposition='outside', marker_color=COLORS['primary'])
        fig_bar_tahun.update_layout(
            title="Jumlah Data per Tahun",
            height=350,
            font=dict(color="#f8fafc"),
            xaxis=dict(gridcolor="#334155", title="Tahun", fixedrange=True),
            yaxis=dict(gridcolor="#334155", title="Jumlah Data", fixedrange=True),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False
        )
        st.plotly_chart(fig_bar_tahun, use_container_width=True, config=NO_ZOOM)
    
    # GRAFIK 2: Pie chart per Tahun
    with col_tahun2:
        fig_pie_tahun = px.pie(
            values=tahun_dist.values,
            names=tahun_dist.index.astype(str),
            hole=0.3,
            color_discrete_sequence=px.colors.sequential.Viridis_r
        )
        fig_pie_tahun.update_traces(textposition='inside', textinfo='percent+label')
        fig_pie_tahun.update_layout(
            title="Persentase Data per Tahun",
            height=350,
            font=dict(color="#f8fafc"),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False
        )
        st.plotly_chart(fig_pie_tahun, use_container_width=True, config=NO_ZOOM)

    st.markdown("---")
    
    # =========================================================
    # BAGIAN 2: DISTRIBUSI STATUS SEKOLAH (Fig 2.2)
    # =========================================================
    st.subheader("Distribusi Status Sekolah")
    
    status_dist = df['Status Sekolah'].value_counts()
    
    col_status1, col_status2 = st.columns(2)
    
    # GRAFIK 3: Bar chart Status Sekolah
    with col_status1:
        fig_bar_status = px.bar(
            x=status_dist.index,
            y=status_dist.values,
            text=status_dist.values,
            color=status_dist.index,
            color_discrete_sequence=[COLORS['primary'], COLORS['secondary']]
        )
        fig_bar_status.update_traces(textposition='outside')
        fig_bar_status.update_layout(
            title="Jumlah Data per Status Sekolah",
            height=350,
            font=dict(color="#f8fafc"),
            xaxis=dict(gridcolor="#334155", title="Status Sekolah", fixedrange=True),
            yaxis=dict(gridcolor="#334155", title="Jumlah Data", fixedrange=True),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False
        )
        st.plotly_chart(fig_bar_status, use_container_width=True, config=NO_ZOOM)
    
    # GRAFIK 4: Pie chart Status Sekolah
    with col_status2:
        fig_pie_status = px.pie(
            values=status_dist.values,
            names=status_dist.index,
            hole=0.3,
            color=status_dist.index,
            color_discrete_sequence=[COLORS['primary'], COLORS['secondary']]
        )
        fig_pie_status.update_traces(textposition='inside', textinfo='percent+label')
        fig_pie_status.update_layout(
            title="Persentase Status Sekolah",
            height=350,
            font=dict(color="#f8fafc"),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False
        )
        st.plotly_chart(fig_pie_status, use_container_width=True, config=NO_ZOOM)

# =========================================================
# HALAMAN 2: EXPLORATORY ANALYSIS (BAGIAN 4 NOTEBOOK)
# =========================================================
elif menu == "📊 Exploratory Analysis":
    st.title("Exploratory Data Analysis (EDA)")
    st.markdown("Eksplorasi rinci karakteristik data berdasarkan jenjang kelas, status sekolah, dan korelasi antar fitur.")
    
    c_f1, c_f2 = st.columns(2)
    with c_f1:
        years = ["Semua Tahun"] + sorted(df["Periode"].unique().tolist())
        sel_year = st.selectbox("Filter Tahun Periode:", years)
    with c_f2:
        regions = ["Semua Wilayah"] + sorted(df["Wilayah"].unique().tolist())
        sel_region = st.selectbox("Filter Wilayah / Provinsi:", regions)

    filtered_eda = df.copy()
    if sel_year != "Semua Tahun":
        filtered_eda = filtered_eda[filtered_eda["Periode"] == sel_year]
    if sel_region != "Semua Wilayah":
        filtered_eda = filtered_eda[filtered_eda["Wilayah"] == sel_region]

    st.write("")
    
    # =========================================================
    # 4.1 DISTRIBUSI PUTUS SEKOLAH PER TINGKAT (Fig 4.1)
    # =========================================================
    st.subheader("Distribusi Putus Sekolah per Tingkat")
    
    df_melted = filtered_eda[LEVEL_COLS].melt(var_name='Tingkat', value_name='Jumlah')
    df_melted['Tingkat'] = df_melted['Tingkat'].str.replace('Tingkat - ', 'Kelas ')
    
    col_e1, col_e2 = st.columns(2, gap="large")
    
    # GRAFIK 5: Boxplot per Tingkat
    with col_e1:
        fig_box = px.box(df_melted, x='Tingkat', y='Jumlah', color='Tingkat',
                        color_discrete_sequence=px.colors.qualitative.Set2)
        fig_box.update_layout(
            title="Boxplot Putus Sekolah per Tingkat",
            height=360,
            font=dict(color="#f8fafc"),
            xaxis=dict(gridcolor="#334155", fixedrange=True),
            yaxis=dict(gridcolor="#334155", title="Jumlah Siswa", fixedrange=True),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig_box, use_container_width=True, config=NO_ZOOM)
    
    # GRAFIK 6: Total per Tingkat (Bar chart)
    with col_e2:
        total_per_tingkat = filtered_eda[LEVEL_COLS].sum()
        total_per_tingkat.index = total_per_tingkat.index.str.replace('Tingkat - ', 'Kelas ')
        fig_bar_tingkat = px.bar(
            x=total_per_tingkat.index, 
            y=total_per_tingkat.values,
            text_auto=",d",
            color_discrete_sequence=px.colors.sequential.Viridis
        )
        fig_bar_tingkat.update_layout(
            title="Total Putus Sekolah per Tingkat",
            height=360,
            font=dict(color="#f8fafc"),
            xaxis=dict(gridcolor="#334155", title="Tingkat Kelas", fixedrange=True),
            yaxis=dict(gridcolor="#334155", title="Total Siswa", fixedrange=True),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig_bar_tingkat, use_container_width=True, config=NO_ZOOM)

    st.markdown("---")
    
    # Statistik Deskriptif (sama notebook)
    st.subheader("Statistik Deskriptif per Tingkat Kelas")
    st.dataframe(tingkat_stats.round(2).style.format("{:,.2f}"), use_container_width=True)
    
    st.markdown("---")
    
    col_violin, col_heat = st.columns(2, gap="large")
    
    # GRAFIK 7: Violin Plot
    with col_violin:
        st.subheader("Violin Plot Putus Sekolah per Tingkat")
        fig_violin = px.violin(df_melted, x='Tingkat', y='Jumlah', color='Tingkat',
                              box=True, points=False,
                              color_discrete_sequence=px.colors.qualitative.Set3)
        fig_violin.update_layout(
            height=360,
            font=dict(color="#f8fafc"),
            xaxis=dict(gridcolor="#334155", fixedrange=True),
            yaxis=dict(gridcolor="#334155", title="Jumlah Siswa", fixedrange=True),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig_violin, use_container_width=True, config=NO_ZOOM)
    
    # GRAFIK 8: Heatmap Korelasi Antar Tingkat - perbaikan
    with col_heat:
        st.subheader("Heatmap Korelasi Antar Tingkat")
        corr_tingkat = filtered_eda[LEVEL_COLS].corr().fillna(0)  # tambah fillna
        fig_corr_tingkat = px.imshow(
            corr_tingkat,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu"  # ganti coolwarm
        )
        fig_corr_tingkat.update_layout(
            height=360,
            font=dict(color="#f8fafc"),
            xaxis=dict(fixedrange=True),
            yaxis=dict(fixedrange=True),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_corr_tingkat, use_container_width=True, config=NO_ZOOM)

    st.markdown("---")
    
    # =========================================================
    # 4.2 ANALISIS PER PROVINSI (Fig 4.2)
    # =========================================================
    st.subheader("Analisis Per Provinsi")
    
    total_per_provinsi_filtered = filtered_eda.groupby('Wilayah')['Jumlah'].sum().sort_values(ascending=False)
    
    col_prov1, col_prov2 = st.columns(2, gap="large")
    
    # GRAFIK 9: Top 10 Provinsi Tertinggi
    with col_prov1:
        top10_prov = total_per_provinsi_filtered.head(10)
        fig_top10 = px.bar(
            x=top10_prov.values,
            y=top10_prov.index,
            orientation='h',
            text_auto=",d",
            color_discrete_sequence=px.colors.sequential.Reds_r
        )
        fig_top10.update_layout(
            title="10 Provinsi dengan Putus Sekolah Tertinggi",
            height=350,
            font=dict(color="#f8fafc"),
            xaxis=dict(gridcolor="#334155", title="Total Putus Sekolah", fixedrange=True),
            yaxis=dict(gridcolor="#334155", autorange="reversed", fixedrange=True),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False
        )
        st.plotly_chart(fig_top10, use_container_width=True, config=NO_ZOOM)
    
    # GRAFIK 10: Bottom 10 Provinsi Terendah
    with col_prov2:
        bottom10_prov = total_per_provinsi_filtered.tail(10)
        fig_bottom10 = px.bar(
            x=bottom10_prov.values,
            y=bottom10_prov.index,
            orientation='h',
            text_auto=",d",
            color_discrete_sequence=px.colors.sequential.Greens_r
        )
        fig_bottom10.update_layout(
            title="10 Provinsi dengan Putus Sekolah Terendah",
            height=350,
            font=dict(color="#f8fafc"),
            xaxis=dict(gridcolor="#334155", title="Total Putus Sekolah", fixedrange=True),
            yaxis=dict(gridcolor="#334155", autorange="reversed", fixedrange=True),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False
        )
        st.plotly_chart(fig_bottom10, use_container_width=True, config=NO_ZOOM)

    st.markdown("---")
    
    # =========================================================
    # 4.3 TREN PUTUS SEKOLAH PER TAHUN (Fig 4.3)
    # =========================================================
    st.subheader("Tren Putus Sekolah per Tahun")
    
    total_per_tahun_filtered = filtered_eda.groupby('Periode')['Jumlah'].sum()
    
    col_tren1, col_tren2 = st.columns(2, gap="large")
    
    # GRAFIK 11: Line Chart
    with col_tren1:
        fig_line = px.line(
            x=total_per_tahun_filtered.index,
            y=total_per_tahun_filtered.values,
            markers=True,
            color_discrete_sequence=[COLORS['primary']]
        )
        fig_line.update_traces(line_width=3, marker_size=10)
        fig_line.update_layout(
            title="Tren Putus Sekolah",
            height=350,
            font=dict(color="#f8fafc"),
            xaxis=dict(gridcolor="#334155", title="Tahun", dtick=1, fixedrange=True),
            yaxis=dict(gridcolor="#334155", title="Total Putus Sekolah", fixedrange=True),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False
        )
        # Tambahkan nilai di atas titik
        for idx, (tahun, val) in enumerate(total_per_tahun_filtered.items()):
            fig_line.add_annotation(
                x=tahun, y=val,
                text=f"{int(val):,}",
                showarrow=False,
                font=dict(size=10, color="#f8fafc"),
                yshift=10
            )
        st.plotly_chart(fig_line, use_container_width=True, config=NO_ZOOM)
    
    # GRAFIK 12: Bar Chart
    with col_tren2:
        fig_bar_tren = px.bar(
            x=total_per_tahun_filtered.index.astype(str),
            y=total_per_tahun_filtered.values,
            text_auto=",d",
            color_discrete_sequence=px.colors.sequential.Viridis
        )
        fig_bar_tren.update_layout(
            title="Total Putus Sekolah per Tahun",
            height=350,
            font=dict(color="#f8fafc"),
            xaxis=dict(gridcolor="#334155", title="Tahun", fixedrange=True),
            yaxis=dict(gridcolor="#334155", title="Total Putus Sekolah", fixedrange=True),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False
        )
        st.plotly_chart(fig_bar_tren, use_container_width=True, config=NO_ZOOM)

    st.markdown("---")
    
    # =========================================================
    # 4.4 STATUS SEKOLAH VS PUTUS SEKOLAH (Fig 4.4)
    # =========================================================
    st.subheader("Status Sekolah vs Putus Sekolah")
    
    status_analysis_filtered = filtered_eda.groupby('Status Sekolah')['Jumlah'].agg(['sum', 'mean', 'median', 'std', 'count'])
    status_analysis_filtered.columns = ['Total', 'Rata-rata', 'Median', 'Std Dev', 'Jumlah Data']
    
    col_stat1, col_stat2, col_stat3 = st.columns(3, gap="large")
    
    # GRAFIK 13: Total per Status
    with col_stat1:
        fig_total_status = px.bar(
            x=status_analysis_filtered.index,
            y=status_analysis_filtered['Total'],
            text_auto=",d",
            color=status_analysis_filtered.index,
            color_discrete_sequence=[COLORS['primary'], COLORS['secondary']]
        )
        fig_total_status.update_layout(
            title="Total Putus Sekolah per Status",
            height=300,
            font=dict(color="#f8fafc"),
            xaxis=dict(gridcolor="#334155", fixedrange=True),
            yaxis=dict(gridcolor="#334155", title="Total", fixedrange=True),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False
        )
        st.plotly_chart(fig_total_status, use_container_width=True, config=NO_ZOOM)
    
    # GRAFIK 14: Rata-rata per Status
    with col_stat2:
        fig_mean_status = px.bar(
            x=status_analysis_filtered.index,
            y=status_analysis_filtered['Rata-rata'],
            text_auto=".2f",
            color=status_analysis_filtered.index,
            color_discrete_sequence=[COLORS['info'], COLORS['secondary']]
        )
        fig_mean_status.update_layout(
            title="Rata-rata Putus Sekolah per Status",
            height=300,
            font=dict(color="#f8fafc"),
            xaxis=dict(gridcolor="#334155", fixedrange=True),
            yaxis=dict(gridcolor="#334155", title="Rata-rata", fixedrange=True),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False
        )
        st.plotly_chart(fig_mean_status, use_container_width=True, config=NO_ZOOM)
    
    # GRAFIK 15: Boxplot per Status
    with col_stat3:
        fig_box_status = px.box(
            filtered_eda,
            x='Status Sekolah',
            y='Jumlah',
            color='Status Sekolah',
            color_discrete_sequence=[COLORS['primary'], COLORS['secondary']]
        )
        fig_box_status.update_layout(
            title="Distribusi Putus Sekolah per Status",
            height=300,
            font=dict(color="#f8fafc"),
            xaxis=dict(gridcolor="#334155", fixedrange=True),
            yaxis=dict(gridcolor="#334155", title="Jumlah", fixedrange=True),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False
        )
        st.plotly_chart(fig_box_status, use_container_width=True, config=NO_ZOOM)

    st.markdown("---")
    
    # =========================================================
    # 4.5 HEATMAP KORELASI (Fig 4.5) - perbaikan
    # =========================================================
    st.subheader("Heatmap Korelasi Lengkap")
    
    # GRAFIK 16: Heatmap Korelasi Lengkap
    corr_matrix_filtered = filtered_eda[LEVEL_COLS + ['Jumlah']].corr().fillna(0)
    fig_corr_full = px.imshow(
        corr_matrix_filtered,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu"  # ganti coolwarm
    )
    fig_corr_full.update_layout(
        height=400,
        font=dict(color="#f8fafc"),
        xaxis=dict(fixedrange=True),
        yaxis=dict(fixedrange=True),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_corr_full, use_container_width=True, config=NO_ZOOM)

# =========================================================
# HALAMAN 3: K-MEANS CLUSTERING (BAGIAN 5 NOTEBOOK)
# =========================================================
elif menu == "🧩 K-Means Clustering":
    st.title("K-Means Clustering Analysis")
    st.markdown("Pengelompokan tingkat kerawanan per Kabupaten/Kota berdasarkan rata-rata proporsi tingkat dan total kasus.")
    
    # =========================================================
    # 5.2 METRIK EVALUASI
    # =========================================================
    c_m1, c_m2, c_m3, c_m4 = st.columns(4)
    with c_m1:
        st.markdown(f"<div class='metric-box'><div class='metric-label'>OPTIMAL K</div><div class='metric-num'>K = {optimal_k}</div><div class='metric-sub'>Rendah, Sedang, Tinggi</div></div>", unsafe_allow_html=True)
    with c_m2:
        st.markdown(f"<div class='metric-box'><div class='metric-label'>SILHOUETTE SCORE</div><div class='metric-num'>{sil_score:.4f}</div><div class='metric-sub'>Evaluasi K-Means</div></div>", unsafe_allow_html=True)
    with c_m3:
        st.markdown(f"<div class='metric-box'><div class='metric-label'>CALINSKI-HARABASZ</div><div class='metric-num'>{ch_score:,.2f}</div><div class='metric-sub'>Separasi Kluster</div></div>", unsafe_allow_html=True)
    with c_m4:
        st.markdown(f"<div class='metric-box'><div class='metric-label'>DAVIES-BOULDIN</div><div class='metric-num'>{db_score:.4f}</div><div class='metric-sub'>Similaritas Kluster</div></div>", unsafe_allow_html=True)

    st.write("")
    
    tab_c1, tab_c2 = st.tabs(["Metrik Evaluasi Cluster", "PCA 2D Visualizer"])
    
    with tab_c1:
        st.subheader("Metrik Evaluasi per k")
        
        # --- 4 Grafik terpisah dalam 2 baris ---
        col1, col2 = st.columns(2)
        
        with col1:
            # GRAFIK 17: Elbow Method (Inertia)
            fig_elbow = px.line(
                x=list(K_range), y=inertia_scores,
                markers=True,
                labels={"x": "Jumlah Cluster (k)", "y": "Inertia"},
                color_discrete_sequence=["#2E86AB"]
            )
            fig_elbow.update_traces(line_width=2, marker_size=8)
            fig_elbow.add_vline(
                x=optimal_k, line_dash="dash", line_color="#ef4444",
                annotation_text=f"k={optimal_k}", annotation_position="top"
            )
            fig_elbow.update_layout(
                title="Elbow Method (Inertia)",
                height=300,
                font=dict(color="#f8fafc"),
                xaxis=dict(gridcolor="#334155", dtick=1, fixedrange=True),
                yaxis=dict(gridcolor="#334155", fixedrange=True),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_elbow, use_container_width=True, config=NO_ZOOM)
            
            # GRAFIK 18: Silhouette Score
            fig_sil = px.line(
                x=list(K_range), y=silhouette_scores,
                markers=True,
                labels={"x": "Jumlah Cluster (k)", "y": "Silhouette Score"},
                color_discrete_sequence=["#E74C3C"]
            )
            fig_sil.update_traces(line_width=2, marker_size=8)
            fig_sil.add_vline(
                x=optimal_k, line_dash="dash", line_color="#ef4444",
                annotation_text=f"k={optimal_k}", annotation_position="top"
            )
            fig_sil.update_layout(
                title="Silhouette Score (higher is better)",
                height=300,
                font=dict(color="#f8fafc"),
                xaxis=dict(gridcolor="#334155", dtick=1, fixedrange=True),
                yaxis=dict(gridcolor="#334155", fixedrange=True),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_sil, use_container_width=True, config=NO_ZOOM)
        
        with col2:
            # GRAFIK 19: Calinski-Harabasz Score
            fig_cal = px.line(
                x=list(K_range), y=calinski_scores,
                markers=True,
                labels={"x": "Jumlah Cluster (k)", "y": "Calinski-Harabasz Score"},
                color_discrete_sequence=["#2ECC71"]
            )
            fig_cal.update_traces(line_width=2, marker_size=8)
            fig_cal.add_vline(
                x=optimal_k, line_dash="dash", line_color="#ef4444",
                annotation_text=f"k={optimal_k}", annotation_position="top"
            )
            fig_cal.update_layout(
                title="Calinski-Harabasz Score (higher is better)",
                height=300,
                font=dict(color="#f8fafc"),
                xaxis=dict(gridcolor="#334155", dtick=1, fixedrange=True),
                yaxis=dict(gridcolor="#334155", fixedrange=True),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_cal, use_container_width=True, config=NO_ZOOM)
            
            # GRAFIK 20: Davies-Bouldin Score
            fig_dav = px.line(
                x=list(K_range), y=davies_scores,
                markers=True,
                labels={"x": "Jumlah Cluster (k)", "y": "Davies-Bouldin Score"},
                color_discrete_sequence=["#A23B72"]
            )
            fig_dav.update_traces(line_width=2, marker_size=8)
            fig_dav.add_vline(
                x=optimal_k, line_dash="dash", line_color="#ef4444",
                annotation_text=f"k={optimal_k}", annotation_position="top"
            )
            fig_dav.update_layout(
                title="Davies-Bouldin Score (lower is better)",
                height=300,
                font=dict(color="#f8fafc"),
                xaxis=dict(gridcolor="#334155", dtick=1, fixedrange=True),
                yaxis=dict(gridcolor="#334155", fixedrange=True),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_dav, use_container_width=True, config=NO_ZOOM)
        
        # Tabel metrik
        st.markdown("---")
        metrics_df = pd.DataFrame({
            'k': list(K_range),
            'Inertia': inertia_scores,
            'Silhouette': silhouette_scores,
            'Calinski': calinski_scores,
            'Davies': davies_scores
        })
        st.dataframe(metrics_df.style.format({
            'Inertia': '{:.2f}',
            'Silhouette': '{:.4f}',
            'Calinski': '{:.2f}',
            'Davies': '{:.4f}'
        }), use_container_width=True)
        
        # Distribusi Kluster (GRAFIK 21 & 22) - pindahkan ke bawah atau di tab ini
        st.markdown("---")
        col_dist1, col_dist2 = st.columns(2)
        
        with col_dist1:
            st.subheader("Distribusi Anggota Cluster")
            cluster_counts = kmeans_data['Cluster_Label'].value_counts()
            fig_pie = px.pie(
                values=cluster_counts.values,
                names=cluster_counts.index,
                hole=0.3,
                color=cluster_counts.index,
                color_discrete_map={
                    "Cluster 0 (Rendah)": "#3498DB",
                    "Cluster 1 (Sedang)": "#2ECC71",
                    "Cluster 2 (Tinggi)": "#E74C3C"
                }
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            fig_pie.update_layout(
                height=350,
                font=dict(color="#f8fafc"),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=10, r=10, t=10, b=10),
                showlegend=False
            )
            st.plotly_chart(fig_pie, use_container_width=True, config=NO_ZOOM)
        
        with col_dist2:
            st.subheader("Jumlah Wilayah per Cluster")
            fig_bar_dist = px.bar(
                x=cluster_counts.index,
                y=cluster_counts.values,
                text_auto=True,
                color=cluster_counts.index,
                color_discrete_map={
                    "Cluster 0 (Rendah)": "#3498DB",
                    "Cluster 1 (Sedang)": "#2ECC71",
                    "Cluster 2 (Tinggi)": "#E74C3C"
                }
            )
            fig_bar_dist.update_layout(
                height=350,
                font=dict(color="#f8fafc"),
                xaxis=dict(gridcolor="#334155", title="Cluster", fixedrange=True),
                yaxis=dict(gridcolor="#334155", title="Jumlah Wilayah", fixedrange=True),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=10, r=10, t=10, b=10),
                showlegend=False
            )
            st.plotly_chart(fig_bar_dist, use_container_width=True, config=NO_ZOOM)
    
    with tab_c2:
        # =========================================================
        # 5.6 VISUALISASI CLUSTER DENGAN PCA 2D (Fig 5.6)
        # =========================================================
        st.subheader("Visualisasi Proyeksi 2D PCA")
        
        pca = PCA(n_components=2, random_state=42)
        X_scaled = scaler_kmeans.transform(kmeans_data[fitur_kmeans])
        pca_coords = pca.fit_transform(X_scaled)
        df_pca = kmeans_data.copy()
        df_pca["PC1"] = pca_coords[:, 0]
        df_pca["PC2"] = pca_coords[:, 1]
        
        # GRAFIK 23: PCA 2D Visualisasi
        fig_pca = px.scatter(
            df_pca,
            x="PC1", y="PC2",
            color="Cluster_Label",
            hover_data=["Kota/Kab", "Jumlah"],
            color_discrete_map={
                "Cluster 0 (Rendah)": "#3498DB",
                "Cluster 1 (Sedang)": "#2ECC71",
                "Cluster 2 (Tinggi)": "#E74C3C"
            }
        )
        fig_pca.update_traces(marker=dict(size=10, opacity=0.7))
        fig_pca.update_layout(
            height=400,
            font=dict(color="#f8fafc"),
            xaxis=dict(
                gridcolor="#334155",
                title=f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% varian)",
                fixedrange=True
            ),
            yaxis=dict(
                gridcolor="#334155",
                title=f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% varian)",
                fixedrange=True
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            )
        )
        # Tambahkan centroid
        centroids_pca = pca.transform(kmeans_model.cluster_centers_)
        fig_pca.add_trace(go.Scatter(
            x=centroids_pca[:, 0],
            y=centroids_pca[:, 1],
            mode='markers',
            marker=dict(
                symbol='x',
                size=15,
                color='red',
                line=dict(width=2, color='black')
            ),
            name='Centroid'
        ))
        st.plotly_chart(fig_pca, use_container_width=True, config=NO_ZOOM)
        
        # GRAFIK 24: PCA 2D dengan Label Kota
        st.subheader("PCA 2D dengan Label Kota (Top 5 per Cluster)")
        
        top_cities = []
        for cluster_id in kmeans_data['Cluster'].unique():
            cluster_cities = kmeans_data[kmeans_data['Cluster'] == cluster_id].nlargest(5, 'Jumlah')
            top_cities.append(cluster_cities)
        top_cities_df = pd.concat(top_cities)
        
        fig_pca_label = px.scatter(
            df_pca,
            x="PC1", y="PC2",
            color="Cluster_Label",
            hover_data=["Kota/Kab", "Jumlah"],
            color_discrete_map={
                "Cluster 0 (Rendah)": "#3498DB",
                "Cluster 1 (Sedang)": "#2ECC71",
                "Cluster 2 (Tinggi)": "#E74C3C"
            }
        )
        fig_pca_label.update_traces(marker=dict(size=8, opacity=0.5))
        
        for idx, row in top_cities_df.iterrows():
            kota = row['Kota/Kab']
            pca_row = df_pca[df_pca['Kota/Kab'] == kota]
            if len(pca_row) > 0:
                fig_pca_label.add_annotation(
                    x=pca_row['PC1'].values[0],
                    y=pca_row['PC2'].values[0],
                    text=kota,
                    font=dict(size=8, color="#f8fafc"),
                    showarrow=False,
                    yshift=5
                )
        fig_pca_label.update_layout(
            height=400,
            font=dict(color="#f8fafc"),
            xaxis=dict(
                gridcolor="#334155",
                title=f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% varian)",
                fixedrange=True
            ),
            yaxis=dict(
                gridcolor="#334155",
                title=f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% varian)",
                fixedrange=True
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            )
        )
        st.plotly_chart(fig_pca_label, use_container_width=True, config=NO_ZOOM)

    st.markdown("---")
    
    # =========================================================
    # 5.7 TABEL RINGKASAN K-MEANS
    # =========================================================
    st.subheader("Tabel Ringkasan Cluster")
    
    cluster_summary = kmeans_data.groupby('Cluster_Label').agg({
        'Kode Kota/Kab': 'count',
        'Jumlah': 'mean'
    }).rename(columns={'Kode Kota/Kab': 'Jumlah_Wilayah'})
    
    proporsi_dominan_map = {}
    for cluster_id in cluster_analysis.index:
        label = cluster_names[cluster_id]
        row = cluster_analysis.loc[cluster_id]
        prop_cols = ['Proporsi_I', 'Proporsi_II', 'Proporsi_III', 'Proporsi_IV', 'Proporsi_V', 'Proporsi_VI']
        max_prop_idx = np.argmax(row[prop_cols].values)
        max_tingkat = ['I', 'II', 'III', 'IV', 'V', 'VI'][max_prop_idx]
        proporsi_dominan_map[label] = f"Tingkat {max_tingkat} ({row[prop_cols[max_prop_idx]]:.3f})"
    
    cluster_summary['Proporsi_Dominan'] = cluster_summary.index.map(proporsi_dominan_map)
    cluster_summary['Persentase'] = (cluster_summary['Jumlah_Wilayah'] / len(kmeans_data) * 100).round(1)
    st.dataframe(cluster_summary, use_container_width=True)
    
    st.subheader("Karakteristik Rata-rata per Cluster")
    st.dataframe(cluster_analysis.reindex(cluster_names.values()).style.format("{:.3f}"), use_container_width=True)

# =========================================================
# HALAMAN 4: RANDOM FOREST MODEL (BAGIAN 6 NOTEBOOK)
# =========================================================
elif menu == "🌲 Random Forest Model":
    st.title("Random Forest Classification")
    st.markdown(f"Evaluasi performa model Supervised Learning (Evaluasi Pada Data Uji Tahun {tahun_test}).")
    
    # =========================================================
    # 6.9 METRIK EVALUASI
    # =========================================================
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"<div class='metric-box'><div class='metric-label'>ACCURACY</div><div class='metric-num'>{accuracy_final*100:.2f}%</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='metric-box'><div class='metric-label'>PRECISION</div><div class='metric-num'>{precision_final*100:.2f}%</div></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='metric-box'><div class='metric-label'>RECALL</div><div class='metric-num'>{recall_final*100:.2f}%</div></div>", unsafe_allow_html=True)
    with c4:
        st.markdown(f"<div class='metric-box'><div class='metric-label'>F1-SCORE</div><div class='metric-num'>{f1_final*100:.2f}%</div></div>", unsafe_allow_html=True)
    with c5:
        imp = (accuracy_final - baseline_accuracy) * 100
        st.markdown(f"<div class='metric-box'><div class='metric-label'>AKURASI VS BASELINE</div><div class='metric-num'>+{imp:.2f}%</div></div>", unsafe_allow_html=True)

    st.write("")
    
    # =========================================================
    # 6.10 CONFUSION MATRIX (Fig 6.10)
    # =========================================================
    col_rf1, col_rf2 = st.columns([1, 1.2], gap="large")
    
    with col_rf1:
        st.subheader(f"Confusion Matrix (Data Uji {tahun_test})")
        labels = ["Rendah", "Sedang", "Tinggi"]
        
        # GRAFIK 25: Confusion Matrix
        fig_cm = px.imshow(
            cm_final,
            x=labels, y=labels,
            text_auto=True,
            color_continuous_scale="Blues",
            labels=dict(x="Prediksi", y="Aktual")
        )
        fig_cm.update_layout(
            height=360,
            font=dict(color="#f8fafc"),
            xaxis=dict(fixedrange=True),
            yaxis=dict(fixedrange=True),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_cm, use_container_width=True, config=NO_ZOOM)
        
        # GRAFIK 26: Normalized Confusion Matrix
        st.subheader("Normalized Confusion Matrix (%)")
        cm_norm = cm_final.astype('float') / cm_final.sum(axis=1)[:, np.newaxis]
        fig_cm_norm = px.imshow(
            cm_norm,
            x=labels, y=labels,
            text_auto=".2%",
            color_continuous_scale="Blues",
            labels=dict(x="Prediksi", y="Aktual")
        )
        fig_cm_norm.update_layout(
            height=360,
            font=dict(color="#f8fafc"),
            xaxis=dict(fixedrange=True),
            yaxis=dict(fixedrange=True),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_cm_norm, use_container_width=True, config=NO_ZOOM)
    
    with col_rf2:
        # =========================================================
        # 6.11 FEATURE IMPORTANCE (Fig 6.11)
        # =========================================================
        st.subheader("Feature Importance Random Forest")
        fi_df = feature_importance.sort_values(by="Importance", ascending=True)
        
        # GRAFIK 27: Feature Importance
        fig_fi = px.bar(
            fi_df,
            x="Importance", y="Fitur",
            orientation="h",
            text_auto=".4f",
            color="Importance",
            color_continuous_scale="viridis"
        )
        fig_fi.update_layout(
            height=360,
            font=dict(color="#f8fafc"),
            xaxis=dict(gridcolor="#334155", fixedrange=True),
            yaxis=dict(
                autorange="reversed",
                gridcolor="#334155",
                title="Fitur",
                fixedrange=True
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_fi, use_container_width=True, config=NO_ZOOM)
        
        # GRAFIK 28: Top 5 Feature Importance
        st.subheader("Top 5 Feature Importance")
        top5_df = feature_importance.head(5).sort_values(by="Importance", ascending=False)
        fig_top5 = px.bar(
            top5_df,
            x="Fitur", y="Importance",
            text_auto=".4f",
            color="Importance",
            color_continuous_scale="plasma"
        )
        fig_top5.update_layout(
            height=200,
            font=dict(color="#f8fafc"),
            xaxis=dict(gridcolor="#334155", fixedrange=True),
            yaxis=dict(gridcolor="#334155", fixedrange=True),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_top5, use_container_width=True, config=NO_ZOOM)

    st.markdown("---")
    
    # =========================================================
    # 6.17 TABEL METRIK PER KELAS
    # =========================================================
    col_rf3, col_rf4 = st.columns(2, gap="large")
    
    with col_rf3:
        st.subheader("Classification Report")
        report_data = []
        for i, class_name in enumerate(["Rendah", "Sedang", "Tinggi"]):
            row = per_class_df[per_class_df['Kelas'] == class_name].iloc[0]
            report_data.append({
                'Kelas': class_name,
                'Precision': f"{row['Precision']*100:.2f}%",
                'Recall': f"{row['Recall']*100:.2f}%",
                'F1-Score': f"{row['F1-Score']*100:.2f}%",
                'Support': int(row['Support'])
            })
        report_data.append({
            'Kelas': 'Macro Avg',
            'Precision': f"{macro_precision*100:.2f}%",
            'Recall': f"{macro_recall*100:.2f}%",
            'F1-Score': f"{macro_f1*100:.2f}%",
            'Support': '-'
        })
        st.dataframe(pd.DataFrame(report_data), use_container_width=True)
    
    with col_rf4:
        st.subheader("Perbandingan Weighted vs Macro")
        comparison_data = {
            'Metrik': ['Precision', 'Recall', 'F1-Score'],
            'Weighted': [f"{precision_final*100:.2f}%", f"{recall_final*100:.2f}%", f"{f1_final*100:.2f}%"],
            'Macro': [f"{macro_precision*100:.2f}%", f"{macro_recall*100:.2f}%", f"{macro_f1*100:.2f}%"]
        }
        st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)

    st.markdown("---")
    
    # =========================================================
    # 6.15 PERBANDINGAN RF VS BASELINE
    # =========================================================
    st.subheader("Perbandingan Random Forest vs Baseline")
    comparison_df = pd.DataFrame({
        'Model': ['Baseline (Most Frequent)', 'Random Forest'],
        'Accuracy': [f"{baseline_accuracy*100:.2f}%", f"{accuracy_final*100:.2f}%"],
        'Precision (Weighted)': ['-', f"{precision_final*100:.2f}%"],
        'Recall (Weighted)': ['-', f"{recall_final*100:.2f}%"],
        'F1-Score (Weighted)': ['-', f"{f1_final*100:.2f}%"],
        'Peningkatan': ['-', f"{(accuracy_final - baseline_accuracy)*100:.2f}%"]
    })
    st.dataframe(comparison_df, use_container_width=True)
    
    st.markdown("---")
    
    # =========================================================
    # 6.18 GRAFIK ACCURACY PER FOLD (Fig 6.18)
    # =========================================================
    if len(fold_accuracies) > 0:
        st.subheader("Akurasi per Fold Validasi")
        
        fold_numbers = [d['fold'] for d in best_fold_details]
        fold_acc_values = [d['accuracy'] for d in best_fold_details]
        fold_labels = [f"{d['train_years']} → {d['val_year']}" for d in best_fold_details]
        
        col_acc1, col_acc2 = st.columns(2, gap="large")
        
        # GRAFIK 29: Bar Chart Accuracy per Fold
        with col_acc1:
            fig_bar_acc = px.bar(
                x=fold_labels,
                y=fold_acc_values,
                text=fold_acc_values,
                color_discrete_sequence=px.colors.sequential.Viridis
            )
            fig_bar_acc.update_traces(textposition='outside', texttemplate='%{text:.3f}')
            fig_bar_acc.update_layout(
                title="Akurasi per Fold Validasi",
                height=350,
                font=dict(color="#f8fafc"),
                xaxis=dict(
                    gridcolor="#334155",
                    title="Fold (Train → Validate)",
                    fixedrange=True,
                    tickangle=30
                ),
                yaxis=dict(gridcolor="#334155", title="Accuracy", fixedrange=True, range=[0, 1]),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                showlegend=False
            )
            fig_bar_acc.add_hline(
                y=mean_acc,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Rata-rata: {mean_acc:.3f}",
                annotation_position="bottom right"
            )
            st.plotly_chart(fig_bar_acc, use_container_width=True, config=NO_ZOOM)
        
        # GRAFIK 30: Line Chart Accuracy per Fold
        with col_acc2:
            fig_line_acc = px.line(
                x=fold_numbers,
                y=fold_acc_values,
                markers=True,
                color_discrete_sequence=[COLORS['primary']]
            )
            fig_line_acc.update_traces(line_width=2, marker_size=8)
            fig_line_acc.update_layout(
                title="Tren Akurasi per Fold",
                height=350,
                font=dict(color="#f8fafc"),
                xaxis=dict(gridcolor="#334155", title="Fold", dtick=1, fixedrange=True),
                yaxis=dict(gridcolor="#334155", title="Accuracy", fixedrange=True, range=[0, 1]),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                showlegend=False
            )
            fig_line_acc.add_hline(
                y=mean_acc,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Rata-rata: {mean_acc:.3f}",
                annotation_position="bottom right"
            )
            for fold, acc in zip(fold_numbers, fold_acc_values):
                fig_line_acc.add_annotation(
                    x=fold, y=acc,
                    text=f"{acc:.3f}",
                    showarrow=False,
                    font=dict(size=9, color="#f8fafc"),
                    yshift=10
                )
            fig_line_acc.add_hrect(
                y0=max(0, mean_acc - std_acc),
                y1=min(1, mean_acc + std_acc),
                fillcolor="rgba(46, 134, 171, 0.2)",
                line_width=0,
                annotation_text=f"±1 Std: {std_acc:.3f}",
                annotation_position="top left"
            )
            st.plotly_chart(fig_line_acc, use_container_width=True, config=NO_ZOOM)

# =========================================================
# HALAMAN 5: PREDICTIVE SIMULATION
# =========================================================
elif menu == "🔮 Predictive Simulation":
    st.title("Predictive Simulation")
    st.markdown("Simulasi prediksi tingkat kerawanan wilayah berdasarkan **Data Tahun Sebelumnya (Lag-1)** & Karakteristik Status.")
    
    st.write("")
    
    with st.form("pred_form"):
        st.subheader("Input Parameter Tahun Sebelumnya (t-1)")
        
        c_i1, c_i2 = st.columns(2)
        with c_i1:
            val_jumlah_lag1 = st.number_input("Total Putus Sekolah Tahun Sebelumnya (Jumlah_Lag1)", min_value=0, value=500, step=10)
            status_input = st.selectbox("Status Sekolah", options=list(le_status.classes_))
            periode_target = st.number_input("Tahun Yang Diprediksi (Periode)", min_value=2024, max_value=2030, value=2025)
        
        with c_i2:
            st.markdown("**Jumlah Siswa Per Kelas Tahun Sebelumnya (t-1)**")
            v_k1 = st.number_input("Jumlah Kelas I (Lag1)", min_value=0, value=80, step=5)
            v_k2 = st.number_input("Jumlah Kelas II (Lag1)", min_value=0, value=70, step=5)
            v_k3 = st.number_input("Jumlah Kelas III (Lag1)", min_value=0, value=90, step=5)
            v_k4 = st.number_input("Jumlah Kelas IV (Lag1)", min_value=0, value=100, step=5)
            v_k5 = st.number_input("Jumlah Kelas V (Lag1)", min_value=0, value=110, step=5)
            v_k6 = st.number_input("Jumlah Kelas VI (Lag1)", min_value=0, value=50, step=5)
            
        st.write("")
        submit_btn = st.form_submit_button("PREDIKSI KATEGORI WILAYAH", use_container_width=True)

    if submit_btn:
        tot_k = v_k1 + v_k2 + v_k3 + v_k4 + v_k5 + v_k6
        denom = tot_k if tot_k > 0 else 1
        
        p_i = v_k1 / denom
        p_ii = v_k2 / denom
        p_iii = v_k3 / denom
        p_iv = v_k4 / denom
        p_v = v_k5 / denom
        p_vi = v_k6 / denom
        
        status_encoded = le_status.transform([status_input])[0]
        
        fitur_rf = [
            'Jumlah_Lag1',
            'Proporsi_I_Lag1',
            'Proporsi_II_Lag1',
            'Proporsi_III_Lag1',
            'Proporsi_IV_Lag1',
            'Proporsi_V_Lag1',
            'Proporsi_VI_Lag1',
            'Status_Encoded',
            'Periode'
        ]
                    
        input_data = pd.DataFrame([[
            val_jumlah_lag1, p_i, p_ii, p_iii, p_iv, p_v, p_vi, status_encoded, periode_target
        ]], columns=fitur_rf)
        
        label_map = {0: "Rendah", 1: "Sedang", 2: "Tinggi"}
        pred_label_num = rf_model.predict(input_data)[0]
        pred_label = label_map[pred_label_num]
        pred_proba = rf_model.predict_proba(input_data)[0]
        max_proba = max(pred_proba) * 100
        
        st.markdown("---")
        st.subheader("Hasil Klasifikasi Prediksi")
        
        card_class = f"pred-card-{pred_label.lower()}"
        
        st.markdown(
            f"""
            <div class='{card_class}'>
                <h3 style='margin:0; font-size:1.1rem; opacity:0.9;'>KATEGORI TINGKAT KERAWANAN</h3>
                <h1 style='font-size: 3.5rem; font-weight: 800; margin: 0.5rem 0;'>{pred_label.upper()}</h1>
                <p style='margin:0; font-size:1rem; font-weight:600;'>Tingkat Keyakinan Model (Confidence): {max_proba:.1f}%</p>
                <p style='margin:0.5rem 0 0 0; font-size:0.85rem; opacity:0.7;'>
                    Threshold: Q1={Q1_final:.2f}, Q3={Q3_final:.2f}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        st.subheader("Detail Probabilitas per Kategori")
        proba_df = pd.DataFrame({
            'Kategori': ['Rendah', 'Sedang', 'Tinggi'],
            'Probabilitas': pred_proba * 100
        })
        fig_proba = px.bar(
            proba_df,
            x='Kategori', y='Probabilitas',
            text_auto=".1f",
            color='Kategori',
            color_discrete_map={'Rendah': '#22c55e', 'Sedang': '#eab308', 'Tinggi': '#ef4444'}
        )
        fig_proba.update_layout(
            height=300,
            font=dict(color="#f8fafc"),
            xaxis=dict(gridcolor="#334155", fixedrange=True),
            yaxis=dict(gridcolor="#334155", title="Probabilitas (%)", fixedrange=True, range=[0, 100]),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False
        )
        st.plotly_chart(fig_proba, use_container_width=True, config=NO_ZOOM)
