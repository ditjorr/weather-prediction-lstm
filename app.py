# =========================================================
# WEATHER PREDICTION USING LSTM
# Prediksi Hujan 1 Jam ke Depan
# =========================================================

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model

# =========================================================
# KONFIGURASI HALAMAN
# =========================================================
st.set_page_config(
    page_title="Weather Prediction LSTM",
    page_icon="🌧️",
    layout="wide"
)

# =========================================================
# PATH PROJECT
# =========================================================
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model" / "best_model.keras"
SCALER_PATH = BASE_DIR / "model" / "scaler_X.joblib"
BEST_THR_PATH = BASE_DIR / "model" / "best_threshold.txt"

# =========================================================
# PARAMETER MODEL
# =========================================================
TIME_STEPS = 12
HORIZON_MINUTES = 60
THRESHOLD = 0.5

FEATURES = [
    "Curah_Hujan_corrected",
    "Curah_Hujan_original",
    "Suhu",
    "Kelembaban",
    "Kecepatan_Angin",
]

TARGET_COL = "Curah_Hujan_corrected"

# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def load_artifacts():
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    with open(BEST_THR_PATH, "r") as f:
        threshold = float(f.read().strip())

    return model, scaler, threshold


model, scaler, THRESHOLD = load_artifacts()

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:

    st.title("🌧️ Weather Prediction")

    st.markdown(
        """
Aplikasi ini menggunakan model
**Long Short-Term Memory (LSTM)**

untuk memprediksi kondisi hujan
**1 jam ke depan**
berdasarkan data historis sensor cuaca.
"""
    )

    st.divider()

    st.subheader("Konfigurasi")

    st.write(f"**Time Steps :** {TIME_STEPS}")

    st.write(f"**Horizon :** {HORIZON_MINUTES} Menit")

    st.write(f"**Threshold :** {THRESHOLD:.3f}")

# =========================================================
# HEADER
# =========================================================
st.title("🌧️ Weather Prediction Using LSTM")

st.caption(
    "Prediksi kondisi hujan 1 jam ke depan menggunakan model Long Short-Term Memory (LSTM)."
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Time Steps", TIME_STEPS)

with col2:
    st.metric("Horizon", "1 Jam")

with col3:
    st.metric("Threshold", f"{THRESHOLD:.3f}")

st.divider()

# =========================================================
# UPLOAD DATA
# =========================================================
uploaded_file = st.file_uploader(
    "Upload file CSV hasil pra-proses",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Silakan upload file CSV terlebih dahulu.")
    st.stop()

# =========================================================
# BACA DATA
# =========================================================
df = pd.read_csv(uploaded_file)

df.columns = [c.strip() for c in df.columns]

# =========================================================
# MEMBANGUN DATETIME
# =========================================================
if "Datetime" not in df.columns:

    df["Datetime"] = pd.to_datetime(
        df["Tanggal"].astype(str)
        + " "
        + df["Jam"].astype(str),
        dayfirst=True,
        errors="coerce",
    )

df = df.sort_values("Datetime").reset_index(drop=True)

# =========================================================
# INFORMASI DATA
# =========================================================
st.subheader("Informasi Dataset")

info1, info2, info3 = st.columns(3)

with info1:
    st.metric("Jumlah Data", len(df))

with info2:
    st.metric("Jumlah Kolom", len(df.columns))

with info3:
    st.metric(
        "Time Steps",
        TIME_STEPS
    )

st.subheader("Preview Dataset")

st.dataframe(
    df.head(10),
    use_container_width=True
)

# =========================================================
# VALIDASI KOLOM
# =========================================================
missing = [
    c
    for c in FEATURES + [TARGET_COL]
    if c not in df.columns
]

if missing:
    st.error(f"Kolom berikut tidak ditemukan:\n\n{missing}")
    st.stop()

# =========================================================
# PREPROCESSING
# =========================================================
for col in FEATURES + [TARGET_COL]:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

df.interpolate(
    method="linear",
    inplace=True,
    limit_direction="both",
)

df.dropna(
    subset=FEATURES + [TARGET_COL],
    inplace=True,
)

df.reset_index(
    drop=True,
    inplace=True,
)

# =========================================================
# LABEL AKTUAL
# =========================================================
df["Kondisi_Aktual"] = df[TARGET_COL].apply(
    lambda x: "Hujan"
    if x > 0
    else "Tidak Hujan"
)

# =========================================================
# VALIDASI PANJANG DATA
# =========================================================
if len(df) < TIME_STEPS:

    st.error(
        f"Minimal diperlukan {TIME_STEPS} baris data."
    )

    st.stop()
# =========================================================
# PREDIKSI
# =========================================================
if st.button("🔍 Jalankan Prediksi", use_container_width=True):

    with st.spinner("Model LSTM sedang melakukan prediksi..."):

        rows = []

        total_data = len(df)
        total_sequence = total_data - TIME_STEPS + 1

        if total_sequence <= 0:
            st.error("Jumlah data tidak cukup untuk membentuk sequence.")
            st.stop()

        progress_bar = st.progress(0)

        for i in range(total_sequence):

            start_idx = i
            end_idx = i + TIME_STEPS

            seq_df = df.iloc[start_idx:end_idx]

            # ===============================
            # INPUT MODEL
            # ===============================
            X_seq = seq_df[FEATURES].values.reshape(
                1,
                TIME_STEPS,
                len(FEATURES)
            )

            X_scaled = scaler.transform(
                X_seq.reshape(-1, X_seq.shape[2])
            ).reshape(X_seq.shape)

            # ===============================
            # PREDIKSI
            # ===============================
            prob = float(model.predict(X_scaled, verbose=0)[0][0])

            pred_label = (
                "Hujan"
                if prob >= THRESHOLD
                else "Tidak Hujan"
            )

            # ===============================
            # WAKTU
            # ===============================
            waktu_awal = pd.to_datetime(
                seq_df["Datetime"].iloc[0]
            )

            waktu_akhir = pd.to_datetime(
                seq_df["Datetime"].iloc[-1]
            )

            waktu_pred = (
                waktu_akhir
                + pd.Timedelta(minutes=HORIZON_MINUTES)
            )

            # ===============================
            # GROUND TRUTH
            # ===============================
            actual_idx = end_idx + TIME_STEPS - 1

            if actual_idx < len(df):

                actual_val = df[TARGET_COL].iloc[actual_idx]

                kondisi_aktual = (
                    "Hujan"
                    if actual_val > 0
                    else "Tidak Hujan"
                )

            else:

                kondisi_aktual = "-"

            # ===============================
            # SIMPAN HASIL
            # ===============================
            rows.append(
                {
                    "Sequence": i + 1,
                    "Rentang Waktu Data":
                        f"{waktu_awal:%Y-%m-%d %H:%M} - "
                        f"{waktu_akhir:%Y-%m-%d %H:%M}",
                    "Waktu Prediksi":
                        waktu_pred.strftime("%Y-%m-%d %H:%M"),
                    "Kondisi Aktual":
                        kondisi_aktual,
                    "Prediksi 1 Jam":
                        pred_label,
                    "Probabilitas":
                        round(prob, 3),
                }
            )

            progress_bar.progress((i + 1) / total_sequence)

        progress_bar.empty()

        # ===============================
        # HASIL
        # ===============================
        hasil_df = pd.DataFrame(rows)

        st.success("✅ Prediksi berhasil dilakukan.")

        st.balloons()

        st.subheader(
            f"Hasil Prediksi ({len(hasil_df)} Sequence)"
        )

        st.dataframe(
            hasil_df,
            use_container_width=True
        )

        # ===============================
        # DOWNLOAD CSV
        # ===============================
        csv = hasil_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="📥 Download Hasil Prediksi",
            data=csv,
            file_name="hasil_prediksi.csv",
            mime="text/csv",
            use_container_width=True,
        )

        # ===============================
        # RINGKASAN
        # ===============================
        st.markdown(
            f"""
### Keterangan

- Prediksi dilakukan menggunakan data historis selama **1 jam terakhir** (**{TIME_STEPS} data**).
- Model yang digunakan adalah **Long Short-Term Memory (LSTM)**.
- Horizon prediksi adalah **1 jam ke depan**.
- Threshold klasifikasi yang digunakan adalah **{THRESHOLD:.3f}**.
- Waktu prediksi terakhir pada data ini adalah **{waktu_pred:%Y-%m-%d %H:%M}**.
"""
        )