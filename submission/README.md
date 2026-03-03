# Proyek Analisis Data: Bike Sharing Dataset

## Setup Environment

### 1) Buat dan aktifkan virtual environment (disarankan)

**Windows (PowerShell)**

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux**

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

## Run Streamlit App

```bash
python -m streamlit run dashboard/dashboard.py --server.port 8502
```

## Catatan Dataset

- Dashboard akan menggunakan file `data/day.csv` dan `data/hour.csv` jika sudah tersedia.
- Jika belum ada, dashboard akan mengunduh dataset dari UCI dan menerapkan proses cleaning (konversi tanggal, label musim/cuaca, dan fitur turunan seperti `temp_c`) sebelum divisualisasikan.
