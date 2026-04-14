=====================================
  CARA MENJALANKAN STUNTINGAPP
=====================================

PERSYARATAN:
- Python 3.9+
- pip

─────────────────────────────────────
LANGKAH 1 — Install dependensi
─────────────────────────────────────
    pip install -r requirements.txt

─────────────────────────────────────
LANGKAH 2 — Setup (WAJIB sekali saja)
─────────────────────────────────────
    python manage.py setup_app

Perintah ini otomatis:
  ✓ Buat semua tabel database (migrate)
  ✓ Buat akun admin & pengunjung
  ✓ Import 746 data training
  ✓ Import 187 data testing

─────────────────────────────────────
LANGKAH 3 — Jalankan server
─────────────────────────────────────
    python manage.py runserver

─────────────────────────────────────
LANGKAH 4 — Buka browser
─────────────────────────────────────
    http://127.0.0.1:8000

    Login Admin : admin / admin123
    Login User  : user  / user123

─────────────────────────────────────
LANGKAH 5 — Latih Model ML
─────────────────────────────────────
1. Login sebagai admin
2. Menu: Machine Learning → Latih Model
3. Lihat evaluasi di: Evaluasi Model
4. Coba prediksi: Prediksi Stunting

=====================================
  DATA YANG SUDAH DISERTAKAN
=====================================
  data/Data_Training_Final.xlsx → 746 data training
  data/Data_Test_Final.xlsx     → 187 data testing

=====================================
