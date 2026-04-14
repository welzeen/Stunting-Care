# StuntingApp - Sistem Deteksi Stunting Balita

Aplikasi web berbasis Django dengan Machine Learning Naive Bayes untuk mendeteksi status stunting balita.

## Fitur Lengkap

### 👤 Manajemen Pengguna
- Login / Logout / Register
- Dua role: **Admin** dan **Pengunjung**
- Profil pengguna (edit nama, email, telepon, alamat)
- Ubah password
- Admin dapat melihat & menghapus pengguna

### 📊 Data Balita (CRUD)
- Tambah, lihat, edit, hapus data balita
- Filter & pencarian (kode, JK, status stunting, dataset)
- Pagination 20 data per halaman
- Import data dari file Excel (.xlsx)
- Export data ke Excel

### 🤖 Machine Learning (Naive Bayes)
- **Latih Model** dari data training
- **Prediksi** status stunting per individu
- **Batch Prediksi** untuk seluruh data testing
- Evaluasi: Akurasi, Presisi, Recall, F1-Score
- **Confusion Matrix** dengan visualisasi chart
- **Classification Report** per kelas
- Riwayat semua prediksi yang pernah dilakukan

### 📈 Dashboard
- Statistik total data (training, testing, prediksi)
- Grafik distribusi status stunting (Doughnut chart)
- Grafik jenis kelamin (Bar chart)
- Progress status gizi
- Tabel data terbaru & riwayat prediksi terbaru

---

## Cara Menjalankan

### 1. Install dependensi
```bash
pip install -r requirements.txt
```

### 2. Inisialisasi database
```bash
python manage.py migrate
```

### 3. Setup aplikasi (buat user + import data Excel)
```bash
# Dengan data Excel (ganti path sesuai lokasi file Anda)
python manage.py setup_app --excel "path/ke/Olah_Data_fix_cm.xlsx"

# Tanpa data Excel (hanya buat user)
python manage.py setup_app --skip-data
```

### 4. Jalankan server
```bash
python manage.py runserver
```

### 5. Buka browser
```
http://127.0.0.1:8000
```

---

## Akun Default

| Username | Password | Role       |
|----------|----------|------------|
| admin    | admin123 | Admin      |
| user     | user123  | Pengunjung |

---

## Alur Penggunaan

1. **Login** sebagai admin
2. **Import data** dari Excel (Menu: Data Balita → Import Excel)
   - Pilih sheet "Preprocessing" sebagai Training
   - Pilih sheet "Data Test" sebagai Testing
3. **Latih Model** (Menu: Machine Learning → Latih Model)
4. **Lihat Evaluasi** (Menu: Machine Learning → Evaluasi Model)
5. **Prediksi** data baru (Menu: Machine Learning → Prediksi Stunting)

---

## Struktur Data

Fitur yang digunakan untuk prediksi:
| Fitur | Keterangan | Nilai |
|-------|-----------|-------|
| Jenis Kelamin | JK | L / P |
| Umur | Kelompok umur | 1-34 Bulan / 35-69 Bulan |
| BB/U | Berat Badan / Umur | Sangat Kurang, Kurang, Normal, Berat Badan Normal, Risiko Lebih |
| TB/U | Tinggi Badan / Umur | Sangat Pendek, Pendek, Normal, Tinggi |
| Status Gizi | Status gizi | Gizi Buruk, Gizi Kurang, Gizi Baik, Risiko Gizi Lebih, Gizi Lebih, Obesitas |
| ASI Eksklusif | Status ASI | Ya / Tidak |

**Label (target):** Status Stunting → `Tidak` / `Potensi Stunting` / `Stunting`

---

## Struktur Proyek

```
stunting_app/
├── manage.py
├── requirements.txt
├── stunting_project/          # Konfigurasi Django
│   ├── settings.py
│   └── urls.py
├── apps/
│   ├── accounts/              # Autentikasi & profil
│   ├── balita/                # CRUD data balita
│   └── ml_engine/             # Machine Learning
├── templates/                 # Template HTML
│   ├── base.html
│   ├── accounts/
│   ├── balita/
│   ├── ml/
│   └── dashboard/
└── static/                    # CSS, JS, gambar
```
