=====================================================
  PANDUAN MIGRASI KE MYSQL (XAMPP)
  StuntingApp — Deteksi Stunting Balita
=====================================================

LANGKAH 1 — Pastikan XAMPP berjalan
-------------------------------------
1. Buka XAMPP Control Panel
2. Klik START pada Apache
3. Klik START pada MySQL
4. Pastikan keduanya berwarna HIJAU

LANGKAH 2 — Buat database di phpMyAdmin
-----------------------------------------
1. Buka browser → http://localhost/phpmyadmin
2. Klik tab "SQL" (di bagian atas)
3. Copy-paste isi file "stunting_db.sql" ke kolom SQL
4. Klik tombol "Go"
5. Pastikan muncul pesan "Database stunting_db created"

   ATAU cara cepat:
   - Klik "New" di panel kiri
   - Isi nama: stunting_db
   - Pilih collation: utf8mb4_unicode_ci
   - Klik Create

LANGKAH 3 — Install library MySQL untuk Python
------------------------------------------------
Buka Command Prompt di folder stunting_app, lalu ketik:

    pip install mysqlclient

  Jika gagal di Windows, coba:
    pip install pymysql

  Jika pakai pymysql, tambahkan 2 baris ini di
  stunting_project/wsgi.py dan manage.py SEBELUM baris lain:
    import pymysql
    pymysql.install_as_MySQLdb()

LANGKAH 4 — Cek konfigurasi settings.py
-----------------------------------------
Buka file stunting_project/settings.py
Cari bagian DATABASES, pastikan sesuai:

    DATABASES = {
        'default': {
            'ENGINE': 'django.db.backends.mysql',
            'NAME':   'stunting_db',   ← nama database
            'USER':   'root',          ← user MySQL (default XAMPP: root)
            'PASSWORD': '',            ← password (default XAMPP: kosong)
            'HOST':   '127.0.0.1',
            'PORT':   '3306',
        }
    }

  Jika XAMPP MySQL Anda punya password, isi PASSWORD-nya.

LANGKAH 5 — Migrate & Setup
------------------------------
Jalankan di Command Prompt (di folder stunting_app):

    python manage.py migrate
    python manage.py setup_app
    python manage.py runserver

LANGKAH 6 — Verifikasi di phpMyAdmin
--------------------------------------
1. Buka http://localhost/phpmyadmin
2. Klik database "stunting_db"
3. Pastikan tabel-tabel berikut sudah terbuat:
   - auth_user
   - accounts_userprofile
   - balita_balita
   - balita_prediksihasil
   - ml_engine_modelnaivebayes
   - django_session
   - dan tabel Django lainnya

SELESAI! Buka http://127.0.0.1:8000
Login: admin / admin123

=====================================================
  TROUBLESHOOTING
=====================================================

ERROR: "No module named 'MySQLdb'"
  → Jalankan: pip install mysqlclient
  → Atau: pip install pymysql (lalu tambahkan kode di wsgi.py)

ERROR: "Access denied for user 'root'"
  → Cek password MySQL Anda di phpMyAdmin
  → Isi field PASSWORD di settings.py

ERROR: "Can't connect to MySQL server"
  → Pastikan MySQL di XAMPP sudah START (warna hijau)
  → Cek HOST dan PORT di settings.py

ERROR: "Unknown database 'stunting_db'"
  → Jalankan dulu script stunting_db.sql di phpMyAdmin

ERROR: "Incorrect string value" / karakter aneh
  → Pastikan database dibuat dengan utf8mb4_unicode_ci
  → Sudah diatur otomatis di script stunting_db.sql

=====================================================
