-- ============================================================
--  STUNTING DB — Script Buat Database di phpMyAdmin / XAMPP
--  Jalankan script ini SEBELUM menjalankan aplikasi
--  Cara: Buka phpMyAdmin → tab SQL → paste → klik Go
-- ============================================================

-- 1. Buat database
CREATE DATABASE IF NOT EXISTS stunting_db
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

-- 2. Gunakan database
USE stunting_db;

-- ============================================================
-- Selesai. Tabel akan dibuat otomatis oleh Django migrate.
-- Lanjut jalankan di command prompt:
--   pip install -r requirements.txt
--   python manage.py migrate
--   python manage.py setup_app
--   python manage.py runserver
-- ============================================================
