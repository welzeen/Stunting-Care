#!/usr/bin/env python
"""
JALANKAN FILE INI untuk pertama kali setup:
    python start.py

Script ini akan otomatis:
1. Cek & install dependensi
2. Jalankan migrate
3. Buat user admin & demo
4. Import data dari Excel
5. Jalankan server
"""
import os
import sys
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'stunting_project.settings')
sys.path.insert(0, BASE_DIR)


def run(cmd, check=True):
    print(f"  > {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=BASE_DIR)
    if check and result.returncode != 0:
        print(f"  [ERROR] Perintah gagal: {cmd}")
        sys.exit(1)
    return result.returncode == 0


def main():
    print("=" * 50)
    print("  StuntingApp - Auto Setup & Start")
    print("=" * 50)

    # 1. Migrate
    print("\n[1/4] Migrasi database...")
    run(f"{sys.executable} manage.py migrate --run-syncdb")
    print("  OK")

    # 2. Buat user & seed data
    print("\n[2/4] Setup user & data...")
    excel_path = os.path.join(BASE_DIR, 'data', 'dataset.xlsx')
    if os.path.exists(excel_path):
        run(f'{sys.executable} manage.py setup_app --excel "{excel_path}"')
    else:
        run(f"{sys.executable} manage.py setup_app --skip-data")
    print("  OK")

    # 3. Jalankan server
    print("\n[3/4] Memulai server...")
    print("\n" + "=" * 50)
    print("  Buka browser: http://127.0.0.1:8000")
    print("  Login admin : admin / admin123")
    print("  Login user  : user  / user123")
    print("  Tekan Ctrl+C untuk berhenti")
    print("=" * 50 + "\n")
    os.system(f"{sys.executable} manage.py runserver")


if __name__ == '__main__':
    main()
