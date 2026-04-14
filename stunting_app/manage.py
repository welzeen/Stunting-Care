#!/usr/bin/env python
import os
import sys


# ── MySQL driver: coba mysqlclient dulu, fallback ke pymysql ──────────────
try:
    import MySQLdb  # mysqlclient
except ImportError:
    try:
        import pymysql
        pymysql.install_as_MySQLdb()
    except ImportError:
        pass  # SQLite tetap bisa jalan


def auto_migrate():
    """Otomatis migrate jika tabel auth_user belum ada."""
    try:
        import django
        django.setup()
        from django.db import connection
        tables = connection.introspection.table_names()
        if 'auth_user' not in tables:
            print("[StuntingApp] Database belum siap, menjalankan migrate otomatis...")
            from django.core.management import call_command
            call_command('migrate', '--run-syncdb', verbosity=1)
            print("[StuntingApp] Migrate selesai.")
    except Exception as e:
        print(f"[StuntingApp] Auto-migrate dilewati: {e}")


def main():
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'stunting_project.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed?"
        ) from exc

    # Otomatis migrate saat runserver jika DB belum siap
    if len(sys.argv) > 1 and sys.argv[1] in ('runserver', 'runserver_plus'):
        auto_migrate()

    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
