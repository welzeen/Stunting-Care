import os

# ── MySQL driver: coba mysqlclient dulu, fallback ke pymysql ──────────────
try:
    import MySQLdb
except ImportError:
    try:
        import pymysql
        pymysql.install_as_MySQLdb()
    except ImportError:
        pass

from django.core.wsgi import get_wsgi_application
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'stunting_project.settings')
application = get_wsgi_application()
