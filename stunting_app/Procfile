web: python manage.py migrate --run-syncdb && python manage.py setup_app && python manage.py collectstatic --noinput && gunicorn stunting_project.wsgi --bind 0.0.0.0:$PORT
