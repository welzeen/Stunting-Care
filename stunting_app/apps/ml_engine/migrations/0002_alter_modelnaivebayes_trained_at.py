from django.db import migrations, models


class Migration(migrations.Migration):
    """
    Mengubah trained_at dari auto_now (NOT NULL) menjadi nullable.
    Wajib jalankan: python manage.py migrate
    """

    dependencies = [
        ('ml_engine', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='modelnaivebayes',
            name='trained_at',
            field=models.DateTimeField(blank=True, null=True, default=None),
        ),
    ]
