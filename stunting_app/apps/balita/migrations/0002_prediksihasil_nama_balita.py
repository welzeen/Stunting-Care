from django.db import migrations, models

class Migration(migrations.Migration):

    dependencies = [
        ('balita', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='prediksihasil',
            name='nama_balita',
            field=models.CharField(blank=True, default='', max_length=100, verbose_name='Nama/Kode Balita'),
        ),
    ]
