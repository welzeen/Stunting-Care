from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='Balita',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('kode_balita', models.CharField(max_length=50, unique=True)),
                ('nama_balita', models.CharField(blank=True, default='', max_length=100)),
                ('jenis_kelamin', models.CharField(choices=[('L', 'Laki-laki'), ('P', 'Perempuan')], max_length=1)),
                ('umur', models.CharField(choices=[('1-34 Bulan', '1-34 Bulan'), ('35-69 Bulan', '35-69 Bulan')], max_length=20)),
                ('berat_badan', models.FloatField(help_text='Berat dalam kg')),
                ('tinggi_badan', models.FloatField(help_text='Tinggi dalam cm')),
                ('status_bbu', models.CharField(choices=[('Sangat Kurang', 'Sangat Kurang'), ('Kurang', 'Kurang'), ('Normal', 'Normal'), ('Berat Badan Normal', 'Berat Badan Normal'), ('Risiko Lebih', 'Risiko Lebih')], max_length=30, verbose_name='BB/U')),
                ('status_tbu', models.CharField(choices=[('Sangat Pendek', 'Sangat Pendek'), ('Pendek', 'Pendek'), ('Normal', 'Normal'), ('Tinggi', 'Tinggi')], max_length=30, verbose_name='TB/U')),
                ('status_gizi', models.CharField(choices=[('Gizi Buruk', 'Gizi Buruk'), ('Gizi Kurang', 'Gizi Kurang'), ('Gizi Baik', 'Gizi Baik'), ('Risiko Gizi Lebih', 'Risiko Gizi Lebih'), ('Gizi Lebih', 'Gizi Lebih'), ('Obesitas', 'Obesitas')], max_length=30)),
                ('status_asi', models.CharField(choices=[('Ya', 'Ya'), ('Tidak', 'Tidak')], max_length=5, verbose_name='ASI Eksklusif')),
                ('status_stunting', models.CharField(blank=True, choices=[('Tidak', 'Tidak Stunting'), ('Potensi Stunting', 'Potensi Stunting'), ('Stunting', 'Stunting')], max_length=20, null=True)),
                ('dataset_type', models.CharField(choices=[('training', 'Training'), ('testing', 'Testing')], default='training', max_length=10)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('created_by', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'verbose_name': 'Data Balita',
                'verbose_name_plural': 'Data Balita',
                'ordering': ['kode_balita'],
            },
        ),
        migrations.CreateModel(
            name='PrediksiHasil',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('jenis_kelamin', models.CharField(max_length=1)),
                ('umur', models.CharField(max_length=20)),
                ('berat_badan', models.FloatField()),
                ('tinggi_badan', models.FloatField()),
                ('status_bbu', models.CharField(max_length=30)),
                ('status_tbu', models.CharField(max_length=30)),
                ('status_gizi', models.CharField(max_length=30)),
                ('status_asi', models.CharField(max_length=5)),
                ('hasil_prediksi', models.CharField(max_length=20)),
                ('probabilitas_tidak', models.FloatField(default=0)),
                ('probabilitas_potensi', models.FloatField(default=0)),
                ('probabilitas_stunting', models.FloatField(default=0)),
                ('predicted_at', models.DateTimeField(auto_now_add=True)),
                ('balita', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, related_name='prediksi', to='balita.balita')),
                ('predicted_by', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'verbose_name': 'Hasil Prediksi',
                'ordering': ['-predicted_at'],
            },
        ),
    ]
