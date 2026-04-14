from django.db import models
from django.contrib.auth.models import User


class Balita(models.Model):
    JK_CHOICES = [('L', 'Laki-laki'), ('P', 'Perempuan')]
    UMUR_CHOICES = [('1-34 Bulan', '1-34 Bulan'), ('35-69 Bulan', '35-69 Bulan')]
    BBU_CHOICES = [
        ('Sangat Kurang', 'Sangat Kurang'),
        ('Kurang', 'Kurang'),
        ('Normal', 'Normal'),
        ('Berat Badan Normal', 'Berat Badan Normal'),
        ('Risiko Lebih', 'Risiko Lebih'),
    ]
    TBU_CHOICES = [
        ('Sangat Pendek', 'Sangat Pendek'),
        ('Pendek', 'Pendek'),
        ('Normal', 'Normal'),
        ('Tinggi', 'Tinggi'),
    ]
    STATUS_GIZI_CHOICES = [
        ('Gizi Buruk', 'Gizi Buruk'),
        ('Gizi Kurang', 'Gizi Kurang'),
        ('Gizi Baik', 'Gizi Baik'),
        ('Risiko Gizi Lebih', 'Risiko Gizi Lebih'),
        ('Gizi Lebih', 'Gizi Lebih'),
        ('Obesitas', 'Obesitas'),
    ]
    ASI_CHOICES = [('Ya', 'Ya'), ('Tidak', 'Tidak')]
    STUNTING_CHOICES = [
        ('Tidak', 'Tidak Stunting'),
        ('Potensi Stunting', 'Potensi Stunting'),
        ('Stunting', 'Stunting'),
    ]
    DATASET_CHOICES = [('training', 'Training'), ('testing', 'Testing')]

    kode_balita = models.CharField(max_length=50, unique=True)
    nama_balita = models.CharField(max_length=100, blank=True, default='')
    jenis_kelamin = models.CharField(max_length=1, choices=JK_CHOICES)
    umur = models.CharField(max_length=20, choices=UMUR_CHOICES)
    berat_badan = models.FloatField(help_text='Berat dalam kg')
    tinggi_badan = models.FloatField(help_text='Tinggi dalam cm')
    status_bbu = models.CharField(max_length=30, choices=BBU_CHOICES, verbose_name='BB/U')
    status_tbu = models.CharField(max_length=30, choices=TBU_CHOICES, verbose_name='TB/U')
    status_gizi = models.CharField(max_length=30, choices=STATUS_GIZI_CHOICES)
    status_asi = models.CharField(max_length=5, choices=ASI_CHOICES, verbose_name='ASI Eksklusif')
    status_stunting = models.CharField(max_length=20, choices=STUNTING_CHOICES, blank=True, null=True)
    dataset_type = models.CharField(max_length=10, choices=DATASET_CHOICES, default='training')
    
    created_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = 'Data Balita'
        verbose_name_plural = 'Data Balita'
        ordering = ['kode_balita']

    def __str__(self):
        return f"{self.kode_balita} - {self.get_jenis_kelamin_display()}"

    def get_stunting_badge(self):
        badges = {
            'Tidak': 'success',
            'Potensi Stunting': 'warning',
            'Stunting': 'danger',
        }
        return badges.get(self.status_stunting, 'secondary')


class PrediksiHasil(models.Model):
    balita = models.ForeignKey(Balita, on_delete=models.CASCADE, related_name='prediksi', null=True, blank=True)
    nama_balita = models.CharField(max_length=100, blank=True, default='', verbose_name='Nama/Kode Balita')
    jenis_kelamin = models.CharField(max_length=1)
    umur = models.CharField(max_length=20)
    berat_badan = models.FloatField()
    tinggi_badan = models.FloatField()
    status_bbu = models.CharField(max_length=30)
    status_tbu = models.CharField(max_length=30)
    status_gizi = models.CharField(max_length=30)
    status_asi = models.CharField(max_length=5)
    
    hasil_prediksi = models.CharField(max_length=20)
    probabilitas_tidak = models.FloatField(default=0)
    probabilitas_potensi = models.FloatField(default=0)
    probabilitas_stunting = models.FloatField(default=0)
    
    predicted_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    predicted_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = 'Hasil Prediksi'
        ordering = ['-predicted_at']

    def __str__(self):
        return f"Prediksi: {self.hasil_prediksi} ({self.predicted_at.strftime('%d/%m/%Y')})"
