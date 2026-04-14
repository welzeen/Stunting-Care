from django.db import models
from django.contrib.auth.models import User
import json


class ModelNaiveBayes(models.Model):
    STATUS_CHOICES = [('trained', 'Trained'), ('untrained', 'Belum Dilatih')]

    nama_model = models.CharField(max_length=100, default='Naive Bayes Stunting')
    akurasi = models.FloatField(default=0)
    presisi = models.FloatField(default=0)
    recall = models.FloatField(default=0)
    f1_score = models.FloatField(default=0)
    
    jumlah_training = models.IntegerField(default=0)
    jumlah_testing = models.IntegerField(default=0)
    
    confusion_matrix_data = models.TextField(default='{}')
    classification_report = models.TextField(default='{}')
    
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='untrained')
    trained_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    trained_at = models.DateTimeField(null=True, blank=True)   # hanya diisi saat benar-benar dilatih
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = 'Model Naive Bayes'
        ordering = ['-trained_at']

    def __str__(self):
        return f"{self.nama_model} - Akurasi: {self.akurasi:.2%}"

    def get_confusion_matrix(self):
        try:
            return json.loads(self.confusion_matrix_data)
        except Exception:
            return {}

    def get_classification_report(self):
        try:
            return json.loads(self.classification_report)
        except Exception:
            return {}

    def get_akurasi_persen(self):
        return round(self.akurasi * 100, 2)

    def get_presisi_persen(self):
        return round(self.presisi * 100, 2)

    def get_recall_persen(self):
        return round(self.recall * 100, 2)

    def get_f1_persen(self):
        return round(self.f1_score * 100, 2)
