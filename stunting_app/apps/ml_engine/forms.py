from django import forms
from apps.balita.models import Balita


class PrediksiForm(forms.Form):
    nama_balita = forms.CharField(
        label='Nama / Kode Balita',
        max_length=100,
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Contoh: Budi / anonymus0001 (opsional)'
        })
    )
    jenis_kelamin = forms.ChoiceField(
        choices=[('L', 'Laki-laki'), ('P', 'Perempuan')],
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    umur = forms.ChoiceField(
        choices=[('1-34 Bulan', '1-34 Bulan'), ('35-69 Bulan', '35-69 Bulan')],
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    status_bbu = forms.ChoiceField(
        label='Status BB/U',
        choices=[
            ('Sangat Kurang', 'Sangat Kurang'),
            ('Kurang', 'Kurang'),
            ('Normal', 'Normal'),
            ('Berat Badan Normal', 'Berat Badan Normal'),
            ('Risiko Lebih', 'Risiko Lebih'),
        ],
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    status_tbu = forms.ChoiceField(
        label='Status TB/U',
        choices=[
            ('Sangat Pendek', 'Sangat Pendek'),
            ('Pendek', 'Pendek'),
            ('Normal', 'Normal'),
            ('Tinggi', 'Tinggi'),
        ],
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    status_gizi = forms.ChoiceField(
        label='Status Gizi',
        choices=[
            ('Gizi Buruk', 'Gizi Buruk'),
            ('Gizi Kurang', 'Gizi Kurang'),
            ('Gizi Baik', 'Gizi Baik'),
            ('Risiko Gizi Lebih', 'Risiko Gizi Lebih'),
            ('Gizi Lebih', 'Gizi Lebih'),
            ('Obesitas', 'Obesitas'),
        ],
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    status_asi = forms.ChoiceField(
        label='ASI Eksklusif',
        choices=[('Ya', 'Ya'), ('Tidak', 'Tidak')],
        widget=forms.Select(attrs={'class': 'form-select'})
    )
