from django import forms
from .models import Balita


class BalitaForm(forms.ModelForm):
    class Meta:
        model = Balita
        fields = [
            'kode_balita', 'nama_balita', 'jenis_kelamin', 'umur',
            'berat_badan', 'tinggi_badan', 'status_bbu', 'status_tbu',
            'status_gizi', 'status_asi', 'status_stunting', 'dataset_type'
        ]
        widgets = {
            'kode_balita': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Contoh: anonymus0001'}),
            'nama_balita': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Nama balita (opsional)'}),
            'jenis_kelamin': forms.Select(attrs={'class': 'form-select'}),
            'umur': forms.Select(attrs={'class': 'form-select'}),
            'berat_badan': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1', 'placeholder': 'kg'}),
            'tinggi_badan': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1', 'placeholder': 'cm'}),
            'status_bbu': forms.Select(attrs={'class': 'form-select'}),
            'status_tbu': forms.Select(attrs={'class': 'form-select'}),
            'status_gizi': forms.Select(attrs={'class': 'form-select'}),
            'status_asi': forms.Select(attrs={'class': 'form-select'}),
            'status_stunting': forms.Select(attrs={'class': 'form-select'}),
            'dataset_type': forms.Select(attrs={'class': 'form-select'}),
        }
        labels = {
            'kode_balita': 'Kode/Nama Balita',
            'nama_balita': 'Nama Lengkap',
            'jenis_kelamin': 'Jenis Kelamin',
            'umur': 'Kelompok Umur',
            'berat_badan': 'Berat Badan (kg)',
            'tinggi_badan': 'Tinggi Badan (cm)',
            'status_bbu': 'Status BB/U',
            'status_tbu': 'Status TB/U',
            'status_gizi': 'Status Gizi',
            'status_asi': 'ASI Eksklusif',
            'status_stunting': 'Status Stunting (Label)',
            'dataset_type': 'Jenis Dataset',
        }


class ImportExcelForm(forms.Form):
    file = forms.FileField(
        label='File Excel (.xlsx)',
        widget=forms.FileInput(attrs={'class': 'form-control', 'accept': '.xlsx,.xls'})
    )
    dataset_type = forms.ChoiceField(
        choices=[('training', 'Data Training'), ('testing', 'Data Testing')],
        widget=forms.Select(attrs={'class': 'form-select'}),
        label='Jenis Dataset'
    )
    overwrite = forms.BooleanField(
        required=False,
        initial=False,
        label='Timpa data yang sudah ada',
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )


class FilterBalitaForm(forms.Form):
    search = forms.CharField(
        required=False,
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Cari kode/nama balita...'})
    )
    jenis_kelamin = forms.ChoiceField(
        required=False,
        choices=[('', 'Semua'), ('L', 'Laki-laki'), ('P', 'Perempuan')],
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    status_stunting = forms.ChoiceField(
        required=False,
        choices=[('', 'Semua'), ('Tidak', 'Tidak'), ('Potensi Stunting', 'Potensi Stunting'), ('Stunting', 'Stunting')],
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    dataset_type = forms.ChoiceField(
        required=False,
        choices=[('', 'Semua'), ('training', 'Training'), ('testing', 'Testing')],
        widget=forms.Select(attrs={'class': 'form-select'})
    )
