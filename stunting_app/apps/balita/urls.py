from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard_view, name='dashboard'),
    path('balita/', views.balita_list_view, name='balita_list'),
    path('balita/tambah/', views.balita_create_view, name='balita_create'),
    path('balita/import/', views.import_excel_view, name='import_excel'),
    path('balita/export/', views.export_excel_view, name='export_excel'),
    path('balita/export-pdf-semua/', views.export_pdf_all_balita_view, name='export_pdf_all_balita'),
    path('balita/kelola-dataset/', views.kelola_dataset_view, name='kelola_dataset'),
    path('balita/upload-dataset/', views.upload_dataset_view, name='upload_dataset'),
    path('balita/upload-split/', views.upload_split_view, name='upload_split'),
    path('balita/hapus-dataset/', views.hapus_dataset_view, name='hapus_dataset'),
    path('balita/<int:pk>/', views.balita_detail_view, name='balita_detail'),
    path('balita/<int:pk>/edit/', views.balita_edit_view, name='balita_edit'),
    path('balita/<int:pk>/hapus/', views.balita_delete_view, name='balita_delete'),
    path('balita/<int:pk>/export-pdf/', views.export_pdf_balita_view, name='export_pdf_balita'),
    path('prediksi/riwayat/', views.prediksi_list_view, name='prediksi_list'),
    path('prediksi/riwayat/<int:pk>/hapus/', views.prediksi_delete_view, name='prediksi_delete'),
    path('prediksi/riwayat/hapus-semua/', views.prediksi_delete_all_view, name='prediksi_delete_all'),
    path('tren-stunting/', views.tren_stunting_view, name='tren_stunting'),
    path('api/stats/', views.api_stats_view, name='api_stats'),
]
