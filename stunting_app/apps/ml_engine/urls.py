from django.urls import path
from . import views

urlpatterns = [
    path('', views.ml_dashboard_view, name='ml_dashboard'),
    path('train/', views.train_model_view, name='train_model'),
    path('prediksi/', views.prediksi_view, name='prediksi'),
    path('batch-predict/', views.batch_predict_view, name='batch_predict'),
    path('evaluasi-data/', views.evaluasi_data_view, name='evaluasi_data'),
    path('korelasi/', views.korelasi_view, name='korelasi'),
]
