from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from django.shortcuts import redirect

urlpatterns = [
    # path('admin/', admin.site.urls),  # Dinonaktifkan
    path('', lambda request: redirect('dashboard'), name='home'),
    path('accounts/', include('apps.accounts.urls')),
    path('dashboard/', include('apps.balita.urls')),
    path('ml/', include('apps.ml_engine.urls')),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
