from django.db.models.signals import post_save
from django.contrib.auth.models import User
from django.dispatch import receiver
from django.db import OperationalError, ProgrammingError


@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    """Buat UserProfile otomatis saat User baru dibuat.
    Pakai get_or_create agar aman bila view sudah duluan membuatnya."""
    if not created:
        return
    try:
        from .models import UserProfile
        UserProfile.objects.get_or_create(user=instance)
    except (OperationalError, ProgrammingError):
        # Tabel belum ada saat migrate pertama — abaikan
        pass
