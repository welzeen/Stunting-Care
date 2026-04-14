from django.shortcuts import render, redirect
from django.contrib.auth import login, logout, authenticate, update_session_auth_hash
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.contrib import messages
from django.db import transaction
from .models import UserProfile
from .forms import LoginForm, RegisterForm, ProfileForm, ChangePasswordForm


def login_view(request):
    if request.user.is_authenticated:
        return redirect('dashboard')
    
    form = LoginForm(request.POST or None)
    if request.method == 'POST' and form.is_valid():
        username = form.cleaned_data['username']
        password = form.cleaned_data['password']
        user = authenticate(request, username=username, password=password)
        if user:
            login(request, user)
            messages.success(request, f'Selamat datang, {user.get_full_name() or user.username}!')
            return redirect('dashboard')
        else:
            messages.error(request, 'Username atau password salah.')
    
    return render(request, 'accounts/login.html', {'form': form})


def logout_view(request):
    logout(request)
    messages.info(request, 'Anda telah berhasil logout.')
    return redirect('login')


def register_view(request):
    if request.user.is_authenticated:
        return redirect('dashboard')
    
    form = RegisterForm(request.POST or None)
    if request.method == 'POST' and form.is_valid():
        with transaction.atomic():
            user = form.save(commit=False)
            user.set_password(form.cleaned_data['password1'])
            user.save()
            # Gunakan get_or_create agar tidak bentrok dengan signal post_save
            profile, _ = UserProfile.objects.get_or_create(user=user)
            profile.role = 'pengunjung'
            profile.save()
            messages.success(request,
                f'Akun "{user.username}" berhasil didaftarkan! '
                f'Silakan login menggunakan username dan password Anda.')
            return redirect('login')
    
    return render(request, 'accounts/register.html', {'form': form})


@login_required
def profile_view(request):
    profile, _ = UserProfile.objects.get_or_create(user=request.user)
    form = ProfileForm(request.POST or None, request.FILES or None, instance=profile, user=request.user)
    
    if request.method == 'POST' and form.is_valid():
        form.save()
        messages.success(request, 'Profil berhasil diperbarui!')
        return redirect('profile')
    
    return render(request, 'accounts/profile.html', {'form': form, 'profile': profile})


@login_required
def change_password_view(request):
    form = ChangePasswordForm(request.user, request.POST or None)
    if request.method == 'POST' and form.is_valid():
        user = form.save()
        update_session_auth_hash(request, user)
        messages.success(request, 'Password berhasil diubah!')
        return redirect('profile')
    
    return render(request, 'accounts/change_password.html', {'form': form})


@login_required
def user_list_view(request):
    profile = getattr(request.user, 'profile', None)
    if not profile or not profile.is_admin():
        messages.error(request, 'Akses ditolak.')
        return redirect('dashboard')
    
    users = User.objects.select_related('profile').all().order_by('-date_joined')
    return render(request, 'accounts/user_list.html', {'users': users})


@login_required
def user_delete_view(request, pk):
    profile = getattr(request.user, 'profile', None)
    if not profile or not profile.is_admin():
        messages.error(request, 'Akses ditolak.')
        return redirect('dashboard')
    
    try:
        user = User.objects.get(pk=pk)
        if user == request.user:
            messages.error(request, 'Tidak dapat menghapus akun sendiri.')
        else:
            user.delete()
            messages.success(request, f'Pengguna {user.username} berhasil dihapus.')
    except User.DoesNotExist:
        messages.error(request, 'Pengguna tidak ditemukan.')
    
    return redirect('user_list')


def forgot_password_view(request):
    """Halaman lupa password — reset via username."""
    if request.user.is_authenticated:
        return redirect('dashboard')

    if request.method == 'POST':
        username = request.POST.get('username', '').strip()
        new_password = request.POST.get('new_password', '').strip()
        confirm_password = request.POST.get('confirm_password', '').strip()

        if not username:
            messages.error(request, 'Username tidak boleh kosong.')
        elif not new_password:
            messages.error(request, 'Password baru tidak boleh kosong.')
        elif len(new_password) < 6:
            messages.error(request, 'Password minimal 6 karakter.')
        elif new_password != confirm_password:
            messages.error(request, 'Konfirmasi password tidak cocok.')
        else:
            try:
                user = User.objects.get(username=username)
                user.set_password(new_password)
                user.save()
                messages.success(request,
                    f'Password untuk akun "{username}" berhasil direset. Silakan login.')
                return redirect('login')
            except User.DoesNotExist:
                messages.error(request, f'Username "{username}" tidak ditemukan.')

    return render(request, 'accounts/forgot_password.html')
