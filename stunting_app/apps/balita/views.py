from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.core.paginator import Paginator
from django.http import HttpResponse, JsonResponse
from django.db.models import Q, Count
import pandas as pd
import io, json
from .models import Balita, PrediksiHasil
from .forms import BalitaForm, ImportExcelForm, FilterBalitaForm


def is_admin(user):
    return hasattr(user, 'profile') and user.profile.is_admin()


@login_required
def dashboard_view(request):
    total_balita = Balita.objects.count()
    total_training = Balita.objects.filter(dataset_type='training').count()
    total_testing = Balita.objects.filter(dataset_type='testing').count()
    total_prediksi = PrediksiHasil.objects.count()

    stunting_counts = Balita.objects.values('status_stunting').annotate(jumlah=Count('id'))
    stunting_data = {'Tidak': 0, 'Potensi Stunting': 0, 'Stunting': 0}
    for item in stunting_counts:
        if item['status_stunting']:
            stunting_data[item['status_stunting']] = item['jumlah']

    jk_counts = Balita.objects.values('jenis_kelamin').annotate(jumlah=Count('id'))
    jk_data = {'L': 0, 'P': 0}
    for item in jk_counts:
        jk_data[item['jenis_kelamin']] = item['jumlah']

    gizi_counts = Balita.objects.values('status_gizi').annotate(jumlah=Count('id')).order_by('-jumlah')[:6]

    recent_balita = Balita.objects.order_by('-created_at')[:5]
    recent_prediksi = PrediksiHasil.objects.order_by('-predicted_at')[:5]

    context = {
        'total_balita': total_balita,
        'total_training': total_training,
        'total_testing': total_testing,
        'total_prediksi': total_prediksi,
        'stunting_data': json.dumps(stunting_data),
        'jk_data': json.dumps(jk_data),
        'gizi_counts': list(gizi_counts),
        'recent_balita': recent_balita,
        'recent_prediksi': recent_prediksi,
    }
    return render(request, 'dashboard/index.html', context)


@login_required
def balita_list_view(request):
    form = FilterBalitaForm(request.GET)
    qs = Balita.objects.all()

    if form.is_valid():
        search = form.cleaned_data.get('search')
        jk = form.cleaned_data.get('jenis_kelamin')
        stunting = form.cleaned_data.get('status_stunting')
        dtype = form.cleaned_data.get('dataset_type')
        if search:
            qs = qs.filter(Q(kode_balita__icontains=search) | Q(nama_balita__icontains=search))
        if jk:
            qs = qs.filter(jenis_kelamin=jk)
        if stunting:
            qs = qs.filter(status_stunting=stunting)
        if dtype:
            qs = qs.filter(dataset_type=dtype)

    paginator = Paginator(qs, 20)
    page_obj = paginator.get_page(request.GET.get('page'))

    context = {
        'page_obj': page_obj,
        'form': form,
        'total': qs.count(),
    }
    return render(request, 'balita/list.html', context)


@login_required
def balita_detail_view(request, pk):
    balita = get_object_or_404(Balita, pk=pk)
    prediksi_list = balita.prediksi.order_by('-predicted_at')[:5]
    return render(request, 'balita/detail.html', {'balita': balita, 'prediksi_list': prediksi_list})


@login_required
def balita_create_view(request):
    if not is_admin(request.user):
        messages.error(request, 'Hanya admin yang dapat menambah data.')
        return redirect('balita_list')

    form = BalitaForm(request.POST or None)
    if request.method == 'POST' and form.is_valid():
        balita = form.save(commit=False)
        balita.created_by = request.user
        balita.save()
        messages.success(request, f'Data balita {balita.kode_balita} berhasil ditambahkan.')
        return redirect('balita_list')

    return render(request, 'balita/form.html', {'form': form, 'title': 'Tambah Data Balita', 'action': 'Simpan'})


@login_required
def balita_edit_view(request, pk):
    if not is_admin(request.user):
        messages.error(request, 'Hanya admin yang dapat mengedit data.')
        return redirect('balita_list')

    balita = get_object_or_404(Balita, pk=pk)
    form = BalitaForm(request.POST or None, instance=balita)
    if request.method == 'POST' and form.is_valid():
        form.save()
        messages.success(request, f'Data balita {balita.kode_balita} berhasil diperbarui.')
        return redirect('balita_detail', pk=pk)

    return render(request, 'balita/form.html', {
        'form': form, 'balita': balita,
        'title': f'Edit Data: {balita.kode_balita}', 'action': 'Update'
    })


@login_required
def balita_delete_view(request, pk):
    if not is_admin(request.user):
        messages.error(request, 'Hanya admin yang dapat menghapus data.')
        return redirect('balita_list')

    balita = get_object_or_404(Balita, pk=pk)
    if request.method == 'POST':
        kode = balita.kode_balita
        balita.delete()
        messages.success(request, f'Data balita {kode} berhasil dihapus.')
        return redirect('balita_list')

    return render(request, 'balita/delete_confirm.html', {'balita': balita})


@login_required
def import_excel_view(request):
    if not is_admin(request.user):
        messages.error(request, 'Hanya admin yang dapat mengimpor data.')
        return redirect('balita_list')

    form = ImportExcelForm(request.POST or None, request.FILES or None)
    if request.method == 'POST' and form.is_valid():
        excel_file = request.FILES['file']
        dataset_type = form.cleaned_data['dataset_type']
        overwrite = form.cleaned_data['overwrite']

        try:
            df = pd.read_excel(excel_file, sheet_name=0)
            # normalize column names
            df.columns = [str(c).strip() for c in df.columns]

            col_map = {
                'Nama/Kode balita': 'kode_balita',
                'JK': 'jenis_kelamin',
                'Umur ': 'umur',
                'Umur': 'umur',
                'Berat ': 'berat_badan',
                'Berat': 'berat_badan',
                'Tinggi': 'tinggi_badan',
                'BB/U': 'status_bbu',
                'TB/U': 'status_tbu',
                'Status Gizi': 'status_gizi',
                'Status ASI Ekslusif': 'status_asi',
                'Status Stunting': 'status_stunting',
            }
            df = df.rename(columns=col_map)

            required = ['kode_balita', 'jenis_kelamin', 'umur', 'berat_badan', 'tinggi_badan',
                        'status_bbu', 'status_tbu', 'status_gizi', 'status_asi']
            missing = [c for c in required if c not in df.columns]
            if missing:
                messages.error(request, f'Kolom tidak ditemukan: {", ".join(missing)}')
                return render(request, 'balita/import.html', {'form': form})

            jk_map = {'L': 'L', 'L ': 'L', 'P': 'P', 'P ': 'P'}
            gizi_map = {
                'Gizi baik': 'Gizi Baik', 'baik': 'Gizi Baik',
                'gizi baik': 'Gizi Baik',
            }
            bbu_map = {'Berat Badan Normal': 'Berat Badan Normal'}

            created, updated, skipped = 0, 0, 0
            for _, row in df.iterrows():
                kode = str(row.get('kode_balita', '')).strip()
                if not kode or kode == 'nan':
                    continue
                try:
                    jk = jk_map.get(str(row.get('jenis_kelamin', '')).strip(),
                                    str(row.get('jenis_kelamin', '')).strip())
                    umur_val = str(row.get('umur', '')).strip()
                    if umur_val not in ['1-34 Bulan', '35-69 Bulan']:
                        umur_val = '35-69 Bulan' if float(umur_val) > 34 else '1-34 Bulan'

                    gizi = str(row.get('status_gizi', '')).strip()
                    gizi = gizi_map.get(gizi, gizi)

                    stunting = str(row.get('status_stunting', '')).strip()
                    if stunting == 'nan':
                        stunting = None

                    data = {
                        'jenis_kelamin': jk,
                        'umur': umur_val,
                        'berat_badan': float(row.get('berat_badan', 0)),
                        'tinggi_badan': float(row.get('tinggi_badan', 0)),
                        'status_bbu': str(row.get('status_bbu', '')).strip(),
                        'status_tbu': str(row.get('status_tbu', '')).strip(),
                        'status_gizi': gizi,
                        'status_asi': str(row.get('status_asi', 'Tidak')).strip(),
                        'status_stunting': stunting,
                        'dataset_type': dataset_type,
                        'created_by': request.user,
                    }

                    obj, was_created = Balita.objects.get_or_create(kode_balita=kode, defaults=data)
                    if was_created:
                        created += 1
                    elif overwrite:
                        for k, v in data.items():
                            setattr(obj, k, v)
                        obj.save()
                        updated += 1
                    else:
                        skipped += 1
                except Exception:
                    skipped += 1

            messages.success(request,
                f'Import selesai: {created} ditambahkan, {updated} diperbarui, {skipped} dilewati.')
            return redirect('balita_list')

        except Exception as e:
            messages.error(request, f'Gagal membaca file: {str(e)}')

    return render(request, 'balita/import.html', {'form': form})


@login_required
def export_excel_view(request):
    qs = Balita.objects.all()
    dtype = request.GET.get('dataset_type', '')
    if dtype:
        qs = qs.filter(dataset_type=dtype)

    data = [{
        'Kode Balita': b.kode_balita,
        'Nama': b.nama_balita,
        'Jenis Kelamin': b.get_jenis_kelamin_display(),
        'Umur': b.umur,
        'Berat (kg)': b.berat_badan,
        'Tinggi (cm)': b.tinggi_badan,
        'BB/U': b.status_bbu,
        'TB/U': b.status_tbu,
        'Status Gizi': b.status_gizi,
        'ASI Eksklusif': b.status_asi,
        'Status Stunting': b.status_stunting or '-',
        'Jenis Dataset': b.dataset_type,
    } for b in qs]

    df = pd.DataFrame(data)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Data Balita')
    output.seek(0)

    response = HttpResponse(
        output.getvalue(),
        content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )
    response['Content-Disposition'] = 'attachment; filename="data_balita.xlsx"'
    return response


@login_required
def prediksi_list_view(request):
    prediksi = PrediksiHasil.objects.select_related('predicted_by').order_by('-predicted_at')
    paginator = Paginator(prediksi, 15)
    page_obj = paginator.get_page(request.GET.get('page'))
    return render(request, 'balita/prediksi_list.html', {'page_obj': page_obj})


@login_required
def prediksi_delete_view(request, pk):
    if not (hasattr(request.user, 'profile') and request.user.profile.is_admin()):
        messages.error(request, 'Hanya admin yang dapat menghapus riwayat prediksi.')
        return redirect('prediksi_list')
    prediksi = get_object_or_404(PrediksiHasil, pk=pk)
    if request.method == 'POST':
        prediksi.delete()
        messages.success(request, 'Riwayat prediksi berhasil dihapus.')
    return redirect('prediksi_list')


@login_required
def prediksi_delete_all_view(request):
    if not (hasattr(request.user, 'profile') and request.user.profile.is_admin()):
        messages.error(request, 'Hanya admin yang dapat menghapus semua riwayat prediksi.')
        return redirect('prediksi_list')
    if request.method == 'POST':
        count = PrediksiHasil.objects.count()
        PrediksiHasil.objects.all().delete()
        messages.success(request, f'{count} riwayat prediksi berhasil dihapus.')
    return redirect('prediksi_list')


@login_required
def api_stats_view(request):
    """API endpoint untuk chart data"""
    stunting_by_jk = {}
    for jk in ['L', 'P']:
        counts = Balita.objects.filter(jenis_kelamin=jk).values('status_stunting').annotate(n=Count('id'))
        stunting_by_jk[jk] = {c['status_stunting']: c['n'] for c in counts if c['status_stunting']}

    umur_counts = Balita.objects.values('umur').annotate(n=Count('id'))

    return JsonResponse({
        'stunting_by_jk': stunting_by_jk,
        'umur_counts': list(umur_counts),
    })


@login_required
def export_pdf_all_balita_view(request):
    """Export laporan PDF semua data balita (dengan filter aktif)."""
    from datetime import datetime
    from .pdf_balita import generate_all_balita_pdf

    qs = Balita.objects.all()
    filters = {}

    search = request.GET.get('search', '').strip()
    jk = request.GET.get('jenis_kelamin', '').strip()
    stunting = request.GET.get('status_stunting', '').strip()
    dtype = request.GET.get('dataset_type', '').strip()

    if search:
        qs = qs.filter(Q(kode_balita__icontains=search) | Q(nama_balita__icontains=search))
        filters['Cari'] = search
    if jk:
        qs = qs.filter(jenis_kelamin=jk)
        filters['JK'] = 'Laki-laki' if jk == 'L' else 'Perempuan'
    if stunting:
        qs = qs.filter(status_stunting=stunting)
        filters['Status Stunting'] = stunting
    if dtype:
        qs = qs.filter(dataset_type=dtype)
        filters['Dataset'] = dtype.capitalize()

    buf = generate_all_balita_pdf(qs, filters=filters if filters else None)
    filename = f"laporan_semua_balita_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
    response = HttpResponse(buf, content_type='application/pdf')
    response['Content-Disposition'] = f'attachment; filename="{filename}"'
    return response


@login_required
def export_pdf_balita_view(request, pk):
    """Export laporan PDF per individu balita."""
    from datetime import datetime
    from django.http import HttpResponse
    from .pdf_balita import generate_balita_pdf

    balita = get_object_or_404(Balita, pk=pk)
    prediksi_list = balita.prediksi.order_by('-predicted_at')[:10]

    buf = generate_balita_pdf(balita, prediksi_list)
    filename = f"laporan_{balita.kode_balita}_{datetime.now().strftime('%Y%m%d')}.pdf"
    response = HttpResponse(buf, content_type='application/pdf')
    response['Content-Disposition'] = f'attachment; filename="{filename}"'
    return response


@login_required
def tren_stunting_view(request):
    """Halaman grafik tren stunting."""
    from django.db.models import Count

    # Per jenis kelamin
    jk_stunting = {}
    for jk, label in [('L', 'Laki-laki'), ('P', 'Perempuan')]:
        counts = (Balita.objects
                  .filter(jenis_kelamin=jk)
                  .exclude(status_stunting__isnull=True)
                  .values('status_stunting')
                  .annotate(n=Count('id')))
        jk_stunting[label] = {c['status_stunting']: c['n'] for c in counts}

    # Per kelompok umur
    umur_stunting = {}
    for umur in ['1-34 Bulan', '35-69 Bulan']:
        counts = (Balita.objects
                  .filter(umur=umur)
                  .exclude(status_stunting__isnull=True)
                  .values('status_stunting')
                  .annotate(n=Count('id')))
        umur_stunting[umur] = {c['status_stunting']: c['n'] for c in counts}

    # Per status gizi
    gizi_stunting = (Balita.objects
                     .exclude(status_stunting__isnull=True)
                     .values('status_gizi', 'status_stunting')
                     .annotate(n=Count('id'))
                     .order_by('status_gizi'))

    gizi_map = {}
    for item in gizi_stunting:
        g = item['status_gizi']
        if g not in gizi_map:
            gizi_map[g] = {}
        gizi_map[g][item['status_stunting']] = item['n']

    # Per dataset type
    dataset_stunting = {}
    for dtype in ['training', 'testing']:
        counts = (Balita.objects
                  .filter(dataset_type=dtype)
                  .exclude(status_stunting__isnull=True)
                  .values('status_stunting')
                  .annotate(n=Count('id')))
        dataset_stunting[dtype] = {c['status_stunting']: c['n'] for c in counts}

    # Summary cards
    total = Balita.objects.exclude(status_stunting__isnull=True).count()
    stunting_total    = Balita.objects.filter(status_stunting='Stunting').count()
    potensi_total     = Balita.objects.filter(status_stunting='Potensi Stunting').count()
    tidak_total       = Balita.objects.filter(status_stunting='Tidak').count()

    context = {
        'jk_stunting':      json.dumps(jk_stunting),
        'umur_stunting':    json.dumps(umur_stunting),
        'gizi_map':         json.dumps(gizi_map),
        'dataset_stunting': json.dumps(dataset_stunting),
        'total':            total,
        'stunting_total':   stunting_total,
        'potensi_total':    potensi_total,
        'tidak_total':      tidak_total,
    }
    return render(request, 'balita/tren_stunting.html', context)


@login_required
def kelola_dataset_view(request):
    """Halaman kelola dataset: upload training/testing & hapus massal."""
    if not is_admin(request.user):
        messages.error(request, 'Hanya admin yang dapat mengelola dataset.')
        return redirect('balita_list')

    from django.db.models import Count

    training_count = Balita.objects.filter(dataset_type='training').count()
    testing_count  = Balita.objects.filter(dataset_type='testing').count()

    context = {
        'training_count': training_count,
        'testing_count':  testing_count,
    }
    return render(request, 'balita/kelola_dataset.html', context)


@login_required
def upload_dataset_view(request):
    """Upload file Excel sebagai training atau testing (replace semua data lama)."""
    if not is_admin(request.user):
        messages.error(request, 'Hanya admin yang dapat mengunggah dataset.')
        return redirect('balita_list')

    if request.method != 'POST':
        return redirect('kelola_dataset')

    file        = request.FILES.get('file')
    dtype       = request.POST.get('dataset_type', '')
    replace_all = request.POST.get('replace_all') == '1'

    if not file:
        messages.error(request, 'Pilih file Excel terlebih dahulu.')
        return redirect('kelola_dataset')
    if dtype not in ('training', 'testing'):
        messages.error(request, 'Jenis dataset tidak valid.')
        return redirect('kelola_dataset')
    if not file.name.endswith(('.xlsx', '.xls')):
        messages.error(request, 'File harus berformat Excel (.xlsx / .xls).')
        return redirect('kelola_dataset')

    try:
        df = pd.read_excel(file, sheet_name=0)
        df.columns = [str(c).strip() for c in df.columns]

        col_map = {
            'Nama/Kode balita': 'kode_balita', 'JK': 'jenis_kelamin',
            'Umur ': 'umur', 'Umur': 'umur',
            'Berat ': 'berat_badan', 'Berat': 'berat_badan',
            'Tinggi': 'tinggi_badan', 'BB/U': 'status_bbu', 'TB/U': 'status_tbu',
            'Status Gizi': 'status_gizi',
            'Status ASI Ekslusif': 'status_asi', 'Status ASI Eksklusif': 'status_asi',
            'Status Stunting': 'status_stunting',
            # Kolom alternatif (nama pendek)
            'kode': 'kode_balita', 'jk': 'jenis_kelamin',
            'bbu': 'status_bbu', 'tbu': 'status_tbu',
            'gizi': 'status_gizi', 'asi': 'status_asi',
            'stunting': 'status_stunting',
            'berat': 'berat_badan', 'tinggi': 'tinggi_badan',
        }
        df = df.rename(columns=col_map)

        # Validasi kolom wajib
        required = ['kode_balita','jenis_kelamin','umur','berat_badan',
                    'tinggi_badan','status_bbu','status_tbu','status_gizi','status_asi']
        missing = [c for c in required if c not in df.columns]
        if missing:
            messages.error(request, f'Kolom tidak ditemukan: {", ".join(missing)}')
            return redirect('kelola_dataset')

        gizi_fix = {
            'gizi baik':'Gizi Baik','baik':'Gizi Baik','gizi buruk':'Gizi Buruk',
            'gizi kurang':'Gizi Kurang','gizi lebih':'Gizi Lebih',
            'risiko gizi lebih':'Risiko Gizi Lebih','obesitas':'Obesitas',
        }
        asi_fix = {'ya':'Ya','y':'Ya','tidak':'Tidak','t':'Tidak','no':'Tidak'}
        bbu_fix = {
            'sangat kurang':'Sangat Kurang','kurang':'Kurang',
            'normal':'Normal','berat badan normal':'Berat Badan Normal',
            'risiko lebih':'Risiko Lebih',
        }
        tbu_fix = {
            'sangat pendek':'Sangat Pendek','pendek':'Pendek',
            'normal':'Normal','tinggi':'Tinggi',
        }
        stunting_fix = {
            'tidak':'Tidak','stunting':'Stunting',
            'potensi stunting':'Potensi Stunting',
        }

        # ── SELALU hapus data lama sesuai dtype sebelum insert ──────────────
        # Ini mencegah konflik kode_balita antara training & testing
        deleted, _ = Balita.objects.filter(dataset_type=dtype).delete()

        # Juga hapus data dengan kode yang sama di dtype lain agar tidak konflik unique
        kode_list = [str(r).strip() for r in df['kode_balita']
                     if str(r).strip() and str(r).strip() != 'nan']
        Balita.objects.filter(kode_balita__in=kode_list).delete()

        # ── Bulk insert ──────────────────────────────────────────────────────
        objs = []
        skipped = 0
        for _, row in df.iterrows():
            kode = str(row.get('kode_balita', '')).strip()
            if not kode or kode == 'nan':
                continue
            try:
                umur_val = str(row.get('umur', '')).strip()
                if umur_val not in ['1-34 Bulan', '35-69 Bulan']:
                    try:    umur_val = '35-69 Bulan' if float(umur_val) > 34 else '1-34 Bulan'
                    except: umur_val = '1-34 Bulan'

                gizi    = str(row.get('status_gizi', '')).strip()
                gizi    = gizi_fix.get(gizi.lower(), gizi)

                asi_raw = str(row.get('status_asi', 'Tidak')).strip()
                asi     = asi_fix.get(asi_raw.lower(), asi_raw)

                bbu_raw = str(row.get('status_bbu', 'Normal')).strip()
                bbu     = bbu_fix.get(bbu_raw.lower(), bbu_raw)

                tbu_raw = str(row.get('status_tbu', 'Normal')).strip()
                tbu     = tbu_fix.get(tbu_raw.lower(), tbu_raw)

                stunt_raw = str(row.get('status_stunting', '')).strip()
                stunting  = None if stunt_raw in ('nan', '', 'None') else \
                            stunting_fix.get(stunt_raw.lower(), stunt_raw)

                objs.append(Balita(
                    kode_balita     = kode,
                    jenis_kelamin   = str(row.get('jenis_kelamin', 'L')).strip()[:1].upper(),
                    umur            = umur_val,
                    berat_badan     = float(row.get('berat_badan', 0) or 0),
                    tinggi_badan    = float(row.get('tinggi_badan', 0) or 0),
                    status_bbu      = bbu,
                    status_tbu      = tbu,
                    status_gizi     = gizi,
                    status_asi      = asi,
                    status_stunting = stunting,
                    dataset_type    = dtype,
                    created_by      = request.user,
                ))
            except Exception:
                skipped += 1

        # Bulk create sekaligus
        Balita.objects.bulk_create(objs, batch_size=200, ignore_conflicts=False)
        created = len(objs)

        label = 'Training' if dtype == 'training' else 'Testing'
        msg = f'Upload {label} selesai: {created} data berhasil ditambahkan'
        if deleted:
            msg += f' ({deleted} data lama dihapus)'
        if skipped:
            msg += f', {skipped} baris dilewati'
        messages.success(request, msg + '.')

    except Exception as e:
        messages.error(request, f'Gagal membaca file: {str(e)}')

    return redirect('kelola_dataset')


@login_required
def hapus_dataset_view(request):
    """Hapus semua data training ATAU testing."""
    if not is_admin(request.user):
        messages.error(request, 'Hanya admin yang dapat menghapus dataset.')
        return redirect('balita_list')

    if request.method != 'POST':
        return redirect('kelola_dataset')

    dtype = request.POST.get('dataset_type', '')
    if dtype not in ('training', 'testing', 'semua'):
        messages.error(request, 'Jenis dataset tidak valid.')
        return redirect('kelola_dataset')

    if dtype == 'semua':
        count, _ = Balita.objects.all().delete()
        messages.success(request, f'Semua data balita ({count} data) berhasil dihapus.')
    else:
        count, _ = Balita.objects.filter(dataset_type=dtype).delete()
        label = 'Training' if dtype == 'training' else 'Testing'
        messages.success(request, f'Data {label} ({count} data) berhasil dihapus.')

    # Jika data training habis, reset status model ke untrained
    if dtype in ('training', 'semua'):
        from apps.ml_engine.models import ModelNaiveBayes
        from django.utils import timezone
        ModelNaiveBayes.objects.filter(status='trained').update(
            status='untrained', jumlah_training=0
        )

    return redirect('kelola_dataset')


@login_required
def upload_split_view(request):
    """Upload 1 file Excel lalu auto-split 80% training / 20% testing."""
    if not is_admin(request.user):
        messages.error(request, 'Hanya admin yang dapat mengunggah dataset.')
        return redirect('kelola_dataset')

    if request.method != 'POST':
        return redirect('kelola_dataset')

    file        = request.FILES.get('file')
    replace_all = request.POST.get('replace_all') == '1'
    split_ratio = float(request.POST.get('split_ratio', '0.8'))  # default 80%
    shuffle     = request.POST.get('shuffle', '1') == '1'

    if not file:
        messages.error(request, 'Pilih file Excel terlebih dahulu.')
        return redirect('kelola_dataset')
    if not file.name.endswith(('.xlsx', '.xls')):
        messages.error(request, 'File harus berformat Excel (.xlsx / .xls).')
        return redirect('kelola_dataset')

    try:
        import random as _random

        df = pd.read_excel(file, sheet_name=0)
        df.columns = [str(c).strip() for c in df.columns]

        # Buang baris header duplikat (kalau ada)
        df = df[df.iloc[:,0].notna()]
        df = df[~df.iloc[:,0].astype(str).str.lower().str.contains('nama|kode|unnamed', na=False)]
        df = df.reset_index(drop=True)

        col_map = {
            'Nama/Kode balita':'kode_balita', 'JK':'jenis_kelamin',
            'Umur ':'umur', 'Umur':'umur',
            'Berat ':'berat_badan', 'Berat':'berat_badan',
            'Tinggi':'tinggi_badan', 'BB/U':'status_bbu', 'TB/U':'status_tbu',
            'Status Gizi':'status_gizi',
            'Status ASI Ekslusif':'status_asi', 'Status ASI Eksklusif':'status_asi',
            'Status Stunting':'status_stunting',
        }
        df = df.rename(columns=col_map)

        required = ['kode_balita','jenis_kelamin','umur','berat_badan',
                    'tinggi_badan','status_bbu','status_tbu','status_gizi','status_asi']
        missing = [c for c in required if c not in df.columns]
        if missing:
            messages.error(request, f'Kolom tidak ditemukan: {", ".join(missing)}')
            return redirect('kelola_dataset')

        # Bersihkan data
        gizi_fix = {
            'gizi baik':'Gizi Baik','baik':'Gizi Baik','gizi buruk':'Gizi Buruk',
            'gizi kurang':'Gizi Kurang','gizi lebih':'Gizi Lebih',
            'risiko gizi lebih':'Risiko Gizi Lebih',
        }

        clean_rows = []
        for _, row in df.iterrows():
            kode = str(row.get('kode_balita','')).strip()
            if not kode or kode == 'nan':
                continue
            try:
                umur_val = str(row.get('umur','')).strip()
                if umur_val not in ['1-34 Bulan','35-69 Bulan']:
                    try:    umur_val = '35-69 Bulan' if float(umur_val)>34 else '1-34 Bulan'
                    except: umur_val = '1-34 Bulan'

                gizi = str(row.get('status_gizi','')).strip()
                gizi = gizi_fix.get(gizi.lower(), gizi)

                stunting = str(row.get('status_stunting','')).strip()
                stunting = None if stunting in ('nan','','None') else stunting

                clean_rows.append({
                    'kode_balita':    kode,
                    'jenis_kelamin':  str(row.get('jenis_kelamin','L')).strip()[:1],
                    'umur':           umur_val,
                    'berat_badan':    float(row.get('berat_badan',0) or 0),
                    'tinggi_badan':   float(row.get('tinggi_badan',0) or 0),
                    'status_bbu':     str(row.get('status_bbu','Normal')).strip(),
                    'status_tbu':     str(row.get('status_tbu','Normal')).strip(),
                    'status_gizi':    gizi,
                    'status_asi':     str(row.get('status_asi','Tidak')).strip(),
                    'status_stunting':stunting,
                })
            except Exception:
                continue

        if not clean_rows:
            messages.error(request, 'Tidak ada data valid yang bisa diproses.')
            return redirect('kelola_dataset')

        # Shuffle sebelum split
        if shuffle:
            _random.seed(42)
            _random.shuffle(clean_rows)

        # Split
        total      = len(clean_rows)
        n_train    = round(total * split_ratio)
        n_test     = total - n_train
        train_rows = clean_rows[:n_train]
        test_rows  = clean_rows[n_train:]

        # Hapus data lama jika replace
        deleted_train = deleted_test = 0
        if replace_all:
            deleted_train, _ = Balita.objects.filter(dataset_type='training').delete()
            deleted_test,  _ = Balita.objects.filter(dataset_type='testing').delete()

        # Simpan ke DB
        def _save_rows(rows, dtype):
            count = 0
            for d in rows:
                try:
                    obj, created = Balita.objects.get_or_create(
                        kode_balita=d['kode_balita'],
                        defaults={**d, 'dataset_type':dtype, 'created_by':request.user}
                    )
                    if not created:
                        for k,v in d.items():
                            setattr(obj, k, v)
                        obj.dataset_type = dtype
                        obj.created_by   = request.user
                        obj.save()
                    count += 1
                except Exception:
                    pass
            return count

        saved_train = _save_rows(train_rows, 'training')
        saved_test  = _save_rows(test_rows,  'testing')

        pct_train = round(saved_train / total * 100, 1) if total else 0
        pct_test  = round(saved_test  / total * 100, 1) if total else 0

        messages.success(request,
            f'Auto-split berhasil! Total {total} data → '
            f'Training: {saved_train} ({pct_train}%), '
            f'Testing: {saved_test} ({pct_test}%).')

    except Exception as e:
        messages.error(request, f'Gagal memproses file: {str(e)}')

    return redirect('kelola_dataset')
