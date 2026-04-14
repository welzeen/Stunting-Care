from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse
from apps.balita.models import Balita, PrediksiHasil
from .models import ModelNaiveBayes
from .forms import PrediksiForm
from . import naive_bayes_service as nbs
import json


@login_required
def ml_dashboard_view(request):
    model_obj = ModelNaiveBayes.objects.order_by('-trained_at').first()

    training_count = Balita.objects.filter(dataset_type='training').exclude(status_stunting__isnull=True).count()
    testing_count = Balita.objects.filter(dataset_type='testing').exclude(status_stunting__isnull=True).count()

    # Jika tidak ada data training sama sekali, reset status model ke untrained
    if model_obj and model_obj.status == 'trained' and training_count == 0:
        try:
            from django.utils import timezone
            model_obj.status = 'untrained'
            model_obj.trained_at = timezone.now()  # tetap isi agar tidak NULL error
            model_obj.save(update_fields=['status', 'trained_at'])
        except Exception:
            pass  # abaikan error jika kolom bermasalah, tampilkan saja halaman

    context = {
        'model_obj': model_obj,
        'training_count': training_count,
        'testing_count': testing_count,
    }
    return render(request, 'ml/dashboard.html', context)


@login_required
def train_model_view(request):
    if not (hasattr(request.user, 'profile') and request.user.profile.is_admin()):
        messages.error(request, 'Hanya admin yang dapat melatih model.')
        return redirect('ml_dashboard')

    if request.method == 'POST':
        try:
            training_qs = Balita.objects.filter(
                dataset_type='training'
            ).exclude(status_stunting__isnull=True).exclude(status_stunting='')

            if training_qs.count() < 10:
                messages.error(request, 'Data training terlalu sedikit (minimal 10 data).')
                return redirect('ml_dashboard')

            model, encoders, X_train, y_train = nbs.train_model(training_qs)

            testing_qs = Balita.objects.filter(
                dataset_type='testing'
            ).exclude(status_stunting__isnull=True).exclude(status_stunting='')

            metrics = None
            if testing_qs.count() > 0:
                metrics = nbs.evaluate_model(model, encoders, testing_qs)

            model_obj, _ = ModelNaiveBayes.objects.get_or_create(id=1)
            model_obj.nama_model = 'Naive Bayes - Klasifikasi Stunting Balita'
            model_obj.jumlah_training = training_qs.count()
            model_obj.status = 'trained'
            model_obj.trained_by = request.user

            from django.utils import timezone
            model_obj.trained_at = timezone.now()  # set manual saat benar-benar dilatih

            if metrics:
                model_obj.akurasi = metrics['akurasi']
                model_obj.presisi = metrics['presisi']
                model_obj.recall = metrics['recall']
                model_obj.f1_score = metrics['f1_score']
                model_obj.jumlah_testing = metrics['jumlah_testing']
                model_obj.confusion_matrix_data = json.dumps(metrics['confusion_matrix'])
                model_obj.classification_report = json.dumps(metrics['classification_report'])

            model_obj.save()
            messages.success(request,
                f'Model berhasil dilatih.')

        except Exception as e:
            messages.error(request, f'Gagal melatih model: {str(e)}')

    return redirect('ml_dashboard')


@login_required
def prediksi_view(request):
    form = PrediksiForm(request.POST or None)
    result = None

    if request.method == 'POST' and form.is_valid():
        input_data = form.cleaned_data
        nama = input_data.get('nama_balita', '').strip()
        try:
            predicted_class, proba_dict = nbs.predict_single(input_data)
            result = {
                'kelas': predicted_class,
                'probabilitas': proba_dict,
                'input': input_data,
                'nama_balita': nama,
            }
            # Save to history
            PrediksiHasil.objects.create(
                nama_balita=nama,
                jenis_kelamin=input_data['jenis_kelamin'],
                umur=input_data['umur'],
                berat_badan=0,
                tinggi_badan=0,
                status_bbu=input_data['status_bbu'],
                status_tbu=input_data['status_tbu'],
                status_gizi=input_data['status_gizi'],
                status_asi=input_data['status_asi'],
                hasil_prediksi=predicted_class,
                probabilitas_tidak=proba_dict.get('Tidak', 0),
                probabilitas_potensi=proba_dict.get('Potensi Stunting', 0),
                probabilitas_stunting=proba_dict.get('Stunting', 0),
                predicted_by=request.user,
            )
        except ValueError as e:
            messages.error(request, str(e))

    return render(request, 'ml/prediksi.html', {'form': form, 'result': result})


@login_required  
def batch_predict_view(request):
    """Predict all testing data that have no label prediction yet"""
    if not (hasattr(request.user, 'profile') and request.user.profile.is_admin()):
        messages.error(request, 'Hanya admin yang dapat menjalankan batch prediksi.')
        return redirect('ml_dashboard')

    if request.method == 'POST':
        try:
            testing_qs = Balita.objects.filter(dataset_type='testing')
            count = 0
            for balita in testing_qs:
                input_data = {
                    'jenis_kelamin': balita.jenis_kelamin,
                    'umur': balita.umur,
                    'status_bbu': balita.status_bbu,
                    'status_tbu': balita.status_tbu,
                    'status_gizi': balita.status_gizi,
                    'status_asi': balita.status_asi,
                }
                try:
                    predicted_class, proba_dict = nbs.predict_single(input_data)
                    PrediksiHasil.objects.create(
                        balita=balita,
                        jenis_kelamin=balita.jenis_kelamin,
                        umur=balita.umur,
                        berat_badan=balita.berat_badan,
                        tinggi_badan=balita.tinggi_badan,
                        status_bbu=balita.status_bbu,
                        status_tbu=balita.status_tbu,
                        status_gizi=balita.status_gizi,
                        status_asi=balita.status_asi,
                        hasil_prediksi=predicted_class,
                        probabilitas_tidak=proba_dict.get('Tidak', 0),
                        probabilitas_potensi=proba_dict.get('Potensi Stunting', 0),
                        probabilitas_stunting=proba_dict.get('Stunting', 0),
                        predicted_by=request.user,
                    )
                    count += 1
                except Exception:
                    pass
            messages.success(request, f'Batch prediksi selesai: {count} data diproses.')
        except Exception as e:
            messages.error(request, f'Error: {str(e)}')

    return redirect('ml_dashboard')


@login_required
def evaluasi_data_view(request):
    """Tampilkan hasil prediksi vs aktual untuk semua data testing."""
    from apps.balita.models import Balita
    from . import naive_bayes_service as nbs

    model, encoders = nbs.load_model()
    if model is None:
        messages.error(request, 'Model belum dilatih. Silakan latih model terlebih dahulu.')
        return redirect('ml_dashboard')

    testing_qs = Balita.objects.filter(
        dataset_type='testing'
    ).exclude(status_stunting__isnull=True).exclude(status_stunting='').order_by('kode_balita')

    if testing_qs.count() == 0:
        messages.error(request, 'Tidak ada data testing.')
        return redirect('ml_dashboard')

    # Prediksi semua data testing
    hasil_list = []
    benar = 0
    for b in testing_qs:
        input_data = {
            'jenis_kelamin': b.jenis_kelamin,
            'umur':          b.umur,
            'status_bbu':    b.status_bbu,
            'status_tbu':    b.status_tbu,
            'status_gizi':   b.status_gizi,
            'status_asi':    b.status_asi,
        }
        try:
            pred, proba = nbs.predict_single(input_data)
            cocok = (pred == b.status_stunting)
            if cocok:
                benar += 1
            hasil_list.append({
                'balita':    b,
                'prediksi':  pred,
                'aktual':    b.status_stunting,
                'cocok':     cocok,
                'prob_tidak':   proba.get('Tidak', 0),
                'prob_potensi': proba.get('Potensi Stunting', 0),
                'prob_stunting':proba.get('Stunting', 0),
            })
        except Exception:
            continue

    total   = len(hasil_list)
    salah   = total - benar
    akurasi = (benar / total * 100) if total > 0 else 0

    # Filter GET
    filter_status = request.GET.get('filter', 'semua')
    if filter_status == 'benar':
        hasil_tampil = [h for h in hasil_list if h['cocok']]
    elif filter_status == 'salah':
        hasil_tampil = [h for h in hasil_list if not h['cocok']]
    else:
        hasil_tampil = hasil_list

    # Pagination
    from django.core.paginator import Paginator
    paginator = Paginator(hasil_tampil, 20)
    page_number = request.GET.get('page', 1)
    page_obj = paginator.get_page(page_number)

    context = {
        'page_obj':      page_obj,
        'total':         total,
        'benar':         benar,
        'salah':         salah,
        'akurasi':       akurasi,
        'filter_status': filter_status,
    }
    return render(request, 'ml/evaluasi_data.html', context)


@login_required
def korelasi_view(request):
    """Halaman analisis korelasi antar variabel menggunakan Cramér's V."""
    from apps.balita.models import Balita
    from . import korelasi_service as ks
    import json

    dataset_type = request.GET.get('dataset_type', 'all')
    qs = Balita.objects.exclude(status_stunting__isnull=True).exclude(status_stunting='')
    if dataset_type == 'training':
        qs = qs.filter(dataset_type='training')
    elif dataset_type == 'testing':
        qs = qs.filter(dataset_type='testing')

    result = ks.hitung_korelasi(qs)

    vars_list = ks.ALL_VARS
    labels = [ks.FITUR_LABEL[v] for v in vars_list]
    matrix_flat = []
    for i, v1 in enumerate(vars_list):
        for j, v2 in enumerate(vars_list):
            matrix_flat.append({
                'x': j, 'y': i,
                'v': result['matrix'][v1][v2],
                'label': f"{ks.FITUR_LABEL[v1]} vs {ks.FITUR_LABEL[v2]}"
            })

    bar_labels = [k['label'] for k in result['korelasi_target']]
    bar_values = [k['persen'] for k in result['korelasi_target']]

    context = {
        'result': result,
        'dataset_type': dataset_type,
        'vars_list': vars_list,
        'labels_json': json.dumps(labels),
        'matrix_json': json.dumps(matrix_flat),
        'bar_labels_json': json.dumps(bar_labels),
        'bar_values_json': json.dumps(bar_values),
    }
    return render(request, 'ml/korelasi.html', context)
