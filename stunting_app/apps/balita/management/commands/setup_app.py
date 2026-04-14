"""
python manage.py setup_app
Otomatis: migrate → buat user → import Data_Training_Final.xlsx & Data_Test_Final.xlsx
"""
from django.core.management.base import BaseCommand
from django.core.management import call_command
from django.contrib.auth.models import User
import os, sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', '..', '..', '..', 'data')


class Command(BaseCommand):
    help = 'Setup otomatis: migrate, user default, import data training & testing'

    def handle(self, *args, **options):
        self.stdout.write(self.style.MIGRATE_HEADING('\n=== Setup StuntingApp ===\n'))

        # ── 1. Migrate ──────────────────────────────────────
        self.stdout.write('1. Migrasi database...')
        call_command('migrate', '--run-syncdb', verbosity=0)
        self.stdout.write(self.style.SUCCESS('   ✓ Selesai'))

        # Import model SETELAH migrate
        from apps.accounts.models import UserProfile
        from apps.balita.models import Balita

        # ── 2. User default ─────────────────────────────────
        self.stdout.write('\n2. Membuat user default...')
        if not User.objects.filter(username='admin').exists():
            admin = User.objects.create_superuser('admin', 'admin@stuntingapp.com', 'admin123')
            admin.first_name = 'Super'; admin.last_name = 'Admin'; admin.save()
            profile, _ = UserProfile.objects.get_or_create(user=admin)
            profile.role = 'admin'
            profile.save()
            self.stdout.write(self.style.SUCCESS('   ✓ Admin: admin / admin123'))
        else:
            admin = User.objects.get(username='admin')
            profile, _ = UserProfile.objects.get_or_create(user=admin)
            profile.role = 'admin'
            profile.save()
            self.stdout.write('   • Admin sudah ada (role dipastikan: admin)')

        if not User.objects.filter(username='user').exists():
            demo = User.objects.create_user('user', 'user@stuntingapp.com', 'user123')
            demo.first_name = 'Demo'; demo.last_name = 'User'; demo.save()
            profile, _ = UserProfile.objects.get_or_create(user=demo)
            profile.role = 'pengunjung'
            profile.save()
            self.stdout.write(self.style.SUCCESS('   ✓ Pengunjung: user / user123'))
        else:
            self.stdout.write('   • User demo sudah ada')

        # ── 3. Import data ──────────────────────────────────
        self.stdout.write('\n3. Import data balita...')
        admin_user = User.objects.get(username='admin')

        train_file = self._find_file('Data_Training_Final.xlsx')
        test_file  = self._find_file('Data_Test_Final.xlsx')

        if not train_file and not test_file:
            self.stdout.write(self.style.WARNING(
                '   ⚠ File data tidak ditemukan di folder "data/"\n'
                '   Pastikan ada: data/Data_Training_Final.xlsx\n'
                '                data/Data_Test_Final.xlsx'
            ))
        else:
            # Hapus data lama agar bersih
            if Balita.objects.exists():
                self.stdout.write('   Menghapus data lama...')
                Balita.objects.all().delete()

            if train_file:
                n = self._import_training(train_file, Balita, admin_user)
                self.stdout.write(self.style.SUCCESS(f'   ✓ Training: {n} data dari {os.path.basename(train_file)}'))
            if test_file:
                n = self._import_testing(test_file, Balita, admin_user)
                self.stdout.write(self.style.SUCCESS(f'   ✓ Testing : {n} data dari {os.path.basename(test_file)}'))

            self.stdout.write(self.style.SUCCESS(
                f'\n   Total data balita: {Balita.objects.count()}'
                f' (Training: {Balita.objects.filter(dataset_type="training").count()}'
                f', Testing: {Balita.objects.filter(dataset_type="testing").count()})'
            ))

        # ── Done ────────────────────────────────────────────
        self.stdout.write(self.style.SUCCESS('\n' + '─' * 45))
        self.stdout.write(self.style.SUCCESS('  Setup selesai!'))
        self.stdout.write(self.style.MIGRATE_HEADING('  python manage.py runserver'))
        self.stdout.write('  Buka: http://127.0.0.1:8000')
        self.stdout.write(self.style.SUCCESS('─' * 45 + '\n'))

    def _find_file(self, filename):
        """Cari file di beberapa lokasi."""
        candidates = [
            os.path.join(DATA_DIR, filename),
            os.path.join(os.getcwd(), 'data', filename),
            os.path.join(os.getcwd(), filename),
        ]
        for p in candidates:
            if os.path.exists(p):
                return p
        return None

    # ── Normalizer helpers ───────────────────────────────────
    @staticmethod
    def _jk(v):
        return {'L':'L','L ':'L',' L':'L','P':'P','P ':'P',' P':'P'}.get(str(v).strip(), str(v).strip()[:1])

    @staticmethod
    def _umur(v):
        v = str(v).strip()
        if v in ['1-34 Bulan','35-69 Bulan']: return v
        try: return '35-69 Bulan' if float(v) > 34 else '1-34 Bulan'
        except: return '35-69 Bulan'

    @staticmethod
    def _bbu(v):
        m = {'berat badan normal':'Berat Badan Normal','normal':'Normal','kurang':'Kurang',
             'sangat kurang':'Sangat Kurang','risiko lebih':'Risiko Lebih'}
        return m.get(str(v).strip().lower(), str(v).strip())

    @staticmethod
    def _tbu(v):
        m = {'normal':'Normal','pendek':'Pendek','sangat pendek':'Sangat Pendek','tinggi':'Tinggi'}
        return m.get(str(v).strip().lower(), str(v).strip())

    @staticmethod
    def _gizi(v):
        m = {'gizi baik':'Gizi Baik','baik':'Gizi Baik','gizi buruk':'Gizi Buruk',
             'gizi kurang':'Gizi Kurang','gizi lebih':'Gizi Lebih',
             'risiko gizi lebih':'Risiko Gizi Lebih','obesitas':'Obesitas'}
        return m.get(str(v).strip().lower(), str(v).strip())

    @staticmethod
    def _asi(v):
        return 'Ya' if str(v).strip().lower() in ['ya','y'] else 'Tidak'

    @staticmethod
    def _stunting(v):
        v = str(v).strip()
        if v in ['nan','','None']: return None
        m = {'tidak':'Tidak','stunting':'Stunting','potensi stunting':'Potensi Stunting'}
        return m.get(v.lower(), v)

    def _import_training(self, path, Balita, user):
        """Import Data_Training_Final.xlsx — kolom sudah bersih di row 0."""
        import pandas as pd
        df = pd.read_excel(path)
        df.columns = [str(c).strip() for c in df.columns]

        # Rename kolom agar konsisten
        rename = {
            'Nama/Kode balita': 'kode', 'JK': 'jk',
            'Umur ': 'umur', 'Umur': 'umur',
            'Berat ': 'berat', 'Berat': 'berat',
            'Tinggi': 'tinggi',
            'BB/U': 'bbu', 'TB/U': 'tbu',
            'Status Gizi': 'gizi',
            'Status ASI Ekslusif': 'asi', 'Status ASI Eksklusif': 'asi',
            'Status Stunting': 'stunting',
        }
        df = df.rename(columns=rename)
        count = 0
        for _, r in df.iterrows():
            kode = str(r.get('kode', '')).strip()
            if not kode or kode == 'nan': continue
            try:
                Balita.objects.create(
                    kode_balita=kode,
                    jenis_kelamin=self._jk(r.get('jk','L')),
                    umur=self._umur(r.get('umur','1-34 Bulan')),
                    berat_badan=float(r.get('berat', 0) or 0),
                    tinggi_badan=float(r.get('tinggi', 0) or 0),
                    status_bbu=self._bbu(r.get('bbu','Normal')),
                    status_tbu=self._tbu(r.get('tbu','Normal')),
                    status_gizi=self._gizi(r.get('gizi','Gizi Baik')),
                    status_asi=self._asi(r.get('asi','Tidak')),
                    status_stunting=self._stunting(r.get('stunting','')),
                    dataset_type='training',
                    created_by=user,
                )
                count += 1
            except Exception as e:
                continue
        return count

    def _import_testing(self, path, Balita, user):
        """Import Data_Test_Final.xlsx — header ada di row 0 sebagai Unnamed."""
        import pandas as pd
        df_raw = pd.read_excel(path, header=0)
        # Ambil 10 kolom pertama saja (kolom data asli)
        df = df_raw.iloc[:, :10].copy()
        df.columns = ['kode','jk','umur','berat','tinggi','bbu','tbu','gizi','asi','stunting']
        # Buang baris yg kode-nya adalah header ulang
        df = df[df['kode'].notna()]
        df = df[~df['kode'].astype(str).str.lower().str.contains('nama|kode|unnamed', na=False)]

        count = 0
        for _, r in df.iterrows():
            kode = str(r.get('kode', '')).strip()
            if not kode or kode == 'nan': continue
            try:
                Balita.objects.create(
                    kode_balita=kode,
                    jenis_kelamin=self._jk(r.get('jk','L')),
                    umur=self._umur(r.get('umur','1-34 Bulan')),
                    berat_badan=float(r.get('berat', 0) or 0),
                    tinggi_badan=float(r.get('tinggi', 0) or 0),
                    status_bbu=self._bbu(r.get('bbu','Normal')),
                    status_tbu=self._tbu(r.get('tbu','Normal')),
                    status_gizi=self._gizi(r.get('gizi','Gizi Baik')),
                    status_asi=self._asi(r.get('asi','Tidak')),
                    status_stunting=self._stunting(r.get('stunting','')),
                    dataset_type='testing',
                    created_by=user,
                )
                count += 1
            except Exception as e:
                continue
        return count
