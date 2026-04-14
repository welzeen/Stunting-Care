"""
korelasi_service.py — Menghitung korelasi antar variabel kategorik
menggunakan Cramér's V (cocok untuk data kategorik, bukan numerik).
"""
import numpy as np
from itertools import combinations


FITUR_LABEL = {
    'jenis_kelamin': 'Jenis Kelamin',
    'umur':          'Umur',
    'status_bbu':    'BB/U',
    'status_tbu':    'TB/U',
    'status_gizi':   'Status Gizi',
    'status_asi':    'ASI Eksklusif',
    'status_stunting': 'Status Stunting',
}

ALL_VARS = list(FITUR_LABEL.keys())


def cramers_v(x, y):
    """
    Menghitung Cramér's V antara dua variabel kategorik.
    Nilai 0 = tidak ada asosiasi, 1 = asosiasi sempurna.
    """
    from collections import Counter

    # Buat contingency table manual
    categories_x = sorted(set(x))
    categories_y = sorted(set(y))

    n = len(x)
    if n == 0:
        return 0.0

    # Hitung frekuensi
    table = np.zeros((len(categories_x), len(categories_y)))
    xy_map = {v: i for i, v in enumerate(categories_x)}
    yy_map = {v: i for i, v in enumerate(categories_y)}

    for xi, yi in zip(x, y):
        if xi in xy_map and yi in yy_map:
            table[xy_map[xi]][yy_map[yi]] += 1

    # Chi-square
    row_sums = table.sum(axis=1)
    col_sums = table.sum(axis=0)
    total = table.sum()

    if total == 0:
        return 0.0

    expected = np.outer(row_sums, col_sums) / total
    with np.errstate(divide='ignore', invalid='ignore'):
        chi2 = np.where(expected > 0, (table - expected) ** 2 / expected, 0).sum()

    r, k = table.shape
    phi2 = chi2 / total
    phi2_corr = max(0, phi2 - ((k - 1) * (r - 1)) / (total - 1))
    r_corr = r - ((r - 1) ** 2) / (total - 1)
    k_corr = k - ((k - 1) ** 2) / (total - 1)

    denom = min(r_corr - 1, k_corr - 1)
    if denom <= 0:
        return 0.0

    return float(np.sqrt(phi2_corr / denom))


def hitung_korelasi(queryset):
    """
    Hitung matriks korelasi Cramér's V untuk semua variabel.
    Returns:
        matrix : dict of dict, matrix[var1][var2] = nilai (0–1)
        distribusi : dict of dict, distribusi[var] = {label: count}
        korelasi_target : list of (var, nilai) diurutkan dari tinggi ke rendah
        total_data : int
    """
    # Ambil semua nilai dari queryset
    data = {var: [] for var in ALL_VARS}
    total_data = 0

    for b in queryset:
        skip = False
        row = {}
        for var in ALL_VARS:
            val = getattr(b, var, None)
            if val is None or str(val).strip() == '':
                skip = True
                break
            row[var] = str(val).strip()
        if skip:
            continue
        for var in ALL_VARS:
            data[var].append(row[var])
        total_data += 1

    # Hitung matriks korelasi
    matrix = {v1: {v2: 0.0 for v2 in ALL_VARS} for v1 in ALL_VARS}
    for v in ALL_VARS:
        matrix[v][v] = 1.0  # diagonal selalu 1

    for v1, v2 in combinations(ALL_VARS, 2):
        val = cramers_v(data[v1], data[v2])
        matrix[v1][v2] = round(val, 4)
        matrix[v2][v1] = round(val, 4)

    # Distribusi per variabel
    from collections import Counter
    distribusi = {}
    for var in ALL_VARS:
        counts = Counter(data[var])
        distribusi[var] = dict(counts)

    # Korelasi terhadap target (status_stunting), diurutkan
    korelasi_target = []
    for var in ALL_VARS:
        if var == 'status_stunting':
            continue
        korelasi_target.append({
            'var': var,
            'label': FITUR_LABEL[var],
            'nilai': round(matrix[var]['status_stunting'], 4),
            'persen': round(matrix[var]['status_stunting'] * 100, 1),
            'kekuatan': _kekuatan(matrix[var]['status_stunting']),
        })

    korelasi_target.sort(key=lambda x: x['nilai'], reverse=True)

    return {
        'matrix': matrix,
        'distribusi': distribusi,
        'korelasi_target': korelasi_target,
        'total_data': total_data,
        'variabel_labels': FITUR_LABEL,
        'all_vars': ALL_VARS,
    }


def _kekuatan(v):
    if v >= 0.5:   return ('Sangat Kuat',  'danger')
    if v >= 0.3:   return ('Kuat',         'warning')
    if v >= 0.1:   return ('Sedang',       'info')
    return ('Lemah', 'secondary')
