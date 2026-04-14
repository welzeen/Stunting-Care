"""
pdf_report.py
Generate laporan PDF evaluasi model Naive Bayes:
  - Halaman 1 : Ringkasan model & metrik
  - Halaman 2+ : Tabel perbandingan data aktual vs prediksi (data testing)
  - Halaman terakhir : Confusion matrix + Akurasi / Presisi / Recall / F1
"""

from io import BytesIO
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether
)
from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.graphics import renderPDF


# ── Palet warna (teal theme) ──────────────────────────────────────────────────
C_TEAL      = colors.HexColor('#0d9488')
C_TEAL_D    = colors.HexColor('#0f766e')
C_TEAL_L    = colors.HexColor('#ccfbf1')
C_DARK      = colors.HexColor('#1c2b27')
C_MUTED     = colors.HexColor('#6b8278')
C_BORDER    = colors.HexColor('#e4eae7')
C_BG        = colors.HexColor('#f4f7f5')
C_WHITE     = colors.white

C_RED       = colors.HexColor('#fef2f2')
C_RED_TXT   = colors.HexColor('#b91c1c')
C_AMBER     = colors.HexColor('#fffbeb')
C_AMBER_TXT = colors.HexColor('#b45309')
C_GREEN     = colors.HexColor('#f0fdf4')
C_GREEN_TXT = colors.HexColor('#15803d')

# ── Styles ────────────────────────────────────────────────────────────────────
def _style(name, **kw):
    base = dict(fontName='Helvetica', fontSize=10, leading=14,
                textColor=C_DARK)
    base.update(kw)
    return ParagraphStyle(name, **base)

ST_TITLE    = _style('title',   fontName='Helvetica-Bold', fontSize=18,
                     textColor=C_WHITE, alignment=TA_CENTER, leading=22)
ST_SUBTITLE = _style('sub',     fontName='Helvetica',      fontSize=10,
                     textColor=colors.HexColor('#a7f3d0'), alignment=TA_CENTER)
ST_H1       = _style('h1',      fontName='Helvetica-Bold', fontSize=13,
                     textColor=C_TEAL_D, spaceAfter=4)
ST_H2       = _style('h2',      fontName='Helvetica-Bold', fontSize=11,
                     textColor=C_DARK, spaceAfter=2)
ST_BODY     = _style('body',    fontSize=9,  textColor=C_DARK, leading=13)
ST_SMALL    = _style('small',   fontSize=8,  textColor=C_MUTED, leading=11)
ST_CENTER   = _style('center',  fontSize=9,  alignment=TA_CENTER, textColor=C_DARK)
ST_BOLD_CTR = _style('boldc',   fontName='Helvetica-Bold', fontSize=9,
                     alignment=TA_CENTER, textColor=C_DARK)


# ── Helper: header/footer callback ───────────────────────────────────────────
class _DocTemplate(SimpleDocTemplate):
    """Custom doc template that adds a footer on every page."""

    def __init__(self, buf, trained_at=None, **kw):
        super().__init__(buf, **kw)
        self._trained_at = trained_at

    def handle_pageBegin(self):
        super().handle_pageBegin()

    def afterPage(self):
        canvas = self.canv
        w, h = A4
        canvas.saveState()
        canvas.setFont('Helvetica', 7)
        canvas.setFillColor(C_MUTED)
        canvas.drawString(2 * cm, 1.2 * cm,
                          f'StuntingApp — Laporan Evaluasi Model Naive Bayes')
        canvas.drawRightString(w - 2 * cm, 1.2 * cm,
                               f'Halaman {self.page}  |  Dicetak: '
                               f'{datetime.now().strftime("%d/%m/%Y %H:%M")}')
        # thin line
        canvas.setStrokeColor(C_BORDER)
        canvas.setLineWidth(0.5)
        canvas.line(2 * cm, 1.5 * cm, w - 2 * cm, 1.5 * cm)
        canvas.restoreState()


# ── Helper: cover header block ────────────────────────────────────────────────
def _cover_header(model_obj):
    """Return a teal header table (acts as banner)."""
    w = A4[0] - 4 * cm  # matches doc left+right margin = 2cm each

    title_p    = Paragraph('Laporan Evaluasi Model', ST_TITLE)
    subtitle_p = Paragraph('Deteksi Stunting Balita — Naive Bayes Classifier', ST_SUBTITLE)

    tbl = Table([[title_p], [subtitle_p]], colWidths=[w])
    tbl.setStyle(TableStyle([
        ('BACKGROUND',  (0, 0), (-1, -1), C_TEAL),
        ('TOPPADDING',  (0, 0), (-1, -1), 18),
        ('BOTTOMPADDING',(0,-1), (-1, -1), 18),
        ('LEFTPADDING', (0, 0), (-1, -1), 20),
        ('RIGHTPADDING',(0, 0), (-1, -1), 20),
        ('ROUNDEDCORNERS', [8]),
    ]))
    return tbl


# ── Helper: metric card row ───────────────────────────────────────────────────
def _metric_cards(model_obj):
    metrics = [
        ('Akurasi',   f"{model_obj.akurasi * 100:.2f}%",  C_TEAL_L,  C_TEAL_D),
        ('Presisi',   f"{model_obj.presisi * 100:.2f}%",  C_GREEN,   C_GREEN_TXT),
        ('Recall',    f"{model_obj.recall * 100:.2f}%",   C_AMBER,   C_AMBER_TXT),
        ('F1-Score',  f"{model_obj.f1_score * 100:.2f}%", colors.HexColor('#ede9fe'),
                                                           colors.HexColor('#6d28d9')),
    ]
    col_w = (A4[0] - 4 * cm) / 4 - 0.15 * cm

    cells = []
    for label, value, bg, fg in metrics:
        cell = Table(
            [[Paragraph(f'<font size="18"><b>{value}</b></font>',
                        ParagraphStyle('mv', fontName='Helvetica-Bold',
                                       fontSize=18, textColor=fg,
                                       alignment=TA_CENTER, leading=22))],
             [Paragraph(label,
                        ParagraphStyle('ml', fontName='Helvetica',
                                       fontSize=8, textColor=fg,
                                       alignment=TA_CENTER))]],
            colWidths=[col_w]
        )
        cell.setStyle(TableStyle([
            ('BACKGROUND',   (0, 0), (-1, -1), bg),
            ('TOPPADDING',   (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING',(0, 0), (-1, -1), 12),
            ('ROUNDEDCORNERS', [6]),
        ]))
        cells.append(cell)

    row = Table([cells], colWidths=[col_w] * 4,
                hAlign='LEFT')
    row.setStyle(TableStyle([
        ('LEFTPADDING',  (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING',   (0, 0), (-1, -1), 0),
        ('BOTTOMPADDING',(0, 0), (-1, -1), 0),
    ]))
    return row


# ── Helper: info summary table ────────────────────────────────────────────────
def _info_table(model_obj):
    trained_at = model_obj.trained_at.strftime('%d %B %Y, %H:%M') \
        if model_obj.trained_at else '-'
    trained_by = model_obj.trained_by.get_full_name() \
        if model_obj.trained_by and model_obj.trained_by.get_full_name() \
        else (model_obj.trained_by.username if model_obj.trained_by else '-')

    rows = [
        ['Nama Model',        model_obj.nama_model],
        ['Algoritma',         'Naive Bayes (CategoricalNB)'],
        ['Jumlah Data Training', f"{model_obj.jumlah_training} data"],
        ['Jumlah Data Testing',  f"{model_obj.jumlah_testing} data"],
        ['Dilatih Oleh',      trained_by],
        ['Tanggal Latih',     trained_at],
    ]

    col_w = (A4[0] - 4 * cm)
    data = [[Paragraph(f'<b>{r[0]}</b>', ST_BODY),
             Paragraph(str(r[1]), ST_BODY)] for r in rows]

    tbl = Table(data, colWidths=[col_w * 0.38, col_w * 0.62])
    tbl.setStyle(TableStyle([
        ('BACKGROUND',   (0, 0), (-1, -1), C_WHITE),
        ('BACKGROUND',   (0, 0), (-1, -1), C_BG),
        ('ROWBACKGROUNDS', (0, 0), (-1, -1), [C_WHITE, C_BG]),
        ('GRID',         (0, 0), (-1, -1), 0.5, C_BORDER),
        ('TOPPADDING',   (0, 0), (-1, -1), 7),
        ('BOTTOMPADDING',(0, 0), (-1, -1), 7),
        ('LEFTPADDING',  (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
    ]))
    return tbl


# ── Helper: comparison table ──────────────────────────────────────────────────
def _stunting_badge_style(val):
    if val == 'Stunting':
        return C_RED, C_RED_TXT
    if val == 'Potensi Stunting':
        return C_AMBER, C_AMBER_TXT
    return C_GREEN, C_GREEN_TXT


def _comparison_table(testing_qs, predicted_map):
    """
    testing_qs    : queryset Balita dataset_type='testing'
    predicted_map : dict { balita.kode_balita : predicted_label }
    """
    col_w_total = A4[0] - 4 * cm
    COL_W = [
        col_w_total * 0.13,  # No
        col_w_total * 0.22,  # Kode Balita
        col_w_total * 0.05,  # JK
        col_w_total * 0.14,  # Umur
        col_w_total * 0.23,  # Aktual
        col_w_total * 0.23,  # Prediksi
    ]

    header = [
        Paragraph('<b>No</b>',          ST_BOLD_CTR),
        Paragraph('<b>Kode Balita</b>', ST_BOLD_CTR),
        Paragraph('<b>JK</b>',          ST_BOLD_CTR),
        Paragraph('<b>Umur</b>',        ST_BOLD_CTR),
        Paragraph('<b>Aktual</b>',      ST_BOLD_CTR),
        Paragraph('<b>Prediksi</b>',    ST_BOLD_CTR),
    ]

    rows = [header]
    style_cmds = [
        ('BACKGROUND',   (0, 0), (-1, 0), C_TEAL),
        ('TEXTCOLOR',    (0, 0), (-1, 0), C_WHITE),
        ('FONTNAME',     (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE',     (0, 0), (-1, 0), 9),
        ('TOPPADDING',   (0, 0), (-1, 0), 8),
        ('BOTTOMPADDING',(0, 0), (-1, 0), 8),
        ('ALIGN',        (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN',       (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID',         (0, 0), (-1, -1), 0.4, C_BORDER),
        ('FONTSIZE',     (0, 1), (-1, -1), 8),
        ('TOPPADDING',   (0, 1), (-1, -1), 5),
        ('BOTTOMPADDING',(0, 1), (-1, -1), 5),
    ]

    correct = 0
    total   = 0

    for i, balita in enumerate(testing_qs, start=1):
        aktual    = balita.status_stunting or '-'
        prediksi  = predicted_map.get(balita.kode_balita, '-')
        is_match  = (aktual == prediksi) and aktual != '-'

        if aktual != '-' and prediksi != '-':
            total += 1
            if is_match:
                correct += 1

        # Row bg: putih / abu alternating
        row_bg = C_WHITE if i % 2 == 0 else C_BG

        row = [
            Paragraph(str(i), ST_CENTER),
            Paragraph(balita.kode_balita, ST_CENTER),
            Paragraph(balita.jenis_kelamin, ST_CENTER),
            Paragraph(balita.umur, ST_CENTER),
            Paragraph(aktual, ST_CENTER),
            Paragraph(prediksi, ST_CENTER),
        ]
        rows.append(row)

        ri = len(rows) - 1
        style_cmds.append(('BACKGROUND', (0, ri), (-1, ri), row_bg))

        # Warna kolom Aktual
        if aktual != '-':
            bg_a, fg_a = _stunting_badge_style(aktual)
            style_cmds.append(('BACKGROUND', (4, ri), (4, ri), bg_a))
            style_cmds.append(('TEXTCOLOR',  (4, ri), (4, ri), fg_a))
            style_cmds.append(('FONTNAME',   (4, ri), (4, ri), 'Helvetica-Bold'))

        # Warna kolom Prediksi
        if prediksi != '-':
            bg_p, fg_p = _stunting_badge_style(prediksi)
            style_cmds.append(('BACKGROUND', (5, ri), (5, ri), bg_p))
            style_cmds.append(('TEXTCOLOR',  (5, ri), (5, ri), fg_p))
            style_cmds.append(('FONTNAME',   (5, ri), (5, ri), 'Helvetica-Bold'))

        # Highlight baris SALAH dengan garis kiri merah
        if aktual != '-' and prediksi != '-' and not is_match:
            style_cmds.append(('LINEAFTER',  (5, ri), (5, ri), 2.5, C_RED_TXT))

    tbl = Table(rows, colWidths=COL_W, repeatRows=1)
    tbl.setStyle(TableStyle(style_cmds))
    return tbl, correct, total


# ── Helper: confusion matrix ──────────────────────────────────────────────────
def _confusion_matrix_table(cm_data, kelas_list):
    """
    cm_data    : dict dari model_obj.get_confusion_matrix()
                 format: { aktual: { prediksi: count } }
    kelas_list : ['Tidak', 'Potensi Stunting', 'Stunting']
    """
    # Bangun matrix dari cm_data
    n = len(kelas_list)
    matrix = [[0] * n for _ in range(n)]

    for i, aktual in enumerate(kelas_list):
        for j, prediksi in enumerate(kelas_list):
            try:
                matrix[i][j] = int(cm_data.get(aktual, {}).get(prediksi, 0))
            except Exception:
                matrix[i][j] = 0

    col_w_total = A4[0] - 4 * cm
    label_w = col_w_total * 0.28
    cell_w  = (col_w_total - label_w) / n

    SHORT = {'Tidak': 'Tidak', 'Potensi Stunting': 'Potensi\nStunting', 'Stunting': 'Stunting'}

    # Header row
    header_row = [Paragraph('', ST_BOLD_CTR)]
    for k in kelas_list:
        header_row.append(Paragraph(f'<b>{SHORT[k]}</b>', ST_BOLD_CTR))

    data = [header_row]
    style_cmds = [
        ('BACKGROUND',   (0, 0), (-1, 0), C_TEAL),
        ('TEXTCOLOR',    (0, 0), (-1, 0), C_WHITE),
        ('FONTNAME',     (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('ALIGN',        (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN',       (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID',         (0, 0), (-1, -1), 0.5, C_BORDER),
        ('TOPPADDING',   (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING',(0, 0), (-1, -1), 8),
        ('FONTSIZE',     (0, 0), (-1, -1), 9),
    ]

    for i, aktual in enumerate(kelas_list):
        row = [Paragraph(f'<b>{SHORT[aktual]}</b>',
                         ParagraphStyle('lbl', fontName='Helvetica-Bold',
                                        fontSize=9, textColor=C_TEAL_D,
                                        alignment=TA_LEFT))]
        for j in range(n):
            val = matrix[i][j]
            if i == j:
                # TP — hijau
                p = Paragraph(f'<b>{val}</b>',
                              ParagraphStyle('tp', fontName='Helvetica-Bold',
                                             fontSize=11, textColor=C_GREEN_TXT,
                                             alignment=TA_CENTER))
                row.append(p)
                ri = i + 1
                style_cmds.append(('BACKGROUND', (j+1, ri), (j+1, ri), C_GREEN))
            else:
                # FP/FN
                color = C_RED_TXT if val > 0 else C_MUTED
                p = Paragraph(str(val),
                              ParagraphStyle('fp', fontName='Helvetica',
                                             fontSize=10, textColor=color,
                                             alignment=TA_CENTER))
                row.append(p)
                ri = i + 1
                if val > 0:
                    style_cmds.append(('BACKGROUND', (j+1, ri), (j+1, ri), C_RED))
                else:
                    style_cmds.append(('BACKGROUND', (j+1, ri), (j+1, ri), C_BG))

        # Row label background
        ri = i + 1
        style_cmds.append(('BACKGROUND', (0, ri), (0, ri), C_TEAL_L))
        data.append(row)

    tbl = Table(data, colWidths=[label_w] + [cell_w] * n)
    tbl.setStyle(TableStyle(style_cmds))
    return tbl


# ── Helper: per-class metrics table ──────────────────────────────────────────
def _per_class_table(cr_data, kelas_list):
    col_w_total = A4[0] - 4 * cm
    COL_W = [
        col_w_total * 0.30,
        col_w_total * 0.175,
        col_w_total * 0.175,
        col_w_total * 0.175,
        col_w_total * 0.175,
    ]

    header = [
        Paragraph('<b>Kelas</b>',     ST_BOLD_CTR),
        Paragraph('<b>Presisi</b>',   ST_BOLD_CTR),
        Paragraph('<b>Recall</b>',    ST_BOLD_CTR),
        Paragraph('<b>F1-Score</b>',  ST_BOLD_CTR),
        Paragraph('<b>Support</b>',   ST_BOLD_CTR),
    ]

    rows  = [header]
    style = [
        ('BACKGROUND',   (0, 0), (-1, 0), C_TEAL),
        ('TEXTCOLOR',    (0, 0), (-1, 0), C_WHITE),
        ('ALIGN',        (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN',       (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID',         (0, 0), (-1, -1), 0.4, C_BORDER),
        ('TOPPADDING',   (0, 0), (-1, -1), 7),
        ('BOTTOMPADDING',(0, 0), (-1, -1), 7),
        ('FONTSIZE',     (0, 0), (-1, -1), 9),
    ]

    for i, kls in enumerate(kelas_list):
        d = cr_data.get(kls, {})
        p   = float(d.get('precision', 0))
        r   = float(d.get('recall', 0))
        f1  = float(d.get('f1-score', 0))
        sup = int(d.get('support', 0))

        bg = C_WHITE if i % 2 == 0 else C_BG

        def _pct(v):
            color = '#15803d' if v >= 0.7 else ('#b45309' if v >= 0.5 else '#b91c1c')
            return Paragraph(f'<font color="{color}"><b>{v*100:.2f}%</b></font>',
                             ST_CENTER)

        rows.append([
            Paragraph(kls, ST_BODY),
            _pct(p), _pct(r), _pct(f1),
            Paragraph(str(sup), ST_CENTER),
        ])
        ri = i + 1
        style.append(('BACKGROUND', (0, ri), (-1, ri), bg))

    # Weighted avg
    wa = cr_data.get('weighted avg', {})
    rows.append([
        Paragraph('<b>Weighted Avg</b>', ST_BODY),
        Paragraph(f'<b>{float(wa.get("precision",0))*100:.2f}%</b>', ST_BOLD_CTR),
        Paragraph(f'<b>{float(wa.get("recall",0))*100:.2f}%</b>',    ST_BOLD_CTR),
        Paragraph(f'<b>{float(wa.get("f1-score",0))*100:.2f}%</b>',  ST_BOLD_CTR),
        Paragraph(f'<b>{int(wa.get("support",0))}</b>',               ST_BOLD_CTR),
    ])
    ri = len(rows) - 1
    style.append(('BACKGROUND', (0, ri), (-1, ri), C_TEAL_L))
    style.append(('TOPLINE',    (0, ri), (-1, ri), 1.5, C_TEAL))

    tbl = Table(rows, colWidths=COL_W)
    tbl.setStyle(TableStyle(style))
    return tbl


# ── MAIN PUBLIC FUNCTION ──────────────────────────────────────────────────────
def generate_evaluation_pdf(model_obj, testing_qs, predicted_map):
    """
    Returns a BytesIO containing the complete PDF.

    Args:
        model_obj     : ModelNaiveBayes instance
        testing_qs    : queryset Balita (dataset_type='testing')
        predicted_map : dict { kode_balita: predicted_label }
    """
    buf = BytesIO()
    doc = _DocTemplate(
        buf,
        trained_at=model_obj.trained_at,
        pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm,  bottomMargin=2.2 * cm,
    )

    story = []
    kelas_list = ['Tidak', 'Potensi Stunting', 'Stunting']

    # ── HALAMAN 1: Cover + Info + Metrik ────────────────────────────────────
    story.append(_cover_header(model_obj))
    story.append(Spacer(1, 0.5 * cm))

    story.append(Paragraph('Informasi Model', ST_H1))
    story.append(HRFlowable(width='100%', thickness=1, color=C_TEAL_L,
                             spaceAfter=6))
    story.append(_info_table(model_obj))
    story.append(Spacer(1, 0.5 * cm))

    story.append(Paragraph('Ringkasan Performa', ST_H1))
    story.append(HRFlowable(width='100%', thickness=1, color=C_TEAL_L,
                             spaceAfter=8))
    story.append(_metric_cards(model_obj))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(
        'Metrik dihitung menggunakan <b>weighted average</b> berdasarkan jumlah sampel per kelas '
        'pada data testing.',
        ST_SMALL))

    story.append(PageBreak())

    # ── HALAMAN 2+: Tabel Perbandingan ──────────────────────────────────────
    story.append(Paragraph('Perbandingan Data Aktual vs Prediksi', ST_H1))
    story.append(HRFlowable(width='100%', thickness=1, color=C_TEAL_L,
                             spaceAfter=6))
    story.append(Paragraph(
        'Tabel berikut menampilkan seluruh data testing beserta label aktual dan hasil '
        'prediksi model. Baris dengan prediksi <font color="#b91c1c"><b>tidak sesuai</b></font> '
        'ditandai dengan border merah pada kolom Prediksi.',
        ST_SMALL))
    story.append(Spacer(1, 0.3 * cm))

    comp_tbl, correct, total = _comparison_table(testing_qs, predicted_map)
    story.append(comp_tbl)
    story.append(Spacer(1, 0.3 * cm))

    total_testing = len(list(testing_qs))
    acc_text = f"{correct}/{total} data benar ({correct/total*100:.2f}%)" if total else "—"
    story.append(Paragraph(
        f'Total: <b>{total_testing}</b> data testing  |  '
        f'Prediksi benar: <b>{acc_text}</b>',
        ST_SMALL))

    story.append(PageBreak())

    # ── HALAMAN TERAKHIR: Confusion Matrix + Metrik Per Kelas ───────────────
    story.append(Paragraph('Confusion Matrix', ST_H1))
    story.append(HRFlowable(width='100%', thickness=1, color=C_TEAL_L,
                             spaceAfter=6))
    story.append(Paragraph(
        'Baris = <b>kelas aktual</b>, Kolom = <b>kelas prediksi</b>.  '
        '<font color="#15803d"><b>Hijau</b></font> = prediksi benar (True Positive).  '
        '<font color="#b91c1c"><b>Merah</b></font> = prediksi salah.',
        ST_SMALL))
    story.append(Spacer(1, 0.3 * cm))

    cm_data = model_obj.get_confusion_matrix()
    if cm_data:
        story.append(KeepTogether([_confusion_matrix_table(cm_data, kelas_list)]))
    else:
        story.append(Paragraph('Data confusion matrix tidak tersedia.', ST_BODY))

    story.append(Spacer(1, 0.7 * cm))

    story.append(Paragraph('Metrik Evaluasi per Kelas', ST_H1))
    story.append(HRFlowable(width='100%', thickness=1, color=C_TEAL_L,
                             spaceAfter=6))

    cr_data = model_obj.get_classification_report()
    if cr_data:
        story.append(_per_class_table(cr_data, kelas_list))
    else:
        story.append(Paragraph('Data classification report tidak tersedia.', ST_BODY))

    story.append(Spacer(1, 0.5 * cm))

    # Kotak ringkasan akhir
    summary_rows = [
        [Paragraph('<b>Metrik</b>',  ST_BOLD_CTR), Paragraph('<b>Nilai</b>', ST_BOLD_CTR)],
        [Paragraph('Akurasi',  ST_BODY), Paragraph(f'{model_obj.akurasi*100:.4f}%',  ST_CENTER)],
        [Paragraph('Presisi',  ST_BODY), Paragraph(f'{model_obj.presisi*100:.4f}%',  ST_CENTER)],
        [Paragraph('Recall',   ST_BODY), Paragraph(f'{model_obj.recall*100:.4f}%',   ST_CENTER)],
        [Paragraph('F1-Score', ST_BODY), Paragraph(f'{model_obj.f1_score*100:.4f}%', ST_CENTER)],
    ]
    col_w = (A4[0] - 4 * cm)
    summary_tbl = Table(summary_rows, colWidths=[col_w * 0.5, col_w * 0.5])
    summary_tbl.setStyle(TableStyle([
        ('BACKGROUND',   (0, 0), (-1, 0), C_TEAL),
        ('TEXTCOLOR',    (0, 0), (-1, 0), C_WHITE),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [C_WHITE, C_BG]),
        ('GRID',         (0, 0), (-1, -1), 0.5, C_BORDER),
        ('ALIGN',        (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN',       (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING',   (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING',(0, 0), (-1, -1), 8),
        ('FONTSIZE',     (0, 0), (-1, -1), 10),
    ]))

    half = (A4[0] - 4 * cm) / 2
    center_wrap = Table([[summary_tbl]], colWidths=[A4[0] - 4 * cm], hAlign='CENTER')
    story.append(KeepTogether([center_wrap]))

    # Build
    doc.build(story)
    buf.seek(0)
    return buf
