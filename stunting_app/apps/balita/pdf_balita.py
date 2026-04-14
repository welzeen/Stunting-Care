"""
pdf_balita.py — Generate laporan PDF per individu balita
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
    HRFlowable, KeepTogether
)

# ── Warna ────────────────────────────────────────────────────────────────────
C_TEAL    = colors.HexColor('#0d9488')
C_TEAL_D  = colors.HexColor('#0f766e')
C_TEAL_L  = colors.HexColor('#ccfbf1')
C_DARK    = colors.HexColor('#1c2b27')
C_MUTED   = colors.HexColor('#6b8278')
C_BORDER  = colors.HexColor('#e4eae7')
C_BG      = colors.HexColor('#f4f7f5')
C_WHITE   = colors.white
C_RED     = colors.HexColor('#fef2f2')
C_RED_T   = colors.HexColor('#b91c1c')
C_AMBER   = colors.HexColor('#fffbeb')
C_AMBER_T = colors.HexColor('#b45309')
C_GREEN   = colors.HexColor('#f0fdf4')
C_GREEN_T = colors.HexColor('#15803d')

def _s(name, **kw):
    base = dict(fontName='Helvetica', fontSize=10, leading=14, textColor=C_DARK)
    base.update(kw)
    return ParagraphStyle(name, **base)

ST_TITLE  = _s('t',  fontName='Helvetica-Bold', fontSize=16, textColor=C_WHITE,
                alignment=TA_CENTER, leading=20)
ST_SUB    = _s('su', fontSize=9,  textColor=colors.HexColor('#a7f3d0'), alignment=TA_CENTER)
ST_H1     = _s('h1', fontName='Helvetica-Bold', fontSize=12, textColor=C_TEAL_D, spaceAfter=4)
ST_BODY   = _s('b',  fontSize=9,  textColor=C_DARK, leading=13)
ST_MUTED  = _s('m',  fontSize=8,  textColor=C_MUTED, leading=11)
ST_CENTER = _s('c',  fontSize=9,  alignment=TA_CENTER, textColor=C_DARK)
ST_BOLD_C = _s('bc', fontName='Helvetica-Bold', fontSize=9, alignment=TA_CENTER, textColor=C_DARK)


class _Doc(SimpleDocTemplate):
    def afterPage(self):
        c = self.canv
        w, _ = A4
        c.saveState()
        c.setFont('Helvetica', 7)
        c.setFillColor(C_MUTED)
        c.drawString(2*cm, 1.2*cm, 'StuntingApp — Laporan Data Balita')
        c.drawRightString(w-2*cm, 1.2*cm,
            f'Halaman {self.page}  |  Dicetak: {datetime.now().strftime("%d/%m/%Y %H:%M")}')
        c.setStrokeColor(C_BORDER)
        c.setLineWidth(0.5)
        c.line(2*cm, 1.5*cm, w-2*cm, 1.5*cm)
        c.restoreState()


def _stunting_colors(val):
    if val == 'Stunting':        return C_RED,   C_RED_T
    if val == 'Potensi Stunting': return C_AMBER, C_AMBER_T
    return C_GREEN, C_GREEN_T


def _header_banner(balita):
    w = A4[0] - 4*cm
    nama = balita.nama_balita or balita.kode_balita
    tbl = Table([
        [Paragraph(f'Laporan Data Balita', ST_TITLE)],
        [Paragraph(f'{nama}  ·  {balita.kode_balita}', ST_SUB)],
    ], colWidths=[w])
    tbl.setStyle(TableStyle([
        ('BACKGROUND',   (0,0),(-1,-1), C_TEAL),
        ('TOPPADDING',   (0,0),(-1,-1), 16),
        ('BOTTOMPADDING',(0,-1),(-1,-1), 16),
        ('LEFTPADDING',  (0,0),(-1,-1), 16),
        ('RIGHTPADDING', (0,0),(-1,-1), 16),
        ('ROUNDEDCORNERS',[8]),
    ]))
    return tbl


def _status_badge(val, w):
    if not val:
        return Table([[Paragraph('Belum Ada Label', ST_CENTER)]], colWidths=[w])
    bg, fg = _stunting_colors(val)
    p = Paragraph(f'<b>{val}</b>',
        ParagraphStyle('sb', fontName='Helvetica-Bold', fontSize=14,
                       textColor=fg, alignment=TA_CENTER, leading=18))
    t = Table([[p]], colWidths=[w])
    t.setStyle(TableStyle([
        ('BACKGROUND',   (0,0),(-1,-1), bg),
        ('TOPPADDING',   (0,0),(-1,-1), 14),
        ('BOTTOMPADDING',(0,0),(-1,-1), 14),
        ('ROUNDEDCORNERS',[8]),
    ]))
    return t


def _info_row(label, value, bg=C_WHITE):
    return [Paragraph(f'<b>{label}</b>', ST_BODY), Paragraph(str(value), ST_BODY)]


def generate_balita_pdf(balita, prediksi_list=None):
    """
    balita       : Balita model instance
    prediksi_list: queryset/list PrediksiHasil (opsional)
    Returns BytesIO PDF
    """
    buf = BytesIO()
    doc = _Doc(buf, pagesize=A4,
               leftMargin=2*cm, rightMargin=2*cm,
               topMargin=2*cm,  bottomMargin=2.2*cm)

    story = []
    W = A4[0] - 4*cm

    # ── BANNER ────────────────────────────────────────────────────────────────
    story.append(_header_banner(balita))
    story.append(Spacer(1, 0.4*cm))

    # ── STATUS STUNTING (besar di atas) ───────────────────────────────────────
    story.append(Paragraph('Status Stunting', ST_H1))
    story.append(HRFlowable(width='100%', thickness=1, color=C_TEAL_L, spaceAfter=6))
    story.append(_status_badge(balita.status_stunting, W))
    story.append(Spacer(1, 0.5*cm))

    # ── DATA IDENTITAS + KESEHATAN (2 kolom) ──────────────────────────────────
    col = W * 0.38

    id_data = [
        _info_row('Kode Balita',   balita.kode_balita),
        _info_row('Nama',          balita.nama_balita or '—'),
        _info_row('Jenis Kelamin', 'Laki-laki' if balita.jenis_kelamin=='L' else 'Perempuan'),
        _info_row('Kelompok Umur', balita.umur),
        _info_row('Berat Badan',   f'{balita.berat_badan} kg'),
        _info_row('Tinggi Badan',  f'{balita.tinggi_badan} cm'),
    ]
    kes_data = [
        _info_row('BB/U',          balita.status_bbu),
        _info_row('TB/U',          balita.status_tbu),
        _info_row('Status Gizi',   balita.status_gizi),
        _info_row('ASI Eksklusif', balita.status_asi),
        _info_row('Jenis Dataset', balita.dataset_type.capitalize()),
        _info_row('Tanggal Input', balita.created_at.strftime('%d/%m/%Y') if balita.created_at else '—'),
    ]

    def _make_tbl(title, rows):
        header = [Paragraph(f'<b>{title}</b>',
                  ParagraphStyle('th', fontName='Helvetica-Bold', fontSize=9,
                                 textColor=C_WHITE, alignment=TA_LEFT))]
        tbl_data = [rows[i] for i in range(len(rows))]
        t = Table([[header[0], '']] + tbl_data, colWidths=[col, W/2 - col])
        cmds = [
            ('BACKGROUND',  (0,0),(-1,0), C_TEAL),
            ('SPAN',        (0,0),(-1,0)),
            ('TOPPADDING',  (0,0),(-1,-1), 6),
            ('BOTTOMPADDING',(0,0),(-1,-1), 6),
            ('LEFTPADDING', (0,0),(-1,-1), 8),
            ('RIGHTPADDING',(0,0),(-1,-1), 8),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),[C_WHITE, C_BG]),
            ('GRID',        (0,0),(-1,-1), 0.4, C_BORDER),
            ('FONTSIZE',    (0,0),(-1,-1), 9),
        ]
        t.setStyle(TableStyle(cmds))
        return t

    left_tbl  = _make_tbl('Data Identitas', id_data)
    right_tbl = _make_tbl('Data Kesehatan', kes_data)

    two_col = Table([[left_tbl, Spacer(0.3*cm, 1), right_tbl]],
                    colWidths=[W/2 - 0.15*cm, 0.3*cm, W/2 - 0.15*cm])
    two_col.setStyle(TableStyle([
        ('VALIGN', (0,0),(-1,-1), 'TOP'),
        ('LEFTPADDING',  (0,0),(-1,-1), 0),
        ('RIGHTPADDING', (0,0),(-1,-1), 0),
        ('TOPPADDING',   (0,0),(-1,-1), 0),
        ('BOTTOMPADDING',(0,0),(-1,-1), 0),
    ]))
    story.append(two_col)
    story.append(Spacer(1, 0.5*cm))

    # ── RIWAYAT PREDIKSI ──────────────────────────────────────────────────────
    if prediksi_list:
        story.append(Paragraph('Riwayat Prediksi', ST_H1))
        story.append(HRFlowable(width='100%', thickness=1, color=C_TEAL_L, spaceAfter=6))

        header_row = [
            Paragraph('<b>No</b>',           ST_BOLD_C),
            Paragraph('<b>Hasil Prediksi</b>',ST_BOLD_C),
            Paragraph('<b>Prob. Tidak</b>',   ST_BOLD_C),
            Paragraph('<b>Prob. Potensi</b>', ST_BOLD_C),
            Paragraph('<b>Prob. Stunting</b>',ST_BOLD_C),
            Paragraph('<b>Tanggal</b>',       ST_BOLD_C),
            Paragraph('<b>Oleh</b>',          ST_BOLD_C),
        ]
        CW = [W*0.05, W*0.20, W*0.13, W*0.13, W*0.13, W*0.21, W*0.15]
        rows = [header_row]
        scmds = [
            ('BACKGROUND',   (0,0),(-1,0), C_TEAL),
            ('TEXTCOLOR',    (0,0),(-1,0), C_WHITE),
            ('ALIGN',        (0,0),(-1,-1), 'CENTER'),
            ('VALIGN',       (0,0),(-1,-1), 'MIDDLE'),
            ('GRID',         (0,0),(-1,-1), 0.4, C_BORDER),
            ('FONTSIZE',     (0,0),(-1,-1), 8),
            ('TOPPADDING',   (0,0),(-1,-1), 5),
            ('BOTTOMPADDING',(0,0),(-1,-1), 5),
        ]
        for i, p in enumerate(prediksi_list, 1):
            bg_p, fg_p = _stunting_colors(p.hasil_prediksi)
            hasil_p = Paragraph(f'<b>{p.hasil_prediksi}</b>',
                ParagraphStyle('hp', fontName='Helvetica-Bold', fontSize=8,
                               textColor=fg_p, alignment=TA_CENTER))
            rows.append([
                Paragraph(str(i), ST_CENTER),
                hasil_p,
                Paragraph(f'{p.probabilitas_tidak:.4f}',   ST_CENTER),
                Paragraph(f'{p.probabilitas_potensi:.4f}', ST_CENTER),
                Paragraph(f'{p.probabilitas_stunting:.4f}',ST_CENTER),
                Paragraph(p.predicted_at.strftime('%d/%m/%Y %H:%M'), ST_CENTER),
                Paragraph(p.predicted_by.username if p.predicted_by else '—', ST_CENTER),
            ])
            ri = i
            row_bg = C_WHITE if i % 2 == 0 else C_BG
            scmds.append(('BACKGROUND', (0,ri),(-1,ri), row_bg))
            scmds.append(('BACKGROUND', (1,ri),(1,ri),  bg_p))

        pred_tbl = Table(rows, colWidths=CW)
        pred_tbl.setStyle(TableStyle(scmds))
        story.append(pred_tbl)
        story.append(Spacer(1, 0.3*cm))

    # ── CATATAN BAWAH ─────────────────────────────────────────────────────────
    story.append(HRFlowable(width='100%', thickness=0.5, color=C_BORDER, spaceAfter=6))
    story.append(Paragraph(
        f'Laporan dicetak pada {datetime.now().strftime("%d %B %Y, %H:%M")} WIB  '
        f'oleh sistem StuntingApp.',
        ST_MUTED))

    doc.build(story)
    buf.seek(0)
    return buf


def generate_all_balita_pdf(balita_queryset, filters=None):
    """
    Generate PDF laporan semua data balita.
    balita_queryset : QuerySet Balita
    filters         : dict info filter aktif (opsional, untuk ditampilkan di header)
    Returns BytesIO PDF
    """
    buf = BytesIO()
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable

    page_w, page_h = landscape(A4)
    margin = 1.5 * cm

    class _AllDoc(SimpleDocTemplate):
        def afterPage(self):
            c = self.canv
            c.saveState()
            c.setFont('Helvetica', 7)
            c.setFillColor(C_MUTED)
            c.drawString(margin, 1.2 * cm, 'StuntingApp — Laporan Data Balita')
            c.drawRightString(page_w - margin, 1.2 * cm,
                f'Halaman {self.page}  |  Dicetak: {datetime.now().strftime("%d/%m/%Y %H:%M")}')
            c.setStrokeColor(C_BORDER)
            c.setLineWidth(0.5)
            c.line(margin, 1.5 * cm, page_w - margin, 1.5 * cm)
            c.restoreState()

    doc = _AllDoc(buf, pagesize=landscape(A4),
                  leftMargin=margin, rightMargin=margin,
                  topMargin=1.5 * cm, bottomMargin=2 * cm)

    story = []
    W = page_w - 2 * margin

    # ── BANNER ────────────────────────────────────────────────────────────────
    banner_tbl = Table([
        [Paragraph('Laporan Data Balita', ST_TITLE)],
        [Paragraph(f'Dicetak: {datetime.now().strftime("%d %B %Y, %H:%M")} WIB  |  Total: {balita_queryset.count()} data', ST_SUB)],
    ], colWidths=[W])
    banner_tbl.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1, -1), C_TEAL),
        ('TOPPADDING',    (0, 0), (-1, -1), 12),
        ('BOTTOMPADDING', (0, -1), (-1, -1), 12),
        ('LEFTPADDING',   (0, 0), (-1, -1), 16),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 16),
        ('ROUNDEDCORNERS', [6]),
    ]))
    story.append(banner_tbl)
    story.append(Spacer(1, 0.3 * cm))

    # ── Info filter ────────────────────────────────────────────────────────────
    if filters:
        filter_parts = [f'{k}: {v}' for k, v in filters.items() if v]
        if filter_parts:
            story.append(Paragraph('Filter aktif: ' + '  |  '.join(filter_parts),
                _s('fi', fontSize=8, textColor=C_MUTED)))
            story.append(Spacer(1, 0.2 * cm))

    # ── TABEL DATA ─────────────────────────────────────────────────────────────
    ST_TH = ParagraphStyle('th_all', fontName='Helvetica-Bold', fontSize=7,
                           textColor=C_WHITE, alignment=1)
    ST_TD = ParagraphStyle('td_all', fontName='Helvetica', fontSize=7,
                           textColor=C_DARK, alignment=1, leading=10)
    ST_TD_L = ParagraphStyle('td_all_l', fontName='Helvetica', fontSize=7,
                             textColor=C_DARK, alignment=0, leading=10)

    def th(text): return Paragraph(f'<b>{text}</b>', ST_TH)
    def td(text, left=False): return Paragraph(str(text) if text else '—', ST_TD_L if left else ST_TD)

    header_row = [
        th('No'), th('Kode Balita'), th('JK'), th('Umur'),
        th('Berat (kg)'), th('Tinggi (cm)'), th('BB/U'), th('TB/U'),
        th('Status Gizi'), th('ASI'), th('Status Stunting'), th('Dataset'),
    ]

    col_w = [
        W * 0.04,   # No
        W * 0.11,   # Kode
        W * 0.04,   # JK
        W * 0.10,   # Umur
        W * 0.07,   # Berat
        W * 0.07,   # Tinggi
        W * 0.09,   # BB/U
        W * 0.08,   # TB/U
        W * 0.13,   # Status Gizi
        W * 0.05,   # ASI
        W * 0.12,   # Stunting
        W * 0.10,   # Dataset
    ]

    rows = [header_row]
    tbl_cmds = [
        ('BACKGROUND',    (0, 0), (-1, 0), C_TEAL),
        ('TEXTCOLOR',     (0, 0), (-1, 0), C_WHITE),
        ('ALIGN',         (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN',        (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTSIZE',      (0, 0), (-1, -1), 7),
        ('GRID',          (0, 0), (-1, -1), 0.3, C_BORDER),
        ('TOPPADDING',    (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]

    for i, b in enumerate(balita_queryset, 1):
        jk_label = 'L' if b.jenis_kelamin == 'L' else 'P'
        stunting_val = b.status_stunting or '—'

        row = [
            td(str(i)),
            td(b.kode_balita, left=True),
            td(jk_label),
            td(b.umur),
            td(f'{b.berat_badan:.1f}'),
            td(f'{b.tinggi_badan:.1f}'),
            td(b.status_bbu),
            td(b.status_tbu),
            td(b.status_gizi, left=True),
            td(b.status_asi),
            td(stunting_val),
            td(b.dataset_type.capitalize()),
        ]
        rows.append(row)

        # Baris alternating
        row_bg = C_WHITE if i % 2 == 0 else C_BG
        tbl_cmds.append(('BACKGROUND', (0, i), (-1, i), row_bg))

        # Warna kolom stunting
        if b.status_stunting:
            bg_s, fg_s = _stunting_colors(b.status_stunting)
            tbl_cmds.append(('BACKGROUND', (10, i), (10, i), bg_s))
            tbl_cmds.append(('TEXTCOLOR',  (10, i), (10, i), fg_s))

    data_tbl = Table(rows, colWidths=col_w, repeatRows=1)
    data_tbl.setStyle(TableStyle(tbl_cmds))
    story.append(data_tbl)
    story.append(Spacer(1, 0.3 * cm))

    # ── RINGKASAN ─────────────────────────────────────────────────────────────
    total = balita_queryset.count()
    stunting_c  = sum(1 for b in balita_queryset if b.status_stunting == 'Stunting')
    potensi_c   = sum(1 for b in balita_queryset if b.status_stunting == 'Potensi Stunting')
    tidak_c     = sum(1 for b in balita_queryset if b.status_stunting == 'Tidak')

    story.append(HRFlowable(width='100%', thickness=0.5, color=C_BORDER, spaceAfter=6))
    story.append(Paragraph(
        f'Ringkasan: Total {total} data  |  '
        f'Stunting: {stunting_c}  |  '
        f'Potensi Stunting: {potensi_c}  |  '
        f'Tidak Stunting: {tidak_c}',
        _s('sum', fontSize=8, textColor=C_MUTED)))

    doc.build(story)
    buf.seek(0)
    return buf
