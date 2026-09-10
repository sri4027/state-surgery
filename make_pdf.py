"""Render docs/one-page-summary.md to a genuinely ONE-page A4 PDF.
Auto-tightens typography until it fits, then asserts page count == 1."""
import re, os, sys
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from pypdf import PdfReader

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MD = open(f"{R}/docs/one-page-summary.md").read()
OUT = sys.argv[1] if len(sys.argv) > 1 else "/mnt/user-data/outputs/one-page-summary.pdf"

def inline(t):
    t = t.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    t = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', t)
    t = re.sub(r'(?<!\w)\*(.+?)\*(?!\w)', r'<i>\1</i>', t)
    return re.sub(r'`(.+?)`', r'<font face="Courier">\1</font>', t)

def build(fs, lead, sa, mar, tfs):
    ss = getSampleStyleSheet()
    body = ParagraphStyle('b', parent=ss['BodyText'], fontName='Helvetica',
                          fontSize=fs, leading=lead, spaceAfter=sa, alignment=4)
    h1 = ParagraphStyle('h1', parent=ss['Title'], fontName='Helvetica-Bold',
                        fontSize=fs + 4.2, leading=fs + 5.4, spaceAfter=5,
                        alignment=0, textColor=colors.HexColor('#2b1a66'))
    sm = ParagraphStyle('s', parent=body, fontSize=tfs, leading=tfs + 1.5,
                        spaceAfter=0, textColor=colors.HexColor('#3c3c48'))
    flow = []
    # split on BLANK LINES into blocks, so a wrapped paragraph reflows as one
    # paragraph instead of one Paragraph per source line
    for block in re.split(r'\n\s*\n', MD.strip()):
        lines = [l.strip() for l in block.split('\n') if l.strip()]
        if not lines:
            continue
        rows = [[c.strip() for c in l.strip('|').split('|')]
                for l in lines if l.startswith('|')]
        if rows:
            rows = [r for r in rows if not set(''.join(r)) <= set('-: ')]
            n = len(rows[0]); wid = (210 - 2 * mar - 1) * mm / n
            tb = Table([[Paragraph(inline(c), sm) for c in r] for r in rows],
                       colWidths=[wid] * n)
            tb.setStyle(TableStyle([
                ('GRID', (0, 0), (-1, -1), 0.28, colors.HexColor('#b9b9cc')),
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#ecebf6')),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('LEFTPADDING', (0, 0), (-1, -1), 2), ('RIGHTPADDING', (0, 0), (-1, -1), 2),
                ('TOPPADDING', (0, 0), (-1, -1), 1.3), ('BOTTOMPADDING', (0, 0), (-1, -1), 1.3)]))
            flow += [Spacer(1, 1.5), tb, Spacer(1, 3)]
            continue
        s = ' '.join(lines)
        if s.startswith('# '):
            flow.append(Paragraph(inline(s[2:]), h1))
        elif s.startswith('*Live artifact'):
            flow.append(Paragraph(inline(s.strip('*')), sm))
        else:
            flow.append(Paragraph(inline(s), body))
    SimpleDocTemplate(OUT, pagesize=A4, leftMargin=mar * mm, rightMargin=mar * mm,
                      topMargin=(mar - 2) * mm, bottomMargin=(mar - 3) * mm,
                      title="Skill Acquisition from Demonstrations: adaptation in state, not weights",
                      author="State Surgery — DataForge 2026 Pathway Track").build(flow)
    return len(PdfReader(OUT).pages)

for fs, lead, sa, mar, tfs in [(7.9,10.0,4.0,13,6.9),(7.5,9.3,3.4,12,6.6),
                               (7.2,8.9,3.0,11,6.4),(6.9,8.5,2.6,11,6.1),
                               (6.6,8.1,2.2,10,5.9)]:
    n = build(fs, lead, sa, mar, tfs)
    print(f"  try {fs}pt/{lead} margin {mar}mm -> {n} page(s)")
    if n == 1:
        print(f"FIT at {fs}pt. {round(os.path.getsize(OUT)/1024,1)} KB")
        break
else:
    raise SystemExit("ERROR: still >1 page — trim the markdown")
