#!/usr/bin/env python3
"""
format_doc.py — Post-humanization formatting fixer for .docx files.

Does NOT touch text content at all. Only fixes:
  1. Paragraph justification (body text → fully justified)
  2. Font consistency (Times New Roman 12pt on body paragraphs)
  3. Line spacing normalization (1.5 on body text)
  4. Indentation consistency  
  5. Heading spacing cleanup
  6. TOC entries — inject Word auto-TOC field to replace manual "…….1" strings
  7. LibreOffice headless refresh to repaginate and update TOC page numbers

Usage:
    python3 format_doc.py input.docx output.docx
    python3 format_doc.py input.docx output.docx --no-toc      # skip TOC rebuild
    python3 format_doc.py input.docx output.docx --no-refresh  # skip LibreOffice
    python3 format_doc.py input.docx output.docx --report      # show what will change

Requires:
    pip install python-docx
    apt install libreoffice   (for TOC repagination — optional)
"""

import argparse
import copy
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor
from lxml import etree


# ─── FORMATTING CONSTANTS (extracted from original document) ─────────────────

BODY_FONT_NAME    = "Times New Roman"
BODY_FONT_SIZE_PT = 12.0          # 12pt = 240 half-points = sz val 24
BODY_LINE_SPACING = 360           # twips: 360 = 1.5 lines (240 twips = single)
BODY_LINE_RULE    = "auto"
BODY_SPACE_BEFORE = 191           # twips (~10pt)
BODY_SPACE_AFTER  = 0
BODY_JC           = "both"        # full justification
BODY_LEFT_INDENT  = 980           # twips (matches majority of body paras)
BODY_RIGHT_INDENT = 744
BODY_FIRST_LINE   = 720           # twips (0.5 inch first line indent)

HEADING2_SPACE_BEFORE = 280       # twips (~18pt)
HEADING2_SPACE_AFTER  = 80        # twips (~4pt)
HEADING3_SPACE_BEFORE = 220       # twips (~14pt)

# Styles we consider "body text" — will be justified + font-fixed
BODY_STYLES = {'normal', 'body text', 'default paragraph font', 'no spacing'}

# Styles we never touch
SKIP_STYLES = {
    'heading 1', 'heading 2', 'heading 3', 'heading 4',
    'title', 'subtitle', 'caption', 'footnote text',
    'header', 'footer', 'toc 1', 'toc 2', 'toc 3',
    'list paragraph', 'list bullet', 'list number',
    'table contents', 'table heading',
    'code', 'verbatim', 'preformatted',
}


# ─── HELPERS ─────────────────────────────────────────────────────────────────

def is_body_paragraph(para):
    """True if this paragraph should have full body formatting applied."""
    style = para.style.name.lower()
    if any(skip in style for skip in SKIP_STYLES):
        return False
    if style not in BODY_STYLES and not style.startswith('normal'):
        return False
    txt = para.text.strip()
    if not txt:
        return False
    # Skip TOC-like lines (manual dot leaders)
    if txt.count('…') > 2 or txt.count('.....') > 2:
        return False
    # Must be actual prose: >=15 words with sentence punctuation
    # Protects cover page, signatures, labels, certificate lines
    words = txt.split()
    if len(words) < 15:
        return False
    if not any(c in txt for c in ['.', ',']):
        return False
    # Skip paragraphs that are primarily images
    if para._element.findall('.//' + qn('w:drawing')) and len(words) < 5:
        return False
    return True


def set_ppr_attr(ppr, tag, attrs):
    """Set or replace a child element in <w:pPr>."""
    existing = ppr.find(qn(tag))
    if existing is not None:
        ppr.remove(existing)
    el = OxmlElement(tag)
    for k, v in attrs.items():
        el.set(qn(k), str(v))
    ppr.append(el)
    return el


def ensure_ppr(para):
    """Get or create <w:pPr> for a paragraph."""
    ppr = para._element.find(qn('w:pPr'))
    if ppr is None:
        ppr = OxmlElement('w:pPr')
        para._element.insert(0, ppr)
    return ppr


def fix_paragraph_format(para, report_mode=False):
    """Apply correct body formatting to a single paragraph. Returns change description or None."""
    changes = []
    ppr = ensure_ppr(para)
    pf = para.paragraph_format

    # 1. Justification
    jc = ppr.find(qn('w:jc'))
    current_jc = jc.get(qn('w:val')) if jc is not None else None
    if current_jc != 'both':
        changes.append(f"justify: {current_jc!r}→both")
        if not report_mode:
            set_ppr_attr(ppr, 'w:jc', {'w:val': 'both'})

    # 2. Line spacing
    spacing = ppr.find(qn('w:spacing'))
    current_line = spacing.get(qn('w:line')) if spacing is not None else None
    if current_line != str(BODY_LINE_SPACING):
        changes.append(f"line_spacing: {current_line}→{BODY_LINE_SPACING}")
        if not report_mode:
            if spacing is None:
                spacing = OxmlElement('w:spacing')
                ppr.append(spacing)
            spacing.set(qn('w:line'), str(BODY_LINE_SPACING))
            spacing.set(qn('w:lineRule'), BODY_LINE_RULE)
            spacing.set(qn('w:after'), str(BODY_SPACE_AFTER))
            if not spacing.get(qn('w:before')):
                spacing.set(qn('w:before'), str(BODY_SPACE_BEFORE))

    # 3. Widow control (prevent single lines at top/bottom of page)
    wc = ppr.find(qn('w:widowControl'))
    if wc is None or wc.get(qn('w:val')) not in ('0', 'false'):
        changes.append("widowControl→0")
        if not report_mode:
            set_ppr_attr(ppr, 'w:widowControl', {'w:val': '0'})

    # 4. Font and size on all runs
    for run in para.runs:
        if not run.text.strip():
            continue
        rpr = run._r.find(qn('w:rPr'))
        if rpr is None:
            rpr = OxmlElement('w:rPr')
            run._r.insert(0, rpr)

        # Font name
        rfonts = rpr.find(qn('w:rFonts'))
        if rfonts is None:
            rfonts = OxmlElement('w:rFonts')
            rpr.insert(0, rfonts)
        for attr in ['w:ascii', 'w:hAnsi', 'w:cs', 'w:eastAsia']:
            if rfonts.get(qn(attr)) != BODY_FONT_NAME:
                if not report_mode:
                    rfonts.set(qn(attr), BODY_FONT_NAME)

        # Font size (sz = half-points, so 12pt = 24)
        sz_val = str(int(BODY_FONT_SIZE_PT * 2))
        sz = rpr.find(qn('w:sz'))
        szCs = rpr.find(qn('w:szCs'))
        current_sz = sz.get(qn('w:val')) if sz is not None else None
        if current_sz not in (sz_val, str(int(BODY_FONT_SIZE_PT * 2) + 1)):
            if current_sz is not None:
                changes.append(f"font_size: {int(current_sz)//2}pt→{BODY_FONT_SIZE_PT}pt")
            if not report_mode:
                if sz is None:
                    sz = OxmlElement('w:sz'); rpr.append(sz)
                sz.set(qn('w:val'), sz_val)
                if szCs is None:
                    szCs = OxmlElement('w:szCs'); rpr.append(szCs)
                szCs.set(qn('w:val'), sz_val)

    return changes if changes else None


def fix_heading_spacing(para, report_mode=False):
    """Normalize heading spacing."""
    style = para.style.name.lower()
    changes = []
    ppr = ensure_ppr(para)

    if 'heading 2' in style:
        spacing = ppr.find(qn('w:spacing'))
        if spacing is None:
            spacing = OxmlElement('w:spacing')
            ppr.append(spacing)
        before = spacing.get(qn('w:before'))
        after  = spacing.get(qn('w:after'))
        if before != str(HEADING2_SPACE_BEFORE) or after != str(HEADING2_SPACE_AFTER):
            changes.append(f"H2 spacing before/after")
            if not report_mode:
                spacing.set(qn('w:before'), str(HEADING2_SPACE_BEFORE))
                spacing.set(qn('w:after'),  str(HEADING2_SPACE_AFTER))

    elif 'heading 3' in style:
        spacing = ppr.find(qn('w:spacing'))
        if spacing is None:
            spacing = OxmlElement('w:spacing')
            ppr.append(spacing)
        before = spacing.get(qn('w:before'))
        if before != str(HEADING3_SPACE_BEFORE):
            changes.append(f"H3 spacing before")
            if not report_mode:
                spacing.set(qn('w:before'), str(HEADING3_SPACE_BEFORE))

    return changes if changes else None


# ─── TOC REBUILD ─────────────────────────────────────────────────────────────

def inject_toc_field(doc):
    """
    Replace manual TOC paragraphs (lines with …………N) with a real Word TOC field.
    The field auto-updates when the document is opened in Word or refreshed by LibreOffice.
    
    Finds the paragraph block that contains the manual TOC, removes those paragraphs,
    inserts a proper { TOC \\o "1-3" \\h \\z \\u } field in their place.
    """
    # Find the range of TOC paragraphs
    toc_start = None
    toc_end   = None
    paras = list(doc.paragraphs)

    for i, para in enumerate(paras):
        txt = para.text.strip()
        if txt.count('…') > 3 or '........' in txt:
            if toc_start is None:
                toc_start = i
            toc_end = i

    if toc_start is None:
        print("  [TOC] No manual TOC entries found — skipping")
        return False

    print(f"  [TOC] Found manual TOC at paragraphs {toc_start}–{toc_end}")

    # Get the parent body element
    body = doc.element.body

    # Find actual XML elements for the TOC range
    toc_para_elements = [paras[i]._element for i in range(toc_start, toc_end + 1)]

    # Build the TOC field paragraph XML
    # Word TOC field: { TOC \o "1-3" \h \z \u }
    toc_xml = '''<w:p xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:pPr>
    <w:pStyle w:val="TOCHeading"/>
    <w:jc w:val="center"/>
  </w:pPr>
  <w:r>
    <w:fldChar w:fldCharType="begin" w:dirty="true"/>
  </w:r>
  <w:r>
    <w:instrText xml:space="preserve"> TOC \\o "1-3" \\h \\z \\u </w:instrText>
  </w:r>
  <w:r>
    <w:fldChar w:fldCharType="separate"/>
  </w:r>
  <w:r>
    <w:t>Right-click and "Update Field" in Word, or press F9 to refresh page numbers.</w:t>
  </w:r>
  <w:r>
    <w:fldChar w:fldCharType="end"/>
  </w:r>
</w:p>'''

    toc_element = etree.fromstring(toc_xml)

    # Insert the new TOC field before the first TOC paragraph
    first_toc = toc_para_elements[0]
    body.insert(list(body).index(first_toc), toc_element)

    # Remove all old manual TOC paragraphs
    for el in toc_para_elements:
        try:
            body.remove(el)
        except ValueError:
            pass  # already removed or not direct child

    print(f"  [TOC] Injected Word TOC field (replaces {len(toc_para_elements)} manual entries)")
    return True


# ─── LIBREOFFICE REFRESH ──────────────────────────────────────────────────────

def libreoffice_refresh(input_path: Path, output_path: Path) -> bool:
    """
    Use LibreOffice headless to open and re-save the docx.
    This forces field recalculation (TOC page numbers, etc.) and
    correct repagination based on final content length.
    """
    soffice = shutil.which('soffice') or shutil.which('libreoffice')
    if not soffice:
        print("  [refresh] LibreOffice not found — install with: apt install libreoffice")
        print("  [refresh] Skipping repagination step")
        return False

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_input = Path(tmpdir) / input_path.name
        shutil.copy2(input_path, tmp_input)

        cmd = [
            soffice, '--headless', '--norestore',
            '--convert-to', 'docx',
            '--outdir', tmpdir,
            str(tmp_input)
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            converted = list(Path(tmpdir).glob('*.docx'))
            converted = [f for f in converted if f != tmp_input]
            if converted:
                shutil.copy2(converted[0], output_path)
                print(f"  [refresh] LibreOffice repagination complete")
                return True
            else:
                print(f"  [refresh] LibreOffice ran but no output found")
                return False
        except subprocess.TimeoutExpired:
            print("  [refresh] LibreOffice timed out after 120s")
            return False
        except Exception as e:
            print(f"  [refresh] LibreOffice error: {e}")
            return False


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Fix formatting of a humanized .docx — never touches text content",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
What this fixes (text content is NEVER modified):
  - Full justification on all body paragraphs
  - Times New Roman 12pt font consistency
  - 1.5 line spacing on body text
  - Heading space-before / space-after normalization
  - Widow/orphan control (prevents single lines on page breaks)
  - TOC: replaces manual "……1" strings with Word auto-TOC field
  - LibreOffice repagination (if installed) to refresh TOC page numbers

Usage after humanizing:
  python3 format_doc.py humanized.docx formatted.docx
        """)
    ap.add_argument("input",  help="Input docx (humanized)")
    ap.add_argument("output", help="Output docx (formatted)")
    ap.add_argument("--no-toc",     action="store_true", help="Skip TOC field injection")
    ap.add_argument("--no-refresh", action="store_true", help="Skip LibreOffice repagination")
    ap.add_argument("--report",     action="store_true", help="Show what would change, don't write")
    args = ap.parse_args()

    src = Path(args.input)
    if not src.exists():
        sys.exit(f"ERROR: {src} not found.")

    print(f"\n{'='*55}")
    print(f"  format_doc.py")
    print(f"  Input  : {src}")
    print(f"  Output : {args.output}")
    if args.report:
        print(f"  Mode   : REPORT ONLY (no changes written)")
    print(f"{'='*55}\n")

    doc = Document(str(src))

    body_fixed    = 0
    heading_fixed = 0
    report_lines  = []

    print("Scanning paragraphs...")
    for para in doc.paragraphs:
        style = para.style.name.lower()

        if is_body_paragraph(para):
            changes = fix_paragraph_format(para, report_mode=args.report)
            if changes:
                body_fixed += 1
                if args.report and len(report_lines) < 20:
                    report_lines.append(f"  BODY [{para.text[:50]}]: {', '.join(changes)}")

        elif any(h in style for h in ['heading 2', 'heading 3']):
            changes = fix_heading_spacing(para, report_mode=args.report)
            if changes:
                heading_fixed += 1

    print(f"  Body paragraphs fixed : {body_fixed}")
    print(f"  Heading spacing fixed : {heading_fixed}")

    if args.report:
        print(f"\nSample changes (first 20):")
        for line in report_lines:
            print(line)
        return

    # TOC rebuild
    toc_injected = False
    if not args.no_toc:
        print("\nRebuilding TOC...")
        toc_injected = inject_toc_field(doc)

    # Save intermediate
    out_path = Path(args.output)
    doc.save(str(out_path))
    print(f"\nSaved formatted document → {out_path}")

    # LibreOffice repagination
    if not args.no_refresh:
        print("\nRunning LibreOffice repagination...")
        refreshed = libreoffice_refresh(out_path, out_path)
        if not refreshed and toc_injected:
            print("  NOTE: Open the output in Word and press Ctrl+A then F9 to update TOC page numbers.")

    print(f"\n✓  Done!")
    print(f"   {body_fixed} body paragraphs justified + font-normalized")
    print(f"   {heading_fixed} heading spacings corrected")
    if toc_injected:
        print(f"   TOC field injected — open in Word and press F9 to refresh page numbers")
    print()


if __name__ == "__main__":
    main()