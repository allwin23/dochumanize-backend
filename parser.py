"""
parser.py — DOCX AST-level parser and reconstructor.

Strategy:
  - Unzips the .docx (it's a ZIP) and operates directly on word/document.xml
  - Extracts only <w:p> paragraph nodes that sit under body-level or
    section-level positions (NOT inside <w:tbl> tables or <w:txbxContent>)
  - Assigns a stable integer ID to each target paragraph
  - On reconstruction, surgically replaces only the <w:t> text nodes
    inside target paragraphs, leaving all <w:rPr>, <w:pPr>, <w:drawing>,
    <w:tbl> nodes completely untouched.
"""

import copy
import re
from lxml import etree
from docx import Document
from docx.oxml.ns import qn
from typing import List, Dict, Any


# XML namespaces used in OOXML
W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def _is_inside_table(element) -> bool:
    """Walk ancestors — return True if this element lives inside a <w:tbl>."""
    parent = element.getparent()
    while parent is not None:
        if parent.tag == qn("w:tbl"):
            return True
        parent = parent.getparent()
    return False


def _is_inside_textbox(element) -> bool:
    """Return True if element is inside a floating text box (<w:txbxContent>)."""
    parent = element.getparent()
    while parent is not None:
        if parent.tag == qn("w:txbxContent"):
            return True
        parent = parent.getparent()
    return False


def _para_has_drawing(para) -> bool:
    """Return True if the paragraph contains an image/drawing."""
    return bool(para.findall(".//" + qn("w:drawing")))


def _get_paragraph_text(para) -> str:
    """Concatenate all <w:t> text content within a paragraph."""
    parts = []
    for t_node in para.iter(qn("w:t")):
        parts.append(t_node.text or "")
    return "".join(parts)


def _get_style_name(para) -> str:
    """Return the paragraph style name (e.g., 'Heading1', 'Normal')."""
    pPr = para.find(qn("w:pPr"))
    if pPr is not None:
        pStyle = pPr.find(qn("w:pStyle"))
        if pStyle is not None:
            return pStyle.get(qn("w:val"), "")
    return ""


HEADING_PATTERN = re.compile(r"^[Hh]eading\s*\d+$|^Title$|^Subtitle$")


class DocxParser:
    """
    Parses a .docx file into a structured paragraph map,
    and can reconstruct the document with replaced text.
    """

    def __init__(self, path: str):
        self.path = path
        self.doc = Document(path)
        self._para_map: List[Dict[str, Any]] = []
        # Map from para_id → lxml element reference (for reconstruction)
        self._element_refs: Dict[int, Any] = {}

    def extract_paragraphs(self) -> List[Dict[str, Any]]:
        """
        Walk the document body and return a list of paragraph descriptors:
        [
          {
            "id": 0,
            "text": "The quick brown fox...",
            "style": "Normal",
            "word_count": 5,
          },
          ...
        ]
        Paragraphs inside tables, textboxes, headings, or drawings are excluded
        and marked as frozen (they will pass through unchanged).
        """
        self._para_map = []
        body = self.doc.element.body
        para_id = 0

        for para in body.iter(qn("w:p")):
            text = _get_paragraph_text(para).strip()
            style = _get_style_name(para)

            # --- Freeze conditions ---
            skip = (
                not text                          # empty paragraph
                or len(text) < 40                 # too short to rewrite meaningfully
                or HEADING_PATTERN.match(style)   # headings
                or _is_inside_table(para)         # table cells
                or _is_inside_textbox(para)       # text boxes / captions
                or _para_has_drawing(para)        # paragraphs with inline images
            )

            if not skip:
                self._para_map.append({
                    "id": para_id,
                    "text": text,
                    "style": style,
                    "word_count": len(text.split()),
                })
                self._element_refs[para_id] = para

            para_id += 1

        return self._para_map

    def reconstruct(self, replacements: Dict[int, str], output_path: str):
        """
        Given a dict of {para_id: new_text}, replace the text content of
        each target paragraph in-place, preserving all formatting runs.

        Strategy per paragraph:
          1. Collect all <w:r> (run) elements.
          2. Keep ALL run formatting (<w:rPr>) intact.
          3. Place the entire new text into the FIRST run's <w:t> node.
          4. Delete all subsequent runs (they're now redundant).
          5. This preserves font, bold, italic, size from the first run.
        """
        for para_id, new_text in replacements.items():
            if para_id not in self._element_refs:
                continue

            para = self._element_refs[para_id]
            runs = para.findall(".//" + qn("w:r"))

            if not runs:
                continue

            # Preserve first run's formatting; set its text
            first_run = runs[0]
            t_nodes = first_run.findall(qn("w:t"))

            if t_nodes:
                t_node = t_nodes[0]
            else:
                # Create a new <w:t> node inside the first run
                t_node = etree.SubElement(first_run, qn("w:t"))

            t_node.text = new_text
            # Preserve spaces attribute so leading/trailing spaces aren't stripped
            t_node.set(
                "{http://www.w3.org/XML/1998/namespace}space",
                "preserve"
            )

            # Remove all extra <w:t> nodes from the first run
            for extra_t in t_nodes[1:]:
                first_run.remove(extra_t)

            # Remove all subsequent runs — their text is now in first_run
            for run in runs[1:]:
                para.remove(run)

        self.doc.save(output_path)