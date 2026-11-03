"""Minimal XLSX writer/reader (standard library only, no openpyxl dependency).

write_xlsx(path, {"Sheet": [header, row, ...]}) writes inline strings with
wrapped text and frozen header rows.  read_xlsx(path) reads such files back,
including files re-saved by Excel/LibreOffice (shared strings).
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET
from xml.sax.saxutils import escape

NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
_ILLEGAL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _col(i: int) -> str:
    s = ""
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        s = chr(65 + r) + s
    return s


def _cell(ref: str, value) -> str:
    if value is None or value == "":
        return f'<c r="{ref}" s="1"/>'
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f'<c r="{ref}" s="1"><v>{value}</v></c>'
    text = escape(_ILLEGAL.sub("", str(value)))
    return f'<c r="{ref}" t="inlineStr" s="1"><is><t xml:space="preserve">{text}</t></is></c>'


def _sheet_xml(rows: list[list], widths: list[int] | None) -> str:
    cols = ""
    if widths:
        cols = "<cols>" + "".join(
            f'<col min="{i + 1}" max="{i + 1}" width="{w}" customWidth="1"/>' for i, w in enumerate(widths)) + "</cols>"
    body = []
    for r, row in enumerate(rows, start=1):
        cells = "".join(_cell(f"{_col(c)}{r}", v) for c, v in enumerate(row))
        body.append(f'<row r="{r}">{cells}</row>')
    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            '<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" '
            'state="frozen"/></sheetView></sheetViews>'
            f'{cols}<sheetData>{"".join(body)}</sheetData></worksheet>')


def write_xlsx(path: Path, sheets: dict[str, list[list]], widths: dict[str, list[int]] | None = None) -> None:
    names = list(sheets)
    ct = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
          '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
          '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
          '<Default Extension="xml" ContentType="application/xml"/>'
          '<Override PartName="/xl/workbook.xml" '
          'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
          '<Override PartName="/xl/styles.xml" '
          'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
          + "".join(f'<Override PartName="/xl/worksheets/sheet{i + 1}.xml" '
                    'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
                    for i in range(len(names))) + "</Types>")
    rels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/'
            'officeDocument" Target="xl/workbook.xml"/></Relationships>')
    wb = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
          '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
          'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>'
          + "".join(f'<sheet name="{escape(n[:31])}" sheetId="{i + 1}" r:id="rId{i + 1}"/>'
                    for i, n in enumerate(names)) + "</sheets></workbook>")
    wb_rels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
               '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
               + "".join(f'<Relationship Id="rId{i + 1}" Type="http://schemas.openxmlformats.org/officeDocument/'
                         f'2006/relationships/worksheet" Target="worksheets/sheet{i + 1}.xml"/>'
                         for i in range(len(names)))
               + f'<Relationship Id="rId{len(names) + 1}" Type="http://schemas.openxmlformats.org/'
                 'officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>')
    styles = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
              '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
              '<fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>'
              '<fills count="1"><fill><patternFill patternType="none"/></fill></fills>'
              '<borders count="1"><border/></borders><cellStyleXfs count="1"><xf/></cellStyleXfs>'
              '<cellXfs count="2"><xf/><xf applyAlignment="1"><alignment wrapText="1" vertical="top"/></xf>'
              '</cellXfs></styleSheet>')
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.xlsx")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", ct)
        z.writestr("_rels/.rels", rels)
        z.writestr("xl/workbook.xml", wb)
        z.writestr("xl/_rels/workbook.xml.rels", wb_rels)
        z.writestr("xl/styles.xml", styles)
        for i, n in enumerate(names):
            z.writestr(f"xl/worksheets/sheet{i + 1}.xml", _sheet_xml(sheets[n], (widths or {}).get(n)))
    tmp.replace(path)


def read_xlsx(path: Path) -> dict[str, list[list[str]]]:
    with zipfile.ZipFile(path) as z:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in z.namelist():
            root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            for si in root.findall("m:si", NS):
                shared.append("".join(t.text or "" for t in si.iter(f"{{{NS['m']}}}t")))
        wb = ET.fromstring(z.read("xl/workbook.xml"))
        rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
        target = {r.get("Id"): r.get("Target") for r in rels}
        out: dict[str, list[list[str]]] = {}
        for sh in wb.find("m:sheets", NS):
            rid = sh.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
            p = target[rid].lstrip("/")
            p = p if p.startswith("xl/") else f"xl/{p}"
            root = ET.fromstring(z.read(p))
            rows: list[list[str]] = []
            for row in root.iter(f"{{{NS['m']}}}row"):
                vals: dict[int, str] = {}
                for c in row.findall("m:c", NS):
                    ref = re.match(r"([A-Z]+)", c.get("r")).group(1)
                    idx = 0
                    for ch in ref:
                        idx = idx * 26 + ord(ch) - 64
                    t = c.get("t")
                    if t == "inlineStr":
                        v = "".join(x.text or "" for x in c.iter(f"{{{NS['m']}}}t"))
                    else:
                        node = c.find("m:v", NS)
                        v = node.text if node is not None and node.text else ""
                        if t == "s" and v:
                            v = shared[int(v)]
                    vals[idx - 1] = v
                rows.append([vals.get(i, "") for i in range(max(vals) + 1)] if vals else [])
            out[sh.get("name")] = rows
        return out
