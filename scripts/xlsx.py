"""Minimal standard-library .xlsx reader and writer (one header row, plain cells).

The reproduction tables are plain tables; this keeps them buildable and checkable without
openpyxl or Excel.
"""

import math
import re
import zipfile
from xml.sax.saxutils import escape

NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG = "http://schemas.openxmlformats.org/package/2006"
NUMBER_FORMAT = "0.####"


def _col(i):
    s = ""
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        s = chr(65 + r) + s
    return s


class _Deterministic:
    """Zip writer with fixed timestamps, so rebuilding a table gives byte-identical files."""

    def __init__(self, zf):
        self.zf = zf

    def writestr(self, name, data):
        info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        self.zf.writestr(info, data)


def write(path, sheets, number_columns=None):
    """sheets: list of (name, header, rows). number_columns: {sheet name: set of column indexes}
    displayed with up to four decimals (values keep full precision)."""
    number_columns = number_columns or {}
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        z = _Deterministic(zf)
        z.writestr("[Content_Types].xml",
            f'<?xml version="1.0" encoding="UTF-8"?><Types xmlns="{PKG}/content-types">'
            f'<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            f'<Default Extension="xml" ContentType="application/xml"/>'
            f'<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            f'<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
            + "".join(f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
                      for i in range(1, len(sheets) + 1)) + "</Types>")
        z.writestr("_rels/.rels", f'<Relationships xmlns="{PKG}/relationships"><Relationship Id="rId1" Type="{REL}/officeDocument" Target="xl/workbook.xml"/></Relationships>')
        z.writestr("xl/workbook.xml", f'<workbook xmlns="{NS}" xmlns:r="{REL}"><sheets>'
            + "".join(f'<sheet name="{escape(n)}" sheetId="{i}" r:id="rId{i}"/>' for i, (n, _, _) in enumerate(sheets, 1))
            + "</sheets></workbook>")
        z.writestr("xl/_rels/workbook.xml.rels", f'<Relationships xmlns="{PKG}/relationships">'
            + "".join(f'<Relationship Id="rId{i}" Type="{REL}/worksheet" Target="worksheets/sheet{i}.xml"/>' for i in range(1, len(sheets) + 1))
            + f'<Relationship Id="rId{len(sheets) + 1}" Type="{REL}/styles" Target="styles.xml"/></Relationships>')
        z.writestr("xl/styles.xml", f'<styleSheet xmlns="{NS}"><numFmts count="1"><numFmt numFmtId="164" formatCode="{NUMBER_FORMAT}"/></numFmts>'
            '<fonts count="2"><font><sz val="11"/><name val="Calibri"/></font><font><b/><sz val="11"/><name val="Calibri"/></font></fonts>'
            '<fills count="2"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill></fills>'
            '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
            '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
            '<cellXfs count="3"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>'
            '<xf numFmtId="164" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/>'
            '<xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0" applyFont="1"/></cellXfs>'
            '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles></styleSheet>')
        for i, (name, header, rows) in enumerate(sheets, 1):
            numeric = number_columns.get(name, set())

            def cell(ref, v, style=0):
                if v is None or v == "":
                    return ""
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    assert math.isfinite(v), (ref, v)
                    return f'<c r="{ref}" s="{style}"><v>{v!r}</v></c>'
                return f'<c r="{ref}" t="inlineStr" s="{style}"><is><t xml:space="preserve">{escape(str(v))}</t></is></c>'

            xml = [f'<worksheet xmlns="{NS}"><sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews><sheetData>']
            xml.append('<row r="1">' + "".join(cell(f"{_col(j)}1", h, 2) for j, h in enumerate(header)) + "</row>")
            for r, row in enumerate(rows, 2):
                xml.append(f'<row r="{r}">' + "".join(cell(f"{_col(j)}{r}", v, 1 if j in numeric else 0) for j, v in enumerate(row)) + "</row>")
            xml.append("</sheetData></worksheet>")
            z.writestr(f"xl/worksheets/sheet{i}.xml", "".join(xml))


def _text(x):
    return re.sub(r"<[^>]+>", "", x).replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"').replace("&apos;", "'").replace("&amp;", "&")


def read(path):
    """{sheet name: list of rows}; numbers as float, text as str, empty cells as None."""
    with zipfile.ZipFile(path) as z:
        names = re.findall(r'<sheet [^>]*name="([^"]+)"', z.read("xl/workbook.xml").decode())
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            shared = [_text(s) for s in re.findall(r"<si>(.*?)</si>", z.read("xl/sharedStrings.xml").decode(), re.S)]
        out = {}
        for i, name in enumerate(names, 1):
            xml = z.read(f"xl/worksheets/sheet{i}.xml").decode()
            rows = []
            for row in re.findall(r"<row[^>]*>(.*?)</row>", xml, re.S):
                cells = {}
                for attrs, body in re.findall(r"<c([^>]*?)(?:/>|>(.*?)</c>)", row, re.S):
                    ref = re.search(r'r="([A-Z]+)', attrs)
                    col = 0
                    if ref:
                        for ch in ref.group(1):
                            col = col * 26 + ord(ch) - 64
                        col -= 1
                    else:
                        col = len(cells)
                    t = re.search(r't="(\w+)"', attrs)
                    t = t.group(1) if t else "n"
                    if not body:
                        val = None
                    elif t == "inlineStr":
                        val = _text(body)
                    elif t == "s":
                        val = shared[int(_text(body))]
                    elif t == "str":
                        val = _text(re.search(r"<v>(.*?)</v>", body, re.S).group(1))
                    else:
                        v = re.search(r"<v>(.*?)</v>", body, re.S)
                        val = float(v.group(1)) if v else None
                    cells[col] = val
                width = max(cells) + 1 if cells else 0
                rows.append([cells.get(j) for j in range(width)])
            out[name] = rows
    return out
