"""Regenerate svg_final/* from svg_editable_text/* with text converted to
outlines (uharfbuzz shaping + fontTools glyph paths, Tahoma/DejaVuSans).
Keeps every non-text element byte-identical. Prints per-file report.
"""
import os
import re
import xml.etree.ElementTree as ET

import uharfbuzz as hb
from fontTools.ttLib import TTFont
from fontTools.pens.svgPathPen import SVGPathPen

SRC = r"docs\thesis\filesCloude\thesis_figures\svg_editable_text"
DST = r"docs\thesis\filesCloude\thesis_figures\svg_final"
WINFonts = r"C:\Windows\Fonts"
MPLFonts = os.path.join(".venv", "Lib", "site-packages", "matplotlib", "mpl-data", "fonts", "ttf")

FONTS = {
    False: [os.path.join(WINFonts, "tahoma.ttf"), os.path.join(MPLFonts, "DejaVuSans.ttf")],
    True: [os.path.join(WINFonts, "tahomabd.ttf"), os.path.join(MPLFonts, "DejaVuSans-Bold.ttf")],
}

RTL_RE = re.compile("[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]")
CTRL_RE = re.compile("[\u202A-\u202E\u200E\u200F\u2066-\u2069]")


class Shaper:
    def __init__(self):
        self.cache = {}

    def font(self, bold):
        if bold not in self.cache:
            tts, hbs, ups = [], [], None
            for p in FONTS[bold]:
                data = open(p, "rb").read()
                face = hb.Face(data)
                hbs.append(hb.Font(face))
                tt = TTFont(p)
                tts.append(tt)
                if ups is None:
                    ups = tt["head"].unitsPerEm
            self.cache[bold] = (tts, hbs, ups)
        return self.cache[bold]

    def shape(self, text, size, bold):
        """Return (svg_path_d_list, total_advance_user_units)."""
        tts, hbs, upm = self.font(bold)
        clean = CTRL_RE.sub("", text)
        rtl = bool(RTL_RE.search(clean))
        for tt, hbf in zip(tts, hbs):
            cmap = tt.getBestCmap()
            if all(ord(c) in cmap or c == " " for c in clean):
                use_tt, use_hb = tt, hbf
                break
        else:
            raise RuntimeError("missing glyph for: " + clean)
        buf = hb.Buffer()
        buf.add_str(clean)
        buf.guess_segment_properties()
        buf.direction = "rtl" if rtl else "ltr"
        hb.shape(use_hb, buf)
        infos, poss = buf.glyph_infos, buf.glyph_positions
        k = size / upm
        glyphset = use_tt.getGlyphSet()
        out, pen_x = [], 0
        for info, pos in zip(infos, poss):
            name = use_tt.getGlyphOrder()[info.codepoint]
            pen = SVGPathPen(glyphset)
            glyphset[name].draw(pen)
            d = pen.getCommands()
            gx = pen_x + pos.x_offset
            gy = pos.y_offset
            if d:
                out.append((d, gx, gy))
            pen_x += pos.x_advance
        return out, pen_x * k, k


NS = {"s": "http://www.w3.org/2000/svg"}
shaper = Shaper()

TEXT_RE = re.compile(r'<text\s+([^>]*)>(.*?)</text>', re.S)
ATTR_RE = re.compile(r'([\w-]+)="([^"]*)"')


def convert(src_path, dst_path):
    raw = open(src_path, encoding="utf-8").read()
    n_text = n_glyph = 0

    def repl(m):
        nonlocal n_text, n_glyph
        attrs = dict(ATTR_RE.findall(m.group(1)))
        txt = m.group(2)
        size = float(attrs.get("font-size", "14"))
        bold = attrs.get("font-weight", "normal") == "bold"
        anchor = attrs.get("text-anchor", "start")
        x = float(attrs.get("x", "0"))
        y = float(attrs.get("y", "0"))
        fill = attrs.get("fill", "#000")
        glyphs, total, k = shaper.shape(txt, size, bold)
        x0 = x - total / 2 if anchor == "middle" else (x - total if anchor == "end" else x)
        parts = [f'<g transform="translate({x0:g},{y:g}) scale({k:.6f},{-k:.6f})">']
        for d, gx, gy in glyphs:
            parts.append(f'<path d="{d}" transform="translate({gx:g},{gy:g})" fill="{fill}"/>')
            n_glyph += 1
        parts.append("</g>")
        n_text += 1
        return "".join(parts)

    out = TEXT_RE.sub(repl, raw)
    open(dst_path, "w", encoding="utf-8").write(out)
    return n_text, n_glyph


if __name__ == "__main__":
    for f in sorted(os.listdir(SRC)):
        if not f.endswith(".svg"):
            continue
        nt, ng = convert(os.path.join(SRC, f), os.path.join(DST, f))
        print(f"{f}: {nt} texts -> {ng} glyph paths")
