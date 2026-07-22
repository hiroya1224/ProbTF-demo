#!/usr/bin/python3
"""Generate the Japanese probabilistic AprilTag implementation deck via UNO.

The script reads the checked-in Phase 3 metrics and overlays on every run.  It
can connect to an existing LibreOffice listener or start an isolated headless
listener itself, then round-trip the generated PPTX and optionally export PDF.
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import zipfile

import uno


PAGE_W = 33867
PAGE_H = 19050
FONT = "Noto Sans CJK JP"
MONO = "Noto Sans Mono CJK JP"
SLIDE_COUNT = 19
ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "docs" / "reports"
METRICS_PATH = REPORT_DIR / "prob_artag_phase3_metrics.json"
OVERLAY_DIR = REPORT_DIR / "prob_artag_phase3_overlays"


def rgb(value):
    return int(value.lstrip("#"), 16)


C = {
    "navy": rgb("0B132B"),
    "ink": rgb("152238"),
    "muted": rgb("5C6B82"),
    "soft": rgb("94A3B8"),
    "light": rgb("F5F7FB"),
    "white": rgb("FFFFFF"),
    "line": rgb("D9E1EC"),
    "cyan": rgb("0891B2"),
    "cyan_light": rgb("8DE8F5"),
    "blue": rgb("2563EB"),
    "blue_dark": rgb("1C3352"),
    "purple": rgb("7C3AED"),
    "green": rgb("059669"),
    "red": rgb("DC2626"),
    "orange": rgb("D97706"),
    "blue_tint": rgb("EAF1FF"),
    "purple_tint": rgb("F1EBFF"),
    "green_tint": rgb("E8F7F1"),
    "red_tint": rgb("FDECEC"),
    "orange_tint": rgb("FFF4E5"),
    "slate_tint": rgb("EDF2F7"),
}


def prop(name, value):
    item = uno.createUnoStruct("com.sun.star.beans.PropertyValue")
    item.Name = name
    item.Value = value
    return item


def point(x, y):
    item = uno.createUnoStruct("com.sun.star.awt.Point")
    item.X = int(x)
    item.Y = int(y)
    return item


def size(w, h):
    item = uno.createUnoStruct("com.sun.star.awt.Size")
    item.Width = int(w)
    item.Height = int(h)
    return item


def enum(type_name, value):
    return uno.Enum(type_name, value)


CURRENT_DOCUMENT = None


def attach_document(doc):
    global CURRENT_DOCUMENT
    CURRENT_DOCUMENT = doc


def raw_shape(page, service, x, y, w, h):
    if CURRENT_DOCUMENT is None:
        raise RuntimeError("No active presentation document")
    shape = CURRENT_DOCUMENT.createInstance(service)
    shape.Position = point(x, y)
    shape.Size = size(w, h)
    page.add(shape)
    return shape


def rect(page, x, y, w, h, fill, line_color=C["line"], radius=0,
         transparency=0, line_width=20):
    shape = raw_shape(page, "com.sun.star.drawing.RectangleShape", x, y, w, h)
    shape.FillStyle = enum("com.sun.star.drawing.FillStyle", "SOLID")
    shape.FillColor = fill
    shape.FillTransparence = int(transparency)
    if line_color is None:
        shape.LineStyle = enum("com.sun.star.drawing.LineStyle", "NONE")
    else:
        shape.LineStyle = enum("com.sun.star.drawing.LineStyle", "SOLID")
        shape.LineColor = line_color
        shape.LineWidth = int(line_width)
    if radius:
        try:
            shape.CornerRadius = int(radius)
        except Exception:
            pass
    return shape


def ellipse(page, x, y, w, h, fill, line_color=None, transparency=0,
            line_width=30):
    shape = raw_shape(page, "com.sun.star.drawing.EllipseShape", x, y, w, h)
    shape.FillStyle = enum("com.sun.star.drawing.FillStyle", "SOLID")
    shape.FillColor = fill
    shape.FillTransparence = int(transparency)
    if line_color is None:
        shape.LineStyle = enum("com.sun.star.drawing.LineStyle", "NONE")
    else:
        shape.LineStyle = enum("com.sun.star.drawing.LineStyle", "SOLID")
        shape.LineColor = line_color
        shape.LineWidth = int(line_width)
    return shape


def line(page, x1, y1, x2, y2, color=C["line"], width=30):
    shape = raw_shape(
        page, "com.sun.star.drawing.LineShape", min(x1, x2), min(y1, y2),
        max(abs(x2 - x1), 1), max(abs(y2 - y1), 1),
    )
    shape.LineStyle = enum("com.sun.star.drawing.LineStyle", "SOLID")
    shape.LineColor = color
    shape.LineWidth = int(width)
    return shape


def text(page, x, y, w, h, value, font_size=14, color=C["ink"], bold=False,
         align="LEFT", valign="TOP", font=FONT):
    shape = raw_shape(page, "com.sun.star.drawing.TextShape", x, y, w, h)
    shape.String = str(value)
    shape.FillStyle = enum("com.sun.star.drawing.FillStyle", "NONE")
    shape.LineStyle = enum("com.sun.star.drawing.LineStyle", "NONE")
    shape.CharFontName = font
    shape.CharFontNameAsian = font
    shape.CharHeight = float(font_size)
    shape.CharHeightAsian = float(font_size)
    shape.CharColor = color
    shape.CharWeight = 150.0 if bold else 100.0
    shape.CharWeightAsian = 150.0 if bold else 100.0
    shape.ParaAdjust = enum("com.sun.star.style.ParagraphAdjust", align)
    shape.TextVerticalAdjust = enum("com.sun.star.drawing.TextVerticalAdjust", valign)
    shape.TextHorizontalAdjust = enum("com.sun.star.drawing.TextHorizontalAdjust", align)
    shape.TextLeftDistance = 0
    shape.TextRightDistance = 0
    shape.TextUpperDistance = 0
    shape.TextLowerDistance = 0
    return shape


def image(page, path, x, y, w, h, border=True):
    if border:
        rect(page, x - 70, y - 70, w + 140, h + 140, C["white"], C["line"], 160)
    shape = raw_shape(page, "com.sun.star.drawing.GraphicObjectShape", x, y, w, h)
    shape.GraphicURL = uno.systemPathToFileUrl(str(Path(path).resolve()))
    return shape


def set_page(page, background=C["light"]):
    page.Width = PAGE_W
    page.Height = PAGE_H
    page.Background = None
    rect(page, 0, 0, PAGE_W, PAGE_H, background, None)


def add_header(page, category, title_value, subtitle, accent):
    text(page, 1500, 500, 9000, 430, category, 10, accent, True)
    text(page, 1500, 1020, 30300, 900, title_value, 25, C["ink"], True)
    text(page, 1500, 1970, 30000, 430, subtitle, 10.5, C["muted"])
    line(page, 1500, 2530, 32300, 2530, C["line"], 15)
    rect(page, 0, 0, 260, PAGE_H, accent, None)


def add_footer(page, number, dark=False):
    line_color = C["blue_dark"] if dark else C["line"]
    label_color = C["soft"] if dark else rgb("9CB2CE")
    line(page, 1500, 18045, 32300, 18045, line_color, 12)
    text(page, 1500, 18135, 12000, 380,
         "ProbTF-demo  |  AprilTag implementation overview", 8.5, label_color)
    text(page, 30900, 18135, 1400, 380, "{:02d}".format(number), 9,
         C["cyan"] if dark else C["blue"], True, "RIGHT")


def pill(page, x, y, w, label, accent, tint, font_size=10):
    rect(page, x, y, w, 560, tint, None, 180)
    text(page, x, y + 85, w, 350, label, font_size, accent, True, "CENTER", "CENTER")


def numbered_card(page, x, y, w, h, number, title_value, body, accent, tint):
    rect(page, x, y, w, h, C["white"], C["line"], 550)
    rect(page, x, y, 90, h, accent, None)
    ellipse(page, x + 430, y + 420, 760, 760, tint)
    text(page, x + 430, y + 520, 760, 470, number, 14, accent, True, "CENTER", "CENTER")
    text(page, x + 1450, y + 350, w - 1900, 650, title_value, 17, C["ink"], True)
    text(page, x + 520, y + 1450, w - 1040, h - 1850, body, 11.5, C["muted"])


def metric_card(page, x, y, w, label, value, accent, tint, note=""):
    rect(page, x, y, w, 2100, C["white"], C["line"], 400)
    rect(page, x, y, 80, 2100, accent, None)
    text(page, x + 420, y + 280, w - 700, 420, label, 10, C["muted"], True)
    text(page, x + 420, y + 760, w - 700, 720, value, 24, accent, True)
    if note:
        text(page, x + 420, y + 1580, w - 700, 320, note, 8.5, C["soft"])


def arrow(page, x, y, color=C["cyan"], size_value=26):
    text(page, x, y, 900, 650, "→", size_value, color, True, "CENTER", "CENTER")


def slide_title(page):
    set_page(page, C["navy"])
    rect(page, 0, 0, 280, PAGE_H, C["cyan"], None)
    ellipse(page, 24400, -3100, 12800, 12800, C["blue"], transparency=68)
    ellipse(page, 21100, 3500, 5400, 5400, C["purple"], transparency=62)
    ellipse(page, 27100, 7200, 9600, 9600, C["cyan"], transparency=76)
    rect(page, 1780, 1440, 7200, 560, C["blue_dark"], rgb("3A5577"), 130)
    text(page, 2200, 1525, 6350, 400,
         "PROBTF-DEMO  •  TECHNICAL OVERVIEW", 10, C["cyan_light"], True, "CENTER")
    text(page, 1780, 3930, 21000, 1400, "確率的 AprilTag Detector", 31, C["white"], True)
    text(page, 1780, 5940, 19000, 830,
         "合成画像から mixture ProbTF edge まで", 20, rgb("DCE8F8"), True)
    text(page, 1780, 7290, 18500, 650,
         "レンダリング  /  幾何推定  /  局所 Hessian  /  ROS", 15, C["cyan_light"])
    line(page, 1800, 8730, 11700, 8730, C["cyan"], 55)
    text(page, 1780, 9360, 20500, 520,
         "pyrender  •  OpenCV IPPE  •  native ProbTF v2", 12, rgb("AFC3DD"))

    rect(page, 28200, 10720, 2600, 2600, C["white"], C["cyan"], 70, line_width=35)
    rect(page, 28630, 11150, 1740, 1740, C["ink"], None)
    rect(page, 29050, 11570, 900, 900, C["white"], None)
    ellipse(page, 27100, 10080, 4700, 1700, C["navy"], C["cyan"], 100, 40)
    ellipse(page, 27450, 11750, 4500, 1600, C["navy"], C["purple"], 100, 40)
    text(page, 28750, 9550, 3300, 600, "p(T_C_M)", 17, C["white"], True, "CENTER")
    text(page, 1780, 16600, 4500, 420, "2026-07-22", 10, C["soft"])
    text(page, 6000, 16600, 4500, 420, "ProbTF-demo", 10, C["soft"])


def slide_phases(page):
    set_page(page)
    add_header(page, "GOAL & PHASES", "目的：曖昧さを edge distribution として公開",
               "既知の transform から観測を作り、detector が復元する分布を閉ループで検証", C["blue"])
    cards = [
        ("1", "Renderer", "RGB / depth / instance\n完全な幾何 ground truth", C["blue"], C["blue_tint"]),
        ("2", "Detector", "ordered corner / IPPE\n局所 posterior mixture", C["purple"], C["purple_tint"]),
        ("3", "Validation", "scenario 別 benchmark\n再現性と失敗境界", C["green"], C["green_tint"]),
        ("4", "Documentation", "数式・API・実測値を\n再生成可能な deck へ", C["orange"], C["orange_tint"]),
    ]
    for index, (num, title_value, body, accent, tint) in enumerate(cards):
        x = 1500 + index * 7900
        numbered_card(page, x, 3500, 7000, 6100, num, title_value, body, accent, tint)
        pill(page, x + 1650, 8650, 3700, "IMPLEMENTED", accent, tint, 9)
        if index < 3:
            arrow(page, x + 7060, 6000)
    rect(page, 1500, 11100, 30700, 4500, C["navy"], None, 650)
    text(page, 2200, 11800, 8700, 450, "設計上の境界", 13, C["cyan_light"], True)
    text(page, 2200, 12600, 13300, 1600,
         "画像処理・PnP・分布構成は detector\nProbTF core は公開分布の保持・合成・照会", 18, C["white"], True)
    text(page, 17200, 11950, 12500, 2100,
         "✓ 粒子表現を使わない\n✓ 二解を最良一解へ潰さない\n✓ ground truth は画像から逆算しない", 13, rgb("C2D1E5"))
    add_footer(page, 2)


def slide_architecture(page):
    set_page(page)
    add_header(page, "ARCHITECTURE", "観測生成と ground truth を別経路にする",
               "wire boundary は画像・metadata・ProbTF v2 message", C["green"])
    stages = [
        (1700, 3900, 5700, "Scene sampler", "T_W_C / T_W_M\nseed / scenario", C["blue"], C["blue_tint"]),
        (8600, 3900, 5700, "pyrender", "synthetic RGB\ndepth / instance", C["blue"], C["blue_tint"]),
        (15500, 3900, 5700, "Corner detector", "ID + ordered y∈R⁸\nΣ_img", C["purple"], C["purple_tint"]),
        (22400, 3900, 5700, "Pose mixture", "IPPE → refine →\n(w,A,m,S,C)", C["purple"], C["purple_tint"]),
        (29200, 3900, 2900, "ProbTF", "/probtf\nedge", C["green"], C["green_tint"]),
    ]
    for x, y, w, title_value, body, accent, tint in stages:
        rect(page, x, y, w, 3700, C["white"], C["line"], 400)
        rect(page, x, y, w, 75, accent, None)
        text(page, x + 350, y + 450, w - 700, 620, title_value, 14, C["ink"], True, "CENTER")
        text(page, x + 350, y + 1350, w - 700, 1250, body, 10.5, C["muted"], False, "CENTER")
    for x in (7470, 14370, 21270, 28200):
        arrow(page, x, 5250)
    line(page, 4500, 8100, 4500, 10300, C["orange"], 35)
    line(page, 4500, 10300, 25200, 10300, C["orange"], 35)
    arrow(page, 25000, 9970, C["orange"], 22)
    rect(page, 1700, 11300, 12600, 3600, C["orange_tint"], rgb("F2C27E"), 450)
    text(page, 2300, 11800, 11200, 560, "解析 ground truth", 14, C["orange"], True)
    text(page, 2300, 12650, 11200, 1250,
         "cv2.projectPoints(T_C_M)\nrendered pixel を逆算しない", 12, C["ink"])
    rect(page, 15500, 11300, 16600, 3600, C["navy"], None, 450)
    text(page, 16100, 11800, 15100, 550, "責務分離", 14, C["cyan_light"], True)
    text(page, 16100, 12600, 15100, 1350,
         "ProbTF core は image / corner / PnP / Hessian を参照しない\n公開された full mixture law だけを graph kernel として扱う", 12.5, C["white"])
    add_footer(page, 3)


def slide_coordinates(page):
    set_page(page)
    add_header(page, "COORDINATES", "座標規約：T_A_B は B → A を写す",
               "OpenCV optical と OpenGL camera の符号反転を一関数へ隔離", C["blue"])
    rect(page, 1600, 3450, 14300, 11800, C["white"], C["line"], 600)
    text(page, 2250, 3950, 12000, 550, "保存する3変換", 15, C["blue"], True)
    text(page, 2450, 5200, 11800, 1900,
         "T_W_C       T_W_M\n\nT_C_M = inv(T_W_C) · T_W_M", 20, C["ink"], True, "CENTER", font=MONO)
    rect(page, 2700, 8200, 11300, 2400, C["navy"], None, 350)
    text(page, 3100, 8720, 10500, 1200,
         "z_C = R(Q) z_M + X\nedge direction:  camera ← marker", 16, C["white"], True, "CENTER", "CENTER", MONO)
    text(page, 2350, 11900, 12000, 1800,
         "W : world\nC : OpenCV / ROS optical camera\nMᵢ : tag i（原点は中心、単位 m）", 12, C["muted"])

    rect(page, 17400, 3450, 14300, 11800, C["white"], C["line"], 600)
    text(page, 18050, 3950, 12500, 550, "camera convention", 15, C["blue"], True)
    rows = [
        ("OpenCV / ROS optical", "x 右  /  y 下  /  z 前方", C["blue"], C["blue_tint"]),
        ("pyrender / OpenGL", "x 右  /  y 上  /  z 後方", C["purple"], C["purple_tint"]),
    ]
    for index, (name, axes, accent, tint) in enumerate(rows):
        y = 5150 + index * 2450
        rect(page, 18100, y, 12900, 1900, tint, None, 300)
        text(page, 18600, y + 300, 5000, 450, name, 11, accent, True)
        text(page, 23500, y + 300, 6900, 500, axes, 12, C["ink"], True)
    text(page, 18400, 10350, 12300, 1500,
         "F = diag(1, −1, −1, 1)\nT_W_GL = T_W_C · F", 18, C["ink"], True, "CENTER", "CENTER", MONO)
    pill(page, 19900, 12750, 9600, "coordinates.py の1箇所だけ", C["green"], C["green_tint"], 11)
    text(page, 18400, 14000, 12300, 550,
         "非ゼロ xyz + roll / pitch / yaw で unit test", 10.5, C["muted"], False, "CENTER")
    add_footer(page, 4)


def slide_corner_order(page):
    set_page(page)
    add_header(page, "CORNER CONTRACT", "corner 順は API の一部として固定",
               "detector・renderer・IPPE の correspondence を専用 adapter で一致させる", C["purple"])
    rect(page, 1600, 3400, 13400, 11900, C["white"], C["line"], 600)
    text(page, 2200, 3900, 12000, 520, "tag frame（y は画像上）", 14, C["purple"], True)
    rect(page, 5100, 5550, 6500, 6500, C["white"], C["ink"], 80, line_width=45)
    labels = [
        (4650, 5000, "p₁  TL\n(−L/2,+L/2)"),
        (10300, 5000, "p₂  TR\n(+L/2,+L/2)"),
        (10300, 12100, "p₃  BR\n(+L/2,−L/2)"),
        (4650, 12100, "p₄  BL\n(−L/2,−L/2)"),
    ]
    for x, y, label in labels:
        ellipse(page, x, y, 760, 760, C["purple"])
        text(page, x - 650, y + 850, 2100, 850, label, 9, C["ink"], True, "CENTER")
    text(page, 6300, 7700, 4100, 700, "AprilTag", 18, C["ink"], True, "CENTER")
    text(page, 6300, 8650, 4100, 850, "1 → 2 → 3 → 4", 14, C["purple"], True, "CENTER")

    rect(page, 16500, 3400, 15200, 11900, C["white"], C["line"], 600)
    text(page, 17150, 3900, 13800, 520, "OpenCV contract", 14, C["purple"], True)
    text(page, 17400, 5000, 13500, 1550,
         "ArucoDetector\nID の面内回転を復号して TL → TR → BR → BL", 14, C["ink"], True)
    arrow(page, 22500, 6750, C["purple"])
    text(page, 17400, 7700, 13500, 1600,
         "solvePnPGeneric(..., SOLVEPNP_IPPE_SQUARE)\n上記 object points を厳密な順で渡す", 13, C["ink"], True)
    rect(page, 17400, 10100, 13500, 2700, C["red_tint"], rgb("F3A6A6"), 350)
    text(page, 17900, 10550, 12500, 1500,
         "参照文書の例示順をそのまま IPPE に渡さない\n上下反転・transpose は正面タグだけでは隠れる", 11.5, C["red"], True)
    pill(page, 18500, 13550, 11300,
         "rendered corner ↔ projectPoints : max ≤ 2 px", C["green"], C["green_tint"], 10.5)
    add_footer(page, 5)


def slide_renderer(page, overlays):
    set_page(page)
    add_header(page, "PHASE 1  |  RENDERER", "可視性を保証してから world pose へ変換",
               "美観よりも geometry・determinism・failure scenario を優先", C["blue"])
    image(page, overlays["moderate"], 1650, 3500, 10240, 7680)
    text(page, 1800, 11450, 9900, 450, "moderate fixture  |  GT / detected / mode axes", 9, C["muted"], False, "CENTER")
    rect(page, 12800, 3500, 8900, 7700, C["white"], C["line"], 500)
    text(page, 13400, 3970, 7600, 550, "render path", 14, C["blue"], True)
    text(page, 13400, 4950, 7600, 4700,
         "• IntrinsicsCamera(K)\n• AprilTag + white margin\n• coded square = 0.12 m\n• one-sided textured quad\n• ambient only / shadow off\n• EGL / OSMesa lazy import\n• RGB + metric depth + instance", 11.5, C["ink"])
    pill(page, 13700, 10100, 7000, "seed → byte-identical", C["green"], C["green_tint"], 10)

    rect(page, 22600, 3500, 9500, 7700, C["white"], C["line"], 500)
    text(page, 23200, 3970, 8200, 550, "scene sampler", 14, C["blue"], True)
    scenarios = [
        ("frontal", "0–10°", C["blue"], C["blue_tint"]),
        ("moderate", "10–45°", C["green"], C["green_tint"]),
        ("oblique", "45–70°", C["orange"], C["orange_tint"]),
        ("small", "24–60 px", C["purple"], C["purple_tint"]),
        ("occluded", "partial", C["red"], C["red_tint"]),
        ("multi_tag", "3–5 IDs", C["cyan"], C["blue_tint"]),
    ]
    for index, (name, value, accent, tint) in enumerate(scenarios):
        x = 23200 + (index % 2) * 4200
        y = 5000 + (index // 2) * 1500
        rect(page, x, y, 3700, 1100, tint, None, 250)
        text(page, x + 250, y + 180, 2000, 350, name, 9.5, accent, True)
        text(page, x + 2200, y + 180, 1200, 350, value, 8.5, C["ink"], True, "RIGHT")
    text(page, 23200, 9750, 8200, 650,
         "z → pixel center → K⁻¹ backproject → pose → reject", 9.5, C["muted"], True, "CENTER")
    rect(page, 1600, 12800, 30500, 2500, C["navy"], None, 450)
    text(page, 2200, 13400, 29200, 1100,
         "camera sequence:  world tags は固定  /  T_W_C だけを滑らかに更新  /  全 frame に完全 annotation", 13, C["white"], True, "CENTER", "CENTER")
    add_footer(page, 6)


def slide_dataset(page):
    set_page(page)
    add_header(page, "DATASET CONTRACT", "RGB だけでなく、復元不能な真値を直接保存",
               "frame directory が Phase 1 → Phase 3 の明示的 wire boundary", C["orange"])
    rect(page, 1600, 3400, 9800, 6600, C["navy"], None, 500)
    text(page, 2200, 3950, 8500, 500, "frames/000000/", 15, C["cyan_light"], True, font=MONO)
    text(page, 2500, 5000, 7900, 3600,
         "├─ rgb.png\n├─ depth.npy          # float32 [m]\n├─ instance_id.png   # uint16\n└─ metadata.json", 14, C["white"], False, font=MONO)
    text(page, 2200, 8950, 8500, 400, "seed 固定で全 file を再生成", 10.5, rgb("C2D1E5"))

    rect(page, 12300, 3400, 9300, 6600, C["white"], C["line"], 500)
    text(page, 12900, 3950, 8100, 500, "metadata.json", 14, C["orange"], True)
    text(page, 12900, 4900, 8000, 4050,
         "camera\n  K / distortion / T_W_C\n\ntag[i]\n  family / id / size_m\n  T_W_M / T_C_M\n  ordered corners / depth\n  front-facing / visibility / size", 10.5, C["ink"], False, font=MONO)

    rect(page, 22500, 3400, 9600, 6600, C["white"], C["line"], 500)
    text(page, 23100, 3950, 8300, 500, "8 degradation toggles", 14, C["orange"], True)
    labels = [
        "Gaussian noise", "defocus blur", "motion blur", "brightness / contrast",
        "radial distortion", "partial occlusion", "background clutter", "rolling shutter",
    ]
    for index, label in enumerate(labels):
        x = 23100 + (index % 2) * 4100
        y = 5000 + (index // 2) * 1050
        pill(page, x, y, 3700, label, C["orange"], C["orange_tint"], 8.5)
    text(page, 23100, 9270, 8300, 350, "各因子を単独 ON / clean へ復帰可能", 9, C["muted"], False, "CENTER")

    rect(page, 1600, 11300, 30500, 4000, C["white"], C["line"], 500)
    text(page, 2200, 11800, 9000, 500, "ground truth policy", 14, C["orange"], True)
    text(page, 2200, 12650, 13200, 1350,
         "corners_px = projectPoints(T_C_M, K, D)\nrendered RGB の edge を逆算しない", 14, C["ink"], True, font=MONO)
    line(page, 16600, 11900, 16600, 14650, C["line"], 20)
    text(page, 17600, 12000, 13000, 1600,
         "radial distortion は camera model と corner に反映\nrolling shutter は単一 global pose を持たない image-space effect と明記", 11, C["muted"])
    add_footer(page, 7)


def slide_ippe(page, overlays):
    set_page(page)
    add_header(page, "PHASE 2  |  PLANAR PNP", "IPPE の二解を離散 hypothesis として保持",
               "reprojection error 最小の一解へ collapse させない", C["purple"])
    image(page, overlays["frontal"], 1600, 3450, 10240, 7680)
    text(page, 1750, 11420, 9900, 500, "frontal: planar ambiguity が重要になる条件", 9, C["muted"], False, "CENTER")

    stages = [
        (12800, "ordered corners\nK / D / L", C["blue"], C["blue_tint"]),
        (18700, "IPPE_SQUARE\nseed 0 / seed 1", C["purple"], C["purple_tint"]),
        (24600, "branch-local\nMahalanobis refine", C["orange"], C["orange_tint"]),
    ]
    for x, title_value, accent, tint in stages:
        rect(page, x, 3900, 5000, 3000, C["white"], C["line"], 420)
        rect(page, x, 3900, 80, 3000, accent, None)
        text(page, x + 300, 4700, 4400, 1250, title_value, 12, C["ink"], True, "CENTER", "CENTER")
    arrow(page, 17800, 5050)
    arrow(page, 23700, 5050)
    text(page, 12800, 7600, 17800, 500,
         "solvePnPGeneric(..., SOLVEPNP_IPPE_SQUARE)", 13, C["purple"], True, "CENTER", font=MONO)

    branch_cards = [
        (13200, 9000, "seed branch 0", "cheirality → line search\nVoronoi guard → local Hessian", C["blue"], C["blue_tint"]),
        (22600, 9000, "seed branch 1", "cheirality → line search\nseed fallback → local Hessian", C["purple"], C["purple_tint"]),
    ]
    for x, y, title_value, body, accent, tint in branch_cards:
        rect(page, x, y, 8400, 3600, tint, None, 450)
        text(page, x + 450, y + 400, 7500, 500, title_value, 13, accent, True)
        text(page, x + 450, y + 1250, 7500, 1300, body, 11, C["ink"])
        pill(page, x + 1700, y + 2800, 5000, "candidate provenance を保持", accent, C["white"], 9)
    rect(page, 12800, 14000, 17800, 1600, C["navy"], None, 350)
    text(page, 13300, 14420, 16800, 650,
         "K は通常2だが固定しない  /  duplicate・non-SPD・負 depth は明示 reject", 11.5, C["white"], True, "CENTER", "CENTER")
    add_footer(page, 8)


def slide_likelihood(page):
    set_page(page)
    add_header(page, "IMAGE LIKELIHOOD", "8×8 covariance を mode・coupling・weight まで伝播",
               "最小実装は σpix² I₈、interface は full covariance を保持", C["purple"])
    rect(page, 1600, 3400, 14500, 11900, C["navy"], None, 600)
    text(page, 2250, 3950, 12800, 500, "observation model", 14, C["cyan_light"], True)
    text(page, 2450, 5050, 12300, 4300,
         "y = (y₁,…,y₄) ∈ R⁸\nΣ_img = σpix² I₈\nσpix = 0.5 px\n\nhⱼ(x,q) = π(x + R(q)pⱼ)\nr(x,q) = h(x,q) − y", 16, C["white"], False, "LEFT", "TOP", MONO)
    line(page, 2400, 10000, 15000, 10000, C["blue_dark"], 20)
    text(page, 2450, 10800, 12300, 2500,
         "Φ(x,q) = ½ rᵀΣ_img⁻¹r\n          + ½V⁻¹ ||x−m₀||²\nV = 10⁶ m²（proper broad prior）", 14, C["cyan_light"], True, font=MONO)

    rect(page, 17500, 3400, 14600, 5500, C["white"], C["line"], 550)
    text(page, 18100, 3900, 13300, 500, "right perturbation", 14, C["purple"], True)
    text(page, 18300, 5000, 12900, 1500,
         "R(u) = Rℓ exp([u]×)\nx = mℓ + δx", 19, C["ink"], True, "CENTER", "CENTER", MONO)
    text(page, 18300, 7050, 12900, 800,
         "rotation update:  R ← R · exp([δu]×)", 11, C["muted"], False, "CENTER", font=MONO)

    rect(page, 17500, 9700, 14600, 5600, C["white"], C["line"], 550)
    text(page, 18100, 10200, 13300, 500, "refinement policy", 14, C["purple"], True)
    rows = [
        ("pinhole / D=0", "analytic right-chart Jacobian", C["green"]),
        ("distortion ≠ 0", "central finite difference", C["orange"]),
        ("both branches", "same Φ + line search + guard", C["purple"]),
        ("invalid local law", "reject with diagnostics", C["red"]),
    ]
    for index, (left, right, accent) in enumerate(rows):
        y = 11200 + index * 850
        ellipse(page, 18300, y, 420, 420, accent)
        text(page, 19000, y - 30, 4300, 400, left, 9.5, C["ink"], True)
        text(page, 23200, y - 30, 7900, 400, right, 9.5, C["muted"])
    add_footer(page, 9)


def slide_hessian(page):
    set_page(page)
    add_header(page, "LOCAL HESSIAN", "pixel uncertainty を 6-DoF joint precision へ",
               "right perturbationの符号と block ordering [δx; u] を固定", C["purple"])
    rect(page, 1600, 3400, 11700, 11800, C["white"], C["line"], 550)
    text(page, 2200, 3950, 10400, 500, "image Jacobian", 14, C["purple"], True)
    text(page, 2400, 5200, 9800, 2900,
         "J_x,j = J_π,j\n\nJ_u,j = −J_π,j Rℓ [pⱼ]×\n\nJ = [ J_x   J_u ] ∈ R⁸ˣ⁶", 16, C["ink"], True, "CENTER", "CENTER", MONO)
    rect(page, 2600, 9300, 9700, 2500, C["purple_tint"], None, 350)
    text(page, 3000, 9850, 8900, 1150,
         "Σ_img = LLᵀ\nr_w=L⁻¹r,  J_w=L⁻¹J", 13, C["purple"], True, "CENTER", "CENTER", MONO)
    text(page, 2600, 12750, 9700, 1150,
         "whitening → correlation / anisotropy を\n全 block に反映", 11, C["muted"], False, "CENTER")

    rect(page, 14500, 3400, 17600, 11800, C["navy"], None, 600)
    text(page, 15100, 3950, 16300, 500, "Gauss–Newton precision", 14, C["cyan_light"], True)
    text(page, 15300, 5050, 15900, 1700,
         "Hℓ = Jᵀ Σ_img⁻¹ J + H_prior", 19, C["white"], True, "CENTER", "CENTER", MONO)
    rect(page, 16500, 7400, 13600, 4000, C["blue_dark"], rgb("3A5577"), 400)
    text(page, 17000, 7900, 12600, 2700,
         "       ┌ H_xx   H_xu ┐\nHℓ  =  │               │\n       └ H_ux   H_uu ┘\n          3×3     3×3", 17, C["white"], True, "CENTER", "CENTER", MONO)
    pill(page, 16400, 12400, 6200, "symmetrize", C["cyan"], C["blue_dark"], 10)
    pill(page, 23700, 12400, 6200, "Cholesky / solve", C["green"], C["green_tint"], 10)
    text(page, 15800, 14000, 15000, 600,
         "H_xx または Schur precision が非 SPD → publish しない", 11, C["red_tint"], True, "CENTER")
    add_footer(page, 10)


def slide_conditional(page):
    set_page(page)
    add_header(page, "HESSIAN → DISTRIBUTION", "S・B・Λ・C は同じ局所 posterior の分解",
               "translation–rotation coupling を捨てずに ProbTF の公開表現へ", C["green"])
    equations = [
        ("Sℓ = H_xx⁻¹", "姿勢 u を固定した後の conditional residual covariance", C["blue"], C["blue_tint"]),
        ("Bℓ = −H_xx⁻¹ H_xu", "回転が変わると条件付き translation mean がどう動くか", C["purple"], C["purple_tint"]),
        ("Λℓ = H_uu − H_ux ·\n       H_xx⁻¹ H_xu", "translation を周辺化した rotation precision", C["green"], C["green_tint"]),
    ]
    for index, (formula, body, accent, tint) in enumerate(equations):
        y = 3400 + index * 3000
        rect(page, 1600, y, 13900, 2500, C["white"], C["line"], 420)
        rect(page, 1600, y, 80, 2500, accent, None)
        text(page, 2150, y + 330, 6400, 1050, formula,
             12.5 if index == 2 else 15, accent, True, font=MONO)
        text(page, 8350, y + 350, 6400, 1150, body, 10.5, C["muted"], False, "LEFT", "CENTER")
    rect(page, 1600, 12900, 13900, 2300, C["red_tint"], rgb("F3A6A6"), 380)
    text(page, 2100, 13450, 12900, 1000,
         "S は translation の marginal covariance ではない\nQ を周辺化すると coupling が位置 law へ寄与", 11.5, C["red"], True, "CENTER")

    rect(page, 16900, 3400, 15200, 11800, C["navy"], None, 550)
    text(page, 17500, 3950, 13900, 500, "local u → public vec(R) coupling", 14, C["cyan_light"], True)
    text(page, 17800, 5050, 13300, 3300,
         "D_rot = [ vec(R[e₁]×)\n          vec(R[e₂]×)\n          vec(R[e₃]×) ]\n\nD_rotᵀD_rot = 2I", 14, C["white"], True, "CENTER", "CENTER", MONO)
    rect(page, 18600, 9000, 11800, 1700, C["blue_dark"], rgb("3A5577"), 300)
    text(page, 19000, 9420, 11000, 750,
         "Cℓ = Bℓ D_rot⁺ = ½ Bℓ D_rotᵀ", 16, C["cyan_light"], True, "CENTER", "CENTER", MONO)
    text(page, 17900, 11700, 13200, 1900,
         "X | Q=q,L=ℓ  ~  N(\n  mℓ + Cℓ(vec R(q) − ρℓ), Sℓ )", 14, C["white"], True, "CENTER", "CENTER", MONO)
    pill(page, 20100, 14200, 8600, "ρℓ = vec R(qℓ) は導出値", C["green"], C["green_tint"], 10)
    add_footer(page, 11)


def slide_bingham_weight(page):
    set_page(page)
    add_header(page, "ORIENTATION & MIXTURE MASS", "Schur precision を Bingham、局所 mass を weight へ",
               "mode の高さだけでなく Hessian volume も考慮", C["green"])
    rect(page, 1600, 3400, 14600, 11800, C["white"], C["line"], 550)
    text(page, 2200, 3950, 13300, 500, "Bingham orientation", 14, C["green"], True)
    text(page, 2400, 5050, 12900, 4400,
         "qℓ(u) = qℓ ⊗ δq(u)\nEℓ = 2 ∂qℓ(u)/∂u |₀\nEℓᵀEℓ = I,   Eℓᵀqℓ = 0\n\nAℓ⁰ = −2 Eℓ Λℓ Eℓᵀ", 15, C["ink"], True, "CENTER", "CENTER", MONO)
    rect(page, 2700, 10300, 11900, 1700, C["green_tint"], None, 300)
    text(page, 3100, 10700, 11100, 800,
         "Aℓ = Aℓ⁰ − tr(Aℓ⁰)/4 · I₄", 14, C["green"], True, "CENTER", "CENTER", MONO)
    text(page, 2600, 13000, 12200, 1050,
         "trace-zero gauge / antipodal q≡−q\nlocal curvature = ½ uᵀΛu", 11, C["muted"], False, "CENTER")

    rect(page, 17500, 3400, 14600, 11800, C["navy"], None, 550)
    text(page, 18100, 3950, 13300, 500, "Laplace component weight", 14, C["cyan_light"], True)
    text(page, 18300, 5200, 12900, 2700,
         "log aℓ = −Φℓ\n          − ½ log det H_xx,ℓ\n          − ½ log det Λℓ", 17, C["white"], True, "CENTER", "CENTER", MONO)
    arrow(page, 23800, 8200, C["cyan"])
    rect(page, 19100, 9300, 11400, 1800, C["blue_dark"], rgb("3A5577"), 300)
    text(page, 19500, 9750, 10600, 800,
         "wℓ = softmax(log aℓ)", 18, C["cyan_light"], True, "CENTER", "CENTER", MONO)
    text(page, 18500, 12100, 12600, 1300,
         "✓ peak height  −Φℓ\n✓ local spread  det(H_xx) det(Λ)", 12, C["white"])
    pill(page, 19300, 14000, 10900, "reprojection error だけでは決めない", C["orange"], C["orange_tint"], 10)
    add_footer(page, 12)


def slide_probtf_mapping(page):
    set_page(page)
    add_header(page, "PROBTF v2 MAPPING", "局所 Laplace mixture を native edge record へ",
               "full stamped distribution・provenance・lossy approximation を外部化", C["green"])
    rect(page, 1500, 3400, 11200, 11900, C["navy"], None, 650)
    text(page, 2150, 3950, 9900, 500, "joint physical action", 13, C["cyan_light"], True)
    text(page, 2300, 5000, 9600, 1550,
         "z_C = R(Q) z_M + X", 20, C["white"], True, "CENTER", "CENTER", MONO)
    text(page, 2300, 7300, 9600, 3500,
         "L ~ Categorical(w₁,…,w_K)\nQ | L=ℓ ~ Bingham(Aℓ)\nX | Q,L=ℓ ~ conditional Gaussian", 13.5, rgb("DCE8F8"), False, "CENTER", "CENTER", MONO)
    line(page, 2500, 11600, 11700, 11600, C["blue_dark"], 20)
    text(page, 2300, 12500, 9600, 1350,
         "parent=camera optical\nchild=apriltag_<id>\nT_C_M / dynamic edge", 11, rgb("AFC3DD"), False, "CENTER")

    mappings = [
        ("BinghamOrientation", "Aℓ + reference quaternion qℓ", C["blue"], C["blue_tint"]),
        ("ConditionalGaussianTranslation", "mean_at_reference=mℓ / residual Sℓ / coupling Cℓ", C["cyan"], C["blue_tint"]),
        ("TransformComponent", "raw_weight=wℓ / component provenance / approximation", C["purple"], C["purple_tint"]),
        ("TransformDistributionStamped", "frames / stamp / edge_id / authority / is_static=false", C["green"], C["green_tint"]),
    ]
    for index, (title_value, body, accent, tint) in enumerate(mappings):
        y = 3400 + index * 2850
        rect(page, 14100, y, 18000, 2350, C["white"], C["line"], 420)
        rect(page, 14100, y, 80, 2350, accent, None)
        ellipse(page, 14600, y + 520, 720, 720, tint)
        text(page, 14600, y + 610, 720, 420, str(index + 1), 12, accent, True, "CENTER", "CENTER")
        text(page, 15700, y + 360, 15400, 500, title_value, 13, C["ink"], True)
        text(page, 15700, y + 1130, 15400, 700, body, 10, C["muted"])
    pill(page, 15700, 15100, 14800,
         "ApproximationKind.TANGENT_SURROGATE / lossy=true", C["orange"], C["orange_tint"], 9.5)
    add_footer(page, 13)


def slide_ros(page, overlays):
    set_page(page)
    add_header(page, "ROS 1 INTEGRATION", "thin wrapper：CameraInfo + Image → /probtf",
               "debug overlay は inspection 専用で estimator へ feedback しない", C["green"])
    image(page, overlays["multi_tag"], 22000, 3800, 9600, 7200)
    text(page, 22200, 11200, 9200, 450, "multi_tag: all refined mode axes + ordered corners", 8.5, C["muted"], False, "CENTER")

    graph = [
        (1700, 4000, 4300, "sensor_msgs/Image", "~image", C["blue"]),
        (1700, 7100, 4300, "CameraInfo", "~camera_info", C["blue"]),
        (8000, 5000, 6500, "prob_artag_detector", "detect → estimate → convert", C["purple"]),
        (16400, 3900, 4200, "ProbTF v2", "/probtf", C["green"]),
        (16400, 7600, 4200, "debug Image", "~debug_image", C["orange"]),
    ]
    for x, y, w, title_value, body, accent in graph:
        rect(page, x, y, w, 2100, C["white"], accent, 350, line_width=30)
        text(page, x + 250, y + 350, w - 500, 500, title_value, 11, C["ink"], True, "CENTER")
        text(page, x + 250, y + 1150, w - 500, 400, body, 9, C["muted"], False, "CENTER")
    arrow(page, 6500, 5200)
    arrow(page, 6500, 7550)
    arrow(page, 14900, 5200, C["green"])
    arrow(page, 14900, 7900, C["orange"])

    rect(page, 1600, 12300, 9600, 3400, C["white"], C["line"], 420)
    text(page, 2150, 12750, 8500, 450, "主要 parameter", 12, C["green"], True)
    text(page, 2150, 13500, 8500, 1600,
         "family / tag_size_m / corner_sigma_px\nmax_iterations / finite_difference_step\ntranslation_prior_variance / spd_tolerance", 9.5, C["ink"], font=MONO)
    rect(page, 12100, 12300, 9300, 3400, C["white"], C["line"], 420)
    text(page, 12650, 12750, 8200, 450, "runtime guard", 12, C["green"], True)
    text(page, 12650, 13500, 8200, 1600,
         "CameraInfo 待ち / processing lock\n重複 frame は drop / invalid edge は reject\nwarn throttle で運用状態を可視化", 9.8, C["ink"])
    rect(page, 22200, 12300, 9400, 3400, C["navy"], None, 420)
    text(page, 22750, 12750, 8300, 450, "debug axes", 12, C["cyan_light"], True)
    text(page, 22750, 13500, 8300, 1500,
         "green: detected quad\naxes: 全 accepted mode\nlabel: ID / mode count / weights", 10, C["white"])
    add_footer(page, 14)


def slide_benchmark(page, metrics, overlays):
    set_page(page)
    add_header(page, "PHASE 3  |  BENCHMARK", "render → detect → estimate → compare を閉ループ化",
               "renderer packageをimportせず、dataset filesをwire boundaryとして読む", C["orange"])
    image(page, overlays["oblique"], 22000, 3550, 9600, 7200)
    text(page, 22200, 10950, 9200, 420, "oblique fixture: GT cyan / detection magenta / all seed axes", 8.5, C["muted"], False, "CENTER")

    flow = [
        (1600, "Phase 1 dataset", "rgb.png\nmetadata.json", C["blue"], C["blue_tint"]),
        (8200, "benchmark", "associate ID\ncompare corners / poses", C["orange"], C["orange_tint"]),
        (14800, "artifacts", "JSON / CSV\noverlays", C["green"], C["green_tint"]),
    ]
    for x, title_value, body, accent, tint in flow:
        rect(page, x, 3900, 5400, 3300, C["white"], C["line"], 420)
        rect(page, x, 3900, 80, 3300, accent, None)
        text(page, x + 300, 4400, 4800, 500, title_value, 12, C["ink"], True, "CENTER")
        text(page, x + 300, 5350, 4800, 1000, body, 10, C["muted"], False, "CENTER")
    arrow(page, 7000, 5200)
    arrow(page, 13600, 5200)

    rect(page, 1600, 8200, 19200, 3400, C["white"], C["line"], 450)
    text(page, 2150, 8650, 18000, 450, "評価 fixture", 12.5, C["orange"], True)
    fixture = metrics["fixture"]
    text(page, 2150, 9450, 17800, 1500,
         "family={}  /  L=0.12 m  /  seed={}  /  σpix={} px\n640×480  /  fx=fy=600 px  /  clean  /  near: t≤{:.2f} m & R≤{:.0f}°".format(
             fixture["family"], fixture["seed"], fixture["corner_sigma_px"],
             fixture["gt_near_translation_threshold_m"], fixture["gt_near_rotation_threshold_deg"]
         ), 10, C["ink"], font=MONO)

    rect(page, 1600, 12600, 30500, 3000, C["navy"], None, 450)
    definitions = [
        ("recall", "associated detection / GT"),
        ("corner RMSE", "ordered 4-corner pixel error"),
        ("near IPPE", "二 seed 内に GT 近傍があるか"),
        ("nearest mode", "閾値正規化距離の oracle"),
        ("gap", "IPPE 二解の reprojection RMSE 差"),
    ]
    for index, (name, body) in enumerate(definitions):
        x = 2100 + index * 5900
        text(page, x, 13100, 5200, 400, name, 10.5, C["cyan_light"], True, "CENTER")
        text(page, x, 13800, 5200, 850, body, 8.5, C["white"], False, "CENTER")
    determinism = metrics["determinism"]
    pill(
        page, 21900, 11750, 9500,
        "{} files byte-identical  |  SHA {}…".format(
            determinism["file_count"], determinism["tree_sha256"][:10]
        ),
        C["green"], C["green_tint"], 8.7,
    )
    add_footer(page, 15)


def _pct(value):
    return "—" if value is None else "{:.0f}%".format(100.0 * float(value))


def _number(value, digits=3, scale=1.0, suffix=""):
    return "—" if value is None else ("{:0." + str(digits) + "f}{}").format(float(value) * scale, suffix)


def slide_results(page, metrics):
    set_page(page)
    add_header(page, "QUANTITATIVE RESULTS", "clean synthetic：遮蔽なし 42 / 42 を正しい ID で検出",
               "accepted mode と rejected seed を混ぜず、scenario 別に集計", C["green"])
    scenarios = metrics["scenarios"]
    non_occluded = [value for name, value in scenarios.items() if name != "occluded"]
    detected = sum(item["correct_id_count"] for item in non_occluded)
    gt_count = sum(item["ground_truth_count"] for item in non_occluded)
    false_positives = sum(item["false_positive_count"] for item in scenarios.values())
    multi = scenarios["multi_tag"]
    metric_card(page, 1600, 3000, 6800, "遮蔽なし correct ID", "{} / {}".format(detected, gt_count), C["green"], C["green_tint"], "recall = 100%")
    metric_card(page, 9000, 3000, 6800, "false positive", str(false_positives), C["blue"], C["blue_tint"], "全6条件")
    metric_card(page, 16400, 3000, 6800, "multi_tag", "{} / {}".format(multi["correct_id_count"], multi["ground_truth_count"]), C["purple"], C["purple_tint"], "10 frames × 3 tags")
    determinism = metrics["determinism"]
    metric_card(
        page, 23800, 3000, 8300, "determinism", "byte-identical",
        C["orange"], C["orange_tint"],
        "{} files  |  SHA {}…".format(
            determinism["file_count"], determinism["tree_sha256"][:10]
        ),
    )

    order = ("frontal", "moderate", "oblique", "small", "multi_tag", "occluded")
    headers = ("condition", "GT", "recall", "ID", "corner RMSE\nmean px", "nearest t\nmean mm", "nearest R\nmean deg", "near /\n2-seed", "gap\nmean px")
    widths = (4100, 2100, 2700, 2400, 3500, 3500, 3500, 3100, 3300)
    x0 = 1700
    y0 = 6000
    row_h = 1350
    x = x0
    for header, width in zip(headers, widths):
        rect(page, x, y0, width, row_h, C["navy"], C["white"], 0, line_width=10)
        text(page, x + 100, y0 + 270, width - 200, 700, header, 8.5, C["white"], True, "CENTER", "CENTER")
        x += width
    for row_index, name in enumerate(order):
        item = scenarios[name]
        y = y0 + (row_index + 1) * row_h
        fill = C["red_tint"] if name == "occluded" else (C["white"] if row_index % 2 == 0 else C["slate_tint"])
        values = (
            name,
            str(item["ground_truth_count"]),
            _pct(item["recall"]),
            _pct(item["id_accuracy_on_associated"]),
            _number(item["corner_rmse_px_mean"], 3),
            _number(item["nearest_mode_translation_error_m_mean"], 1, 1000.0),
            _number(item["nearest_mode_rotation_error_deg_mean"], 3),
            (
                "—"
                if item["two_ippe_candidate_tag_count"] == 0
                else "{} / {}".format(
                    item["gt_near_ippe_candidate_count"],
                    item["two_ippe_candidate_tag_count"],
                )
            ),
            _number(item["ippe_reprojection_rmse_gap_px_mean"], 3),
        )
        x = x0
        for column, (value, width) in enumerate(zip(values, widths)):
            rect(page, x, y, width, row_h, fill, C["line"], 0, line_width=10)
            color = C["red"] if name == "occluded" else (C["ink"] if column == 0 else C["muted"])
            text(page, x + 80, y + 350, width - 160, 520, value, 8.8, color, column == 0, "CENTER", "CENTER")
            x += width
    text(page, 1800, 16000, 30000, 500,
         "nearest mode = GT proximityを使うoracle指標。二解平均を単一 pose 精度として扱っていない。", 9, C["muted"])
    text(page, 1800, 16500, 30000, 430,
         "IPPE 2-candidate coverage: 遮蔽なし 42/42 = 100%  /  occluded 0/3 = 0%（near は条件付きのため —）",
         8.5, C["muted"])
    add_footer(page, 16)


def slide_weak_depth(page, metrics, overlays):
    set_page(page)
    add_header(page, "OBSERVABILITY", "小さい平面 tag は clean image でも depth が弱い",
               "1 px 前後の corner error が並進 z へ大きく増幅", C["orange"])
    image(page, overlays["frontal"], 1600, 3400, 8000, 6000)
    image(page, overlays["small"], 10500, 3400, 8000, 6000)
    text(page, 1700, 9700, 7800, 380, "frontal / projected edge ≥ 50 px group", 8.5, C["muted"], False, "CENTER")
    text(page, 10600, 9700, 7800, 380, "small / projected edge ≈ 29–40 px", 8.5, C["muted"], False, "CENTER")

    derived = metrics["derived"]["metrics"]
    small = metrics["scenarios"]["small"]
    rect(page, 19600, 3400, 12500, 6600, C["white"], C["line"], 500)
    text(page, 20200, 3900, 11200, 500, "scale comparison", 14, C["orange"], True)
    text(page, 20200, 5000, 4300, 430, "edge ≥ 50 px", 10, C["green"], True)
    text(page, 24700, 5000, 6500, 430,
         "{} tags".format(derived["tag_count"]), 10, C["ink"], True, "RIGHT")
    text(page, 20200, 5700, 11000, 1500,
         "corner {:.3f} px  /  t {:.2f} mm  /  R {:.3f}°".format(
             derived["corner_rmse_px_mean"],
             1000.0 * derived["nearest_mode_translation_error_m_mean"],
             derived["nearest_mode_rotation_error_deg_mean"],
         ), 11, C["ink"], True)
    line(page, 20200, 7550, 31200, 7550, C["line"], 18)
    text(page, 20200, 8050, 4300, 430, "small", 10, C["orange"], True)
    text(page, 20200, 8700, 11000, 900,
         "corner {:.3f} px  /  t {:.1f} mm  /  R {:.3f}°".format(
             small["corner_rmse_px_mean"],
             1000.0 * small["nearest_mode_translation_error_m_mean"],
             small["nearest_mode_rotation_error_deg_mean"],
         ), 11, C["ink"], True)

    rect(page, 1600, 11200, 30500, 4300, C["navy"], None, 500)
    text(page, 2200, 11700, 9100, 500, "幾何的な理由", 13, C["cyan_light"], True)
    text(page, 2200, 12600, 9100, 1100,
         "projected size ℓ ≈ fL/Z\nδZ ≈ Z²/(fL) · δℓ", 16, C["white"], True, "CENTER", "CENTER", MONO)
    line(page, 12400, 11900, 12400, 14600, C["blue_dark"], 20)
    text(page, 13400, 11700, 17000, 500, "実画像へ持ち込むとき", 13, C["cyan_light"], True)
    text(page, 13400, 12600, 17000, 1400,
         "1  projected-size gate\n2  複数 frame の ProbTF fusion\n3  IMU / depth / multi-tag constraint", 12, C["white"])
    add_footer(page, 17)


def slide_occlusion_limits(page, metrics, overlays):
    set_page(page)
    add_header(page, "FAILURE BOUNDARY", "強い遮蔽：0 / 3 検出、ただし安全に missed と記録",
               "recall 達成ではなく、検出不能入力で node / benchmark が壊れない確認", C["red"])
    image(page, overlays["occluded"], 1600, 3450, 10240, 7680)
    occluded = metrics["scenarios"]["occluded"]
    metric_card(page, 12900, 3500, 5500, "detections", str(occluded["detection_count"]), C["red"], C["red_tint"])
    metric_card(page, 19100, 3500, 5500, "missed", str(occluded["missed_count"]), C["orange"], C["orange_tint"])
    metric_card(page, 25300, 3500, 6500, "frame errors", str(occluded["error_frame_count"]), C["green"], C["green_tint"])

    rect(page, 12900, 6400, 18900, 4700, C["white"], C["line"], 450)
    text(page, 13500, 6900, 17600, 500, "graceful handling", 13, C["red"], True)
    text(page, 13500, 7850, 17600, 2100,
         "• no corners → no IPPE → no invalid ProbTF edge\n• false positive = 0\n• report row は null / missed を明示\n• overlay 生成も例外なし", 11.5, C["ink"])

    rect(page, 1600, 12400, 30500, 3300, C["navy"], None, 450)
    text(page, 2200, 12850, 5200, 430, "未評価 / limits", 12, C["cyan_light"], True)
    limits = [
        "実 camera calibration・printer・optics",
        "複合 degradation 下の recall",
        "mixture weight の coverage / NLL",
        "連続時系列の graph fusion",
    ]
    for index, label in enumerate(limits):
        x = 2200 + (index % 2) * 14500
        y = 13600 + (index // 2) * 900
        ellipse(page, x, y, 430, 430, C["orange"])
        text(page, x + 700, y - 20, 12800, 500, label, 10.5, C["white"])
    add_footer(page, 18)


def slide_reproduce(page):
    set_page(page)
    continuation = " " + chr(92) + "\n"
    dataset_commands = (
        "prob-artag-generate-dataset --output /tmp/prob-artag/frontal/dataset"
        + continuation
        + "  --seed 2 --scenario frontal --frames 3 --overwrite\n"
        + "prob-artag-benchmark /tmp/prob-artag/frontal/dataset"
        + continuation
        + "  /tmp/prob-artag/frontal/report"
    )
    deck_command = (
        "/usr/bin/python3 docs/slides/generate_prob_artag_detector_slides.py"
        + continuation
        + "  --output docs/slides/prob_artag_detector_implementation_overview_ja.pptx"
    )
    add_header(page, "REPRODUCE & HANDOFF", "同じ seed・config・wire boundary から再生成",
               "dataset → benchmark → aggregate → slides を command line で追跡可能", C["blue"])
    rect(page, 1600, 3400, 19600, 11500, C["navy"], None, 520)
    text(page, 2200, 3900, 18400, 450, "1  environment", 12, C["cyan_light"], True)
    text(page, 2300, 4600, 17900, 1600,
         "source /home/leus/catkin_ws/devel/setup.bash\nexport PYOPENGL_PLATFORM=egl\n"
         "export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json",
         8.4, C["white"], font=MONO)
    text(page, 2200, 6700, 18400, 450, "2  dataset + benchmark", 12, C["cyan_light"], True)
    text(page, 2300, 7400, 17900, 2400,
         dataset_commands, 9.2, C["white"], font=MONO)
    text(page, 2200, 10500, 18400, 450, "3  regenerate this deck", 12, C["cyan_light"], True)
    text(page, 2300, 11200, 17900, 1800,
         deck_command, 8.5, C["white"], font=MONO)
    pill(page, 4300, 13600, 14200, "PPTX round-trip + optional PDF export", C["green"], C["green_tint"], 10)

    rect(page, 22500, 3400, 9600, 3100, C["white"], C["line"], 450)
    text(page, 23100, 3900, 8300, 450, "inputs", 13, C["blue"], True)
    text(page, 23100, 4750, 8300, 1200,
         "Phase 1 / 2 source\nprob_artag_phase3_evaluation_ja.md\n"
         "prob_artag_phase3_metrics.json\noverlays/*.png", 8, C["ink"], font=MONO)
    rect(page, 22500, 7100, 9600, 7800, C["white"], C["line"], 450)
    text(page, 23100, 7600, 8300, 450, "takeaways", 13, C["blue"], True)
    takeaways = [
        ("1", "座標・corner order を先に固定", C["blue"]),
        ("2", "IPPE branch と coupling を保持", C["purple"]),
        ("3", "distribution を native ProbTF edge へ", C["green"]),
        ("4", "弱観測と遮蔽を数値で境界化", C["orange"]),
    ]
    for index, (num, body, accent) in enumerate(takeaways):
        y = 8650 + index * 1350
        ellipse(page, 23200, y, 700, 700, accent)
        text(page, 23200, y + 90, 700, 430, num, 12, C["white"], True, "CENTER", "CENTER")
        text(page, 24200, y + 40, 6900, 700, body, 10.5, C["ink"], True, "LEFT", "CENTER")
    add_footer(page, 19)


def _resolve_desktop(target, attempts):
    local = uno.getComponentContext()
    resolver = local.ServiceManager.createInstanceWithContext(
        "com.sun.star.bridge.UnoUrlResolver", local
    )
    error = None
    for _ in range(attempts):
        try:
            context = resolver.resolve(target)
            return context.ServiceManager.createInstanceWithContext(
                "com.sun.star.frame.Desktop", context
            )
        except Exception as exc:
            error = exc
            time.sleep(0.1)
    raise RuntimeError("Cannot connect to LibreOffice UNO: {}".format(error))


def connect_or_launch(port):
    socket_target = (
        "uno:socket,host=127.0.0.1,port={};urp;StarOffice.ComponentContext".format(port)
    )
    try:
        return _resolve_desktop(socket_target, 3), None, None
    except RuntimeError:
        pass
    executable = shutil.which("libreoffice") or shutil.which("soffice")
    if not executable:
        raise RuntimeError("LibreOffice executable was not found")
    temporary_root = tempfile.TemporaryDirectory(prefix="prob-artag-slides-")
    profile_path = Path(temporary_root.name) / "lo-profile"
    pipe_name = "prob_artag_slides_{}_{}".format(os.getpid(), port)
    pipe_target = "uno:pipe,name={};urp;StarOffice.ComponentContext".format(pipe_name)
    command = [
        executable, "--headless", "--nologo", "--nodefault", "--nolockcheck",
        "--norestore", "--nofirststartwizard",
        "-env:UserInstallation={}".format(
            uno.systemPathToFileUrl(str(profile_path))
        ),
        "--accept=pipe,name={};urp;StarOffice.ServiceManager".format(pipe_name),
    ]
    environment = os.environ.copy()
    environment.setdefault("SAL_USE_VCLPLUGIN", "svp")
    process = subprocess.Popen(
        command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=environment
    )
    try:
        desktop = _resolve_desktop(pipe_target, 120)
    except Exception:
        process.terminate()
        temporary_root.cleanup()
        raise
    return desktop, process, temporary_root


def build_deck(desktop, metrics, overlays, output_path, pdf_path=None):
    hidden = (prop("Hidden", True),)
    doc = desktop.loadComponentFromURL("private:factory/simpress", "_blank", 0, hidden)
    try:
        attach_document(doc)
        pages = doc.getDrawPages()
        while pages.getCount() > 1:
            pages.remove(pages.getByIndex(pages.getCount() - 1))
        while pages.getCount() < SLIDE_COUNT:
            pages.insertNewByIndex(pages.getCount())
        builders = [
            lambda p: slide_title(p),
            lambda p: slide_phases(p),
            lambda p: slide_architecture(p),
            lambda p: slide_coordinates(p),
            lambda p: slide_corner_order(p),
            lambda p: slide_renderer(p, overlays),
            lambda p: slide_dataset(p),
            lambda p: slide_ippe(p, overlays),
            lambda p: slide_likelihood(p),
            lambda p: slide_hessian(p),
            lambda p: slide_conditional(p),
            lambda p: slide_bingham_weight(p),
            lambda p: slide_probtf_mapping(p),
            lambda p: slide_ros(p, overlays),
            lambda p: slide_benchmark(p, metrics, overlays),
            lambda p: slide_results(p, metrics),
            lambda p: slide_weak_depth(p, metrics, overlays),
            lambda p: slide_occlusion_limits(p, metrics, overlays),
            lambda p: slide_reproduce(p),
        ]
        for index, builder in enumerate(builders):
            builder(pages.getByIndex(index))
        properties = doc.getDocumentProperties()
        properties.Title = "確率的 AprilTag Detector 実装説明資料"
        properties.Subject = "Renderer, planar pose mixture, ProbTF v2, ROS, and benchmark"
        properties.Author = "ProbTF-demo"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        doc.storeAsURL(
            uno.systemPathToFileUrl(str(output_path.resolve())),
            (prop("FilterName", "Impress MS PowerPoint 2007 XML"), prop("Overwrite", True)),
        )
    finally:
        doc.close(True)

    reopened = desktop.loadComponentFromURL(
        uno.systemPathToFileUrl(str(output_path.resolve())), "_blank", 0, hidden
    )
    try:
        pages = reopened.getDrawPages()
        if pages.getCount() != SLIDE_COUNT:
            raise AssertionError("Expected {} slides, got {}".format(SLIDE_COUNT, pages.getCount()))
        for index in range(SLIDE_COUNT):
            page = pages.getByIndex(index)
            if page.Width != PAGE_W or page.Height != PAGE_H:
                raise AssertionError(
                    "slide {} size {}x{} != {}x{}".format(
                        index + 1, page.Width, page.Height, PAGE_W, PAGE_H
                    )
                )
        if pdf_path is not None:
            pdf_path.parent.mkdir(parents=True, exist_ok=True)
            reopened.storeToURL(
                uno.systemPathToFileUrl(str(pdf_path.resolve())),
                (prop("FilterName", "impress_pdf_Export"), prop("Overwrite", True)),
            )
    finally:
        reopened.close(True)
    with zipfile.ZipFile(str(output_path), "r") as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise AssertionError("PPTX ZIP CRC failure: {}".format(bad_member))
        names = set(archive.namelist())
        if "ppt/slides/slide19.xml" not in names:
            raise AssertionError("PPTX does not contain slide19.xml")
    print("roundtrip OK: slides={} size={}x{} output={}".format(
        SLIDE_COUNT, PAGE_W, PAGE_H, output_path
    ))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().with_name(
            "prob_artag_detector_implementation_overview_ja.pptx"
        ),
    )
    parser.add_argument("--pdf", type=Path, help="optional PDF verification export")
    parser.add_argument("--port", type=int, default=2083)
    return parser.parse_args()


def main():
    args = parse_args()
    with METRICS_PATH.open("r", encoding="utf-8") as stream:
        metrics = json.load(stream)
    overlays = {
        name: OVERLAY_DIR / (name + ".png")
        for name in ("frontal", "moderate", "oblique", "small", "occluded", "multi_tag")
    }
    missing = [str(path) for path in overlays.values() if not path.is_file()]
    if missing:
        raise RuntimeError("missing overlay inputs: {}".format(", ".join(missing)))
    desktop = process = profile = None
    try:
        desktop, process, profile = connect_or_launch(args.port)
        build_deck(desktop, metrics, overlays, args.output, args.pdf)
    finally:
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        if profile is not None:
            profile.cleanup()


if __name__ == "__main__":
    main()
