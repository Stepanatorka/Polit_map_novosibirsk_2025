import sys
import os
import threading
import queue
from PyQt5 import QtCore, QtGui, QtWidgets
from PIL import Image, ImageEnhance
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

# ---------------------------
# Данные по районам (1..38)
# ---------------------------
districts_data = [
    (1, 50256, 47893, 21581, 6449, 19863, 6449, 21581, 843, 27187, 0, 0,
     1330, 4.74, 3275, 11.68, 1114, 3.97, 152, 0.54, 1866, 6.66, 173, 0.62, 2947, 10.51, 16330, 58.26),
    (2, 50445, 49530, 17770, 5633, 26127, 5633, 17751, 690, 22694, 0, 0,
     1355, 5.79, 2598, 11.11, 1104, 4.72, 131, 0.56, 1466, 6.27, 115, 0.49, 2745, 11.74, 13180, 56.36),
    (3, 50874, 48930, 18857, 4766, 25307, 4766, 18854, 622, 21791, 0, 0,
     2171, 9.60, 1977, 8.75, 1511, 6.68, 131, 0.58, 1481, 6.55, 108, 0.48, 2486, 10.99, 11926, 52.76),
    (4, 50700, 49410, 16927, 4379, 28104, 4379, 16880, 563, 22998, 0, 0,
     5340, 22.61, 2377, 10.06, 1062, 4.50, 134, 0.57, 1569, 6.55, 112, 0.47, 1974, 8.36, 10430, 44.16),
    (5, 50855, 44100, 19187, 6735, 18178, 6735, 19186, 597, 20696, 0, 0,
     1288, 6.06, 2374, 11.17, 895, 4.21, 158, 0.74, 1797, 8.45, 118, 0.56, 2188, 10.29, 11878, 55.87),
    (6, 50887, 58733, 20629, 1737, 36368, 1737, 20494, 854, 25324, 0, 1,
     2037, 7.86, 2648, 10.22, 1139, 4.39, 147, 0.57, 2367, 9.13, 143, 0.55, 2399, 9.26, 14444, 55.72),
    (7, 62489, 46990, 17889, 7544, 21557, 7544, 17805, 843, 21377, 0, 0,
     2371, 10.67, 1894, 8.52, 1858, 8.36, 223, 1.00, 2518, 11.33, 140, 0.63, 1856, 8.35, 10517, 47.31),
    (8, 50225, 50403, 19515, 4392, 26496, 4392, 19507, 862, 24506, 0, 0,
     1141, 4.50, 2999, 11.83, 917, 3.62, 139, 0.55, 1653, 6.52, 136, 0.54, 2591, 10.22, 14930, 58.90),
    (9, 52746, 52400, 19515, 4392, 26496, 4392, 19507, 862, 23037, 0, 0,
     3971, 16.62, 2444, 10.23, 1002, 4.19, 172, 0.72, 1604, 6.71, 127, 0.53, 2037, 8.52, 11680, 48.87),
    (10, 54936, 52400, 14580, 3868, 33952, 3868, 14558, 795, 17631, 0, 0,
     1042, 5.66, 2342, 12.71, 2106, 11.43, 158, 0.86, 1231, 6.68, 136, 0.74, 1660, 9.01, 8956, 48.61),
    (11, 51370, 47100, 14405, 552, 32143, 552, 14383, 614, 14321, 0, 0,
     1249, 8.36, 2310, 15.47, 827, 5.54, 91, 0.61, 934, 6.25, 55, 0.37, 1932, 12.94, 6923, 46.35),
    (12, 51730, 46300, 13419, 1167, 31714, 1167, 13411, 470, 14108, 0, 0,
     1248, 8.56, 1836, 12.59, 1177, 8.07, 126, 0.86, 810, 5.56, 99, 0.68, 1607, 11.02, 7205, 49.42),
    (13, 50943, 48660, 26091, 8696, 13873, 8696, 26088, 605, 34179, 0, 0,
     1738, 8.56, 2505, 7.20, 914, 2.63, 173, 0.50, 1204, 3.46, 151, 0.43, 1603, 4.61, 25891, 74.43),
    (14, 50851, 49440, 24509, 7279, 17652, 7279, 24507, 733, 31053, 0, 0,
     1964, 6.18, 4379, 13.78, 1217, 3.83, 176, 0.55, 1081, 3.40, 163, 0.51, 1693, 5.33, 20380, 64.12),
    (15, 51666, 48960, 17587, 7327, 24046, 7327, 17584, 912, 23999, 0, 0,
     1164, 4.67, 2992, 12.01, 1489, 5.98, 185, 0.74, 2519, 10.11, 192, 0.77, 1846, 7.41, 13612, 54.64),
    (16, 52933, 47100, 12917, 742, 33441, 742, 12907, 381, 13268, 0, 0,
     1649, 12.08, 1311, 9.61, 1480, 10.84, 110, 0.81, 709, 5.19, 105, 0.77, 1403, 10.28, 6501, 47.63),
    (17, 60519, 55719, 19315, 2908, 33496, 2908, 19307, 745, 21470, 0, 0,
     964, 4.34, 1837, 8.27, 1727, 7.77, 153, 0.69, 1033, 4.65, 139, 0.63, 1476, 6.64, 14141, 63.66),
    (18, 60963, 58230, 19398, 2708, 36124, 2708, 19365, 1162, 20911, 0, 0,
     1560, 7.07, 2193, 9.94, 2016, 9.13, 199, 0.90, 1239, 5.61, 142, 0.64, 1554, 7.04, 12008, 54.40),

    (19, 60401, 60401, 15481, 0, 0, 0, 15481, 0, 15481, 0, 0,
     1396, 9.02, 1600, 10.33, 1484, 9.59, 186, 1.20, 921, 5.95, 154, 1.00, 1912, 12.35, 7828, 50.56),
    (20, 59232, 59232, 16333, 0, 0, 0, 16333, 0, 16333, 0, 0,
     1467, 8.99, 1295, 7.93, 1117, 6.84, 132, 0.81, 777, 4.76, 119, 0.73, 3700, 22.66, 7726, 47.29),
    (21, 59438, 59438, 15111, 0, 0, 0, 15111, 0, 15111, 0, 0,
     1180, 7.81, 1134, 7.51, 1934, 12.80, 301, 1.99, 862, 5.71, 553, 3.66, 2244, 14.86, 6903, 45.69),
    (22, 60868, 60868, 15119, 0, 0, 0, 15119, 0, 15119, 0, 0,
     904, 5.98, 923, 6.11, 5380, 35.59, 182, 1.20, 807, 5.34, 251, 1.66, 1662, 11.00, 5010, 33.15),
    (23, 60250, 60250, 15613, 0, 0, 0, 15613, 0, 15613, 0, 0,
     892, 5.71, 1334, 8.55, 2026, 12.98, 321, 2.06, 1021, 6.54, 292, 1.87, 1665, 10.66, 8062, 51.66),
    (24, 57500, 57500, 15268, 0, 0, 0, 15268, 0, 15268, 0, 0,
     689, 4.51, 1275, 8.35, 1301, 8.52, 171, 1.12, 2229, 14.59, 2743, 17.96, 1287, 8.43, 5573, 36.50),
    (25, 58900, 58900, 13076, 0, 0, 0, 13076, 0, 13076, 0, 0,
     755, 5.77, 1156, 8.84, 1628, 12.45, 109, 0.83, 898, 6.87, 246, 1.88, 1175, 8.99, 7109, 54.37),
    (26, 61300, 61300, 16037, 0, 0, 0, 16037, 0, 16037, 0, 0,
     909, 5.67, 1119, 6.98, 1130, 7.05, 132, 0.82, 1029, 6.42, 87, 0.54, 1118, 6.97, 10513, 65.55),
    (27, 59100, 59100, 13121, 0, 0, 0, 13121, 0, 13121, 0, 0,
     779, 5.94, 1071, 8.16, 1101, 8.39, 127, 0.97, 1591, 12.13, 91, 0.69, 1245, 9.49, 7116, 54.23),
    (28, 60083, 60083, 15293, 0, 0, 0, 15293, 0, 15293, 0, 0,
     908, 5.94, 1187, 7.76, 1949, 12.75, 228, 1.49, 1099, 7.19, 111, 0.73, 1648, 10.77, 8163, 53.37),
    (29, 59600, 59600, 12715, 0, 0, 0, 12715, 0, 12715, 0, 0,
     835, 6.57, 1152, 9.06, 1309, 10.29, 125, 0.98, 1045, 8.22, 80, 0.63, 1369, 10.77, 6800, 53.49),
    (30, 59700, 59700, 15185, 0, 0, 0, 15185, 0, 15185, 0, 0,
     1265, 8.33, 1382, 9.10, 1292, 8.51, 171, 1.13, 1208, 7.96, 101, 0.67, 2230, 14.69, 7536, 49.63),
    (31, 58000, 58000, 14241, 0, 0, 0, 14241, 0, 14241, 0, 0,
     771, 5.41, 1264, 8.88, 1286, 9.03, 184, 1.29, 877, 6.16, 106, 0.74, 3259, 22.89, 6494, 45.59),
    (32, 56000, 56000, 15664, 0, 0, 0, 15664, 0, 15664, 0, 0,
     1588, 10.14, 1168, 7.46, 1502, 9.59, 221, 1.41, 833, 5.32, 104, 0.66, 1843, 11.76, 8405, 53.64),
    (33, 56000, 56000, 14320, 0, 0, 0, 14320, 0, 14320, 0, 0,
     794, 5.55, 946, 6.61, 1433, 10.01, 190, 1.33, 913, 6.38, 119, 0.83, 2358, 16.47, 7567, 52.86),
    (34, 58500, 58500, 13732, 0, 0, 0, 13732, 0, 13732, 0, 0,
     739, 5.38, 1033, 7.52, 1311, 9.55, 129, 0.94, 1029, 7.49, 105, 0.77, 1029, 7.49, 8357, 60.88),
    (35, 59600, 59600, 15763, 0, 0, 0, 15763, 0, 15763, 0, 0,
     2441, 15.49, 999, 6.34, 1724, 10.94, 134, 0.85, 1169, 7.42, 106, 0.67, 4009, 25.44, 5181, 32.87),
    (36, 56650, 56650, 17236, 0, 0, 0, 17236, 0, 17236, 0, 0,
     1362, 7.90, 935, 5.42, 1059, 6.14, 124, 0.72, 868, 5.03, 88, 0.51, 6481, 37.61, 6319, 36.66),
    (37, 53800, 53800, 14049, 0, 0, 0, 14049, 0, 14049, 0, 0,
     2230, 15.87, 956, 6.80, 1569, 11.17, 245, 1.74, 834, 5.94, 95, 0.68, 2852, 20.30, 5268, 37.51),
    (38, 55900, 55900, 15951, 0, 0, 0, 15951, 0, 15951, 0, 0,
     806, 5.05, 1232, 7.72, 2130, 13.36, 259, 1.62, 870, 5.45, 159, 1.00, 1983, 12.43, 8512, 53.37),
]

FIELD_NAMES = [
    "district",
    "voters_registered",
    "ballots_received",
    "ballots_issued_in_place",
    "ballots_issued_out",
    "ballots_cancelled",
    "ballots_in_mobile_box",
    "ballots_in_stationary_box",
    "invalid_ballots",
    "valid_ballots",
    "lost_ballots",
    "not_accounted_ballots",
    "spravedlivaya",
    "spravedlivaya_percent",
    "ldpr",
    "ldpr_percent",
    "novye_lyudi",
    "novye_lyudi_percent",
    "zelyonye",
    "zelyonye_percent",
    "pensionery",
    "pensionery_percent",
    "rodina",
    "rodina_percent",
    "kprf",
    "kprf_percent",
    "edro",
    "edro_percent"
]

PARTY_COLORS = {
    "Единая Россия": "#2E4EA4",
    "КПРФ": "#CC1111",
    "ЛДПР": "#4488CC",
    "Справедливая Россия": "#FF9933",
    "Новые Люди": "#47C2C0",
    "Родина": "#FF3840",
    "Партия Пенсионеров": "#FFB5A8",
    "Российская Партия зеленые": "#009933",
    "Самовыдвижение": "#9B96AB"
}

PARTIES = [
    ("edro", "Единая Россия"),
    ("kprf", "КПРФ"),
    ("ldpr", "ЛДПР"),
    ("spravedlivaya", "Справедливая Россия"),
    ("novye_lyudi", "Новые Люди"),
    ("rodina", "Родина"),
    ("pensionery", "Партия Пенсионеров"),
    ("zelyonye", "Российская Партия зеленые"),
    ("samovydvizhenie", "Самовыдвижение"),
]

DIST_COUNT = len(districts_data)
ALPHA_THRESHOLD = 10
MAX_MASK_SIZE = (800, 800)

try:
    RESAMPLE = Image.Resampling.LANCZOS
except Exception:
    RESAMPLE = Image.LANCZOS

def resource_path(*p):
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, *p)

def pil_to_qimage(pil):
    try:
        from PIL.ImageQt import ImageQt as PILImageQt
        return PILImageQt(pil)
    except Exception:
        pil_rgba = pil.convert("RGBA")
        w, h = pil_rgba.size
        data = pil_rgba.tobytes("raw", "RGBA")
        bytes_per_line = 4 * w
        try:
            fmt = QtGui.QImage.Format_RGBA8888
        except Exception:
            fmt = QtGui.QImage.Format_ARGB32
        return QtGui.QImage(data, w, h, bytes_per_line, fmt)

# ---------------------------
# Кеш статических данных
# ---------------------------
def build_static_cache():
    cache = {}
    for tpl in districts_data:
        if not tpl:
            continue
        if len(tpl) != len(FIELD_NAMES):
            continue
        row = dict(zip(FIELD_NAMES, tpl))
        try:
            did = int(row.get("district"))
        except Exception:
            continue
        row.setdefault("name", f"Район {did}")
        cache[did] = row
    return cache

STATIC_CACHE = build_static_cache()
NOVOSIBIRSK_CACHE = {k: v for k, v in STATIC_CACHE.items() if 19 <= k <= 38}

def _autopct_small(pct):
    return f"{pct:.1f}%" if pct >= 1.0 else ""

# ---------------------------
# Worker for overlay images
# ---------------------------
class OverlayWorker(threading.Thread):
    def __init__(self, task_q, result_q, files, size):
        super().__init__(daemon=True)
        self.task_q = task_q
        self.result_q = result_q
        self.files = files
        self.size = size

    def run(self):
        while True:
            d = self.task_q.get()
            if d is None:
                break
            f = self.files.get(d)
            if not f or not os.path.exists(f):
                self.result_q.put((d, None))
                continue
            try:
                pil = Image.open(f).convert("RGBA")
                pil.thumbnail(MAX_MASK_SIZE, RESAMPLE)
                pil = pil.resize(self.size, RESAMPLE)
                bright = ImageEnhance.Brightness(pil).enhance(1.6)
                bright = ImageEnhance.Contrast(bright).enhance(1.15)
                r, g, b, a = bright.split()
                a = a.point(lambda p: min(255, int(p * 1.4)))
                bright.putalpha(a)
                self.result_q.put((d, bright))
            except Exception:
                self.result_q.put((d, None))

# ---------------------------
# ResultsDialog
# ---------------------------
class ResultsDialog(QtWidgets.QDialog):
    def __init__(self, parent, district, rowdict):
        super().__init__(parent)
        self.setWindowTitle(f"Район {district} — результаты")
        self.resize(780, 520)
        layout = QtWidgets.QVBoxLayout(self)

        reg = int(rowdict.get("voters_registered", 0) or 0)
        valid = int(rowdict.get("valid_ballots", 0) or 0)
        pct = (valid / reg * 100.0) if reg > 0 else 0.0

        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel(f"Зарегистрированных избирателей: <b>{reg}</b>"))
        top.addStretch()
        top.addWidget(QtWidgets.QLabel(f"Действительных бюллетеней: <b>{valid}</b>"))
        top.addSpacing(12)
        top.addWidget(QtWidgets.QLabel(f"Доля действительных: <b>{pct:.2f}%</b>"))
        layout.addLayout(top)

        fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), dpi=100, gridspec_kw={'width_ratios':[2,1]})
        self.canvas = FigureCanvas(fig)
        layout.addWidget(self.canvas)

        labels = []
        counts = []
        colors = []
        for key, name in PARTIES:
            val = rowdict.get(key, 0)
            try:
                cnt = float(val)
            except Exception:
                cnt = 0.0
            labels.append(name)
            counts.append(cnt)
            colors.append(PARTY_COLORS.get(name, "#888888"))

        total_votes = sum(counts)
        ax_pie = axes[0]
        if total_votes <= 0:
            ax_pie.text(0.5, 0.5, "Нет данных по голосам", ha="center", va="center", fontsize=12)
            ax_pie.axis("off")
        else:
            pie_res = ax_pie.pie(
                counts,
                labels=None,
                autopct=_autopct_small,
                startangle=90,
                colors=colors,
                textprops={'fontsize': 8},
                wedgeprops=dict(edgecolor='w')
            )
            wedges = pie_res[0] if isinstance(pie_res, tuple) and len(pie_res) >= 1 else pie_res
            legend_labels = [
                f"{name} — {int(cnt)} ({(cnt/total_votes*100):.1f}%)" for name, cnt in zip(labels, counts)
            ]
            ax_pie.axis("equal")
            ax_pie.set_title("Распределение голосов по партиям")
            ax_pie.legend(wedges, legend_labels, loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8)

        ax_bar = axes[1]
        bars_reg = ax_bar.barh([0], [reg], height=0.4, color="#6C8CD5", label="Зарегистрированных избирателей")
        bars_val = ax_bar.barh([0.6], [valid], height=0.4, color="#4CAF50", label="Действительных бюллетеней")
        ax_bar.set_yticks([0, 0.6])
        ax_bar.set_yticklabels(["Зарегистрированных избирателей", "Действительных бюллетеней"])
        ax_bar.invert_yaxis()
        ax_bar.set_xlabel("Число")
        ax_bar.set_title("Числа по району")
        try:
            ax_bar.bar_label(bars_reg, fmt='%d', padding=3)
            ax_bar.bar_label(bars_val, fmt='%d', padding=3)
        except Exception:
            ax_bar.text(reg, 0, str(int(reg)), va='center', ha='left', fontsize=9)
            ax_bar.text(valid, 0.6, str(int(valid)), va='center', ha='left', fontsize=9)

        table = QtWidgets.QTableWidget(self)
        table.setColumnCount(3)
        table.setHorizontalHeaderLabels(["Партия", "Голоса", "Процент"])
        table.horizontalHeader().setStretchLastSection(True)
        rows = []
        for (key, label), cnt in zip(PARTIES, counts):
            pct_party = (cnt / total_votes * 100) if total_votes > 0 else 0.0
            rows.append((label, int(cnt) if cnt == int(cnt) else cnt, f"{pct_party:.2f}%"))
        table.setRowCount(len(rows))
        for i, (lab, cnt, pct_party) in enumerate(rows):
            table.setItem(i, 0, QtWidgets.QTableWidgetItem(str(lab)))
            table.setItem(i, 1, QtWidgets.QTableWidgetItem(str(cnt)))
            table.setItem(i, 2, QtWidgets.QTableWidgetItem(str(pct_party)))
        table.resizeColumnsToContents()
        layout.addWidget(table)

        btn = QtWidgets.QPushButton("Закрыть")
        btn.clicked.connect(self.accept)
        layout.addWidget(btn)

        fig.tight_layout()
        self.canvas.draw()

# ---------------------------
# Map view for NovosibirskExpand (zoom + pan)
# ---------------------------
class MapGraphicsView(QtWidgets.QGraphicsView):
    def __init__(self, pixmap: QtGui.QPixmap = None, parent=None):
        super().__init__(parent)
        self.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)
        self.setViewportUpdateMode(QtWidgets.QGraphicsView.BoundingRectViewportUpdate)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)

        self._scene = QtWidgets.QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item = None
        self._zoom = 1.0
        self._zoom_min = 0.1
        self._zoom_max = 10.0
        self._panning = False
        self._pan_start = QtCore.QPoint()

        if pixmap and not pixmap.isNull():
            self.set_pixmap(pixmap)

    def set_pixmap(self, pixmap: QtGui.QPixmap):
        self._scene.clear()
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._scene.setSceneRect(self._pixmap_item.boundingRect())
        self.fit_to_view()

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 1.25 if delta > 0 else 0.8
        self._apply_zoom(factor)

    def _apply_zoom(self, factor):
        new_zoom = self._zoom * factor
        if new_zoom < self._zoom_min or new_zoom > self._zoom_max:
            return
        self._zoom = new_zoom
        self.scale(factor, factor)

    def zoom_in(self):
        self._apply_zoom(1.25)

    def zoom_out(self):
        self._apply_zoom(0.8)

    def reset_zoom(self):
        self.resetTransform()
        self._zoom = 1.0
        self.fit_to_view()

    def fit_to_view(self):
        if self._pixmap_item is None:
            return
        self.fitInView(self._scene.sceneRect(), QtCore.Qt.KeepAspectRatio)
        self._zoom = 1.0

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self._panning = True
            self.setCursor(QtCore.Qt.ClosedHandCursor)
            self._pan_start = event.pos()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._panning:
            delta = event.pos() - self._pan_start
            self._pan_start = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton and self._panning:
            self._panning = False
            self.setCursor(QtCore.Qt.ArrowCursor)
            return
        super().mouseReleaseEvent(event)

# ---------------------------
# NovosibirskExpand: список округов + карта (без выбора файла)
# ---------------------------
class NovosibirskExpand(QtWidgets.QDialog):
    def __init__(self, parent, cache):
        super().__init__(parent)
        self.setWindowTitle("Novosibirsk — Округа и карта")
        self.resize(1100, 720)
        self.cache = cache

        main_layout = QtWidgets.QVBoxLayout(self)

        # toolbar with zoom controls
        toolbar = QtWidgets.QHBoxLayout()
        btn_zoom_in = QtWidgets.QPushButton("+")
        btn_zoom_in.setFixedWidth(36)
        btn_zoom_out = QtWidgets.QPushButton("−")
        btn_zoom_out.setFixedWidth(36)
        btn_fit = QtWidgets.QPushButton("Fit")
        lbl_hint = QtWidgets.QLabel("Двойной клик по карте — открыть полноразмер")
        lbl_hint.setStyleSheet("color: gray;")
        toolbar.addWidget(btn_zoom_in)
        toolbar.addWidget(btn_zoom_out)
        toolbar.addWidget(btn_fit)
        toolbar.addStretch()
        toolbar.addWidget(lbl_hint)
        main_layout.addLayout(toolbar)

        # main content: list left, map right
        content = QtWidgets.QHBoxLayout()
        main_layout.addLayout(content, 1)

        # left: list of districts
        left_w = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_w)
        left_layout.setContentsMargins(6, 6, 6, 6)
        left_layout.addWidget(QtWidgets.QLabel("Список округов"))
        self.list_widget = QtWidgets.QListWidget()
        for did in sorted(self.cache.keys()):
            row = self.cache[did]
            name = row.get("name", f"Округ {did}")
            item = QtWidgets.QListWidgetItem(f"{did} — {name}")
            item.setData(QtCore.Qt.UserRole, did)
            self.list_widget.addItem(item)
        left_layout.addWidget(self.list_widget, 1)
        btn_open = QtWidgets.QPushButton("Открыть")
        btn_close = QtWidgets.QPushButton("Закрыть")
        hbtn = QtWidgets.QHBoxLayout()
        hbtn.addWidget(btn_open)
        hbtn.addWidget(btn_close)
        left_layout.addLayout(hbtn)
        content.addWidget(left_w, 30)  # 30% width

        # right: map view
        right_w = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_w)
        right_layout.setContentsMargins(6, 6, 6, 6)

        # try to load map automatically from images/Novosibirsk_map.png or any file with "novo" in name
        map_pix = None
        try:
            images_dir = resource_path("images")
            default = os.path.join(images_dir, "Novosibirsk_map.png")
            if os.path.exists(default):
                map_pix = QtGui.QPixmap(default)
            else:
                if os.path.isdir(images_dir):
                    for fn in os.listdir(images_dir):
                        low = fn.lower()
                        if low.endswith((".png", ".jpg", ".jpeg", ".bmp")) and ("novo" in low or "novos" in low):
                            cand = os.path.join(images_dir, fn)
                            map_pix = QtGui.QPixmap(cand)
                            if not map_pix.isNull():
                                break
        except Exception:
            map_pix = None

        if map_pix is None or map_pix.isNull():
            placeholder = QtWidgets.QLabel("Карта Novosibirsk не найдена в папке images")
            placeholder.setAlignment(QtCore.Qt.AlignCenter)
            placeholder.setStyleSheet("color: gray;")
            right_layout.addWidget(placeholder)
            self.map_view = None
        else:
            self.map_view = MapGraphicsView(map_pix)
            right_layout.addWidget(self.map_view, 1)

        content.addWidget(right_w, 70)  # 70% width

        # connections
        btn_open.clicked.connect(self.open_selected)
        btn_close.clicked.connect(self.accept)
        self.list_widget.itemDoubleClicked.connect(self.on_item_double)
        self.list_widget.currentItemChanged.connect(self.on_selection_changed)

        if self.map_view:
            btn_zoom_in.clicked.connect(self.map_view.zoom_in)
            btn_zoom_out.clicked.connect(self.map_view.zoom_out)
            btn_fit.clicked.connect(self.map_view.fit_to_view)
            self.map_view.mouseDoubleClickEvent = lambda ev: self._open_full_map()

    def on_item_double(self, item):
        did = item.data(QtCore.Qt.UserRole)
        self.open_district(did)

    def open_selected(self):
        item = self.list_widget.currentItem()
        if not item:
            return
        did = item.data(QtCore.Qt.UserRole)
        self.open_district(did)

    def open_district(self, did):
        row = self.cache.get(did)
        if row is None:
            QtWidgets.QMessageBox.information(self, "Нет данных", f"Данных по округу {did} нет")
            return
        dlg = ResultsDialog(self, did, row)
        dlg.exec_()

    def on_selection_changed(self, current, previous):
        # placeholder for future: center map on selection if coordinates available
        pass

    def _open_full_map(self):
        if not self.map_view or not self.map_view._pixmap_item:
            QtWidgets.QMessageBox.information(self, "Карта не найдена", "Карта не загружена")
            return
        pix = self.map_view._pixmap_item.pixmap()
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Novosibirsk — полноразмерная карта")
        dlg.resize(1200, 900)
        v = QtWidgets.QVBoxLayout(dlg)
        lbl = QtWidgets.QLabel()
        lbl.setAlignment(QtCore.Qt.AlignCenter)
        lbl.setPixmap(pix)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(lbl)
        v.addWidget(scroll, 1)
        btn = QtWidgets.QPushButton("Закрыть")
        btn.clicked.connect(dlg.accept)
        v.addWidget(btn)
        dlg.exec_()

# ---------------------------
# Overall stats dialog
# ---------------------------
class OverallStatsDialog(QtWidgets.QDialog):
    def __init__(self, parent, static_cache, novosib_cache):
        super().__init__(parent)
        self.setWindowTitle("Общие проценты по области")
        self.resize(900, 600)
        layout = QtWidgets.QVBoxLayout(self)

        total_reg = 0
        total_valid = 0
        party_totals = {p[0]: 0.0 for p in PARTIES}
        for d, row in static_cache.items():
            try:
                total_reg += int(row.get("voters_registered", 0) or 0)
            except Exception:
                pass
            try:
                total_valid += int(row.get("valid_ballots", 0) or 0)
            except Exception:
                pass
            for key, _ in PARTIES:
                try:
                    party_totals[key] += float(row.get(key, 0) or 0)
                except Exception:
                    pass
        for d, row in novosib_cache.items():
            try:
                total_reg += int(row.get("voters_registered", 0) or 0)
            except Exception:
                pass
            try:
                total_valid += int(row.get("valid_ballots", 0) or 0)
            except Exception:
                pass
            for key, _ in PARTIES:
                try:
                    party_totals[key] += float(row.get(key, 0) or 0)
                except Exception:
                    pass

        pct_valid = (total_valid / total_reg * 100.0) if total_reg > 0 else 0.0

        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel(f"Всего зарегистрировано: <b>{total_reg}</b>"))
        top.addStretch()
        top.addWidget(QtWidgets.QLabel(f"Всего действительных бюллетеней: <b>{total_valid}</b>"))
        top.addSpacing(12)
        top.addWidget(QtWidgets.QLabel(f"Доля действительных: <b>{pct_valid:.2f}%</b>"))
        layout.addLayout(top)

        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=100)
        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)

        labels = []
        counts = []
        colors = []
        for key, name in PARTIES:
            labels.append(name)
            counts.append(party_totals.get(key, 0.0))
            colors.append(PARTY_COLORS.get(name, "#888888"))

        total_party_votes = sum(counts)
        if total_party_votes <= 0:
            ax.text(0.5, 0.5, "Нет данных по партиям", ha="center", va="center")
            ax.axis("off")
        else:
            pie_res = ax.pie(counts, labels=None, autopct=_autopct_small, startangle=90, colors=colors,
                             textprops={'fontsize': 8}, wedgeprops=dict(edgecolor='w'))
            wedges = pie_res[0] if isinstance(pie_res, tuple) and len(pie_res) >= 1 else pie_res
            legend_labels = [f"{name} — {int(cnt)} ({(cnt/total_party_votes*100):.1f}%)" for name, cnt in zip(labels, counts)]
            ax.axis("equal")
            ax.set_title("Распределение партий по области")
            ax.legend(wedges, legend_labels, loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8)

        table = QtWidgets.QTableWidget(self)
        table.setColumnCount(3)
        table.setHorizontalHeaderLabels(["Партия", "Голоса (всего)", "Доля от партийных голосов, %"])
        table.setRowCount(len(PARTIES))
        for i, (key, name) in enumerate(PARTIES):
            cnt = int(party_totals.get(key, 0.0))
            pct_party = (party_totals.get(key, 0.0) / total_party_votes * 100.0) if total_party_votes > 0 else 0.0
            table.setItem(i, 0, QtWidgets.QTableWidgetItem(name))
            table.setItem(i, 1, QtWidgets.QTableWidgetItem(str(cnt)))
            table.setItem(i, 2, QtWidgets.QTableWidgetItem(f"{pct_party:.2f}%"))
        table.resizeColumnsToContents()
        layout.addWidget(table, 1)

        btn = QtWidgets.QPushButton("Закрыть")
        btn.clicked.connect(self.accept)
        layout.addWidget(btn)

        fig.tight_layout()
        canvas.draw()

# ---------------------------
# MapView: dynamic overlay masks + context menu
# ---------------------------
class MapView(QtWidgets.QGraphicsView):
    districtClicked = QtCore.pyqtSignal(int)

    def __init__(self, map_path, masks_dir, scale=4, debounce_ms=80):
        super().__init__()
        self.setMouseTracking(True)

        if not os.path.exists(map_path):
            QtWidgets.QMessageBox.critical(None, "Ошибка", f"Карта не найдена:\n{map_path}")
            raise FileNotFoundError(map_path)
        self.base_pix = QtGui.QPixmap(map_path)
        if self.base_pix.isNull():
            QtWidgets.QMessageBox.critical(None, "Ошибка", f"Не удалось загрузить изображение: {map_path}")
            raise RuntimeError("Failed to load base pixmap")

        self.scene = QtWidgets.QGraphicsScene(self)
        self.setScene(self.scene)
        w, h = self.base_pix.width(), self.base_pix.height()
        self.scene.setSceneRect(0, 0, w, h)
        self.base_item = self.scene.addPixmap(self.base_pix)
        self.base_item.setZValue(0)
        try:
            self.base_item.setCacheMode(QtWidgets.QGraphicsItem.DeviceCoordinateCache)
        except Exception:
            pass

        transparent = QtGui.QPixmap(self.base_pix.size())
        transparent.fill(QtCore.Qt.transparent)
        self.transparent_pix = transparent
        self.overlay_item = self.scene.addPixmap(self.transparent_pix)
        self.overlay_item.setZValue(1)
        try:
            self.overlay_item.setCacheMode(QtWidgets.QGraphicsItem.DeviceCoordinateCache)
        except Exception:
            pass

        self.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)
        self.setViewportUpdateMode(QtWidgets.QGraphicsView.BoundingRectViewportUpdate)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setMinimumSize(min(1200, w), min(800, h))
        self.fitInView(self.scene.sceneRect(), QtCore.Qt.KeepAspectRatio)

        self.hit_scale = max(1, int(scale))
        self.debounce_ms = int(debounce_ms)
        self.size = (w, h)

        # load mask files (numeric names like 1.png, 2.png, ...)
        self.files = {}
        try:
            for fname in os.listdir(masks_dir):
                if not fname.lower().endswith(".png"):
                    continue
                name = os.path.splitext(fname)[0]
                if not name.isdigit():
                    continue
                idx = int(name)
                self.files[idx] = os.path.join(masks_dir, fname)
        except Exception:
            pass

        # build label_map
        self.label_map = np.zeros((h, w), dtype=np.uint8)
        for idx, fpath in self.files.items():
            try:
                img = Image.open(fpath).convert("RGBA").resize((w, h), RESAMPLE)
                alpha = np.array(img.getchannel("A"))
                self.label_map[alpha > ALPHA_THRESHOLD] = idx
            except Exception:
                pass

        self.label_map_small = self.label_map[::self.hit_scale, ::self.hit_scale]

        self.task_q = queue.Queue()
        self.result_q = queue.Queue()
        self.pending = set()
        self.cache = {}
        self.worker = OverlayWorker(self.task_q, self.result_q, self.files, self.size)
        self.worker.start()

        self._deb = QtCore.QTimer(self)
        self._deb.setSingleShot(True)
        self._deb.timeout.connect(self._process_motion)
        self._last_pos = None

        self._poll = QtCore.QTimer(self)
        self._poll.timeout.connect(self._poll_results)
        self._poll.start(50)

        self._zoom = 1.0
        self._zoom_min = 0.2
        self._zoom_max = 6.0
        self._panning = False
        self._pan_start = QtCore.QPoint()

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 1.25 if delta > 0 else 0.8
        new_zoom = self._zoom * factor
        if new_zoom < self._zoom_min or new_zoom > self._zoom_max:
            return
        self._zoom = new_zoom
        super().scale(factor, factor)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.RightButton:
            self._panning = True
            self.setCursor(QtCore.Qt.ClosedHandCursor)
            self._pan_start = event.pos()
            return
        if event.button() == QtCore.Qt.LeftButton:
            p = self.mapToScene(event.pos()).toPoint()
            x, y = p.x(), p.y()
            if 0 <= x < self.label_map.shape[1] and 0 <= y < self.label_map.shape[0]:
                did = int(self.label_map[y, x])
                if did != 0:
                    self.districtClicked.emit(did)
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._panning:
            delta = event.pos() - self._pan_start
            self._pan_start = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            return
        p = self.mapToScene(event.pos()).toPoint()
        self._last_pos = (p.x(), p.y())
        self._deb.start(self.debounce_ms)

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.RightButton and self._panning:
            self._panning = False
            self.setCursor(QtCore.Qt.ArrowCursor)
            return
        super().mouseReleaseEvent(event)

    def contextMenuEvent(self, event):
        p = self.mapToScene(event.pos()).toPoint()
        x, y = p.x(), p.y()
        did = 0
        if 0 <= x < self.label_map.shape[1] and 0 <= y < self.label_map.shape[0]:
            did = int(self.label_map[y, x])
        menu = QtWidgets.QMenu(self)
        if did == 0:
            menu.addAction("Нет района под курсором")
        else:
            menu.addAction(f"Район {did}")
        act_novo = menu.addAction("Открыть Novosibirsk — список округов")
        act_novo.triggered.connect(lambda: self.districtClicked.emit(19))
        menu.exec_(event.globalPos())

    def _process_motion(self):
        if self._last_pos is None:
            return
        x, y = self._last_pos
        sx, sy = x // self.hit_scale, y // self.hit_scale
        h_small, w_small = self.label_map_small.shape
        if sx < 0 or sy < 0 or sx >= w_small or sy >= h_small:
            self._set_overlay(None)
            return
        found = int(self.label_map_small[sy, sx])
        self._set_overlay(found if found != 0 else None)

    def _set_overlay(self, did):
        if did is None:
            self.overlay_item.setPixmap(self.transparent_pix)
            return
        if did in self.cache:
            self.overlay_item.setPixmap(self.cache[did])
            return
        if did not in self.pending:
            self.pending.add(did)
            self.task_q.put(did)
            self.overlay_item.setPixmap(self.transparent_pix)

    def _poll_results(self):
        try:
            while True:
                d, pil = self.result_q.get_nowait()
                self.pending.discard(d)
                if pil is None:
                    continue
                qim = pil_to_qimage(pil)
                pix = QtGui.QPixmap.fromImage(qim)
                self.cache[d] = pix
                self.overlay_item.setPixmap(pix)
        except queue.Empty:
            pass

    def shutdown(self):
        try:
            self.task_q.put(None)
            self.worker.join(timeout=1)
        except Exception:
            pass

# ---------------------------
# Main window
# ---------------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, map_path, masks_dir):
        super().__init__()
        self.setWindowTitle("Election Map")
        self.resize(1400, 900)

        menubar = self.menuBar()
        stats_menu = menubar.addMenu("Статистика")
        overall_action = QtWidgets.QAction("Общие проценты по области", self)
        overall_action.triggered.connect(self.show_overall_stats)
        stats_menu.addAction(overall_action)
        novo_action = QtWidgets.QAction("Novosibirsk — список округов", self)
        novo_action.triggered.connect(self.open_novosibirsk)
        stats_menu.addAction(novo_action)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        hbox = QtWidgets.QHBoxLayout(central)

        self.map_view = MapView(map_path, masks_dir, scale=4, debounce_ms=80)
        hbox.addWidget(self.map_view, 1)

        right = QtWidgets.QWidget()
        right.setFixedWidth(260)
        v = QtWidgets.QVBoxLayout(right)
        v.setContentsMargins(8, 8, 8, 8)

        lbl = QtWidgets.QLabel("Нажмите на район для деталей")
        lbl.setWordWrap(True)
        v.addWidget(lbl)

        v.addSpacing(6)
        self.btn_reset = QtWidgets.QPushButton("Сброс зума")
        self.btn_reset.clicked.connect(self.reset_zoom)
        v.addWidget(self.btn_reset)

        v.addStretch()
        hbox.addWidget(right)

        self.static_cache = STATIC_CACHE

        self.map_view.districtClicked.connect(self.on_district_clicked)

        self.status = self.statusBar()
        self.status.showMessage("Готово")

    def on_district_clicked(self, did):
        if int(did) == 19:
            self.open_novosibirsk()
            return
        row = self.static_cache.get(did)
        if row is None:
            QtWidgets.QMessageBox.information(self, "Нет данных", f"Данных по району {did} нет")
            return
        dlg = ResultsDialog(self, did, row)
        dlg.exec_()

    def open_novosibirsk(self):
        dlg = NovosibirskExpand(self, NOVOSIBIRSK_CACHE)
        dlg.exec_()

    def show_overall_stats(self):
        dlg = OverallStatsDialog(self, self.static_cache, NOVOSIBIRSK_CACHE)
        dlg.exec_()

    def reset_zoom(self):
        try:
            self.map_view.resetTransform()
            self.map_view.fitInView(self.map_view.scene.sceneRect(), QtCore.Qt.KeepAspectRatio)
            self.map_view._zoom = 1.0
        except Exception:
            pass

    def closeEvent(self, ev):
        try:
            self.map_view.shutdown()
        except Exception:
            pass
        super().closeEvent(ev)

# ---------------------------
# Entry point
# ---------------------------
def main():
    app = QtWidgets.QApplication(sys.argv)
    map_path = resource_path("images", "map.png")
    masks_dir = resource_path("images")
    try:
        w = MainWindow(map_path, masks_dir)
    except Exception as e:
        print("Ошибка инициализации:", e)
        return
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
