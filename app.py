# Copyright (c) 2025-2026 Hannes Göök
# MIT License - GDForge
# https://github.com/hannesgook/gdforge

import sys
import numpy as np
import math
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QDoubleSpinBox, QSpinBox, QCheckBox, QGraphicsView, QGraphicsScene,
    QFormLayout, QLineEdit, QScrollArea, QGraphicsDropShadowEffect, QFrame, QSplitter
)
from PySide6.QtGui import QPainter, QPainterPath, QPen, QBrush, QPalette, QColor, QFont, QTransform
from PySide6.QtCore import Qt, QPropertyAnimation, QEasingCurve, QSize, Signal, QTimer
import pyqtgraph as pg
import soundfile as sf
import sounddevice as sd
import threading
import time as _time
from PySide6.QtWidgets import QSplitter, QGroupBox
from PySide6.QtGui import QColor, QFont, QTransform
import os, shutil
from gd_serialize import build_wave_ramps_45deg, wave_make_clones, build_ramps_along_path_by_spacing, build_level_xml

from settings import AppSettings
from audio_analysis import analyze_audio
from generator import generate_level
from gd_serialize import build_k4_polyline, serialize_gmd, build_k4_orb_arc

Y_SHIFT = 0.0
GROUND_LINE_Y = 0.0

class Preview(QWidget):
    visibleRangeChanged = Signal(float, float)
    seekRequested = Signal(float) # emits time in seconds

    def __init__(self):
        super().__init__()

        self.level = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Title label (styled like PyQtGraph titles)
        self.title_label = QLabel("Level Preview")
        self.title_label.setStyleSheet("""
            QLabel {
                background-color: #202028;
                color: #cccccc;
                padding: 5px;
                font-size: 10pt;
                font-weight: bold;
            }
        """)
        self.title_label.setAlignment(Qt.AlignCenter)

        # Graphics view
        self.view = QGraphicsView()
        self.view.setScene(QGraphicsScene(self.view))
        self.view.setRenderHint(QPainter.Antialiasing, True)
        self.view.setDragMode(QGraphicsView.ScrollHandDrag)
        self.view.setBackgroundBrush(QColor(32, 32, 40))
        self.view.setFrameShape(QGraphicsView.NoFrame)
        self.view.scale(1.0, -1.0)

        # Smooth zoom settings
        self.view.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.view.setResizeAnchor(QGraphicsView.AnchorUnderMouse)

        # Track view changes to sync with audio plot
        self.view.horizontalScrollBar().valueChanged.connect(self.report_visible_range)
        self.view.verticalScrollBar().valueChanged.connect(self.report_visible_range)

        # Intercept double-clicks for seek
        self.view.viewport().installEventFilter(self)

        # Playhead line in scene coordinates
        self._playhead_line = None
        self._playhead_visible = False

        layout.addWidget(self.title_label)
        layout.addWidget(self.view)

    def eventFilter(self, obj, event):
        from PySide6.QtCore import QEvent
        if obj is self.view.viewport() and event.type() == QEvent.MouseButtonDblClick:
            if self.level is not None and self.level.units_per_second > 0:
                scene_pos = self.view.mapToScene(event.pos())
                t = scene_pos.x() / self.level.units_per_second
                t = max(0.0, t)
                self.seekRequested.emit(t)
                return True
        return super().eventFilter(obj, event)

    def wheelEvent(self, event):
        scale_factor = 1.08
        factor = scale_factor if event.angleDelta().y() > 0 else 1.0 / scale_factor
        self.view.scale(factor, factor)
        self.report_visible_range()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.report_visible_range()

    def report_visible_range(self, *args):
        if self.level is None:
            return
        port = self.view.viewport().rect()
        scene_poly = self.view.mapToScene(port)
        rect = scene_poly.boundingRect()
        start_x = rect.left()
        end_x = rect.right()
        if self.level.units_per_second > 0:
            start_t = start_x / self.level.units_per_second
            end_t = end_x / self.level.units_per_second
            self.visibleRangeChanged.emit(start_t, end_t)

    def set_playhead(self, t, visible=True, auto_scroll=True):
        if self.level is None or self.level.units_per_second <= 0:
            return
        x_pos = float(t) * self.level.units_per_second
        if self._playhead_line is None:
            # create the line the first time it is needed.
            sc = self.view.scene()
            sr = sc.sceneRect()
            from PySide6.QtWidgets import QGraphicsLineItem
            pen = QPen(QColor(0, 230, 255), 2)
            pen.setCosmetic(True)
            self._playhead_line = sc.addLine(0, sr.top(), 0, sr.bottom(), pen)
            self._playhead_line.setZValue(100)
        self._playhead_line.setVisible(visible)
        if not visible:
            return
        # Move by adjusting X translation - zero allocation, zero scene rebuild.
        self._playhead_line.setX(x_pos)
        if auto_scroll:
            port = self.view.viewport().rect()
            scene_poly = self.view.mapToScene(port)
            rect = scene_poly.boundingRect()
            if x_pos < rect.left() + rect.width() * 0.1 or x_pos > rect.right() - rect.width() * 0.1:
                self.view.centerOn(x_pos, 0)

    def show_level(self, level, settings=None):
        self.level = level
        sc = self.view.scene()
        sc.clear()
        self._playhead_line = None # scene.clear() destroys all items; force lazy re-create

        if level is None or len(level.t_samp) < 2:
            sc.setSceneRect(-200, -200, 400, 400)
            return

        x = (level.t_samp * level.units_per_second).astype(np.float64)
        y_ceil = settings.path.wave_corridor_units * 30.0 if settings is not None else 9999.0
        is_wave = settings is not None and bool(settings.path.start_as_wave)
        y_top = y_ceil - 15.0 if is_wave else float('inf')
        y = np.clip(level.y_samp + Y_SHIFT, 15.0, y_top)

        main_path = QPainterPath()
        main_path.moveTo(0.0, float(y[0]))
        main_path.lineTo(float(x[0]), float(y[0]))
        for i in range(1, len(x)):
            main_path.lineTo(float(x[i]), float(y[i]))

        gd_unit = 30.0
        x_min = float(x.min())
        x_max = float(x.max())
        y_min = float(y.min())
        y_max = float(y.max())

        view_margin = 200.0
        x_range_min = min(x_min, 0.0) - view_margin
        x_range_max = x_max + view_margin
        y_range_min = y_min - 100
        y_range_max = y_max + 100

        def cosmetic_pen(color, width=1, style=Qt.SolidLine):
            p = QPen(color, width)
            p.setCosmetic(True)
            p.setStyle(style)
            return p

        grid_pen = cosmetic_pen(QColor(40, 40, 50), 3, Qt.DotLine)
        mark_pen = cosmetic_pen(QColor(80, 80, 90), 2)
        tick_pen = cosmetic_pen(QColor(120, 120, 130), 2)
        path_pen = cosmetic_pen(QColor(240, 240, 240), 2)
        ref_pen = cosmetic_pen(QColor(255, 80, 80), 2)
        ground_pen = cosmetic_pen(QColor(100, 100, 140), 4)
        time_pen = cosmetic_pen(QColor(255, 255, 0), 2)

        label_font = QFont("Arial", 8)
        label_transform = QTransform()
        label_transform.scale(1, -1)

        # Batch vertical grid lines into one path
        vgrid_path = QPainterPath()
        vmark_path = QPainterPath()
        vtick_path = QPainterPath()

        x_start_g = int(np.floor(x_range_min / gd_unit)) * gd_unit
        x_end_g   = int(np.ceil(x_range_max  / gd_unit)) * gd_unit
        grid_xs = np.arange(x_start_g, x_end_g + gd_unit, gd_unit)
        grid_xs = grid_xs[grid_xs >= -0.001]

        # Limit label density: aim for at most ~200 labels regardless of song length.
        label_step = max(1, int(np.ceil(len(grid_xs) / 200)))

        ground_y_val = GROUND_LINE_Y
        grid_y_min_v = max(y_range_min, ground_y_val)

        for idx_g, gx in enumerate(grid_xs):
            gx = float(gx)
            vgrid_path.moveTo(gx, grid_y_min_v)
            vgrid_path.lineTo(gx, y_range_max)
            vmark_path.moveTo(gx - 5, ground_y_val)
            vmark_path.lineTo(gx + 5, ground_y_val)
            vtick_path.moveTo(gx, Y_SHIFT - 0)
            vtick_path.lineTo(gx, Y_SHIFT - 10)
            # Labels are still individual items (unavoidable for text)
            if gx >= -0.001 and (idx_g % label_step == 0):
                ti = sc.addText(f"{int(gx / gd_unit)}", label_font)
                ti.setDefaultTextColor(QColor(120, 120, 130))
                ti.setPos(gx - 15, Y_SHIFT - 20)
                ti.setTransform(label_transform)

        sc.addPath(vgrid_path, grid_pen)
        sc.addPath(vmark_path, mark_pen)
        sc.addPath(vtick_path, tick_pen)

        # Batch horizontal grid lines into one path
        hgrid_path = QPainterPath()
        hmark_path = QPainterPath()
        htick_path = QPainterPath()

        y_start_g = int(np.floor(y_range_min / gd_unit)) * gd_unit
        y_end_g   = int(np.ceil(y_range_max  / gd_unit)) * gd_unit
        grid_ys = np.arange(y_start_g, y_end_g + gd_unit, gd_unit)
        grid_ys = grid_ys[grid_ys >= ground_y_val]

        drawn_labels = set()
        for gy in grid_ys:
            gy = float(gy)
            hgrid_path.moveTo(0, gy)
            hgrid_path.lineTo(x_range_max, gy)
            hmark_path.moveTo(0, gy - 5)
            hmark_path.lineTo(0, gy + 5)
            htick_path.moveTo(-15, gy)
            htick_path.lineTo(-5,  gy)

            gd_block_height = int(round((gy - GROUND_LINE_Y) / gd_unit))
            if gd_block_height not in drawn_labels:
                drawn_labels.add(gd_block_height)
                ti = sc.addText(f"{gd_block_height}", label_font)
                ti.setDefaultTextColor(QColor(120, 120, 130))
                ti.setPos(-45, gy + 3)
                ti.setTransform(label_transform)

        sc.addPath(hgrid_path, grid_pen)
        sc.addPath(hmark_path, mark_pen)
        sc.addPath(htick_path, tick_pen)

        sc.addPath(main_path, path_pen)

        # reference, ground, ceiling lines
        sc.addLine(0, y_range_min, 0, y_range_max, ref_pen)
        sc.addLine(x_range_min, GROUND_LINE_Y, x_range_max, GROUND_LINE_Y, ground_pen)

        if settings is not None and settings.path.start_as_wave:
            ceiling_y = settings.path.wave_corridor_units * 30.0
            ceil_pen = cosmetic_pen(QColor(80, 80, 100), 3)
            sc.addLine(x_range_min, ceiling_y, x_range_max, ceiling_y, ceil_pen)

        # beat ticks
        beats_path = QPainterPath()
        for t in level.times:
            xx = float(t * level.units_per_second)
            beats_path.moveTo(xx, -10.0)
            beats_path.lineTo(xx, +10.0)
        sc.addPath(beats_path, time_pen)

        # orb circles (vectorized interp, individual ellipses)
        if level.orb_times is not None and len(level.orb_times) > 0:
            orb_colors = {
                0: QColor(255, 255, 0),
                1: QColor(255, 0, 255),
                2: QColor(0, 150, 255),
                3: QColor(0, 255, 0),
            }
            orb_xs = level.orb_times * level.units_per_second
            orb_ys = np.interp(level.orb_times, level.t_samp, level.y_samp) + Y_SHIFT

            for ox, oy, typ in zip(orb_xs, orb_ys, level.orb_types):
                color = orb_colors.get(int(typ), QColor(255, 255, 0))
                pen = QPen(color, 2)
                pen.setCosmetic(True)
                brush = QColor(color)
                brush.setAlpha(150)
                sc.addEllipse(float(ox) - 8, float(oy) - 8, 16, 16, pen, brush)

        # ramp preview
        def parse_k4_positions(k4_plain):
            ramps = []
            if not k4_plain:
                return ramps
            for obj in k4_plain.split(";"):
                obj = obj.strip()
                if not obj:
                    continue
                parts = obj.split(",")
                if len(parts) < 6:
                    continue
                d = {}
                for i in range(0, len(parts) - 1, 2):
                    k = parts[i].strip()
                    v = parts[i + 1].strip()
                    if not k:
                        continue
                    try:
                        d[int(k)] = v
                    except Exception:
                        continue
                if 2 not in d or 3 not in d or 6 not in d:
                    continue
                try:
                    ramps.append((float(d[2]), float(d[3]), float(d[6])))
                except Exception:
                    pass
            return ramps

        if settings is not None:
            try:
                if bool(settings.path.start_as_wave):
                    above, below = wave_make_clones(
                        y_samp=level.y_samp,
                        gap_units=settings.path.wave_clone_gap_units,
                        y_min=settings.path.y_min,
                        y_max=settings.path.y_max,
                        y_ceil=settings.path.y_ceil
                    )

                    if bool(settings.path.wave_place_ramps) and above is not None and below is not None:
                        ramp_spacing = float(settings.path.wave_ramp_size_units) * math.sqrt(2.0)

                        k4_top = build_ramps_along_path_by_spacing(
                            times_s=level.t_samp,
                            y_s=above + Y_SHIFT,
                            units_per_second=level.units_per_second,
                            start_offset_s=settings.path.start_offset_s,
                            spacing_units=ramp_spacing,
                            ramp_id=settings.path.wave_ramp_id,
                            y_add=15.0,
                            extra_rot_deg=float(settings.path.wave_ramp_extra_rotation_deg),
                            rotate_180=True,
                            invert_if_top=bool(settings.path.wave_ramp_invert_top),
                            invert_if_bottom=bool(settings.path.wave_ramp_invert_bottom)
                        )

                        k4_bottom = build_ramps_along_path_by_spacing(
                            times_s=level.t_samp,
                            y_s=below + Y_SHIFT,
                            units_per_second=level.units_per_second,
                            start_offset_s=settings.path.start_offset_s,
                            spacing_units=ramp_spacing,
                            ramp_id=settings.path.wave_ramp_id,
                            y_add=15.0,
                            extra_rot_deg=float(settings.path.wave_ramp_extra_rotation_deg),
                            rotate_180=False,
                            invert_if_top=bool(settings.path.wave_ramp_invert_top),
                            invert_if_bottom=bool(settings.path.wave_ramp_invert_bottom)
                        )

                        top_ramps = parse_k4_positions(k4_top)
                        bot_ramps = parse_k4_positions(k4_bottom)

                        base_len = float(settings.path.wave_ramp_size_units) * math.sqrt(2.0)
                        base_len = max(12.0, min(120.0, base_len))

                        pen_top = cosmetic_pen(QColor(255, 170, 60), 2)
                        pen_bottom = cosmetic_pen(QColor(80, 170, 255), 2)

                        # Batch ramp triangles into two QPainterPath items
                        def draw_ramps_batched(ramps, pen, y_src, prefer_lower_apex, invert_flag):
                            want_lower = not (bool(prefer_lower_apex) ^ bool(invert_flag))
                            path = QPainterPath()
                            for rx, ry, rot_deg in ramps:
                                ry -= 15.0
                                i = int(np.searchsorted(x, rx))
                                i = max(1, min(i, len(y_src) - 2))

                                dy_prev = float(y_src[i] - y_src[i - 1])
                                dy_next = float(y_src[i + 1] - y_src[i])
                                kink = (dy_prev != 0.0 and dy_next != 0.0 and (dy_prev > 0) != (dy_next > 0))
                                L = base_len * (0.65 if kink else 1.0)

                                ang = math.radians(rot_deg + 45.0)
                                ux = math.cos(ang)
                                uy = math.sin(ang)
                                hx = ux * (L * 0.5)
                                hy = uy * (L * 0.5)

                                x1r, y1r = rx - hx, ry - hy
                                x2r, y2r = rx + hx, ry + hy
                                px2, py2 = -uy, ux
                                vxa = rx + px2 * (L * 0.5)
                                vya = ry + py2 * (L * 0.5)
                                vxb = rx - px2 * (L * 0.5)
                                vyb = ry - py2 * (L * 0.5)
                                if want_lower:
                                    vx, vy = (vxa, vya) if (vya < vyb) else (vxb, vyb)
                                else:
                                    vx, vy = (vxa, vya) if (vya > vyb) else (vxb, vyb)

                                path.moveTo(x1r, y1r); path.lineTo(x2r, y2r)
                                path.moveTo(x1r, y1r); path.lineTo(vx,  vy)
                                path.moveTo(x2r, y2r); path.lineTo(vx,  vy)

                            sc.addPath(path, pen)

                        draw_ramps_batched(top_ramps, pen_top,    above, True,  bool(settings.path.wave_ramp_invert_top))
                        draw_ramps_batched(bot_ramps, pen_bottom, below, False, bool(settings.path.wave_ramp_invert_bottom))

            except Exception:
                pass

        # scene rect
        margin_x = margin_y = 200.0
        final_x_min = min(float(x.min()), 0.0) - margin_x
        final_x_max = float(x.max()) + margin_x
        final_y_min = float(y.min()) - margin_y
        final_y_max = float(y.max()) + margin_y
        sc.setSceneRect(final_x_min, final_y_min, final_x_max - final_x_min, final_y_max - final_y_min)

        self.report_visible_range()


class AudioPlot(QWidget):
    seekRequested = Signal(float) # emits time in seconds

    def __init__(self):
        super().__init__()

        pg.setConfigOptions(antialias=True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.wave = pg.PlotWidget()
        self.env  = pg.PlotWidget()

        self.wave.setBackground((32, 32, 40))
        self.env.setBackground((32, 32, 40))

        self.wave.showGrid(x=True, y=True, alpha=0.2)
        self.env.showGrid(x=True, y=True, alpha=0.2)

        self.wave.setLabel("bottom", "Time", units="s")
        self.wave.setLabel("left", "Amplitude")
        self.env.setLabel("bottom", "Time", units="s")
        self.env.setLabel("left", "Env / Threshold")

        self.wave.setTitle("Waveform (full song)")
        self.env.setTitle("Envelope used for peak detection")

        layout.addWidget(self.wave, 2)
        layout.addWidget(self.env, 1)

        self._wave_curve = self.wave.plot([], [], pen=pg.mkPen((235, 235, 235), width=1))
        self._env_curve  = self.env.plot([],  [], pen=pg.mkPen((180, 220, 255), width=2))

        self._thr_line = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen((255, 120, 120), width=2))
        self.env.addItem(self._thr_line)

        self._peak_item_wave = None
        self._peak_item_env  = None

        self._playhead_wave = pg.InfiniteLine(pos=0, angle=90, movable=False,
                                               pen=pg.mkPen((0, 230, 255), width=2))
        self._playhead_env  = pg.InfiniteLine(pos=0, angle=90, movable=False,
                                               pen=pg.mkPen((0, 230, 255), width=2))
        self.wave.addItem(self._playhead_wave)
        self.env.addItem(self._playhead_env)
        self._playhead_wave.setVisible(False)
        self._playhead_env.setVisible(False)

        # highlight region for sync with Preview
        self._view_region = pg.LinearRegionItem(
            values=(0, 0),
            orientation=pg.LinearRegionItem.Vertical,
            movable=False,
            brush=pg.mkBrush(255, 0, 0, 40),
            pen=pg.mkPen(None)
        )
        self.wave.addItem(self._view_region)

        self._view_region_env = pg.LinearRegionItem(
            values=(0, 0),
            orientation=pg.LinearRegionItem.Vertical,
            movable=False,
            brush=pg.mkBrush(255, 0, 0, 40),
            pen=pg.mkPen(None)
        )
        self.env.addItem(self._view_region_env)

        # Intercept double-clicks for seek on both plots
        self.wave.viewport().installEventFilter(self)
        self.env.viewport().installEventFilter(self)

    def eventFilter(self, obj, event):
        from PySide6.QtCore import QEvent
        for pw in (self.wave, self.env):
            if obj is pw.viewport() and event.type() == QEvent.MouseButtonDblClick:
                vb = pw.plotItem.vb
                mouse_pt = vb.mapSceneToView(event.pos())
                t = max(0.0, float(mouse_pt.x()))
                self.seekRequested.emit(t)
                return True
        return super().eventFilter(obj, event)

    def set_playhead(self, t, visible=True):
        self._playhead_wave.setVisible(visible)
        self._playhead_env.setVisible(visible)
        if visible:
            self._playhead_wave.setValue(t)
            self._playhead_env.setValue(t)

    def set_visible_range(self, t_start, t_end):
        self._view_region.setRegion([t_start, t_end])
        self._view_region_env.setRegion([t_start, t_end])

    def _clear_peaks(self):
        """Remove the batched peak PlotDataItems."""
        if self._peak_item_wave is not None:
            self.wave.removeItem(self._peak_item_wave)
            self._peak_item_wave = None
        if self._peak_item_env is not None:
            self.env.removeItem(self._peak_item_env)
            self._peak_item_env = None

    def set_audio(self, y, sr, env, t_env, peak_times, thr):
        y = np.asarray(y, dtype=np.float32)
        env = np.asarray(env, dtype=np.float32)
        t_env = np.asarray(t_env, dtype=np.float64)
        peak_times = np.asarray(peak_times, dtype=np.float64)

        n = len(y)
        if n <= 0:
            return

        dur = n / float(sr)
        max_points = 200_000
        step = max(1, n // max_points)

        y_ds = y[::step]
        t_ds = (np.arange(len(y_ds), dtype=np.float64) * step) / float(sr)

        self._wave_curve.setData(t_ds, y_ds)
        self._env_curve.setData(t_env, env)
        self._thr_line.setValue(float(thr))

        self._clear_peaks()

        # Batch all peak lines into a single PlotDataItem per plot using NaN separators.
        # replaces N individual InfiniteLines with 2 total plot items (fast)
        if len(peak_times) > 0:
            y_lo_wave = float(y_ds.min()) * 1.2 if y_ds.min() < 0 else -float(abs(y_ds).max()) * 1.2
            y_hi_wave = float(abs(y_ds).max()) * 1.2
            y_lo_env  = 0.0
            y_hi_env  = float(env.max()) * 1.2 if env.max() > 0 else 1.0

            # build arrays with NaN gaps: [t, t, NaN, t, t, NaN, ...]
            n_peaks = len(peak_times)
            peaks_x = np.empty(n_peaks * 3, dtype=np.float64)
            peaks_x[0::3] = peak_times
            peaks_x[1::3] = peak_times
            peaks_x[2::3] = np.nan

            peaks_y_wave = np.empty(n_peaks * 3, dtype=np.float64)
            peaks_y_wave[0::3] = y_lo_wave
            peaks_y_wave[1::3] = y_hi_wave
            peaks_y_wave[2::3] = np.nan

            peaks_y_env = np.empty(n_peaks * 3, dtype=np.float64)
            peaks_y_env[0::3] = y_lo_env
            peaks_y_env[1::3] = y_hi_env
            peaks_y_env[2::3] = np.nan

            peak_pen = pg.mkPen((120, 255, 160), width=1)
            self._peak_item_wave = self.wave.plot(peaks_x, peaks_y_wave, pen=peak_pen)
            self._peak_item_env  = self.env.plot(peaks_x, peaks_y_env,  pen=peak_pen)

        self.wave.setXRange(0.0, dur, padding=0.01)
        self.env.setXRange(0.0, dur, padding=0.01)


MODERN_STYLESHEET = """
/* Global Reset and Base Colors */
QWidget {
    background-color: #2b2b36;
    color: #e0e0e0;
    font-family: 'Segoe UI', sans-serif;
    font-size: 10pt;
}

/* Scrollbars */
QScrollBar:vertical {
    border: none;
    background: #2b2b36;
    width: 10px;
    margin: 0px 0px 0px 0px;
}
QScrollBar::handle:vertical {
    background: #4a4a58;
    min-height: 20px;
    border-radius: 5px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    border: none;
    background: none;
}

QScrollBar:horizontal {
    border: none;
    background: #2b2b36;
    height: 10px;
    margin: 0px 0px 0px 0px;
}
QScrollBar::handle:horizontal {
    background: #4a4a58;
    min-width: 20px;
    border-radius: 5px;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    border: none;
    background: none;
}

/* Splitter */
QSplitter::handle {
    background-color: #3b3b48;
}
QSplitter::handle:hover {
    background-color: #007acc;
}

/* Buttons */
QPushButton {
    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #3d3d4a, stop:1 #32323d);
    border: 1px solid #4a4a58;
    border-radius: 6px;
    color: #ffffff;
    padding: 6px 16px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #4a4a58, stop:1 #3d3d4a);
    border: 1px solid #5a5a68;
}
QPushButton:pressed {
    background-color: #25252e;
    border: 1px solid #007acc;
}

/* Input Fields */
QLineEdit, QSpinBox, QDoubleSpinBox {
    background-color: #202028;
    border: 1px solid #3b3b48;
    border-radius: 4px;
    padding: 4px;
    padding-right: 20px;
    selection-background-color: #007acc;
    color: #eeeeee;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border: 1px solid #007acc;
}

/* SpinBox Buttons */
QSpinBox::up-button, QDoubleSpinBox::up-button {
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 18px;
    border-left: 1px solid #3b3b48;
    background: #2b2b36;
    border-top-right-radius: 4px;
}
QSpinBox::down-button, QDoubleSpinBox::down-button {
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    width: 18px;
    border-left: 1px solid #3b3b48;
    background: #2b2b36;
    border-bottom-right-radius: 4px;
}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
    background: #3d3d4a;
}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-bottom: 4px solid #cccccc;
    width: 0;
    height: 0;
    margin: 4px;
}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 4px solid #cccccc;
    width: 0;
    height: 0;
    margin: 4px;
}

/* Form Layout Labels */
QLabel {
    color: #cccccc;
}

/* Checkboxes */
QCheckBox {
    spacing: 5px;
    color: #cccccc;
}
QCheckBox::indicator {
    width: 18px;
    height: 18px;
    background-color: #202028;
    border: 1px solid #3b3b48;
    border-radius: 3px;
}
QCheckBox::indicator:checked {
    background-color: #007acc;
    border: 1px solid #007acc;
    image: url(checkbox_check.png);
}
QCheckBox::indicator:hover {
    border: 1px solid #007acc;
}
"""


def apply_modern_theme(app):
    app.setStyle("Fusion")

    palette = QPalette()
    dark_bg = QColor(43, 43, 54)
    palette.setColor(QPalette.Window, dark_bg)
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(32, 32, 40))
    palette.setColor(QPalette.AlternateBase, dark_bg)
    palette.setColor(QPalette.ToolTipBase, Qt.white)
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, dark_bg)
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(palette)

    app.setStyleSheet(MODERN_STYLESHEET)

from PySide6.QtCore import QObject, Signal as _Signal

class _AudioWorker(QObject):
    done = _Signal(object) # emits the (y, sr, env, t_env, peak_times, thr) tuple
    error = _Signal(str)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GDForge")
        self.resize(1200, 800)

        self.settings = AppSettings()
        self.audio_path = None
        self.times = np.array([], dtype=np.float64)
        self.level = None

        self._play_start_wall = None
        self._play_start_t = 0.0
        self._play_duration = 0.0
        self._is_playing = False
        self._play_stream = None
        self._play_sf_handle = None
        self._playhead_timer  = QTimer(self)
        self._playhead_timer.setInterval(30) # ~33fps
        self._playhead_timer.timeout.connect(self._on_playhead_tick)

        # Debounce timer: waits 300ms after last param change before regenerating.
        # Prevents analyze_audio from running on every spinbox tick.
        self._regen_timer = QTimer(self)
        self._regen_timer.setSingleShot(True)
        self._regen_timer.setInterval(300)
        self._regen_timer.timeout.connect(self._regen_all_now)

        # worker signal bridge
        self._audio_worker = _AudioWorker()
        self._audio_worker.done.connect(self._apply_audio_result)
        self._audio_worker.error.connect(lambda msg: self.statusBar().showMessage(f"Error: {msg}", 8000))

        # main splitter layout
        main_splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(main_splitter)

        # left Sidebar
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame) if hasattr(QFrame, 'NoFrame') else None
        scroll.setStyleSheet("QScrollArea { border: none; background-color: transparent; }")

        left_content = QWidget()
        left_content_layout = QVBoxLayout(left_content)
        left_content_layout.setAlignment(Qt.AlignTop)
        left_content_layout.setContentsMargins(15, 15, 15, 15)
        left_content_layout.setSpacing(15)

        scroll.setWidget(left_content)
        left_layout.addWidget(scroll)

        left_container.setStyleSheet("background-color: #2b2b36;")

        self.lbl_audio = QLabel("Audio: (none)")
        btn_load = QPushButton("Load audio")
        btn_load.clicked.connect(self.on_load_audio)
        btn_load.setCursor(Qt.PointingHandCursor)

        form = QFormLayout()
        form.setSpacing(10)

        self.ed_level_name = QLineEdit(self.settings.export.level_name)
        self.ed_creator    = QLineEdit(self.settings.export.creator_name)

        self.sp_song = QSpinBox()
        self.sp_song.setRange(1, 10**9)
        self.sp_song.setValue(self.settings.export.song_id)

        form.addRow("Level name", self.ed_level_name)
        form.addRow("Creator", self.ed_creator)
        form.addRow("Song ID", self.sp_song)

        self.sp_percentile = QDoubleSpinBox()
        self.sp_percentile.setRange(1.0, 99.0)
        self.sp_percentile.setDecimals(1)
        self.sp_percentile.setSingleStep(1.0)
        self.sp_percentile.setValue(self.settings.peaks.peak_percentile)
        self.sp_percentile.setKeyboardTracking(False)
        self.sp_percentile.valueChanged.connect(self.on_params_changed)

        self.chk_onset = QCheckBox("Use onset envelope")
        self.chk_onset.setChecked(self.settings.peaks.use_onset_env)
        self.chk_onset.stateChanged.connect(self.on_params_changed)

        self.chk_start_as_wave = QCheckBox("Start as wave")
        self.chk_start_as_wave.setChecked(bool(self.settings.path.start_as_wave))
        self.chk_start_as_wave.stateChanged.connect(self.on_params_changed)

        self.sp_wave_gap = QDoubleSpinBox()
        self.sp_wave_gap.setRange(0.0, 5000.0)
        self.sp_wave_gap.setDecimals(1)
        self.sp_wave_gap.setSingleStep(10.0)
        self.sp_wave_gap.setValue(self.settings.path.wave_clone_gap_units)
        self.sp_wave_gap.setKeyboardTracking(False)
        self.sp_wave_gap.valueChanged.connect(self.on_params_changed)

        self.chk_wave_ramps = QCheckBox("Place ramps on wave rails")
        self.chk_wave_ramps.setChecked(self.settings.path.wave_place_ramps)
        self.chk_wave_ramps.stateChanged.connect(self.on_params_changed)

        form.addRow("Peak percentile", self.sp_percentile)
        form.addRow("", self.chk_onset)
        form.addRow("", self.chk_start_as_wave)
        form.addRow("Wave rail gap", self.sp_wave_gap)
        form.addRow("", self.chk_wave_ramps)

        btn_export = QPushButton("Export .gmd")
        btn_export.clicked.connect(self.on_export)

        # Playback controls
        playback_layout = QHBoxLayout()
        self.btn_play = QPushButton("Play")
        self.btn_play.clicked.connect(self.on_play_stop)
        self.btn_play.setEnabled(False)
        self.btn_play.setCursor(Qt.PointingHandCursor)
        self.btn_play.setStyleSheet("""
            QPushButton { background-color: #1a6b2e; border-color: #2d9e45; color: #ffffff; }
            QPushButton:hover { background-color: #217a35; }
            QPushButton:pressed { background-color: #155526; }
            QPushButton:disabled { background-color: #333; color: #777; }
        """)
        self.lbl_time = QLabel("0:00.0 / 0:00.0")
        self.lbl_time.setStyleSheet("color: #aaaaaa; font-family: monospace;")
        playback_layout.addWidget(self.btn_play)
        playback_layout.addWidget(self.lbl_time)

        self.lbl_stats = QLabel("Stats:")

        self.lbl_help = QLabel(
            "Preview meaning:\n"
            "Waveform: raw audio amplitude\n"
            "Envelope: signal used for peak detection\n"
            "Red line: peak threshold\n"
            "Green lines: detected peaks\n"
            "Level preview: white line = wave path, ticks = peaks"
        )
        self.lbl_help.setStyleSheet("color: #bbbbbb; font-style: italic;")

        left_content_layout.addWidget(self.lbl_audio)
        left_content_layout.addWidget(btn_load)
        left_content_layout.addSpacing(10)
        left_content_layout.addLayout(form)
        left_content_layout.addSpacing(10)
        left_content_layout.addWidget(btn_export)
        left_content_layout.addLayout(playback_layout)
        left_content_layout.addWidget(self.lbl_stats)
        left_content_layout.addWidget(self.lbl_help)
        left_content_layout.addStretch()

        self.preview = Preview()
        self.audio_plot = AudioPlot()

        self.preview.visibleRangeChanged.connect(self.audio_plot.set_visible_range)
        self.preview.seekRequested.connect(self._seek_to)
        self.audio_plot.seekRequested.connect(self._seek_to)

        right = QSplitter(Qt.Vertical)
        right.setHandleWidth(8)
        right.addWidget(self.audio_plot)
        right.addWidget(self.preview)

        main_splitter.addWidget(left_container)
        main_splitter.addWidget(right)
        main_splitter.setSizes([350, 850])
        main_splitter.setCollapsible(0, False)
        right.setSizes([450, 350])

    def pull_ui_to_settings(self):
        self.settings.export.level_name = self.ed_level_name.text().strip()
        self.settings.export.creator_name = self.ed_creator.text().strip()
        self.settings.export.song_id = int(self.sp_song.value())

        self.settings.peaks.peak_percentile = float(self.sp_percentile.value())
        self.settings.peaks.use_onset_env = bool(self.chk_onset.isChecked())

        self.settings.path.wave_ramp_extra_rotation_deg = 0.0
        self.settings.path.wave_clone_gap_units = float(self.sp_wave_gap.value())
        self.settings.path.wave_place_ramps = bool(self.chk_wave_ramps.isChecked())
        self.settings.path.start_as_wave = bool(self.chk_start_as_wave.isChecked())

        self.update_wave_controls_enabled()

    def update_wave_controls_enabled(self):
        on = bool(self.chk_start_as_wave.isChecked())
        self.sp_wave_gap.setEnabled(on)
        self.chk_wave_ramps.setEnabled(on)

    def on_load_audio(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select audio", "", "Audio (*.mp3 *.wav *.ogg *.flac);;All files (*.*)"
        )
        if not path:
            return
        if self._is_playing:
            self._stop_playback()
        self.audio_path = path
        self.lbl_audio.setText(f"Audio: {path}")
        self.regen_all()

    def on_params_changed(self):
        if not self.audio_path:
            return
        self._regen_timer.start()

    def regen_all(self):
        self._regen_timer.stop()
        self._regen_all_now()

    def _regen_all_now(self):
        if not self.audio_path:
            return

        self.pull_ui_to_settings()
        self.statusBar().showMessage("Analyzing audio...")

        audio_path = self.audio_path
        ps = self.settings.peaks
        worker = self._audio_worker

        def _thread_fn():
            try:
                result = analyze_audio(audio_path, ps)
                worker.done.emit(result)
            except Exception as exc:
                worker.error.emit(str(exc))

        threading.Thread(target=_thread_fn, daemon=True).start()

    def _apply_audio_result(self, result):
        # Called on the main thread via the _AudioWorker.done signal.
        y, sr, env, t_env, peak_times, thr = result
        self.times = peak_times
        self.audio_plot.set_audio(y, sr, env, t_env, peak_times, thr)

        duration_s = float(len(y)) / float(sr) if sr else 0.0
        self._play_duration = duration_s
        self.btn_play.setEnabled(True)

        self.level = generate_level(self.times, self.settings, end_time_s=duration_s)
        self.preview.show_level(self.level, self.settings)
        self.lbl_stats.setText(
            f"Stats: peaks={len(self.times)}  "
            f"samples={len(self.level.t_samp)}  "
            f"ups={self.level.units_per_second:.2f}"
        )
        self.statusBar().showMessage("Ready.", 3000)

    def _seek_to(self, t):
        if not self.audio_path:
            return
        t = max(0.0, min(t, self._play_duration))
        self._stop_playback()
        self.audio_plot.set_playhead(t, visible=True)
        self.preview.set_playhead(t, visible=True, auto_scroll=True)
        self.lbl_time.setText(f"{self._fmt_time(t)} / {self._fmt_time(self._play_duration)}")
        self._start_playback(start_t=t)
        self._playhead_timer.start()
        self.btn_play.setFocus()

    def on_play_stop(self):
        if self._is_playing:
            self._stop_playback()
            self._set_btn_play()
            self.audio_plot.set_playhead(0, visible=False)
            self.preview.set_playhead(0, visible=False)
            self.lbl_time.setText(f"0:00.0 / {self._fmt_time(self._play_duration)}")
        else:
            self._start_playback(start_t=0.0)
            self._playhead_timer.start()

    def _set_btn_play(self):
        self.btn_play.setText("Play")
        self.btn_play.setStyleSheet("""
            QPushButton { background-color: #1a6b2e; border-color: #2d9e45; color: #ffffff; }
            QPushButton:hover { background-color: #217a35; }
            QPushButton:pressed { background-color: #155526; }
            QPushButton:disabled { background-color: #333; color: #777; }
        """)

    def _set_btn_stop(self):
        self.btn_play.setText("⏹  Stop")
        self.btn_play.setStyleSheet("""
            QPushButton { background-color: #7a2020; border-color: #b03030; color: #ffffff; }
            QPushButton:hover { background-color: #8a2828; }
            QPushButton:pressed { background-color: #5a1818; }
        """)


    def _start_playback(self, start_t=0.0):
        if not self.audio_path:
            return
        try:
            info = sf.info(self.audio_path)
            file_sr = info.samplerate
            channels = info.channels
        except Exception as e:
            self.statusBar().showMessage(f"Cannot open audio: {e}", 5000)
            return

        start_frame = max(0, int(start_t * file_sr))
        try:
            sf_handle = sf.SoundFile(self.audio_path)
            sf_handle.seek(start_frame)
        except Exception as e:
            self.statusBar().showMessage(f"Cannot open audio: {e}", 5000)
            return

        CHUNK = 512

        def _audio_callback(outdata, frames, time_info, status):
            data = sf_handle.read(frames, dtype='float32', always_2d=True)
            if len(data) < frames:
                outdata[:len(data)] = data
                if len(data) < frames:
                    outdata[len(data):] = 0
                raise sd.CallbackStop()
            outdata[:] = data

        def _finished_callback():
            self._is_playing = False

        try:
            stream = sd.OutputStream(
                samplerate=file_sr,
                channels=channels,
                dtype='float32',
                blocksize=CHUNK,
                callback=_audio_callback,
                finished_callback=_finished_callback,
            )
        except Exception as e:
            self.statusBar().showMessage(f"Cannot open audio stream: {e}", 5000)
            sf_handle.close()
            return

        self._play_sf_handle = sf_handle
        self._play_stream = stream
        self._play_start_t = start_t
        self._play_duration = info.frames / float(file_sr)
        self._is_playing = True          # set BEFORE stream.start()
        self._set_btn_stop()
        stream.start()
        self._play_start_wall = _time.perf_counter() + stream.latency

    def _stop_playback(self):
        self._is_playing = False
        try:
            if self._play_stream is not None:
                self._play_stream.stop()
                self._play_stream.close()
        except Exception:
            pass
        self._play_stream = None
        try:
            if hasattr(self, '_play_sf_handle') and self._play_sf_handle is not None:
                self._play_sf_handle.close()
        except Exception:
            pass
        self._play_sf_handle = None
        try:
            sd.stop()
        except Exception:
            pass
        self._playhead_timer.stop()

    def _on_playhead_tick(self):
        if not self._is_playing or self._play_start_wall is None:
            self._stop_playback()
            self._set_btn_play()
            self.audio_plot.set_playhead(0, visible=False)
            self.preview.set_playhead(0, visible=False)
            self.lbl_time.setText(f"0:00.0 / {self._fmt_time(self._play_duration)}")
            return
        elapsed = _time.perf_counter() - self._play_start_wall + self._play_start_t
        if elapsed >= self._play_duration:
            self._stop_playback()
            self._set_btn_play()
            self.audio_plot.set_playhead(0, visible=False)
            self.preview.set_playhead(0, visible=False)
            self.lbl_time.setText(f"0:00.0 / {self._fmt_time(self._play_duration)}")
            return
        self.audio_plot.set_playhead(elapsed, visible=True)
        self.preview.set_playhead(elapsed, visible=True, auto_scroll=True)
        self.lbl_time.setText(f"{self._fmt_time(elapsed)} / {self._fmt_time(self._play_duration)}")

    @staticmethod
    def _fmt_time(t):
        m = int(t) // 60
        s = t - m * 60
        return f"{m}:{s:04.1f}"

    def on_export(self):
        if self.audio_path is None or self.level is None:
            self.statusBar().showMessage("Load audio and generate first.", 5000)
            return

        self.pull_ui_to_settings()

        out_path, _ = QFileDialog.getSaveFileName(
            self, "Export .gmd", "", "GMD (*.gmd);;All files (*.*)"
        )
        if not out_path:
            return
        if not out_path.lower().endswith(".gmd"):
            out_path += ".gmd"

        if bool(self.settings.path.start_as_wave):
            y_ceil = self.settings.path.wave_corridor_units * 30.0
            y_samp_clamped = np.clip(self.level.y_samp, 15.0, y_ceil - 15.0)
            k4 = build_k4_polyline(
                self.level.t_samp,
                y_samp_clamped,
                self.level.units_per_second,
                self.settings.path.start_offset_s,
                start_as_wave=True,
                spacing_units=self.settings.path.block_spacing_units,
                y_add=0.0,
            )
        else:
            y_samp_clamped = np.maximum(self.level.y_samp, 15.0)
            k4 = build_k4_orb_arc(
                self.level.t_samp,
                y_samp_clamped,
                self.level.units_per_second,
                self.settings.path.start_offset_s,
                orb_times=self.level.orb_times,
                orb_types=self.level.orb_types,
                y_add=0.0,
                orb_id_yellow=36,
                orb_id_purple=141,
                orb_id_blue=84,
                orb_id_green=1022,
                cube_portal_id=12,
                start_with_cube=True,
                orb_x_offset=0.0,
                orb_y_offset=0.0,
                spacing_units=self.settings.path.block_spacing_units,
            )

        if self.settings.path.start_as_wave:
            above, below = wave_make_clones(
                y_samp=self.level.y_samp,
                gap_units=self.settings.path.wave_clone_gap_units,
                y_min=self.settings.path.y_min,
                y_max=self.settings.path.y_max,
                y_ceil=self.settings.path.y_ceil
            )

            if self.settings.path.wave_place_ramps and above is not None and below is not None:
                ramp_spacing = float(self.settings.path.wave_ramp_size_units) * math.sqrt(2.0)

                k4 += build_ramps_along_path_by_spacing(
                    times_s=self.level.t_samp,
                    y_s=above,
                    units_per_second=self.level.units_per_second,
                    start_offset_s=self.settings.path.start_offset_s,
                    spacing_units=ramp_spacing,
                    ramp_id=self.settings.path.wave_ramp_id,
                    y_add=0.0,
                    extra_rot_deg=float(self.settings.path.wave_ramp_extra_rotation_deg),
                    rotate_180=True,
                    invert_if_top=bool(self.settings.path.wave_ramp_invert_top),
                    invert_if_bottom=bool(self.settings.path.wave_ramp_invert_bottom)
                )

                k4 += build_ramps_along_path_by_spacing(
                    times_s=self.level.t_samp,
                    y_s=below,
                    units_per_second=self.level.units_per_second,
                    start_offset_s=self.settings.path.start_offset_s,
                    spacing_units=ramp_spacing,
                    ramp_id=self.settings.path.wave_ramp_id,
                    y_add=0.0,
                    extra_rot_deg=float(self.settings.path.wave_ramp_extra_rotation_deg),
                    rotate_180=False,
                    invert_if_top=bool(self.settings.path.wave_ramp_invert_top),
                    invert_if_bottom=bool(self.settings.path.wave_ramp_invert_bottom)
                )

        level_xml = build_level_xml(
            level_name=self.settings.export.level_name,
            creator_name=self.settings.export.creator_name,
            k4_plain=k4,
            custom_song_id=self.settings.export.song_id
        )
        final_xml = '<?xml version="1.0"?><plist version="1.0" gjver="2.0">' + level_xml + '</plist>'

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(final_xml)

        from PySide6.QtWidgets import QMessageBox
        msg = QMessageBox(self)
        msg.setWindowTitle("Exported")
        msg.setText(f"Level exported to:\n{out_path}\n\nImport the .gmd file into Geometry Dash using GDShare.")
        msg.setIcon(QMessageBox.Information)
        msg.exec()

        gd_dir = os.path.expandvars(r"%localappdata%\GeometryDash")
        if os.path.isdir(gd_dir):
            target_mp3 = os.path.join(gd_dir, f"{self.settings.export.song_id}.mp3")
            try:
                shutil.copy2(self.audio_path, target_mp3)
            except Exception as e:
                self.statusBar().showMessage(f"Exported .gmd, but failed to copy song: {e}", 7000)
                return

        self.statusBar().showMessage(f"Exported: {out_path}", 7000)


def main():
    print("===========================================")
    print("               GDForge - app.py            ")
    print("    Copyright (c) 2025-2026 Hannes Göök    ")
    print("           MIT License - GDForge           ")
    print("   https://github.com/hannesgook/gdforge   ")
    print("===========================================")

    app = QApplication(sys.argv)
    apply_modern_theme(app)
    w = MainWindow()

    # Fade-in animation
    w.setWindowOpacity(0)
    w.show()
    anim = QPropertyAnimation(w, b"windowOpacity")
    anim.setDuration(600)
    anim.setStartValue(0)
    anim.setEndValue(1)
    anim.setEasingCurve(QEasingCurve.OutCubic)
    anim.start()

    w._intro_anim = anim

    sys.exit(app.exec())


if __name__ == "__main__":
    main()