import sys
import numpy as np
import math
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QDoubleSpinBox, QSpinBox, QCheckBox, QGraphicsView, QGraphicsScene,
    QFormLayout, QLineEdit
)
from PySide6.QtGui import QPainter, QPainterPath, QPen
from PySide6.QtCore import Qt
import pyqtgraph as pg
from PySide6.QtWidgets import QSplitter, QGroupBox
from PySide6.QtGui import QColor
import os, shutil
from gd_serialize import build_wave_ramps_45deg, wave_make_clones, build_ramps_along_path_by_spacing, build_level_xml

from settings import AppSettings
from audio_analysis import analyze_audio
from generator import generate_level
from gd_serialize import build_k4_polyline, serialize_gmd, build_k4_orb_arc

class Preview(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.Antialiasing, True)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setBackgroundBrush(QColor(12, 12, 16))
        self.scale(1.0, -1.0)

    def wheelEvent(self, event):
        factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
        self.scale(factor, factor)

    def show_level(self, level, settings=None):
        sc = self.scene()
        sc.clear()

        if level is None or len(level.t_samp) < 2:
            sc.setSceneRect(-200, -200, 400, 400)
            return

        x = (level.t_samp * level.units_per_second).astype(np.float64)
        y = level.y_samp.astype(np.float64)

        path = QPainterPath()
        path.moveTo(float(x[0]), float(y[0]))
        for i in range(1, len(x)):
            path.lineTo(float(x[i]), float(y[i]))

        sc.addPath(path, QPen(QColor(240, 240, 240), 2))

        for t in level.times:
            xx = float(t * level.units_per_second)
            sc.addLine(xx, -40.0, xx, +40.0, QPen(QColor(120, 120, 120), 1))

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
                            y_s=above,
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
                            y_s=below,
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


                        pen_top = QPen(QColor(255, 170, 60), 2)
                        pen_bottom = QPen(QColor(80, 170, 255), 2)

                        def draw_ramps(ramps, pen, y_src, prefer_lower_apex, invert_flag):
                            want_lower = not (bool(prefer_lower_apex) ^ bool(invert_flag))

                            for rx, ry, rot_deg in ramps:
                                ry -= 15.0

                                i = int(np.searchsorted(x, rx))
                                if i < 1:
                                    i = 1
                                if i > len(y_src) - 2:
                                    i = len(y_src) - 2

                                dy_prev = float(y_src[i] - y_src[i - 1])
                                dy_next = float(y_src[i + 1] - y_src[i])

                                kink = (dy_prev != 0.0 and dy_next != 0.0 and (dy_prev > 0) != (dy_next > 0))
                                L = base_len * (0.65 if kink else 1.0)

                                ang = math.radians(rot_deg + 45.0)
                                ux = math.cos(ang)
                                uy = math.sin(ang)

                                hx = ux * (L * 0.5)
                                hy = uy * (L * 0.5)

                                x1, y1 = rx - hx, ry - hy
                                x2, y2 = rx + hx, ry + hy

                                px, py = -uy, ux
                                vx_a, vy_a = rx + px * (L * 0.5), ry + py * (L * 0.5)
                                vx_b, vy_b = rx - px * (L * 0.5), ry - py * (L * 0.5)

                                if want_lower:
                                    vx, vy = (vx_a, vy_a) if (vy_a < vy_b) else (vx_b, vy_b)
                                else:
                                    vx, vy = (vx_a, vy_a) if (vy_a > vy_b) else (vx_b, vy_b)

                                sc.addLine(x1, y1, x2, y2, pen)
                                sc.addLine(x1, y1, vx, vy, pen)
                                sc.addLine(x2, y2, vx, vy, pen)

                        draw_ramps(top_ramps, pen_top, above, prefer_lower_apex=True,  invert_flag=bool(settings.path.wave_ramp_invert_top))
                        draw_ramps(bot_ramps, pen_bottom, below, prefer_lower_apex=False, invert_flag=bool(settings.path.wave_ramp_invert_bottom))

            except Exception:
                pass

        xmin, xmax = float(x.min()), float(x.max())
        ymin, ymax = float(y.min()), float(y.max())
        pad = 200.0
        sc.setSceneRect(xmin - pad, ymin - pad, (xmax - xmin) + 2 * pad, (ymax - ymin) + 2 * pad)

class AudioPlot(QWidget):
    def __init__(self):
        super().__init__()

        pg.setConfigOptions(antialias=True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.wave = pg.PlotWidget()
        self.env = pg.PlotWidget()

        self.wave.setBackground((12, 12, 16))
        self.env.setBackground((12, 12, 16))

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
        self._env_curve = self.env.plot([], [], pen=pg.mkPen((180, 220, 255), width=2))

        self._thr_line = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen((255, 120, 120), width=2))
        self.env.addItem(self._thr_line)

        self._peak_lines = []

    def _clear_peaks(self):
        for item in self._peak_lines:
            self.env.removeItem(item)
            self.wave.removeItem(item)
        self._peak_lines = []

    def set_audio(self, y, sr, env, t_env, peak_times, thr):
        y = np.asarray(y, dtype=np.float32)
        env = np.asarray(env, dtype=np.float32)
        t_env = np.asarray(t_env, dtype=np.float64)
        peak_times = np.asarray(peak_times, dtype=np.float64)

        # Downsample waveform for speed
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

        # Mark peaks on both plots as vertical lines
        for pt in peak_times:
            lw = pg.InfiniteLine(pos=float(pt), angle=90, movable=False, pen=pg.mkPen((120, 255, 160), width=1))
            le = pg.InfiniteLine(pos=float(pt), angle=90, movable=False, pen=pg.mkPen((120, 255, 160), width=1))
            self.wave.addItem(lw)
            self.env.addItem(le)
            self._peak_lines.append(lw)
            self._peak_lines.append(le)

        self.wave.setXRange(0.0, dur, padding=0.01)
        self.env.setXRange(0.0, dur, padding=0.01)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GD Mapper (PySide6)")

        self.settings = AppSettings()
        self.audio_path = None
        self.times = np.array([], dtype=np.float64)
        self.level = None

        root = QWidget()
        self.setCentralWidget(root)

        layout = QHBoxLayout(root)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setAlignment(Qt.AlignTop)

        self.lbl_audio = QLabel("Audio: (none)")
        btn_load = QPushButton("Load audio")
        btn_load.clicked.connect(self.on_load_audio)

        form = QFormLayout()

        self.ed_level_name = QLineEdit(self.settings.export.level_name)
        self.ed_creator = QLineEdit(self.settings.export.creator_name)

        self.sp_song = QSpinBox()
        self.sp_song.setRange(1, 10**9)
        self.sp_song.setValue(self.settings.export.song_id)

        form.addRow("Level name", self.ed_level_name)
        form.addRow("Creator", self.ed_creator)
        form.addRow("Song ID", self.sp_song)

        self.sp_speed = QDoubleSpinBox()
        self.sp_speed.setRange(0.1, 10.0)
        self.sp_speed.setDecimals(3)
        self.sp_speed.setSingleStep(0.05)
        self.sp_speed.setValue(self.settings.path.speed_mult)
        self.sp_speed.valueChanged.connect(self.on_params_changed)

        self.sp_minsep = QDoubleSpinBox()
        self.sp_minsep.setRange(0.01, 1.0)
        self.sp_minsep.setDecimals(3)
        self.sp_minsep.setSingleStep(0.01)
        self.sp_minsep.setValue(self.settings.peaks.min_sep_s)
        self.sp_minsep.valueChanged.connect(self.on_params_changed)

        self.sp_percentile = QDoubleSpinBox()
        self.sp_percentile.setRange(1.0, 99.0)
        self.sp_percentile.setDecimals(1)
        self.sp_percentile.setSingleStep(1.0)
        self.sp_percentile.setValue(self.settings.peaks.peak_percentile)
        self.sp_percentile.valueChanged.connect(self.on_params_changed)

        self.chk_onset = QCheckBox("Use onset envelope")
        self.chk_onset.setChecked(self.settings.peaks.use_onset_env)
        self.chk_onset.stateChanged.connect(self.on_params_changed)


        self.chk_start_as_wave = QCheckBox("Start as wave")
        self.chk_start_as_wave.setChecked(bool(self.settings.path.start_as_wave))
        self.chk_start_as_wave.stateChanged.connect(self.on_params_changed)


        self.sp_wave_angle = QDoubleSpinBox()
        self.sp_wave_angle.setRange(1.0, 89.0)
        self.sp_wave_angle.setDecimals(1)
        self.sp_wave_angle.setSingleStep(1.0)
        self.sp_wave_angle.setValue(self.settings.path.wave_angle_deg)
        self.sp_wave_angle.valueChanged.connect(self.on_params_changed)

        self.sp_wave_gap = QDoubleSpinBox()
        self.sp_wave_gap.setRange(0.0, 5000.0)
        self.sp_wave_gap.setDecimals(1)
        self.sp_wave_gap.setSingleStep(10.0)
        self.sp_wave_gap.setValue(self.settings.path.wave_clone_gap_units)
        self.sp_wave_gap.valueChanged.connect(self.on_params_changed)

        self.chk_wave_ramps = QCheckBox("Place ramps on wave rails")
        self.chk_wave_ramps.setChecked(self.settings.path.wave_place_ramps)
        self.chk_wave_ramps.stateChanged.connect(self.on_params_changed)

        self.chk_start_as_wave = QCheckBox("Start as wave")
        self.chk_start_as_wave.setChecked(bool(self.settings.path.start_as_wave))
        self.chk_start_as_wave.stateChanged.connect(self.on_params_changed)


        self.sp_ramp_size = QDoubleSpinBox()
        self.sp_ramp_size.setRange(5.0, 200.0)
        self.sp_ramp_size.setDecimals(1)
        self.sp_ramp_size.setSingleStep(5.0)
        self.sp_ramp_size.setValue(self.settings.path.wave_ramp_size_units)
        self.sp_ramp_size.valueChanged.connect(self.on_params_changed)

        form.addRow("Speed mult", self.sp_speed)
        form.addRow("Min sep (s)", self.sp_minsep)
        form.addRow("Peak percentile", self.sp_percentile)
        form.addRow("", self.chk_onset)
        form.addRow("", self.chk_start_as_wave)
        form.addRow("Wave angle", self.sp_wave_angle)
        form.addRow("Wave rail gap", self.sp_wave_gap)
        form.addRow("", self.chk_wave_ramps)
        form.addRow("Ramp size (units)", self.sp_ramp_size)

        btn_gen = QPushButton("Generate preview")
        btn_gen.clicked.connect(self.regen_all)

        btn_export = QPushButton("Export .gmd")
        btn_export.clicked.connect(self.on_export)

        self.lbl_stats = QLabel("Stats: -")

        self.lbl_help = QLabel(
            "Preview meaning:\n"
            "• Waveform: raw audio amplitude\n"
            "• Envelope: signal used for peak detection\n"
            "• Red line: peak threshold\n"
            "• Green lines: detected peaks\n"
            "• Level preview: white line = wave path, ticks = peaks"
        )
        self.lbl_help.setStyleSheet("color: #ddd;")

        left_layout.addWidget(self.lbl_audio)
        left_layout.addWidget(btn_load)
        left_layout.addSpacing(10)
        left_layout.addLayout(form)
        left_layout.addSpacing(10)
        left_layout.addWidget(btn_gen)
        left_layout.addWidget(btn_export)
        left_layout.addWidget(self.lbl_stats)
        left_layout.addWidget(self.lbl_help)

        self.preview = Preview()
        self.audio_plot = AudioPlot()

        right = QSplitter(Qt.Vertical)
        right.addWidget(self.audio_plot)
        right.addWidget(self.preview)
        right.setSizes([450, 350])

        layout.addWidget(left, 0)
        layout.addWidget(right, 1)

    def pull_ui_to_settings(self):
        self.settings.export.level_name = self.ed_level_name.text().strip()
        self.settings.export.creator_name = self.ed_creator.text().strip()
        self.settings.export.song_id = int(self.sp_song.value())

        self.settings.path.speed_mult = float(self.sp_speed.value())
        self.settings.peaks.min_sep_s = float(self.sp_minsep.value())
        self.settings.peaks.peak_percentile = float(self.sp_percentile.value())
        self.settings.peaks.use_onset_env = bool(self.chk_onset.isChecked())

        self.settings.path.wave_angle_deg = float(self.sp_wave_angle.value())
        # Match audio2gmd.py default: ramps get no extra rotation unless explicitly set elsewhere.
        self.settings.path.wave_ramp_extra_rotation_deg = 0.0

        self.settings.path.wave_clone_gap_units = float(self.sp_wave_gap.value())
        self.settings.path.wave_place_ramps = bool(self.chk_wave_ramps.isChecked())
        self.settings.path.wave_ramp_size_units = float(self.sp_ramp_size.value())

        self.settings.path.start_as_wave = bool(self.chk_start_as_wave.isChecked())
        self.update_wave_controls_enabled()


    def update_wave_controls_enabled(self):
        on = bool(self.chk_start_as_wave.isChecked())
        self.sp_wave_angle.setEnabled(on)
        self.sp_wave_gap.setEnabled(on)
        self.chk_wave_ramps.setEnabled(on)
        self.sp_ramp_size.setEnabled(on)

    def on_load_audio(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select audio", "", "Audio (*.mp3 *.wav *.ogg *.flac);;All files (*.*)")
        if not path:
            return
        self.audio_path = path
        self.lbl_audio.setText(f"Audio: {path}")
        self.regen_all()

    def on_params_changed(self):
        if not self.audio_path:
            return
        self.regen_all()

    def regen_all(self):
        if not self.audio_path:
            return
        self.pull_ui_to_settings()

        y, sr, env, t_env, peak_times, thr = analyze_audio(self.audio_path, self.settings.peaks)
        self.times = peak_times
        self.audio_plot.set_audio(y, sr, env, t_env, peak_times, thr)

        duration_s = float(len(y)) / float(sr) if sr else 0.0
        self.level = generate_level(self.times, self.settings, end_time_s=duration_s)


        self.preview.show_level(self.level, self.settings)
        self.lbl_stats.setText(f"Stats: peaks={len(self.times)} samples={len(self.level.t_samp)} ups={self.level.units_per_second:.2f}")

    def on_export(self):
        import math
        import os
        import shutil
        from PySide6.QtWidgets import QFileDialog

        if self.audio_path is None or self.level is None:
            self.statusBar().showMessage("Load audio and generate first.", 5000)
            return

        self.pull_ui_to_settings()

        out_path, _ = QFileDialog.getSaveFileName(self, "Export .gmd", "", "GMD (*.gmd);;All files (*.*)")
        if not out_path:
            return
        if not out_path.lower().endswith(".gmd"):
            out_path += ".gmd"

        if bool(self.settings.path.start_as_wave):
            k4 = build_k4_polyline(
                self.level.t_samp,
                self.level.y_samp,
                self.level.units_per_second,
                self.settings.path.start_offset_s,
                start_as_wave=True
            )
        else:
            k4 = build_k4_orb_arc(
                self.level.t_samp,
                self.level.y_samp,
                self.level.units_per_second,
                self.settings.path.start_offset_s,
                orb_times=self.level.orb_times,
                orb_types=self.level.orb_types,
                y_add=15.0,
                orb_id_yellow=36,
                orb_id_purple=141,
                orb_id_blue=84,
                orb_id_green=1022,
                cube_portal_id=12,
                start_with_cube=True,
                orb_x_offset=0.0,
                orb_y_offset=15.0,
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
                    y_add=15.0,
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
                    y_add=15.0,
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
    app = QApplication(sys.argv)
    w = MainWindow()
    w.resize(1400, 800)
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
