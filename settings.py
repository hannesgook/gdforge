from dataclasses import dataclass
from typing import Optional

BASE_UNITS_PER_SEC = 311.58

@dataclass
class PeakSettings:
    sr: int = 44100
    hop: int = 256
    frame: int = 1024
    use_onset_env: bool = False
    peak_percentile: float = 75.0
    peak_abs: float = 0.0
    min_sep_s: float = 0.09


@dataclass
class PathSettings:
    speed_mult: float = 1.0
    start_offset_s: float = 0.0

    dx_units: float = 4.0
    y_start: float = 0.0
    y_min: float = -120.0
    y_max: float = 9999.0
    y_ceil: float = 2200.0

    start_as_wave: bool = True
    wave_angle_deg: float = 45.0
    wave_dx_units: Optional[float] = None
    wave_dir_up: bool = True
    wave_corridor_units: float = 9.0
    wave_margin_units: float = 0.5

    wave_clone_gap_units: float = 30.0
    wave_place_ramps: bool = False
    wave_ramp_id: int = 309
    wave_ramp_size_units: float = 30.0
    wave_ramp_extra_rotation_deg: float = 45.0
    wave_ramp_invert_top: bool = False
    wave_ramp_invert_bottom: bool = False

@dataclass
class ExportSettings:
    song_id: int = 777777
    level_name: str = "Auto Map"
    creator_name: str = "Player"

@dataclass
class AppSettings:
    peaks: PeakSettings = PeakSettings()
    path: PathSettings = PathSettings()
    export: ExportSettings = ExportSettings()

    def units_per_second(self) -> float:
        return BASE_UNITS_PER_SEC * float(self.path.speed_mult)
