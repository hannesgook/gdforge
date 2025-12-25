from dataclasses import dataclass
import numpy as np
import math
from settings import AppSettings

@dataclass
class LevelData:
    times: np.ndarray
    t_samp: np.ndarray
    y_samp: np.ndarray
    units_per_second: float

def sample_wave_by_dx(times_s, units_per_second, dx_units, y_start, y_min, y_max, y_ceil, angle_deg, start_dir_up, corridor_units, margin_units):
    if len(times_s) == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    U = float(units_per_second)
    dt = float(dx_units) / U
    m = math.tan(math.radians(angle_deg))

    ceiling_units = min(float(y_ceil), float(corridor_units) * 30.0)

    out_t = [float(times_s[0])]
    out_y = [max(0.0, float(y_start))]

    dir_up = bool(start_dir_up)

    for i in range(len(times_s) - 1):
        t0 = float(times_s[i])
        t1 = float(times_s[i + 1])
        t = t0 + dt
        y_cur = out_y[-1]

        while t < t1 - 1e-9:
            if dir_up and y_cur >= ceiling_units - margin_units:
                dir_up = False
            elif (not dir_up) and y_cur <= margin_units:
                dir_up = True

            s = 1.0 if dir_up else -1.0
            y_cur = y_cur + s * m * (dt * U)

            y_cur = max(0.0, min(float(y_ceil), y_cur))
            y_cur = max(float(y_min), min(float(y_max), y_cur))

            out_t.append(t)
            out_y.append(y_cur)
            t += dt

        out_t.append(t1)
        out_y.append(out_y[-1])

        dir_up = not dir_up

    return np.asarray(out_t, dtype=np.float64), np.asarray(out_y, dtype=np.float64)

def generate_level(times: np.ndarray, settings: AppSettings) -> LevelData:
    ups = settings.units_per_second()

    if len(times) == 0:
        return LevelData(times=times, t_samp=np.array([], dtype=np.float64), y_samp=np.array([], dtype=np.float64), units_per_second=ups)

    p = settings.path
    dx = p.wave_dx_units if p.wave_dx_units is not None else p.dx_units

    t_samp, y_samp = sample_wave_by_dx(
        times_s=times,
        units_per_second=ups,
        dx_units=dx,
        y_start=max(0.0, p.y_start),
        y_min=p.y_min,
        y_max=p.y_max,
        y_ceil=p.y_ceil,
        angle_deg=p.wave_angle_deg,
        start_dir_up=p.wave_dir_up,
        corridor_units=p.wave_corridor_units,
        margin_units=p.wave_margin_units,
    )

    return LevelData(times=times, t_samp=t_samp, y_samp=y_samp, units_per_second=ups)
