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
    orb_times: np.ndarray
    orb_types: np.ndarray

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

def _choose_orb_types_from_times(times: np.ndarray) -> np.ndarray:
    if len(times) == 0:
        return np.array([], dtype=np.int32)

    dt = np.diff(times)
    if len(dt) == 0:
        return np.array([], dtype=np.int32)

    med = float(np.median(dt))
    types = np.zeros(len(dt), dtype=np.int32)

    for i, d in enumerate(dt):
        if d < med * 0.85:
            types[i] = 1  # pink (smaller impulse)
        else:
            types[i] = 0  # yellow (bigger impulse)

    for i in range(5, len(types), 8):
        types[i] = 2  # blue (gravity flip event)

    return types

def sample_cube_arcs_from_orbs(
    times: np.ndarray,
    orb_types: np.ndarray,
    units_per_second: float,
    dx_units: float,
    y_start: float,
    y_min: float,
    y_max: float,
    y_ceil: float,
    g: float = -2600.0,
    impulse_yellow: float = 1050.0,
    impulse_pink: float = 820.0,
    impulse_blue: float = 1050.0,
    blue_flips_gravity: bool = True,
):
    if len(times) == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    U = float(units_per_second)
    dt = float(dx_units) / U

    t0 = float(times[0])
    t1 = float(times[-1])

    n = int(max(2, math.ceil((t1 - t0) / dt))) + 1
    t_samp = np.linspace(t0, t1, n, dtype=np.float64)

    y_samp = np.empty_like(t_samp)
    y_samp[0] = max(0.0, float(y_start))

    vy = 0.0
    g_cur = float(g)
    next_orb_i = 0

    for i in range(1, len(t_samp)):
        ti = float(t_samp[i])

        while next_orb_i < len(orb_types) and float(times[next_orb_i + 1]) <= ti + 1e-12:
            typ = int(orb_types[next_orb_i])
            if typ == 0:
                vy += float(impulse_yellow)
            elif typ == 1:
                vy += float(impulse_pink)
            else:
                vy += float(impulse_blue)
                if blue_flips_gravity:
                    g_cur = -g_cur
            next_orb_i += 1

        vy += g_cur * dt
        y = float(y_samp[i - 1]) + vy * dt

        ceiling_units = min(float(y_ceil), float(30.0 * 9999.0))
        y = max(0.0, min(float(ceiling_units), y))
        y = max(float(y_min), min(float(y_max), y))

        if y <= float(y_min) + 1e-6 and vy < 0.0:
            vy = 0.0
        if y >= float(y_max) - 1e-6 and vy > 0.0:
            vy = 0.0

        y_samp[i] = y

    return t_samp, y_samp
ARC_V0_YELLOW = 590.85
ARC_G = 2727.35
PURPLE_V0_MULT = 0.71

def floor_collision_time(y0, v0, g):
    a = 0.5 * g
    b = -v0
    c = -y0
    disc = b*b - 4*a*c
    if disc < 0 or abs(a) < 1e-12:
        return None
    r = math.sqrt(disc)
    t1 = (-b - r) / (2*a)
    t2 = (-b + r) / (2*a)
    ts = [t for t in (t1, t2) if t > 1e-9]
    return min(ts) if ts else None

def sample_arcs_by_dx_segmented(times_s, seq, units_per_second, v0_y, v0_p, g, y_start, y_min, y_max, y_ceil, dx_units):
    if len(times_s) == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    U = float(units_per_second)
    dt = float(dx_units) / U

    out_t = [float(times_s[0])]
    y0 = max(0.0, float(y_start))
    out_y = [y0]

    for i in range(len(times_s) - 1):
        t0 = float(times_s[i])
        t1 = float(times_s[i + 1])
        dur = t1 - t0

        c = seq[i]
        v0 = float(v0_y if c == 0 else v0_p)

        t_hit = floor_collision_time(y0, v0, g)
        hit_in_seg = (t_hit is not None) and (t_hit <= dur + 1e-9)

        t = t0 + dt

        if not hit_in_seg:
            while t < t1 - 1e-9:
                tau = t - t0
                y = y0 + v0 * tau - 0.5 * g * tau * tau
                y = max(0.0, min(float(y_ceil), y))
                y = max(float(y_min), min(float(y_max), y))
                out_t.append(t)
                out_y.append(y)
                t += dt

            tau1 = dur
            y1 = y0 + v0 * tau1 - 0.5 * g * tau1 * tau1
            y1 = max(0.0, min(float(y_ceil), y1))
            y1 = max(float(y_min), min(float(y_max), y1))
            out_t.append(t1)
            out_y.append(y1)
            y0 = y1
        else:
            t_collide = t0 + float(t_hit)

            while t < t_collide - 1e-9:
                tau = t - t0
                y = y0 + v0 * tau - 0.5 * g * tau * tau
                y = max(0.0, min(float(y_ceil), y))
                y = max(float(y_min), min(float(y_max), y))
                out_t.append(t)
                out_y.append(y)
                t += dt

            out_t.append(t_collide)
            out_y.append(0.0)

            if t_collide < t1 - 1e-9:
                out_t.append(t1)
                out_y.append(0.0)
            else:
                if abs(t_collide - t1) > 1e-9:
                    out_t.append(t1)
                    out_y.append(0.0)

            y0 = 0.0

    return np.asarray(out_t, dtype=np.float64), np.asarray(out_y, dtype=np.float64)

def choose_orb_sequence_from_beats(n):
    if n <= 0:
        return np.array([], dtype=np.int32)
    seq = np.zeros(n, dtype=np.int32)
    for i in range(n):
        seq[i] = 0 if (i % 2 == 0) else 1
    return seq

def apply_floor_safety(seq, times_s, units_per_second, v0_y, v0_p, g, y_start, y_min, y_max, y_ceil, dx_units, floor_eps=35.0, max_iters=6):
    seq2 = np.asarray(seq, dtype=np.int32).copy()
    for _ in range(max_iters):
        t_samp, y_samp = sample_arcs_by_dx_segmented(
            times_s=times_s,
            seq=seq2,
            units_per_second=units_per_second,
            v0_y=v0_y,
            v0_p=v0_p,
            g=g,
            y_start=y_start,
            y_min=y_min,
            y_max=y_max,
            y_ceil=y_ceil,
            dx_units=dx_units,
        )

        changed = False
        for i in range(len(times_s) - 1):
            t0 = float(times_s[i])
            y0 = float(np.interp(t0, t_samp, y_samp))
            if y0 <= float(floor_eps) and seq2[i] == 1:
                seq2[i] = 0
                changed = True

        if not changed:
            break

    return seq2

def generate_level(times: np.ndarray, settings: AppSettings) -> LevelData:
    ups = settings.units_per_second()

    if len(times) == 0:
        return LevelData(
            times=times,
            t_samp=np.array([], dtype=np.float64),
            y_samp=np.array([], dtype=np.float64),
            units_per_second=ups,
            orb_times=np.array([], dtype=np.float64),
            orb_types=np.array([], dtype=np.int32),
        )

    p = settings.path
    dx = p.wave_dx_units if p.wave_dx_units is not None else p.dx_units

    if bool(p.start_as_wave):
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
        orb_types = np.array([], dtype=np.int32)
        orb_times = np.array([], dtype=np.float64)
    else:
        if len(times) < 2:
            t_samp = times.astype(np.float64, copy=False)
            y_samp = np.zeros_like(t_samp)
            orb_times = np.array([], dtype=np.float64)
            orb_types = np.array([], dtype=np.int32)
        else:
            orb_types = choose_orb_sequence_from_beats(len(times) - 1)
            orb_types = apply_floor_safety(
                seq=orb_types,
                times_s=times,
                units_per_second=ups,
                v0_y=ARC_V0_YELLOW,
                v0_p=ARC_V0_YELLOW * PURPLE_V0_MULT,
                g=ARC_G,
                y_start=max(0.0, p.y_start),
                y_min=p.y_min,
                y_max=p.y_max,
                y_ceil=p.y_ceil,
                dx_units=dx,
            )

            t_samp, y_samp = sample_arcs_by_dx_segmented(
                times_s=times,
                seq=orb_types,
                units_per_second=ups,
                v0_y=ARC_V0_YELLOW,
                v0_p=ARC_V0_YELLOW * PURPLE_V0_MULT,
                g=ARC_G,
                y_start=max(0.0, p.y_start),
                y_min=p.y_min,
                y_max=p.y_max,
                y_ceil=p.y_ceil,
                dx_units=dx,
            )

            orb_times = times[:-1].astype(np.float64, copy=False)



    return LevelData(
        times=times,
        t_samp=t_samp,
        y_samp=y_samp,
        units_per_second=ups,
        orb_times=orb_times,
        orb_types=orb_types,
    )
