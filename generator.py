# Copyright (c) 2025-2026 Hannes Göök
# MIT License - GDForge
# https://github.com/hannesgook/gdforge

from dataclasses import dataclass
import numpy as np
import math
from settings import AppSettings
from typing import Optional


def _extend_orb_events_to_end(
    beat_times: np.ndarray,
    orb_types: np.ndarray,
    end_time_s: float,
    y0: float,
    y_ceil: float,
    start_inverted: bool = False,
    eps: float = 20.0,
):
    if len(beat_times) == 0:
        return beat_times, orb_types

    bt = beat_times.astype(np.float64, copy=False)
    ot = orb_types.astype(np.int32, copy=False)

    if not np.isfinite(end_time_s) or end_time_s <= float(bt[-1]) + 1e-6:
        return bt, ot

    if len(bt) >= 2:
        base_dt = float(np.median(np.diff(bt)))
    else:
        base_dt = 0.35
    base_dt = max(0.12, min(0.75, base_dt))

    times_out = list(bt)
    types_out = list(ot)

    grav_sign = -1 if start_inverted else +1
    y = float(y0)

    for i in range(len(ot)):
        if int(ot[i]) == 2:
            grav_sign *= -1

    t = float(bt[-1]) + base_dt
    while t < float(end_time_s) - 1e-6:
        if grav_sign == +1:
            if y <= eps:
                typ = 0
            elif y >= float(y_ceil) - eps:
                typ = 2
                grav_sign *= -1
            else:
                typ = 0 if (len(types_out) % 2 == 0) else 1
        else:
            if y >= float(y_ceil) - eps:
                typ = 0
            elif y <= eps:
                typ = 2
                grav_sign *= -1
            else:
                typ = 0 if (len(types_out) % 2 == 0) else 1

        times_out.append(float(t))
        types_out.append(int(typ))

        y = float(y_ceil) if grav_sign == -1 else 0.0
        t += base_dt

    return np.asarray(times_out, dtype=np.float64), np.asarray(types_out, dtype=np.int32)


def sample_cube_arcs_from_orb_events(
    orb_times: np.ndarray,
    orb_types: np.ndarray,
    units_per_second: float,
    dx_units: float,
    y_start: float,
    y_min: float,
    y_max: float,
    y_ceil: float,
    g: float = -2727.35,
    impulse_yellow: float = 590.85,
    impulse_pink: float = 590.85 * 0.71,
    impulse_blue: float = 590.85,
    blue_flips_gravity: bool = True,
    blue_cap_delay_x_units=45,
):
    if len(orb_times) < 2:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    blue_theta0_deg = 45.0
    blue_theta1_deg = 69.0
    blue_theta_ramp_dx = 1.5

    U = float(units_per_second)
    dt = float(dx_units) / U
    tan_cap = math.tan(math.radians(69.0))
    v_cap = U * tan_cap
    blue_cap_delay_tau = float(blue_cap_delay_x_units) / U

    y0 = max(0.0, float(y_start))
    g_cur = float(g)

    # Pre-allocate output lists - we'll extend with numpy arrays per segment
    all_t = [float(orb_times[0])]
    all_y = [y0]

    def clamp_y(y):
        y = max(0.0, min(float(y_ceil), y))
        y = max(float(y_min), min(float(y_max), y))
        return y

    for i in range(len(orb_times) - 1):
        t0 = float(orb_times[i])
        t1 = float(orb_times[i + 1])
        dur = t1 - t0
        if dur <= 1e-9:
            continue

        typ = int(orb_types[i])

        # BLUE orb: linear slope ramp
        if typ == 2:
            if blue_flips_gravity:
                g_cur = -g_cur

            y_bound = 15.0 if g_cur < 0.0 else float(y_ceil)
            theta0 = math.radians(float(blue_theta0_deg))
            theta1 = math.radians(float(blue_theta1_deg))
            ramp_dx = max(1e-6, float(blue_theta_ramp_dx))

            grav_sign = +1 if (g_cur < 0.0) else -1
            sign_dir = +1.0 if grav_sign == -1 else -1.0

            dxu_blue = min(float(dx_units), float(ramp_dx) / 5.0)
            dxu_blue = max(0.05, dxu_blue)
            dt_blue = dxu_blue / U

            # Vectorize the ramp phase
            n_steps = max(1, int(math.ceil(dur / dt_blue)))
            taus_blue = np.arange(1, n_steps + 1) * dt_blue
            taus_blue = taus_blue[taus_blue < dur - 1e-9]

            x_progs = np.arange(1, len(taus_blue) + 1) * dxu_blue
            alphas = np.clip(x_progs / ramp_dx, 0.0, 1.0)
            thetas = theta0 + (theta1 - theta0) * alphas
            slopes = sign_dir * np.tan(thetas)

            # Cumulative y
            y_cur = y0
            hit = False
            seg_t = []
            seg_y = []

            for j, (tau, slope) in enumerate(zip(taus_blue, slopes)):
                y_next = y_cur + slope * dxu_blue
                if (g_cur < 0.0 and y_next < y_bound) or (g_cur > 0.0 and y_next > y_bound):
                    y_next = y_bound
                    hit = True
                y_next = clamp_y(y_next)
                seg_t.append(t0 + tau)
                seg_y.append(y_next)
                y_cur = y_next
                if hit:
                    # Vectorize the flat fill
                    t_after = t0 + tau + dt_blue
                    if t_after < t1 - 1e-9:
                        flat_ts = np.arange(t_after, t1 - 1e-9, dt_blue)
                        seg_t.extend(flat_ts.tolist())
                        seg_y.extend([clamp_y(y_bound)] * len(flat_ts))
                    break

            all_t.extend(seg_t)
            all_y.extend(seg_y)

            if not hit:
                all_t.append(t1)
                all_y.append(clamp_y(y_cur))
                y0 = clamp_y(y_cur)
            else:
                all_t.append(t1)
                all_y.append(clamp_y(y_bound))
                y0 = y_bound
            continue

        # YELLOW / PINK / (duplicate blue path below typ==2 check already handled above)
        is_inverted = (g_cur > 0.0)

        if typ == 1:
            v0_mag = abs(float(impulse_pink))
            v0 = (+v0_mag) if (g_cur < 0.0) else (-v0_mag)
        else:
            v0_mag = abs(float(impulse_yellow))
            v0 = (+v0_mag) if (g_cur < 0.0) else (-v0_mag)

        y_bound = 15.0 if g_cur < 0.0 else float(y_ceil)

        # Cap velocity
        tau_cap_candidates = []
        for s in (-1.0, +1.0):
            if abs(g_cur) > 1e-12:
                tau = (s * v_cap - v0) / g_cur
                if 0.0 <= tau <= dur:
                    tau_cap_candidates.append(tau)
        tau_cap = min(tau_cap_candidates) if tau_cap_candidates else None
        cap_trigger_tau = tau_cap

        # Vectorized ballistic phase
        n_steps = max(1, int(math.ceil(dur / dt)))
        taus = np.arange(1, n_steps + 1) * dt
        taus = taus[taus < dur - 1e-9]

        if len(taus) == 0:
            # short segment, emit endpoint
            y1 = y0 + v0 * dur + 0.5 * g_cur * dur * dur
            y1 = clamp_y(min(y_bound, y1) if g_cur < 0.0 else max(y_bound, y1))
            all_t.append(t1)
            all_y.append(y1)
            y0 = y1
            continue

        # Split into ballistic and capped phases
        if cap_trigger_tau is not None:
            mask_ballistic = taus < cap_trigger_tau
            taus_b = taus[mask_ballistic]
            taus_c = taus[~mask_ballistic]
        else:
            taus_b = taus
            taus_c = np.array([], dtype=np.float64)

        # Ballistic phase
        ys_b = y0 + v0 * taus_b + 0.5 * g_cur * taus_b * taus_b
        np.clip(ys_b, float(y_min), float(y_max), out=ys_b)
        np.clip(ys_b, 15.0, float(y_ceil), out=ys_b)

        # Check for boundary hit in ballistic phase
        if g_cur < 0.0:
            hit_mask = ys_b <= y_bound
        else:
            hit_mask = ys_b >= y_bound

        hit_idx = np.argmax(hit_mask) if np.any(hit_mask) else -1

        if hit_idx >= 0:
            # Emit up to hit, then fill flat
            all_t.extend((t0 + taus_b[:hit_idx + 1]).tolist())
            ys_b[:hit_idx + 1] = np.clip(ys_b[:hit_idx + 1], y_bound if g_cur < 0.0 else -np.inf,
                                          y_bound if g_cur > 0.0 else np.inf)
            all_y.extend(ys_b[:hit_idx + 1].tolist())
            # Flat fill for rest
            t_after = t0 + taus_b[hit_idx] + dt
            if t_after < t1 - 1e-9:
                flat_ts = np.arange(t_after, t1 - 1e-9, dt)
                all_t.extend(flat_ts.tolist())
                all_y.extend([float(y_bound)] * len(flat_ts))
            all_t.append(t1)
            all_y.append(float(y_bound))
            y0 = float(y_bound)
            continue

        all_t.extend((t0 + taus_b).tolist())
        all_y.extend(ys_b.tolist())

        # Capped linear phase
        if len(taus_c) > 0 and cap_trigger_tau is not None:
            y_at_cap = y0 + v0 * cap_trigger_tau + 0.5 * g_cur * cap_trigger_tau * cap_trigger_tau
            v_at_cap = v0 + g_cur * cap_trigger_tau
            slope_sign = 1.0 if v_at_cap >= 0.0 else -1.0
            m_cap = slope_sign * tan_cap

            dt_from_cap = taus_c - cap_trigger_tau
            ys_c = y_at_cap + m_cap * U * dt_from_cap
            np.clip(ys_c, float(y_min), float(y_max), out=ys_c)
            np.clip(ys_c, 15.0, float(y_ceil), out=ys_c)

            if g_cur < 0.0:
                hit_mask_c = ys_c <= y_bound
            else:
                hit_mask_c = ys_c >= y_bound

            hit_idx_c = np.argmax(hit_mask_c) if np.any(hit_mask_c) else -1

            if hit_idx_c >= 0:
                all_t.extend((t0 + taus_c[:hit_idx_c + 1]).tolist())
                all_y.extend(ys_c[:hit_idx_c + 1].tolist())
                t_after = t0 + taus_c[hit_idx_c] + dt
                if t_after < t1 - 1e-9:
                    flat_ts = np.arange(t_after, t1 - 1e-9, dt)
                    all_t.extend(flat_ts.tolist())
                    all_y.extend([float(y_bound)] * len(flat_ts))
                all_t.append(t1)
                all_y.append(float(y_bound))
                y0 = float(y_bound)
                continue

            all_t.extend((t0 + taus_c).tolist())
            all_y.extend(ys_c.tolist())

        # Segment end
        if cap_trigger_tau is None or len(taus_c) == 0:
            y1 = y0 + v0 * dur + 0.5 * g_cur * dur * dur
        else:
            rem_dt = dur - taus_c[-1] if len(taus_c) > 0 else 0.0
            y1 = all_y[-1] + m_cap * U * rem_dt if 'm_cap' in dir() else all_y[-1]

        y1 = clamp_y(y1)
        if g_cur < 0.0 and y1 < y_bound:
            y1 = y_bound
        elif g_cur > 0.0 and y1 > y_bound:
            y1 = y_bound

        all_t.append(t1)
        all_y.append(y1)
        y0 = y1

    return np.asarray(all_t, dtype=np.float64), np.asarray(all_y, dtype=np.float64)


@dataclass
class LevelData:
    times: np.ndarray
    t_samp: np.ndarray
    y_samp: np.ndarray
    units_per_second: float
    orb_times: np.ndarray
    orb_types: np.ndarray

def sample_wave_by_dx(times_s, units_per_second, dx_units, y_start, y_min, y_max, y_ceil,
                      angle_deg, start_dir_up, corridor_units, margin_units):
    # Vectorized
    if len(times_s) == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    U = float(units_per_second)
    dt = float(dx_units) / U
    m = math.tan(math.radians(angle_deg))
    dy_step = m * float(dx_units)

    ceiling_units = min(float(y_ceil), float(corridor_units) * 30.0)
    y_lo = float(margin_units)
    y_hi = ceiling_units - float(margin_units)

    # Build the full time array across all segments at once
    all_t = []
    all_y = []

    y_cur = max(0.0, float(y_start))
    dir_up = bool(start_dir_up)

    all_t.append(float(times_s[0]))
    all_y.append(y_cur)

    last_seg = len(times_s) - 2

    for i in range(len(times_s) - 1):
        t0 = float(times_s[i])
        t1 = float(times_s[i + 1])
        dur = t1 - t0

        # Build time array for this segment
        seg_ts = np.arange(t0 + dt, t1 - 1e-9, dt)

        if len(seg_ts) > 0:
            # Direction switching
            ys = np.empty(len(seg_ts), dtype=np.float64)
            for j in range(len(seg_ts)):
                if dir_up and y_cur >= y_hi:
                    dir_up = False
                elif (not dir_up) and y_cur <= y_lo:
                    dir_up = True
                s = 1.0 if dir_up else -1.0
                y_cur = float(np.clip(y_cur + s * dy_step, max(0.0, float(y_min)), min(float(y_ceil), float(y_max))))
                ys[j] = y_cur

            all_t.append(seg_ts)
            all_y.append(ys)

        # Segment endpoint
        if i == last_seg:
            t_prev = seg_ts[-1] if len(seg_ts) > 0 else t0
            if dir_up and y_cur >= y_hi:
                dir_up = False
            elif (not dir_up) and y_cur <= y_lo:
                dir_up = True
            s = 1.0 if dir_up else -1.0
            rem_dx = max(0.0, (t1 - t_prev) * U)
            y_end = float(np.clip(y_cur + s * m * rem_dx, max(0.0, float(y_min)), min(float(y_ceil), float(y_max))))
            all_t.append(np.array([t1]))
            all_y.append(np.array([y_end]))
        else:
            last_y = y_cur if len(seg_ts) == 0 else float(all_y[-1][-1])
            all_t.append(np.array([t1]))
            all_y.append(np.array([last_y]))

        dir_up = not dir_up

    # Flatten all segments
    out_t_parts = []
    out_y_parts = []
    out_t_parts.append(np.array([all_t[0]], dtype=np.float64))
    out_y_parts.append(np.array([all_y[0]], dtype=np.float64))
    for item in all_t[1:]:
        out_t_parts.append(np.asarray(item, dtype=np.float64).ravel())
    for item in all_y[1:]:
        out_y_parts.append(np.asarray(item, dtype=np.float64).ravel())

    return np.concatenate(out_t_parts), np.concatenate(out_y_parts)


def _choose_orb_types_from_times(times: np.ndarray) -> np.ndarray:
    if len(times) == 0:
        return np.array([], dtype=np.int32)
    dt = np.diff(times)
    if len(dt) == 0:
        return np.array([], dtype=np.int32)
    med = float(np.median(dt))
    types = np.where(dt < med * 0.85, 1, 0).astype(np.int32)
    types[5::8] = 2 # blue gravity flip every 8 beats starting at index 5

    # Second pass: track gravity state and apply safety rules:
    # - rightside-up + gap >= 1s + blue orb -> downgrade to yellow (no sky drift)
    # - upside-down + gap >= 1s -> force blue (prevent ceiling drift)
    inverted = False
    for i in range(len(types)):
        gap = float(dt[i])
        if inverted and gap >= 1.0:
            types[i] = 2 # force blue to flip back before drifting to ceiling
        elif not inverted and types[i] == 2 and gap >= 1.0:
            types[i] = 0 # downgrade blue, player would drift too high
        if types[i] == 2:
            inverted = not inverted

    return types

ARC_V0_YELLOW = 590.85
ARC_G = 2727.35
PURPLE_V0_MULT = 0.71

def floor_collision_time(y0, v0, g):
    a = 0.5 * g
    b = -v0
    c = -y0
    disc = b * b - 4 * a * c
    if disc < 0 or abs(a) < 1e-12:
        return None
    r = math.sqrt(disc)
    t1 = (-b - r) / (2 * a)
    t2 = (-b + r) / (2 * a)
    ts = [t for t in (t1, t2) if t > 1e-9]
    return min(ts) if ts else None


def sample_arcs_by_dx_segmented(times_s, seq, units_per_second, v0_y, v0_p, g,
                                 y_start, y_min, y_max, y_ceil, dx_units):
    if len(times_s) == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    U = float(units_per_second)
    dt = float(dx_units) / U
    y0 = max(0.0, float(y_start))

    all_t = [float(times_s[0])]
    all_y = [y0]

    for i in range(len(times_s) - 1):
        t0 = float(times_s[i])
        t1 = float(times_s[i + 1])
        dur = t1 - t0

        c = seq[i]
        v0 = float(v0_y if c == 0 else v0_p)

        t_hit = floor_collision_time(y0, v0, g)
        hit_in_seg = (t_hit is not None) and (t_hit <= dur + 1e-9)

        # Build time array for this segment
        seg_ts = np.arange(t0 + dt, t1 - 1e-9, dt)

        if not hit_in_seg:
            if len(seg_ts) > 0:
                taus = seg_ts - t0
                ys = y0 + v0 * taus - 0.5 * g * taus * taus
                np.clip(ys, max(0.0, float(y_min)), min(float(y_ceil), float(y_max)), out=ys)
                all_t.append(seg_ts)
                all_y.append(ys)

            tau1 = dur
            y1 = float(np.clip(y0 + v0 * tau1 - 0.5 * g * tau1 * tau1,
                                max(0.0, float(y_min)), min(float(y_ceil), float(y_max))))
            all_t.append(np.array([t1]))
            all_y.append(np.array([y1]))
            y0 = y1
        else:
            t_collide = t0 + float(t_hit)
            pre_ts = seg_ts[seg_ts < t_collide - 1e-9]
            if len(pre_ts) > 0:
                taus = pre_ts - t0
                ys = y0 + v0 * taus - 0.5 * g * taus * taus
                np.clip(ys, max(0.0, float(y_min)), min(float(y_ceil), float(y_max)), out=ys)
                all_t.append(pre_ts)
                all_y.append(ys)

            all_t.append(np.array([t_collide, t1]))
            all_y.append(np.array([15, 15]))
            y0 = 15

    # Flatten
    out_t = np.concatenate([np.atleast_1d(np.asarray(x, dtype=np.float64)) for x in all_t])
    out_y = np.concatenate([np.atleast_1d(np.asarray(x, dtype=np.float64)) for x in all_y])
    return out_t, out_y


def choose_orb_sequence_from_beats(n):
    if n <= 0:
        return np.array([], dtype=np.int32)
    seq = np.zeros(n, dtype=np.int32)
    seq[1::2] = 1
    return seq


def apply_floor_safety(seq, times_s, units_per_second, v0_y, v0_p, g,
                        y_start, y_min, y_max, y_ceil, dx_units,
                        floor_eps=35.0, max_iters=6):
    """Vectorized floor safety - replaces the per-beat interp loop."""
    seq2 = np.asarray(seq, dtype=np.int32).copy()

    for _ in range(max_iters):
        t_samp, y_samp = sample_arcs_by_dx_segmented(
            times_s=times_s, seq=seq2, units_per_second=units_per_second,
            v0_y=v0_y, v0_p=v0_p, g=g, y_start=y_start,
            y_min=y_min, y_max=y_max, y_ceil=y_ceil, dx_units=dx_units,
        )

        # Vectorized interp for all beat times at once
        y_at_beats = np.interp(times_s[:-1], t_samp, y_samp)
        fix_mask = (y_at_beats <= float(floor_eps)) & (seq2 == 1)
        if not np.any(fix_mask):
            break
        seq2[fix_mask] = 0

    return seq2


def generate_level(times: np.ndarray, settings: AppSettings,
                   end_time_s: Optional[float] = None) -> LevelData:
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
    dx_sim = dx

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
        y_samp = np.maximum(y_samp, 0.0)
        orb_types = np.array([], dtype=np.int32)
        orb_times = np.array([], dtype=np.float64)
    else:
        if len(times) < 2:
            orb_times = np.array([], dtype=np.float64)
            orb_types = np.array([], dtype=np.int32)
            t_samp = np.array([], dtype=np.float64)
            y_samp = np.array([], dtype=np.float64)
        else:
            orb_types = _choose_orb_types_from_times(times).astype(np.int32, copy=False)
            orb_times = times[:-1].astype(np.float64, copy=False)

            if end_time_s is not None:
                orb_times, orb_types = _extend_orb_events_to_end(
                    beat_times=orb_times, orb_types=orb_types,
                    end_time_s=float(end_time_s),
                    y0=max(0.0, p.y_start), y_ceil=p.y_ceil,
                    start_inverted=False, eps=20.0,
                )

            t_samp, y_samp = sample_cube_arcs_from_orb_events(
                orb_times=orb_times, orb_types=orb_types,
                units_per_second=ups, dx_units=dx_sim,
                y_start=max(0.0, p.y_start),
                y_min=p.y_min, y_max=p.y_max, y_ceil=p.y_ceil,
                g=-2727.35, impulse_yellow=590.85,
                impulse_pink=590.85 * 0.71, impulse_blue=590.85,
                blue_flips_gravity=True,
            )
            y_samp = np.maximum(y_samp, 0.0)

            # Vectorized orb ground filter
            if len(orb_times) > 0 and len(t_samp) > 0:
                ground_threshold = 15.1
                y_at_orbs = np.interp(orb_times, t_samp, y_samp)
                keep_mask = (y_at_orbs > ground_threshold) | (orb_types != 0)
                orb_times = orb_times[keep_mask]
                orb_types = orb_types[keep_mask]

    return LevelData(
        times=times,
        t_samp=t_samp,
        y_samp=y_samp,
        units_per_second=ups,
        orb_times=orb_times,
        orb_types=orb_types,
    )