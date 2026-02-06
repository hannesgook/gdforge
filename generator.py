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

    times_out = [float(x) for x in bt]
    types_out = [int(x) for x in ot]

    grav_sign = -1 if start_inverted else +1
    y = float(y0)

    i = 0
    for i in range(len(bt) - 1):
        if i < len(ot) and int(ot[i]) == 2:
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
    
    blue_theta0_deg=45.0
    blue_theta1_deg=69.0
    blue_theta_ramp_dx=1.5


    U = float(units_per_second)
    dt = float(dx_units) / U
    tan_cap = math.tan(math.radians(69.0))
    v_cap = U * tan_cap

    blue_cap_delay_tau = float(blue_cap_delay_x_units) / U

    out_t = [float(orb_times[0])]
    y0 = max(0.0, float(y_start))
    out_y = [y0]

    g_cur = float(g)  # signed acceleration for y(t) integration

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

        # Gravity sign tracking:
        # g_cur < 0 => normal (bound=0)
        # g_cur > 0 => inverted (bound=y_ceil)

        if typ == 2:
            # BLUE: flip gravity instantly (like audio2gmd.py)
            if blue_flips_gravity:
                g_cur = -g_cur

            y_bound = 0.0 if g_cur < 0.0 else float(y_ceil)

            theta0 = math.radians(float(blue_theta0_deg))
            theta1 = math.radians(float(blue_theta1_deg))
            ramp_dx = max(1e-6, float(blue_theta_ramp_dx))

            # In audio2gmd.py:
            # sign_dir = +1 if grav_sign == -1 else -1
            # grav_sign == +1 means normal, -1 means inverted
            grav_sign = +1 if (g_cur < 0.0) else -1
            sign_dir = +1.0 if grav_sign == -1 else -1.0

            def slope_for_dx(xu):
                a = xu / ramp_dx
                if a < 0.0: a = 0.0
                if a > 1.0: a = 1.0
                th = theta0 + (theta1 - theta0) * a
                return sign_dir * math.tan(th)

            # Force enough samples to see the 45->69 ramp.
            # We want multiple steps within ramp_dx (1.5 units).
            dxu_blue = min(float(dx_units), float(ramp_dx) / 20.0)  # 20 steps across the ramp
            dxu_blue = max(0.05, dxu_blue)  # don't go too tiny
            dt_blue = dxu_blue / U


            t = t0 + dt_blue
            x_prog = 0.0
            y_cur = y0
            hit = False

            while t < t1 - 1e-9:
                dxu = dxu_blue
                m = slope_for_dx(x_prog)
                y_next = y_cur + m * dxu

                if (g_cur < 0.0 and y_next < y_bound) or (g_cur > 0.0 and y_next > y_bound):
                    y_next = y_bound
                    hit = True

                y_next = clamp_y(y_next)
                out_t.append(t)
                out_y.append(y_next)

                y_cur = y_next
                x_prog += dxu

                if hit:
                    t_flat = t + dt_blue
                    while t_flat < t1 - 1e-9:
                        out_t.append(t_flat)
                        out_y.append(clamp_y(y_bound))
                        t_flat += dt_blue
                    break

                t += dt_blue

            if not hit:
                rem_t = t1 - (t - dt_blue)
                rem_dx = rem_t * U

                m = slope_for_dx(x_prog)
                y_end = y_cur + m * rem_dx

                if (g_cur < 0.0 and y_end < y_bound) or (g_cur > 0.0 and y_end > y_bound):
                    y_end = y_bound

                y_end = clamp_y(y_end)
                out_t.append(t1)
                out_y.append(y_end)
                y0 = y_end
            else:
                out_t.append(t1)
                out_y.append(clamp_y(y_bound))
                y0 = y_bound

            continue


        # Determine "inverted" from current gravity sign
        # g_cur < 0 => normal (player on ground, bound=0)
        # g_cur > 0 => inverted (player on ceiling, bound=y_ceil)
        is_inverted = (g_cur > 0.0)

        if typ == 2:
            # Blue: flip gravity first
            if blue_flips_gravity:
                g_cur = -g_cur
            is_inverted = (g_cur > 0.0)

            v0_mag = abs(float(impulse_blue))
            # audio2gmd.py behavior: blue uses opposite sign rule vs yellow/purple
            v0 = (-v0_mag) if (g_cur < 0.0) else (+v0_mag)

        elif typ == 1:
            v0_mag = abs(float(impulse_pink))
            # yellow/purple rule: normal -> +, inverted -> -
            v0 = (+v0_mag) if (g_cur < 0.0) else (-v0_mag)

        else:
            v0_mag = abs(float(impulse_yellow))
            v0 = (+v0_mag) if (g_cur < 0.0) else (-v0_mag)


        # Current gravity "boundary"
        # If g_cur < 0: normal gravity pulls down, boundary is ground (0)
        # If g_cur > 0: inverted pulls up, boundary is ceiling (y_ceil)
        y_bound = 0.0 if g_cur < 0.0 else float(y_ceil)

        # Ballistic until slope hits cap, then straight line at tan_cap
        # v(t) = v0 + g_cur * tau
        # slope dy/dx = v/U
        tau_cap_candidates = []
        for s in (-1.0, +1.0):
            # solve v0 + g*tau = s*v_cap  -> tau = (s*v_cap - v0)/g
            if abs(g_cur) > 1e-12:
                tau = (s * v_cap - v0) / g_cur
                if 0.0 <= tau <= dur:
                    tau_cap_candidates.append(tau)
        tau_cap = min(tau_cap_candidates) if tau_cap_candidates else None

        cap_trigger_tau = tau_cap
        if typ == 2 and cap_trigger_tau is not None:
            cap_trigger_tau = max(float(cap_trigger_tau), blue_cap_delay_tau)


        def y_ballistic(tau):
            return y0 + v0 * tau + 0.5 * g_cur * tau * tau

        # Determine cap line slope sign at tau_cap (matches audio2gmd style)
        cap_active = False
        y_cap = None
        m_cap = None

        t = t0 + dt
        y_cur = y0

        hit = False

        while t < t1 - 1e-9:
            tau = t - t0

            if (not cap_active) and (cap_trigger_tau is not None) and (tau >= cap_trigger_tau - 1e-12):
                cap_active = True
                y_cap = y_ballistic(float(tau_cap))
                v_at_cap = v0 + g_cur * float(tau_cap)
                slope_sign = 1.0 if v_at_cap >= 0.0 else -1.0
                m_cap = slope_sign * tan_cap
                # reset current y to the cap point at this exact moment
                y_cur = y_cap

            if not cap_active:
                y_next = y_ballistic(tau)
            else:
                # move linearly by capped slope
                dxu = dt * U
                y_next = y_cur + float(m_cap) * dxu

            # boundary collision: if we cross the bound, snap and fill flat
            if (g_cur < 0.0 and y_next < y_bound) or (g_cur > 0.0 and y_next > y_bound):
                y_next = y_bound
                hit = True

            y_next = clamp_y(y_next)
            out_t.append(t)
            out_y.append(y_next)

            y_cur = y_next

            if hit:
                # fill rest of segment on the boundary
                t_fill = t + dt
                while t_fill < t1 - 1e-9:
                    out_t.append(t_fill)
                    out_y.append(clamp_y(y_bound))
                    t_fill += dt
                break

            t += dt

        # segment end point
        if not hit:
            if not cap_active:
                y1 = y_ballistic(dur)
            else:
                # continue capped line to end
                rem_t = max(0.0, t1 - (t - dt))
                rem_dx = rem_t * U
                y1 = y_cur + float(m_cap) * rem_dx

            if (g_cur < 0.0 and y1 < y_bound) or (g_cur > 0.0 and y1 > y_bound):
                y1 = y_bound

            y1 = clamp_y(y1)
            out_t.append(t1)
            out_y.append(y1)
            y0 = y1
        else:
            out_t.append(t1)
            out_y.append(clamp_y(y_bound))
            y0 = y_bound

    return np.asarray(out_t, dtype=np.float64), np.asarray(out_y, dtype=np.float64)


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

    last_seg_index = len(times_s) - 2

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

        if i == last_seg_index:
            if dir_up and y_cur >= ceiling_units - margin_units:
                dir_up = False
            elif (not dir_up) and y_cur <= margin_units:
                dir_up = True

            s = 1.0 if dir_up else -1.0

            t_prev = out_t[-1]
            rem_dx = max(0.0, (t1 - t_prev) * U)

            y_end = y_cur + s * m * rem_dx
            y_end = max(0.0, min(float(y_ceil), y_end))
            y_end = max(float(y_min), min(float(y_max), y_end))

            out_t.append(t1)
            out_y.append(y_end)
        else:
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

def generate_level(times: np.ndarray, settings: AppSettings, end_time_s: Optional[float] = None) -> LevelData:
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
    dx_sim = 0.1

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
            orb_times = np.array([], dtype=np.float64)
            orb_types = np.array([], dtype=np.int32)
            t_samp = np.array([], dtype=np.float64)
            y_samp = np.array([], dtype=np.float64)
        else:
            # 1) choose one orb type per SEGMENT (beat i -> beat i+1)
            orb_types = _choose_orb_types_from_times(times).astype(np.int32, copy=False)   # len(times)-1

            # 2) place an orb at the START of each segment, INCLUDING the first beat
            orb_times = times[:-1].astype(np.float64, copy=False)                          # len(times)-1

            # 3) optionally extend beyond last beat
            if end_time_s is not None:
                orb_times, orb_types = _extend_orb_events_to_end(
                    beat_times=orb_times,
                    orb_types=orb_types,
                    end_time_s=float(end_time_s),
                    y0=max(0.0, p.y_start),
                    y_ceil=p.y_ceil,
                    start_inverted=False,
                    eps=20.0,
                )

            # 4) sample arcs driven by these orb events
            t_samp, y_samp = sample_cube_arcs_from_orb_events(
                orb_times=orb_times,
                orb_types=orb_types,
                units_per_second=ups,
                dx_units=dx_sim,
                y_start=max(0.0, p.y_start),
                y_min=p.y_min,
                y_max=p.y_max,
                y_ceil=p.y_ceil,
                g=-2727.35,
                impulse_yellow=590.85,
                impulse_pink=590.85 * 0.71,
                impulse_blue=590.85,
                blue_flips_gravity=True,
            )


    return LevelData(
        times=times,
        t_samp=t_samp,
        y_samp=y_samp,
        units_per_second=ups,
        orb_times=orb_times,
        orb_types=orb_types,
    )
