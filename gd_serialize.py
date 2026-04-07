# Copyright (c) 2025-2026 Hannes Göök
# MIT License - GDForge
# https://github.com/hannesgook/gdforge

from xml.sax.saxutils import escape
import numpy as np
import math

# XML helpers (unchanged)
def xml_kv_string(k, v): return f"<k>{escape(k)}</k><s>{escape(v)}</s>"
def xml_kv_int(k, v):    return f"<k>{escape(k)}</k><i>{v}</i>"
def xml_kv_true(k):       return f"<k>{escape(k)}</k><t />"
def xml_kv_real(k, v):   return f"<k>{escape(k)}</k><r>{v}</r>"

def build_level_xml(level_name, creator_name, k4_plain, custom_song_id):
    parts = []
    parts.append(xml_kv_int("kCEK", 4))
    parts.append(xml_kv_string("k2", level_name))
    parts.append(xml_kv_string("k4", k4_plain))
    parts.append(xml_kv_string("k5", creator_name))
    parts.append(xml_kv_string("k101", ",".join(["0"] * 20)))
    parts.append(xml_kv_true("k13"))
    parts.append(xml_kv_int("k21", 2))
    parts.append(xml_kv_int("k16", 1))
    parts.append(xml_kv_int("k80", 7))
    parts.append(xml_kv_int("k50", 45))
    parts.append(xml_kv_true("k47"))
    parts.append(xml_kv_int("k48", 1))
    parts.append(xml_kv_real("kI1", 237.717))
    parts.append(xml_kv_real("kI2", 151.044))
    parts.append(xml_kv_real("kI3", 0.1))
    parts.append(xml_kv_int("k8", 1))
    parts.append(xml_kv_int("k45", int(custom_song_id)))
    return "<dict>" + "".join(parts) + "</dict>"


def build_k4_polyline(t_samp, y_samp, units_per_second, start_offset_s,
                       block_id=1764, y_add=15.0, start_as_wave=False,
                       spacing_units=4.0):
    header = "kS38,1_40_2_125_3_255_11_255_12_255_13_255_4_-1_6_1000_7_1_15_1_18_0_8_1,kA13,0,kA6,1,kA16,1,kA15,1,k128,0;"
    objs = []

    start_x = 0.0
    start_y = float(y_add)

    if start_as_wave:
        objs.append(f"1,660,2,{start_x:.3f},3,{start_y:.3f},155,1;")

    objs.append(
        build_objects_along_path_by_spacing(
            times_s=t_samp, y_s=y_samp,
            units_per_second=units_per_second,
            start_offset_s=start_offset_s,
            spacing_units=spacing_units,
            block_id=int(block_id),
            y_add=float(y_add),
        )
    )

    return header + "".join(objs)


# build_objects_along_path_by_spacing
def build_objects_along_path_by_spacing(
    times_s, y_s, units_per_second, start_offset_s,
    spacing_units, block_id, y_add=15.0
):
    """
    Computes cumulative arc-length along the path,
    samples at evenly-spaced intervals, and builds the object string in one pass.
    """
    if len(times_s) == 0:
        return ""

    U = float(units_per_second)
    xs = (np.asarray(times_s, dtype=np.float64) + float(start_offset_s)) * U
    ys = np.asarray(y_s, dtype=np.float64) + float(y_add)

    # Cumulative arc length
    dxs = np.diff(xs)
    dys = np.diff(ys)
    seg_lens = np.hypot(dxs, dys)
    cum = np.empty(len(xs), dtype=np.float64)
    cum[0] = 0.0
    np.cumsum(seg_lens, out=cum[1:])
    total = cum[-1]

    if total < 1e-9:
        return f"1,{int(block_id)},2,{xs[0]:.3f},3,{ys[0]:.3f},155,1;"

    # Evenly-spaced sample distances
    sample_dists = np.arange(0.0, total, float(spacing_units))

    # Interpolate x and y at those distances
    px = np.interp(sample_dists, cum, xs)
    py = np.interp(sample_dists, cum, ys)

    bid = int(block_id)
    parts = [f"1,{bid},2,{x:.3f},3,{y:.3f},155,1;" for x, y in zip(px, py)]
    return "".join(parts)


def build_k4_orb_arc(
    t_samp, y_samp, units_per_second, start_offset_s,
    orb_times, orb_types,
    y_add=15.0,
    block_id=1764,
    orb_id_yellow=36,
    orb_id_purple=141,
    orb_id_blue=84,
    orb_id_green=1022,
    cube_portal_id=12,
    start_with_cube=True,
    orb_x_offset=0.0,
    orb_y_offset=15.0,
    spacing_units=4.0,
):
    header = "kS38,1_40_2_125_3_255_11_255_12_255_13_255_4_-1_6_1000_7_1_15_1_18_0_8_1,kA13,0,kA6,1,kA16,1,kA15,1,k128,0;"
    objs = []

    U = float(units_per_second)
    off = float(start_offset_s)
    ya = float(y_add)

    if start_with_cube:
        objs.append(f"1,{int(cube_portal_id)},2,0.000,3,{ya:.3f},155,1;")

    objs.append(
        build_objects_along_path_by_spacing(
            times_s=t_samp, y_s=y_samp,
            units_per_second=units_per_second,
            start_offset_s=start_offset_s,
            spacing_units=float(spacing_units),
            block_id=int(block_id),
            y_add=ya,
        )
    )

    if orb_times is None or orb_types is None:
        return header + "".join(objs)

    orb_times = np.asarray(orb_times, dtype=np.float64)
    orb_types = np.asarray(orb_types, dtype=np.int32)

    if len(orb_times) == 0:
        return header + "".join(objs)

    t_s = np.asarray(t_samp, dtype=np.float64)
    y_s_arr = np.asarray(y_samp, dtype=np.float64)

    id_by_type = {
        0: int(orb_id_yellow),
        1: int(orb_id_purple),
        2: int(orb_id_blue),
        3: int(orb_id_green),
    }

    # Vectorized orb placement
    orb_xs = orb_times * U + off * U + float(orb_x_offset)
    y_on_arc = np.interp(orb_times, t_s, y_s_arr)
    orb_ys = np.maximum(0.0, y_on_arc) + float(orb_y_offset)

    for ox, oy, typ in zip(orb_xs, orb_ys, orb_types):
        oid = id_by_type.get(int(typ), int(orb_id_yellow))
        objs.append(f"1,{oid},2,{ox:.3f},3,{oy:.3f},155,1;")

    return header + "".join(objs)

def serialize_gmd(level_name, creator_name, song_id, k4_plain):
    level_xml = build_level_xml(level_name, creator_name, k4_plain, song_id)
    return '<?xml version="1.0"?><plist version="1.0" gjver="2.0">' + level_xml + "</plist>"

def build_wave_ramps_45deg(t_samp, y_samp, units_per_second, start_offset_s,
                            ramp_id=309, ramp_size_units=30.0, y_add=15.0):
    if len(t_samp) < 2:
        return ""

    U = float(units_per_second)
    xs = (np.asarray(t_samp, dtype=np.float64) + float(start_offset_s)) * U
    ys = (np.asarray(y_samp, dtype=np.float64) + float(y_add))

    spacing = float(ramp_size_units) * math.sqrt(2.0)
    objs = []

    dist_acc = 0.0
    for i in range(1, len(xs)):
        x0, y0 = float(xs[i - 1]), float(ys[i - 1])
        x1, y1 = float(xs[i]),     float(ys[i])
        dx = x1 - x0
        dy = y1 - y0
        seg = math.hypot(dx, dy)
        if seg <= 1e-9:
            continue

        while dist_acc + seg >= spacing - 1e-9:
            need = spacing - dist_acc
            a = need / seg
            px = x0 + a * dx
            py = y0 + a * dy
            rot = 45.0 if dy >= 0.0 else -45.0
            objs.append(f"1,{int(ramp_id)},2,{px:.3f},3,{py:.3f},6,{rot:.3f},155,1;")
            x0, y0 = px, py
            dx = x1 - x0
            dy = y1 - y0
            seg = math.hypot(dx, dy)
            dist_acc = 0.0

        dist_acc += seg

    return "".join(objs)

def wave_make_clones(y_samp, gap_units, y_min, y_max, y_ceil):
    if gap_units is None:
        return None, None
    gap = float(gap_units)
    if gap <= 0.0:
        return None, None
    cap_max = min(float(y_ceil), float(y_max))
    above = np.clip(np.asarray(y_samp, dtype=np.float64) + gap, max(float(y_min), 0.0), cap_max)
    below = np.clip(np.asarray(y_samp, dtype=np.float64) - gap, float(y_min), cap_max)
    return above, below

Y_SHIFT = 15.0

def build_ramps_along_path_by_spacing(
    times_s, y_s, units_per_second, start_offset_s,
    spacing_units, ramp_id, y_add=15.0,
    extra_rot_deg=0.0, rotate_180=False,
    invert_if_top=False, invert_if_bottom=False
):
    if len(times_s) == 0:
        return ""

    y_add = float(y_add) - 15.0
    U = float(units_per_second)
    xs = (np.asarray(times_s, dtype=np.float64) + float(start_offset_s)) * U
    ys = np.asarray(y_s, dtype=np.float64) + float(y_add) + float(Y_SHIFT)
    n = len(xs)
    if n < 2:
        return ""

    is_top = bool(rotate_180)
    flip_base = (is_top and invert_if_top) or ((not is_top) and invert_if_bottom)

    def normalize_angle(a):
        a = float(a)
        while a <= -180.0: a += 360.0
        while a > 180.0:   a -= 360.0
        return a

    def final_rot_from_dxdy(dx, dy):
        eps = 1e-6
        if abs(dx) < eps and abs(dy) < eps:
            base = 0.0
        elif dx >= 0.0 and dy < 0.0:
            base = 180.0
        elif dx >= 0.0 and dy >= 0.0:
            base = 0.0
        elif dx < 0.0 and dy >= 0.0:
            base = 180.0
        else:
            base = -90.0
        if rotate_180:
            base += 180.0
        base += float(extra_rot_deg)
        return normalize_angle(base)

    def emit_str(x, y, ang_deg, flipped):
        if flipped:
            return f"1,{ramp_id},2,{x:.3f},3,{y:.3f},5,1,6,{ang_deg:.3f},155,1;"
        return f"1,{ramp_id},2,{x:.3f},3,{y:.3f},6,{ang_deg:.3f},155,1;"

    def place_leg(x_leg, y_leg, dx0, dy0):
        END_MARGIN = 0
        if x_leg[-1] > xs[-1] - END_MARGIN:
            return ""
        if len(x_leg) < 2:
            return ""

        rot = final_rot_from_dxdy(dx0, dy0)
        eps = 1e-6
        downward = dy0 < -eps

        if downward:
            rot = normalize_angle(rot - 90.0)

        leg_flip = False if downward else flip_base
        spacing = float(spacing_units)

        leg_len_x = float(x_leg[-1] - x_leg[0])
        too_short = leg_len_x < spacing / 1.5

        dx0_leg = float(x_leg[-1] - x_leg[0])
        dy0_leg = float(y_leg[-1] - y_leg[0])
        m = 0.0 if abs(dx0_leg) < 1e-9 else (dy0_leg / dx0_leg)

        x0 = float(x_leg[0])
        y0_val = float(y_leg[0])
        x1 = float(x_leg[-1])

        x_start = x0 + 15.0
        y_start = y0_val + m * (x_start - x0)
        x_end = x1 - 15.0
        y_end = y0_val + m * (x_end - x0)

        if not too_short:
            objs = [emit_str(x_start, y_start, rot, leg_flip)]
            # Vectorized intermediate ramps
            ramp_spacing = (spacing / 3.0) * 2
            if ramp_spacing > 1e-6 and x_end - x_start > ramp_spacing:
                mid_xs = np.arange(x_start + ramp_spacing, x_end, ramp_spacing)
                if len(mid_xs) > 0:
                    t_vals = (mid_xs - x_start) / max(x_end - x_start, 1e-9)
                    mid_ys = y_start + t_vals * (y_end - y_start)
                    for mx, my in zip(mid_xs, mid_ys):
                        objs.append(emit_str(float(mx), float(my), rot, leg_flip))
            objs.append(emit_str(x_end, y_end, rot, leg_flip))
            return "".join(objs)

        objs = []
        if is_top:
            if dy0 > eps:
                objs.append(emit_str(x_start, y_start, rot, leg_flip))
            elif dy0 < -eps:
                objs.append(emit_str(x_end, y_end, rot, leg_flip))
            else:
                objs.append(emit_str(x_start, y_start, rot, leg_flip))
                objs.append(emit_str(x_end, y_end, rot, leg_flip))
        else:
            if dy0 > eps:
                objs.append(emit_str(x_end, y_end, rot, leg_flip))
            elif dy0 < -eps:
                objs.append(emit_str(x_start, y_start, rot, leg_flip))
            else:
                objs.append(emit_str(x_start, y_start, rot, leg_flip))
                objs.append(emit_str(x_end, y_end, rot, leg_flip))
        return "".join(objs)

    # leg detection
    seg_dx = np.diff(xs)
    seg_dy = np.diff(ys)

    # "same direction" = both dy same sign (or both zero)
    dy_sign = np.sign(seg_dy) # -1, 0, +1
    # A change happens when consecutive dy_signs have opposite nonzero signs
    direction_change = np.zeros(n - 2, dtype=bool)
    for i in range(n - 2):
        d1y = float(seg_dy[i])
        d2y = float(seg_dy[i + 1])
        # same_dir if both >=0 or both <0
        same = (d1y >= 0.0 and d2y >= 0.0) or (d1y < 0.0 and d2y < 0.0)
        direction_change[i] = not same

    result_objs = []
    leg_start = 0
    change_indices = np.where(direction_change)[0] + 1 # indices where leg ends

    for ci in change_indices:
        i = int(ci)
        x_leg = xs[leg_start:i + 1]
        y_leg = ys[leg_start:i + 1]
        result_objs.append(place_leg(x_leg, y_leg, float(seg_dx[leg_start]), float(seg_dy[leg_start])))
        leg_start = i

    # Last leg
    x_leg = xs[leg_start:n]
    y_leg = ys[leg_start:n]
    result_objs.append(place_leg(x_leg, y_leg, float(seg_dx[leg_start]), float(seg_dy[leg_start])))

    return "".join(result_objs)