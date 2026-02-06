from xml.sax.saxutils import escape
import numpy as np
import math

def xml_kv_string(k, v): return f"<k>{escape(k)}</k><s>{escape(v)}</s>"
def xml_kv_int(k, v): return f"<k>{escape(k)}</k><i>{v}</i>"
def xml_kv_true(k): return f"<k>{escape(k)}</k><t />"
def xml_kv_real(k, v): return f"<k>{escape(k)}</k><r>{v}</r>"

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

def build_k4_polyline(t_samp, y_samp, units_per_second, start_offset_s, block_id=1764, y_add=15.0, start_as_wave=False):
    header = "kS38,1_40_2_125_3_255_11_255_12_255_13_255_4_-1_6_1000_7_1_15_1_18_0_8_1,kA13,0,kA6,1,kA16,1,kA15,1,k128,0;"
    objs = []

    start_x = 0.0
    start_y = float(y_add)

    if start_as_wave:
        objs.append(f"1,660,2,{start_x:.3f},3,{start_y:.3f},155,1;")

    for t, y in zip(t_samp, y_samp):
        x = (float(t) + float(start_offset_s)) * float(units_per_second)
        yy = float(y) + float(y_add)
        objs.append(f"1,{block_id},2,{x:.3f},3,{yy:.3f},155,1;")

    return header + "".join(objs)

def build_objects_along_path_by_spacing(
    times_s, y_s, units_per_second, start_offset_s,
    spacing_units, block_id, y_add=15.0
):
    if len(times_s) == 0:
        return ""
    U = float(units_per_second)
    xs = (np.asarray(times_s, dtype=np.float64) + float(start_offset_s)) * U
    ys = np.asarray(y_s, dtype=np.float64) + float(y_add)

    objs = []
    px, py = float(xs[0]), float(ys[0])
    objs.append(f"1,{int(block_id)},2,{px:.3f},3,{py:.3f},155,1;")
    remain = float(spacing_units)

    for i in range(1, len(xs)):
        x1, y1 = float(xs[i-1]), float(ys[i-1])
        x2, y2 = float(xs[i]),   float(ys[i])
        dx = x2 - x1
        dy = y2 - y1
        seg = math.hypot(dx, dy)
        if seg <= 1e-9:
            continue

        while seg >= remain - 1e-9:
            a = remain / seg
            px = x1 + a * dx
            py = y1 + a * dy
            objs.append(f"1,{int(block_id)},2,{px:.3f},3,{py:.3f},155,1;")
            x1, y1 = px, py
            dx = x2 - x1
            dy = y2 - y1
            seg = math.hypot(dx, dy)
            remain = float(spacing_units)

        remain -= seg
        if remain < 1e-9:
            remain = float(spacing_units)

    return "".join(objs)

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
):
    header = "kS38,1_40_2_125_3_255_11_255_12_255_13_255_4_-1_6_1000_7_1_15_1_18_0_8_1,kA13,0,kA6,1,kA16,1,kA15,1,k128,0;"
    objs = []

    U = float(units_per_second)
    off = float(start_offset_s)
    ya = float(y_add)

    if start_with_cube:
        objs.append(f"1,{int(cube_portal_id)},2,0.000,3,{ya:.3f},155,1;")

    # Place WHITE CIRCLES along the arc ONCE (spacing in GD units)
    objs.append(
        build_objects_along_path_by_spacing(
            times_s=t_samp,
            y_s=y_samp,
            units_per_second=units_per_second,
            start_offset_s=start_offset_s,
            spacing_units=1.0,   # 1.0–2.0 is sane
            block_id=int(block_id),
            y_add=ya
        )
    )

    if orb_times is None or orb_types is None:
        return header + "".join(objs)

    orb_times = np.asarray(orb_times, dtype=np.float64)
    orb_types = np.asarray(orb_types, dtype=np.int32)

    if len(orb_times) == 0:
        return header + "".join(objs)

    t_s = np.asarray(t_samp, dtype=np.float64)
    y_s = np.asarray(y_samp, dtype=np.float64)

    id_by_type = {
        0: int(orb_id_yellow),
        1: int(orb_id_purple),
        2: int(orb_id_blue),
        3: int(orb_id_green),
    }

    # Place ORBS only at orb_times (beats + your extension)
    for t, typ in zip(orb_times, orb_types):
        x = (float(t) + off) * U + float(orb_x_offset)
        y_on_arc = float(np.interp(float(t), t_s, y_s))
        yy = max(0.0, y_on_arc) + float(orb_y_offset)
        oid = id_by_type.get(int(typ), int(orb_id_yellow))
        objs.append(f"1,{oid},2,{x:.3f},3,{yy:.3f},155,1;")

    return header + "".join(objs)



def serialize_gmd(level_name, creator_name, song_id, k4_plain):
    level_xml = build_level_xml(level_name, creator_name, k4_plain, song_id)
    return '<?xml version="1.0"?><plist version="1.0" gjver="2.0">' + level_xml + "</plist>"

def build_wave_ramps_45deg(t_samp, y_samp, units_per_second, start_offset_s, ramp_id=309, ramp_size_units=30.0, y_add=15.0):
    if len(t_samp) < 2:
        return ""

    U = float(units_per_second)
    xs = (np.asarray(t_samp, dtype=np.float64) + float(start_offset_s)) * U
    ys = (np.asarray(y_samp, dtype=np.float64) + float(y_add))

    spacing = float(ramp_size_units) * math.sqrt(2.0)
    objs = []

    dist_acc = 0.0
    last_x = float(xs[0])
    last_y = float(ys[0])

    for i in range(1, len(xs)):
        x0, y0 = float(xs[i - 1]), float(ys[i - 1])
        x1, y1 = float(xs[i]), float(ys[i])

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

            # 45° ramps: up-slope => +45, down-slope => -45
            rot = 45.0 if dy >= 0.0 else -45.0

            objs.append(f"1,{int(ramp_id)},2,{px:.3f},3,{py:.3f},6,{rot:.3f},155,1;")

            x0, y0 = px, py
            dx = x1 - x0
            dy = y1 - y0
            seg = math.hypot(dx, dy)
            dist_acc = 0.0

        dist_acc += seg
        last_x, last_y = x1, y1

    return "".join(objs)


import math
import numpy as np

Y_SHIFT = 15.0

def build_ramps_along_path_by_spacing(
    times_s, y_s, units_per_second, start_offset_s,
    spacing_units, ramp_id, y_add=15.0,
    extra_rot_deg=0.0, rotate_180=False,
    invert_if_top=False, invert_if_bottom=False
):
    if len(times_s) == 0:
        return ""
    y_add = 0
    U = float(units_per_second)
    xs = (np.asarray(times_s, dtype=np.float64) + float(start_offset_s)) * U
    ys = np.asarray(y_s, dtype=np.float64) + float(y_add) + float(Y_SHIFT)

    is_top = bool(rotate_180)
    flip_for_this_call = (is_top and invert_if_top) or ((not is_top) and invert_if_bottom)

    def emit(x, y, ang_deg, flipped):
        if flipped:
            # 5,1 = vertical flip in GD. If your editor uses 4,1 for this, swap to 4.
            return f"1,{ramp_id},2,{x:.3f},3,{y:.3f},5,1,6,{ang_deg:.3f},155,1;"
        else:
            return f"1,{ramp_id},2,{x:.3f},3,{y:.3f},6,{ang_deg:.3f},155,1;"



    def normalize_angle(a):
        a = float(a)
        while a <= -180.0:
            a += 360.0
        while a > 180.0:
            a -= 360.0
        return a

    def base_cardinal_from_dxdy(dx, dy):
        eps = 1e-6
        if abs(dx) < eps and abs(dy) < eps:
            return 0.0

        if dx >= 0.0 and dy < 0.0:
            return 180
        if dx >= 0.0 and dy >= 0.0:
            return 0.0
        if dx < 0.0 and dy >= 0.0:
            return 180.0
        return -90.0

    def final_rot_from_dxdy(dx, dy):
        a = base_cardinal_from_dxdy(dx, dy)
        if rotate_180:
            a += 180.0
        a += float(extra_rot_deg)
        return normalize_angle(a)

    def same_dir(dx1, dy1, dx2, dy2):
        eps = 1e-6
        if abs(dx1) < eps and abs(dx2) < eps and abs(dy1) < eps and abs(dy2) < eps:
            return True
        return (dy1 >= 0.0 and dy2 >= 0.0) or (dy1 < 0.0 and dy2 < 0.0)

    def place_leg(x_leg, y_leg, dx0, dy0):
        END_MARGIN = 0//1e-9

        if x_leg[-1] > xs[-1] - END_MARGIN:
            return ""

        if len(x_leg) < 2:
            return ""

        rot = final_rot_from_dxdy(dx0, dy0)
        objs = []

        eps = 1e-6
        is_top = bool(rotate_180)
        downward = dy0 < -eps

        if downward:
            rot = normalize_angle(rot - 90.0)

        # Base flip from flags
        base_flip = (is_top and invert_if_top) or ((not is_top) and invert_if_bottom)

        # On downward-tilting segments: both top and bottom MUST be inverted
        leg_flip = False if downward else base_flip

        spacing = float(spacing_units)

        # leg horizontal length
        leg_len_x = float(x_leg[-1] - x_leg[0])
        too_short = leg_len_x < spacing/1.5

        dx0 = float(x_leg[-1] - x_leg[0])
        dy0_leg = float(y_leg[-1] - y_leg[0])
        m = 0.0 if abs(dx0) < 1e-9 else (dy0_leg / dx0)

        x0 = float(x_leg[0])
        y0 = float(y_leg[0])
        x1 = float(x_leg[-1])
        y1 = float(y_leg[-1])

        x_start = x0 + 15.0
        y_start = y0 + m * (x_start - x0)

        x_end = x1 - 15.0
        y_end = y0 + m * (x_end - x0)


        if not too_short:
            objs.append(emit(x_start, y_start, rot, leg_flip))
            
            # Add intermediate ramps between start and end
            # Use a smaller spacing - divide by 3 or 4 for more ramps
            ramp_spacing = (spacing / 3.0)*2
            current_x = x_start + ramp_spacing
            while current_x <= x_end:  # Stop before overlapping end ramp
                # Interpolate y position based on x position
                t = (current_x - x_start) / (x_end - x_start)
                current_y = y_start + t * (y_end - y_start)
                objs.append(emit(current_x, current_y, rot, leg_flip))
                current_x += ramp_spacing
            
            objs.append(emit(x_end, y_end, rot, leg_flip))
            return "".join(objs)

        # too short segment: apply your rules
        is_top = bool(rotate_180)

        if is_top:
            # top ramp rules
            if dy0 > eps:
                # top ramp at start of an increasing segment
                objs.append(emit(x_start, y_start, rot, leg_flip))
            elif dy0 < -eps:
                # top ramp at end of a decreasing segment
                objs.append(emit(x_end, y_end, rot, leg_flip))
            else:
                # flat: just keep both to avoid killing ramps everywhere
                objs.append(emit(x_start, y_start, rot, leg_flip))
                objs.append(emit(x_end, y_end, rot, leg_flip))
        else:
            # bottom ramp rules
            if dy0 > eps:
                # bottom ramp at end of an increasing segment
                objs.append(emit(x_end, y_end, rot, leg_flip))
            elif dy0 < -eps:
                # bottom ramp at start of a decreasing segment
                objs.append(emit(x_start, y_start, rot, leg_flip))
            else:
                # flat: same fallback
                objs.append(emit(x_start, y_start, rot, leg_flip))
                objs.append(emit(x_end, y_end, rot, leg_flip))

        return "".join(objs)

    n = len(xs)
    if n < 2:
        return ""

    seg_dx = []
    seg_dy = []
    for i in range(n - 1):
        seg_dx.append(float(xs[i+1] - xs[i]))
        seg_dy.append(float(ys[i+1] - ys[i]))

    result_objs = []
    leg_start = 0
    for i in range(1, n - 1):
        if not same_dir(seg_dx[i-1], seg_dy[i-1], seg_dx[i], seg_dy[i]):
            x_leg = xs[leg_start:i+1]
            y_leg = ys[leg_start:i+1]
            result_objs.append(
                place_leg(x_leg, y_leg, seg_dx[leg_start], seg_dy[leg_start])
            )
            leg_start = i

    x_leg = xs[leg_start:n]
    y_leg = ys[leg_start:n]
    result_objs.append(
        place_leg(x_leg, y_leg, seg_dx[leg_start], seg_dy[leg_start])
    )

    return "".join(result_objs)

def wave_make_clones(y_samp, gap_units, y_min, y_max, y_ceil):
    if gap_units is None:
        return None, None

    gap = float(gap_units)
    if gap <= 0.0:
        return None, None

    cap_max = min(float(y_ceil), float(y_max))

    above = np.asarray(y_samp, dtype=np.float64) + gap
    above = np.clip(above, max(float(y_min), 0.0), cap_max)

    below = np.asarray(y_samp, dtype=np.float64) - gap
    below = np.clip(below, float(y_min), cap_max)

    return above, below
