"""
EAS Wind Pressure Estimator  (Components & Cladding, walls, h <= 60 ft)

Estimating tool for sizing window walls / curtain wall / storefront against wind.
The estimator enters the wind speed manually (from the ASCE Hazard Tool / local code).

Method: ASCE 7 Chapter 30, Part 1 (low-rise, enclosed / partially enclosed), wall zones 4 and 5.
    qh = 0.00256 * Kz * Kzt * Kd * Ke * V^2
    p  = qh * (GCp - GCpi)
Pressures are ULTIMATE (strength level). ASD = 0.6 x ultimate.

NOT for design or permit use. The project engineer's / specification's design pressure governs.

Run:  streamlit run wind_pressure_app.py
"""

import hmac
import math
import re
import time
from pathlib import Path

import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
PSF_TO_PA = 47.880259
PSF_TO_KNM2 = 0.04788026
ASD_FACTOR = 0.6
KD_CC = 0.85
MIN_PSF = 16.0  # minimum C&C pressure (ultimate), either direction

# Exposure: (alpha, zg [ft])
EXPOSURE = {"B": (7.0, 1200.0), "C": (9.5, 900.0), "D": (11.5, 700.0)}

# GCp for walls, h <= 60 ft (Figure 30.3-1), at 10 sf and at 500 sf; log-linear in between
GCP = {
    4: {"pos": (1.0, 0.7), "neg": (-1.1, -0.8)},
    5: {"pos": (1.0, 0.7), "neg": (-1.4, -0.8)},
}


# ----------------------------------------------------------------------------
# Calculation functions (no Streamlit code in here, so they can be tested)
# ----------------------------------------------------------------------------
def kz_factor(h_ft: float, exposure: str) -> float:
    alpha, zg = EXPOSURE[exposure]
    z = max(h_ft, 15.0)
    return 2.01 * (z / zg) ** (2.0 / alpha)


def ke_factor(elevation_ft: float) -> float:
    return math.exp(-0.0000362 * elevation_ft)


def velocity_pressure(v_mph, kz, kzt, kd, ke) -> float:
    return 0.00256 * kz * kzt * kd * ke * v_mph ** 2


def gcp(zone: int, sign: str, area_sf: float) -> float:
    """Log-interpolated GCp between 10 sf and 500 sf (held constant outside)."""
    lo, hi = GCP[zone][sign]
    a = min(max(area_sf, 10.0), 500.0)
    t = math.log(a / 10.0) / math.log(50.0)
    return lo + (hi - lo) * t


def pressure(q_psf, zone, area_sf, gcpi):
    """Return dict with positive/negative ultimate pressure (psf) and flags."""
    gp, gn = gcp(zone, "pos", area_sf), gcp(zone, "neg", area_sf)
    pos = q_psf * (gp + gcpi)
    neg = q_psf * (gn - gcpi)
    return {
        "gcp_pos": gp,
        "gcp_neg": gn,
        "pos": max(pos, MIN_PSF),
        "neg": min(neg, -MIN_PSF),
        "min_governs": pos < MIN_PSF or neg > -MIN_PSF,
    }


def member_area(span_ft: float, trib_ft: float) -> float:
    """Effective wind area of a mullion/transom: span x max(trib, span/3)."""
    return span_ft * max(trib_ft, span_ft / 3.0)


def zone5_width(least_dim_ft: float, h_ft: float) -> float:
    a = min(0.10 * least_dim_ft, 0.40 * h_ft)
    return max(a, 0.04 * least_dim_ft, 3.0)


def parse_length(text: str) -> float:
    """Parse a length to feet. Accepts 9.5, 2'-6", 3'-5 3/4", 9' 1 3/4", 30", 30 in, 4'."""
    s = text.strip().lower()
    for a, b in (("’", "'"), ("′", "'"), ("”", '"'), ("″", '"'), ("“", '"')):
        s = s.replace(a, b)
    if not s:
        raise ValueError("empty")

    def inches(part: str) -> float:
        part = part.strip().lstrip("-").strip()
        part = part.replace('"', " ").replace("in", " ").strip()
        if not part:
            return 0.0
        total = 0.0
        for tok in part.split():
            if "/" in tok:
                n, d = tok.split("/")
                total += float(n) / float(d)
            else:
                total += float(tok)
        return total

    if "'" in s or "ft" in s:
        sep = "'" if "'" in s else "ft"
        feet_s, rest = s.split(sep, 1)
        return float(feet_s) + inches(rest) / 12.0
    if s.endswith('"') or s.endswith("in"):
        return inches(s) / 12.0
    return float(s)  # plain number = feet


def parse_list(text: str):
    items = [t for t in re.split(r"[,;\n]", text) if t.strip()]
    return [parse_length(t) for t in items]


P1_CLASSES = [(1, 400), (2, 800), (3, 1200), (4, 1600), (5, 2000)]  # EN 12210 / EN 13116 style, Pa


def wind_load_class(p_pa: float):
    """Lowest class whose test pressure P1 is at least the design load. None if above class 5."""
    for cls, p1 in P1_CLASSES:
        if p1 >= p_pa:
            return cls, p1
    return None, None


def deflection_class(denominator: float) -> str:
    """Frontal deflection class (A 1/150, B 1/200, C 1/300) at least as strict as the spec limit L/denominator."""
    for letter, d in (("A", 150), ("B", 200), ("C", 300)):
        if d >= denominator:
            return f"{letter} (< 1/{d})"
    return "C (< 1/300), stricter than any EN class is needed: ask the factory"


def conv(psf: float) -> dict:
    return {"psf": psf, "Pa": psf * PSF_TO_PA, "kN/m2": psf * PSF_TO_KNM2}


# ----------------------------------------------------------------------------
# Settings fixed by the Technical Director (NOT estimator inputs)
# ----------------------------------------------------------------------------
FACTORY_CONVENTION = "characteristic"  # "characteristic", "ultimate" or "asd"
GAMMA = 1.5  # safety factor the factory software applies to the entered Wind load (characteristic only)


def convention_factor() -> float:
    if FACTORY_CONVENTION == "ultimate":
        return 1.0
    if FACTORY_CONVENTION == "asd":
        return ASD_FACTOR
    return 1.0 / GAMMA


def convention_text() -> str:
    return {
        "ultimate": "ultimate load (factor 1.0)",
        "asd": "ASD load (0.6 x ultimate)",
        "characteristic": f"service load (ultimate / {GAMMA})",
    }[FACTORY_CONVENTION]


def analyze(zone, q, gcpi, bays, rows, span, anchor_area):
    """All members for one wall zone. Returns tables and the value to give the factory."""
    n = len(bays)
    mull = []
    for i in range(n + 1):
        if i == 0:
            name, trib = "Left jamb", bays[0] / 2
        elif i == n:
            name, trib = "Right jamb", bays[-1] / 2
        else:
            name, trib = f"Mullion {i}", (bays[i - 1] + bays[i]) / 2
        area = member_area(span, trib)
        p = pressure(q, zone, area, gcpi)
        gov_p = max(p["pos"], -p["neg"])
        mull.append({
            "Member": name, "Span (ft)": span, "Trib. width (ft)": trib, "Eff. area (sf)": area,
            "p+ ult (psf)": p["pos"], "p- ult (psf)": p["neg"],
            "Governing ult (kN/m2)": gov_p * PSF_TO_KNM2,
            "Line load ult (kN/m)": gov_p * PSF_TO_KNM2 * trib * 0.3048,
        })
    mull_df = pd.DataFrame(mull)

    longest = max(bays)
    tr = []
    nr = len(rows)
    for j in range(nr + 1):
        if j == 0:
            name, trib = "Head", rows[0] / 2
        elif j == nr:
            name, trib = "Sill", rows[-1] / 2
        else:
            name, trib = f"Transom {j}", (rows[j - 1] + rows[j]) / 2
        area = member_area(longest, trib)
        p = pressure(q, zone, area, gcpi)
        tr.append({
            "Member": name, "Span (ft)": longest, "Trib. height (ft)": trib, "Eff. area (sf)": area,
            "p+ ult (psf)": p["pos"], "p- ult (psf)": p["neg"],
            "Governing ult (kN/m2)": max(p["pos"], -p["neg"]) * PSF_TO_KNM2,
        })
    tr_df = pd.DataFrame(tr)

    gl = []
    for ri, rh in enumerate(rows, 1):
        for bi, bw in enumerate(bays, 1):
            area = bw * rh
            p = pressure(q, zone, area, gcpi)
            gl.append({
                "Lite": f"R{ri}-B{bi}", "Width (ft)": bw, "Height (ft)": rh, "Area (sf)": area,
                "p+ ult (psf)": p["pos"], "p- ult (psf)": p["neg"],
                "Governing ult (kN/m2)": max(p["pos"], -p["neg"]) * PSF_TO_KNM2,
            })
    glass_df = pd.DataFrame(gl)

    anchors = pressure(q, zone, anchor_area, gcpi)

    gov = mull_df.loc[mull_df["Governing ult (kN/m2)"].idxmax()]
    gov_psf = float(gov["Governing ult (kN/m2)"]) / PSF_TO_KNM2
    gp = pressure(q, zone, float(gov["Eff. area (sf)"]), gcpi)
    send_psf = gov_psf * convention_factor()
    return {
        "zone": zone, "mull": mull_df, "trans": tr_df, "glass": glass_df, "anchors": anchors,
        "gov_member": gov["Member"], "gov_dir": "suction" if -gp["neg"] >= gp["pos"] else "positive",
        "gov_psf": gov_psf, "send_psf": send_psf,
        "send_knm2": send_psf * PSF_TO_KNM2, "send_pa": send_psf * PSF_TO_PA,
    }


# ----------------------------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------------------------
def fmt_ft_in(ft: float) -> str:
    """Feet to a feet-inches string, rounded to the nearest 1/4 inch (for the sketch labels)."""
    total_in = round(ft * 12 * 4) / 4
    feet = int(total_in // 12)
    inch = total_in - feet * 12
    whole = int(inch)
    frac = ["", " 1/4", " 1/2", " 3/4"][int(round((inch - whole) * 4)) % 4]
    return f"{feet}'-{whole}{frac}\""


def sketch_svg(bays, rows) -> str:
    """Dimensioned sketch of the window wall: panels named R(row)-B(bay), jambs J and mullions M."""
    W, H = sum(bays), sum(rows)
    ml, mt, mr, mb = 80, 66, 26, 52
    scale = min(440.0 / W, 340.0 / H)
    dw, dh = W * scale, H * scale
    width, height = ml + dw + mr, mt + dh + mb
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.0f} {height:.0f}" '
         f'width="{width:.0f}" height="{height:.0f}" font-family="Arial, Helvetica, sans-serif">',
         '<rect width="100%" height="100%" fill="white"/>']
    xs = [ml]
    for b in bays:
        xs.append(xs[-1] + b * scale)
    ys = [mt]
    for r in rows:
        ys.append(ys[-1] + r * scale)

    # panels
    for ri, rh in enumerate(rows):
        for bi, bw in enumerate(bays):
            x0, x1, y0, y1 = xs[bi], xs[bi + 1], ys[ri], ys[ri + 1]
            s.append(f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{x1 - x0:.1f}" height="{y1 - y0:.1f}" '
                     f'fill="#dbeafe" stroke="#1f2937" stroke-width="3"/>')
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            big = min(x1 - x0, y1 - y0) > 36
            s.append(f'<text x="{cx:.1f}" y="{cy:.1f}" text-anchor="middle" font-size="{13 if big else 9}" '
                     f'font-weight="bold" fill="#1e3a8a">R{ri + 1}-B{bi + 1}</text>')
            if (x1 - x0) > 74 and (y1 - y0) > 44:
                s.append(f'<text x="{cx:.1f}" y="{cy + 14:.1f}" text-anchor="middle" font-size="10" '
                         f'fill="#374151">{fmt_ft_in(bw)} x {fmt_ft_in(rh)}</text>')

    # top dimensions (panel widths, then total)
    for bi, bw in enumerate(bays):
        x0, x1 = xs[bi], xs[bi + 1]
        y = mt - 12
        s.append(f'<line x1="{x0:.1f}" y1="{y}" x2="{x1:.1f}" y2="{y}" stroke="#6b7280" stroke-width="1"/>')
        for x in (x0, x1):
            s.append(f'<line x1="{x:.1f}" y1="{y - 4}" x2="{x:.1f}" y2="{y + 4}" stroke="#6b7280" stroke-width="1"/>')
        s.append(f'<text x="{(x0 + x1) / 2:.1f}" y="{y - 5}" text-anchor="middle" '
                 f'font-size="{10 if (x1 - x0) > 50 else 8}" fill="#111827">{fmt_ft_in(bw)}</text>')
    y = mt - 38
    s.append(f'<line x1="{xs[0]:.1f}" y1="{y}" x2="{xs[-1]:.1f}" y2="{y}" stroke="#111827" stroke-width="1.3"/>')
    for x in (xs[0], xs[-1]):
        s.append(f'<line x1="{x:.1f}" y1="{y - 4}" x2="{x:.1f}" y2="{y + 4}" stroke="#111827" stroke-width="1.3"/>')
    s.append(f'<text x="{(xs[0] + xs[-1]) / 2:.1f}" y="{y - 5}" text-anchor="middle" font-size="11" '
             f'font-weight="bold" fill="#111827">{fmt_ft_in(W)}</text>')

    # left dimensions (panel heights, then total)
    for ri, rh in enumerate(rows):
        y0, y1 = ys[ri], ys[ri + 1]
        x = ml - 14
        s.append(f'<line x1="{x}" y1="{y0:.1f}" x2="{x}" y2="{y1:.1f}" stroke="#6b7280" stroke-width="1"/>')
        for yy in (y0, y1):
            s.append(f'<line x1="{x - 4}" y1="{yy:.1f}" x2="{x + 4}" y2="{yy:.1f}" stroke="#6b7280" stroke-width="1"/>')
        s.append(f'<text x="{x - 6}" y="{(y0 + y1) / 2:.1f}" text-anchor="middle" font-size="{10 if (y1 - y0) > 50 else 8}" '
                 f'fill="#111827" transform="rotate(-90 {x - 6} {(y0 + y1) / 2:.1f})">{fmt_ft_in(rh)}</text>')
    x = ml - 46
    s.append(f'<line x1="{x}" y1="{ys[0]:.1f}" x2="{x}" y2="{ys[-1]:.1f}" stroke="#111827" stroke-width="1.3"/>')
    for yy in (ys[0], ys[-1]):
        s.append(f'<line x1="{x - 4}" y1="{yy:.1f}" x2="{x + 4}" y2="{yy:.1f}" stroke="#111827" stroke-width="1.3"/>')
    s.append(f'<text x="{x - 6}" y="{(ys[0] + ys[-1]) / 2:.1f}" text-anchor="middle" font-size="11" font-weight="bold" '
             f'fill="#111827" transform="rotate(-90 {x - 6} {(ys[0] + ys[-1]) / 2:.1f})">{fmt_ft_in(H)}</text>')

    # vertical members under the drawing
    n = len(bays)
    for i, x in enumerate(xs):
        name = "J" if i in (0, n) else f"M{i}"
        s.append(f'<text x="{x:.1f}" y="{ys[-1] + 16:.1f}" text-anchor="middle" font-size="11" '
                 f'font-weight="bold" fill="#b45309">{name}</text>')
    s.append(f'<text x="{ml + dw / 2:.1f}" y="{ys[-1] + 36:.1f}" text-anchor="middle" font-size="10" fill="#4b5563">'
             f'Panels: R = row from the top, B = bay from the left.  J = jamb, M = vertical mullion.</text>')
    s.append("</svg>")
    return "".join(s)


def show_logo():
    """Show the EAS logo if eas_logo.png/.jpg/.svg/.webp sits next to this file."""
    try:
        here = Path(__file__).resolve().parent
    except NameError:
        here = Path.cwd()
    for ext in ("png", "jpg", "jpeg", "svg", "webp"):
        p = here / f"eas_logo.{ext}"
        if p.exists():
            st.image(str(p), width=260)
            return
    st.markdown("**European Architectural Supply**")


# ----------------------------------------------------------------------------
# Login (credentials live in Streamlit secrets, never in this file)
#
#   .streamlit/secrets.toml  (local)  or  App settings > Secrets  (Streamlit Cloud)
#       [credentials.users]
#       username = "password"
# ----------------------------------------------------------------------------
MAX_ATTEMPTS = 5
LOCKOUT_SECONDS = 60


def _load_users() -> dict:
    try:
        users = st.secrets["credentials"]["users"]
        return {str(k).strip().lower(): str(v) for k, v in users.items()}
    except Exception:
        return {}


def check_login(username: str, password: str, users: dict) -> bool:
    """Constant-time comparison. Usernames are not case sensitive, passwords are."""
    key = username.strip().lower()
    stored = users.get(key)
    if stored is None:
        hmac.compare_digest(password.encode(), b"x" * 16)  # keep timing similar
        return False
    return hmac.compare_digest(password.encode(), stored.encode())


def require_login() -> None:
    """Show the login form and stop the script until the user is signed in."""
    if st.session_state.get("auth_user"):
        return
    users = _load_users()
    show_logo()
    st.title("Wind Load for Window Walls")
    if not users:
        st.error("Login is not configured. Add [credentials.users] to the Streamlit secrets.")
        st.stop()

    locked_until = st.session_state.get("locked_until", 0.0)
    wait = int(locked_until - time.time())
    if wait > 0:
        st.error(f"Too many failed attempts. Try again in {wait} seconds.")
        st.stop()

    with st.form("login"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in", type="primary")
    if submitted:
        if check_login(username, password, users):
            st.session_state["auth_user"] = username.strip().lower()
            st.session_state["failed"] = 0
            st.rerun()
        fails = st.session_state.get("failed", 0) + 1
        st.session_state["failed"] = fails
        if fails >= MAX_ATTEMPTS:
            st.session_state["locked_until"] = time.time() + LOCKOUT_SECONDS
            st.session_state["failed"] = 0
        st.error("Wrong username or password.")
    st.stop()


def main():
    st.set_page_config(page_title="EAS Wind Load", layout="centered")
    require_login()
    with st.sidebar:
        st.write(f"Signed in as **{st.session_state['auth_user']}**")
        if st.button("Sign out"):
            st.session_state.clear()
            st.rerun()
    show_logo()
    st.title("Wind Load for Window Walls")
    st.caption("Enter the wind speed, building height and window wall layout. "
               "You get the kN/m² to give the factory.")

    # ---------------- Main inputs ----------------
    c1, c2, c3 = st.columns(3)
    v = c1.number_input("Wind speed (mph)", min_value=50.0, max_value=250.0, value=115.0, step=1.0,
                        help="Ultimate wind speed from the ASCE Hazard Tool for the building's Risk Category (usually II). "
                             "Link below.")
    h = c2.number_input("Building height (ft)", min_value=5.0, value=21.0, step=1.0,
                        help="Mean roof height of the building.")
    exposure = c3.selectbox(
        "Surroundings", ["C", "B", "D"],
        format_func=lambda x: {"C": "C: open terrain (default)", "B": "B: trees / suburban",
                               "D": "D: open coast"}[x],
        help="If you do not know, leave C. It is the conservative choice.")

    st.caption(
        "Wind speed: use the [ASCE Hazard Tool](https://ascehazardtool.org/). Enter the project address, "
        "choose the ASCE 7 edition the local code uses (7-16 or 7-22), select the building's Risk Category "
        "(II for most buildings), and read the wind speed. Not sure? Ask the architect or engineer."
    )

    g1, g2 = st.columns(2)
    bays_txt = g1.text_input("Panel widths, left to right", "2'-6\", 4'-0\", 2'-6\"",
                             help="Width of each panel between vertical mullions, separated by commas. "
                                  "Example: 2'-6\", 4'-0\", 2'-6\"  or  2.5, 4, 2.5. "
                                  "No vertical mullion? Enter the full width only, e.g. 9'-0\".")
    rows_txt = g2.text_input("Panel heights, top to bottom", "3'-5 3/4\", 5'-8\"",
                             help="Height of each panel between horizontal bars, top to bottom, separated by commas. "
                                  "No horizontal bar? Enter the full height only, e.g. 9'-1 3/4\".")

    # Sketch of what was entered, so the layout can be checked at a glance
    try:
        _b, _r = parse_list(bays_txt), parse_list(rows_txt)
        if _b and _r and min(_b + _r) > 0:
            st.image(sketch_svg(_b, _r))
    except Exception:
        pass

    # ---------------- Advanced (defaults are fine) ----------------
    with st.expander("Advanced (leave as default unless you know)"):
        a1, a2 = st.columns(2)
        kzt = a1.number_input("Topographic factor Kzt", min_value=1.0, value=1.0, step=0.05,
                              help="1.0 for flat sites. Hills and ridges can raise it.")
        elev = a2.number_input("Ground elevation (ft)", min_value=0.0, value=0.0, step=50.0,
                               help="0 keeps the conservative Ke = 1.0.")
        partially = st.checkbox("Partially enclosed building",
                                help="For example, wind-borne debris region without impact-rated glazing. Raises internal pressure.")
        a3, a4 = st.columns(2)
        least_dim = a3.number_input("Least building width (ft), 0 = unknown", min_value=0.0, value=0.0, step=1.0,
                                    help="Only used to show how far from a corner Zone 5 reaches.")
        span_txt = a4.text_input("Mullion span (blank = full height)", "",
                                 help="Distance between mullion anchors. Blank uses the full height of the window wall.")
        a5, a6 = st.columns(2)
        anchor_area = a5.number_input("Anchor area (sf)", min_value=1.0, value=10.0, step=1.0)
        defl_denom = a6.selectbox("Deflection limit in spec (L / ...)", [150, 175, 200, 240, 300], index=1)

    gcpi = 0.55 if partially else 0.18
    if h > 60:
        st.error("This tool covers buildings up to 60 ft. Use the engineer's pressures for taller buildings.")
        st.stop()
    try:
        bays = parse_list(bays_txt)
        rows = parse_list(rows_txt)
        span = parse_length(span_txt) if span_txt.strip() else sum(rows)
    except Exception:
        st.error("Could not read a length. Use forms like 2'-6\", 3'-5 3/4\" or 2.5.")
        st.stop()
    if not bays or not rows or min(bays + rows + [span]) <= 0:
        st.error("Enter at least one panel width and one panel height.")
        st.stop()

    kz = kz_factor(h, exposure)
    ke = ke_factor(elev)
    q = velocity_pressure(v, kz, kzt, KD_CC, ke)
    res = {z: analyze(z, q, gcpi, bays, rows, span, anchor_area) for z in (4, 5)}

    # ---------------- Result ----------------
    st.divider()
    st.subheader("Wind load to give the factory")
    cols = st.columns(2)
    labels = {4: "Window away from corners (Zone 4)", 5: "Window near a corner (Zone 5)"}
    for col, z in zip(cols, (4, 5)):
        r = res[z]
        w_cls, w_p1 = wind_load_class(r["send_pa"])
        with col.container(border=True):
            st.caption(labels[z])
            st.metric("Wind load", f"{r['send_knm2']:.2f} kN/m²")
            st.write(f"{r['send_pa']:,.0f} Pa  ·  {r['send_psf']:.1f} psf")
            st.caption(f"Wind load class {w_cls if w_cls else 'above 5'}"
                       f"{f' (P1 {w_p1} Pa)' if w_cls else ''}  ·  deflection class "
                       f"{deflection_class(defl_denom).split(' ')[0]}")

    a_zone = zone5_width(least_dim, h) if least_dim > 0 else None
    st.info("Use the Zone 5 value if the window is within "
            + (f"{a_zone:.1f} ft" if a_zone else "about 10% of the building width (at least 3 ft)")
            + " of a building corner. Otherwise use Zone 4. If you are unsure, use Zone 5.")

    # ---------------- Details ----------------
    with st.expander("Show calculation details"):
        z = st.radio("Zone", [4, 5], horizontal=True, format_func=lambda x: f"Zone {x}")
        r = res[z]
        st.caption(f"Velocity pressure qh = {q:.2f} psf (Kz {kz:.3f}, Ke {ke:.3f}, Kd {KD_CC}). "
                   f"Governing mullion: {r['gov_member']}, {r['gov_dir']}. "
                   f"Ultimate {r['gov_psf']:.1f} psf = {r['gov_psf'] * PSF_TO_KNM2:.2f} kN/m2; "
                   f"ASD {r['gov_psf'] * ASD_FACTOR:.1f} psf.")
        t1, t2, t3, t4 = st.tabs(["Mullions", "Head / transoms / sill", "Glass", "Anchors"])
        t1.dataframe(r["mull"].round(3), hide_index=True, width="stretch")
        t2.dataframe(r["trans"].round(3), hide_index=True, width="stretch")
        t3.dataframe(r["glass"].round(3), hide_index=True, width="stretch")
        pa = r["anchors"]
        t4.write(f"Anchors at {anchor_area:.0f} sf: +{pa['pos'] * PSF_TO_KNM2:.2f} / "
                 f"-{-pa['neg'] * PSF_TO_KNM2:.2f} kN/m2 ultimate (ASD = 0.6 x these).")

    with st.expander("Method and limits"):
        st.markdown(
            f"""
- qh = 0.00256 · Kz · Kzt · Kd · Ke · V², p = qh · (GCp − GCpi), minimum 16 psf ultimate.
- Wall coefficients (h ≤ 60 ft): Zone 4 +1.0/−1.1 at 10 sf to +0.7/−0.8 at 500 sf. Zone 5 +1.0/−1.4 to +0.7/−0.8. The 10% wall reduction allowed for roof slopes of 10° or less is not applied (conservative).
- Mullions: effective area = span × larger of (tributary width, span ÷ 3). The value shown is the highest of all mullions and jambs.
- Value given to the factory is entered as {convention_text()}. This is a setting at the top of the app file (FACTORY_CONVENTION, GAMMA).
- Not covered: buildings above 60 ft, roofs and skylights, wind-borne debris and impact glazing, wind tunnel pressures, tornado loads, local amendments.
- Estimating only. The specification or the project engineer's design pressure governs.
            """
        )


if __name__ == "__main__":
    main()
