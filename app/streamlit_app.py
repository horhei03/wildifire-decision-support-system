"""
Wildfire Crew Extraction Warning System - Interactive Dashboard

Loads FARSITE fire simulations and computes real-time crew danger
based on fire-proximity analysis from arrival-time rasters.
Works for any number of crews at any position on the map.
"""

import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.farsite_parser import FARSITEParser

# ── Constants ────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = Path("C:/Users/horhe/OneDrive/Documents/DSS")

DOWNSAMPLE_FACTOR = 5
TIMESTEP_MINUTES = 5
DS_RESOLUTION = 25.0  # meters per cell after 5x downsample (5 m × 5)

PATCHES = [
    ("P1_darkforest",    "P1 - Dark Forest"),
    ("P2_shrub",         "P2 - Shrubland"),
    ("P3_drygrass",      "P3 - Dry Grassland"),
    ("P4_forestshrub",   "P4 - Forest-Shrub Mix"),
    ("P5_mixedshrub",    "P5 - Mixed Shrub"),
    ("P6_forestedslope", "P6 - Forested Slope"),
    ("P7_grassgrey",     "P7 - Grey Grassland"),
    ("P8_shrubgrass",    "P8 - Shrub-Grass"),
]

SCENARIOS = [
    ("ws12_wd90_dry",      "12 km/h, 90°, Dry"),
    ("ws12_wd270_dry",     "12 km/h, 270°, Dry"),
    ("ws12_wd270_extreme", "12 km/h, 270°, Extreme"),
    ("ws25_wd90_dry",      "25 km/h, 90°, Dry"),
    ("ws25_wd270_dry",     "25 km/h, 270°, Dry"),
    ("ws25_wd270_extreme", "25 km/h, 270°, Extreme"),
]

DEFAULT_CREWS = [
    {"name": "Alpha",   "row": 60,  "col": 100, "color": "#e74c3c"},
    {"name": "Bravo",   "row": 100, "col": 200, "color": "#3498db"},
    {"name": "Charlie", "row": 160, "col": 100, "color": "#2ecc71"},
    {"name": "Delta",   "row": 160, "col": 220, "color": "#f39c12"},
    {"name": "Echo",    "row": 220, "col": 140, "color": "#9b59b6"},
    {"name": "Foxtrot", "row": 260, "col": 260, "color": "#1abc9c"},
]

CREW_NAMES = [
    "Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot",
    "Golf", "Hotel", "India", "Juliet",
]
CREW_COLORS = [
    "#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6",
    "#1abc9c", "#e67e22", "#2980b9", "#c0392b", "#16a085",
]

SEVERITY_COLORS = {
    "LOW": "#27ae60",
    "MODERATE": "#f39c12",
    "HIGH": "#e74c3c",
    "CRITICAL": "#8e44ad",
}

SEVERITY_GLOW = {
    "LOW": "0 0 8px #27ae60",
    "MODERATE": "0 0 12px #f39c12",
    "HIGH": "0 0 16px #e74c3c, 0 0 32px #e74c3c44",
    "CRITICAL": "0 0 20px #8e44ad, 0 0 40px #8e44ad66, 0 0 60px #8e44ad33",
}

# ── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="WILDFIRE DSS — Crew Safety Command",
    page_icon="W",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS — Mission Control Theme ───────────────────────────────────────

st.markdown("""
<style>
    /* ─── Google Fonts ─── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;700&display=swap');

    /* ─── Hide Streamlit branding ─── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}

    /* ─── Global ─── */
    .stApp {
        background: linear-gradient(170deg, #0a0a14 0%, #0d1117 40%, #111827 100%);
        font-family: 'Inter', -apple-system, sans-serif;
    }

    /* ─── Sidebar — Tactical Panel ─── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
        border-right: 1px solid #ff6b3520;
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #FF6B35;
        font-family: 'Inter', sans-serif;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-size: 0.85rem;
        border-bottom: 1px solid #FF6B3530;
        padding-bottom: 6px;
    }
    section[data-testid="stSidebar"] hr {
        border-color: #FF6B3520;
    }

    /* ─── Main content text ─── */
    .stApp, .stMarkdown, .stMarkdown p, .stCaption {
        color: #c9d1d9;
    }

    /* ─── Custom header bar ─── */
    .header-bar {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #1a1a2e 100%);
        border: 1px solid #FF6B3530;
        border-radius: 8px;
        padding: 12px 24px;
        margin-bottom: 16px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 4px 20px rgba(255, 107, 53, 0.08), inset 0 1px 0 rgba(255,255,255,0.03);
    }
    .header-title {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        font-size: 1.5rem;
        color: #FF6B35;
        text-transform: uppercase;
        letter-spacing: 3px;
        text-shadow: 0 0 20px #FF6B3555;
    }
    .header-subtitle {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        color: #8b949e;
        letter-spacing: 2px;
        text-transform: uppercase;
    }
    .header-info {
        display: flex;
        gap: 24px;
        align-items: center;
    }
    .header-info-item {
        text-align: center;
    }
    .header-info-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.6rem;
        color: #6e7681;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .header-info-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.9rem;
        color: #e6edf3;
        font-weight: 600;
    }
    .header-status-live {
        display: inline-block;
        width: 8px; height: 8px;
        background: #27ae60;
        border-radius: 50%;
        margin-right: 6px;
        animation: pulse-green 2s infinite;
    }

    /* ─── Crew status cards ─── */
    .crew-card {
        background: linear-gradient(135deg, #161b22 0%, #1c2333 100%);
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 10px;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    .crew-card:hover {
        border-color: #58a6ff40;
        box-shadow: 0 2px 16px rgba(88, 166, 255, 0.1);
    }
    .crew-card-alert {
        border-color: #DC143C60;
        box-shadow: 0 0 20px rgba(220, 20, 60, 0.15);
        animation: pulse-border 2s infinite;
    }
    .crew-card-critical {
        border-color: #8e44ad80;
        box-shadow: 0 0 25px rgba(142, 68, 173, 0.25);
        animation: pulse-border-critical 1.5s infinite;
    }
    .crew-card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
    }
    .crew-name {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        font-size: 0.95rem;
        color: #e6edf3;
    }
    .crew-icon {
        font-size: 1rem;
        margin-right: 6px;
    }
    .status-badge {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem;
        font-weight: 700;
        padding: 3px 10px;
        border-radius: 12px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .badge-safe {
        background: #27ae6020;
        color: #27ae60;
        border: 1px solid #27ae6040;
    }
    .badge-moderate {
        background: #f39c1220;
        color: #f39c12;
        border: 1px solid #f39c1240;
    }
    .badge-high {
        background: #DC143C20;
        color: #DC143C;
        border: 1px solid #DC143C40;
        animation: blink 1.5s infinite;
    }
    .badge-critical {
        background: #8e44ad25;
        color: #c27bdb;
        border: 1px solid #8e44ad60;
        animation: blink-fast 0.8s infinite;
    }
    .crew-metrics {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 8px;
        margin: 8px 0;
    }
    .crew-metric {
        text-align: center;
        padding: 6px;
        background: #0d111740;
        border-radius: 6px;
    }
    .crew-metric-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.55rem;
        color: #6e7681;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .crew-metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.1rem;
        font-weight: 700;
    }
    .danger-bar-container {
        background: #21262d;
        border-radius: 4px;
        height: 6px;
        overflow: hidden;
        margin-top: 6px;
    }
    .danger-bar-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.5s ease;
    }

    /* ─── Alert banner ─── */
    .alert-banner {
        background: linear-gradient(135deg, #DC143C15, #DC143C08);
        border: 1px solid #DC143C40;
        border-radius: 8px;
        padding: 10px 16px;
        margin-bottom: 10px;
        text-align: center;
    }
    .alert-banner-text {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        font-size: 0.9rem;
        color: #DC143C;
        text-shadow: 0 0 10px #DC143C55;
        animation: blink 2s infinite;
    }
    .safe-banner {
        background: linear-gradient(135deg, #27ae6010, #27ae6005);
        border: 1px solid #27ae6030;
        border-radius: 8px;
        padding: 10px 16px;
        margin-bottom: 10px;
        text-align: center;
    }
    .safe-banner-text {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        font-size: 0.85rem;
        color: #27ae60;
    }

    /* ─── Section headers ─── */
    .section-header {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        font-size: 0.8rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 12px;
        padding-bottom: 6px;
        border-bottom: 1px solid #21262d;
    }

    /* ─── Deploy button ─── */
    .deploy-btn-container {
        margin-top: 12px;
    }
    .deploy-btn {
        display: block;
        width: 100%;
        padding: 12px;
        background: linear-gradient(135deg, #DC143C, #b91030);
        color: white;
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        border: 2px solid #DC143C;
        border-radius: 8px;
        text-align: center;
        cursor: pointer;
        box-shadow: 0 0 20px #DC143C40;
        transition: all 0.3s ease;
    }
    .deploy-btn:hover {
        box-shadow: 0 0 30px #DC143C66;
        transform: translateY(-1px);
    }

    /* ─── Playback controls ─── */
    .playback-bar {
        background: linear-gradient(135deg, #161b22, #1c2333);
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 10px 16px;
    }

    /* ─── Timeline footer ─── */
    .timeline-info {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        color: #8b949e;
        text-align: center;
    }
    .timeline-time {
        font-size: 1rem;
        font-weight: 700;
        color: #FF6B35;
    }

    /* ─── Footer ─── */
    .footer-bar {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.6rem;
        color: #484f58;
        text-align: center;
        letter-spacing: 1px;
        text-transform: uppercase;
        padding-top: 8px;
        border-top: 1px solid #21262d;
    }

    /* ─── Wind compass ─── */
    .wind-compass { display: flex; align-items: center; gap: 8px; }
    .compass-ring {
        width: 42px; height: 42px;
        border: 1px solid #30363d;
        border-radius: 50%;
        position: relative;
        background: radial-gradient(circle, #1a1a2e, #0d1117);
        flex-shrink: 0;
    }
    .compass-ring .cl {
        position: absolute;
        font-family: 'JetBrains Mono', monospace;
        font-size: 7px; color: #484f58;
    }
    .cl-n { top: 2px; left: 50%; transform: translateX(-50%); }
    .cl-s { bottom: 2px; left: 50%; transform: translateX(-50%); }
    .cl-e { top: 50%; right: 3px; transform: translateY(-50%); }
    .cl-w { top: 50%; left: 3px; transform: translateY(-50%); }
    .compass-wind-info { text-align: center; line-height: 1.3; }
    .compass-speed {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem; font-weight: 700; color: #e6edf3;
    }
    .compass-dir-text {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.55rem; color: #6e7681;
    }

    /* ─── Crew 2-column grid ─── */
    .crew-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 8px;
    }
    .crew-grid .crew-card { margin-bottom: 0; }
    .crew-grid .crew-metric-value { font-size: 0.95rem; }
    .crew-grid .crew-metric-label { font-size: 0.5rem; }

    /* ─── Trend arrow ─── */
    .trend-indicator {
        font-size: 0.85rem; font-weight: 700;
        margin-left: 3px; vertical-align: middle;
    }

    /* ─── Animations ─── */
    @keyframes pulse-green {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.3; }
    }
    @keyframes pulse-border {
        0%, 100% { border-color: #DC143C60; }
        50% { border-color: #DC143C20; }
    }
    @keyframes pulse-border-critical {
        0%, 100% { border-color: #8e44ad80; box-shadow: 0 0 25px rgba(142, 68, 173, 0.25); }
        50% { border-color: #8e44ad30; box-shadow: 0 0 10px rgba(142, 68, 173, 0.1); }
    }
    @keyframes blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    @keyframes blink-fast {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.3; }
    }

    /* ─── Streamlit widget overrides ─── */
    .stSlider > div > div {
        color: #c9d1d9;
    }
    div[data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace;
    }
    .stSelectbox label, .stSlider label, .stNumberInput label {
        font-family: 'Inter', sans-serif;
        font-size: 0.8rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Fix streamlit containers */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-color: #30363d;
        background: transparent;
    }
</style>
""", unsafe_allow_html=True)


# ── Helper Functions ─────────────────────────────────────────────────────────

def convert_to_time_series(arrival_time, flame_length, rate_of_spread,
                           timestep_minutes=5, downsample_factor=5):
    if downsample_factor > 1:
        H, W = arrival_time.shape
        nH = H // downsample_factor
        nW = W // downsample_factor

        arrival_time = arrival_time[:nH * downsample_factor, :nW * downsample_factor]
        arrival_time = arrival_time.reshape(nH, downsample_factor, nW, downsample_factor)
        arrival_time = np.nanmin(arrival_time, axis=(1, 3))

        flame_length = flame_length[:nH * downsample_factor, :nW * downsample_factor]
        flame_length = flame_length.reshape(nH, downsample_factor, nW, downsample_factor)
        flame_length = np.nanmax(flame_length, axis=(1, 3))

        rate_of_spread = rate_of_spread[:nH * downsample_factor, :nW * downsample_factor]
        rate_of_spread = rate_of_spread.reshape(nH, downsample_factor, nW, downsample_factor)
        rate_of_spread = np.nanmax(rate_of_spread, axis=(1, 3))

    arrival_ds = arrival_time.copy()
    flame_ds = flame_length.copy()

    valid_times = arrival_time[~np.isnan(arrival_time)]
    if len(valid_times) == 0:
        raise ValueError("No valid arrival times found in this simulation")

    max_time = int(np.ceil(valid_times.max()))
    num_timesteps = max_time // timestep_minutes + 1

    H, W = arrival_time.shape
    flame_series = np.zeros((num_timesteps, H, W), dtype=np.float32)
    spread_series = np.zeros((num_timesteps, H, W), dtype=np.float32)

    for t in range(num_timesteps):
        current_time = t * timestep_minutes
        burned = (arrival_time <= current_time) & ~np.isnan(arrival_time)
        flame_series[t][burned] = flame_length[burned]
        spread_series[t][burned] = rate_of_spread[burned]

    return flame_series, spread_series, arrival_ds, flame_ds


def compute_crew_danger(arrival_ds, flame_ds, crew_row, crew_col,
                        current_time, buffer_cells=8, prediction_horizon=10):
    H, W = arrival_ds.shape
    crew_row = max(0, min(H - 1, crew_row))
    crew_col = max(0, min(W - 1, crew_col))

    r0 = max(0, crew_row - buffer_cells)
    r1 = min(H, crew_row + buffer_cells + 1)
    c0 = max(0, crew_col - buffer_cells)
    c1 = min(W, crew_col + buffer_cells + 1)

    neigh_arrival = arrival_ds[r0:r1, c0:c1]
    neigh_flame = flame_ds[r0:r1, c0:c1]
    valid = neigh_arrival[~np.isnan(neigh_arrival)]

    if len(valid) == 0:
        return dict(probability=0.0, time_to_fire=float("inf"),
                    severity="LOW", flame_nearby=0.0)

    min_arrival = float(valid.min())
    time_to_fire = min_arrival - current_time

    arrived_mask = (neigh_arrival <= current_time) & ~np.isnan(neigh_arrival)
    flame_nearby = float(np.nanmax(neigh_flame[arrived_mask])) if arrived_mask.any() else 0.0

    if time_to_fire <= 0:
        penetration = arrived_mask.sum() / arrived_mask.size
        prob = 0.6 + 0.4 * penetration
    elif time_to_fire <= prediction_horizon:
        prob = 0.3 + 0.3 * (1.0 - time_to_fire / prediction_horizon)
    elif time_to_fire <= prediction_horizon * 3:
        frac = (time_to_fire - prediction_horizon) / (prediction_horizon * 2)
        prob = 0.05 + 0.25 * (1.0 - frac)
    else:
        prob = max(0.0, 0.05 * (1.0 - (time_to_fire - prediction_horizon * 3) / 200))

    prob = float(np.clip(prob, 0.0, 1.0))

    if prob >= 0.8:
        severity = "CRITICAL"
    elif prob >= 0.6:
        severity = "HIGH"
    elif prob >= 0.3:
        severity = "MODERATE"
    else:
        severity = "LOW"

    return dict(
        probability=prob,
        time_to_fire=max(time_to_fire, 0.0),
        severity=severity,
        flame_nearby=flame_nearby,
    )


def parse_wind_from_scenario(scenario_key):
    """Parse wind speed (km/h), direction (degrees), and moisture from scenario name."""
    import re
    m = re.match(r'ws(\d+)_wd(\d+)_(.+)', scenario_key)
    if m:
        return int(m.group(1)), int(m.group(2)), m.group(3).capitalize()
    return None, None, None


def compute_burned_stats(flame_frame):
    """Compute burned area stats from a single flame frame."""
    total_cells = flame_frame.size
    burned_cells = int((flame_frame > 0).sum())
    burned_pct = burned_cells / total_cells * 100
    max_flame = float(np.nanmax(flame_frame)) if burned_cells > 0 else 0.0
    return burned_pct, max_flame, burned_cells


def get_trend_arrow(current_prob, prev_prob):
    """Return trend arrow and color based on danger change."""
    delta = current_prob - prev_prob
    if delta > 0.1:
        return "\u2191", "#DC143C"
    elif delta > 0.03:
        return "\u2197", "#f39c12"
    elif delta > -0.03:
        return "\u2192", "#6e7681"
    elif delta > -0.1:
        return "\u2198", "#27ae60"
    else:
        return "\u2193", "#27ae60"


# ── Cached Loaders ───────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading FARSITE simulation…", max_entries=3)
def load_simulation(patch_folder, scenario):
    scenario_path = DATA_ROOT / patch_folder / "Outputs" / scenario
    parser = FARSITEParser(str(scenario_path))

    arrival_time, _ = parser.load_arrival_time()
    flame_length, _ = parser.load_flame_length()
    rate_of_spread, _ = parser.load_rate_of_spread()

    return convert_to_time_series(
        arrival_time, flame_length, rate_of_spread,
        timestep_minutes=TIMESTEP_MINUTES,
        downsample_factor=DOWNSAMPLE_FACTOR,
    )


# ── HTML Components ──────────────────────────────────────────────────────────

def render_header(patch_label, scenario_label, current_time, n_crews, n_alerts,
                  wind_speed=None, wind_dir=None, moisture=None,
                  burned_pct=0, max_flame=0):
    now = datetime.now().strftime("%H:%M:%S")
    alert_class = "color: #DC143C; text-shadow: 0 0 10px #DC143C55;" if n_alerts > 0 else "color: #27ae60;"
    status_text = f"{n_alerts} ALERT{'S' if n_alerts != 1 else ''}" if n_alerts > 0 else "ALL CLEAR"

    # Wind compass HTML
    wind_html = ""
    if wind_speed is not None and wind_dir is not None:
        arrow_deg = (wind_dir + 180) % 360
        dir_names = {0: "N", 45: "NE", 90: "E", 135: "SE", 180: "S", 225: "SW", 270: "W", 315: "NW"}
        from_label = dir_names.get(wind_dir, f"{wind_dir}\u00b0")
        moisture_color = "#DC143C" if moisture and moisture.upper() == "EXTREME" else "#8b949e"
        moisture_str = f'<div class="compass-dir-text" style="color: {moisture_color};">{moisture}</div>' if moisture else ""
        wind_html = f'<div class="header-info-item"><div class="header-info-label">Wind</div><div class="wind-compass"><div class="compass-ring"><span class="cl cl-n">N</span><span class="cl cl-s">S</span><span class="cl cl-e">E</span><span class="cl cl-w">W</span><div style="position:absolute;top:5px;left:50%;width:2px;height:16px;margin-left:-1px;background:#DC143C;transform-origin:bottom center;transform:rotate({arrow_deg}deg);border-radius:1px 1px 0 0;"></div><div style="position:absolute;top:50%;left:50%;width:5px;height:5px;background:#FF6B35;border-radius:50%;transform:translate(-50%,-50%);"></div></div><div class="compass-wind-info"><div class="compass-speed">{wind_speed} km/h</div><div class="compass-dir-text">from {from_label}</div>{moisture_str}</div></div></div>'

    burned_color = "#DC143C" if burned_pct > 20 else ("#f39c12" if burned_pct > 5 else "#27ae60")
    flame_color = "#DC143C" if max_flame > 3 else ("#f39c12" if max_flame > 1 else "#6e7681")

    st.markdown(f"""
    <div class="header-bar">
        <div>
            <div class="header-title">Wildfire DSS</div>
            <div class="header-subtitle">Crew Extraction Warning System</div>
        </div>
        <div class="header-info">
            <div class="header-info-item">
                <div class="header-info-label">Terrain</div>
                <div class="header-info-value">{patch_label}</div>
            </div>
            {wind_html}
            <div class="header-info-item">
                <div class="header-info-label">Sim Time</div>
                <div class="header-info-value" style="color: #FF6B35;">{current_time} min</div>
            </div>
            <div class="header-info-item">
                <div class="header-info-label">Burned</div>
                <div class="header-info-value" style="color: {burned_color};">{burned_pct:.1f}%</div>
            </div>
            <div class="header-info-item">
                <div class="header-info-label">Max Flame</div>
                <div class="header-info-value" style="color: {flame_color};">{max_flame:.1f}m</div>
            </div>
            <div class="header-info-item">
                <div class="header-info-label">Crews</div>
                <div class="header-info-value">{n_crews}</div>
            </div>
            <div class="header-info-item">
                <div class="header-info-label">Status</div>
                <div class="header-info-value" style="{alert_class}">
                    <span class="header-status-live"></span>{status_text}
                </div>
            </div>
            <div class="header-info-item">
                <div class="header-info-label">Clock</div>
                <div class="header-info-value">{now}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_crew_card_html(crew, danger, threshold, trend=None):
    """Build crew card HTML string. Returns the HTML (does not render directly)."""
    prob = danger["probability"]
    sev = danger["severity"]
    is_danger = prob > threshold
    ttf = danger["time_to_fire"]
    flame = danger["flame_nearby"]

    # Badge styling
    badge_map = {
        "LOW": ("badge-safe", "SAFE"),
        "MODERATE": ("badge-moderate", "WATCH"),
        "HIGH": ("badge-high", "ALERT"),
        "CRITICAL": ("badge-critical", "CRITICAL"),
    }
    badge_class, badge_text = badge_map[sev]
    if not is_danger and sev in ("HIGH", "CRITICAL"):
        badge_class, badge_text = "badge-moderate", "WATCH"

    card_class = "crew-card"
    if sev == "CRITICAL" and is_danger:
        card_class += " crew-card-critical"
    elif is_danger:
        card_class += " crew-card-alert"

    # Color for probability
    prob_color = SEVERITY_COLORS[sev]
    bar_gradient = f"linear-gradient(90deg, {prob_color}, {prob_color}88)"
    bar_width = max(prob * 100, 1)

    # ETA display
    if ttf <= 0:
        eta_str = "NOW"
        eta_color = "#DC143C"
    elif ttf < 9999:
        eta_str = f"{ttf:.0f}m"
        eta_color = "#f39c12" if ttf < 10 else "#c9d1d9"
    else:
        eta_str = "\u2014"
        eta_color = "#6e7681"

    # Flame display
    flame_str = f"{flame:.1f}m" if flame > 0 else "\u2014"
    flame_color = "#DC143C" if flame > 3 else ("#f39c12" if flame > 1 else "#6e7681")

    crew_icon = "\u26a0\ufe0f" if is_danger else "\U0001f464"

    # Trend arrow
    trend_html = ""
    if trend:
        arrow, arrow_color = trend
        trend_html = f'<span class="trend-indicator" style="color: {arrow_color};">{arrow}</span>'

    return f"""
    <div class="{card_class}">
        <div class="crew-card-header">
            <span class="crew-name">
                <span class="crew-icon">{crew_icon}</span>
                {crew['name']}{trend_html}
            </span>
            <span class="status-badge {badge_class}">{badge_text}</span>
        </div>
        <div class="crew-metrics">
            <div class="crew-metric">
                <div class="crew-metric-label">Danger</div>
                <div class="crew-metric-value" style="color: {prob_color}; text-shadow: {SEVERITY_GLOW[sev]};">{prob:.0%}</div>
            </div>
            <div class="crew-metric">
                <div class="crew-metric-label">Fire ETA</div>
                <div class="crew-metric-value" style="color: {eta_color};">{eta_str}</div>
            </div>
            <div class="crew-metric">
                <div class="crew-metric-label">Flame</div>
                <div class="crew-metric-value" style="color: {flame_color};">{flame_str}</div>
            </div>
            <div class="crew-metric">
                <div class="crew-metric-label">Position</div>
                <div class="crew-metric-value" style="color: #6e7681; font-size: 0.7rem;">({crew['row']},{crew['col']})</div>
            </div>
        </div>
        <div class="danger-bar-container">
            <div class="danger-bar-fill" style="width: {bar_width}%; background: {bar_gradient};"></div>
        </div>
    </div>
    """


# ── Plotly Visualisations ────────────────────────────────────────────────────

def create_fire_heatmap(flame_frame, crews, dangers, threshold, arrival_ds=None):
    fig = go.Figure()

    # Terrain underlay: show burnable area with subtle green tint
    display_data = flame_frame.copy()
    max_fl = max(float(np.nanmax(flame_frame)), 1.0)
    if arrival_ds is not None:
        non_burnable = np.isnan(arrival_ds) & (flame_frame == 0)
        display_data[non_burnable] = np.nan
        unburned_burnable = (flame_frame == 0) & ~np.isnan(arrival_ds)
        if unburned_burnable.any():
            at_vals = arrival_ds.copy()
            at_valid = ~np.isnan(at_vals)
            if at_valid.any():
                at_min, at_max = float(np.nanmin(at_vals)), float(np.nanmax(at_vals))
                if at_max > at_min:
                    at_vals[at_valid] = (at_vals[at_valid] - at_min) / (at_max - at_min)
                else:
                    at_vals[at_valid] = 0.5
            display_data[unburned_burnable] = (0.001 + 0.003 * at_vals[unburned_burnable]) * max_fl

    fire_cs = [
        [0.0,   "rgba(0,0,0,0)"],
        [0.0005, "#0b1710"],
        [0.002, "#0e1a12"],
        [0.004, "#121f16"],
        [0.006, "#1a0a00"],
        [0.02,  "#3d1503"],
        [0.08,  "#6b2006"],
        [0.20,  "#b7410e"],
        [0.40,  "#e65100"],
        [0.60,  "#ff6d00"],
        [0.80,  "#ff9100"],
        [1.0,   "#ffcc02"],
    ]

    # Fire heatmap (with terrain underlay baked in)
    fig.add_trace(go.Heatmap(
        z=np.flipud(display_data),
        colorscale=fire_cs,
        colorbar=dict(
            title=dict(text="Flame (m)", font=dict(color="#8b949e", size=11)),
            len=0.6,
            thickness=12,
            tickfont=dict(color="#6e7681", size=10),
            bgcolor="rgba(0,0,0,0)",
            bordercolor="#30363d",
            borderwidth=1,
        ),
        zmin=0,
        zmax=max_fl,
        hovertemplate="Row: %{y}<br>Col: %{x}<br>Flame: %{z:.1f} m<extra></extra>",
    ))

    H = flame_frame.shape[0]
    for i, crew in enumerate(crews):
        row, col = crew["row"], crew["col"]
        d = dangers[i]
        prob = d["probability"]
        sev = d["severity"]
        is_danger = prob > threshold
        color = SEVERITY_COLORS[sev]
        symbol = "x" if is_danger else "circle"
        ttf = d["time_to_fire"]
        ttf_str = f"{ttf:.0f} min" if ttf < 9999 else "No threat"

        # Outer glow ring for danger
        if is_danger:
            fig.add_trace(go.Scatter(
                x=[col], y=[H - row],
                mode="markers",
                marker=dict(size=30, color=color, opacity=0.2, symbol="circle"),
                showlegend=False, hoverinfo="skip",
            ))

        fig.add_trace(go.Scatter(
            x=[col], y=[H - row],
            mode="markers+text",
            marker=dict(size=14, color=color, symbol=symbol,
                        line=dict(width=2, color="white")),
            text=[crew["name"]],
            textposition="top center",
            textfont=dict(color="white", size=10, family="Inter"),
            name=f"{crew['name']}: {prob:.0%} ({sev})",
            hovertemplate=(
                f"<b>{crew['name']}</b><br>"
                f"Grid: ({row}, {col})<br>"
                f"Danger: {prob:.0%} ({sev})<br>"
                f"Fire ETA: {ttf_str}<br>"
                f"Flame nearby: {d['flame_nearby']:.1f} m"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        height=540,
        xaxis=dict(
            title=dict(text="Column", font=dict(color="#6e7681", size=11)),
            constrain="domain",
            gridcolor="#21262d",
            tickfont=dict(color="#484f58", size=9),
            showline=True, linecolor="#30363d",
        ),
        yaxis=dict(
            title=dict(text="Row", font=dict(color="#6e7681", size=11)),
            scaleanchor="x", constrain="domain",
            gridcolor="#21262d",
            tickfont=dict(color="#484f58", size=9),
            showline=True, linecolor="#30363d",
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1,
            font=dict(size=9, color="#8b949e", family="Inter"),
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=50, r=20, t=30, b=50),
        plot_bgcolor="#0d1117",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c9d1d9"),
    )
    return fig


def create_danger_timeline(timeline_data, crews, threshold, current_t):
    fig = go.Figure()
    times = [t * TIMESTEP_MINUTES for t in timeline_data["timesteps"]]

    for i, crew in enumerate(crews):
        probs = [timeline_data["probabilities"][j][i] for j in range(len(times))]
        fig.add_trace(go.Scatter(
            x=times, y=probs, mode="lines",
            name=crew["name"],
            line=dict(color=crew.get("color", "#3498db"), width=2),
        ))

    fig.add_hline(y=threshold, line_dash="dash", line_color="#DC143C",
                  annotation_text=f"Threshold ({threshold:.0%})",
                  annotation_font_color="#DC143C")
    fig.add_vline(x=current_t * TIMESTEP_MINUTES, line_dash="dot",
                  line_color="#FF6B35",
                  annotation_text="Now", annotation_font_color="#FF6B35")

    fig.update_layout(
        height=200,
        xaxis=dict(
            title=dict(text="Time (min)", font=dict(color="#6e7681", size=10)),
            gridcolor="#21262d",
            tickfont=dict(color="#484f58", size=9),
        ),
        yaxis=dict(
            title=dict(text="Danger", font=dict(color="#6e7681", size=10)),
            range=[0, 1.05],
            gridcolor="#21262d",
            tickfont=dict(color="#484f58", size=9),
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            font=dict(size=9, color="#8b949e"),
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=50, r=20, t=30, b=40),
        plot_bgcolor="#0d1117",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c9d1d9"),
    )
    return fig


# ── Session State ────────────────────────────────────────────────────────────

def init_state():
    defaults = {
        "current_t": 0,
        "is_playing": False,
        "danger_cache": {},
        "timeline_cache": {},
        "prev_patch": None,
        "prev_scenario": None,
        "crews": [dict(c) for c in DEFAULT_CREWS],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    init_state()

    # ── SIDEBAR — Tactical Control Panel ──────────────────────────────
    st.sidebar.markdown("""
    <div style="text-align:center; padding: 8px 0 16px 0;">
        <span style="font-size: 2rem; color: #FF6B35;">W</span><br>
        <span style="font-family: 'Inter', sans-serif; font-weight: 800; font-size: 0.9rem;
              color: #FF6B35; letter-spacing: 3px; text-transform: uppercase;">
            CONTROL PANEL
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.header("📍 Mission Parameters")

    patch_idx = st.sidebar.selectbox(
        "Landscape Patch",
        range(len(PATCHES)),
        format_func=lambda i: PATCHES[i][1],
    )
    patch_folder = PATCHES[patch_idx][0]

    scenario_idx = st.sidebar.selectbox(
        "Weather Scenario",
        range(len(SCENARIOS)),
        format_func=lambda i: SCENARIOS[i][1],
    )
    scenario = SCENARIOS[scenario_idx][0]

    # Reset caches when simulation changes
    if (st.session_state.prev_patch != patch_folder
            or st.session_state.prev_scenario != scenario):
        st.session_state.danger_cache = {}
        st.session_state.timeline_cache = {}
        st.session_state.current_t = 0
        st.session_state.is_playing = False
        st.session_state.prev_patch = patch_folder
        st.session_state.prev_scenario = scenario

    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Threat Analysis")

    buffer_m = st.sidebar.slider(
        "Safety Buffer", 100, 500, 200, 25,
        help="Buffer radius around crew (meters)", format="%d m")
    prediction_horizon = st.sidebar.slider(
        "Prediction Horizon", 5, 30, 10, 5,
        help="Look-ahead window (minutes)", format="%d min")
    threshold = st.sidebar.slider(
        "Alert Threshold", 0.0, 1.0, 0.5, 0.05,
        help="Probability above which a crew receives a danger alert")

    buffer_cells = int(buffer_m / DS_RESOLUTION)

    st.sidebar.markdown("---")

    # ── Crew position editor ─────────────────────────────────────────
    with st.sidebar.expander("👥 Crew Positions", expanded=False):
        num_crews = st.number_input(
            "Number of crews", 1, 10, len(st.session_state.crews))

        while len(st.session_state.crews) < num_crews:
            idx = len(st.session_state.crews)
            st.session_state.crews.append({
                "name": CREW_NAMES[idx % len(CREW_NAMES)],
                "row": 160, "col": 160,
                "color": CREW_COLORS[idx % len(CREW_COLORS)],
            })
        st.session_state.crews = st.session_state.crews[:num_crews]

        for i, crew in enumerate(st.session_state.crews):
            c1, c2 = st.columns(2)
            with c1:
                crew["row"] = st.number_input(
                    f"{crew['name']} Row", 0, 319, crew["row"], key=f"cr_{i}")
            with c2:
                crew["col"] = st.number_input(
                    f"{crew['name']} Col", 0, 319, crew["col"], key=f"cc_{i}")

        if st.button("Reset to Defaults"):
            st.session_state.crews = [dict(c) for c in DEFAULT_CREWS]
            st.session_state.danger_cache = {}
            st.session_state.timeline_cache = {}
            st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.header("▶️ Playback")

    speed = st.sidebar.select_slider(
        "Speed", options=[0.5, 1.0, 2.0, 4.0], value=1.0,
        format_func=lambda x: f"{x}×")

    btn_cols = st.sidebar.columns(3)
    with btn_cols[0]:
        if st.button("⏪", use_container_width=True, help="Step back"):
            st.session_state.current_t = max(0, st.session_state.current_t - 1)
            st.session_state.is_playing = False
    with btn_cols[1]:
        play_label = "⏸" if st.session_state.is_playing else "▶"
        if st.button(play_label, use_container_width=True, type="primary"):
            st.session_state.is_playing = not st.session_state.is_playing
    with btn_cols[2]:
        step_fwd = st.button("⏩", use_container_width=True, help="Step forward")

    # ── LOAD DATA ────────────────────────────────────────────────────────
    try:
        flame_series, spread_series, arrival_ds, flame_ds = load_simulation(
            patch_folder, scenario)
    except Exception as e:
        st.error(f"Failed to load simulation: {e}")
        return

    max_t = flame_series.shape[0] - 1
    if st.session_state.current_t > max_t:
        st.session_state.current_t = max_t
    if step_fwd:
        st.session_state.current_t = min(max_t, st.session_state.current_t + 1)
        st.session_state.is_playing = False

    # ── COMPUTE DANGER ───────────────────────────────────────────────────
    t = st.session_state.current_t
    current_time = t * TIMESTEP_MINUTES
    crews = st.session_state.crews

    crew_key = "|".join(f"{c['row']},{c['col']}" for c in crews)
    cache_key = f"{patch_folder}_{scenario}_{t}_{buffer_cells}_{prediction_horizon}_{crew_key}"

    if cache_key not in st.session_state.danger_cache:
        dangers = [
            compute_crew_danger(arrival_ds, flame_ds, c["row"], c["col"],
                                current_time, buffer_cells, prediction_horizon)
            for c in crews
        ]
        st.session_state.danger_cache[cache_key] = dangers

    dangers = st.session_state.danger_cache[cache_key]

    # ── COMPUTE TIMELINE ─────────────────────────────────────────────────
    tl_key = f"{patch_folder}_{scenario}_{buffer_cells}_{prediction_horizon}_{crew_key}"

    if tl_key not in st.session_state.timeline_cache:
        timesteps = list(range(0, max_t + 1))
        all_probs = []
        for ts in timesteps:
            ct = ts * TIMESTEP_MINUTES
            row_probs = [
                compute_crew_danger(arrival_ds, flame_ds, c["row"], c["col"],
                                    ct, buffer_cells, prediction_horizon)["probability"]
                for c in crews
            ]
            all_probs.append(row_probs)
        st.session_state.timeline_cache[tl_key] = {
            "timesteps": timesteps,
            "probabilities": all_probs,
        }

    timeline_data = st.session_state.timeline_cache[tl_key]
    n_alerts = sum(1 for d in dangers if d["probability"] > threshold)

    # ── WIND & BURNED STATS ──────────────────────────────────────────────
    wind_speed, wind_dir, moisture = parse_wind_from_scenario(scenario)
    burned_pct, max_flame_val, _ = compute_burned_stats(flame_series[t])

    # ── HEADER BAR ───────────────────────────────────────────────────────
    render_header(
        PATCHES[patch_idx][1], SCENARIOS[scenario_idx][1],
        current_time, len(crews), n_alerts,
        wind_speed=wind_speed, wind_dir=wind_dir, moisture=moisture,
        burned_pct=burned_pct, max_flame=max_flame_val,
    )

    # ── MAIN CONTENT ─────────────────────────────────────────────────────
    col_map, col_status = st.columns([3, 1])

    with col_map:
        st.markdown('<div class="section-header">📡 Fire Progression Map</div>',
                    unsafe_allow_html=True)

        fig = create_fire_heatmap(flame_series[t], crews, dangers, threshold, arrival_ds)
        st.plotly_chart(fig, use_container_width=True, key="heatmap")

        st.markdown('<div class="section-header">Danger Timeline</div>',
                    unsafe_allow_html=True)
        tl_fig = create_danger_timeline(timeline_data, crews, threshold, t)
        st.plotly_chart(tl_fig, use_container_width=True, key="timeline")

    with col_status:
        st.markdown('<div class="section-header">👥 Crew Monitoring</div>',
                    unsafe_allow_html=True)

        # Alert / safe banner
        if n_alerts > 0:
            st.markdown(f"""
            <div class="alert-banner">
                <div class="alert-banner-text">
                    {n_alerts} ACTIVE ALERT{'S' if n_alerts > 1 else ''}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="safe-banner">
                <div class="safe-banner-text">All crews safe</div>
            </div>
            """, unsafe_allow_html=True)

        # Compute trends (compare to 3 steps back)
        trends = []
        for i in range(len(crews)):
            prev_t_idx = max(0, t - 3)
            prev_prob = timeline_data["probabilities"][prev_t_idx][i]
            trends.append(get_trend_arrow(dangers[i]["probability"], prev_prob))

        # Sort crews by danger (highest first)
        crew_data = list(zip(crews, dangers, trends))
        crew_data.sort(key=lambda x: x[1]["probability"], reverse=True)

        # Crew cards (2-column grid)
        grid_cols = st.columns(2)
        for idx, (crew_item, danger_item, trend_item) in enumerate(crew_data):
            with grid_cols[idx % 2]:
                st.markdown(
                    render_crew_card_html(crew_item, danger_item, threshold, trend_item),
                    unsafe_allow_html=True,
                )

        # Deploy Extraction button
        if n_alerts > 0:
            st.markdown("""
            <div class="deploy-btn-container">
                <div class="deploy-btn">🚁 Deploy Extraction</div>
            </div>
            """, unsafe_allow_html=True)

    # ── TIMELINE SLIDER ──────────────────────────────────────────────────
    st.markdown("---")
    tc1, tc2 = st.columns([5, 1])

    with tc1:
        st.session_state.timeline_slider = t

        def _on_timeline_change():
            st.session_state.current_t = st.session_state.timeline_slider
            st.session_state.is_playing = False

        st.slider("Timeline", 0, max_t, format="Step %d",
                  key="timeline_slider", on_change=_on_timeline_change)

    with tc2:
        st.markdown(f"""
        <div style="text-align: center; padding-top: 8px;">
            <div class="timeline-time">{current_time} min</div>
            <div class="timeline-info">{t} / {max_t} steps</div>
        </div>
        """, unsafe_allow_html=True)

    # ── FOOTER ───────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="footer-bar">
        Wildfire DSS — Crew Extraction Warning System · Bachelor's Thesis ·
        FARSITE-based Decision Support · {len(crews)} crews monitored ·
        Buffer: {buffer_m}m · Horizon: {prediction_horizon}min
    </div>
    """, unsafe_allow_html=True)

    # ── AUTO-PLAY ────────────────────────────────────────────────────────
    if st.session_state.is_playing:
        if st.session_state.current_t < max_t:
            time.sleep(1.0 / speed)
            st.session_state.current_t += 1
            st.rerun()
        else:
            st.session_state.is_playing = False


if __name__ == "__main__":
    main()
