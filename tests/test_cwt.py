import numpy as np
import pandas as pd
from pathlib import Path
import pywt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import find_peaks

from src.processing.extract_golden_standard import extract_golden_standard
from src.processing.newbee_surface_extraction import extract_surface_segments

# --- 1. Load Data ---
accel_cols = ['acceleration_Pelvis_x', 'acceleration_Pelvis_y', 'acceleration_Pelvis_z']
xsens_file = Path(__file__).resolve().parent.parent / "data" / "newbee" / "courseA" / "id03" / "xsens.csv"
labels_file = Path(__file__).resolve().parent.parent / "data" / "newbee" / "courseA" / "id03" / "labels.csv"

start_end_times = extract_surface_segments(str(labels_file), target_mode="stairs_down")
df = pd.read_csv(xsens_file, usecols=['time'] + accel_cols)

time_s = df['time'].to_numpy() / 1000.0
fs = round(1 / np.mean(np.diff(time_s)))
acc_mag = np.linalg.norm(df[accel_cols].to_numpy(), axis=1)
acc_z = df[accel_cols[2]].to_numpy()

y_HS, y_FO = extract_golden_standard(pd.read_csv(labels_file))
hs_times_s = y_HS[:, 0] / 1000.0
fo_times_s = y_FO[:, 0] / 1000.0


# --- 2. CWT, Timing Extraction & Metrics Functions ---
def compute_cwt(signal, fs, step_freq, wavelet='mexh'):
    fc = pywt.central_frequency(wavelet)
    scale = (fc * fs) / step_freq
    coefs, _ = pywt.cwt(signal, scales=[scale], wavelet=wavelet)
    return coefs[0], scale

def predict_step_timing_from_cwt(cwt_output, time_array, min_distance=None):
    peaks, _ = find_peaks(cwt_output, distance=min_distance)
    return peaks, time_array[peaks], cwt_output[peaks]

def predict_foot_off_from_secondary_cwt(cwt1_output, time_array, hs_peak_indices, fs, step_freq):
    cwt2_output, _ = compute_cwt(cwt1_output, fs, step_freq=step_freq, wavelet='mexh')
    fo_indices = []
    for i in range(len(hs_peak_indices) - 1):
        idx_start = hs_peak_indices[i]
        idx_end = hs_peak_indices[i + 1]
        if idx_end > idx_start + 1:
            segment_min_idx = idx_start + np.argmin(cwt2_output[idx_start:idx_end])
            fo_indices.append(segment_min_idx)
    fo_indices = np.array(fo_indices, dtype=int)
    if len(fo_indices) == 0:
        return np.array([]), np.array([]), cwt2_output
    return time_array[fo_indices], cwt2_output[fo_indices], cwt2_output

def calculate_metrics(true_times, pred_times, tol=0.3):
    """Calculates F1, Mean Error, and Std Error with strictly 1-to-1 matching."""
    if len(true_times) == 0 and len(pred_times) == 0:
        return 1.0, np.nan, np.nan
    if len(true_times) == 0 or len(pred_times) == 0:
        return 0.0, np.nan, np.nan
        
    pairs = []
    for i, t in enumerate(true_times):
        for j, p in enumerate(pred_times):
            dist = abs(t - p)
            if dist <= tol:
                pairs.append((dist, i, j))
    
    pairs.sort(key=lambda x: x[0])
    matched_true, matched_pred, errors = set(), set(), []
    
    for dist, i, j in pairs:
        if i not in matched_true and j not in matched_pred:
            matched_true.add(i)
            matched_pred.add(j)
            errors.append(pred_times[j] - true_times[i])
            
    tp = len(matched_true)
    fp = len(pred_times) - tp
    fn = len(true_times) - tp
    
    f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    mean_err = np.mean(errors) * 1000 if errors else np.nan
    std_err = np.std(errors) * 1000 if errors else np.nan
        
    return f1, mean_err, std_err

def create_vline_coords(x_points, y_min, y_max):
    if len(x_points) == 0:
        return np.array([]), np.array([])
    xs = np.column_stack([x_points, x_points, np.full_like(x_points, np.nan)]).ravel()
    ys = np.tile([y_min, y_max, np.nan], len(x_points))
    return xs, ys


# --- 3. Precompute across Target Frequencies ---
frequencies = np.round(np.arange(1.0, 5.1, 0.1), 1)

cwt_mag_by_freq, cwt_z_by_freq = [], []
cwt2_mag_by_freq, cwt2_z_by_freq = [], []
wavelet_t_by_freq, wavelet_y_by_freq = [], []

hs_pred_mag_x, hs_pred_mag_y = [], []
fo_pred_mag_x, fo_pred_mag_y = [], []
hs_pred_z_x, hs_pred_z_y = [], []
fo_pred_z_x, fo_pred_z_y = [], []

freq_tables = []
scopes = ["All Segments"] + [f"Seg {j+1}" for j in range(len(start_end_times))]
headers = ['Scope', 'HS F1 (Mag)', 'HS Err (Mag)', 'FO F1 (Mag)', 'FO Err (Mag)', 'HS F1 (Z)', 'HS Err (Z)', 'FO F1 (Z)', 'FO Err (Z)']
wavelet_time_window = np.linspace(-2.5, 2.5, 1000)

for freq in frequencies:
    c_mag, scale = compute_cwt(acc_mag, fs, freq)
    c_z, _ = compute_cwt(acc_z, fs, freq)
    
    cwt_mag_by_freq.append(c_mag)
    cwt_z_by_freq.append(c_z)
    
    min_dist = max(1, int(0.5 * fs / freq))

    hs_idx_mag, hs_t_mag, hs_val_mag = predict_step_timing_from_cwt(c_mag, time_s, min_distance=min_dist)
    hs_idx_z, hs_t_z, hs_val_z = predict_step_timing_from_cwt(c_z, time_s, min_distance=min_dist)
    hs_pred_mag_x.append(hs_t_mag); hs_pred_mag_y.append(hs_val_mag)
    hs_pred_z_x.append(hs_t_z); hs_pred_z_y.append(hs_val_z)

    fo_t_mag, fo_val_mag, cwt2_mag = predict_foot_off_from_secondary_cwt(c_mag, time_s, hs_idx_mag, fs, freq)
    fo_t_z, fo_val_z, cwt2_z = predict_foot_off_from_secondary_cwt(c_z, time_s, hs_idx_z, fs, freq)
    cwt2_mag_by_freq.append(cwt2_mag)
    cwt2_z_by_freq.append(cwt2_z)
    fo_pred_mag_x.append(fo_t_mag); fo_pred_mag_y.append(fo_val_mag)
    fo_pred_z_x.append(fo_t_z); fo_pred_z_y.append(fo_val_z)

    s_time = scale / fs
    tau = wavelet_time_window / s_time
    norm_const = 2.0 / (np.sqrt(3.0) * (np.pi ** 0.25) * np.sqrt(s_time))
    psi = norm_const * (1.0 - tau**2) * np.exp(-0.5 * (tau**2))
    wavelet_t_by_freq.append(wavelet_time_window)
    wavelet_y_by_freq.append(psi)

    # Calculate Table Metrics for this Frequency
    cols = [[] for _ in range(9)]
    for j, scope in enumerate(scopes):
        cols[0].append(scope)
        if j == 0:
            t_start, t_end = 0, np.inf
        else:
            t_start = start_end_times["start_time"].iloc[j-1] / 1000.0
            t_end = start_end_times["end_time"].iloc[j-1] / 1000.0
            
        t_hs = hs_times_s[(hs_times_s >= t_start) & (hs_times_s <= t_end)]
        t_fo = fo_times_s[(fo_times_s >= t_start) & (fo_times_s <= t_end)]
        p_hs_m = hs_t_mag[(hs_t_mag >= t_start) & (hs_t_mag <= t_end)]
        p_fo_m = fo_t_mag[(fo_t_mag >= t_start) & (fo_t_mag <= t_end)]
        p_hs_z = hs_t_z[(hs_t_z >= t_start) & (hs_t_z <= t_end)]
        p_fo_z = fo_t_z[(fo_t_z >= t_start) & (fo_t_z <= t_end)]
        
        metrics = [
            calculate_metrics(t_hs, p_hs_m), calculate_metrics(t_fo, p_fo_m),
            calculate_metrics(t_hs, p_hs_z), calculate_metrics(t_fo, p_fo_z)
        ]
        
        for col_idx, (f1, m_err, s_err) in enumerate(metrics):
            cols[1 + col_idx*2].append(f"{f1:.2f}")
            cols[2 + col_idx*2].append(f"{m_err:.0f} ± {s_err:.0f} ms" if not np.isnan(m_err) else "-")
            
    freq_tables.append(cols)

default_idx = int(np.where(frequencies == 1.5)[0][0])


# --- 4. Subplot Setup ---
fig = make_subplots(
    rows=6, cols=1,
    shared_xaxes=False, vertical_spacing=0.04,
    row_heights=[0.14, 0.14, 0.14, 0.14, 0.14, 0.30],
    specs=[[{"type": "xy"}], [{"type": "xy"}], [{"type": "xy"}], [{"type": "xy"}], [{"type": "xy"}], [{"type": "table"}]],
    subplot_titles=(
        'Acceleration Magnitude', 'CWT Magnitude (HS: Peaks | FO: 2nd CWT Minima)',
        'Pelvis Z Acceleration', 'CWT Pelvis Z (HS: Peaks | FO: 2nd CWT Minima)',
        'Mexican Hat Wavelet Form', 'Prediction Statistics (300ms Tolerance)'
    )
)

fig.add_trace(go.Scatter(x=time_s, y=acc_mag, mode='lines', name='Acc Mag', line=dict(color='#1f77b4')), row=1, col=1)
fig.add_trace(go.Scatter(x=time_s, y=cwt_mag_by_freq[default_idx], mode='lines', name='CWT Acc Mag', line=dict(color='#ff7f0e')), row=2, col=1)
fig.add_trace(go.Scatter(x=hs_pred_mag_x[default_idx], y=hs_pred_mag_y[default_idx], mode='markers', name='Pred HS (Mag)', marker=dict(symbol='triangle-up', size=8, color='#d62728'), legendgroup='pred_hs'), row=2, col=1)
fig.add_trace(go.Scatter(x=time_s, y=cwt2_mag_by_freq[default_idx], mode='lines', name='2nd CWT Acc Mag', line=dict(color='#9467bd')), row=2, col=1)
fig.add_trace(go.Scatter(x=fo_pred_mag_x[default_idx], y=fo_pred_mag_y[default_idx], mode='markers', name='Pred FO (Mag)', marker=dict(symbol='triangle-down', size=8, color='#9467bd'), legendgroup='pred_fo'), row=2, col=1)
fig.add_trace(go.Scatter(x=time_s, y=acc_z, mode='lines', name='Acc Pelvis Z', line=dict(color='#2ca02c')), row=3, col=1)
fig.add_trace(go.Scatter(x=time_s, y=cwt_z_by_freq[default_idx], mode='lines', name='CWT Pelvis Z', line=dict(color='#17becf')), row=4, col=1)
fig.add_trace(go.Scatter(x=hs_pred_z_x[default_idx], y=hs_pred_z_y[default_idx], mode='markers', name='Pred HS (Z)', marker=dict(symbol='triangle-up', size=8, color='#d62728'), legendgroup='pred_hs', showlegend=False), row=4, col=1)
fig.add_trace(go.Scatter(x=time_s, y=cwt2_z_by_freq[default_idx], mode='lines', name='2nd CWT Pelvis Z', line=dict(color='#9467bd')), row=4, col=1)
fig.add_trace(go.Scatter(x=fo_pred_z_x[default_idx], y=fo_pred_z_y[default_idx], mode='markers', name='Pred FO (Z)', marker=dict(symbol='triangle-down', size=8, color='#9467bd'), legendgroup='pred_fo', showlegend=False), row=4, col=1)
fig.add_trace(go.Scatter(x=wavelet_t_by_freq[default_idx], y=wavelet_y_by_freq[default_idx], mode='lines', name='Scaled mexh', line=dict(color='#333333', width=2)), row=5, col=1)

# Add Golden Standards
all_mag_cwt = np.concatenate(cwt_mag_by_freq)
all_z_cwt = np.concatenate(cwt_z_by_freq)
y_bounds = [
    (float(np.min(acc_mag)), float(np.max(acc_mag))),
    (float(np.min(np.concatenate([all_mag_cwt, np.concatenate(cwt2_mag_by_freq)]))), float(np.max(np.concatenate([all_mag_cwt, np.concatenate(cwt2_mag_by_freq)])))),
    (float(np.min(acc_z)), float(np.max(acc_z))),
    (float(np.min(np.concatenate([all_z_cwt, np.concatenate(cwt2_z_by_freq)]))), float(np.max(np.concatenate([all_z_cwt, np.concatenate(cwt2_z_by_freq)])))),
]

for row_idx, (ymin, ymax) in enumerate(y_bounds, start=1):
    vl_x_hs, vl_y_hs = create_vline_coords(hs_times_s, ymin, ymax)
    fig.add_trace(go.Scatter(x=vl_x_hs, y=vl_y_hs, mode='lines', line=dict(color='green', width=1, dash='dash'), opacity=0.45, name='True HS', legendgroup='true_hs', showlegend=(row_idx == 1), hoverinfo='skip'), row=row_idx, col=1)
    
    vl_x_fo, vl_y_fo = create_vline_coords(fo_times_s, ymin, ymax)
    fig.add_trace(go.Scatter(x=vl_x_fo, y=vl_y_fo, mode='lines', line=dict(color='magenta', width=1, dash='dot'), opacity=0.45, name='True FO', legendgroup='true_fo', showlegend=(row_idx == 1), hoverinfo='skip'), row=row_idx, col=1)

# Add Statistical Table Trace
fig.add_trace(go.Table(
    header=dict(values=headers, font=dict(size=11, color='white'), align="center", fill_color='#1f77b4'),
    cells=dict(values=freq_tables[default_idx], align="center", font=dict(size=11))
), row=6, col=1)
table_idx = len(fig.data) - 1

# Controls
dropdown_buttons = []
for i in range(len(start_end_times)):
    t_start = float(start_end_times["start_time"].iloc[i] / 1000.0)
    t_end = float(start_end_times["end_time"].iloc[i] / 1000.0)
    dropdown_buttons.append(dict(
        method="relayout",
        args=[{"xaxis.range": [t_start, t_end], "xaxis2.range": [t_start, t_end], "xaxis3.range": [t_start, t_end], "xaxis4.range": [t_start, t_end]}],
        label=f"Seg {i+1} ({t_start:.1f}s - {t_end:.1f}s)"
    ))

slider_steps = []
for i, freq in enumerate(frequencies):
    step = dict(
        method="restyle",
        args=[{
            "x": [time_s, hs_pred_mag_x[i], time_s, fo_pred_mag_x[i], time_s, hs_pred_z_x[i], time_s, fo_pred_z_x[i], wavelet_t_by_freq[i], None],
            "y": [cwt_mag_by_freq[i], hs_pred_mag_y[i], cwt2_mag_by_freq[i], fo_pred_mag_y[i], cwt_z_by_freq[i], hs_pred_z_y[i], cwt2_z_by_freq[i], fo_pred_z_y[i], wavelet_y_by_freq[i], None],
            "cells.values": [None]*9 + [freq_tables[i]]
        }, [1, 2, 3, 4, 6, 7, 8, 9, 10, table_idx]],
        label=f"{freq:.1f} Hz"
    )
    slider_steps.append(step)

init_x_min = float(start_end_times["start_time"].iloc[0] / 1000.0)
init_x_max = float(start_end_times["end_time"].iloc[0] / 1000.0)

fig.update_layout(
    title="Interactive CWT Gait Predictions & Statistics",
    height=1400, hovermode="x unified", showlegend=True,
    xaxis=dict(range=[init_x_min, init_x_max], matches='x4', showticklabels=False),
    xaxis2=dict(matches='x4', showticklabels=False), xaxis3=dict(matches='x4', showticklabels=False),
    xaxis4=dict(title='Gait Time (seconds)'), xaxis5=dict(title='Wavelet Time (seconds)', range=[-1.5, 1.5]),
    yaxis=dict(title='Acc (m/s²)'), yaxis2=dict(title='CWT Mag'),
    yaxis3=dict(title='Pelvis Z (m/s²)'), yaxis4=dict(title='CWT Z'), yaxis5=dict(title='Wavelet Amp'),
    sliders=[dict(active=default_idx, currentvalue={"prefix": "Wavelet Frequency: "}, pad={"t": 55}, steps=slider_steps)],
    updatemenus=[dict(type="dropdown", direction="down", x=1.0, y=1.08, showactive=True, buttons=dropdown_buttons)]
)

fig.show()