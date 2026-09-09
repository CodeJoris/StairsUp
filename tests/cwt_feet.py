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
rightfoot_cols = ['acceleration_RightFoot_x', 'acceleration_RightFoot_y', 'acceleration_RightFoot_z']
leftfoot_cols = ['acceleration_LeftFoot_x', 'acceleration_LeftFoot_y', 'acceleration_LeftFoot_z']
xsens_file = Path(__file__).resolve().parent.parent / "data" / "newbee" / "courseA" / "id04" / "xsens.csv"
labels_file = Path(__file__).resolve().parent.parent / "data" / "newbee" / "courseA" / "id04" / "labels.csv"

start_end_times = extract_surface_segments(str(labels_file), target_mode="stairs_down")
usecols = ['time'] + rightfoot_cols + leftfoot_cols
df = pd.read_csv(xsens_file, usecols=usecols)

time_s = df['time'].to_numpy() / 1000.0
fs = round(1 / np.mean(np.diff(time_s)))

# Process Left and Right Foot signals
acc_mag_R = np.linalg.norm(df[rightfoot_cols].to_numpy(), axis=1)
acc_mag_L = np.linalg.norm(df[leftfoot_cols].to_numpy(), axis=1)
acc_z_R = df[rightfoot_cols[2]].to_numpy()
acc_z_L = df[leftfoot_cols[2]].to_numpy()

y_HS, y_FO = extract_golden_standard(pd.read_csv(labels_file))
hs_times_s = y_HS[:, 0] / 1000.0
fo_times_s = y_FO[:, 0] / 1000.0


# --- 2. CWT and Timing Extraction Functions ---
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

def create_vline_coords(x_points, y_min, y_max):
    if len(x_points) == 0:
        return np.array([]), np.array([])
    xs = np.column_stack([x_points, x_points, np.full_like(x_points, np.nan)]).ravel()
    ys = np.tile([y_min, y_max, np.nan], len(x_points))
    return xs, ys

def process_signal_for_freq(sig, fs, freq, time_array):
    c1, scale = compute_cwt(sig, fs, freq)
    min_dist = max(1, int(0.5 * fs / freq))
    hs_idx, hs_t, hs_val = predict_step_timing_from_cwt(c1, time_array, min_distance=min_dist)
    fo_t, fo_val, c2 = predict_foot_off_from_secondary_cwt(c1, time_array, hs_idx, fs, freq)
    return c1, hs_t, hs_val, c2, fo_t, fo_val, scale


# --- 3. Precompute across Target Frequencies ---
frequencies = np.round(np.arange(1.0, 5.1, 0.1), 1)

data_cache = {k: [] for k in [
    'c1_rm', 'hst_rm', 'hsv_rm', 'c2_rm', 'fot_rm', 'fov_rm',
    'c1_lm', 'hst_lm', 'hsv_lm', 'c2_lm', 'fot_lm', 'fov_lm',
    'c1_rz', 'hst_rz', 'hsv_rz', 'c2_rz', 'fot_rz', 'fov_rz',
    'c1_lz', 'hst_lz', 'hsv_lz', 'c2_lz', 'fot_lz', 'fov_lz',
    'wav_t', 'wav_y'
]}

wavelet_time_window = np.linspace(-2.5, 2.5, 1000)

for freq in frequencies:
    # Right Mag
    c1, ht, hv, c2, ft, fv, scale = process_signal_for_freq(acc_mag_R, fs, freq, time_s)
    data_cache['c1_rm'].append(c1); data_cache['hst_rm'].append(ht); data_cache['hsv_rm'].append(hv)
    data_cache['c2_rm'].append(c2); data_cache['fot_rm'].append(ft); data_cache['fov_rm'].append(fv)
    
    # Left Mag
    c1, ht, hv, c2, ft, fv, _ = process_signal_for_freq(acc_mag_L, fs, freq, time_s)
    data_cache['c1_lm'].append(c1); data_cache['hst_lm'].append(ht); data_cache['hsv_lm'].append(hv)
    data_cache['c2_lm'].append(c2); data_cache['fot_lm'].append(ft); data_cache['fov_lm'].append(fv)

    # Right Z
    c1, ht, hv, c2, ft, fv, _ = process_signal_for_freq(acc_z_R, fs, freq, time_s)
    data_cache['c1_rz'].append(c1); data_cache['hst_rz'].append(ht); data_cache['hsv_rz'].append(hv)
    data_cache['c2_rz'].append(c2); data_cache['fot_rz'].append(ft); data_cache['fov_rz'].append(fv)
    
    # Left Z
    c1, ht, hv, c2, ft, fv, _ = process_signal_for_freq(acc_z_L, fs, freq, time_s)
    data_cache['c1_lz'].append(c1); data_cache['hst_lz'].append(ht); data_cache['hsv_lz'].append(hv)
    data_cache['c2_lz'].append(c2); data_cache['fot_lz'].append(ft); data_cache['fov_lz'].append(fv)

    # Wavelet profile
    s_time = scale / fs
    tau = wavelet_time_window / s_time
    psi = (2.0 / (np.sqrt(3.0) * (np.pi ** 0.25) * np.sqrt(s_time))) * (1.0 - tau**2) * np.exp(-0.5 * (tau**2))
    data_cache['wav_t'].append(wavelet_time_window)
    data_cache['wav_y'].append(psi)

default_idx = int(np.where(frequencies == 1.5)[0][0])


# --- 4. Subplot Setup ---
fig = make_subplots(
    rows=7, cols=1, shared_xaxes=False, vertical_spacing=0.05,
    row_heights=[0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
    subplot_titles=(
        'Foot Acceleration Magnitude (Left & Right)',
        'CWT1 Magnitude (HS: Peaks)',
        'CWT2 Magnitude (FO: Minima)',
        'Foot Z Acceleration (Left & Right)',
        'CWT1 Foot Z (HS: Peaks)',
        'CWT2 Foot Z (FO: Minima)',
        'Mexican Hat Wavelet Form (Time Domain)'
    )
)

colors = {'L': '#1f77b4', 'R': '#ff7f0e', 'L_pred': '#9467bd', 'R_pred': '#d62728'}

# Row 1: Raw Mag [Traces 0, 1]
fig.add_trace(go.Scatter(x=time_s, y=acc_mag_L, mode='lines', name='Acc Mag L', line=dict(color=colors['L'])), row=1, col=1)
fig.add_trace(go.Scatter(x=time_s, y=acc_mag_R, mode='lines', name='Acc Mag R', line=dict(color=colors['R'])), row=1, col=1)

# Row 2: CWT1 Mag + HS [Traces 2, 3, 4, 5]
fig.add_trace(go.Scatter(x=time_s, y=data_cache['c1_lm'][default_idx], mode='lines', name='CWT1 Mag L', line=dict(color=colors['L'])), row=2, col=1)
fig.add_trace(go.Scatter(x=data_cache['hst_lm'][default_idx], y=data_cache['hsv_lm'][default_idx], mode='markers', name='Pred HS L', marker=dict(symbol='triangle-up', size=8, color=colors['L_pred'])), row=2, col=1)
fig.add_trace(go.Scatter(x=time_s, y=data_cache['c1_rm'][default_idx], mode='lines', name='CWT1 Mag R', line=dict(color=colors['R'])), row=2, col=1)
fig.add_trace(go.Scatter(x=data_cache['hst_rm'][default_idx], y=data_cache['hsv_rm'][default_idx], mode='markers', name='Pred HS R', marker=dict(symbol='triangle-up', size=8, color=colors['R_pred'])), row=2, col=1)

# Row 3 : CWT2 Mag + FO [Traces 6, 7, 8, 9]
fig.add_trace(go.Scatter(x=time_s, y=data_cache['c2_lm'][default_idx], mode='lines', name='CWT2 Mag L', line=dict(color=colors['L'])), row=3, col=1)
fig.add_trace(go.Scatter(x=data_cache['fot_lm'][default_idx], y=data_cache['fov_lm'][default_idx], mode='markers', name='Pred FO L', marker=dict(symbol='triangle-down', size=8, color=colors['L_pred'])), row=3, col=1)
fig.add_trace(go.Scatter(x=time_s, y=data_cache['c2_rm'][default_idx], mode='lines', name='CWT2 Mag R', line=dict(color=colors['R'])), row=3, col=1)
fig.add_trace(go.Scatter(x=data_cache['fot_rm'][default_idx], y=data_cache['fov_rm'][default_idx], mode='markers', name='Pred FO R', marker=dict(symbol='triangle-down', size=8, color=colors['R_pred'])), row=3, col=1)

# Row 4: Raw Acc Z [Traces 10, 11]
fig.add_trace(go.Scatter(x=time_s, y=acc_z_L, mode='lines', name='Acc Z L', line=dict(color=colors['L'])), row=4, col=1)
fig.add_trace(go.Scatter(x=time_s, y=acc_z_R, mode='lines', name='Acc Z R', line=dict(color=colors['R'])), row=4, col=1)

# Row 5: CWT1 Z + HS [Traces 12, 13, 14, 15]
fig.add_trace(go.Scatter(x=time_s, y=data_cache['c1_lz'][default_idx], mode='lines', name='CWT1 Z L', line=dict(color=colors['L'])), row=5, col=1)
fig.add_trace(go.Scatter(x=data_cache['hst_lz'][default_idx], y=data_cache['hsv_lz'][default_idx], mode='markers', name='Pred HS Z L', marker=dict(symbol='triangle-up', size=8, color=colors['L_pred']), showlegend=False), row=5, col=1)
fig.add_trace(go.Scatter(x=time_s, y=data_cache['c1_rz'][default_idx], mode='lines', name='CWT1 Z R', line=dict(color=colors['R'])), row=5, col=1)
fig.add_trace(go.Scatter(x=data_cache['hst_rz'][default_idx], y=data_cache['hsv_rz'][default_idx], mode='markers', name='Pred HS Z R', marker=dict(symbol='triangle-up', size=8, color=colors['R_pred']), showlegend=False), row=5, col=1)

# Row 6: CWT2 Z + FO [Traces 16, 17, 18, 19]
fig.add_trace(go.Scatter(x=time_s, y=data_cache['c2_lz'][default_idx], mode='lines', name='CWT2 Z L', line=dict(color=colors['L'])), row=6, col=1)
fig.add_trace(go.Scatter(x=data_cache['fot_lz'][default_idx], y=data_cache['fov_lz'][default_idx], mode='markers', name='Pred FO Z L', marker=dict(symbol='triangle-down', size=8, color=colors['L_pred']), showlegend=False), row=6, col=1)
fig.add_trace(go.Scatter(x=time_s, y=data_cache['c2_rz'][default_idx], mode='lines', name='CWT2 Z R', line=dict(color=colors['R'])), row=6, col=1)
fig.add_trace(go.Scatter(x=data_cache['fot_rz'][default_idx], y=data_cache['fov_rz'][default_idx], mode='markers', name='Pred FO Z R', marker=dict(symbol='triangle-down', size=8, color=colors['R_pred']), showlegend=False), row=6, col=1)

# Row 7: Wavelet Visual [Trace 20]
fig.add_trace(go.Scatter(x=data_cache['wav_t'][default_idx], y=data_cache['wav_y'][default_idx], mode='lines', name='Scaled mexh', line=dict(color='#333333', width=2)), row=7, col=1)


# Add True Events Background Lines spanning global L/R min-max
y_bounds = [
    (min(np.min(acc_mag_L), np.min(acc_mag_R)), max(np.max(acc_mag_L), np.max(acc_mag_R))),
    (min(np.min(data_cache['c1_lm']), np.min(data_cache['c1_rm'])), max(np.max(data_cache['c1_lm']), np.max(data_cache['c1_rm']))),
    (min(np.min(data_cache['c2_lm']), np.min(data_cache['c2_rm'])), max(np.max(data_cache['c2_lm']), np.max(data_cache['c2_rm']))),
    (min(np.min(acc_z_L), np.min(acc_z_R)), max(np.max(acc_z_L), np.max(acc_z_R))),
    (min(np.min(data_cache['c1_lz']), np.min(data_cache['c1_rz'])), max(np.max(data_cache['c1_lz']), np.max(data_cache['c1_rz']))),
    (min(np.min(data_cache['c2_lz']), np.min(data_cache['c2_rz'])), max(np.max(data_cache['c2_lz']), np.max(data_cache['c2_rz']))),
]

for row_idx, (ymin, ymax) in enumerate(y_bounds, start=1):
    vl_x_hs, vl_y_hs = create_vline_coords(hs_times_s, float(ymin), float(ymax))
    fig.add_trace(go.Scatter(x=vl_x_hs, y=vl_y_hs, mode='lines', line=dict(color='green', width=1, dash='dash'), opacity=0.45, name='True HS', legendgroup='true_hs', showlegend=(row_idx == 1), hoverinfo='skip'), row=row_idx, col=1)
    
    vl_x_fo, vl_y_fo = create_vline_coords(fo_times_s, float(ymin), float(ymax))
    fig.add_trace(go.Scatter(x=vl_x_fo, y=vl_y_fo, mode='lines', line=dict(color='magenta', width=1, dash='dot'), opacity=0.45, name='True FO', legendgroup='true_fo', showlegend=(row_idx == 1), hoverinfo='skip'), row=row_idx, col=1)

# Navigation Controls
dropdown_buttons = []
for i in range(len(start_end_times)):
    t_start = float(start_end_times["start_time"].iloc[i] / 1000.0)
    t_end = float(start_end_times["end_time"].iloc[i] / 1000.0)
    dropdown_buttons.append(dict(
        method="relayout",
        args=[{"xaxis.range": [t_start, t_end], "xaxis2.range": [t_start, t_end], "xaxis3.range": [t_start, t_end], "xaxis4.range": [t_start, t_end], "xaxis5.range": [t_start, t_end], "xaxis6.range": [t_start, t_end]}],
        label=f"Seg {i+1} ({t_start:.1f}s - {t_end:.1f}s)"
    ))

# --- 6. Controls: Slider Mapping exactly 17 dynamic traces ---
slider_steps = []
for i, freq in enumerate(frequencies):
    step = dict(
        method="restyle",
        args=[
            {
                "x": [
                    time_s, data_cache['hst_lm'][i], time_s, data_cache['hst_rm'][i],      # 2, 3, 4, 5
                    time_s, data_cache['fot_lm'][i], time_s, data_cache['fot_rm'][i],      # 6, 7, 8, 9
                    time_s, data_cache['hst_lz'][i], time_s, data_cache['hst_rz'][i],      # 12, 13, 14, 15
                    time_s, data_cache['fot_lz'][i], time_s, data_cache['fot_rz'][i],      # 16, 17, 18, 19
                    data_cache['wav_t'][i]                                                 # 20
                ],
                "y": [
                    data_cache['c1_lm'][i], data_cache['hsv_lm'][i], data_cache['c1_rm'][i], data_cache['hsv_rm'][i], 
                    data_cache['c2_lm'][i], data_cache['fov_lm'][i], data_cache['c2_rm'][i], data_cache['fov_rm'][i],
                    data_cache['c1_lz'][i], data_cache['hsv_lz'][i], data_cache['c1_rz'][i], data_cache['hsv_rz'][i], 
                    data_cache['c2_lz'][i], data_cache['fov_lz'][i], data_cache['c2_rz'][i], data_cache['fov_rz'][i],
                    data_cache['wav_y'][i]
                ]
            },
            [2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        ],
        label=f"{freq:.1f} Hz"
    )
    slider_steps.append(step)

init_x_min = float(start_end_times["start_time"].iloc[0] / 1000.0)
init_x_max = float(start_end_times["end_time"].iloc[0] / 1000.0)

fig.update_layout(
    title="CWT Gait Events (Left & Right Feet): Heel Strike & Foot Off",
    height=1400,
    hovermode="x unified",
    showlegend=True,
    xaxis=dict(range=[init_x_min, init_x_max], matches='x6', showticklabels=False),
    xaxis2=dict(matches='x6', showticklabels=False),
    xaxis3=dict(matches='x6', showticklabels=False),
    xaxis4=dict(matches='x6', showticklabels=False),
    xaxis5=dict(matches='x6', showticklabels=False),
    xaxis6=dict(title='Gait Time (seconds)'),
    xaxis7=dict(title='Wavelet Time (seconds)', range=[-1.5, 1.5]),
    yaxis=dict(title='Acc Mag'),
    yaxis2=dict(title='CWT1 Mag'),
    yaxis3=dict(title='CWT2 Mag'),
    yaxis4=dict(title='Acc Z'),
    yaxis5=dict(title='CWT1 Z'),
    yaxis6=dict(title='CWT2 Z'),
    yaxis7=dict(title='Wavelet Amp'),
    sliders=[dict(
        active=default_idx,
        currentvalue={"prefix": "Wavelet Frequency: "},
        pad={"t": 55},
        steps=slider_steps
    )],
    updatemenus=[dict(
        type="dropdown",
        direction="down",
        x=1.0,
        y=1.05,
        showactive=True,
        buttons=dropdown_buttons
    )]
)

fig.show()