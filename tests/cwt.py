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
xsens_file = Path(__file__).resolve().parent.parent / "data" / "newbee" / "courseA" / "id01" / "xsens.csv"
labels_file = Path(__file__).resolve().parent.parent / "data" / "newbee" / "courseA" / "id01" / "labels.csv"

start_end_times = extract_surface_segments(str(labels_file), target_mode="stairs_down")
df = pd.read_csv(xsens_file, usecols=['time'] + accel_cols)

time_s = df['time'].to_numpy() / 1000.0
fs = round(1 / np.mean(np.diff(time_s)))
acc_mag = np.linalg.norm(df[accel_cols].to_numpy(), axis=1)
acc_z = df[accel_cols[2]].to_numpy()

y_HS, y_FO = extract_golden_standard(pd.read_csv(labels_file))
hs_times_s = y_HS[:, 0] / 1000.0
fo_times_s = y_FO[:, 0] / 1000.0


# --- 2. CWT and Timing Extraction Functions ---
def compute_cwt(signal, fs, step_freq, wavelet='gaus2'):
    fc = pywt.central_frequency(wavelet)
    scale = (fc * fs) / step_freq
    coefs, _ = pywt.cwt(signal, scales=[scale], wavelet=wavelet)
    return coefs[0], scale


def predict_step_timing_from_cwt(cwt_output, time_array, min_distance=None):
    """Initial Contact (Heel Strike) identified from primary CWT peaks."""
    peaks, _ = find_peaks(cwt_output, distance=min_distance)
    return peaks, time_array[peaks], cwt_output[peaks]


def predict_foot_off_from_secondary_cwt(cwt1_output, time_array, hs_peak_indices, fs, step_freq):
    """
    Applies a second CWT to the primary CWT output and searches for local 
    minima between consecutive heel strikes to identify Final Contact (Foot Off).
    """
    # Second CWT on top of the first CWT output
    cwt2_output, _ = compute_cwt(cwt1_output, fs, step_freq=step_freq, wavelet='gaus2')
    
    fo_indices = []
    # Search for local minima between successive heel strikes
    for i in range(len(hs_peak_indices) - 1):
        idx_start = hs_peak_indices[i]
        idx_end = hs_peak_indices[i + 1]
        
        # Look within the step interval for the minimum of CWT2
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


# --- 3. Precompute across Target Frequencies ---
frequencies = np.round(np.arange(1.0, 5.1, 0.1), 1)

cwt_mag_by_freq = []
cwt_z_by_freq = []
cwt2_mag_by_freq = []
cwt2_z_by_freq = []
wavelet_t_by_freq = []
wavelet_y_by_freq = []

hs_pred_mag_x, hs_pred_mag_y = [], []
fo_pred_mag_x, fo_pred_mag_y = [], []
hs_pred_z_x, hs_pred_z_y = [], []
fo_pred_z_x, fo_pred_z_y = [], []

wavelet_time_window = np.linspace(-2.5, 2.5, 1000)

for freq in frequencies:
    # 1st CWT
    c_mag, scale = compute_cwt(acc_mag, fs, freq)
    c_z, _ = compute_cwt(acc_z, fs, freq)
    
    cwt_mag_by_freq.append(c_mag)
    cwt_z_by_freq.append(c_z)
    
    min_dist = max(1, int(0.5 * fs / freq))

    # Initial Contact (HS) via CWT-1 peaks
    hs_idx_mag, hs_t_mag, hs_val_mag = predict_step_timing_from_cwt(c_mag, time_s, min_distance=min_dist)
    hs_idx_z, hs_t_z, hs_val_z = predict_step_timing_from_cwt(c_z, time_s, min_distance=min_dist)
    
    hs_pred_mag_x.append(hs_t_mag)
    hs_pred_mag_y.append(hs_val_mag)
    hs_pred_z_x.append(hs_t_z)
    hs_pred_z_y.append(hs_val_z)

    # Final Contact (FO) via Secondary CWT minima
    fo_t_mag, fo_val_mag, cwt2_mag_output = predict_foot_off_from_secondary_cwt(c_mag, time_s, hs_idx_mag, fs, freq)
    fo_t_z, fo_val_z, cwt2_z_output = predict_foot_off_from_secondary_cwt(c_z, time_s, hs_idx_z, fs, freq)

    cwt2_mag_by_freq.append(cwt2_mag_output)
    cwt2_z_by_freq.append(cwt2_z_output)
    
    fo_pred_mag_x.append(fo_t_mag)
    fo_pred_mag_y.append(fo_val_mag)
    fo_pred_z_x.append(fo_t_z)
    fo_pred_z_y.append(fo_val_z)

    # Scaled wavelet profile
    s_time = scale / fs
    tau = wavelet_time_window / s_time
    norm_const = 2.0 / (np.sqrt(3.0) * (np.pi ** 0.25) * np.sqrt(s_time))
    psi = - norm_const * (1.0 - tau**2) * np.exp(-0.5 * (tau**2))
    wavelet_t_by_freq.append(wavelet_time_window)
    wavelet_y_by_freq.append(psi)

default_idx = int(np.where(frequencies == 1.5)[0][0])


# --- 4. Subplot Setup ---
fig = make_subplots(
    rows=7,
    cols=1,
    shared_xaxes=False,
    vertical_spacing=0.05,
    row_heights=[0.20, 0.20, 0.20, 0.20, 0.20, 0.2, 0.2],
    subplot_titles=(
        'Acceleration Magnitude',
        'CWT Magnitude (HS: Peaks)',
        '2nd CWT Magnitude (FO: Minima)',
        'Pelvis Z Acceleration',
        'CWT Pelvis Z (HS: Peaks)',
        '2nd CWT Pelvis Z (FO: Minima)',
        'Mexican Hat Wavelet Form (Time Domain)'
    )
)

# Row 1: Raw Mag
fig.add_trace(go.Scatter(x=time_s, y=acc_mag, mode='lines', name='Acc Mag', line=dict(color='#1f77b4')), row=1, col=1)

# Row 2: CWT Mag + HS Events
fig.add_trace(go.Scatter(x=time_s, y=cwt_mag_by_freq[default_idx], mode='lines', name='CWT Acc Mag', line=dict(color='#ff7f0e')), row=2, col=1)
fig.add_trace(go.Scatter(x=hs_pred_mag_x[default_idx], y=hs_pred_mag_y[default_idx], mode='markers', name='Pred HS (CWT1 Peaks)', marker=dict(symbol='triangle-up', size=8, color='#d62728'), legendgroup='pred_hs'), row=2, col=1)

# Row 3 : CWT2 + FO events
fig.add_trace(go.Scatter(x=time_s, y=cwt2_mag_by_freq[default_idx], mode='lines', name='2nd CWT Acc Mag', line=dict(color='#9467bd')), row=3, col=1)
fig.add_trace(go.Scatter(x=fo_pred_mag_x[default_idx], y=fo_pred_mag_y[default_idx], mode='markers', name='Pred FO (2nd CWT Min)', marker=dict(symbol='triangle-down', size=8, color='#d62728'), legendgroup='pred_fo'), row=3, col=1)

# Row 4: Raw Acc Z
fig.add_trace(go.Scatter(x=time_s, y=acc_z, mode='lines', name='Acc Pelvis Z', line=dict(color='#2ca02c')), row=4, col=1)

# Row 5: CWT Z + Events
fig.add_trace(go.Scatter(x=time_s, y=cwt_z_by_freq[default_idx], mode='lines', name='CWT Pelvis Z', line=dict(color='#17becf')), row=5, col=1)
fig.add_trace(go.Scatter(x=hs_pred_z_x[default_idx], y=hs_pred_z_y[default_idx], mode='markers', name='Pred HS', marker=dict(symbol='triangle-up', size=8, color='#d62728'), legendgroup='pred_hs', showlegend=False), row=5, col=1)

# Row 6: CWT2 Z + FO events
fig.add_trace(go.Scatter(x=time_s, y=cwt2_z_by_freq[default_idx], mode='lines', name='2nd CWT Pelvis Z', line=dict(color='#9467bd')), row=6, col=1)
fig.add_trace(go.Scatter(x=fo_pred_z_x[default_idx], y=fo_pred_z_y[default_idx], mode='markers', name='Pred FO', marker=dict(symbol='triangle-down', size=8, color='#d62728'), legendgroup='pred_fo', showlegend=False), row=6, col=1)

# Row 7: Wavelet Visual
fig.add_trace(go.Scatter(x=wavelet_t_by_freq[default_idx], y=wavelet_y_by_freq[default_idx], mode='lines', name='Scaled mexh', line=dict(color='#333333', width=2)), row=7, col=1)



# Add Golden Standards
all_mag_cwt = np.concatenate(cwt_mag_by_freq)
all_z_cwt = np.concatenate(cwt_z_by_freq)
all_mag_cwt2 = np.concatenate(cwt2_mag_by_freq)
all_z_cwt2 = np.concatenate(cwt2_z_by_freq)
y_bounds = [
    (float(np.min(acc_mag)), float(np.max(acc_mag))),
    (float(np.min(all_mag_cwt)), float(np.max(all_mag_cwt))),
    (float(np.min(all_mag_cwt2)), float(np.max(all_mag_cwt2))),
    (float(np.min(acc_z)), float(np.max(acc_z))),
    (float(np.min(all_z_cwt)), float(np.max(all_z_cwt))),
    (float(np.min(all_z_cwt2)), float(np.max(all_z_cwt2))),
]

for row_idx, (ymin, ymax) in enumerate(y_bounds, start=1):
    vl_x_hs, vl_y_hs = create_vline_coords(hs_times_s, ymin, ymax)
    fig.add_trace(go.Scatter(x=vl_x_hs, y=vl_y_hs, mode='lines', line=dict(color='green', width=1, dash='dash'), opacity=0.45, name='True HS', legendgroup='true_hs', showlegend=(row_idx == 1), hoverinfo='skip'), row=row_idx, col=1)
    
    vl_x_fo, vl_y_fo = create_vline_coords(fo_times_s, ymin, ymax)
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

# --- 6. Controls: Slider & Navigation ---
slider_steps = []
for i, freq in enumerate(frequencies):
    step = dict(
        method="restyle",
        args=[
            {
                # Exactly 9 items matching the 9 updated traces
                "x": [
                    time_s,                 # Trace 1: CWT1 Mag
                    hs_pred_mag_x[i],       # Trace 2: Pred HS Mag
                    time_s,                 # Trace 3: CWT2 Mag
                    fo_pred_mag_x[i],       # Trace 4: Pred FO Mag
                    time_s,                 # Trace 6: CWT1 Z
                    hs_pred_z_x[i],         # Trace 7: Pred HS Z
                    time_s,                 # Trace 8: CWT2 Z
                    fo_pred_z_x[i],         # Trace 9: Pred FO Z
                    wavelet_t_by_freq[i]    # Trace 10: Wavelet Visual
                ],
                "y": [
                    cwt_mag_by_freq[i],     # Trace 1: CWT1 Mag
                    hs_pred_mag_y[i],       # Trace 2: Pred HS Mag
                    cwt2_mag_by_freq[i],    # Trace 3: CWT2 Mag
                    fo_pred_mag_y[i],       # Trace 4: Pred FO Mag
                    cwt_z_by_freq[i],       # Trace 6: CWT1 Z
                    hs_pred_z_y[i],         # Trace 7: Pred HS Z
                    cwt2_z_by_freq[i],      # Trace 8: CWT2 Z
                    fo_pred_z_y[i],         # Trace 9: Pred FO Z
                    wavelet_y_by_freq[i]    # Trace 10: Wavelet Visual
                ]
            },
            [1, 2, 3, 4, 6, 7, 8, 9, 10]    # Exact 9 trace indices
        ],
        label=f"{freq:.1f} Hz"
    )
    slider_steps.append(step)

init_x_min = float(start_end_times["start_time"].iloc[0] / 1000.0)
init_x_max = float(start_end_times["end_time"].iloc[0] / 1000.0)

fig.update_layout(
    title="CWT Gait Events: Heel Strike (1st CWT Peaks) & Foot Off (2nd CWT Minima)",
    height=1200,
    hovermode="x unified",
    showlegend=True,
    # Synchronize all 6 signal subplots (xaxis through xaxis6)
    xaxis=dict(range=[init_x_min, init_x_max], matches='x6', showticklabels=False),
    xaxis2=dict(matches='x6', showticklabels=False),
    xaxis3=dict(matches='x6', showticklabels=False),
    xaxis4=dict(matches='x6', showticklabels=False),
    xaxis5=dict(matches='x6', showticklabels=False),
    xaxis6=dict(title='Gait Time (seconds)'),
    # Independent axis for the 7th subplot (Wavelet profile)
    xaxis7=dict(title='Wavelet Time (seconds)', range=[-1.5, 1.5]),
    yaxis=dict(title='Acc (m/s²)'),
    yaxis2=dict(title='CWT Mag'),
    yaxis3=dict(title='CWT2 Mag'),
    yaxis4=dict(title='Pelvis Z (m/s²)'),
    yaxis5=dict(title='CWT Z'),
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
        y=1.08,
        showactive=True,
        buttons=dropdown_buttons
    )]
)

fig.show()