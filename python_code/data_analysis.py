import os
import re
import warnings
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import linear_sum_assignment
from python_code.Toolboxes.PyShoe.pyshoe_export import clean_raw_zupt_mask
import plotly.express as px
import pandas as pd

# Suppress runtime warnings for mean calculations on empty slices
warnings.filterwarnings('ignore', category=RuntimeWarning)

class GaitAnalysisPipeline:
    def __init__(self, data_dir: str, tolerance_ms: float = 50.0):
        self.data_dir = Path(data_dir)
        self.tolerance_ms = tolerance_ms
        self.df_all_steps = pd.DataFrame()
        self.df_all_segments = pd.DataFrame()

    def _split_feet(self, array) -> tuple[np.ndarray, np.ndarray]:
        if array is None or (isinstance(array, (list, np.ndarray)) and len(array) == 0):
            return np.array([]), np.array([])
            
        array = np.asarray(array)
        if array.ndim == 1 or array.shape[1] < 2:
            return array.flatten().astype(float), np.array([])
            
        right_foot = array[array[:, 1] == 0][:, 0].astype(float)
        left_foot = array[array[:, 1] == 1][:, 0].astype(float)
        
        return right_foot, left_foot

    def _match_events(self, t_true: np.ndarray, t_pred: np.ndarray) -> list:
        step_records = []
        if len(t_true) == 0 and len(t_pred) == 0:
            return step_records
        if len(t_true) == 0:
            return [{'error_ms': np.nan, 'classification': 'FP'} for _ in t_pred]
        if len(t_pred) == 0:
            return [{'error_ms': np.nan, 'classification': 'FN'} for _ in t_true]

        diff_matrix = np.abs(t_pred[:, None] - t_true[None, :])
        cost_matrix = diff_matrix.copy()
        cost_matrix[cost_matrix > self.tolerance_ms] = 1e9

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched_pred, matched_true = set(), set()

        for p_idx, t_idx in zip(row_ind, col_ind):
            error = t_pred[p_idx] - t_true[t_idx]
            if abs(error) <= self.tolerance_ms:
                step_records.append({'error_ms': error, 'classification': 'TP'})
                matched_pred.add(p_idx)
                matched_true.add(t_idx)

        for p_idx in range(len(t_pred)):
            if p_idx not in matched_pred:
                step_records.append({'error_ms': np.nan, 'classification': 'FP'})

        for t_idx in range(len(t_true)):
            if t_idx not in matched_true:
                step_records.append({'error_ms': np.nan, 'classification': 'FN'})

        return step_records

    def parse_and_match(self):
        print(f"Scanning for .mat files in: {self.data_dir}...")
        mat_files = list(self.data_dir.rglob("*.mat"))
        
        all_steps, all_segments = [], []

        for file_path in mat_files:
            try:
                detector = file_path.parent.name
                file_stem = file_path.stem
                
                parts = file_stem.split('_')
                if len(parts) >= 3 and "clip" in parts:
                    clip_idx = parts.index("clip")
                    clip_id = parts[clip_idx + 1]
                    subject_id = parts[0]
                    course = "_".join(parts[1:clip_idx])
                else:
                    subject_id = parts[0]
                    course = parts[1] if len(parts) > 1 else "unknown"
                    clip_id = parts[-1]

                mat = loadmat(str(file_path), squeeze_me=True, struct_as_record=False)
                if 'results' not in mat: continue
                res = mat['results']
                
                events = {
                    'HS': (getattr(res, 'y_HS', []), getattr(res, 'y_hat_HS', [])),
                    'FO': (getattr(res, 'y_FO', []), getattr(res, 'y_hat_FO', []))
                }

                for event_type, (true_arr, pred_arr) in events.items():
                    true_r, true_l = self._split_feet(true_arr)
                    pred_r, pred_l = self._split_feet(pred_arr)

                    matched_r = self._match_events(true_r, pred_r)
                    for rec in matched_r: rec['foot'] = 'R'
                    
                    matched_l = self._match_events(true_l, pred_l)
                    for rec in matched_l: rec['foot'] = 'L'
                    
                    matched_records = matched_r + matched_l
                    
                    for rec in matched_records:
                        rec.update({'detector': detector, 'subject': subject_id, 'course': course, 'clip_id': clip_id, 'event_type': event_type})
                        all_steps.append(rec)

                    df_temp = pd.DataFrame(matched_records)
                    if not df_temp.empty:
                        tp = (df_temp['classification'] == 'TP').sum()
                        fp = (df_temp['classification'] == 'FP').sum()
                        fn = (df_temp['classification'] == 'FN').sum()
                        
                        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

                        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0
                        f2 = (1 + 4.0) * (precision * recall) / ((4.0 * precision) + recall) if precision + recall > 0 else 0.0
                        
                        tp_errors = df_temp.loc[df_temp['classification'] == 'TP', 'error_ms']
                        mae = tp_errors.abs().mean() if not tp_errors.empty else np.nan
                        std_err = tp_errors.std() if len(tp_errors) > 1 else 0.0

                        all_segments.append({
                            'detector': detector,
                            'subject': subject_id,
                            'course': course,
                            'clip_id': clip_id,
                            'event_type': event_type,
                            'precision': precision,
                            'recall': recall,
                            'f1_score': f1,
                            'f2_score': f2, 
                            'mean_absolute_error_ms': mae,
                            'std_error_ms': std_err
                        })
            except Exception as e:
                print(f"Error processing {file_path.name}: {str(e)}")

        self.df_all_steps = pd.DataFrame(all_steps)
        self.df_all_segments = pd.DataFrame(all_segments)
        self.filtered_df = None

    def generate_summary_table_without_worst(self, num: int = 0):
        if self.df_all_segments.empty:
            print("No data available to summarize.")
            return

        print(f"\n### Detector Performance Summary (Excluding {num} worst segments per group)")
        
        if num > 0:
            # Sort by F1 score descending (best at the top, worst at the bottom)
            sorted_df = self.df_all_segments.sort_values(by='f1_score', ascending=False)
            
            # cumcount(ascending=False) numbers rows from the bottom of each group up 
            # (0 = worst, 1 = second worst, etc.). 
            # Keeping rows >= num cleanly slices off the bottom 'num' rows.
            mask = sorted_df.groupby(['detector', 'event_type']).cumcount(ascending=False) >= num
            self.filtered_df = sorted_df[mask].reset_index(drop=True)
            
            if self.filtered_df.empty:
                print(f"Warning: Dropping the worst {num} segments removed ALL data.")
                print("Make sure you have more segments per group than the number you are trying to drop.")
                return
        else:
            self.filtered_df = self.df_all_segments

        # Aggregating metrics on the filtered dataset
        summary = self.filtered_df.groupby(['detector', 'event_type']).agg(
            f1_mean=('f1_score', 'mean'),
            f1_std=('f1_score', 'std'),          
            f2_mean=('f2_score', 'mean'),
            precision_mean=('precision', 'mean'),
            precision_std=('precision', 'std'),
            recall_mean=('recall', 'mean'),
            mae_mean=('mean_absolute_error_ms', 'mean'),
            mae_std=('mean_absolute_error_ms', 'std'),
            retained_segments=('f1_score', 'count') # Tracks remaining segments after the cut
        ).round(3)

        print(summary.to_markdown())
        return summary

    def visualize(self):
        if self.df_all_steps.empty: return
        sns.set_theme(style="whitegrid", palette="muted")
        
        tp_steps = self.df_all_steps[self.df_all_steps['classification'] == 'TP']
        
        # --- SPLIT TIMING ERROR HISTOGRAM ---
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True, sharex=True)
        
        sns.histplot(data=tp_steps[tp_steps['event_type'] == 'HS'], 
                     x='error_ms', hue='detector', kde=True, element="step", alpha=0.3, ax=axes[0])
        axes[0].set_title('Timing Errors: Heel Strike (HS)', fontsize=14)
        axes[0].set_xlabel('Timing Error (ms)')

        sns.histplot(data=tp_steps[tp_steps['event_type'] == 'FO'], 
                     x='error_ms', hue='detector', kde=True, element="step", alpha=0.3, ax=axes[1])
        axes[1].set_title('Timing Errors: Foot Off (FO)', fontsize=14)
        axes[1].set_xlabel('Timing Error (ms)')

        plt.tight_layout()
        plt.show()

        # --- BOXPLOTS ---
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        sns.boxplot(
            data=self.filtered_df, 
            x='detector', 
            y='f1_score', 
            hue='event_type', 
            ax=axes[0]
        )
        axes[0].set_title('Consistency: F1-Score per Staircase Segment', fontsize=14)
        axes[0].set_ylabel('F1-Score (0.0 to 1.0)')
        
        sns.boxplot(data=self.filtered_df, x='detector', y='mean_absolute_error_ms', hue='event_type', ax=axes[1])
        axes[1].set_title('Consistency: Mean Absolute Timing Error (MAE)')
        plt.show()

    def visualize_extremes(self, detector: str, metric: str = 'f1_score', event: str = 'FO'):
        """Plots raw acceleration, true steps, and predicted steps for best, avg, worst segments."""
        df_det = self.filtered_df[(self.filtered_df['detector'] == detector) & (self.filtered_df['event_type'] == event)]
        
        if df_det.empty:
            print(f"No data found for detector '{detector}' and event '{event}'.")
            return

        # Sort to find the Worst, Median, and Best segments
        df_sorted = df_det.sort_values(by=metric).dropna(subset=[metric]).reset_index(drop=True)
        if len(df_sorted) < 3:
            print("Not enough segments to plot extremes.")
            return

        worst, average, best = df_sorted.iloc[0], df_sorted.iloc[len(df_sorted) // 2], df_sorted.iloc[-1]
        # worst, average, best = df_sorted.iloc[0], df_sorted.iloc[1], df_sorted.iloc[2]
        segments = [('Worst', worst), ('Average', average), ('Best', best)]
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=False)
        fig.suptitle(f"Detector: {detector.upper()} | Prediction Extremes (Event: {event}, Metric: {metric})", fontsize=16)

        for ax, (label, row) in zip(axes, segments):
            file_name = f"{row['subject']}_{row['course']}_{row['clip_id']}.mat"
            file_path = self.data_dir / detector / file_name
            
            try:
                mat = loadmat(str(file_path), squeeze_me=True, struct_as_record=False)
                res = mat['results']
                
                # Check if raw data exists in the .mat file
                # if not hasattr(res, 'time') or not hasattr(res, 'acc_mag_l'):
                #     ax.set_title(f"{label}: {file_name} — [MISSING RAW ACCELERATION]")
                #     ax.text(0.5, 0.5, "Raw 'time' and 'acc_mag_l' missing from .mat file.\nUpdate pyshoe_export.py.", ha='center', va='center')
                #     continue
                
                t_raw = res.time
                acc_l, acc_r = None, None  # Pre-initialize to avoid NameError

                if hasattr(res, "acc_mag_l"):
                    acc_l = res.acc_mag_l
                    ax.plot(t_raw, acc_l, label='Left Accel', color="#9f9f9f", linewidth=1.5, alpha=0.6, zorder = 3)

                if hasattr(res, "acc_mag_r"):
                    acc_r = res.acc_mag_r
                    ax.plot(t_raw, acc_r, label='Right Accel', color="#3f3f3f", linewidth=1.5, alpha=0.6, zorder = 3)
                    
                # Extract true vs predicted events
                true_events = getattr(res, f'y_{event}')
                pred_events = getattr(res, f'y_hat_{event}')
                true_r, true_l = self._split_feet(true_events)
                pred_r, pred_l = self._split_feet(pred_events)
                
                # Safe evaluation of max limits for NumPy arrays
                max_l = np.max(acc_l) if (acc_l is not None and acc_l.size > 0) else 0
                max_r = np.max(acc_r) if (acc_r is not None and acc_r.size > 0) else 0
                y_max = max(max_l, max_r)

                # Overlay True/Pred (Left)
                if len(true_l) > 0:
                    ax.vlines(true_l, ymin=0, ymax=y_max, color='#2ca02c', linestyle='-', linewidth=2, alpha=0.8, label='True (L)')
                if len(pred_l) > 0:
                    ax.vlines(pred_l, ymin=0, ymax=y_max, color='#2ca02c', linestyle='--', linewidth=2, alpha=0.8, label='Pred (L)')
                    
                # Overlay True/Pred (Right)
                if len(true_r) > 0:
                    ax.vlines(true_r, ymin=0, ymax=y_max, color='#1f77b4', linestyle='-', linewidth=2, alpha=0.8, label='True (R)')
                if len(pred_r) > 0:
                    ax.vlines(pred_r, ymin=0, ymax=y_max, color='#1f77b4', linestyle='--', linewidth=2, alpha=0.8, label='Pred (R)')

                ax.set_title(f"{label} Segment: {file_name} | {metric} = {row[metric]:.3f} | MAE = {row['mean_absolute_error_ms']:.1f}ms")
                ax.set_ylabel("Accel (m/s²)")
                ax.set_xlim([t_raw[0], t_raw[-1]])
                
                handles, labels_lgd = ax.get_legend_handles_labels()
                by_label = dict(zip(labels_lgd, handles))
                ax.legend(by_label.values(), by_label.keys(), loc='upper right')
                
            except Exception as e:
                ax.set_title(f"Error loading {file_name}: {e}")

        plt.xlabel("Time (ms)")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def generate_roc_curve(self, detector: str = 'shoe', num_thresholds: int = 100):
        """Generates an event-based ROC curve for HS and FO independently."""
        
        if detector != 'shoe':
            print("ROC generation currently requires raw T statistics exported (only implemented for 'shoe').")
            return
            
        print(f"Loading datasets and sweeping thresholds for {detector.upper()}...")
        
        # Determine valid filenames based on the filtered dataframe
        valid_filenames = None
        if self.filtered_df is not None and not self.filtered_df.empty:
            df_det = self.filtered_df[self.filtered_df['detector'] == detector]
            if not df_det.empty:
                # Separate the remaining files by event type
                df_hs = df_det[df_det['event_type'] == 'HS']
                df_fo = df_det[df_det['event_type'] == 'FO']
                
                hs_files = set(f"{row['subject']}_{row['course']}_{row['clip_id']}.mat" for _, row in df_hs.iterrows())
                fo_files = set(f"{row['subject']}_{row['course']}_{row['clip_id']}.mat" for _, row in df_fo.iterrows())
                
                # STRICT FILTER: Only keep files that survived BOTH F1 cuts
                valid_filenames = hs_files.intersection(fo_files)
                
                print(f"Filter active: Generating ROC using only the {len(valid_filenames)} strictly retained clips.")
            else:
                print(f"Warning: No retained clips found for detector '{detector}'. Check your dataframe.")
                return
        else:
            print("No filtered summary found. Generating ROC using ALL available clips.")

        # 1. Cache the raw data per clip
        clips_data = []
        mat_files = list((self.data_dir / detector).rglob("*.mat"))
        
        for file_path in mat_files:
            if valid_filenames is not None and file_path.name not in valid_filenames:
                continue

            mat = loadmat(str(file_path), squeeze_me=True, struct_as_record=False)
            res = mat['results']
            if not hasattr(res, 'T_l') or len(res.T_l) == 0: continue
                
            t_raw = res.time
            fs = 1000.0 / np.mean(np.diff(t_raw)) 
            
            # Extract ground truth arrays directly
            true_hs_l = self._split_feet(res.y_HS)[1]
            true_fo_l = self._split_feet(res.y_FO)[1]
                    
            clips_data.append({
                'T': res.T_l,
                'time': t_raw,
                'true_hs': true_hs_l,
                'true_fo': true_fo_l,
                'fs': fs
            })

        if not clips_data:
            print("No valid T-statistics found matching the filter criteria.")
            return

        # 2. Establish threshold sweep range
        all_T = np.concatenate([c['T'] for c in clips_data])
        min_T, max_T = np.percentile(all_T, 1), np.percentile(all_T, 99)
        thresholds = np.logspace(np.log10(max(min_T, 1e-5)), np.log10(max_T), num_thresholds)
        
        roc_data = []

        # 3. Sweep thresholds and apply pipeline matching per-clip
        for gamma in thresholds:
            tp_hs, fp_hs, p_hs = 0, 0, 0
            tp_fo, fp_fo, p_fo = 0, 0, 0
            
            for clip in clips_data:
                T = clip['T']
                time = clip['time']
                W = max(2, int(clip['fs'] * 0.04))
                
                # Raw GLRT decision
                raw_mask = T < gamma
                
                # Centralized cleanup
                _, hs_pred, fo_pred = clean_raw_zupt_mask(raw_mask, time, W, min_spacing_ms=600, to_fo_delay_ms = 150)
                
                # Match HS Events
                matched_hs = self._match_events(clip['true_hs'], hs_pred)
                tp_hs += sum(1 for m in matched_hs if m['classification'] == 'TP')
                fp_hs += sum(1 for m in matched_hs if m['classification'] == 'FP')
                p_hs += len(clip['true_hs'])
                
                # Match FO Events
                matched_fo = self._match_events(clip['true_fo'], fo_pred)
                tp_fo += sum(1 for m in matched_fo if m['classification'] == 'TP')
                fp_fo += sum(1 for m in matched_fo if m['classification'] == 'FP')
                p_fo += len(clip['true_fo'])

            # Store HS Metrics
            pd_hs = tp_hs / p_hs if p_hs > 0 else 0
            pfa_hs = fp_hs / p_hs if p_hs > 0 else 0
            roc_data.append({'PFA': pfa_hs, 'PD': pd_hs, 'Threshold': gamma, 'Event': 'HS'})
            
            # Store FO Metrics
            pd_fo = tp_fo / p_fo if p_fo > 0 else 0
            pfa_fo = fp_fo / p_fo if p_fo > 0 else 0
            roc_data.append({'PFA': pfa_fo, 'PD': pd_fo, 'Threshold': gamma, 'Event': 'FO'})

        # 4. Plot interactive split ROC
        roc_df = pd.DataFrame(roc_data)
        fig = px.line(roc_df, x='PFA', y='PD', color='Event', hover_data=['Threshold'], markers=True, 
                      title=f'Event-Based ROC Curve: {detector.upper()} (Filtered)')
        
    # --- NEW: OVERLAY KIELMAT OPERATING POINTS ---
        if not self.df_all_steps.empty:
            # Assumes your folder is named exactly "kielmat"
            kielmat_steps = self.df_all_steps[self.df_all_steps['detector'] == 'kielmat'] 
            
            for event in ['HS', 'FO']:
                km_ev = kielmat_steps[kielmat_steps['event_type'] == event]
                if not km_ev.empty:
                    tp = (km_ev['classification'] == 'TP').sum()
                    fp = (km_ev['classification'] == 'FP').sum()
                    fn = (km_ev['classification'] == 'FN').sum()
                    
                    p = tp + fn
                    pd_val = tp / p if p > 0 else 0
                    pfa_val = fp / p if p > 0 else 0
                    
                    # Define distinct colors for HS and FO to match the line plot
                    marker_color = '#1f77b4' if event == 'HS' else '#ef553b' 
                    
                    fig.add_scatter(
                        x=[pfa_val], 
                        y=[pd_val],
                        mode='markers+text',
                        marker=dict(size=16, symbol='star', color=marker_color, line=dict(width=2, color='black')),
                        text=[f"KielMAT {event}"],
                        textposition="top center",
                        name=f"KielMAT {event} (Fixed)"
                    )
        # ---------------------------------------------

        # Adjust axes to reflect the event-based metrics
        fig.update_xaxes(title="False Alarm Rate (False Positives per True Event)", range=[0, 0.5]) 
        fig.update_yaxes(title="Probability of Detection (Recall)", range=[0, 1.0])
        fig.show()

    def plot_threshold_tuning(self, detector: str = 'shoe', num_thresholds: int = 80):
        """Generates a Precision-Recall-F1 vs Threshold tuning curve for event detection."""
        
        if detector != 'shoe':
            print("Tuning generation currently requires raw T statistics exported (only implemented for 'shoe').")
            return
            
        print(f"Loading datasets and sweeping thresholds for {detector.upper()}...")
        
        valid_filenames = None
        if self.filtered_df is not None and not self.filtered_df.empty:
            df_det = self.filtered_df[self.filtered_df['detector'] == detector]
            if not df_det.empty:
                df_hs = df_det[df_det['event_type'] == 'HS']
                df_fo = df_det[df_det['event_type'] == 'FO']
                hs_files = set(f"{row['subject']}_{row['course']}_{row['clip_id']}.mat" for _, row in df_hs.iterrows())
                fo_files = set(f"{row['subject']}_{row['course']}_{row['clip_id']}.mat" for _, row in df_fo.iterrows())
                valid_filenames = hs_files.intersection(fo_files)
                print(f"Filter active: Tuning using only the {len(valid_filenames)} strictly retained clips.")
            else:
                return
        
        clips_data = []
        mat_files = list((self.data_dir / detector).rglob("*.mat"))
        
        for file_path in mat_files:
            if valid_filenames is not None and file_path.name not in valid_filenames: continue
            mat = loadmat(str(file_path), squeeze_me=True, struct_as_record=False)
            res = mat['results']
            if not hasattr(res, 'T_l') or len(res.T_l) == 0: continue
                
            t_raw = res.time
            fs = 1000.0 / np.mean(np.diff(t_raw)) 
            
            clips_data.append({
                'T': res.T_l,
                'time': t_raw,
                'true_hs': self._split_feet(res.y_HS)[1],
                'true_fo': self._split_feet(res.y_FO)[1],
                'fs': fs
            })

        if not clips_data: return

        all_T = np.concatenate([c['T'] for c in clips_data])
        min_T, max_T = np.percentile(all_T, 1), np.percentile(all_T, 99)
        thresholds = np.logspace(np.log10(max(min_T, 1e-5)), np.log10(max_T), num_thresholds)
        
        tuning_data = []

        for gamma in thresholds:
            tp_hs, fp_hs, p_hs = 0, 0, 0
            tp_fo, fp_fo, p_fo = 0, 0, 0
            
            for clip in clips_data:
                W = max(2, int(clip['fs'] * 0.04))
                raw_mask = clip['T'] < gamma
                
                _, hs_pred, fo_pred = clean_raw_zupt_mask(raw_mask, clip['time'], W, min_spacing_ms=600, to_fo_delay_ms = 150)
                
                matched_hs = self._match_events(clip['true_hs'], hs_pred)
                tp_hs += sum(1 for m in matched_hs if m['classification'] == 'TP')
                fp_hs += sum(1 for m in matched_hs if m['classification'] == 'FP')
                p_hs += len(clip['true_hs'])
                
                matched_fo = self._match_events(clip['true_fo'], fo_pred)
                tp_fo += sum(1 for m in matched_fo if m['classification'] == 'TP')
                fp_fo += sum(1 for m in matched_fo if m['classification'] == 'FP')
                p_fo += len(clip['true_fo'])

            # Calculate HS Metrics
            prec_hs = tp_hs / (tp_hs + fp_hs) if (tp_hs + fp_hs) > 0 else 0
            rec_hs = tp_hs / p_hs if p_hs > 0 else 0
            f1_hs = 2 * (prec_hs * rec_hs) / (prec_hs + rec_hs) if (prec_hs + rec_hs) > 0 else 0
            
            tuning_data.append({'Threshold': gamma, 'Metric': 'Precision', 'Value': prec_hs, 'Event': 'HS'})
            tuning_data.append({'Threshold': gamma, 'Metric': 'Recall', 'Value': rec_hs, 'Event': 'HS'})
            tuning_data.append({'Threshold': gamma, 'Metric': 'F1-Score', 'Value': f1_hs, 'Event': 'HS'})

            # Calculate FO Metrics
            prec_fo = tp_fo / (tp_fo + fp_fo) if (tp_fo + fp_fo) > 0 else 0
            rec_fo = tp_fo / p_fo if p_fo > 0 else 0
            f1_fo = 2 * (prec_fo * rec_fo) / (prec_fo + rec_fo) if (prec_fo + rec_fo) > 0 else 0
            
            tuning_data.append({'Threshold': gamma, 'Metric': 'Precision', 'Value': prec_fo, 'Event': 'FO'})
            tuning_data.append({'Threshold': gamma, 'Metric': 'Recall', 'Value': rec_fo, 'Event': 'FO'})
            tuning_data.append({'Threshold': gamma, 'Metric': 'F1-Score', 'Value': f1_fo, 'Event': 'FO'})

        df_tuning = pd.DataFrame(tuning_data)
        
        # Plotting
        fig = px.line(df_tuning, x='Threshold', y='Value', color='Metric', facet_row='Event',
                      log_x=True, title=f'Detector Tuning Curve: {detector.upper()}',
                      color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c'])
        
        fig.update_layout(hovermode="x unified")
        fig.update_yaxes(range=[0, 1.05])
        fig.show()

if __name__ == "__main__":
    TARGET_SURFACE = "stairs_up"  # Change this to your target surface if needed
    TARGET_DATASET = 'newbee'  # Change this to your target dataset if needed
    EXCLUDE_WORST_N = 0  # Number of worst segments to drop for summary statistics

    # Argparse
    parser = argparse.ArgumentParser(description="Gait Analysis Pipeline")
    parser.add_argument('-d', '--detector', type=str, help="Specify detector to visualize extremes (e.g., -d shoe, -d ared)")
    args = parser.parse_args()

    # Path to .mat files
    STAIRS_DIRECTORY = Path(__file__).resolve().parent.parent / "data" / f"{TARGET_DATASET}_{TARGET_SURFACE}" 
    # you can change this to your actual data directory with the surface you insolated
    # eg. WALK_DIRECTORY = Path(__file__).resolve().parent.parent / "data" / "walk"

    pipeline = GaitAnalysisPipeline(data_dir=STAIRS_DIRECTORY, tolerance_ms=300.0)
    pipeline.parse_and_match()
    pipeline.generate_summary_table_without_worst(EXCLUDE_WORST_N)
    
    # Plotting Logic
    if args.detector:
        print(f"\nGenerating diagnostic extreme plots for '{args.detector}' detector...")
        pipeline.visualize_extremes(detector=args.detector.lower(), metric='f1_score', event='HS')
        pipeline.visualize_extremes(detector=args.detector.lower(), metric='f1_score', event='FO')
    else:
        # Default global visualization if no command line switch is provided
        pipeline.visualize()
        pipeline.generate_roc_curve()
        pipeline.plot_threshold_tuning()