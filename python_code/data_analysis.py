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

    def generate_summary_table(self):
        if self.df_all_segments.empty:
            print("No data available to summarize.")
            return

        print("\n### Detector Performance Summary (Prioritizing F1-Score)")
        
        # Aggregating F1 mean/std and F2 mean
        summary = self.df_all_segments.groupby(['detector', 'event_type']).agg(
            f1_mean=('f1_score', 'mean'),
            f1_std=('f1_score', 'std'),          # Re-adding F1 standard deviation
            f2_mean=('f2_score', 'mean'),
            precision_mean=('precision', 'mean'),
            precision_std=('precision', 'std'),
            recall_mean=('recall', 'mean'),
            mae_mean=('mean_absolute_error_ms', 'mean'),
            mae_std=('mean_absolute_error_ms', 'std')
        ).round(3)

        print(summary.to_markdown())
        return summary

    def visualize(self):
        if self.df_all_steps.empty: return
        sns.set_theme(style="whitegrid", palette="muted")
        
        plt.figure(figsize=(10, 6))
        tp_steps = self.df_all_steps[self.df_all_steps['classification'] == 'TP']
        sns.histplot(data=tp_steps, x='error_ms', hue='detector', kde=True, element="step", alpha=0.3)
        plt.title('Distribution of Timing Errors ($t_{pred} - t_{true}$)', fontsize=14)
        plt.xlabel('Timing Error (ms)')
        plt.show()

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        sns.boxplot(
            data=self.df_all_segments, 
            x='detector', 
            y='f1_score', # Changed from f2_score
            hue='event_type', 
            ax=axes[0]
        )
        axes[0].set_title('Consistency: F1-Score per Staircase Segment', fontsize=14)
        axes[0].set_ylabel('F1-Score (0.0 to 1.0)')
        
        sns.boxplot(data=self.df_all_segments, x='detector', y='mean_absolute_error_ms', hue='event_type', ax=axes[1])
        axes[1].set_title('Consistency: Mean Absolute Timing Error (MAE)')
        plt.show()

    def visualize_extremes(self, detector: str, metric: str = 'f1_score', event: str = 'FO'):
        """Plots raw acceleration, true steps, and predicted steps for best, avg, worst segments."""
        df_det = self.df_all_segments[(self.df_all_segments['detector'] == detector) & (self.df_all_segments['event_type'] == event)]
        
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
                if not hasattr(res, 'time') or not hasattr(res, 'acc_mag_l'):
                    ax.set_title(f"{label}: {file_name} — [MISSING RAW ACCELERATION]")
                    ax.text(0.5, 0.5, "Raw 'time' and 'acc_mag_l' missing from .mat file.\nUpdate pyshoe_export.py.", ha='center', va='center')
                    continue
                
                t_raw = res.time
                acc_l = res.acc_mag_l
                
                # Plot the raw acceleration curve (Left foot as an example)
                ax.plot(t_raw, acc_l, label='Left Accel Magnitude', color='#d3d3d3', linewidth=1.5)
                
                # Extract true vs predicted events
                true_events = getattr(res, f'y_{event}')
                pred_events = getattr(res, f'y_hat_{event}')
                true_r, true_l = self._split_feet(true_events)
                pred_r, pred_l = self._split_feet(pred_events)
                
                # Overlay True (Left)
                if len(true_l) > 0:
                    ax.vlines(true_l, ymin=min(acc_l), ymax=max(acc_l), color='#2ca02c', linestyle='-', linewidth=2, alpha=0.8, label='True Event')
                
                # Overlay Pred (Left)
                if len(pred_l) > 0:
                    ax.vlines(pred_l, ymin=min(acc_l), ymax=max(acc_l), color='#d62728', linestyle='--', linewidth=2, alpha=0.8, label='Predicted Event')

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


if __name__ == "__main__":
    # 1. Add Argparse
    parser = argparse.ArgumentParser(description="Gait Analysis Pipeline")
    parser.add_argument('-d', '--detector', type=str, help="Specify detector to visualize extremes (e.g., -d shoe, -d ared)")
    args = parser.parse_args()

    # 2. Existing Data Processing
    STAIRS_DIRECTORY = Path(__file__).resolve().parent.parent / "data" / "stairs_up"
    WALK_DIRECTORY = Path(__file__).resolve().parent.parent / "data" / "walk"
    AMBULATION_STAIRS = Path(__file__).resolve().parent.parent / "data" / "ambulation_out"

    pipeline = GaitAnalysisPipeline(data_dir=AMBULATION_STAIRS, tolerance_ms=300.0)
    pipeline.parse_and_match()
    pipeline.generate_summary_table()
    
    # 3. Dynamic Plotting Logic
    if args.detector:
        print(f"\nGenerating diagnostic extreme plots for '{args.detector}' detector...")
        pipeline.visualize_extremes(detector=args.detector.lower(), metric='f2_score', event='HS')
        pipeline.visualize_extremes(detector=args.detector.lower(), metric='f1_score', event='FO')
    else:
        # Default global visualization if no command line switch is provided
        pipeline.visualize()