import os
import re
import warnings
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
        """
        Initializes the analysis pipeline.
        
        Parameters:
        -----------
        data_dir : str
            The root directory containing the processed .mat files (e.g., 'data/toolbox1').
        tolerance_ms : float
            The time window (± ms) to consider a predicted step a True Positive.
        """
        self.data_dir = Path(data_dir)
        self.tolerance_ms = tolerance_ms
        self.df_all_steps = pd.DataFrame()
        self.df_all_segments = pd.DataFrame()

    def _extract_timestamps(self, array) -> np.ndarray:
        """Safely extracts the first column (time) from MATLAB array structures."""
        if array is None or (isinstance(array, (list, np.ndarray)) and len(array) == 0):
            return np.array([])
        array = np.asarray(array)
        # Handle cases where array is flat or 2D
        if array.ndim == 2 and array.shape[1] >= 1:
            return array[:, 0].astype(float)
        return array.flatten().astype(float)

    def _match_events(self, t_true: np.ndarray, t_pred: np.ndarray) -> list:
        """
        Matches predicted timestamps to ground-truth using optimal bipartite matching.
        """
        step_records = []
        
        # Handle edge cases where one or both arrays are empty
        if len(t_true) == 0 and len(t_pred) == 0:
            return step_records
        if len(t_true) == 0:
            return [{'error_ms': np.nan, 'classification': 'FP'} for _ in t_pred]
        if len(t_pred) == 0:
            return [{'error_ms': np.nan, 'classification': 'FN'} for _ in t_true]

        # Calculate absolute difference cost matrix
        diff_matrix = np.abs(t_pred[:, None] - t_true[None, :])
        
        # Penalize matches outside the tolerance window so they are never paired
        cost_matrix = diff_matrix.copy()
        cost_matrix[cost_matrix > self.tolerance_ms] = 1e9

        # Optimal bipartite matching
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched_pred = set()
        matched_true = set()

        for p_idx, t_idx in zip(row_ind, col_ind):
            error = t_pred[p_idx] - t_true[t_idx] # t_pred - t_true
            if abs(error) <= self.tolerance_ms:
                step_records.append({'error_ms': error, 'classification': 'TP'})
                matched_pred.add(p_idx)
                matched_true.add(t_idx)

        # Flag remaining predictions as False Positives
        for p_idx in range(len(t_pred)):
            if p_idx not in matched_pred:
                step_records.append({'error_ms': np.nan, 'classification': 'FP'})

        # Flag remaining ground-truths as False Negatives
        for t_idx in range(len(t_true)):
            if t_idx not in matched_true:
                step_records.append({'error_ms': np.nan, 'classification': 'FN'})

        return step_records

    def parse_and_match(self):
        """Locates all .mat files, parses arrays, matches steps, and builds DataFrames."""
        print(f"Scanning for .mat files in: {self.data_dir}...")
        mat_files = list(self.data_dir.rglob("*.mat"))
        
        all_steps = []
        all_segments = []

        # Regex to safely parse metadata: {detector}/{id}_{course}_clip_{clip_id}.mat
        # Adapts based on directory structure assumptions
        for file_path in mat_files:
            try:
                # Extract metadata from paths
                detector = file_path.parent.name
                file_stem = file_path.stem
                
                # Assuming format: id_course_clip_X or similar
                # Using a generic split to capture identifiers
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

                # Load MATLAB file
                mat = loadmat(str(file_path), squeeze_me=True, struct_as_record=False)
                if 'results' not in mat:
                    continue
                
                res = mat['results']
                
                # Dictionary mapping event names to structure properties
                events = {
                    'HS': (getattr(res, 'y_HS', []), getattr(res, 'y_hat_HS', [])),
                    'FO': (getattr(res, 'y_FO', []), getattr(res, 'y_hat_FO', []))
                }

                for event_type, (true_arr, pred_arr) in events.items():
                    t_true = self._extract_timestamps(true_arr)
                    t_pred = self._extract_timestamps(pred_arr)

                    # Match events
                    matched_records = self._match_events(t_true, t_pred)
                    
                    # Store step-level records
                    for rec in matched_records:
                        rec.update({
                            'detector': detector,
                            'subject': subject_id,
                            'course': course,
                            'clip_id': clip_id,
                            'event_type': event_type
                        })
                        all_steps.append(rec)

                    # Calculate Segment-level metrics
                    df_temp = pd.DataFrame(matched_records)
                    if not df_temp.empty:
                        tp = (df_temp['classification'] == 'TP').sum()
                        fp = (df_temp['classification'] == 'FP').sum()
                        fn = (df_temp['classification'] == 'FN').sum()
                        
                        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                        
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
                            'mean_absolute_error_ms': mae,
                            'std_error_ms': std_err
                        })

            except Exception as e:
                print(f"Error processing {file_path.name}: {str(e)}")

        self.df_all_steps = pd.DataFrame(all_steps)
        self.df_all_segments = pd.DataFrame(all_segments)
        print(f"Processed {len(self.df_all_segments)} segment configurations resulting in {len(self.df_all_steps)} total steps evaluated.")

    def generate_summary_table(self):
        """Calculates and prints a clean markdown summary table grouped by detector."""
        if self.df_all_segments.empty:
            print("No data available to summarize.")
            return

        print("\n### Detector Performance Summary")
        
        # Calculate summary statistics across all clips for each detector
        summary = self.df_all_segments.groupby(['detector', 'event_type']).agg(
            f1_mean=('f1_score', 'mean'),
            f1_std=('f1_score', 'std'),
            precision_mean=('precision', 'mean'),
            recall_mean=('recall', 'mean'),
            mae_mean=('mean_absolute_error_ms', 'mean'),
            error_std_mean=('std_error_ms', 'mean')
        ).round(3)

        print(summary.to_markdown())
        return summary

    def visualize(self):
        """Generates the three requested analytical plots using Seaborn."""
        if self.df_all_steps.empty:
            print("No data to visualize.")
            return

        sns.set_theme(style="whitegrid", palette="muted")
        
        # ---------------------------------------------------------
        # Plot 1: Timing Error Distribution (Hist + KDE)
        # ---------------------------------------------------------
        plt.figure(figsize=(10, 6))
        tp_steps = self.df_all_steps[self.df_all_steps['classification'] == 'TP']
        sns.histplot(
            data=tp_steps, 
            x='error_ms', 
            hue='detector', 
            kde=True, 
            element="step", 
            stat="density", 
            common_norm=False,
            alpha=0.3
        )
        plt.title('Distribution of Timing Errors ($t_{pred} - t_{true}$)', fontsize=14)
        plt.xlabel('Timing Error (ms)', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.axvline(0, color='black', linestyle='--', linewidth=1.5)
        plt.tight_layout()
        plt.show()

        # ---------------------------------------------------------
        # Plot 2: Global Confusion Matrix Counts
        # ---------------------------------------------------------
        plt.figure(figsize=(10, 6))
        sns.countplot(
            data=self.df_all_steps, 
            x='detector', 
            hue='classification', 
            palette={'TP': '#2ca02c', 'FP': '#d62728', 'FN': '#ff7f0e'}
        )
        plt.title('Global Step Classifications (TP, FP, FN) by Detector', fontsize=14)
        plt.xlabel('Detector Algorithm', fontsize=12)
        plt.ylabel('Total Count', fontsize=12)
        plt.legend(title='Classification')
        plt.tight_layout()
        plt.show()

        # ---------------------------------------------------------
        # Plot 3: Consistency Boxplots (F1-Score and MAE per segment)
        # ---------------------------------------------------------
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # F1 Score Consistency
        sns.boxplot(
            data=self.df_all_segments, 
            x='detector', 
            y='f1_score', 
            hue='event_type', 
            ax=axes[0]
        )
        axes[0].set_title('Consistency: F1-Score per Staircase Segment', fontsize=14)
        axes[0].set_ylabel('F1-Score (0.0 to 1.0)')
        axes[0].set_xlabel('Detector')
        axes[0].set_ylim(-0.05, 1.05)

        # MAE Consistency
        sns.boxplot(
            data=self.df_all_segments, 
            x='detector', 
            y='mean_absolute_error_ms', 
            hue='event_type', 
            ax=axes[1]
        )
        axes[1].set_title('Consistency: Mean Absolute Timing Error (MAE)', fontsize=14)
        axes[1].set_ylabel('MAE (ms)')
        axes[1].set_xlabel('Detector')

        plt.tight_layout()
        plt.show()

# ==========================================
# Execution Block
# ==========================================
if __name__ == "__main__":
    # Point this to your root directory containing the /toolbox1/ foldertree
    STAIRS_DIRECTORY = Path(__file__).resolve().parent.parent / "data" / "stairs"
    WALK_DIRECTORY = Path(__file__).resolve().parent.parent / "data" / "walk"
    
    # Initialize pipeline with a 50ms acceptance window
    pipeline = GaitAnalysisPipeline(data_dir=WALK_DIRECTORY, tolerance_ms=300.0)
    
    # 1. Parse Data & Match Steps
    pipeline.parse_and_match()
    
    # 2. Output Statistics
    pipeline.generate_summary_table()
    
    # 3. Generate Visualizations
    pipeline.visualize()