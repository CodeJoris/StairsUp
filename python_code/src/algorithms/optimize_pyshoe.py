"""
Hyperparameter optimization for PyShoe zero-velocity thresholds (G)
Target: optimize G for 'stairs_up' segments using LOSO cross-validation.

Run from the repository `python_code/` root as:

    python -m src.algorithms.optimize_pyshoe

Use the -d switch to only optimize one detector

    python -m src.algorithms.optimize_pyshoe -d mbgtd

"""
from __future__ import annotations
import sys
import traceback
import tempfile
from pathlib import Path
import json
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
import os
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from python_code.src.preprocessing.extractStairs import get_stair_segments
from python_code.src.preprocessing.extractGoldenStandard import (
    extract_golden_standard,
    validate_golden_standard_input,
)
from python_code.Toolboxes.PyShoe import pyshoe_export
from python_code.Toolboxes.PyShoe.ins_tools.INS import INS
from concurrent.futures import ProcessPoolExecutor, as_completed

DATA_PATH = ROOT / "data"
TOL_MS = 150
STATIONARY_BUFFER_MS = 2000
MAX_PREDICTED_EVENTS = 300
DATA_SET_ANCHOR = "data_set"
CLIPS_CACHE_PATH = Path(__file__).resolve().parent / "stair_clips_cache.pkl"
PLOTS_ON = True
MIN_STANCE_MS = 350.0  # Safe threshold (real stance is ~600ms+) tweak based on sampling frequency this was made to be more than 15 frames at 60Hz

REQUIRED_LABEL_COLS = [
    "time",
    "insoles_RightFoot_is_step",
    "insoles_RightFoot_is_lifted",
    "insoles_LeftFoot_is_step",
    "insoles_LeftFoot_is_lifted",
    "walk_mode",
]
REQUIRED_SENSOR_COLS = ["time"] + pyshoe_export.LEFT_FOOT_COLS + pyshoe_export.RIGHT_FOOT_COLS

CHECKPOINT_PATH = Path(__file__).resolve().parent / "pyshoe_optimization_checkpoint.json"
SUMMARY_PATH = Path(__file__).resolve().parent / "pyshoe_optimized_specs_stairs.json"

def plot_predictions_vs_truth(clip_df, y_HS_true, y_HS_pred, y_FO_true, y_FO_pred, G_val, course, sid, detector):
    """
    Overlays predicted gait events onto raw IMU data to visually debug F1 scores.
    """
    plt.figure(figsize=(14, 6))
    
    # Plot a reference sensor signal (Right Foot Z-axis acceleration)
    time = clip_df['time']
    if 'acceleration_RightFoot_z' in clip_df.columns:
        acc_z = clip_df['acceleration_RightFoot_z']
        plt.plot(time, acc_z, label='Right Foot Accel Z', color='lightgray', alpha=0.8)
        
    # Safely extract 1D time arrays (assuming column 0 is time in ms)
    hs_true = y_HS_true[:, 0] if y_HS_true.ndim == 2 else y_HS_true
    hs_pred = y_HS_pred[:, 0] if y_HS_pred.ndim == 2 else y_HS_pred
    fo_true = y_FO_true[:, 0] if y_FO_true.ndim == 2 else y_FO_true
    fo_pred = y_FO_pred[:, 0] if y_FO_pred.ndim == 2 else y_FO_pred

    # Plot Ground Truth (Solid Lines)
    for t in hs_true:
        plt.axvline(x=t, color='green', linestyle='-', linewidth=0.5, alpha = 0.7,
                    label='True HS' if t == hs_true[0] else "")
    for t in fo_true:
        plt.axvline(x=t, color='blue', linestyle='-', linewidth=0.5, alpha = 0.7,
                    label='True FO' if t == fo_true[0] else "")

    # Plot Predictions (Dashed Lines)
    for t in hs_pred:
        plt.axvline(x=t, color='limegreen', linestyle='--', linewidth=0.5, alpha = 0.7,
                    label='Pred HS' if len(hs_pred) > 0 and t == hs_pred[0] else "")
    for t in fo_pred:
        plt.axvline(x=t, color='dodgerblue', linestyle='--', linewidth=0.5, alpha = 0.7,
                    label='Pred FO' if len(fo_pred) > 0 and t == fo_pred[0] else "")

    plt.title(f"PyShoe Predictions vs. Truth (G = {G_val:.3f})", fontsize=14)
    plt.xlabel('Time (ms)', fontsize=12)
    plt.ylabel('Acceleration (m/s²)', fontsize=12)
    
    # Deduplicate legend labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right')
    
    plt.tight_layout()
    file_name = f"{course}_{sid}_G_{G_val:.3f}.pdf"
    save_path = Path(__file__).resolve().parent / "plots" / detector / file_name
    plt.savefig(save_path)
    plt.close('all')

def atomic_write_json(path: Path, payload: dict) -> None:
    """
    Write JSON atomically via a temp file in the same directory.

    Uses ``os.replace`` so readers never see a partially written file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        suffix=".json.tmp", prefix=f"{path.stem}_", dir=str(path.parent)
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


def task_key(detector: str, fold_key: str) -> str:
    """Build a stable identifier for a detector/fold optimization task."""
    return f"{detector}|{fold_key}"


def parse_course_subject_from_xsens_path(path: Path) -> Tuple[str, str]:
    """
    Extract course and subject id from an xsens.csv path.

    Expected layout: ``.../data_set/<course>/<subject>/xsens.csv``.

    Raises:
        ValueError: When the path does not match the expected layout.
    """
    parts = path.parts
    try:
        anchor_idx = parts.index(DATA_SET_ANCHOR)
    except ValueError as exc:
        raise ValueError(
            f"Sensor path must contain '{DATA_SET_ANCHOR}' segment: {path}"
        ) from exc
    if len(parts) < anchor_idx + 4:
        raise ValueError(f"Sensor path too short after '{DATA_SET_ANCHOR}': {path}")
    course = parts[anchor_idx + 1]
    sid = parts[anchor_idx + 2]
    if parts[anchor_idx + 3] != "xsens.csv":
        raise ValueError(f"Expected xsens.csv after subject folder: {path}")
    return course, sid


def validate_csv_columns(df: pd.DataFrame, required: List[str], context: str) -> None:
    """Raise ValueError when required columns are absent."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{context}: missing columns {missing}")


def validate_clip_payload(clip_meta: dict, context: str = "clip") -> None:
    """
    Validate a clip dict before expensive INS evaluation.

    Raises:
        ValueError: When required keys, shapes, or column sets are invalid.
    """
    for key in ("sensor_df", "y_HS", "y_FO"):
        if key not in clip_meta:
            raise ValueError(f"{context}: missing key '{key}'")

    sensor_df = clip_meta["sensor_df"]
    if not isinstance(sensor_df, pd.DataFrame) or sensor_df.empty:
        raise ValueError(f"{context}: sensor_df must be a non-empty DataFrame")
    validate_csv_columns(sensor_df, REQUIRED_SENSOR_COLS, f"{context}.sensor_df")

    for arr_key in ("y_HS", "y_FO"):
        arr = clip_meta[arr_key]
        if not isinstance(arr, np.ndarray):
            raise ValueError(f"{context}: {arr_key} must be a numpy array")
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(
                f"{context}: {arr_key} must have shape (N, 2), got {arr.shape}"
            )


def list_all_subject_files() -> List[Path]:
    """
    Find all ``xsens.csv`` files under ``DATA_PATH``.

    Returns:
        Sorted list of sensor CSV paths (may be empty).
    """
    files = sorted(DATA_PATH.rglob("xsens.csv"))
    print(f"Found {len(files)} xsens.csv files under {DATA_PATH}")
    return files


def get_stair_clips_for_subject(course: str, sid: str) -> List[dict]:
    """
    Extract stair-up clip ground truth for one subject/course pair.

    Reads ``labels.csv``, finds continuous ``stairs_up`` blocks, extracts
    golden-standard HS/FO arrays, and trims only clearly incomplete gait
    cycles at segment boundaries (FO before first HS, HS after last FO).

    Args:
        course: Course folder under ``data_set`` (e.g. ``courseA``).
        sid: Subject id folder (e.g. ``id01``).

    Returns:
        List of clip dicts with keys ``segment_id``, ``start_time``,
        ``end_time``, ``y_HS``, ``y_FO``. Empty when no valid segments exist.

    Raises:
        FileNotFoundError: If labels.csv is missing.
        ValueError: If required label columns are missing.
    """
    file_path = DATA_PATH / DATA_SET_ANCHOR / course / sid / "labels.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"Labels file not found: {file_path}")

    stair_blocks = get_stair_segments(str(file_path), target_mode="stairs_up")
    if stair_blocks.empty:
        return []

    print(f"Loading labels for {course}/{sid}: {file_path}")
    data = pd.read_csv(file_path)
    validate_csv_columns(data, REQUIRED_LABEL_COLS, str(file_path))
    validate_golden_standard_input(data)

    y_HS_full, y_FO_full = extract_golden_standard(data)

    clips = []
    for idx, block in stair_blocks.iterrows():
        start_buffered = max(0, block.start_time - STATIONARY_BUFFER_MS)
        mask_HS = (y_HS_full[:, 0] >= start_buffered) & (y_HS_full[:, 0] <= block.end_time)
        mask_FO = (y_FO_full[:, 0] >= start_buffered) & (y_FO_full[:, 0] <= block.end_time)

        y_HS_clip = y_HS_full[mask_HS]
        y_FO_clip = y_FO_full[mask_FO]

        if len(y_FO_clip) > 0 and len(y_HS_clip) > 0 and y_FO_clip[0, 0] < y_HS_clip[0, 0]:
            y_FO_clip = y_FO_clip[1:]

        if len(y_HS_clip) > 0 and len(y_FO_clip) > 0 and y_HS_clip[-1, 0] > y_FO_clip[-1, 0]:
            y_HS_clip = y_HS_clip[:-1]

        if len(y_HS_clip) == 0 or len(y_FO_clip) == 0:
            print(f"  Skipping segment {idx} for {sid}_{course}: incomplete cycles")
            continue

        clips.append({
            "segment_id": int(idx),
            "start_time": int(block.start_time),
            "end_time": int(block.end_time),
            "y_HS": y_HS_clip,
            "y_FO": y_FO_clip,
        })

    print(f"  Extracted {len(clips)} valid stairs_up clips for {course}/{sid}")
    return clips


def slice_sensor_with_buffer(raw_df: pd.DataFrame, start_time: int, end_time: int) -> pd.DataFrame:
    """
    Slice sensor data to ``[start_time - buffer, end_time]``.

    Args:
        raw_df: Full xsens dataframe with a ``time`` column (milliseconds).
        start_time: Clip start in milliseconds.
        end_time: Clip end in milliseconds.

    Returns:
        Buffered slice including ``STATIONARY_BUFFER_MS`` before ``start_time``.
    """
    start = max(0, start_time - STATIONARY_BUFFER_MS)
    print(
        f"    Slicing sensor data [{start} .. {end_time}] "
        f"(buffered from {start_time}) -> {len(raw_df)} rows available"
    )
    sliced = raw_df[(raw_df.time >= start) & (raw_df.time <= end_time)].reset_index(drop=True)
    print(f"    Sliced df has {len(sliced)} rows")
    return sliced


def _strip_padding_artifact(event_indices: np.ndarray) -> np.ndarray:
    """
    Remove HS/FO index 0 caused by prepending False before the ZV vector.

    Only index 0 is removed; later events are kept even when the clip
    starts in stance.
    """
    if event_indices.size > 0 and event_indices[0] == 0:
        return event_indices[1:]
    return event_indices


def get_predictions_for_clip(
    clip_df: pd.DataFrame, detector: str, G_val: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run PyShoe INS on a sensor clip and return predicted HS/FO times.

    Args:
        clip_df: Sensor dataframe for one clip (must include IMU columns).
        detector: PyShoe detector name (e.g. ``shoe``, ``ared``).
        G_val: Threshold passed to ``INS.baseline``.

    Returns:
        Tuple of sorted 1D arrays ``(hs_times_ms, fo_times_ms)``.

    Raises:
        ValueError: If required sensor columns are missing.
    """
    validate_csv_columns(clip_df, REQUIRED_SENSOR_COLS, "get_predictions_for_clip")

    right_cols = pyshoe_export.RIGHT_FOOT_COLS
    left_cols = pyshoe_export.LEFT_FOOT_COLS

    imu_left = clip_df[left_cols].to_numpy()
    imu_right = clip_df[right_cols].to_numpy()
    time = clip_df["time"].to_numpy().squeeze()

    print(f"    Predicting clip with detector={detector}, G={G_val:.6g}, rows={len(clip_df)}")

    def events_for_foot(imu: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ins = INS(
            imu,
            sigma_a=0.00098,
            sigma_w=8.7266463e-5,
            T=1.0 / pyshoe_export.SAMPLING_FREQUENCY,
        )
        ins.baseline(W=5, G=G_val, detector=detector)
        padded = np.insert(ins.zv, 0, False).astype(int)
        diff = np.diff(padded)
        hs_idx = _strip_padding_artifact(np.where(diff == 1)[0])
        fo_idx = _strip_padding_artifact(np.where(diff == -1)[0])
        hs_times = time[hs_idx] if hs_idx.size else np.array([])
        fo_times = time[fo_idx] if fo_idx.size else np.array([])

        # Drop lone initial Foot Off (if recording started mid-swing)
        if len(fo_times) > 0 and len(hs_times) > 0:
            if fo_times[0] < hs_times[0]:
                fo_times = fo_times[1:]
                
        # Drop lone trailing Heel Strike (if recording ended in stance)
        if len(hs_times) > 0 and len(fo_times) > 0:
            if hs_times[-1] > fo_times[-1]:
                hs_times = hs_times[:-1]

        # Enforce a Minimum Stance Time to void rapid false-positive flickers
        if len(hs_times) == len(fo_times) and len(hs_times) > 0:
            # Calculate the duration the foot was "on the ground" for each step
            stance_durations = fo_times - hs_times
            
            # Create a boolean mask of only the steps that lasted long enough
            valid_mask = stance_durations >= MIN_STANCE_MS
            
            # Apply the mask to instantly drop the glitches from both arrays
            hs_times = hs_times[valid_mask]
            fo_times = fo_times[valid_mask]

        return hs_times, fo_times

    hs_l, fo_l = events_for_foot(imu_left)
    hs_r, fo_r = events_for_foot(imu_right)

    # Helper to create Nx2 arrays: [time_ms, foot_label]
    def tag_foot(times, label):
        if times.size == 0: return np.empty((0, 2))
        return np.column_stack((times, np.full(len(times), label)))

    hs_l_tagged = tag_foot(hs_l, 1)
    hs_r_tagged = tag_foot(hs_r, 0)
    fo_l_tagged = tag_foot(fo_l, 1)
    fo_r_tagged = tag_foot(fo_r, 0)

    # Combine left and right
    all_HS = np.vstack((hs_l_tagged, hs_r_tagged)) if len(hs_l) or len(hs_r) else np.empty((0, 2))
    all_FO = np.vstack((fo_l_tagged, fo_r_tagged)) if len(fo_l) or len(fo_r) else np.empty((0, 2))

    # Sort sequentially by time (column 0)
    hs_sorted = all_HS[all_HS[:, 0].argsort()] if len(all_HS) else all_HS
    fo_sorted = all_FO[all_FO[:, 0].argsort()] if len(all_FO) else all_FO
    print(f"    Predicted {len(hs_sorted)} HS and {len(fo_sorted)} FO events")
    return hs_sorted, fo_sorted


def hungarian_match(
    true_times: np.ndarray, pred_times: np.ndarray, tol: float = TOL_MS
) -> Tuple[int, int, int]:
    """
    Match predicted event times to ground truth with Hungarian assignment.

    Args:
        true_times: 1D ground-truth timestamps (ms).
        pred_times: 1D predicted timestamps (ms).
        tol: Maximum allowed absolute difference for a match (ms).

    Returns:
        Tuple ``(TP, FP, FN)``.
    """
    true_times = np.asarray(true_times).ravel()
    pred_times = np.asarray(pred_times).ravel()

    if len(pred_times) == 0 and len(true_times) == 0:
        return 0, 0, 0
    if len(pred_times) == 0:
        return 0, 0, len(true_times)
    if len(true_times) == 0:
        return 0, len(pred_times), 0

    cost = np.abs(true_times[:, None] - pred_times[None, :])
    large = 1e9
    cost_masked = np.where(cost <= tol, cost, large)

    row_ind, col_ind = linear_sum_assignment(cost_masked)
    matched = sum(1 for r, c in zip(row_ind, col_ind) if cost_masked[r, c] < large)

    tp = matched
    fp = len(pred_times) - tp
    fn = len(true_times) - tp
    print(f"      Matching: TP={tp}, FP={fp}, FN={fn}")
    return tp, fp, fn


def f1_from_counts(tp: int, fp: int, fn: int) -> float:
    """Compute F1 from confusion counts; returns 0.0 when TP is zero."""
    if tp == 0:
        return 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if prec + rec == 0:
        return 0.0
    return 2 * (prec * rec) / (prec + rec)

def f_beta_from_counts(tp: int, fp: int, fn: int, beta: float = 2.0) -> float:
    """
    Compute the F-beta score. 
    beta > 1 penalizes False Negatives more heavily (emphasizes recall).
    beta = 2 makes recall twice as important as precision.
    """
    if tp == 0:
        return 0.0
        
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    if prec + rec == 0:
        return 0.0
        
    beta_sq = beta ** 2
    return (1 + beta_sq) * (prec * rec) / ((beta_sq * prec) + rec)

def evaluate_G_on_clips(clips: List[dict], detector: str, G_val: float) -> float:
    """
    Aggregate F1 across clips for one detector/threshold pair.

    Args:
        clips: Clip dicts with ``sensor_df``, ``y_HS``, ``y_FO``.
        detector: PyShoe detector name.
        G_val: Threshold value for INS.

    Returns:
        Combined HS+FO F1 score in [0, 1].

    Raises:
        ValueError: When a clip payload is invalid.
    """
    tp_total = fp_total = fn_total = 0
    print(f"  Evaluating detector={detector} G={G_val:.6g} on {len(clips)} clips")

    for i, clip_meta in enumerate(clips, 1):
        validate_clip_payload(clip_meta, context=f"clip {i}")
        clip_df = clip_meta["sensor_df"]

        y_HS_true = clip_meta["y_HS"]
        y_FO_true = clip_meta["y_FO"]
        y_HS_pred, y_FO_pred = get_predictions_for_clip(clip_df, detector, G_val)

        if len(y_HS_pred) > MAX_PREDICTED_EVENTS or len(y_FO_pred) > MAX_PREDICTED_EVENTS:
            return 0.0

        # Helper to split an Nx2 array into Right (0) and Left (1) 1D time arrays
        def split_feet(arr):
            if len(arr) == 0: return np.array([]), np.array([])
            return arr[arr[:, 1] == 0][:, 0], arr[arr[:, 1] == 1][:, 0]

        true_hs_r, true_hs_l = split_feet(y_HS_true)
        true_fo_r, true_fo_l = split_feet(y_FO_true)
        pred_hs_r, pred_hs_l = split_feet(y_HS_pred)
        pred_fo_r, pred_fo_l = split_feet(y_FO_pred)

        # Only plot the very first clip of a specific G value to avoid spamming 100 plots
        if i == 1 and PLOTS_ON:
            plot_predictions_vs_truth(
                clip_df, 
                true_hs_r, pred_hs_r, 
                true_fo_r, pred_fo_r, 
                G_val, clip_meta["course"], clip_meta["id"], detector
            )
        # ----------------------------

        if len(y_HS_pred) > MAX_PREDICTED_EVENTS or len(y_FO_pred) > MAX_PREDICTED_EVENTS:
            print(
                f"    [WARNING] G={G_val:.6g} exceeded event cap "
                f"({len(y_HS_pred)} HS, {len(y_FO_pred)} FO). Rejecting threshold."
            )
            return 0.0

        # Match Right Foot
        tp_hs_r, fp_hs_r, fn_hs_r = hungarian_match(true_hs_r, pred_hs_r, tol=TOL_MS)
        tp_fo_r, fp_fo_r, fn_fo_r = hungarian_match(true_fo_r, pred_fo_r, tol=TOL_MS)
        
        # Match Left Foot
        tp_hs_l, fp_hs_l, fn_hs_l = hungarian_match(true_hs_l, pred_hs_l, tol=TOL_MS)
        tp_fo_l, fp_fo_l, fn_fo_l = hungarian_match(true_fo_l, pred_fo_l, tol=TOL_MS)

        # Combine totals
        tp_total += tp_hs_r + tp_fo_r + tp_hs_l + tp_fo_l
        fp_total += fp_hs_r + fp_fo_r + fp_hs_l + fp_fo_l
        fn_total += fn_hs_r + fn_fo_r + fn_hs_l + fn_fo_l

        print(
            f"    Clip {i}/{len(clips)}: "
            f"TP={(tp_hs_r + tp_fo_r + tp_hs_l + tp_fo_l)}, FP={(fp_hs_r + fp_fo_r + fp_hs_l + fp_fo_l)}, FN={(fn_hs_r + fn_fo_r + fn_hs_l + fn_fo_l)}"
        )

    f1 = f_beta_from_counts(tp_total, fp_total, fn_total, beta=2.0)
    print(f"  -> Aggregated TP={tp_total}, FP={fp_total}, FN={fn_total}, F1={f1:.3f}")
    return f1


def build_all_clips_with_sensor() -> List[dict]:
    """
    Build stairs_up clips with buffered sensor slices attached.

    Per-file failures are logged and skipped so one bad subject does not
    abort the entire run.

    Returns:
        List of clip dicts ready for LOSO optimization.
    """
    # --- CHECK FOR CACHE ---
    if CLIPS_CACHE_PATH.exists():
        print(f"Loading cached clips from {CLIPS_CACHE_PATH}...")
        with open(CLIPS_CACHE_PATH, "rb") as f:
            return pickle.load(f)
    # -----------------------

    clips: List[dict] = []
    xsens_files = list_all_subject_files()
    print(f"Building clips from {len(xsens_files)} sensor files")

    for f in xsens_files:
        try:
            course, sid = parse_course_subject_from_xsens_path(f)
        except ValueError as exc:
            print(f" Skipping {f}: {exc}")
            continue

        print(f" Processing file: {course}/{sid} -> {f}")
        try:
            stair_clips = get_stair_clips_for_subject(course, sid)
            if not stair_clips:
                print(f"   No stair clips for {course}/{sid}")
                continue

            raw_df = pd.read_csv(f)
            validate_csv_columns(raw_df, REQUIRED_SENSOR_COLS, str(f))

            for c in stair_clips:
                c_copy = c.copy()
                c_copy["course"] = course
                c_copy["id"] = sid
                c_copy["sensor_df"] = slice_sensor_with_buffer(
                    raw_df, c_copy["start_time"], c_copy["end_time"]
                )
                validate_clip_payload(c_copy, context=f"{course}/{sid} segment {c_copy['segment_id']}")
                clips.append(c_copy)
        except Exception as exc:
            print(f"   Failed to build clips for {course}/{sid}: {exc}")
            traceback.print_exc()

    print(f"Built total {len(clips)} clips with sensor data attached")

    # --- SAVE TO CACHE ---
    print(f"Saving clips to cache at {CLIPS_CACHE_PATH}...")
    with open(CLIPS_CACHE_PATH, "wb") as f:
        pickle.dump(clips, f)
    # ---------------------

    return clips


def build_loso_folds(all_clips: List[dict]) -> List[dict]:
    """
    Build leave-one-subject-out folds with deterministic subject ordering.

    Returns:
        List of dicts with keys ``train``, ``test``, ``key`` (subject id).
    """
    subjects = sorted({c["id"] for c in all_clips})
    folds = []
    for sid in subjects:
        train_clips = [c for c in all_clips if c["id"] != sid]
        test_clips = [c for c in all_clips if c["id"] == sid]
        folds.append({"train": train_clips, "test": test_clips, "key": sid})
    return folds


def _grid_search_threshold(
    detector: str,
    training_clips: List[dict],
    search_values: np.ndarray,
    to_G,
) -> Tuple[float, float, List[Tuple[float, float]]]:
    """
    Evaluate a grid of candidate thresholds and return the best F1.

    Returns:
        ``(best_G, best_f1, history)`` where history is ``[(G, f1), ...]``.
    """
    best_f1 = -1.0
    best_G = float(to_G(search_values[0]))
    history: List[Tuple[float, float]] = []

    for x in search_values:
        G = float(to_G(x))
        f1 = evaluate_G_on_clips(training_clips, detector, G)
        history.append((G, f1))
        if f1 > best_f1:
            best_f1 = f1
            best_G = G

    return best_G, best_f1, history


def optimize_detector_G(
    detector: str, training_clips: List[dict]
) -> Tuple[float, dict]:
    """
    Optimize threshold G using coarse-to-fine grid search.

    ``minimize_scalar`` is avoided because the F1 surface is jagged and
    non-convex. A log/linear coarse grid is followed by a short local
    refinement around the best coarse point.

    Args:
        detector: PyShoe detector name.
        training_clips: Training clips (must pass ``validate_clip_payload``).

    Returns:
        Tuple ``(G_opt, search_info)`` where ``search_info`` records the
        coarse/fine grids and convergence metadata.
    """
    if not training_clips:
        raise ValueError(f"No training clips provided for detector {detector}")

    for i, clip in enumerate(training_clips, 1):
        validate_clip_payload(clip, context=f"training clip {i}")

    print(f" Optimizing detector {detector} on {len(training_clips)} training clips")

    if detector == "shoe":
        coarse_x = np.linspace(9.0, 10.0, 20)
        to_G = lambda x: 10 ** x
        from_x = lambda g: np.log10(g)
        fine_half_width = 0.15
        fine_points = 7
    # elif detector == "mbgtd":
    #     coarse_x = np.linspace(40.0, 100.0, 20)  # Search from 40 to 100
    #     to_G = lambda x: x
    #     from_x = lambda g: g
    #     fine_half_width = 3.0  # Slightly wider fine search given the large range
    #     fine_points = 9
    else:
        coarse_x = np.linspace(0.1, 200.0, 20) 
        to_G = lambda x: x
        from_x = lambda g: g
        fine_half_width = 2.0
        fine_points = 9

    best_G, best_f1, coarse_history = _grid_search_threshold(
        detector, training_clips, coarse_x, to_G
    )

    center_x = from_x(best_G)
    fine_x = np.linspace(center_x - fine_half_width, center_x + fine_half_width, fine_points)
    if detector == "shoe":
        fine_x = np.clip(fine_x, 8.0, 10.0)
    # elif detector == "mbgtd":
    #     fine_x = np.clip(fine_x, 40.0, 100.0)
    else:
        fine_x = np.clip(fine_x, 0.1, 200.0)

    fine_G, fine_f1, fine_history = _grid_search_threshold(
        detector, training_clips, fine_x, to_G
    )

    if fine_f1 >= best_f1:
        G_opt, best_f1 = fine_G, fine_f1
        stage = "fine"
    else:
        G_opt, best_f1 = best_G, best_f1
        stage = "coarse"

    search_info = {
        "method": "coarse_to_fine_grid",
        "best_stage": stage,
        "best_f1": best_f1,
        "coarse_evaluations": len(coarse_history),
        "fine_evaluations": len(fine_history),
        "coarse_best": max(coarse_history, key=lambda t: t[1])[0] if coarse_history else None,
    }

    print(f"  Optimized {detector}: G*={G_opt:.6g} (train F1={best_f1:.3f}, stage={stage})")
    return G_opt, search_info


def process_single_fold(fold_data: dict, detector: str) -> dict:
    """
    Worker entry point: optimize G on training clips and evaluate on test clips.

    Args:
        fold_data: Dict with ``train``, ``test``, and ``key`` (subject id).
        detector: PyShoe detector to optimize.

    Returns:
        Dict with ``detector``, ``fold_key``, ``G_star``, ``f1_test``,
        and optional ``search_info``.

    Raises:
        ValueError: When fold payloads are invalid or empty.
    """
    key = fold_data["key"]
    train_clips = fold_data["train"]
    test_clips = fold_data["test"]

    if not train_clips:
        raise ValueError(f"Fold {key}/{detector}: training clips are empty")
    if not test_clips:
        raise ValueError(f"Fold {key}/{detector}: test clips are empty")

    G_star, search_info = optimize_detector_G(detector, train_clips)
    f1_test = evaluate_G_on_clips(test_clips, detector, G_star)

    return {
        "detector": detector,
        "fold_key": key,
        "G_star": G_star,
        "f1_test": f1_test,
        "search_info": search_info,
    }


def empty_checkpoint() -> dict:
    """Return a fresh checkpoint structure."""
    return {
        "completed": [],
        "failed": [],
        "optimized_results": {},
    }


def load_checkpoint(path: Optional[Path] = None) -> dict:
    """
    Load an existing checkpoint or return an empty structure.

    Returns:
        Checkpoint dict with ``completed``, ``failed``, and
        ``optimized_results`` keys.
    """
    path = path or CHECKPOINT_PATH
    if not path.exists():
        return empty_checkpoint()
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"Warning: could not load checkpoint ({exc}); starting fresh.")
        return empty_checkpoint()

    for key in ("completed", "failed", "optimized_results"):
        data.setdefault(key, [] if key != "optimized_results" else {})
    return data


def completed_task_keys(checkpoint: dict) -> Set[str]:
    """Return task keys already recorded as completed."""
    return {task_key(item["detector"], item["fold_key"]) for item in checkpoint.get("completed", [])}


def rebuild_optimized_results(completed: List[dict]) -> Dict[str, dict]:
    """Aggregate per-detector G and F1 lists from completed fold records."""
    results: Dict[str, dict] = {}
    for item in completed:
        det = item["detector"]
        bucket = results.setdefault(det, {"G_vals": [], "f1_tests": []})
        bucket["G_vals"].append(item["G_star"])
        bucket["f1_tests"].append(item["f1_test"])
    return results


def save_checkpoint(checkpoint: dict, path: Optional[Path] = None) -> None:
    """Persist checkpoint atomically."""
    path = path or CHECKPOINT_PATH
    atomic_write_json(path, checkpoint)


def aggregate_summary(
    detectors: List[str],
    optimized_results: Dict[str, dict],
    default_specs: dict,
) -> dict:
    """
    Build final summary dict, tolerating detectors with zero successful folds.
    """
    summary = {}
    for det in detectors:
        bucket = optimized_results.get(det, {"G_vals": [], "f1_tests": []})
        gs = bucket.get("G_vals", [])
        f1s = bucket.get("f1_tests", [])

        if det == "shoe" and gs:
            mean_G = float(10 ** np.mean(np.log10(gs)))
        elif gs:
            mean_G = float(np.mean(gs))
        else:
            mean_G = None

        mean_f1 = float(np.mean(f1s)) if f1s else None
        summary[det] = {
            "default_G": float(default_specs[det]["G"]),
            "optimized_G_mean": mean_G,
            "optimized_mean_test_F1": mean_f1,
            "successful_folds": len(gs),
        }
    return summary


def format_summary_cell(value: Optional[float], precision: int = 6) -> str:
    """Format summary numeric values for markdown output."""
    if value is None:
        return "n/a"
    if precision == 3:
        return f"{value:.3f}"
    return f"{value:.6g}"


def print_summary_table(summary: dict) -> None:
    """Print a markdown table that handles missing fold results gracefully."""
    print("\n| Detector | Default G | Optimized G* (mean) | Optimized mean F1 | Folds |")
    print("|---|---:|---:|---:|---:|")
    for det, row in summary.items():
        if det == "checkpoint":
            continue

        print(
            f"| {det} | {format_summary_cell(row['default_G'])} | "
            f"{format_summary_cell(row['optimized_G_mean'])} | "
            f"{format_summary_cell(row['optimized_mean_test_F1'], precision=3)} | "
            f"{row.get('successful_folds', 0)} |"
        )


def main() -> None:
    """
    Run LOSO PyShoe threshold optimization with restartable checkpointing.

    Loads an existing checkpoint when present, skips completed detector/fold
    tasks, writes incremental progress atomically, and emits a final summary
    JSON even when some folds fail.
    """
    parser = argparse.ArgumentParser(description="Optimize PyShoe G-values.")
    parser.add_argument(
        "-d", "--detector", 
        type=str, 
        choices=["shoe", "ared", "amvd", "mbgtd", "all"], 
        default="all", 
        help="Run optimization for a specific detector (default: all)"
    )
    args = parser.parse_args()

    # Determine which detectors to process
    all_detectors = ["shoe", "ared", "amvd", "mbgtd"]
    target_detectors = all_detectors if args.detector == "all" else [args.detector]
    
    print(f"Starting optimization for: {', '.join(target_detectors)}")

    all_clips = build_all_clips_with_sensor()
    if not all_clips:
        print("No clips found. Exiting")
        return

    folds = build_loso_folds(all_clips)
    detectors = pyshoe_export.DETECTORS
    default_specs = pyshoe_export.SPECS

    checkpoint = load_checkpoint()
    checkpoint["optimized_results"] = rebuild_optimized_results(checkpoint.get("completed", []))
    done = completed_task_keys(checkpoint)

    safe_cores = max(1, int(os.cpu_count() // 2))
    total_tasks = len(folds) * len(detectors)
    pending = sum(
        1 for det in detectors for fold in folds if task_key(det, fold["key"]) not in done
    )

    print(
        f"Starting True LOSO with {len(folds)} folds using {safe_cores} CPU cores "
        f"({pending}/{total_tasks} tasks pending)..."
    )

    tasks = []
    with ProcessPoolExecutor(max_workers=safe_cores) as executor:
        for det in target_detectors:
            for fold in folds:
                key = task_key(det, fold["key"])
                if key in done:
                    print(f"Skipping completed task {key}")
                    continue
                future = executor.submit(process_single_fold, fold, det)
                tasks.append((future, det, fold["key"]))

        for i, (future, det, fold_key) in enumerate(tasks, 1):
            key = task_key(det, fold_key)
            try:
                res = future.result()
                checkpoint["completed"].append({
                    "detector": res["detector"],
                    "fold_key": res["fold_key"],
                    "G_star": res["G_star"],
                    "f1_test": res["f1_test"],
                })
                checkpoint["optimized_results"] = rebuild_optimized_results(checkpoint["completed"])
                save_checkpoint(checkpoint)

                print(
                    f"[{i}/{len(tasks)}] Finished {key} -> "
                    f"G*={res['G_star']:.6g}, test F1={res['f1_test']:.3f}"
                )
            except Exception as exc:
                tb = traceback.format_exc()
                checkpoint["failed"].append({
                    "detector": det,
                    "fold_key": fold_key,
                    "error": str(exc),
                    "traceback": tb,
                })
                save_checkpoint(checkpoint)
                print(f"[{i}/{len(tasks)}] FAILED {key}: {exc}\n{tb}")

    summary = aggregate_summary(detectors, checkpoint["optimized_results"], default_specs)
    summary["checkpoint"] = {
        "completed_tasks": len(checkpoint.get("completed", [])),
        "failed_tasks": len(checkpoint.get("failed", [])),
    }

    print_summary_table(summary)
    atomic_write_json(SUMMARY_PATH, summary)
    print(f"\nSaved optimized specs to: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
