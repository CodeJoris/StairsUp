"""
Hyperparameter optimization for PyShoe zero-velocity thresholds (G)
Target: optimize G for 'stairs_up' segments using LOSO cross-validation.

Run from the repository `python_code/` root as:

    python -m src.algorithms.optimize_pyshoe

"""
from __future__ import annotations
import sys
from pathlib import Path
import json
from typing import List, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar, linear_sum_assignment
import os


# Ensure imports for local toolbox/processing paths when run from python_code/ root
ROOT = Path(__file__).resolve().parents[3]  # python_code/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from python_code.src.preprocessing.extractStairs import get_stair_segments
from python_code.src.preprocessing.extractGoldenStandard import extract_golden_standard
from python_code.Toolboxes.PyShoe import pyshoe_export
from python_code.Toolboxes.PyShoe.ins_tools.INS import INS
from concurrent.futures import ProcessPoolExecutor, as_completed

DATA_PATH = ROOT / "data"
TOL_MS = 50
STATIONARY_BUFFER_MS = 2000


def list_all_subject_files() -> List[Path]:
    """
    Find all `xsens.csv` files under the configured `DATA_PATH`.

    Returns:
        List[Path]: List of file paths to sensor CSVs (may be empty).
    """
    files = list(DATA_PATH.rglob("xsens.csv"))
    print(f"Found {len(files)} xsens.csv files under {DATA_PATH}")
    return files


def get_stair_clips_for_subject(course: str, sid: str):
    """
    Extract stair-up clips' ground-truth for a single subject and course.

    This function reads `labels.csv` for the given `course` and `sid`, finds
    continuous `stairs_up` blocks via `get_stair_segments`, extracts the golden
    standard heel-strike (`y_HS`) and toe-off (`y_FO`) timestamps using
    `extract_golden_standard`, and cleans edge cases (dropping incomplete
    first/last events).

    Args:
        course (str): Course name folder under `data_set` (e.g., "courseA").
        sid (str): Subject id folder (e.g., "id01").

    Returns:
        list[dict]: A list of clip metadata dicts with keys ``segment_id``,
            ``start_time``, ``end_time``, ``y_HS``, ``y_FO``. Returns an empty
            list when no stairs_up segments exist.

    Raises:
        FileNotFoundError: If the labels file does not exist.
    """
    file_path = DATA_PATH / "data_set" / course / sid / "labels.csv"
    stair_blocks = get_stair_segments(str(file_path), target_mode="stairs_up")
    if stair_blocks.empty:
        return []
    print(f"Loading labels for {course}/{sid}: {file_path}")
    data = pd.read_csv(file_path, usecols=["time", "insoles_RightFoot_is_step", "insoles_RightFoot_is_lifted", "insoles_LeftFoot_is_step", "insoles_LeftFoot_is_lifted"]) 
    y_HS_full, y_FO_full = extract_golden_standard(data)

    clips = []
    for idx, block in stair_blocks.iterrows():
        mask_HS = (y_HS_full[:, 0] >= block.start_time) & (y_HS_full[:, 0] <= block.end_time)
        mask_FO = (y_FO_full[:, 0] >= block.start_time) & (y_FO_full[:, 0] <= block.end_time)

        y_HS_clip = y_HS_full[mask_HS]
        y_FO_clip = y_FO_full[mask_FO]

        # Drop initial FO before first HS
        if len(y_FO_clip) > 0 and len(y_HS_clip) > 0 and y_FO_clip[0, 0] < y_HS_clip[0, 0]:
            y_FO_clip = y_FO_clip[1:]

        # Drop trailing HS if after last FO
        if len(y_HS_clip) > 0 and len(y_FO_clip) > 0 and y_HS_clip[-1, 0] > y_FO_clip[-1, 0]:
            y_HS_clip = y_HS_clip[:-1]

        # Safety Guard: Skip appending if this staircase doesn't contain at least one complete gait cycle
        if len(y_HS_clip) == 0 or len(y_FO_clip) == 0:
            print(f"  Skipping segment {idx} for {sid}_{course}: incomplete cycles")
            continue

        clips.append({
            'segment_id': int(idx),
            'start_time': int(block.start_time),
            'end_time': int(block.end_time),
            'y_HS': y_HS_clip,
            'y_FO': y_FO_clip
        })

    print(f"  Extracted {len(clips)} valid stairs_up clips for {course}/{sid}")
    return clips


def slice_sensor_with_buffer(raw_df: pd.DataFrame, start_time: int, end_time: int) -> pd.DataFrame:
    """
    Slice raw sensor dataframe to [start_time - buffer, end_time].

    Args:
        raw_df (pd.DataFrame): Full xsens sensor dataframe with a `time` column.
        start_time (int): Clip start time in milliseconds.
        end_time (int): Clip end time in milliseconds.

    Returns:
        pd.DataFrame: Sliced dataframe including a stationary buffer before
            `start_time` to allow INS initialization.
    """
    start = max(0, start_time - STATIONARY_BUFFER_MS)
    print(f"    Slicing sensor data [{start} .. {end_time}] (buffered from {start_time}) -> {len(raw_df)} rows available")
    sliced = raw_df[(raw_df.time >= start) & (raw_df.time <= end_time)].reset_index(drop=True)
    print(f"    Sliced df has {len(sliced)} rows")
    return sliced


def get_predictions_for_clip(clip_df: pd.DataFrame, detector: str, G_val: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run PyShoe INS baseline for a given detector and threshold on a sensor clip.

    The function prepares left/right 6-axis IMU arrays, runs
    ``INS.baseline(W=5, G=G_val, detector=detector)`` for each foot, and
    extracts heel-strike (HS) and foot-off (FO) timestamps by diffing the
    zero-velocity boolean vector.

    Args:
        clip_df (pd.DataFrame): Sensor dataframe sliced to the clip window.
        detector (str): One of PyShoe detectors (e.g., 'shoe', 'ared').
        G_val (float): Threshold value passed to INS (for 'shoe' this is the
            linear G; upstream code may optimize log10-space).

    Returns:
        Tuple[np.ndarray, np.ndarray]: Sorted arrays of predicted HS times and
            FO times (milliseconds).
    """
    # Prepare imu arrays as in pyshoe_export
    RIGHT_FOOT_COLS = ['acceleration_RightFoot_x', 'acceleration_RightFoot_y', 'acceleration_RightFoot_z', 'angularVelocity_RightFoot_x', 'angularVelocity_RightFoot_y', 'angularVelocity_RightFoot_z']
    LEFT_FOOT_COLS = ['acceleration_LeftFoot_x', 'acceleration_LeftFoot_y', 'acceleration_LeftFoot_z', 'angularVelocity_LeftFoot_x', 'angularVelocity_LeftFoot_y', 'angularVelocity_LeftFoot_z']

    imu_left = clip_df[LEFT_FOOT_COLS].to_numpy()
    imu_right = clip_df[RIGHT_FOOT_COLS].to_numpy()
    time = clip_df['time'].to_numpy().squeeze()

    print(f"    Predicting clip with detector={detector}, G={G_val:.6g}, rows={len(clip_df)}")
    # Left
    ins_left = INS(imu_left, sigma_a=0.00098, sigma_w=8.7266463e-5, T=1.0/pyshoe_export.SAMPLING_FREQUENCY)
    ins_left.baseline(W=5, G=G_val, detector=detector)
    zv_left = ins_left.zv
    padded_left = np.insert(zv_left, 0, False).astype(int)
    diff_left = np.diff(padded_left)
    HS_left = np.where(diff_left == 1)[0]
    FO_left = np.where(diff_left == -1)[0]
    if len(zv_left) > 0 and zv_left[0]:
        HS_left = HS_left[1:]
        FO_left = FO_left[1:]
    HS_times_left = time[HS_left] if HS_left.size else np.array([])
    FO_times_left = time[FO_left] if FO_left.size else np.array([])

    # Right
    ins_right = INS(imu_right, sigma_a=0.00098, sigma_w=8.7266463e-5, T=1.0/pyshoe_export.SAMPLING_FREQUENCY)
    ins_right.baseline(W=5, G=G_val, detector=detector)
    zv_right = ins_right.zv
    padded_right = np.insert(zv_right, 0, False).astype(int)
    diff_right = np.diff(padded_right)
    HS_right = np.where(diff_right == 1)[0]
    FO_right = np.where(diff_right == -1)[0]
    if len(zv_right) > 0 and zv_right[0]:
        HS_right = HS_right[1:]
        FO_right = FO_right[1:]
    HS_times_right = time[HS_right] if HS_right.size else np.array([])
    FO_times_right = time[FO_right] if FO_right.size else np.array([])

    all_HS = np.concatenate((HS_times_left, HS_times_right)) if HS_times_left.size or HS_times_right.size else np.array([])
    all_FO = np.concatenate((FO_times_left, FO_times_right)) if FO_times_left.size or FO_times_right.size else np.array([])

    hs_sorted = np.sort(all_HS)
    fo_sorted = np.sort(all_FO)
    print(f"    Predicted {len(hs_sorted)} HS and {len(fo_sorted)} FO events")
    return hs_sorted, fo_sorted


def hungarian_match(true_times: np.ndarray, pred_times: np.ndarray, tol: float = TOL_MS) -> Tuple[int, int, int]:
    """
    Match predicted event times to true times using the Hungarian algorithm.

    A prediction and a true event are considered matchable if their absolute
    time difference is <= `tol`. Assignments with difference > `tol` are
    forbidden. The function returns counts of true positives (TP), false
    positives (FP) and false negatives (FN).

    Args:
        true_times (np.ndarray): 1D array of ground-truth event timestamps.
        pred_times (np.ndarray): 1D array of predicted event timestamps.
        tol (float): Matching tolerance in milliseconds. Default is 50 ms.

    Returns:
        Tuple[int, int, int]: (TP, FP, FN)
    """
    if len(pred_times) == 0 and len(true_times) == 0:
        return 0, 0, 0
    if len(pred_times) == 0:
        return 0, 0, len(true_times)
    if len(true_times) == 0:
        return 0, len(pred_times), 0

    # cost matrix
    cost = np.abs(true_times[:, None] - pred_times[None, :])
    # forbid assignments greater than tol by setting large cost
    LARGE = 1e9
    cost_masked = np.where(cost <= tol, cost, LARGE)

    row_ind, col_ind = linear_sum_assignment(cost_masked)
    # count matches where cost <= tol
    matched = 0
    for r, c in zip(row_ind, col_ind):
        if cost_masked[r, c] < LARGE:
            matched += 1

    TP = matched
    FP = len(pred_times) - TP
    FN = len(true_times) - TP
    print(f"      Matching: TP={TP}, FP={FP}, FN={FN}")
    return TP, FP, FN


def f1_from_counts(tp: int, fp: int, fn: int) -> float:
    """
    Compute F1 score given TP, FP and FN counts.

    Args:
        tp (int): True positives count.
        fp (int): False positives count.
        fn (int): False negatives count.

    Returns:
        float: F1 score in [0, 1]. Returns 0.0 when TP == 0.
    """
    if tp == 0:
        return 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if prec + rec == 0:
        return 0.0
    return 2 * (prec * rec) / (prec + rec)


def evaluate_G_on_clips(clips: List[dict], detector: str, G_val: float) -> float:
    """
    Evaluate a given threshold `G_val` across a list of clips and return
    the aggregated F1 score for heel-strike detections.

    Args:
        clips (List[dict]): List of clip metadata dicts; each must include
            ``sensor_df`` (pd.DataFrame) and ``y_HS`` (np.ndarray).
        detector (str): Detector name passed to PyShoe.
        G_val (float): Threshold passed to INS.

    Returns:
        float: Aggregated F1 score across all clips.
    """
    TP = FP = FN = 0
    print(f"  Evaluating detector={detector} G={G_val:.6g} on {len(clips)} clips")
    for i, clip_meta in enumerate(clips, 1):
        if 'sensor_df' in clip_meta:
            sensor_df = clip_meta['sensor_df']
        else:
            raise RuntimeError('clip_meta missing sensor_df')

        clip_df = sensor_df
        # Extract both true arrays
        y_HS_true = clip_meta['y_HS'][:, 0]
        y_FO_true = clip_meta['y_FO'][:, 0]

        # Get both prediction arrays
        y_HS_pred, y_FO_pred = get_predictions_for_clip(clip_df, detector, G_val)

        # --- NEW SAFETY VALVE ---
        if len(y_HS_pred) > 300 or len(y_FO_pred) > 300:
            print(f"    [WARNING] G={G_val:.6g} caused noise explosion ({len(y_HS_pred)} steps). Rejecting.")
            return 0.0  # Immediately punish this threshold
        # ------------------------

        # Match both event types
        tp_hs, fp_hs, fn_hs = hungarian_match(y_HS_true, y_HS_pred, tol=TOL_MS)
        tp_fo, fp_fo, fn_fo = hungarian_match(y_FO_true, y_FO_pred, tol=TOL_MS)
        
        # Aggregate
        TP += (tp_hs + tp_fo)
        FP += (fp_hs + fp_fo)
        FN += (fn_hs + fn_fo)
        
        print(f"    Clip {i}/{len(clips)}: TP={(tp_hs+tp_fo)}, FP={(fp_hs+fp_fo)}, FN={(fn_hs+fn_fo)}")

    f1 = f1_from_counts(TP, FP, FN)
    print(f"  -> Aggregated TP={TP}, FP={FP}, FN={FN}, F1={f1:.3f}")
    return f1


def build_all_clips_with_sensor() -> List[dict]:
    """
    Build a list of all stairs_up clips with their sliced sensor data.

    Scans for all `xsens.csv` files, extracts stair-up clip ground truth using
    `get_stair_clips_for_subject`, slices the raw sensor data with a buffer and
    attaches the dataframe to each clip dict under the key ``sensor_df``.

    Returns:
        List[dict]: List of clip dicts ready for evaluation/optimization.
    """
    clips = []
    xsens_files = list_all_subject_files()
    print(f"Building clips from {len(xsens_files)} sensor files")
    for f in xsens_files:
        sid = f.parts[-2]
        course = f.parts[-3]
        print(f" Processing file: {course}/{sid} -> {f}")
        stair_clips = get_stair_clips_for_subject(course, sid)
        if not stair_clips:
            print(f"   No stair clips for {course}/{sid}")
            continue

        raw_df = pd.read_csv(f)
        for c in stair_clips:
            c_copy = c.copy()
            c_copy['course'] = course
            c_copy['id'] = sid
            # slice raw df with buffer
            c_copy['sensor_df'] = slice_sensor_with_buffer(raw_df, c_copy['start_time'], c_copy['end_time'])
            clips.append(c_copy)

    print(f"Built total {len(clips)} clips with sensor data attached")
    return clips


def optimize_detector_G(detector: str, training_clips: List[dict]):
    """
    Optimize the G threshold for a single detector using bounded scalar
    minimization. The objective is `1 - F1` computed on `training_clips`.

    For the 'shoe' detector the search is performed in log10-space between
    [6, 10] and the returned G is 10**x. For the other detectors the search
    is performed in linear space between [0.1, 50.0].

    Args:
        detector (str): Detector name.
        training_clips (List[dict]): Clips used for optimization (must include
            `sensor_df` and `y_HS`).

    Returns:
        Tuple[float, OptimizeResult]: Optimized G value and the SciPy result.
    """
    print(f" Optimizing detector {detector} on {len(training_clips)} training clips")
    # objective returns 1 - F1
    if detector == 'shoe':
        def obj(x):
            G = 10 ** x
            f1 = evaluate_G_on_clips(training_clips, detector, G)
            return 1.0 - f1

        res = minimize_scalar(obj, bounds=(8.0, 10.0), method='bounded', options={'maxiter': 15})
        G_opt = 10 ** res.x
    else:
        def obj(G):
            f1 = evaluate_G_on_clips(training_clips, detector, G)
            return 1.0 - f1

        res = minimize_scalar(obj, bounds=(0.1, 50.0), method='bounded', options={'maxiter': 15})
        G_opt = res.x

    print(f"  Optimized {detector}: G*={G_opt:.6g}")
    return G_opt, res

def process_single_fold(fold_data: dict, detector: str) -> dict:
    """Worker function to optimize and test a single fold on a separate CPU core."""
    key = fold_data['key']
    train_clips = fold_data['train']
    test_clips = fold_data['test']
    
    # 1. Train
    G_star, _ = optimize_detector_G(detector, train_clips)
    
    # 2. Test
    f1_test = evaluate_G_on_clips(test_clips, detector, G_star)
    
    return {
        'detector': detector,
        'fold_key': key,
        'G_star': G_star,
        'f1_test': f1_test
    }

# def main():
#     all_clips = build_all_clips_with_sensor()
#     if not all_clips:
#         print('No clips found. Exiting')
#         return

#     # Build True LOSO folds: isolate completely by Subject ID
#     subjects = list(set([c['id'] for c in all_clips]))
#     folds = []
    
#     for sid in subjects:
#         train_clips = [c for c in all_clips if c['id'] != sid]
#         test_clips = [c for c in all_clips if c['id'] == sid]
#         folds.append({'train': train_clips, 'test': test_clips, 'key': sid})

#     detectors = pyshoe_export.DETECTORS
#     default_specs = pyshoe_export.SPECS

#     optimized_results = {d: {'G_vals': [], 'f1_tests': []} for d in detectors}

#     print(f"Starting LOSO with {len(folds)} folds")
#     for fi, fold in enumerate(folds, 1):
#         print(f"\nFold {fi}/{len(folds)}: Hold-out Subject = {fold['key']}")
#         for det in detectors:
#             G_star, res = optimize_detector_G(det, fold['train'])
#             optimized_results[det]['G_vals'].append(G_star)

#             # evaluate on held-out test clips
#             f1_test = evaluate_G_on_clips(fold['test'], det, G_star)
#             optimized_results[det]['f1_tests'].append(f1_test)

#             print(f"Fold {fold['key']} detector {det} -> G*={G_star:.6g}, test F1={f1_test:.3f}")

#     # Aggregate
#     summary = {}
#     for det in detectors:
#         Gs = optimized_results[det]['G_vals']
#         f1s = optimized_results[det]['f1_tests']
        
#         if det == 'shoe' and Gs:
#             # Geometric mean for log-scaled variables
#             mean_G = float(10 ** np.mean(np.log10(Gs)))
#         else:
#             mean_G = float(np.mean(Gs)) if Gs else None
            
#         mean_f1 = float(np.mean(f1s)) if f1s else None
        
#         summary[det] = {
#             'default_G': float(default_specs[det]['G']),
#             'optimized_G_mean': mean_G,
#             'optimized_mean_test_F1': mean_f1
#         }

#     # Print markdown table
#     print("\n| Detector | Default G | Optimized G* (mean) | Optimized mean F1 |")
#     print("|---|---:|---:|---:|")
#     for det in detectors:
#         row = summary[det]
#         print(f"| {det} | {row['default_G']:.6g} | {row['optimized_G_mean']:.6g} | {row['optimized_mean_test_F1']:.3f} |")

#     # Save JSON
#     out_path = Path(__file__).resolve().parent / 'pyshoe_optimized_specs_stairs.json'
#     with open(out_path, 'w') as fh:
#         json.dump(summary, fh, indent=2)

#     print(f"\nSaved optimized specs to: {out_path}")

def main():
    all_clips = build_all_clips_with_sensor()
    if not all_clips:
        print('No clips found. Exiting')
        return

    # Build True LOSO folds: isolate completely by Subject ID
    subjects = list(set([c['id'] for c in all_clips]))
    folds = []
    
    for sid in subjects:
        train_clips = [c for c in all_clips if c['id'] != sid]
        test_clips = [c for c in all_clips if c['id'] == sid]
        folds.append({'train': train_clips, 'test': test_clips, 'key': sid})

    detectors = pyshoe_export.DETECTORS
    default_specs = pyshoe_export.SPECS

    optimized_results = {d: {'G_vals': [], 'f1_tests': []} for d in detectors}

    # Calculate a safe number of workers (e.g., 90% your total CPU cores)
    # Using max(1, ...) ensures it doesn't try to assign 0 workers on small machines
    safe_cores = max(1, int(os.cpu_count() * 0.9))
    
    print(f"Starting True LOSO with {len(folds)} folds using {safe_cores} CPU cores...")    

    
    # --- MULTIPROCESSING DISPATCHER ---
    tasks = []
    with ProcessPoolExecutor(max_workers=safe_cores) as executor:
        for det in detectors:
            for fold in folds:
                # Submit task to a CPU core
                future = executor.submit(process_single_fold, fold, det)
                tasks.append(future)
                
        # Collect results as they finish
        for i, future in enumerate(as_completed(tasks), 1):
            try:
                res = future.result()
                det = res['detector']
                
                optimized_results[det]['G_vals'].append(res['G_star'])
                optimized_results[det]['f1_tests'].append(res['f1_test'])
                
                print(f"[{i}/{len(folds) * len(detectors)}] Finished Fold {res['fold_key']} for {det} -> G*={res['G_star']:.6g}, test F1={res['f1_test']:.3f}")
                
                # --- NEW CHECKPOINTING ---
                checkpoint_path = Path(__file__).resolve().parent / 'pyshoe_optimization_checkpoint.json'
                with open(checkpoint_path, 'w') as f:
                    json.dump(optimized_results, f, indent=2)
                # -------------------------

            except Exception as exc:
                print(f"A fold generated an exception: {exc}")

    # Aggregate
    summary = {}
    for det in detectors:
        Gs = optimized_results[det]['G_vals']
        f1s = optimized_results[det]['f1_tests']
        
        if det == 'shoe' and Gs:
            # Geometric mean for log-scaled variables
            mean_G = float(10 ** np.mean(np.log10(Gs)))
        else:
            mean_G = float(np.mean(Gs)) if Gs else None
            
        mean_f1 = float(np.mean(f1s)) if f1s else None
        
        summary[det] = {
            'default_G': float(default_specs[det]['G']),
            'optimized_G_mean': mean_G,
            'optimized_mean_test_F1': mean_f1
        }

    # Print markdown table
    print("\n| Detector | Default G | Optimized G* (mean) | Optimized mean F1 |")
    print("|---|---:|---:|---:|")
    for det in detectors:
        row = summary[det]
        print(f"| {det} | {row['default_G']:.6g} | {row['optimized_G_mean']:.6g} | {row['optimized_mean_test_F1']:.3f} |")

    # Save JSON
    out_path = Path(__file__).resolve().parent / 'pyshoe_optimized_specs_stairs.json'
    with open(out_path, 'w') as fh:
        json.dump(summary, fh, indent=2)

    print(f"\nSaved optimized specs to: {out_path}")


if __name__ == '__main__':
    main()
