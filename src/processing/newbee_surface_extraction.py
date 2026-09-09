from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

REQUIRED_LABEL_COLS = ["time", "walk_mode"]


def extract_surface_segments(f_label: str, target_mode: str = "stairs_up") -> pd.DataFrame:
    """
    Parse a labels.csv file and extract continuous blocks of a walking mode.

    Args:
        f_label: Path to labels.csv containing ``time`` and ``walk_mode``.
        target_mode: Activity label to isolate (e.g. ``stairs_up``).

    Returns:
        DataFrame with ``start_time`` and ``end_time`` columns, one row per
        continuous segment. Empty (with both columns present) when no rows
        match ``target_mode``.

    Raises:
        FileNotFoundError: If ``f_label`` does not exist.
        ValueError: If required columns are missing from the CSV.
    """
    if not os.path.exists(f_label):
        raise FileNotFoundError(f"Label file not found at: {f_label}")

    df = pd.read_csv(f_label, usecols=REQUIRED_LABEL_COLS)

    is_target = df["walk_mode"] == target_mode
    if not is_target.any():
        return pd.DataFrame(columns=["start_time", "end_time"])

    df["segment_id"] = (is_target != is_target.shift()).cumsum()
    target_df = df[is_target]

    segments = (
        target_df.groupby("segment_id")["time"]
        .agg(["min", "max"])
        .reset_index(drop=True)
    )
    segments.columns = ["start_time", "end_time"]
    return segments


def parse_course_subject_from_label_path(f_label: str) -> tuple[str, str]:
    """
    Extract course and subject id from a labels path anchored at ``data_set``.

    Expected layout: ``.../data_set/<course>/<subject>/labels.csv``.

    Args:
        f_label: Path to a labels.csv file.

    Returns:
        Tuple ``(course_name, subject_id)``.

    Raises:
        ValueError: If the path does not contain the expected anchor segments.
    """
    parts = Path(f_label).parts
    try:
        anchor_idx = parts.index("data_set")
    except ValueError as exc:
        raise ValueError(
            f"Label path must contain 'data_set' segment: {f_label}"
        ) from exc
    if len(parts) < anchor_idx + 4:
        raise ValueError(
            f"Label path too short after 'data_set' anchor: {f_label}"
        )
    course_name = parts[anchor_idx + 1]
    subj_id = parts[anchor_idx + 2]
    if parts[anchor_idx + 3] != "labels.csv":
        raise ValueError(
            f"Expected labels.csv immediately after subject folder: {f_label}"
        )
    return course_name, subj_id


def export_stair_clips(f_sensor: str, f_label: str, output_dir: str) -> Optional[pd.DataFrame]:
    """
    Slice raw IMU sensor data into stairs_up clips and write CSV files.

    Uses ``get_stair_segments`` for boundaries and names outputs
    ``<subj>_<course>_stairsup_clip_<index>.csv``.

    Args:
        f_sensor: Path to raw sensor CSV with a ``time`` column (milliseconds).
        f_label: Path to labels CSV with ``time`` and ``walk_mode``.
        output_dir: Destination directory (created if missing).

    Returns:
        Segment boundary DataFrame, or ``None`` when no stairs_up segments
        exist in the labels file.

    Raises:
        FileNotFoundError: If sensor or label file is missing.
        ValueError: If the label path layout is invalid.
    """
    if not os.path.exists(f_sensor):
        raise FileNotFoundError(f"Raw sensor file not found at: {f_sensor}")
    if not os.path.exists(f_label):
        raise FileNotFoundError(f"Ground-truth label file not found at: {f_label}")

    segments = extract_surface_segments(f_label, target_mode="stairs_up")
    if segments.empty:
        print(f"No 'stairs_up' segments found for {Path(f_label).name}. Skipping.")
        return None

    course_name, subj_id = parse_course_subject_from_label_path(f_label)

    print(f"Loading raw sensor data: {Path(f_sensor).name}...")
    df_sensor = pd.read_csv(f_sensor)

    os.makedirs(output_dir, exist_ok=True)

    for idx, row in segments.iterrows():
        clip = df_sensor[
            (df_sensor.time >= row["start_time"]) & (df_sensor.time <= row["end_time"])
        ]
        filename = f"{subj_id}_{course_name}_stairsup_clip_{idx}.csv"
        out_path = os.path.join(output_dir, filename)
        clip.to_csv(out_path, index=False)
        print(f"   Successfully exported: {filename} ({len(clip)} rows)")

    return segments


if __name__ == "__main__":
    print("Testing extract_surface_segments logic execution...")
    path = r"C:\Users\theil\Documents\0_summer2026\0_git\StairsUp\data\data_set\courseA\id01\labels.csv"
    segments_df = extract_surface_segments(path)
    for start, end in zip(segments_df["start_time"], segments_df["end_time"]):
        print(start, end)
