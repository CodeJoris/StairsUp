import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from python_code.src.preprocessing.extractGoldenStandard import (
    extract_golden_standard,
    validate_golden_standard_input,
    pair_events,
    align_fo_to_hs,
    remove_t0,
)
from python_code.src.preprocessing.extractStairs import (
    get_stair_segments,
    parse_course_subject_from_label_path,
)
from python_code.src.algorithms import optimize_pyshoe as opt


@pytest.fixture
def mock_label_file(tmp_path):
    csv_path = tmp_path / "mock_labels.csv"
    data = {
        "time": [10, 20, 30, 40, 50, 60, 70],
        "walk_mode": ["flat", "flat", "stairs_up", "stairs_up", "stairs_up", "flat", "flat"],
    }
    pd.DataFrame(data).to_csv(csv_path, index=False)
    return str(csv_path)


def test_get_stair_segments_extracts_correct_bounds(mock_label_file):
    result = get_stair_segments(mock_label_file, target_mode="stairs_up")
    assert len(result) == 1
    assert result.loc[0, "start_time"] == 30
    assert result.loc[0, "end_time"] == 50


def test_get_stair_segments_empty_if_no_match(mock_label_file):
    result = get_stair_segments(mock_label_file, target_mode="running")
    assert len(result) == 0
    assert list(result.columns) == ["start_time", "end_time"]


def test_get_stair_segments_handles_missing_file():
    with pytest.raises(FileNotFoundError):
        get_stair_segments("this_file_does_not_exist.csv")


def test_parse_course_subject_from_label_path():
    path = "/data/data_set/courseA/id01/labels.csv"
    course, sid = parse_course_subject_from_label_path(path)
    assert course == "courseA"
    assert sid == "id01"


def test_parse_course_subject_from_label_path_invalid():
    with pytest.raises(ValueError):
        parse_course_subject_from_label_path("/tmp/labels.csv")


def _golden_standard_df():
    """Minimal labels frame with one HS/FO pair per foot."""
    rows = []
    for t, r_step, r_lift, l_step, l_lift in [
        (100, False, False, False, False),
        (200, True, False, False, False),
        (300, False, True, False, False),
        (400, False, False, True, False),
        (500, False, False, False, True),
    ]:
        rows.append({
            "time": t,
            "insoles_RightFoot_is_step": r_step,
            "insoles_RightFoot_is_lifted": r_lift,
            "insoles_LeftFoot_is_step": l_step,
            "insoles_LeftFoot_is_lifted": l_lift,
        })
    return pd.DataFrame(rows)


def _minimal_sensor_df():
    cols = {c: [0.0] for c in opt.REQUIRED_SENSOR_COLS}
    return pd.DataFrame(cols)


def test_extract_golden_standard_returns_paired_events():
    y_hs, y_fo = extract_golden_standard(_golden_standard_df())
    assert y_hs.shape[1] == 2
    assert y_fo.shape[1] == 2
    assert len(y_hs) == 2
    assert np.all(y_hs[:, 0] < y_fo[:, 0])


def test_extract_golden_standard_missing_columns():
    df = pd.DataFrame({"time": [1, 2]})
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_golden_standard_input(df)


def test_extract_golden_standard_empty_events():
    df = pd.DataFrame({
        "time": [1, 2],
        "insoles_RightFoot_is_step": [False, False],
        "insoles_RightFoot_is_lifted": [False, False],
        "insoles_LeftFoot_is_step": [False, False],
        "insoles_LeftFoot_is_lifted": [False, False],
    })
    y_hs, y_fo = extract_golden_standard(df)
    assert y_hs.shape == (0, 2)
    assert y_fo.shape == (0, 2)


def test_remove_t0_drops_only_zero():
    assert np.array_equal(remove_t0(np.array([0, 10, 20])), np.array([10, 20]))
    assert np.array_equal(remove_t0(np.array([5, 10])), np.array([5, 10]))


def test_align_fo_to_hs_trims_early_fo():
    ic = np.array([100, 300])
    tc = np.array([50, 200])
    trimmed = align_fo_to_hs(ic, tc)
    assert np.array_equal(trimmed, np.array([200]))


def test_pair_events_drops_unpaired_hs():
    ic = np.array([100, 200, 300])
    tc = np.array([150])
    paired = pair_events(ic, tc, foot_label=0)
    assert len(paired) == 1
    assert paired.iloc[0]["InitialContact"] == 100


def test_hungarian_match_empty_arrays():
    assert opt.hungarian_match(np.array([]), np.array([])) == (0, 0, 0)
    assert opt.hungarian_match(np.array([10, 20]), np.array([])) == (0, 0, 2)
    assert opt.hungarian_match(np.array([]), np.array([10, 20])) == (0, 2, 0)


def test_hungarian_match_tolerance_boundary():
    true = np.array([100.0])
    pred_inside = np.array([140.0])
    pred_outside = np.array([160.0])
    assert opt.hungarian_match(true, pred_inside, tol=50)[0] == 1
    assert opt.hungarian_match(true, pred_outside, tol=50)[0] == 0


def test_build_loso_folds_deterministic_order():
    clips = [
        {"id": "id03", "x": 1},
        {"id": "id01", "x": 2},
        {"id": "id02", "x": 3},
        {"id": "id01", "x": 4},
    ]
    folds = opt.build_loso_folds(clips)
    assert [f["key"] for f in folds] == ["id01", "id02", "id03"]
    assert len(folds[0]["test"]) == 2
    assert len(folds[0]["train"]) == 2


def test_parse_course_subject_from_xsens_path():
    path = Path("/project/data/data_set/courseB/id07/xsens.csv")
    course, sid = opt.parse_course_subject_from_xsens_path(path)
    assert course == "courseB"
    assert sid == "id07"


def test_parse_course_subject_from_xsens_path_invalid():
    with pytest.raises(ValueError):
        opt.parse_course_subject_from_xsens_path(Path("/tmp/xsens.csv"))


def test_strip_padding_artifact_only_index_zero():
    idx = np.array([0, 5, 10])
    stripped = opt._strip_padding_artifact(idx)
    assert np.array_equal(stripped, np.array([5, 10]))
    idx_no_zero = np.array([3, 7])
    assert np.array_equal(opt._strip_padding_artifact(idx_no_zero), idx_no_zero)


def test_atomic_write_json_replaces_file(tmp_path):
    target = tmp_path / "state.json"
    opt.atomic_write_json(target, {"a": 1})
    opt.atomic_write_json(target, {"b": 2})
    with open(target, encoding="utf-8") as fh:
        data = json.load(fh)
    assert data == {"b": 2}
    assert not list(tmp_path.glob("*.tmp"))


def test_checkpoint_restart_skips_completed(tmp_path, monkeypatch):
    ckpt = tmp_path / "checkpoint.json"
    monkeypatch.setattr(opt, "CHECKPOINT_PATH", ckpt)

    existing = {
        "completed": [
            {"detector": "ared", "fold_key": "id01", "G_star": 2.0, "f1_test": 0.5}
        ],
        "failed": [],
        "optimized_results": {},
    }
    opt.atomic_write_json(ckpt, existing)

    loaded = opt.load_checkpoint()
    done = opt.completed_task_keys(loaded)
    assert "ared|id01" in done
    rebuilt = opt.rebuild_optimized_results(loaded["completed"])
    assert rebuilt["ared"]["G_vals"] == [2.0]


def test_aggregate_summary_handles_no_successful_folds():
    summary = opt.aggregate_summary(
        ["shoe", "ared"],
        {"shoe": {"G_vals": [], "f1_tests": []}},
        opt.pyshoe_export.SPECS,
    )
    assert summary["shoe"]["optimized_G_mean"] is None
    assert summary["shoe"]["optimized_mean_test_F1"] is None
    assert summary["shoe"]["successful_folds"] == 0
    assert opt.format_summary_cell(None) == "n/a"


def test_validate_clip_payload_rejects_bad_shape():
    with pytest.raises(ValueError, match="shape"):
        opt.validate_clip_payload({
            "sensor_df": _minimal_sensor_df(),
            "y_HS": np.array([1, 2]),
            "y_FO": np.array([[1, 0]]),
        })


def test_format_summary_table_does_not_crash(capsys):
    summary = {
        "ared": {
            "default_G": 2.0,
            "optimized_G_mean": None,
            "optimized_mean_test_F1": None,
            "successful_folds": 0,
        }
    }
    opt.print_summary_table(summary)
    out = capsys.readouterr().out
    assert "n/a" in out
