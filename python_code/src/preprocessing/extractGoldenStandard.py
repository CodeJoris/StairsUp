import numpy as np
import pandas as pd

REQUIRED_GOLDEN_STANDARD_COLS = [
    "time",
    "insoles_RightFoot_is_step",
    "insoles_RightFoot_is_lifted",
    "insoles_LeftFoot_is_step",
    "insoles_LeftFoot_is_lifted",
]


def validate_golden_standard_input(output: pd.DataFrame) -> None:
    """
    Validate that a labels dataframe contains columns required by
    ``extract_golden_standard``.

    Args:
        output: Labels dataframe expected to hold insole event columns.

    Raises:
        TypeError: If ``output`` is not a pandas DataFrame.
        ValueError: If one or more required columns are missing.
    """
    if not isinstance(output, pd.DataFrame):
        raise TypeError("extract_golden_standard expects a pandas DataFrame")
    missing = [c for c in REQUIRED_GOLDEN_STANDARD_COLS if c not in output.columns]
    if missing:
        raise ValueError(f"Missing required columns for golden standard extraction: {missing}")


def extract_golden_standard(output: pd.DataFrame):
    """
    Extract heel-strike (HS) and toe-off (FO) golden-standard event times.

    Reads insole step/lifted boolean columns for both feet, pairs each HS
    with the next FO on the same foot, and returns sorted Nx2 arrays where
    column 0 is time (ms) and column 1 is foot label (0=right, 1=left).

    Args:
        output: DataFrame with ``REQUIRED_GOLDEN_STANDARD_COLS``.

    Returns:
        Tuple[np.ndarray, np.ndarray]: ``(y_HS, y_FO)`` each shaped (N, 2).
            Either array may be empty (shape (0, 2)) when no valid pairs exist.

    Raises:
        TypeError: If ``output`` is not a DataFrame.
        ValueError: If required columns are missing.
    """
    validate_golden_standard_input(output)

    def get_event_times(df, col_name):
        mask = (df[col_name] == "True") | (df[col_name] == True)
        return df.loc[mask, "time"].values

    ic_r = get_event_times(output, "insoles_RightFoot_is_step")
    tc_r = get_event_times(output, "insoles_RightFoot_is_lifted")
    ic_l = get_event_times(output, "insoles_LeftFoot_is_step")
    tc_l = get_event_times(output, "insoles_LeftFoot_is_lifted")

    ic_r = remove_t0(ic_r)
    ic_l = remove_t0(ic_l)

    tc_r = align_fo_to_hs(ic_r, tc_r)
    tc_l = align_fo_to_hs(ic_l, tc_l)

    paired_r = pair_events(ic_r, tc_r, 0)
    paired_l = pair_events(ic_l, tc_l, 1)

    if paired_r.empty and paired_l.empty:
        empty = np.empty((0, 2))
        return empty, empty.copy()

    timings = pd.concat([paired_r, paired_l], ignore_index=True)
    timings = timings.sort_values(by="InitialContact").reset_index(drop=True)

    y_HS = timings[["InitialContact", "LeftStance"]].to_numpy()
    y_FO = timings[["TerminalContact", "LeftStance"]].to_numpy()

    return y_HS, y_FO


def remove_t0(ic: np.ndarray) -> np.ndarray:
    """
    Drop a heel-strike at exactly t=0 (recording has not started walking).

    Args:
        ic: 1D array of initial-contact timestamps.

    Returns:
        Filtered array; unchanged when empty or first event is not at zero.
    """
    if len(ic) > 0 and ic[0] == 0:
        return ic[1:]
    return ic


def align_fo_to_hs(ic: np.ndarray, tc: np.ndarray) -> np.ndarray:
    """
    Remove toe-off events that occur before the first heel-strike on a foot.

    Args:
        ic: Heel-strike timestamps for one foot.
        tc: Toe-off timestamps for the same foot.

    Returns:
        Toe-off array trimmed so the first FO is not before the first HS.
    """
    if len(ic) > 0 and len(tc) > 0:
        while len(tc) > 0 and tc[0] < ic[0]:
            tc = tc[1:]
    return tc


def pair_events(ic: np.ndarray, tc: np.ndarray, foot_label: int) -> pd.DataFrame:
    """
    Pair each heel-strike with the first following toe-off before the next HS.

    Unpaired HS events (no FO before the next HS) are dropped.

    Args:
        ic: Heel-strike timestamps for one foot.
        tc: Toe-off timestamps for the same foot.
        foot_label: 0 for right, 1 for left (stored in ``LeftStance`` column).

    Returns:
        DataFrame with columns ``InitialContact``, ``TerminalContact``,
        ``LeftStance``. Empty when no valid pairs exist.
    """
    if len(ic) == 0:
        return pd.DataFrame(columns=["InitialContact", "TerminalContact", "LeftStance"])

    tc_matched = np.full(len(ic), np.nan)

    for i in range(len(ic)):
        upper_bound = ic[i + 1] if i < len(ic) - 1 else np.inf
        valid_tcs = tc[(tc > ic[i]) & (tc < upper_bound)]
        if len(valid_tcs) > 0:
            tc_matched[i] = valid_tcs[0]

    valid = ~np.isnan(tc_matched)
    if not valid.any():
        return pd.DataFrame(columns=["InitialContact", "TerminalContact", "LeftStance"])

    return pd.DataFrame({
        "InitialContact": ic[valid],
        "TerminalContact": tc_matched[valid],
        "LeftStance": np.full(np.sum(valid), foot_label),
    })
