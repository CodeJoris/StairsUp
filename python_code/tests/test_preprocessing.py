import pandas as pd
import pytest
import numpy as np
from python_code.src.preprocessing.newbee_surface_extraction import get_stair_segments

@pytest.fixture
def mock_label_file(tmp_path):
    """Creates a temporary, fake labels.csv file for testing."""
    csv_path = tmp_path / "mock_labels.csv"
    
    # Simulating a file where someone walks on flat ground, 
    # goes up stairs (time 30-50), and returns to flat ground.
    data = {
        'time': [10, 20, 30, 40, 50, 60, 70],
        'walk_mode': ['flat', 'flat', 'stairs_up', 'stairs_up', 'stairs_up', 'flat', 'flat']
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    return str(csv_path)


def test_get_stair_segments_extracts_correct_bounds(mock_label_file):
    """Ensure it correctly finds the single start and end time for stairs."""
    result = get_stair_segments(mock_label_file, target_mode="stairs_up")
    
    # Assertions: check if the logic holds up
    assert len(result) == 1
    assert result.loc[0, 'start_time'] == 30
    assert result.loc[0, 'end_time'] == 50


def test_get_stair_segments_empty_if_no_match(mock_label_file):
    """Ensure it returns an empty dataframe gracefully if a mode isn't present."""
    result = get_stair_segments(mock_label_file, target_mode="running")
    
    assert len(result) == 0
    assert list(result.columns) == ['start_time', 'end_time']


def test_get_stair_segments_handles_missing_file():
    """Ensure it throws a proper FileNotFoundError if the path is invalid."""
    with pytest.raises(FileNotFoundError):
        get_stair_segments("this_file_does_not_exist.csv")