# StairsUp: Step Detection Analysis

This repository provides a foundation for evaluating and analyzing step-detection algorithms, specifically focusing on stair ambulation using real-world IMU data. It processes and evaluates real, distinct datasets (the NEWBEE and OSF databases) without merging them, testing algorithms like PyShoe and KielMAT to demonstrate technical proficiency in signal processing and gait analysis.

---

## Directory Structure

```text
|-- data
|   |-- ambulation_out									<-- output data from the OSF Dataset (on stairs up sections)
|   |-- data_set										<-- NEWBEE Dataset
|	|	`-- CourseA
|	|		`-- id01
|	|			|-- label.csv							<-- Contains ground truth timings and surface labels
|	|			`-- xsens.csv							<-- IMU data
|	|
|   |-- stair_ambulation_dataset
|	|	`-- healthy
|	|		`-- subject_01
|	|			|-- part_1
|	|			|	|-- imu\							<-- Binary IMU data
|	|			|	`-- manual_annotation_z_level.csv	<-- Contains ground truth timings and surface labels
|	|			`-- part_2
|   `-- stairs_up										<-- output data from the NEWBEE dataset (on stairs up sections)
|-- python_code
|   |-- Toolboxes\
|   |-- src\
|   |-- tests\
|   |-- data_analysis.py
|   |-- newbee_runner.py
|   `-- osf_runner.py
`-- requirements.txt
```

---

## Getting Started

**Recommended Python version:** 3.11

1. Clone the repository and navigate to the project root.
2. Ensure the NEWBEE and OSF databases are downloaded and placed in the `data` directory.
3. Install the required dependencies:

```shell
StairsUp> pip install -r requirements.txt
```

---

## Pipeline Overview

### 1. Running the Data Extraction and Processing Code

The pipeline uses separate runners for the distinct datasets to extract the real ground-truth events and evaluate them through the toolboxes. Run the `osf_runner` and `newbee_runner` located in the `python_code` directory to create output `.mat` files later used for data analysis.

```shell
StairsUp> python -m python_code.newbee_runner
StairsUp> python -m python_code.osf_runner
```

### 2. Hyperparameter Optimization

You can optimize the PyShoe zero-velocity threshold (`G`) parameter using Leave-One-Subject-Out (LOSO) cross-validation. This targets the `stairs_up` segments specifically to maximize the F1-Score.

```shell
StairsUp> python -m python_code.src.algorithms.optimize_pyshoe
```

Note: You can optimize a specific detector using the `-d` flag (e.g., `python -m python_code.src.algorithms.optimize_pyshoe -d mbgtd`).

### 3. Data Analysis and Visualization

Run the `data_analysis.py` script to generate summary statistics (Precision, Recall, F1-Score, MAE) and match predictions to the real ground truth using Hungarian matching.

```shell
StairsUp> python python_code/data_analysis.py
```

You can also generate detector-specific plots showcasing the worst, median, and best predictions using the `-d` switch:

```shell
StairsUp> python python_code/data_analysis.py -d shoe
```

---

## Using on Custom Data

To apply the PyShoe pipeline to other data segments, utilize the `pyshoe_process_single_file` function found in the `pyshoe_export.py` script:

```python
from python_code.Toolboxes.PyShoe.pyshoe_export import pyshoe_process_single_file

def pyshoe_process_single_file(
    segment: pd.DataFrame, 
    has_gravity: bool,
    course: str, 
    id: str, 
    clip_id: int,
    true_hs_r: pd.DataFrame | np.ndarray, 
    true_fo_r: pd.DataFrame | np.ndarray, 
    true_hs_l: pd.DataFrame | np.ndarray, 
    true_fo_l: pd.DataFrame | np.ndarray, 
    output_path: Path,
    fs: float
) -> str:
```
