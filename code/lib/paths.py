"""
Data and output locations shared by all scripts.

Default layout (the repository sits two levels below the project folder):

    MODS_project/
        data/
            CULPRIT/                                    CULPRIT-data_20210407.xlsx
            eicu-collaborative-research-database-2.0/   raw eICU tables
                preprocessed_MACRO/                     written by eICU_extract_CS_patients.py
        CULPRIT_project/
            MACRO_experiments/                          this repository

Set the environment variable MACRO_DATA_DIR to use another data folder.
"""

import os
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("MACRO_DATA_DIR", REPO_DIR.parents[1] / "data"))

CULPRIT_DIR = DATA_DIR / "CULPRIT"
EICU_DIR = DATA_DIR / "eicu-collaborative-research-database-2.0"
EICU_PREPROCESSED_DIR = EICU_DIR / "preprocessed_MACRO"

# Results of the current run go to output/<RUN_NAME>/ (the "output/..." folders
# named in the script docstrings are relative to OUTPUT_DIR). MACRO_QUICK=1
# (smoke runs, see lib/experiment_config.py) writes to output_quick/<RUN_NAME>/.
RUN_NAME = "rev2"
OUTPUT_DIR = (
    REPO_DIR
    / ("output_quick" if os.environ.get("MACRO_QUICK", "0") == "1" else "output")
    / RUN_NAME
)

# Missing-value baseline results are kept next to their script
# (missing_value_analysis/output/rev2/missing_values for the full run)
MISSING_VALUES_DIR = (
    REPO_DIR
    / "code"
    / "1-model_training"
    / "missing_value_analysis"
    / OUTPUT_DIR.relative_to(REPO_DIR)
    / "missing_values"
)
