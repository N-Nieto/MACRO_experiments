"""
Data and output locations shared by all scripts.

Default layout (the repository sits two levels below the project folder):

    MODS_project/
        data/
            CULPRIT/                                    CULPRIT-data_20210407.xlsx, Shorten_table.xlsx
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

# MACRO_QUICK=1 (smoke runs, see lib/experiment_config.py) writes to output_quick/
OUTPUT_DIR = REPO_DIR / (
    "output_quick" if os.environ.get("MACRO_QUICK", "0") == "1" else "output"
)
