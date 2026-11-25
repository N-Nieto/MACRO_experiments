# Machine learning models for early prognosis prediction in cardiogenic shock

## About

The Forschungszentrum Jülich Machine Learning Library

It is currently being developed and maintained at the [Applied Machine Learning](https://www.fz-juelich.de/en/inm/inm-7/research-groups/applied-machine-learning-aml) group at [Forschungszentrum Juelich](https://www.fz-juelich.de/en), Germany.


## Overview

**Code to reproduce the results in "Machine learning models for early prognosis prediction in cardiogenic shock".** This repository has all script used to generated the results presented in the paper. 

# Steps to reproduce the results

**Clone the repository:**

```bash
git clone https://github.com/N-Nieto/MACRO_experiments.git
cd MACRO_experiments
```

## Python environment

The environment is managed with [uv](https://docs.astral.sh/uv/): dependencies are declared in
`pyproject.toml` and the exact versions are pinned in `uv.lock` (Python >= 3.13).

```bash
uv sync                 # create .venv with the locked versions (incl. pytest, ipykernel)
uv run pytest           # run the tests (synthetic data only, no patient data needed)
```

Run each script from its own folder, e.g.

```bash
cd code/Admission_model && uv run python Admission_model_experiment.py
```

To update all packages to their latest releases: `uv lock --upgrade && uv sync`.


## Data location

All scripts read the data from the folders defined in `code/lib/paths.py`:
`<parent of CULPRIT_project>/data/CULPRIT` (CULPRIT-SHOCK export and `Shorten_table.xlsx`) and
`<parent of CULPRIT_project>/data/eicu-collaborative-research-database-2.0` (raw eICU tables;
`preprocessed_MACRO/` is created by `code/eICU_extract_CS_patients.py`).
To use another folder, set `MACRO_DATA_DIR`, e.g. `MACRO_DATA_DIR=/path/to/data uv run python ...`.

## Data access
### CULPRIT-SCHOCK dataset
The dataset is not publicly available but access can be granted upon request. Please contact the corresponding author [n.nieto@fz-juelich.de]

### eICU
eICU dataset is publicly available and the code to extract the used cohort from the raw data is presented in 
`/code/eICU_extract_CS_patients.py`

## Admission model

The Admission model aim to replicate the real life scenario were the patient suffers a Cardiogenic schock (CS) and only immediately available features (like age, height, weight and comorbilities) can be used. Importantly this model does not use any laboratory value data, as is assume this data will take hours to be collected.

## Full model
The Full model incorporates the lab values acquired in the first 24 hs after CS. 

## Score models
The traditional risk scores were benchmarked, building a Logistic Regression (LG) that uses a single score as input feature and predicts the 30-days mortality as output. BOSMAN score, IAMP II, 

## Extra analysis
To compare with the XGBoost architecture used in the Admission and Full models, a Logistic Regression model was developed. The model was only developed for the admission features, as LG can not handle missing values. 

# [Web demo](https://macro.appliedml.org/)
We developed a web demo to showcase how the models could be used. Note that the demo only allows the user to load pre-existing synthetic data, developed to showcase the model behavior.

## Citation
Soon

## Licensing

MACRO is released under the AGPL v3 license:

MACRO, FZJuelich AML machine learning library.
Copyright (C) 2020, authors of MACRO.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.