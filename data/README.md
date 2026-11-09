# Raw data inventory

The project’s raw CSV files are intentionally retained in [`../working/`](../working/) so the notebooks can run without changing their original paths.  They are the competition-provided source tables:

- `train.csv` — training encounter IDs and `readmit_30d` target
- `test.csv` — test encounter IDs
- `sample_submission.csv` — required Kaggle submission schema
- `index_encounters.csv` — encounter-level index table
- `patients.csv` and `hospitals.csv` — patient and hospital attributes
- `diagnoses.csv` and `icd10_lookup.csv` — diagnosis records and code metadata
- `lab_results.csv` and `vitals.csv` — time-stamped clinical measurements
- `medications.csv`, `procedures.csv`, and `unit_transfers.csv` — treatment/activity tables
- `prior_utilization.csv` — historical utilization records
- `discharge_planning.csv` — discharge-planning attributes
- `data_dictionary.csv` — column definitions

`final_columns.csv` is not a raw source table; it is the final-model feature appendix needed for the report.

If this repository is shared without the raw data, download the competition data through Kaggle, place every CSV above in `working/`, and preserve the filenames exactly.
