# Hospital Readmission Within 30 Days

Course project for Introduction to Machine Learning (IBA Karachi).  The task is to predict `readmit_30d` for each test-set hospital encounter.  Predictions are evaluated with ROC-AUC.

## Final result

Three submissions were selected for the final Kaggle evaluation.  The best selected result was **0.75739 private ROC-AUC**:

| Submission | Private ROC-AUC | Public ROC-AUC | Final selection |
| --- | ---: | ---: | --- |
| `submission_v19_rank_old60_stump10_gam30.csv` | 0.75739 | 0.74790 | Yes — best selected |
| `submission_v17_conservative_gam.csv` | 0.75733 | 0.74792 | Yes |
| `submission_v14_rankblend.csv` | 0.75720 | 0.74792 | Yes |

The three files are in [`Current best submission v11/`](Current%20best%20submission%20v11/).  `submission_v18_oof_stack.csv` achieved 0.75740 private ROC-AUC, but it was not one of the three submissions selected for final evaluation.

## Repository contents

| Location | Contents |
| --- | --- |
| [`working/`](working/) | Raw competition CSV tables, thirteen ordered notebooks, utility scripts, and the project data dictionary. |
| [`Current best submission v11/`](Current%20best%20submission%20v11/) | Final submission candidates, saved OOF/test predictions used by blends, feature-column appendix, and earlier pipeline scripts. |
| [`data/README.md`](data/README.md) | Raw-data inventory and expected file names. |
| [`submissions/README.md`](submissions/README.md) | Final-submission inventory and verification notes. |

## Reproducing the pipeline

1. Create and activate a Python environment, then install the packages:

   ```bash
   pip install -r requirements.txt
   ```

2. Confirm the raw tables listed in [`data/README.md`](data/README.md) are present in `working/`.
3. Open the notebooks in numerical order.  Each notebook explains its purpose and reads artifacts produced by earlier notebooks.  Set `KAGGLE_DATA` to the raw-table folder and `KAGGLE_WORK` to a writable output folder if prompted in a notebook.
4. Run notebooks `01` through `13` in order.  The final stacking notebook creates submission candidates from out-of-fold predictions.  The saved prediction arrays under `Current best submission v11/` document the blends used for the final candidates.

### Exact v19 reproduction path

The final selected file is reproducible from the raw tables through the following focused path (after the feature-engineering notebooks have produced `features2.parquet` and `features3.parquet`):

```bash
python working/run_best_blend.py
python working/train_stumps.py
python working/train_enriched_gam.py
python working/create_v19_rank_blend.py
```

The last step rank-averages the three aligned probability vectors with weights 60% base blend, 10% shallow boosted stump model, and 30% enriched GAM-style boosted-stump model.  It writes `Current best submission v11/submission_v19_rank_old60_stump10_gam30.csv`.

The notebooks deliberately use out-of-fold predictions for blend/stack model selection and keep the test set separate until producing submission probabilities.

## Data and reproducibility notes

- The repository contains course-competition data only.  Do not add patient-identifying data, credentials, or any non-competition data.
- `final_columns.csv` is included as the report appendix describing the modeling columns and their source tables.
- Generated caches, model binaries, notebook checkpoints, and large intermediate feature matrices are excluded through `.gitignore`; they can be regenerated from the raw tables and notebooks.
- Runtime is approximately 4–5 hours for the full notebook sequence and requires roughly 7 GB RAM, according to the project run notes.

## Report handoff checklist

- [x] Raw source tables are retained.
- [x] Ordered notebooks document the pipeline from raw CSVs to submissions.
- [x] The three final selected submission CSVs are retained.
- [x] `final_columns.csv` is retained for the required column/source appendix.
- [x] Dependency and run instructions are documented.
