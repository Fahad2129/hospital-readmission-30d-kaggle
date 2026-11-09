# Hospital readmission challenge — notebooks

Thirteen notebooks that contain all the code produced in this project, in the order it was run.
Each notebook starts with a plain-language explanation and the results we observed. **No outputs are stored in the files**; the numbers in the text come from our runs.

## Before you start
1. Unzip the competition data into one folder (it must contain `train.csv`, `test.csv`, `sample_submission.csv`, `index_encounters.csv`, `patients.csv` and the other tables).
2. In the first code cell of each notebook set `DATA` (that folder) and `WORK` (an empty folder for results). All notebooks must use the **same** `WORK` folder, since they pass files to each other.
   You can instead set the environment variables `KAGGLE_DATA` and `KAGGLE_WORK`.
3. Packages: `pip install lightgbm xgboost catboost scikit-learn pandas pyarrow scipy interpret jupyter`. Needs ~7 GB RAM; 2 CPU cores is enough.

## Order and what each does
| # | Notebook | Needs | Produces | Approx. time |
|---|---|---|---|---|
| 01 | data_profiling | CSVs | (exploration only) | 5 min |
| 02 | feature_engineering | CSVs | `features2.parquet` | 3 min |
| 03 | extra_features | 02 | `features3..6.parquet` | 6 min |
| 04 | lightgbm_baseline_tuning_ablation | 02, 03 | `search.json`, `oof_*` | 1 h |
| 05 | first_submission_lgbm_lr_blend | 02 | **`submission.csv`** | 15 min |
| 06 | enriched_feature_models | 03, 04 | `n_*` predictions | 40 min |
| 07 | xgboost_and_bagging | 02, 05 | `xgb_*` predictions | 20 min |
| 08 | catboost_mlp_extratrees_ensemble | 02, 05, 07 | `submission_v2_lgb_lr_cat.csv` | 10 min |
| 09 | naive_bayes_random_forest | 02 | **NB, RF and NB+RF ensemble submissions** | 12 min |
| 10 | linear_regression_and_svm | 02 | **linear-regression and SVM submissions** | 15 min |
| 11 | additive_models_gam_ebm_and_variants | 03, 04 | `v3_*` predictions | 45 min |
| 12 | negative_results_and_diagnostics | 03–06, 11 | `v3_gam_ix/ms4/f6`, `v3_woe_*` | 45 min |
| 13 | final_stacking_and_submissions | 05–12 | **`submission_v3_additive_stack.csv`** (recommended), `submission_v3_hillclimb.csv`, `submission_v4_stack.csv` | 20 min |

Total for a full re-run: roughly 4–5 hours. If you only want the best submission, you need 02 → 03 → 04 (the search part) → 05 → 06 → 07 → 08 → 09 → 10 → 11 → 12 → 13 because the stack uses all saved predictions; the saved prediction files from our run can also be dropped straight into `WORK` to skip straight to 13.

## Honest notes
- Run order matters; later notebooks read files written by earlier ones.
- The 5-fold split (`random_state=0`) is the same everywhere. Never change it in one notebook only, because the stacking in 13 relies on all out-of-fold predictions being aligned.
- Results can differ in the 4th decimal on another machine or library version (LightGBM threading).
- Notebook 03 step 3.4 and notebook 12 mention an experiment (LightGBM on `features6`) that we never finished; the additive-model result is reported instead.
- Notebook 04's random search ran 6 of a planned 12 trials; change `N_TRIALS` to rerun more.
