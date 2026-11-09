# Final submission artifacts

The canonical CSVs are kept in [`../final_submissions_and_artifacts/`](../final_submissions_and_artifacts/), alongside the saved out-of-fold and test prediction arrays used during blending.

| File | Private ROC-AUC | Public ROC-AUC | Status |
| --- | ---: | ---: | --- |
| `submission_v19_rank_old60_stump10_gam30.csv` | 0.75739 | 0.74790 | Best of the three final selections |
| `submission_v17_conservative_gam.csv` | 0.75733 | 0.74792 | Final selection |
| `submission_v14_rankblend.csv` | 0.75720 | 0.74792 | Final selection |

Each submission has the Kaggle-required columns:

```text
encounter_id,readmit_30d
```

`readmit_30d` contains probabilities, not hard 0/1 predictions.  `submission_v18_oof_stack.csv` is retained as an additional experiment; despite its marginally higher displayed private score (0.75740), it was not among the three selected final submissions.

The documented v19 reproduction code is [`../working/create_v19_rank_blend.py`](../working/create_v19_rank_blend.py).  Its component-model scripts save `stump_submission.csv` and `enriched_gam_submission.csv` beside these final artifacts before the rank blend is created.

For the grader-facing end-to-end path, run [`../final_pipeline.ipynb`](../final_pipeline.ipynb). It writes its regenerated submission to `reproducibility_runs/` rather than replacing any canonical submission artifact.
