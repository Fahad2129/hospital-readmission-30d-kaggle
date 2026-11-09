"""Create the final selected v19 rank blend.

Inputs are the base blend produced by ``run_best_blend.py`` and the shallow
LightGBM/GAM candidates produced by ``train_stumps.py`` and
``train_enriched_gam.py``.  Blending ranks (rather than raw probabilities)
keeps the final ordering robust to calibration differences between models.
"""
from pathlib import Path

import pandas as pd
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "working"
FINAL = ROOT / "final_submissions_and_artifacts"


def ranks(values):
    """Return normalized average ranks while preserving within-model ordering."""
    return rankdata(values, method="average") / len(values)


base = pd.read_csv(WORK / "submission.csv")
stump = pd.read_csv(FINAL / "stump_submission.csv")
gam = pd.read_csv(FINAL / "enriched_gam_submission.csv")

for candidate, name in [(stump, "stump"), (gam, "enriched GAM")]:
    if candidate.encounter_id.tolist() != base.encounter_id.tolist():
        raise ValueError(f"{name} predictions are not aligned to the base submission IDs.")

prediction = (
    0.60 * ranks(base.readmit_30d)
    + 0.10 * ranks(stump.readmit_30d)
    + 0.30 * ranks(gam.readmit_30d)
)
submission = pd.DataFrame(
    {"encounter_id": base.encounter_id, "readmit_30d": prediction}
)
assert submission.readmit_30d.notna().all()
assert submission.readmit_30d.between(0, 1).all()

output = FINAL / "submission_v19_rank_old60_stump10_gam30.csv"
submission.to_csv(output, index=False)
print(f"Wrote {output} with {len(submission):,} encounter predictions.")
