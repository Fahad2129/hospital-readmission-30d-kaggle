"""Train notebook-03 enriched-feature boosted stumps for blending."""
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT=Path(__file__).resolve().parents[1]; V=ROOT/"final_submissions_and_artifacts"; D=ROOT/"working"
X=pd.read_parquet(D/"features3.parquet")
tr=pd.read_csv(D/"train.csv"); te=pd.read_csv(D/"test.csv"); ss=pd.read_csv(D/"sample_submission.csv")
T=tr.merge(X,on="encounter_id",how="left",validate="one_to_one"); S=te.merge(X,on="encounter_id",how="left",validate="one_to_one")
y=T.readmit_30d.to_numpy(); cats=[c for c in X if not pd.api.types.is_numeric_dtype(X[c]) and c!="encounter_id"]
for c in cats: T[c]=T[c].astype("category"); S[c]=S[c].astype("category")
drop=("hrs_","visits_","n_transfers","first_unit","last_unit","icu_","had_ed","ward_after","proc_","n_proc")
features=[c for c in T if c not in ("encounter_id","readmit_30d") and not c.startswith(drop)]
params=dict(objective="binary",learning_rate=.1,num_leaves=2,max_depth=1,min_child_samples=100,
            feature_fraction=.5,lambda_l2=20,max_bin=63,bagging_freq=1,bagging_fraction=.8,
            verbosity=-1,n_jobs=2)
o=np.zeros(len(T)); p=np.zeros(len(S)); folds=list(StratifiedKFold(5,shuffle=True,random_state=0).split(T,y))
for f,(a,b) in enumerate(folds):
    model=lgb.train({**params,"seed":f},lgb.Dataset(T.iloc[a][features],label=y[a]),5000,
                    valid_sets=[lgb.Dataset(T.iloc[b][features],label=y[b])],
                    callbacks=[lgb.early_stopping(200,verbose=False),lgb.log_evaluation(0)])
    o[b]=model.predict(T.iloc[b][features],num_iteration=model.best_iteration)
    p+=model.predict(S[features],num_iteration=model.best_iteration)/5
    print(f"fold={f+1} iter={model.best_iteration}",flush=True)
print("enriched GAM OOF",roc_auc_score(y,o),flush=True)
np.save(V/"enriched_gam_oof.npy",o); np.save(V/"enriched_gam_test.npy",p)
sub=pd.DataFrame({"encounter_id":te.encounter_id,"readmit_30d":p})
assert sub.encounter_id.tolist()==ss.encounter_id.tolist() and sub.readmit_30d.between(0,1).all() and sub.readmit_30d.notna().all()
sub.to_csv(V/"enriched_gam_submission.csv",index=False)
