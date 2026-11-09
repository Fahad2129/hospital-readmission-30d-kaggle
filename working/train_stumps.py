"""Train a diverse shallow LightGBM model for OOF blend evaluation."""
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT=Path(__file__).resolve().parents[1]; V=ROOT/"Current best submission v11"; D=ROOT/"working"
X=pd.read_csv(V/"final_dataset.csv.gz",compression="gzip")
tr=pd.read_csv(D/"train.csv"); te=pd.read_csv(D/"test.csv"); ss=pd.read_csv(D/"sample_submission.csv")
T=tr.merge(X,on="encounter_id",how="left",validate="one_to_one"); S=te.merge(X,on="encounter_id",how="left",validate="one_to_one")
y=T.readmit_30d.to_numpy(); cat=[c for c in X if not pd.api.types.is_numeric_dtype(X[c]) and c not in ["encounter_id","attending_id"]]
for c in cat: T[c]=T[c].astype("category"); S[c]=S[c].astype("category")
features=[c for c in T if c not in ("encounter_id","readmit_30d","attending_id")]
params=dict(objective="binary",verbosity=-1,n_jobs=1,learning_rate=.03,num_leaves=2,max_depth=1,
            min_child_samples=250,feature_fraction=.7,bagging_fraction=.85,bagging_freq=1,
            lambda_l1=1,lambda_l2=20,max_bin=127)
o=np.zeros(len(T)); p=np.zeros(len(S)); splits=list(StratifiedKFold(5,shuffle=True,random_state=0).split(T,y))
for f,(a,b) in enumerate(splits):
    model=lgb.train({**params,"seed":101+f},lgb.Dataset(T.iloc[a][features],label=y[a]),5000,
        valid_sets=[lgb.Dataset(T.iloc[b][features],label=y[b])],
        callbacks=[lgb.early_stopping(250,verbose=False),lgb.log_evaluation(0)])
    o[b]=model.predict(T.iloc[b][features],num_iteration=model.best_iteration)
    p+=model.predict(S[features],num_iteration=model.best_iteration)/5
    print(f"fold={f+1} iter={model.best_iteration}",flush=True)
print("stump OOF",roc_auc_score(y,o),flush=True)
np.save(V/"stump_oof.npy",o); np.save(V/"stump_test.npy",p)
sub=pd.DataFrame({"encounter_id":te.encounter_id,"readmit_30d":p})
assert sub.encounter_id.tolist()==ss.encounter_id.tolist() and sub.readmit_30d.between(0,1).all() and sub.readmit_30d.notna().all()
sub.to_csv(V/"stump_submission.csv",index=False)
