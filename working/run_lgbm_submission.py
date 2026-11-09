"""Train a reproducible five-fold LightGBM model and write submission.csv."""
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

D=Path(__file__).resolve().parent
X=pd.read_parquet(D/"features2.parquet")
tr=pd.read_csv(D/"train.csv"); te=pd.read_csv(D/"test.csv")
catc=[c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c]) and c!="encounter_id"]
for c in catc: X[c]=X[c].astype("category")
T=X.merge(tr,on="encounter_id",how="inner"); S=te.merge(X,on="encounter_id",how="left")
y=T.readmit_30d.to_numpy()
drop=("hrs_","visits_","n_transfers","first_unit","last_unit","icu_","had_ed","ward_after","proc_","n_proc")
feats=[c for c in T if c not in ("encounter_id","readmit_30d") and not c.startswith(drop)]
params=dict(objective="binary",verbose=-1,n_jobs=2,bagging_fraction=.8,bagging_freq=1,
            cat_smooth=50,max_cat_to_onehot=8,num_leaves=7,min_child_samples=200,
            feature_fraction=.3,lambda_l2=20,learning_rate=.04,seed=101)
folds=StratifiedKFold(5,shuffle=True,random_state=0)
oof=np.zeros(len(T)); pred=np.zeros(len(S))
for f,(a,b) in enumerate(folds.split(T,y)):
    dtrain=lgb.Dataset(T.iloc[a][feats],label=y[a],free_raw_data=True)
    dvalid=lgb.Dataset(T.iloc[b][feats],label=y[b],reference=dtrain,free_raw_data=True)
    model=lgb.train(params,dtrain,num_boost_round=4000,valid_sets=[dvalid],
                    callbacks=[lgb.early_stopping(200,verbose=False),lgb.log_evaluation(0)])
    oof[b]=model.predict(T.iloc[b][feats],num_iteration=model.best_iteration)
    pred+=model.predict(S[feats],num_iteration=model.best_iteration)/5
    print(f"fold {f+1}/5 best_iter={model.best_iteration}",flush=True)
print("OOF ROC-AUC",round(roc_auc_score(y,oof),6),flush=True)
sample=pd.read_csv(D/"sample_submission.csv")
sub=pd.DataFrame({"encounter_id":te.encounter_id,"readmit_30d":pred})
assert sub.encounter_id.tolist()==sample.encounter_id.tolist()
assert sub.readmit_30d.notna().all() and sub.readmit_30d.between(0,1).all()
sub.to_csv(D/"submission.csv",index=False)
np.save(D/"lgbm_oof.npy",oof)
print("Saved",D/"submission.csv","rows",len(sub),"mean",round(float(pred.mean()),6),flush=True)
