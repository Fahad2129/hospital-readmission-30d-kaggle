"""Compare v11 LightGBM with fold-safe target-encoded identifiers."""
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT=Path(__file__).resolve().parents[1]
V11=ROOT/"final_submissions_and_artifacts"
X=pd.read_csv(V11/"final_dataset.csv.gz",compression="gzip")
tr=pd.read_csv(ROOT/"working"/"train.csv"); te=pd.read_csv(ROOT/"working"/"test.csv")
sample=pd.read_csv(ROOT/"working"/"sample_submission.csv")
T=tr.merge(X,on="encounter_id",how="left",validate="one_to_one")
S=te.merge(X,on="encounter_id",how="left",validate="one_to_one")
y=T.readmit_30d.to_numpy()
cats=[c for c in X if not pd.api.types.is_numeric_dtype(X[c]) and c!="encounter_id"]
cats=[c for c in cats if c!="attending_id"]  # v11 exclusion, tested as numeric encoded signal below
for c in cats: T[c]=T[c].astype("category"); S[c]=S[c].astype("category")
drop=("hrs_","visits_","n_transfers","first_unit","last_unit","icu_","had_ed","ward_after","proc_","n_proc")
feats=[c for c in T if c not in ("encounter_id","readmit_30d","attending_id") and not c.startswith(drop)]
ids=[c for c in ["attending_id","hospital_id","service_line","attending_specialty","primary_chapter"] if c in T]
folds=list(StratifiedKFold(5,shuffle=True,random_state=0).split(T,y))
params=dict(objective="binary",verbose=-1,n_jobs=2,bagging_fraction=.8,bagging_freq=1,
            cat_smooth=50,max_cat_to_onehot=8,num_leaves=7,min_child_samples=200,
            feature_fraction=.3,lambda_l2=20,learning_rate=.05)
base_o=np.zeros(len(T)); enc_o=np.zeros(len(T)); base_t=np.zeros(len(S)); enc_t=np.zeros(len(S))
for f,(a,b) in enumerate(folds):
    trn=T.iloc[a].copy(); val=T.iloc[b].copy(); tst=S.copy()
    # OOF encodings use only each fold's training labels; test encodings use all labels.
    prior=float(y[a].mean()); test_prior=float(y.mean())
    for c in ids:
        key=trn[c].astype("string").fillna("__NA__")
        stats=pd.DataFrame({"k":key.to_numpy(),"y":y[a]}).groupby("k").y.agg(["sum","count"])
        mp=(stats["sum"]+prior*30)/(stats["count"]+30)
        # Leave-one-out target encoding for the rows used to train this fold.
        # This prevents each row's own label from entering its encoded feature.
        trn[f"te_{c}"]=((key.map(stats["sum"]).to_numpy()-y[a]+prior*30)/
                          (key.map(stats["count"]).to_numpy()-1+30))
        val[f"te_{c}"]=val[c].astype("string").fillna("__NA__").map(mp).fillna(prior).to_numpy()
        keyall=T[c].astype("string").fillna("__NA__")
        statsall=pd.DataFrame({"k":keyall.to_numpy(),"y":y}).groupby("k").y.agg(["sum","count"])
        mpall=(statsall["sum"]+test_prior*30)/(statsall["count"]+30)
        tst[f"te_{c}"]=tst[c].astype("string").fillna("__NA__").map(mpall).fillna(test_prior).to_numpy()
    add=[f"te_{c}" for c in ids]; allfeat=feats+add
    for mode,train_df,valid_df,test_df,acc_o,acc_t in [
        ("base",trn,val,S,base_o,base_t),
        ("target",trn,val,tst,enc_o,enc_t)]:
        ff=feats if mode=="base" else allfeat
        model=lgb.train({**params,"seed":100+f},lgb.Dataset(train_df[ff],label=y[a]),num_boost_round=3000,
                        valid_sets=[lgb.Dataset(valid_df[ff],label=y[b])],
                        callbacks=[lgb.early_stopping(150,verbose=False),lgb.log_evaluation(0)])
        acc_o[b]=model.predict(valid_df[ff],num_iteration=model.best_iteration)
        acc_t+=model.predict(test_df[ff],num_iteration=model.best_iteration)/5
        print(f"fold={f+1} mode={mode} best_iter={model.best_iteration}",flush=True)
    print(f"running AUC base={roc_auc_score(y,base_o):.6f} target={roc_auc_score(y,enc_o):.6f}",flush=True)
auc_base=roc_auc_score(y,base_o); auc_enc=roc_auc_score(y,enc_o)
print("FINAL",auc_base,auc_enc,flush=True)
np.save(V11/"candidate_base_oof.npy",base_o); np.save(V11/"candidate_encoded_oof.npy",enc_o)
if auc_enc > auc_base:
    sub=pd.DataFrame({"encounter_id":te.encounter_id,"readmit_30d":enc_t})
    assert sub.encounter_id.tolist()==sample.encounter_id.tolist()
    assert sub.readmit_30d.notna().all() and sub.readmit_30d.between(0,1).all()
    sub.to_csv(V11/"submission_v12_target_encoded.csv",index=False)
    print("WROTE",V11/"submission_v12_target_encoded.csv",flush=True)
else:
    print("No candidate written: target encoding did not improve this CV run.",flush=True)
