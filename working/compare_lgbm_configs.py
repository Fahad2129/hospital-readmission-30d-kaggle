"""Compare v11's two alternate LightGBM configs on its fixed 5-fold split."""
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

ROOT=Path(__file__).resolve().parents[1]; V11=ROOT/"Current best submission v11"; DATA=ROOT/"working"
X=pd.read_csv(V11/"final_dataset.csv.gz",compression="gzip")
tr=pd.read_csv(DATA/"train.csv"); te=pd.read_csv(DATA/"test.csv"); ss=pd.read_csv(DATA/"sample_submission.csv")
cat=[c for c in X if not pd.api.types.is_numeric_dtype(X[c]) and c!="encounter_id"]
T=X.copy(); S=X[X.encounter_id.isin(te.encounter_id)].set_index("encounter_id").loc[te.encounter_id].reset_index()
for c in cat: T[c]=T[c].astype("category"); S[c]=S[c].astype("category")
T=T.merge(tr,on="encounter_id",how="inner",validate="one_to_one"); y=T.readmit_30d.to_numpy()
feats=[c for c in T if c not in ("encounter_id","readmit_30d","attending_id")]
folds=list(StratifiedKFold(5,shuffle=True,random_state=0).split(T,y))
base=dict(objective="binary",verbose=-1,n_jobs=1,bagging_fraction=.8,bagging_freq=1,cat_smooth=50,max_cat_to_onehot=8)
cfgs=[dict(num_leaves=7,min_child_samples=200,feature_fraction=.3,lambda_l2=20,learning_rate=.05),
      dict(num_leaves=4,max_depth=2,min_child_samples=300,feature_fraction=.3,lambda_l2=30,learning_rate=.07),
      dict(num_leaves=5,min_child_samples=300,feature_fraction=.2,lambda_l2=50,learning_rate=.07,max_bin=63)]
oofs=[]; tests=[]
for ci,cfg in enumerate(cfgs,1):
    o=np.zeros(len(T)); t=np.zeros(len(S)); p={**base,**cfg}
    for f,(a,b) in enumerate(folds):
        p["seed"]=100+f
        m=lgb.train(p,lgb.Dataset(T.iloc[a][feats],label=y[a]),3000,
                    valid_sets=[lgb.Dataset(T.iloc[b][feats],label=y[b])],
                    callbacks=[lgb.early_stopping(150,verbose=False),lgb.log_evaluation(0)])
        o[b]=m.predict(T.iloc[b][feats],num_iteration=m.best_iteration)
        t+=m.predict(S[feats],num_iteration=m.best_iteration)/5
        print(f"cfg{ci} fold{f+1} iter={m.best_iteration}",flush=True)
    print(f"cfg{ci} AUC={roc_auc_score(y,o):.6f}",flush=True)
    np.save(V11/f"cfg{ci}_oof.npy",o); np.save(V11/f"cfg{ci}_test.npy",t); oofs.append(o); tests.append(t)
base_o=oofs[0]; base_t=tests[0]
def logit(p): p=np.clip(p,1e-6,1-1e-6); return np.log(p/(1-p))
print("blend OOF:")
for w in [0,.25,.5,.75,1]:
    z=(1-w)*logit(base_o)+w*np.mean([logit(v) for v in oofs[1:]],axis=0)
    print(f"alt_weight={w:.2f} AUC={roc_auc_score(y,z):.6f}",flush=True)
# choose the best of these few predeclared weights; submit only when better than baseline
best=(roc_auc_score(y,logit(base_o)),0.0)
for w in [.25,.5,.75,1.0]:
    z=(1-w)*logit(base_o)+w*np.mean([logit(v) for v in oofs[1:]],axis=0)
    auc=roc_auc_score(y,z)
    if auc>best[0]: best=(auc,w)
print("best",best,"baseline",roc_auc_score(y,logit(base_o)),flush=True)
if best[1]>0:
    w=best[1]; pred=(1-w)*logit(base_t)+w*np.mean([logit(v) for v in tests[1:]],axis=0)
    pred=1/(1+np.exp(-pred))
    sub=pd.DataFrame({"encounter_id":te.encounter_id,"readmit_30d":pred})
    assert sub.encounter_id.tolist()==ss.encounter_id.tolist() and sub.readmit_30d.between(0,1).all() and sub.readmit_30d.notna().all()
    sub.to_csv(V11/"submission_v12_lgbm_blend.csv",index=False)
    print("wrote",V11/"submission_v12_lgbm_blend.csv",flush=True)
