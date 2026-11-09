from pathlib import Path
import os
DATA = str(Path(__file__).resolve().parent)
WORK = DATA
import pandas as pd, numpy as np, lightgbm as lgb, sys, json
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
feat_file=f'{WORK}/features2.parquet'; drop_prefix=("hrs_","visits_","n_transfers","first_unit","last_unit","icu_","had_ed","ward_after","proc_","n_proc")
D=DATA
X=pd.read_parquet(feat_file); tr=pd.read_csv(f'{D}/train.csv'); te=pd.read_csv(f'{D}/test.csv')
catc=[c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c]) and c!='encounter_id']
Xg=X.copy()
for c in catc: Xg[c]=Xg[c].astype('category')
T=Xg.merge(tr,on='encounter_id'); S=te.merge(Xg,on='encounter_id',how='left'); y=T.readmit_30d.values
assert len(S)==len(te) and S.encounter_id.tolist()==te.encounter_id.tolist()
feats=[c for c in T.columns if c not in('encounter_id','readmit_30d') and not c.startswith(drop_prefix)]
base=dict(objective='binary',verbose=-1,n_jobs=2,bagging_fraction=0.8,bagging_freq=1,cat_smooth=50,max_cat_to_onehot=8)
cfgs=[dict(num_leaves=7,min_child_samples=200,feature_fraction=0.3,lambda_l2=20,learning_rate=0.04),
      dict(num_leaves=4,max_depth=2,min_child_samples=300,feature_fraction=0.3,lambda_l2=30,learning_rate=0.05),
      dict(num_leaves=5,min_child_samples=300,feature_fraction=0.2,lambda_l2=50,learning_rate=0.05,max_bin=63)]
folds=list(StratifiedKFold(5,shuffle=True,random_state=0).split(T,y))
oof=[];tst=[]
for i,c in enumerate(cfgs):
    p={**base,**c}; o=np.zeros(len(T)); t=np.zeros(len(S))
    for f,(a,b) in enumerate(folds):
        for seed in (1,2):
            p['seed']=seed*100+f
            m=lgb.train(p,lgb.Dataset(T.loc[a,feats],y[a]),4000,valid_sets=[lgb.Dataset(T.loc[b,feats],y[b])],callbacks=[lgb.early_stopping(200,verbose=False)])
            o[b]+=m.predict(T.loc[b,feats],num_iteration=m.best_iteration)/2
            t+=m.predict(S[feats],num_iteration=m.best_iteration)/10   # 5 folds x 2 seeds
    print('cfg',i,'OOF AUC %.5f'%roc_auc_score(y,o),flush=True); oof.append(o); tst.append(t)
# logistic regression
num=[c for c in feats if pd.api.types.is_numeric_dtype(T[c]) and T[c].notna().sum()>0 and T[c].nunique()>1]
cat=[c for c in feats if c in catc]
A=pd.concat([T[num],S[num]]); miss=A.columns[A.isna().mean()>0.01]
lo,hi=T[num].quantile(.005),T[num].quantile(.995)
def prep(df):
    N=df[num].clip(lo,hi,axis=1); N=N.fillna(T[num].clip(lo,hi,axis=1).median())
    M=df[miss].isna().astype(int).add_suffix('_na'); C=df[cat].astype(object).fillna('NA')
    return pd.concat([N,M,C],axis=1)
Zall=pd.get_dummies(pd.concat([prep(T),prep(S)],ignore_index=True),columns=cat,dtype=float).astype(float)
Zt,Zs=Zall.iloc[:len(T)].values,Zall.iloc[len(T):].values
o=np.zeros(len(T)); t=np.zeros(len(S))
for a,b in folds:
    sc=StandardScaler().fit(Zt[a]); m=LogisticRegression(C=0.02,max_iter=500).fit(sc.transform(Zt[a]),y[a])
    o[b]=m.predict_proba(sc.transform(Zt[b]))[:,1]; t+=m.predict_proba(sc.transform(Zs))[:,1]/5
print('LR OOF AUC %.5f'%roc_auc_score(y,o)); 
lg=lambda p:np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6)))
gb_o=np.mean([lg(v) for v in oof],0); gb_t=np.mean([lg(v) for v in tst],0)
print('GBM avg OOF AUC %.5f'%roc_auc_score(y,gb_o))
best=(0,0)
for w in [0,.1,.2,.3,.4]:
    a=roc_auc_score(y,(1-w)*gb_o+w*lg(o)); print('w_lr',w,'%.5f'%a)
    if a>best[0]: best=(a,w)
w=best[1]; fo=(1-w)*gb_o+w*lg(o); ft=(1-w)*gb_t+w*lg(t)
print('FINAL blend OOF AUC %.5f (w_lr=%.1f)'%(best[0],w))
# Calibrate monotonically-preserving: Platt on OOF logit so outputs are true probabilities
pl=LogisticRegression(C=1e6).fit(fo.reshape(-1,1),y); pt_=pl.predict_proba(ft.reshape(-1,1))[:,1]
sub=pd.DataFrame({'encounter_id':te.encounter_id,'readmit_30d':pt_.round(6)})
sub.to_csv(f'{WORK}/submission.csv',index=False); np.save(f'{WORK}/final_oof.npy',pl.predict_proba(fo.reshape(-1,1))[:,1])
print(sub.describe(), 'mean pred',pt_.mean(),'train rate',y.mean())