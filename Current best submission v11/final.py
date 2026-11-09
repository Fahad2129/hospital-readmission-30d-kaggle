import pandas as pd, numpy as np, lightgbm as lgb, warnings, os; warnings.filterwarnings('ignore')
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from feat2 import add
os.makedirs('out',exist_ok=True)
X=add(pd.read_parquet('work/features.parquet')).drop(columns=['admit_year','days_since_start','admit_month'])
X.to_parquet('out/final_dataset.parquet'); 
tr=pd.read_csv('/mnt/user-data/uploads/train.csv'); te=pd.read_csv('/mnt/user-data/uploads/test.csv')
cat=[c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c]) and c!='encounter_id']
Xg=X.copy()
for c in cat+['hospital_id']: Xg[c]=Xg[c].astype('category')
T=Xg.merge(tr,on='encounter_id'); S=te.merge(Xg,on='encounter_id',how='left'); y=T.readmit_30d.values
feats=[c for c in T.columns if c not in('encounter_id','readmit_30d','attending_id')]
base=dict(objective='binary',verbose=-1,n_jobs=1,bagging_fraction=0.8,bagging_freq=1,cat_smooth=50,max_cat_to_onehot=8)
cfgs=[dict(num_leaves=7,min_child_samples=200,feature_fraction=0.3,lambda_l2=20,learning_rate=0.05),
      dict(num_leaves=4,max_depth=2,min_child_samples=300,feature_fraction=0.3,lambda_l2=30,learning_rate=0.07),
      dict(num_leaves=5,min_child_samples=300,feature_fraction=0.2,lambda_l2=50,learning_rate=0.07,max_bin=63)]
folds=list(StratifiedKFold(5,shuffle=True,random_state=0).split(T,y))
lg=lambda p:np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6)))
# logistic regression first (fast)
num=[c for c in feats if pd.api.types.is_numeric_dtype(T[c]) and T[c].notna().sum()>0 and T[c].nunique()>1]
cc=[c for c in feats if c in cat or c=='hospital_id']
miss=[c for c in num if pd.concat([T[c],S[c]]).isna().mean()>0.01]
lo,hi=T[num].quantile(.005),T[num].quantile(.995); med=T[num].clip(lo,hi,axis=1).median()
def prep(df):
    N=df[num].clip(lo,hi,axis=1).fillna(med); M=df[miss].isna().astype(int).add_suffix('_na')
    C=df[cc].astype(object).fillna('NA'); return pd.concat([N,M,C],axis=1)
Z=pd.get_dummies(pd.concat([prep(T),prep(S)],ignore_index=True),columns=cc,dtype=float)
Zt,Zs=Z.iloc[:len(T)].values,Z.iloc[len(T):].values
o=np.zeros(len(T)); t=np.zeros(len(S))
for a,b in folds:
    sc=StandardScaler().fit(Zt[a]); m=LogisticRegression(C=0.02,max_iter=300).fit(sc.transform(Zt[a]),y[a])
    o[b]=m.predict_proba(sc.transform(Zt[b]))[:,1]; t+=m.predict_proba(sc.transform(Zs))[:,1]/5
print('LR OOF %.5f'%roc_auc_score(y,o),flush=True); np.save('out/lr_oof.npy',o); np.save('out/lr_tst.npy',t)
for i,c in enumerate(cfgs):
    p={**base,**c}; o=np.zeros(len(T)); t=np.zeros(len(S))
    for f,(a,b) in enumerate(folds):
        p['seed']=100+f
        m=lgb.train(p,lgb.Dataset(T.loc[a,feats],y[a]),3000,valid_sets=[lgb.Dataset(T.loc[b,feats],y[b])],callbacks=[lgb.early_stopping(150,verbose=False)])
        o[b]=m.predict(T.loc[b,feats],num_iteration=m.best_iteration); t+=m.predict(S[feats],num_iteration=m.best_iteration)/5
    print('cfg',i,'OOF %.5f'%roc_auc_score(y,o),flush=True); np.save(f'out/g{i}_oof.npy',o); np.save(f'out/g{i}_tst.npy',t)
print('DONE')
