import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
tr=pd.read_csv('/mnt/user-data/uploads/train.csv'); te=pd.read_csv('/mnt/user-data/uploads/test.csv')
X=pd.read_parquet('out/final_dataset.parquet'); y=tr.merge(X[['encounter_id']],on='encounter_id').readmit_30d.values
lg=lambda p:np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6)))
g_o=np.mean([lg(np.load(f'out/g{i}_oof.npy')) for i in range(3)],0); g_t=np.mean([lg(np.load(f'out/g{i}_tst.npy')) for i in range(3)],0)
l_o=lg(np.load('out/lr_oof.npy')); l_t=lg(np.load('out/lr_tst.npy'))
print('GBM avg %.5f'%roc_auc_score(y,g_o)); best=(0,0)
for w in [0,.1,.2,.3,.4,.5]:
    a=roc_auc_score(y,(1-w)*g_o+w*l_o); print(w,'%.5f'%a); best=max(best,(a,w))
w=best[1]; fo=(1-w)*g_o+w*l_o; ft=(1-w)*g_t+w*l_t
pl=LogisticRegression(C=1e6).fit(fo.reshape(-1,1),y); p=pl.predict_proba(ft.reshape(-1,1))[:,1]
sub=pd.DataFrame({'encounter_id':te.encounter_id,'readmit_30d':p.round(6)})
ss=pd.read_csv('/mnt/user-data/uploads/sample_submission.csv')
assert sub.encounter_id.tolist()==ss.encounter_id.tolist() and sub.readmit_30d.between(0,1).all() and sub.readmit_30d.notna().all()
import os; os.makedirs('/mnt/user-data/outputs',exist_ok=True)
sub.to_csv('/mnt/user-data/outputs/submission_v11.csv',index=False)
print('FINAL OOF %.5f w_lr=%.1f'%(best[0],w), sub.readmit_30d.describe())
# dataset + column list
cols=[]
def src(c):
    for pre,s in [('lab_','lab_results'),('vit_','vitals'),('med','medications'),('pu_','prior_utilization'),('tr_','unit_transfers'),('n_proc','procedures'),('dx','diagnoses+icd10_lookup'),('n_dx','diagnoses'),('n_chronic','diagnoses+icd10_lookup'),('frac_chronic','diagnoses+icd10_lookup'),('primary_','diagnoses+icd10_lookup'),('followup','discharge_planning'),('home_health','discharge_planning'),('education','discharge_planning'),('dp_','discharge_planning'),
        ('bed_size','hospitals'),('teaching','hospitals'),('urban','hospitals'),('ownership','hospitals'),('sex','patients'),('race','patients'),('language','patients'),('marital','patients'),('insurance','patients'),('zip3','patients'),('deprivation','patients'),('age','patients+index_encounters'),('egfr','derived (labs, patients)'),('bun_cr','derived (labs)'),('shock','derived (vitals)'),('pulse','derived (vitals)'),('hb_','derived (labs)')]:
        if c.startswith(pre): return s
    return 'index_encounters (or derived)'
pd.DataFrame({'column':X.columns,'source_table':[src(c) for c in X.columns],'dtype':[str(X[c].dtype) for c in X.columns]}).to_csv('/mnt/user-data/outputs/final_columns.csv',index=False)
X.to_csv('/mnt/user-data/outputs/final_dataset.csv.gz',index=False,compression='gzip')
import shutil
for f in ['build_features.py','feat2.py','final.py','blend.py']: shutil.copy(f,'/mnt/user-data/outputs/'+f)
print(X.shape)
