"""Evaluate a restrained meta blend of every saved OOF candidate."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT=Path(__file__).resolve().parents[1]; V=ROOT/"Current best submission v11"; D=ROOT/"working"
tr=pd.read_csv(D/"train.csv"); te=pd.read_csv(D/"test.csv"); ss=pd.read_csv(D/"sample_submission.csv")
# final_oof.npy follows the encounter order in features2.parquet, so align by ID.
ids=pd.read_parquet(D/"features2.parquet",columns=["encounter_id"])
old_ids=ids.merge(tr,on="encounter_id",how="inner").encounter_id
old=pd.DataFrame({"encounter_id":old_ids,"p":np.load(ROOT/"final_oof.npy")}).set_index("encounter_id").reindex(tr.encounter_id).p.to_numpy()
names=["old","cfg1","cfg2","cfg3","stump","gam"]
oofs=[old]+[np.load(V/f"{n}_oof.npy") for n in ["cfg1","cfg2","cfg3","stump","enriched_gam"]]
tests=[pd.read_csv(ROOT/"submission.csv").readmit_30d.to_numpy()]+[np.load(V/f"{n}_test.npy") for n in ["cfg1","cfg2","cfg3","stump","enriched_gam"]]
y=tr.readmit_30d.to_numpy()
for n,o in zip(names,oofs): print("single",n,round(roc_auc_score(y,o),6))
rank=lambda p:rankdata(p,method="average")/len(p)
R=np.column_stack([rank(p) for p in oofs]); Rt=np.column_stack([rank(p) for p in tests])
v14=.8*R[:,0]+.2*R[:,1]
print("v14 rank OOF",round(roc_auc_score(y,v14),6))
uniform=R.mean(1); print("uniform ranks",round(roc_auc_score(y,uniform),6))
# Greedy non-negative rank ensemble selection, with a small fixed iteration count.
sel=[]; total=np.zeros(len(y))
for i in range(30):
    scores=[roc_auc_score(y,(total+R[:,j])/(i+1)) for j in range(R.shape[1])]
    j=int(np.argmax(scores)); sel.append(j); total+=R[:,j]
    if i in [4,9,19,29]: print("greedy",i+1,round(max(scores),6),{names[k]:round(sel.count(k)/(i+1),2) for k in set(sel)})
# Ridge-stabilized logistic stack; compare several strengths on repeated held-out folds.
mu=R.mean(0); sd=R.std(0); Z=(R-mu)/sd; Zt=(Rt-mu)/sd
best=None; scores_by_c={}
for C in [.001,.003,.01,.03,.1]:
    sscores=[]; held=[]
    for seed in [0,7,13]:
        o=np.zeros(len(y))
        for a,b in StratifiedKFold(5,shuffle=True,random_state=seed).split(Z,y):
            m=LogisticRegression(C=C,max_iter=1000).fit(Z[a],y[a]); o[b]=m.decision_function(Z[b])
        score=roc_auc_score(y,o); sscores.append(score)
        if seed==7: held=o
    scores_by_c[C]=(float(np.mean(sscores)),held)
    print("stack",C,[round(s,6) for s in sscores],"mean",round(np.mean(sscores),6),flush=True)
    if best is None or np.mean(sscores)>best[0]: best=(float(np.mean(sscores)),C,held)
basecv=roc_auc_score(y,v14); print("v14",basecv,"best stack mean",best[0],"C",best[1],flush=True)
if best[0] > basecv:
    C=best[1]; m=LogisticRegression(C=C,max_iter=1000).fit(Z,y)
    z=m.decision_function(Zt)
    # monotonic sigmoid keeps the Kaggle field as probabilities
    sub=pd.DataFrame({"encounter_id":te.encounter_id,"readmit_30d":expit(z)})
    assert sub.encounter_id.tolist()==ss.encounter_id.tolist() and sub.readmit_30d.notna().all() and sub.readmit_30d.between(0,1).all()
    out=V/"submission_v18_oof_stack.csv"; sub.to_csv(out,index=False); print("wrote",out,flush=True)
else: print("No stack submission written: repeated CV did not beat v14.",flush=True)
