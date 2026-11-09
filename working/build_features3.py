from pathlib import Path
DATA=Path(__file__).resolve().parent
WORK=DATA
import pandas as pd, numpy as np
D=DATA; rd=lambda n,**k: pd.read_csv(f'{D}/{n}.csv',**k)
ie=rd('index_encounters',parse_dates=['admit_dt','discharge_dt'])[['encounter_id','admit_dt','discharge_dt']]
# ---- labs (same cleaning as features.py) ----
lb=rd('lab_results',parse_dates=['collected_dt']).drop_duplicates(); lb['lab']=lb.lab_name.str.lower()
v=lb.value.values.copy(); u=lb.unit.values; nm=lb.lab.values
def mult(mask,f):
    global v; v=np.where(mask,v*f,v)
m=lambda lab,unit:(nm==lab)&(u==unit)
mult(m('albumin','g/L'),0.1); mult((nm=='albumin')&pd.isna(u)&(v>10),0.1); mult(m('creatinine','umol/L'),1/88.4); mult((nm=='creatinine')&pd.isna(u)&(v>25),1/88.4)
mult(m('glucose','mmol/L'),18.016); mult((nm=='glucose')&pd.isna(u)&(v<=30),18.016); mult(m('hemoglobin','g/L'),0.1); mult((nm=='hemoglobin')&pd.isna(u)&(v>30),0.1)
rng={'albumin':(1,6),'bnp':(5,40000),'bun':(2,250),'creatinine':(0.3,20),'glucose':(30,1000),'hba1c':(3,20),'hemoglobin':(3,22),'inr':(0.7,15),'lactate':(0.3,30),'potassium':(1.5,10),'sodium':(100,175),'troponin_i':(0,60),'wbc':(0.5,100)}
lo=np.array([rng[x][0] for x in nm]); hi=np.array([rng[x][1] for x in nm])
for _ in range(2): v=np.where(v>hi,v/10,v)
v=np.where((v<lo)&(v*10<=hi),v*10,v); v=np.where((v>hi)|(v<lo),np.nan,v); lb['v']=v
lb=lb.dropna(subset=['v']).merge(ie,on='encounter_id')
lb['t']=(lb.collected_dt-lb.admit_dt).dt.total_seconds()/86400; lb['hdc']=(lb.discharge_dt-lb.collected_dt).dt.total_seconds()/3600; lb['los']=(lb.discharge_dt-lb.admit_dt).dt.total_seconds()/86400
lb['tv']=lb.t*lb.v; lb['tt']=lb.t**2; lb['one']=1.0
g=lb.groupby(['encounter_id','lab'])
s=g[['one','t','v','tv','tt']].sum(); den=s.one*s.tt-s.t**2
E=pd.DataFrame({'slope':np.where(den>1e-6,(s.one*s.tv-s.t*s.v)/den,np.nan)},index=s.index)
E['std']=g.v.std()
w1=lb[lb.t<=1].groupby(['encounter_id','lab']).v.agg(adm24_mean='mean',adm24_max='max')
w2=lb[lb.hdc<=24].groupby(['encounter_id','lab']).v.agg(dc24_mean='mean',dc24_max='max',dc24_min='min')
pk=lb.loc[g.v.idxmax().dropna().astype(int).values].set_index(['encounter_id','lab']); E['peak_h_to_dc']=pk.hdc/(pk.los*24)
E=E.join(w1).join(w2); E['dc_minus_adm']=E.dc24_mean-E.adm24_mean
E=E.unstack('lab'); E.columns=[f'lx_{b}_{a}' for a,b in E.columns]
# composites: z-scored severity at discharge and at admission
last=lb.sort_values(['encounter_id','t']).groupby(['encounter_id','lab']).v.last().unstack(); first=lb.sort_values(['encounter_id','t']).groupby(['encounter_id','lab']).v.first().unstack()
mx=lb.groupby(['encounter_id','lab']).v.max().unstack(); mn=lb.groupby(['encounter_id','lab']).v.min().unstack()
def z(df,col,sign=1): 
    x=df[col]; return sign*(np.log(x) if col in('bnp','bun','creatinine','troponin_i','lactate','wbc','inr') else x).pipe(lambda s:(s-s.mean())/s.std()).clip(-4,4)
sev={'creatinine':1,'bun':1,'bnp':1,'albumin':-1,'hemoglobin':-1,'wbc':1,'sodium':-1,'lactate':1,'troponin_i':1,'glucose':1,'inr':1}
C=pd.DataFrame(index=last.index)
C['sev_last']=pd.concat([z(last,k,sg) for k,sg in sev.items() if k in last],axis=1).mean(axis=1)
C['sev_first']=pd.concat([z(first,k,sg) for k,sg in sev.items() if k in first],axis=1).mean(axis=1)
C['sev_worst']=pd.concat([z(mx if sg==1 else mn,k,sg) for k,sg in sev.items() if k in mx],axis=1).mean(axis=1)
C['kidney_sev']=(z(last,'creatinine')+z(last,'bun')).fillna(0)
crit=((last.sodium<135)|(last.sodium>145)).astype(float)+(last.potassium>5.0)+(last.potassium<3.5)+(last.creatinine>2)+(last.hemoglobin<10)+(last.bun>40)+(last.glucose>250)+(last.albumin<3)+(last.wbc>12)+(last.bnp>1000)
C['n_critical_last']=crit
del lb
# ---- vitals ----
vt=rd('vitals').drop_duplicates(); vt['t']=pd.to_datetime(vt.charttime,format='%m/%d/%Y %H:%M')
x=vt.value.values.copy(); ty=vt.vital_type.values
x=np.where((ty=='TEMP')&(x>50),(x-32)*5/9,x); x=np.where((ty=='HR')&((x>220)|(x<25)),np.nan,x); x=np.where((ty=='SPO2')&(x<60),np.nan,x)
vt['x']=x; vt=vt.dropna(subset=['x']).merge(ie,on='encounter_id'); vt['td']=(vt.t-vt.admit_dt).dt.total_seconds()/86400; vt['hdc']=(vt.discharge_dt-vt.t).dt.total_seconds()/3600
vt['tx']=vt.td*vt.x; vt['tt']=vt.td**2; vt['one']=1.0
g=vt.groupby(['encounter_id','vital_type']); s=g[['one','td','x','tx','tt']].sum(); den=s.one*s.tt-s.td**2
V=pd.DataFrame({'slope':np.where(den>1e-6,(s.one*s.tx-s.td*s.x)/den,np.nan)},index=s.index)
V=V.join(vt[vt.hdc<=12].groupby(['encounter_id','vital_type']).x.agg(dc12_mean='mean',dc12_min='min',dc12_max='max'))
V=V.join(vt[vt.hdc<=24].groupby(['encounter_id','vital_type']).x.agg(dc24_min='min',dc24_max='max',dc24_std='std'))
V=V.unstack('vital_type'); V.columns=[f'vx_{b}_{a}' for a,b in V.columns]
d12=vt[vt.hdc<=12]
inst=pd.DataFrame({'unstable_12h':((d12.vital_type=='SBP')&(d12.x<100)).groupby(d12.encounter_id).sum()+((d12.vital_type=='HR')&(d12.x>100)).groupby(d12.encounter_id).sum()+((d12.vital_type=='SPO2')&(d12.x<92)).groupby(d12.encounter_id).sum()+((d12.vital_type=='RR')&(d12.x>22)).groupby(d12.encounter_id).sum()+((d12.vital_type=='TEMP')&(d12.x>38)).groupby(d12.encounter_id).sum()})
inst['readings_12h']=d12.groupby('encounter_id').size(); inst['unstable_12h_frac']=inst.unstable_12h/inst.readings_12h
OUT=pd.concat([E,C,V,inst],axis=1); print(OUT.shape)
OUT.reset_index().to_parquet(f'{WORK}/extras.parquet')
import pandas as pd
a = pd.read_parquet(f'{WORK}/features2.parquet'); b = pd.read_parquet(f'{WORK}/extras.parquet')
c3 = a.merge(b, on='encounter_id', how='left'); print(c3.shape); c3.to_parquet(f'{WORK}/features3.parquet')