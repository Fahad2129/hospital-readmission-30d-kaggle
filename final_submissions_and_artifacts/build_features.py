"""Raw tables -> one row per encounter_id (train + test). Saves features.parquet"""
from pathlib import Path
import pandas as pd, numpy as np, warnings, re
warnings.filterwarnings('ignore')
ROOT = Path(__file__).resolve().parents[1]
D = ROOT / 'working'
OUT = ROOT / 'reproducibility_runs' / 'legacy_features'
import os; os.makedirs(OUT, exist_ok=True)

tr = pd.read_csv(f'{D}/train.csv'); te = pd.read_csv(f'{D}/test.csv')
ids = pd.concat([tr[['encounter_id']], te[['encounter_id']]], ignore_index=True)
ie = pd.read_csv(f'{D}/index_encounters.csv', parse_dates=['admit_dt', 'discharge_dt'])
ie = ie.drop_duplicates('encounter_id')
pt = pd.read_csv(f'{D}/patients.csv').drop_duplicates('patient_id')
hosp = pd.read_csv(f'{D}/hospitals.csv')
X = ids.merge(ie, on='encounter_id', how='left').merge(pt, on='patient_id', how='left').merge(hosp, on='hospital_id', how='left')

# ---------- cleaning of encounter / patient / hospital fields ----------
for c in ['admission_type', 'admission_source', 'discharge_disposition']:
    X[c] = X[c].replace({'Not Available': np.nan, 'Not Mapped': np.nan}).fillna('Unknown')
X['insurance_type'] = X['insurance_type'].str.title().replace({'Self-Pay': 'Self-pay'})
X['race_ethnicity'] = X['race_ethnicity'].replace({'?': np.nan})
X['sex'] = X['sex'].replace({'U': np.nan})
X['los_days'] = (X.discharge_dt - X.admit_dt).dt.total_seconds() / 86400
X['admit_hour'] = X.admit_dt.dt.hour; X['admit_dow'] = X.admit_dt.dt.dayofweek
X['admit_month'] = X.admit_dt.dt.month; X['admit_year'] = X.admit_dt.dt.year
X['disch_hour'] = X.discharge_dt.dt.hour; X['disch_dow'] = X.discharge_dt.dt.dayofweek
X['age'] = X.admit_year - X.birth_year
X.loc[(X.age < 0) | (X.age > 110), 'age'] = np.nan
X['weight_missing'] = X.admit_weight_kg.isna().astype(int)
X['bmi_proxy'] = X.admit_weight_kg  # no height available
X['spec_missing'] = X.attending_specialty.isna().astype(int)
X['zip3'] = X['zip3'].astype('float')
# frequency encodings (computed on all encounter ids; no target involved)
for c in ['hospital_id', 'attending_id', 'zip3']:
    X[c + '_cnt'] = X.groupby(c)[c].transform('size')
base = X[['encounter_id', 'patient_id', 'admit_dt', 'discharge_dt']].copy()
X = X.drop(columns=['patient_id', 'admit_dt', 'discharge_dt', 'birth_year', 'bmi_proxy'])
tmp = base.set_index('encounter_id')
adm = tmp.admit_dt; dis = tmp.discharge_dt

# ---------- diagnoses ----------
d = pd.read_csv(f'{D}/diagnoses.csv')
d['code'] = d.icd10_code.astype(str).str.upper().str.replace('.', '', regex=False).str.strip()
d = d.drop_duplicates(['encounter_id', 'code'])
lk = pd.read_csv(f'{D}/icd10_lookup.csv'); lk['code'] = lk.icd10_code.str.replace('.', '', regex=False)
lk = lk.drop_duplicates('code')
d = d.merge(lk[['code', 'chapter', 'chronic_flag']], on='code', how='left')
g = d.groupby('encounter_id')
dx = pd.DataFrame({
    'n_dx': g.size(), 'n_chronic_dx': g.chronic_flag.sum(),
    'n_dx_poa_n': g.present_on_admission.apply(lambda s: (s == 'N').sum()),
    'n_dx_chapters': g.chapter.nunique(),
})
dx['frac_chronic'] = dx.n_chronic_dx / dx.n_dx
prim = d.sort_values(['encounter_id', 'is_primary', 'seq'], ascending=[True, False, True]).drop_duplicates('encounter_id')
dx['primary_dx'] = prim.set_index('encounter_id').code
dx['primary_chapter'] = prim.set_index('encounter_id').chapter
# multi-hot of every code (82 canonical codes) + chapter counts
mh = pd.crosstab(d.encounter_id, d.code).clip(upper=1).add_prefix('dx_')
ch = pd.crosstab(d.encounter_id, d.chapter).add_prefix('dxch_')
dx = dx.join(mh).join(ch)
X = X.merge(dx.reset_index(), on='encounter_id', how='left')

# ---------- labs ----------
lab = pd.read_csv(f'{D}/lab_results.csv', parse_dates=['collected_dt']).drop_duplicates()
conv = {('Albumin', 'g/L'): 0.1, ('Creatinine', 'umol/L'): 1 / 88.4, ('Glucose', 'mmol/L'): 18.016, ('Hemoglobin', 'g/L'): 0.1}
f = np.array([conv.get((n, u), 1.0) for n, u in zip(lab.lab_name, lab.unit)]) if False else None
lab['k'] = [conv.get(k, 1.0) for k in zip(lab.lab_name, lab.unit)]
lab['v'] = lab.value * lab.k
rng = {'Albumin': (1, 6), 'Bnp': (5, 35000), 'Bun': (2, 250), 'Creatinine': (0.3, 20), 'Glucose': (30, 1000),
       'Hba1C': (3, 20), 'Hemoglobin': (3, 22), 'Inr': (0.8, 15), 'Lactate': (0.3, 30), 'Potassium': (1.5, 9),
       'Sodium': (100, 180), 'Troponin_I': (0, 50), 'Wbc': (0.5, 100)}
lo = lab.lab_name.map({k: v[0] for k, v in rng.items()}); hi = lab.lab_name.map({k: v[1] for k, v in rng.items()})
dec = (lab.v > hi) & (lab.v / 10 >= lo) & (lab.v / 10 <= hi)   # x10 decimal-shift errors
lab.loc[dec, 'v'] = lab.loc[dec, 'v'] / 10
lab.loc[(lab.v > hi) | (lab.v < lo), 'v'] = np.nan
lab = lab.dropna(subset=['v']).sort_values(['encounter_id', 'collected_dt'])
lab['name'] = lab.lab_name.str.lower()
lab['h_from_adm'] = (lab.collected_dt - lab.encounter_id.map(adm)).dt.total_seconds() / 3600
lab['h_to_dis'] = (lab.encounter_id.map(dis) - lab.collected_dt).dt.total_seconds() / 3600
lab['abn'] = lab.abnormal_flag.isin(['H', 'L']).astype(int)
gl = lab.groupby(['encounter_id', 'name'])
agg = gl.v.agg(['last', 'first', 'mean', 'min', 'max', 'std', 'count'])
agg['delta'] = agg['last'] - agg['first']
agg['h_last_to_dis'] = gl.h_to_dis.last()
agg['abn_frac'] = gl.abn.mean()
agg = agg.unstack('name'); agg.columns = [f'lab_{b}_{a}' for a, b in agg.columns]
agg['lab_n_total'] = lab.groupby('encounter_id').size()
agg['lab_n_types'] = lab.groupby('encounter_id').name.nunique()
agg['lab_abn_total'] = lab.groupby('encounter_id').abn.sum()
agg['lab_abn_last_frac'] = lab.groupby('encounter_id').abn.mean()
X = X.merge(agg.reset_index(), on='encounter_id', how='left')
del lab

# ---------- vitals ----------
v = pd.read_csv(f'{D}/vitals.csv').drop_duplicates()
v['t'] = pd.to_datetime(v.charttime, format='%m/%d/%Y %H:%M', errors='coerce')
m = (v.vital_type == 'TEMP') & (v.value > 45); v.loc[m, 'value'] = (v.loc[m, 'value'] - 32) * 5 / 9   # F -> C
bounds = {'HR': (20, 250), 'SBP': (50, 260), 'DBP': (20, 160), 'RR': (4, 60), 'SPO2': (50, 100), 'TEMP': (33, 42)}
lo = v.vital_type.map({k: b[0] for k, b in bounds.items()}); hi = v.vital_type.map({k: b[1] for k, b in bounds.items()})
v.loc[(v.value < lo) | (v.value > hi), 'value'] = np.nan
v = v.dropna(subset=['value', 't']).sort_values(['encounter_id', 't'])
v['h_to_dis'] = (v.encounter_id.map(dis) - v.t).dt.total_seconds() / 3600
gv = v.groupby(['encounter_id', 'vital_type'])
va = gv.value.agg(['last', 'first', 'mean', 'min', 'max', 'std', 'count'])
va['delta'] = va['last'] - va['first']
last24 = v[v.h_to_dis <= 24].groupby(['encounter_id', 'vital_type']).value.mean().rename('last24')
va = va.join(last24)
va = va.unstack('vital_type'); va.columns = [f'vit_{b}_{a}' for a, b in va.columns]
va['vit_n_total'] = v.groupby('encounter_id').size()
X = X.merge(va.reset_index(), on='encounter_id', how='left')
del v

# ---------- medications ----------
md = pd.read_csv(f'{D}/medications.csv').drop_duplicates()
md['drug'] = md.drug_name.str.lower().str.strip()
md['sd'] = pd.to_datetime(md.start_dt.astype(str), format='%Y%m%d', errors='coerce')
md['ed'] = pd.to_datetime(md.stop_dt.astype('Int64').astype(str), format='%Y%m%d', errors='coerce')
md['dur'] = (md.ed - md.sd).dt.days
md.loc[md.dur < 0, 'dur'] = np.nan
md['cont'] = md.continued_at_discharge
gm = md.groupby('encounter_id')
ma = pd.DataFrame({'med_n': gm.size(), 'med_n_drugs': gm.drug.nunique(), 'med_n_classes': gm.drug_class.nunique(),
                   'med_cont_sum': gm.cont.sum(), 'med_cont_frac': gm.cont.mean(), 'med_dur_mean': gm.dur.mean(),
                   'med_no_stop_frac': gm.ed.apply(lambda s: s.isna().mean())})
for r in ['IV', 'SC', 'INH', 'PO']:
    ma[f'med_route_{r}'] = md[md.route == r].groupby('encounter_id').size()
cl = pd.crosstab(md.encounter_id, md.drug_class).clip(upper=1).add_prefix('medc_')
cc = md[md.cont == 1].groupby(['encounter_id', 'drug_class']).size().unstack().notna().astype(int).add_prefix('medcont_')
ma = ma.join(cl).join(cc)
for c in [c for c in ma.columns if c.startswith('med_route')]: ma[c] = ma[c].fillna(0)
X = X.merge(ma.reset_index(), on='encounter_id', how='left')
X['med_any'] = X.med_n.notna().astype(int)

# ---------- discharge planning ----------
dp = pd.read_csv(f'{D}/discharge_planning.csv').drop_duplicates('encounter_id')
dp['home_health_referral'] = dp.home_health_referral.map({'Y': 1, 'N': 0})
dp['education_done'] = dp.education_done.map({'Yes': 1, 'No': 0})
dp['dp_row'] = 1
X = X.merge(dp, on='encounter_id', how='left')
X['dp_row'] = X.dp_row.fillna(0)

# ---------- prior utilization (per patient, before index admission) ----------
pu = pd.read_csv(f'{D}/prior_utilization.csv', parse_dates=['visit_dt']).drop_duplicates()
pid = base.set_index('encounter_id').patient_id
b2 = base[['encounter_id', 'patient_id', 'admit_dt']]
pu = b2.merge(pu, on='patient_id', how='inner')
pu['days_before'] = (pu.admit_dt - pu.visit_dt).dt.days
pu = pu[pu.days_before >= 0]
pa = pu.groupby('encounter_id').agg(pu_n=('days_before', 'size'), pu_last_days=('days_before', 'min'),
                                     pu_first_days=('days_before', 'max'))
for t in ['ED', 'IP', 'OP']:
    s = pu[pu.visit_type == t]
    pa[f'pu_{t}_n'] = s.groupby('encounter_id').size()
    pa[f'pu_{t}_last_days'] = s.groupby('encounter_id').days_before.min()
    for w in [30, 90, 180, 365]:
        pa[f'pu_{t}_{w}d'] = s[s.days_before <= w].groupby('encounter_id').size()
for w in [30, 90, 180, 365]:
    pa[f'pu_all_{w}d'] = pu[pu.days_before <= w].groupby('encounter_id').size()
pa['pu_n_hosp'] = pu.groupby('encounter_id').hospital_id.nunique()
sameh = pu.merge(X[['encounter_id', 'hospital_id']], on='encounter_id', suffixes=('', '_idx'))
pa['pu_same_hosp_n'] = sameh[sameh.hospital_id == sameh.hospital_id_idx].groupby('encounter_id').size()
X = X.merge(pa.reset_index(), on='encounter_id', how='left')
puc = [c for c in X.columns if c.startswith('pu_') and c not in ('pu_last_days', 'pu_first_days') and 'last_days' not in c]
X[puc] = X[puc].fillna(0)

# ---------- unit transfers (small summary; earlier ablation: weak) ----------
ut = pd.read_csv(f'{D}/unit_transfers.csv', parse_dates=['in_dt', 'out_dt']).drop_duplicates()
ut['hrs'] = (ut.out_dt - ut.in_dt).dt.total_seconds() / 3600
ut.loc[ut.hrs < 0, 'hrs'] = np.nan
ua = ut.groupby('encounter_id').agg(tr_n=('unit', 'size'))
for u in ['ICU', 'STEPDOWN', 'WARD', 'ED']:
    ua[f'tr_hrs_{u}'] = ut[ut.unit == u].groupby('encounter_id').hrs.sum()
ua['tr_last_unit'] = ut.sort_values('in_dt').groupby('encounter_id').unit.last()
X = X.merge(ua.reset_index(), on='encounter_id', how='left')
for u in ['ICU', 'STEPDOWN', 'WARD', 'ED']: X[f'tr_hrs_{u}'] = X[f'tr_hrs_{u}'].fillna(0)

# ---------- procedures ----------
pr = pd.read_csv(f'{D}/procedures.csv').drop_duplicates()
pra = pr.groupby('encounter_id').size().rename('n_proc')
X = X.merge(pra.reset_index(), on='encounter_id', how='left'); X['n_proc'] = X.n_proc.fillna(0)

X.to_parquet(f'{OUT}/features.parquet')
print(X.shape)
