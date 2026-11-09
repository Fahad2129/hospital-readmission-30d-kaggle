"""Build one row per encounter from the competition's raw CSV tables.

Run from the project folder: python make_readmission_features.py
Writes features2.parquet and final_columns.csv beside the raw CSV files.
"""
from pathlib import Path
import numpy as np
import pandas as pd

D = Path(__file__).resolve().parent
rd = lambda name, **kw: pd.read_csv(D / f"{name}.csv", **kw)

ie = rd("index_encounters", parse_dates=["admit_dt", "discharge_dt"])
pt = rd("patients")
hs = rd("hospitals")
F = ie.copy()
F["los_days"] = (F.discharge_dt - F.admit_dt).dt.total_seconds() / 86400
for name, col in [("admit_hour", "admit_dt"), ("admit_dow", "admit_dt"),
                  ("admit_month", "admit_dt"), ("admit_year", "admit_dt"),
                  ("dc_hour", "discharge_dt"), ("dc_dow", "discharge_dt")]:
    dt = F[col]
    attr = {"hour": dt.dt.hour, "dow": dt.dt.dayofweek,
            "month": dt.dt.month, "year": dt.dt.year}[name.split("_")[-1]]
    F[name] = attr
F["dc_weekend"] = (F.dc_dow >= 5).astype(int)
F["admit_t"] = (F.admit_dt - pd.Timestamp("2019-01-01")).dt.days
for c in ["admission_type", "admission_source", "discharge_disposition"]:
    F[c] = F[c].replace({"Not Available": np.nan, "Not Mapped": np.nan})
F["weight_missing"] = F.admit_weight_kg.isna().astype(int)
F["spec_missing"] = F.attending_specialty.isna().astype(int)
F["attending_n"] = F.groupby("attending_id").encounter_id.transform("size")
F["hospital_n"] = F.groupby("hospital_id").encounter_id.transform("size")
pt.loc[pt.birth_year <= 1900, "birth_year"] = np.nan
pt["race_ethnicity"] = pt.race_ethnicity.replace({"?": "Unknown"})
pt["insurance_type"] = pt.insurance_type.str.title()
pt["sex"] = pt.sex.replace({"U": np.nan})
F = F.merge(pt, on="patient_id", how="left").merge(hs, on="hospital_id", how="left")
F["age"] = F.admit_year - F.birth_year
F["zip3_num"] = pd.to_numeric(F.zip3, errors="coerce")
F["bmi_proxy_missing"] = F.admit_weight_kg.isna().astype(int)
F["ins_missing"] = F.insurance_type.isna().astype(int)
F = F.drop(columns=["zip3", "birth_year"])
adm = F[["encounter_id", "patient_id", "admit_dt", "discharge_dt"]]

# Diagnoses: restore missing ICD decimal and aggregate encounter level burdens.
dx = rd("diagnoses").drop_duplicates()
lk = rd("icd10_lookup")
def norm(code):
    code = str(code).strip().upper().replace(" ", "")
    return code if "." in code or len(code) <= 3 else code[:3] + "." + code[3:]
dx["code"] = dx.icd10_code.map(norm)
dx = dx.merge(lk, left_on="code", right_on="icd10_code", how="left", suffixes=("", "_lk"))
g = dx.groupby("encounter_id")
D1 = pd.DataFrame({"n_dx": g.size(), "n_dx_unique": g.code.nunique(),
                   "n_chronic": g.chronic_flag.sum(),
                   "n_poa_N": g.present_on_admission.apply(lambda s: (s == "N").sum()),
                   "n_poa_missing": g.present_on_admission.apply(lambda s: s.isna().sum()),
                   "n_poa_UW": g.present_on_admission.apply(lambda s: s.isin(["U", "W"]).sum())})
prim = dx.sort_values(["encounter_id", "is_primary", "seq"], ascending=[True, False, True]).drop_duplicates("encounter_id").set_index("encounter_id")[["code", "chapter", "chronic_flag"]]
prim.columns = ["prim_code", "prim_chapter", "prim_chronic"]
D1 = D1.join(prim)
D1 = D1.join(dx.groupby(["encounter_id", "chapter"]).size().unstack(fill_value=0).add_prefix("ch_"))
freq = dx.groupby("code").encounter_id.nunique()
keep = freq[freq >= 300].index
mh = dx[dx.code.isin(keep)].assign(v=1).pivot_table(index="encounter_id", columns="code", values="v", aggfunc="max", fill_value=0)
mh.columns = ["dx_" + c.replace(".", "_") for c in mh.columns]
D1 = D1.join(mh)
fam = {"hf": lambda c: c.startswith("I50") or c == "I11.0", "ami": lambda c: c.startswith("I21"), "afib": lambda c: c.startswith("I48"),
       "stroke": lambda c: c in ("I63.9", "G45.9", "Z86.73"), "copd": lambda c: c.startswith("J44"), "pna": lambda c: c in ("J18.9", "J15.9", "J13"),
       "dm": lambda c: c.startswith("E11") or c == "Z79.4", "ckd": lambda c: c.startswith("N18") or c in ("E11.22", "N17.9"),
       "cancer": lambda c: c.startswith("C"), "mets": lambda c: c in ("C79.9", "C80.1"), "dementia": lambda c: c == "F03.90",
       "psych": lambda c: c.startswith("F") and c != "F03.90", "sym": lambda c: c.startswith("R"), "fluid": lambda c: c in ("E86.0", "E87.1", "E87.6"),
       "noncompl": lambda c: c == "Z91.19", "pressure_ulcer": lambda c: c == "L89.90", "tobacco": lambda c: c in ("Z72.0", "F17.210")}
cm = dx[["encounter_id", "code"]].copy()
for k, fn in fam.items(): cm["f_" + k] = cm.code.map(fn).astype(int)
FM = cm.drop(columns="code").groupby("encounter_id").sum()
w = {"f_hf":1,"f_ami":1,"f_dementia":1,"f_copd":1,"f_dm":1,"f_ckd":2,"f_cancer":2,"f_mets":4,"f_stroke":1}
FM["charlson_like"] = sum(FM[k].clip(0, 1) * v for k, v in w.items())
D1 = D1.join(FM)

# Labs: normalize common units, repair obvious decimal slips, summarize trajectory.
lb = rd("lab_results", parse_dates=["collected_dt"]).drop_duplicates()
lb["lab"] = lb.lab_name.str.lower(); v = lb.value.values.copy(); u = lb.unit.values; nm = lb.lab.values
def mult(mask, factor):
    global v
    v = np.where(mask, v * factor, v)
m = lambda lab, unit: (nm == lab) & (u == unit)
mult(m("albumin", "g/L"), .1); mult((nm == "albumin") & pd.isna(u) & (v > 10), .1)
mult(m("creatinine", "umol/L"), 1/88.4); mult((nm == "creatinine") & pd.isna(u) & (v > 25), 1/88.4)
mult(m("glucose", "mmol/L"), 18.016); mult((nm == "glucose") & pd.isna(u) & (v <= 30), 18.016)
mult(m("hemoglobin", "g/L"), .1); mult((nm == "hemoglobin") & pd.isna(u) & (v > 30), .1)
rng = {"albumin":(1,6),"bnp":(5,40000),"bun":(2,250),"creatinine":(.3,20),"glucose":(30,1000),"hba1c":(3,20),"hemoglobin":(3,22),"inr":(.7,15),"lactate":(.3,30),"potassium":(1.5,10),"sodium":(100,175),"troponin_i":(0,60),"wbc":(.5,100)}
lo = np.array([rng[x][0] for x in nm]); hi = np.array([rng[x][1] for x in nm])
for _ in range(2): v = np.where(v > hi, v / 10, v)
v = np.where((v < lo) & (v*10 <= hi), v*10, v); v = np.where((v > hi) | (v < lo), np.nan, v)
lb["v"] = v
lb = lb.dropna(subset=["v"]).merge(adm, on="encounter_id").sort_values(["encounter_id", "collected_dt"])
lb["h_to_dc"] = (lb.discharge_dt-lb.collected_dt).dt.total_seconds()/3600
lb["H"] = (lb.abnormal_flag == "H").astype(int); lb["L"] = (lb.abnormal_flag == "L").astype(int)
g = lb.groupby(["encounter_id", "lab"])
A = g.v.agg(first="first", last="last", min="min", max="max", mean="mean", n="size")
A["delta"] = A["last"]-A["first"]; A["h_last"] = g.h_to_dc.last(); A["hfrac"] = g.H.mean(); A["lfrac"] = g.L.mean()
A = A.unstack("lab"); A.columns = [f"lab_{lab}_{stat}" for stat, lab in A.columns]
A["lab_total_n"] = lb.groupby("encounter_id").size(); A["lab_n_types"] = lb.groupby("encounter_id").lab.nunique()
A["lab_flag_H_n"] = lb.groupby("encounter_id").H.sum(); A["lab_flag_L_n"] = lb.groupby("encounter_id").L.sum()

# Vitals: convert temperature and filter impossible heart-rate / oxygen readings.
vt = rd("vitals").drop_duplicates(); vt["t"] = pd.to_datetime(vt.charttime, format="%m/%d/%Y %H:%M")
x = vt.value.values.copy(); ty = vt.vital_type.values
x = np.where((ty == "TEMP") & (x > 50), (x-32)*5/9, x)
x = np.where((ty == "HR") & ((x > 220) | (x < 25)), np.nan, x); x = np.where((ty == "SPO2") & (x < 60), np.nan, x)
vt["x"] = x; vt = vt.dropna(subset=["x"]).merge(adm, on="encounter_id").sort_values(["encounter_id", "t"])
vt["h_to_dc"] = (vt.discharge_dt-vt.t).dt.total_seconds()/3600
g = vt.groupby(["encounter_id", "vital_type"])
V = g.x.agg(first="first", last="last", min="min", max="max", mean="mean", std="std"); V["delta"] = V["last"]-V["first"]
V = V.join(vt[vt.h_to_dc <= 24].groupby(["encounter_id", "vital_type"]).x.mean().rename("last24"))
V = V.join(vt[vt.h_to_dc <= 48].groupby(["encounter_id", "vital_type"]).x.mean().rename("last48")).unstack("vital_type")
V.columns = [f"vit_{typ}_{stat}" for stat, typ in V.columns]
for k, op, t in [("HR", ">",100),("SBP","<",100),("SPO2","<",92),("RR",">",22),("TEMP",">",38)]:
    s = vt[vt.vital_type == k]; b = s.x > t if op == ">" else s.x < t
    V[f"vit_{k}_abn_frac"] = b.groupby(s.encounter_id).mean()
    s2=s[s.h_to_dc <= 24]; b2=s2.x > t if op == ">" else s2.x < t
    V[f"vit_{k}_abn_last24"] = b2.groupby(s2.encounter_id).mean()
V["vit_shock_idx_last"] = V["vit_HR_last"] / V["vit_SBP_last"]
V["vit_pulse_pressure_last"] = V["vit_SBP_last"] - V["vit_DBP_last"]
V["vit_n"] = vt.groupby("encounter_id").size()

# Medications: normalize text and summarize drug classes and common names.
md=rd("medications"); md["drug_name"]=md.drug_name.str.strip().str.lower(); md=md.drop_duplicates()
md["s"]=pd.to_datetime(md.start_dt.astype(str),format="%Y%m%d",errors="coerce")
md["e"]=pd.to_datetime(md.stop_dt.astype("Int64").astype(str),format="%Y%m%d",errors="coerce")
md=md.merge(adm,on="encounter_id"); md["start_rel"]=(md.s-md.admit_dt.dt.normalize()).dt.days; md["dur"]=(md.e-md.s).dt.days
md["home"]=(md.start_rel<0).astype(int); md["stop_missing"]=md.e.isna().astype(int); md["cont_missing"]=md.continued_at_discharge.isna().astype(int)
g=md.groupby("encounter_id")
M=pd.DataFrame({"n_med":g.size(),"n_med_unique":g.drug_name.nunique(),"n_med_classes":g.drug_class.nunique(),"med_cont_sum":g.continued_at_discharge.sum(),"med_cont_frac":g.continued_at_discharge.mean(),"med_cont_missing":g.cont_missing.sum(),"med_stop_missing":g.stop_missing.sum(),"med_home_n":g.home.sum(),"med_dur_mean":g.dur.mean(),"med_start_rel_mean":g.start_rel.mean(),"med_n_IV":g.route.apply(lambda s:(s=="IV").sum()),"med_n_SC":g.route.apply(lambda s:(s=="SC").sum())})
M["med_not_cont_n"]=M.n_med-M.med_cont_sum-M.med_cont_missing
M=M.join(md.groupby(["encounter_id","drug_class"]).size().unstack(fill_value=0).add_prefix("mc_"))
M=M.join(md[md.continued_at_discharge==1].groupby(["encounter_id","drug_class"]).size().unstack(fill_value=0).add_prefix("mcont_"))
dn=md.drug_name.value_counts(); dm=md[md.drug_name.isin(dn[dn>=1500].index)].assign(v=1).pivot_table(index="encounter_id",columns="drug_name",values="v",aggfunc="max",fill_value=0)
dm.columns=["drug_"+c for c in dm.columns]; M=M.join(dm)

# Prior utilization strictly precedes the index admission.
pu=rd("prior_utilization",parse_dates=["visit_dt"]).drop_duplicates()
a2=adm.merge(F[["encounter_id","hospital_id"]],on="encounter_id")
pu=pu.merge(a2[["encounter_id","patient_id","admit_dt","hospital_id"]],on="patient_id",suffixes=("","_idx"))
pu["dbef"]=(pu.admit_dt-pu.visit_dt).dt.days; pu["same_h"]=(pu.hospital_id==pu.hospital_id_idx).astype(int)
U=pd.DataFrame(index=a2.encounter_id.unique())
for typ in ["ED","IP","OP"]:
    s=pu[pu.visit_type==typ]
    for window in [30,90,180,365,730]: U[f"pu_{typ}_{window}"]=s[s.dbef<=window].groupby("encounter_id").size()
    U[f"pu_{typ}_last_days"]=s.groupby("encounter_id").dbef.min(); U[f"pu_{typ}_samehosp"]=s.groupby("encounter_id").same_h.sum()
cols=[c for c in U if c.startswith("pu_") and not c.endswith("_days")]; U[cols]=U[cols].fillna(0)
U["pu_total"]=pu.groupby("encounter_id").size().fillna(0); U["pu_n_hosp"]=pu.groupby("encounter_id").hospital_id.nunique()
U["pu_any_days"]=pu.groupby("encounter_id").dbef.min(); U["pu_acute_365"]=U.pu_ED_365+U.pu_IP_365; U["pu_acute_180"]=U.pu_ED_180+U.pu_IP_180; U["pu_has_history"]=(U.pu_total>0).astype(int)

# Reproduce the original notebook's procedure and transfer features. They are
# retained for the auditable table; the documented LightGBM recipe drops them.
pr=rd("procedures").drop_duplicates(); pr["d"]=pd.to_datetime(pr.proc_dt,format="%d-%b-%Y")
pr=pr.merge(adm,on="encounter_id"); pr["proc_day"]=(pr.d-pr.admit_dt.dt.normalize()).dt.days; pr["code"]=pr.code.astype(str)
g=pr.groupby("encounter_id")
P=pd.DataFrame({"n_proc":g.size(),"proc_first_day":g.proc_day.min(),"proc_last_day":g.proc_day.max()})
P["proc_code"]=pr.sort_values("proc_day").drop_duplicates("encounter_id").set_index("encounter_id").code
topc=pr.code.value_counts(); topc=topc[topc>=150].index
pm=pr[pr.code.isin(topc)].assign(v=1).pivot_table(index="encounter_id",columns="code",values="v",aggfunc="max",fill_value=0)
pm.columns=["proc_"+c for c in pm.columns]; P=P.join(pm)
ut=rd("unit_transfers",parse_dates=["in_dt","out_dt"]).drop_duplicates().merge(adm,on="encounter_id")
ut["hrs"]=(ut.out_dt-ut.in_dt).dt.total_seconds()/3600; ut=ut.sort_values(["encounter_id","in_dt"])
T=ut.groupby(["encounter_id","unit"]).hrs.sum().unstack(fill_value=0).add_prefix("hrs_")
T=T.join(ut.groupby(["encounter_id","unit"]).size().unstack(fill_value=0).add_prefix("visits_"))
g=ut.groupby("encounter_id"); T["n_transfers"]=g.size(); T["first_unit"]=g.unit.first(); T["last_unit"]=g.unit.last()
icu=ut[ut.unit=="ICU"].copy(); icu["rel"]=(icu.in_dt-icu.admit_dt).dt.total_seconds()/3600; icu["gap_to_dc"]=(icu.discharge_dt-icu.out_dt).dt.total_seconds()/3600
T["icu_first_rel_h"]=icu.groupby("encounter_id").rel.min(); T["icu_to_dc_h"]=icu.groupby("encounter_id").gap_to_dc.min()
T["icu_any"]=(T.visits_ICU>0).astype(int) if "visits_ICU" in T else 0; T["had_ed"]=(T.visits_ED>0).astype(int); T["ward_after_icu_h"]=T.icu_to_dc_h

# Discharge planning.
dp=rd("discharge_planning").drop_duplicates("encounter_id").set_index("encounter_id")
dp["dp_present"]=1; dp["dp_followup_missing"]=dp.followup_scheduled_days.isna().astype(int)
dp["home_health_referral"]=dp.home_health_referral.map({"Y":1,"N":0}); dp["education_done"]=dp.education_done.map({"Yes":1,"No":0})
dp=dp.add_prefix("dp_").rename(columns={"dp_dp_present":"dp_present","dp_dp_followup_missing":"dp_followup_missing"})

X=F.set_index("encounter_id")
for table in [D1,P,A,V,M,T,U,dp]: X=X.join(table,how="left")
X["n_proc"]=X.n_proc.fillna(0)
for c in [c for c in X if c.startswith(("mc_","mcont_","drug_","dx_","ch_"))]: X[c]=X[c].fillna(0)
X["no_meds"]=X.n_med.isna().astype(int); X["n_med"]=X.n_med.fillna(0); X["no_labs"]=X.lab_total_n.isna().astype(int); X["dp_present"]=X.dp_present.fillna(0)
X["dx_per_day"]=X.n_dx/X.los_days; X["meds_per_day"]=X.n_med/X.los_days; X["labs_per_day"]=X.lab_total_n/X.los_days
female=X.sex=="F"
def egfr(cr):
    k=np.where(female,.7,.9); a=np.where(female,-.241,-.302)
    return 142*np.minimum(cr/k,1)**a*np.maximum(cr/k,1)**-1.2*.9938**X.age*np.where(female,1.012,1)
X["egfr_first"]=egfr(X.lab_creatinine_first); X["egfr_last"]=egfr(X.lab_creatinine_last); X["egfr_max_cr"]=egfr(X.lab_creatinine_max)
X["bun_cr_ratio"]=X.lab_bun_last/X.lab_creatinine_last; X["cr_ratio_last_first"]=X.lab_creatinine_last/X.lab_creatinine_first
X["acute_adm"]=X.admission_type.isin(["Emergency","Urgent","Trauma"]).astype(int)
los_pts=pd.cut(X.los_days,[-1,1,2,3,4,7,14,1e9],labels=[0,1,2,3,4,5,7]).astype(float)
cl=X.charlson_like.fillna(0); X["lace"]=los_pts+3*X.acute_adm+np.where(cl>=4,5,cl)+np.minimum(X.pu_ED_180.fillna(0),4)
X=X.drop(columns=["patient_id","admit_dt","discharge_dt","attending_id"],errors="ignore").reset_index()
assert X.encounter_id.is_unique and len(X)==len(ie), "Expected one feature row per encounter"
X.to_parquet(D/"features2.parquet",index=False)

# Explicit provenance for every feature.
def source(c):
    if c=="encounter_id": return "index_encounters", "join key"
    if c.startswith("lab_") or c in {"no_labs"}: return "lab_results", "cleaned and aggregated by encounter and lab"
    if c.startswith("vit_"): return "vitals", "temperature normalized; encounter-level summary"
    if c.startswith(("dx_","ch_","prim_","n_dx","n_chronic","n_poa","f_","charlson")): return "diagnoses + icd10_lookup", "ICD normalized and aggregated by encounter"
    if c.startswith(("mc_","mcont_","drug_","med_","n_med")) or c=="no_meds": return "medications", "normalized drug text; encounter-level aggregation"
    if c.startswith("pu_"): return "prior_utilization", "prior-to-admission visit counts and summaries"
    if c.startswith("dp_"): return "discharge_planning", "cleaned encounter-level fields"
    if c in hs.columns: return "hospitals", "joined by hospital_id"
    if c in pt.columns or c in {"age","zip3_num","bmi_proxy_missing","ins_missing"}: return "patients", "cleaned patient fields joined by patient_id"
    return "index_encounters", "encounter field or derived date/admission feature"
pd.DataFrame([{"column":c,"source_table":source(c)[0],"creation":source(c)[1]} for c in X.columns]).to_csv(D/"final_columns.csv",index=False)
print(f"Wrote {D/'features2.parquet'}: {X.shape}; {D/'final_columns.csv'}")
