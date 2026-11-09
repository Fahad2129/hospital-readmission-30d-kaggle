import pandas as pd, numpy as np
def add(X):
    X=X.copy()
    age=X.age; fem=(X.sex=='F').astype(float)
    cr=X.lab_creatinine_last
    k=np.where(fem==1,0.7,0.9); a=np.where(fem==1,-0.241,-0.302)
    for nm in ['last','mean','max','first']:
        cr=X[f'lab_creatinine_{nm}']
        X[f'egfr_{nm}']=142*np.minimum(cr/k,1)**a*np.maximum(cr/k,1)**-1.2*0.9938**age*np.where(fem==1,1.012,1)
    X['bun_cr_last']=X.lab_bun_last/X.lab_creatinine_last
    X['bun_cr_mean']=X.lab_bun_mean/X.lab_creatinine_mean
    X['shock_idx_last']=X.vit_HR_last/X.vit_SBP_last; X['shock_idx_mean']=X.vit_HR_mean/X.vit_SBP_mean
    X['pulse_pr_mean']=X.vit_SBP_mean-X.vit_DBP_mean
    X['hb_last_minus_first']=X.lab_hemoglobin_last-X.lab_hemoglobin_first
    X['lab_per_day']=X.lab_n_total/X.los_days; X['vit_per_day']=X.vit_n_total/X.los_days
    X['pu_ip_ed_365']=X.pu_IP_365d*2+X.pu_ED_365d
    X['pu_rate_per_yr']=X.pu_n/ ((X.pu_first_days.fillna(0)+30)/365)
    X['dx_per_day']=X.n_dx/X.los_days
    X['age_x_chronic']=X.age*X.n_chronic_dx
    X['abn_x_dx']=X.lab_abn_total/X.lab_n_total.clip(lower=1)
    X['days_since_start']=(X.admit_year-2019)*365+X.admit_month*30.4
    X['fu_missing']=X.followup_scheduled_days.isna().astype(int)
    X['dp_gaps']=X.med_reconciliation_done.isna().astype(int)+X.education_done.isna().astype(int)+X.followup_scheduled_days.isna().astype(int)
    X['dis_to_snf']=(X.discharge_disposition=='Skilled Nursing Facility').astype(int)
    return X
def te_oof(T,S,cols,y,folds,sm=30):
    """out-of-fold smoothed target encoding; returns train and test frames"""
    prior=y.mean(); Tn={}; Sn={}
    for c in cols:
        o=np.full(len(T),np.nan)
        for a,b in folds:
            st=pd.DataFrame({'k':T[c].iloc[a].astype(object).values,'y':y[a]}).groupby('k').y.agg(['sum','count'])
            enc=(st['sum']+prior*sm)/(st['count']+sm)
            o[b]=T[c].iloc[b].astype(object).map(enc).fillna(prior).values
        st=pd.DataFrame({'k':T[c].astype(object).values,'y':y}).groupby('k').y.agg(['sum','count'])
        enc=(st['sum']+prior*sm)/(st['count']+sm)
        Tn['te_'+c]=o; Sn['te_'+c]=S[c].astype(object).map(enc).fillna(prior).values
    return pd.DataFrame(Tn,index=T.index),pd.DataFrame(Sn,index=S.index)
