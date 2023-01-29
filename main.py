import streamlit as st
import pandas as pd
import xgboost as xgb
import numpy as np

st.header('STEMI Death Prediction by Some features (Beta-Version)')

#Load model
model = xgb.XGBClassifier(max_depth=5, n_estimators=50)
model.load_model('death_model.json')

st.subheader('Age')
left_column, right_column = st.columns(2)
with left_column:
    age = st.text_input('AGE', key='age')

st.subheader('Blood Pressure')
left_column, right_column = st.columns(2)
with left_column:
    sbp = st.text_input('SBP', key='sbp')
with right_column:
    dbp = st.text_input('DBP', key='dbp')

st.subheader('HR/RR/O2 Sat')
left_column, right_column = st.columns(2)
with left_column:
    rr = st.text_input('Respiratory Rate', key='rr')
    hr = st.text_input('Heart rate', key='hr')
with right_column:
    oxygen = st.text_input('Oxygen saturation', key='o2sat')


st.subheader('Chronic illness')
left_column, right_column = st.columns(2)
with left_column:
    dm = st.checkbox('Diabetes Mellitus')
    ht = st.checkbox('Hypertension')
    dlp = st.checkbox('Dyslipidemia')
    smoking = st.checkbox('Smoking')
    cva = st.checkbox('Cardiovascula disease')
    pad = st.checkbox('Peripheral arterial disease')
    copd = st.checkbox('Chronic Obstructive Lung Disease')
    ckd = st.checkbox('CKD stage >= 3')
with right_column:
    fam_stemi = st.checkbox('Familial Hx. of Myocardial infarction')
    pri_mi = st.checkbox('Prior Myocardial infarction')
    pri_hf = st.checkbox('Prior Heart failure')
    pri_pci = st.checkbox('Prior Percutaneous coronary intervention')
    pri_cabg = st.checkbox('Prior Coronary Artery Bypass Graft Surgery')
    dialysis = st.checkbox('Prior dialysis')

st.subheader('At admission')
left_column, right_column = st.columns(2)
with left_column:
    car_shock = st.checkbox('Cardiogenic shock')
    hf = st.checkbox('Heart failure')
    ett = st.checkbox('Endotracheal tube intubated')
    cpr = st.checkbox('Cardiopulmonary resuscitation')
    ext_pace = st.checkbox('On external pacemaker')
    temp_pace = st.checkbox('On temporary pacemaker')
with left_column:
    af = st.checkbox('Atrial fibrillation/flutter')
    svt = st.checkbox('Supraventricular tachycardia')
    non_sus_vt = st.checkbox('Nonsustained ventricular tachycardia')
    vt_vf = st.checkbox('Ventricular tachycardia/fibrillation')
    vt_vf_defib = st.checkbox('Defibrillated ventricular tachycardia/fibrillation')
    chb = st.checkbox('Complete heart block')
    sec_av_block = st.checkbox('Second degree AV-Block')

st.subheader('Echo/Trop_t')
left_column, right_column = st.columns(2)
with left_column:
    echo = st.checkbox('Echo')
    trop_t = st.checkbox('Elevated Troponin')
with right_column:
    ef = st.text_input('EF:', key='ef')

st.subheader('Killip')
left_column, right_column = st.columns(2)
with left_column:
    killip = st.radio('Killip classification:', ['1', '2', '3', '4'])

if st.button('Make Prediction'):
    list_feature = ['age', 'hr','sbp','dbp','rr','o2sat','dm','ht','dlp','smoking','no_smoking','familia_hx','prior_mi','prior_hf','prior_pci','prior_cabg','cva','pad','copd','ckd_stage3','dialysis','cardiogenic_shock','hf','ett','no_ett','tachy_arrhy','af','svt','non_sus_vt','vt_vf','vt_vf_defib','cpr','brady_arrhy','chb','2_av_block','on_ext_pace','on_tpm','echo','ef','elev_trop_t','no_elev_trop_t','killip_1','killip_2','killip_3','killip_4',]
    feature_dict =  dict.fromkeys(list_feature, 0)
    if (int(age) <= 0) | (int(hr) <= 0) | (int(sbp) <= 0) | (int(dbp) <= 0) | (int(rr) <= 0) | (int(oxygen) <= 0):
        st.write('Please complete the field')
    else:
        feature_dict['age'] = int(age)
        feature_dict['hr'] = int(hr)
        feature_dict['sbp'] = int(sbp)
        feature_dict['dbp'] = int(dbp)
        feature_dict['rr'] = int(rr)
        feature_dict['o2sat'] = int(oxygen)
        if dm:
            feature_dict['dm'] = 1
        if ht:
            feature_dict['ht'] = 1
        if dlp:
            feature_dict['dlp'] = 1
        if smoking:
            feature_dict['smoking'] = 1
        else:
            feature_dict['no_smoking'] = 1
        if fam_stemi:
            feature_dict['familia_hx'] = 1
        if pri_mi:
            feature_dict['prior_mi'] = 1
        if pri_hf:
            feature_dict['prior_hf'] = 1
        if pri_pci:
            feature_dict['prior_pci'] = 1
        if pri_cabg:
            feature_dict['prior_cabg'] = 1
        if cva:
            feature_dict['cva'] = 1
        if pad:
            feature_dict['pad'] = 1
        if copd:
            feature_dict['copd'] = 1
        if ckd:
            feature_dict['ckd_stage3'] = 1
        if dialysis:
            feature_dict['dialysis'] = 1
        if car_shock:
            feature_dict['cardiogenic_shock'] = 1
        if hf:
            feature_dict['hf'] = 1
        if ett:
            feature_dict['ett'] = 1
        else:
            feature_dict['no_ett'] = 1
        if af:
            feature_dict['tachy_arrhy'] = 1
            feature_dict['af'] = 1
        if svt:
            feature_dict['tachy_arrhy'] = 1
            feature_dict['svt'] = 1
        if non_sus_vt:
            feature_dict['tachy_arrhy'] = 1
            feature_dict['non_sus_vt'] = 1
        if vt_vf:
            feature_dict['tachy_arrhy'] = 1
            feature_dict['vt_vf'] = 1
        if vt_vf_defib:
            feature_dict['tachy_arrhy'] = 1
            feature_dict['vt_vf_defib'] = 1
        if cpr:
            feature_dict['cpr'] = 1
        if chb:
            feature_dict['brady_arrhy'] = 1
            feature_dict['chb'] = 1
        if sec_av_block:
            feature_dict['brady_arrhy'] = 1
            feature_dict['2_av_block'] = 1
        if ext_pace:
            feature_dict['on_ext_pace'] = 1
        if temp_pace:
            feature_dict['on_tpm'] = 1
        if echo:
            feature_dict['echo'] = 1
            feature_dict['ef'] = ef
        if trop_t:
            feature_dict['elev_trop_t'] = 1
        else:
            feature_dict['no_elev_trop_t'] = 1
        if killip == '1':
            feature_dict['killip_1'] = 1
        elif killip == '2':
            feature_dict['killip_2'] = 1
        elif killip == '3':
            feature_dict['killip_3'] = 1
        elif killip == '4':
            feature_dict['killip_4'] = 1
    feature_df = pd.DataFrame.from_dict(feature_dict, orient='index', columns=['value'])
    feature_df = feature_df.transpose().reset_index(drop=True)
    prediction = model.predict(feature_df)
    st.write(f'This case most likely to {prediction}')


# def make_inp(inp_abo, inp_day, inp_month):
#     keys = ['ABOGROUP_A', 'ABOGROUP_AB', 'ABOGROUP_B', 'ABOGROUP_O', 'weekday_friday', 'weekday_monday', 'weekday_saturday', 'weekday_sunday', 'weekday_thursday', 'weekday_tuesday', 'weekday_wednesday', 'month_April', 'month_August', 'month_December', 'month_February', 'month_January', 'month_July', 'month_June', 'month_March' ,'month_May', 'month_November', 'month_October', 'month_September']
#     feature_dict = dict.fromkeys(keys, 0)
#     for k in feature_dict.keys():
#         if k == f"ABOGROUP_{inp_abo}":
#             feature_dict[k] = 1
#         elif k == f"weekday_{inp_day}":
#             feature_dict[k] = 1
#         elif k == f"month_{inp_month}":
#             feature_dict[k] = 1
#         else:
#             feature_dict[k] = 0
#     cols_order = ['ABOGROUP_A', 'ABOGROUP_B', 'ABOGROUP_AB', 'ABOGROUP_O', 'weekday_friday', 'weekday_monday', 'weekday_saturday', 'weekday_sunday', 'weekday_thursday', 'weekday_tuesday', 'weekday_wednesday', 'month_January', 'month_February', 'month_March', 'month_April', 'month_May', 'month_June', 'month_July', 'month_August', 'month_September', 'month_October', 'month_November', 'month_December']
#     feature_df = pd.DataFrame.from_dict(feature_dict, orient='index', columns=['value'])
#     feature_df = feature_df.transpose().reset_index(drop=True)
#     feature_df = feature_df[cols_order]
#     return feature_df

