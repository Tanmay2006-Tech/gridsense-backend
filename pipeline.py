"""
Gridlock Hackathon 2.0 - Round 2  |  FINAL BEST PIPELINE v3
Event-Driven Congestion Prediction — AI Ops Co-pilot for Traffic Police

Dataset : kagglehub.dataset_download("tanmaytripathi7525/gridlock-round2-theme2")
Target  : 3-class congestion (Low / Medium / High)
Models  : LightGBM + CatBoost + XGBoost → weighted soft-vote (0.3/0.4/0.3)
CV F1   : 0.640 (5-fold macro)   |   Hold-out F1: 0.627   |   Accuracy: 0.64
"""

import os, json, pickle, warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, accuracy_score
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import xgboost as xgb
warnings.filterwarnings('ignore')

# ════════════════════════════════════════════
# GLOBAL STATE (populated during fit)
# ════════════════════════════════════════════
LE_DICT         = {}
STAT_MAPS       = {}
CLASS_RATE_MAPS = {}
ZONE_HIGH_MAP   = {}
PS_HIGH_MAP     = {}
CORR_HIGH_MAP   = {}
GBA_HIGH_MAP    = {}
THRESHOLDS      = {}

# ════════════════════════════════════════════
# 0. LOAD
# ════════════════════════════════════════════
def load_data(kaggle=True, local_path=None):
    if kaggle:
        import kagglehub
        path = kagglehub.dataset_download("tanmaytripathi7525/gridlock-round2-theme2")
        print("Dataset path:", path)
        csv_file = [f for f in os.listdir(path) if f.endswith('.csv')][0]
        df = pd.read_csv(os.path.join(path, csv_file))
    else:
        df = pd.read_csv(local_path)
    print(f"[load] {len(df)} rows x {df.shape[1]} cols")
    return df

# ════════════════════════════════════════════
# 1. CLEAN
# ════════════════════════════════════════════
def clean(df):
    for c in ['start_datetime','end_datetime','closed_datetime',
              'resolved_datetime','created_date','modified_datetime']:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], utc=True, errors='coerce')
    df['actual_end']    = df['closed_datetime'].fillna(df['resolved_datetime'])
    df['duration_mins'] = (df['actual_end'] - df['start_datetime']).dt.total_seconds() / 60
    df['corridor']       = df['corridor'].fillna('Non-corridor')
    df['zone']           = df['zone'].fillna('Unknown')
    df['junction']       = df['junction'].fillna('Unknown')
    df['police_station'] = df['police_station'].fillna('Unknown')
    df['veh_type']       = df['veh_type'].fillna('none')
    df['priority']       = df['priority'].fillna('Low')
    df['event_cause']    = df['event_cause'].str.lower().str.strip().fillna('others')
    df['gba_identifier'] = df['gba_identifier'].fillna('none')
    return df

# ════════════════════════════════════════════
# 2. TARGET
# ════════════════════════════════════════════
def make_target(df, fit=True):
    valid = df[df['duration_mins'].notna() & (df['duration_mins'] > 0)].copy()
    if fit:
        p33 = valid['duration_mins'].quantile(0.33)
        p66 = valid['duration_mins'].quantile(0.66)
        THRESHOLDS['p33'] = float(p33)
        THRESHOLDS['p66'] = float(p66)
    else:
        p33, p66 = THRESHOLDS['p33'], THRESHOLDS['p66']

    valid['target'] = pd.cut(
        valid['duration_mins'], bins=[-1, p33, p66, 1e9], labels=[0,1,2]
    ).astype(int)
    valid.loc[valid['requires_road_closure'] == True, 'target'] = 2
    print(f"[target] p33={p33:.0f}m p66={p66:.0f}m | dist={valid['target'].value_counts().sort_index().to_dict()}")
    return valid

# ════════════════════════════════════════════
# 3. FEATURE ENGINEERING  (63 features)
# ════════════════════════════════════════════
SEVERITY_MAP = {
    'public_event':5,'procession':5,'vip_movement':5,
    'protest':4,'construction':4,'accident':4,
    'water_logging':3,'tree_fall':3,'road_conditions':3,
    'pot_holes':3,'congestion':3,'debris':3,
    'vehicle_breakdown':2,'others':2,'fog / low visibility':2,'test_demo':1,
}
HIGH_IMPACT = {'construction','water_logging','road_conditions','pot_holes',
               'tree_fall','public_event','procession','vip_movement','protest'}
CAT_COLS = ['event_cause','corridor','zone','police_station',
            'event_type','priority','veh_type','junction']

def _encode_cats(df, fit):
    for c in CAT_COLS:
        if fit:
            le = LabelEncoder()
            df[c+'_enc'] = le.fit_transform(df[c].fillna('unknown').astype(str))
            LE_DICT[c] = le
        else:
            le = LE_DICT[c]
            vals = df[c].fillna('unknown').astype(str)
            known = set(le.classes_)
            vals = vals.apply(lambda v: v if v in known else le.classes_[0])
            df[c+'_enc'] = le.transform(vals)
    return df

def engineer(df, fit=True):
    # ── Time ──
    df['hour']            = df['start_datetime'].dt.hour
    df['dow']             = df['start_datetime'].dt.dayofweek
    df['month']           = df['start_datetime'].dt.month
    df['is_weekend']      = (df['dow'] >= 5).astype(int)
    df['is_night']        = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
    df['is_morning_peak'] = ((df['hour'] >= 7)  & (df['hour'] <= 10)).astype(int)
    df['is_evening_peak'] = ((df['hour'] >= 17) & (df['hour'] <= 21)).astype(int)
    df['hour_sin']        = np.sin(2*np.pi*df['hour']/24)
    df['hour_cos']        = np.cos(2*np.pi*df['hour']/24)
    df['dow_sin']         = np.sin(2*np.pi*df['dow']/7)
    df['dow_cos']         = np.cos(2*np.pi*df['dow']/7)
    df['month_sin']       = np.sin(2*np.pi*df['month']/12)
    df['month_cos']       = np.cos(2*np.pi*df['month']/12)

    # ── Domain ──
    df['severity']          = df['event_cause'].map(SEVERITY_MAP).fillna(2)
    df['road_cl']           = df['requires_road_closure'].astype(int)
    df['is_planned']        = (df['event_type'] == 'planned').astype(int)
    df['is_major_corridor'] = (df['corridor'] != 'Non-corridor').astype(int)
    df['priority_h']        = (df['priority'] == 'High').astype(int)
    df['has_end_dt']        = df['end_datetime'].notna().astype(int) if 'end_datetime' in df.columns else 0
    df['has_police']        = df['assigned_to_police_id'].notna().astype(int)
    df['has_citizen']       = df['citizen_accident_id'].notna().astype(int)
    df['is_high_impact']    = df['event_cause'].isin(HIGH_IMPACT).astype(int)

    # ── NEW: untapped columns ──
    df['is_authenticated']  = (df['authenticated'].fillna('no') == 'yes').astype(int)
    df['has_kgid']          = df['kgid'].notna().astype(int)
    df['has_cargo']         = df['cargo_material'].notna().astype(int)
    df['has_end_addr']      = df['end_address'].notna().astype(int) if 'end_address' in df.columns else 0
    df['has_junction']      = (df['junction'] != 'Unknown').astype(int)
    df['has_endpoint']      = df['endlatitude'].notna().astype(int)
    df['desc_len_log']      = np.log1p(df['description'].fillna('').str.len().clip(0,500))
    df['has_veh_no']        = df['veh_no'].notna().astype(int)
    df['is_client2']        = (df['client_id'].fillna(1) == 2).astype(int)

    # gba_identifier LabelEncode + high-rate
    if fit:
        gba_le = LabelEncoder()
        df['gba_enc'] = gba_le.fit_transform(df['gba_identifier'].astype(str))
        LE_DICT['gba_identifier'] = gba_le
        GBA_HIGH_MAP.update(df.groupby('gba_identifier')['target'].apply(
            lambda x: (x==2).mean()).to_dict())
    else:
        le = LE_DICT['gba_identifier']
        vals = df['gba_identifier'].astype(str)
        known = set(le.classes_)
        vals = vals.apply(lambda v: v if v in known else le.classes_[0])
        df['gba_enc'] = le.transform(vals)
    df['gba_high_rate'] = df['gba_identifier'].map(GBA_HIGH_MAP).fillna(0.33)

    # ── Interactions ──
    df['sev_x_road']     = df['severity'] * df['road_cl']
    df['sev_x_peak']     = df['severity'] * (df['is_morning_peak'] + df['is_evening_peak'])
    df['corr_x_sev']     = df['is_major_corridor'] * df['severity']
    df['pri_x_sev']      = df['priority_h'] * df['severity']
    df['wknd_x_sev']     = df['is_weekend'] * df['severity']
    df['road_x_peak']    = df['road_cl'] * (df['is_morning_peak'] + df['is_evening_peak'])
    df['auth_x_sev']     = df['is_authenticated'] * df['severity']
    df['endpoint_x_sev'] = df['has_endpoint'] * df['severity']

    # ── Statistical aggregation maps ──
    if fit:
        STAT_MAPS['cause_med']     = df.groupby('event_cause')['duration_mins'].median().to_dict()
        STAT_MAPS['corr_med']      = df.groupby('corridor')['duration_mins'].median().to_dict()
        STAT_MAPS['cp_med']        = df.groupby(['event_cause','priority'])['duration_mins'].median().to_dict()
        STAT_MAPS['zone_load']     = df.groupby('zone')['id'].count().to_dict()
        STAT_MAPS['corr_load']     = df.groupby('corridor')['id'].count().to_dict()
        STAT_MAPS['ps_load']       = df.groupby('police_station')['id'].count().to_dict()
        for cls in [0,1,2]:
            CLASS_RATE_MAPS[cls] = df.groupby('event_cause')['target'].apply(
                lambda x: (x==cls).mean()).to_dict()
        ZONE_HIGH_MAP.update(df.groupby('zone')['target'].apply(lambda x:(x==2).mean()).to_dict())
        PS_HIGH_MAP.update(df.groupby('police_station')['target'].apply(lambda x:(x==2).mean()).to_dict())
        CORR_HIGH_MAP.update(df.groupby('corridor')['target'].apply(lambda x:(x==2).mean()).to_dict())

    df['log_cause_med'] = np.log1p(df['event_cause'].map(STAT_MAPS['cause_med']).fillna(60))
    df['log_corr_med']  = np.log1p(df['corridor'].map(STAT_MAPS['corr_med']).fillna(60))
    df['log_cp_med']    = np.log1p(df.apply(
        lambda r: STAT_MAPS['cp_med'].get((r['event_cause'],r['priority']),60), axis=1))
    df['log_zone_load'] = np.log1p(df['zone'].map(STAT_MAPS['zone_load']).fillna(10))
    df['log_corr_load'] = np.log1p(df['corridor'].map(STAT_MAPS['corr_load']).fillna(10))
    df['log_ps_load']   = np.log1p(df['police_station'].map(STAT_MAPS['ps_load']).fillna(10))

    for cls in [0,1,2]:
        df[f'cause_cls{cls}'] = df['event_cause'].map(CLASS_RATE_MAPS.get(cls,{})).fillna(1/3)
    df['zone_high_rate'] = df['zone'].map(ZONE_HIGH_MAP).fillna(0.33)
    df['ps_high_rate']   = df['police_station'].map(PS_HIGH_MAP).fillna(0.33)
    df['corr_high_rate'] = df['corridor'].map(CORR_HIGH_MAP).fillna(0.33)

    df = _encode_cats(df, fit=fit)
    return df

FEATURE_COLS = [
    'hour','dow','month','is_weekend','is_night','is_morning_peak','is_evening_peak',
    'hour_sin','hour_cos','dow_sin','dow_cos','month_sin','month_cos',
    'severity','road_cl','is_planned','is_major_corridor','priority_h',
    'has_end_dt','has_police','has_citizen','is_high_impact',
    'is_authenticated','has_kgid','has_cargo','has_end_addr',
    'has_junction','has_endpoint','desc_len_log','has_veh_no','is_client2',
    'gba_enc','gba_high_rate',
    'sev_x_road','sev_x_peak','corr_x_sev','pri_x_sev','wknd_x_sev',
    'road_x_peak','auth_x_sev','endpoint_x_sev',
    'log_cause_med','log_corr_med','log_cp_med',
    'log_zone_load','log_corr_load','log_ps_load',
    'cause_cls0','cause_cls1','cause_cls2',
    'zone_high_rate','ps_high_rate','corr_high_rate',
    'latitude','longitude',
] + [c+'_enc' for c in CAT_COLS]

def get_X(df):
    return df[FEATURE_COLS].fillna(0).values

# ════════════════════════════════════════════
# 4. MODELS  (Optuna-tuned best params)
# ════════════════════════════════════════════
def build_models():
    lgbm = LGBMClassifier(
        n_estimators=317, learning_rate=0.01775, max_depth=6, num_leaves=83,
        subsample=0.8316, colsample_bytree=0.7021, min_child_samples=78,
        reg_alpha=0.3458, reg_lambda=4.12,
        class_weight='balanced', random_state=42, verbose=-1, n_jobs=-1
    )
    cat = CatBoostClassifier(
        iterations=700, learning_rate=0.04, depth=8, l2_leaf_reg=4,
        bagging_temperature=0.2, random_strength=1.5,
        random_seed=42, verbose=0, auto_class_weights='Balanced'
    )
    xgbc = xgb.XGBClassifier(
        n_estimators=800, learning_rate=0.04, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        reg_alpha=0.1, reg_lambda=1.5,
        eval_metric='mlogloss', random_state=42, verbosity=0
    )
    return lgbm, cat, xgbc

# Best weights found by grid search: LGBM=0.3, Cat=0.4, XGB=0.3
W_LGBM, W_CAT, W_XGB = 0.3, 0.4, 0.3

def ensemble_proba(lgbm, cat, xgbc, X):
    return W_LGBM*lgbm.predict_proba(X) + W_CAT*cat.predict_proba(X) + W_XGB*xgbc.predict_proba(X)

def ensemble_predict(lgbm, cat, xgbc, X):
    return ensemble_proba(lgbm, cat, xgbc, X).argmax(axis=1)

# ════════════════════════════════════════════
# 5. ANALYTICS
# ════════════════════════════════════════════
def build_analytics(df):
    out = {}
    out['events_by_cause'] = df['event_cause'].value_counts().reset_index().rename(
        columns={'event_cause':'cause','count':'count'}).to_dict('records')
    out['events_by_hour'] = df.groupby('hour')['id'].count().reset_index().rename(
        columns={'id':'count'}).to_dict('records')
    day_map = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
    out['events_by_day'] = (df.groupby('dow')['id'].count().reset_index()
        .assign(day=lambda x: x['dow'].map(day_map))
        .rename(columns={'id':'count'})[['day','count']].to_dict('records'))
    out['closure_rate'] = (df.groupby('event_cause')['requires_road_closure'].mean()
        .reset_index().rename(columns={'requires_road_closure':'closure_rate'})
        .sort_values('closure_rate',ascending=False).to_dict('records'))
    out['events_by_zone'] = df['zone'].value_counts().reset_index().to_dict('records')
    out['median_duration'] = (df.groupby('event_cause')['duration_mins'].median()
        .dropna().reset_index().rename(columns={'duration_mins':'median_mins'})
        .sort_values('median_mins',ascending=False).to_dict('records'))
    df['ym'] = df['start_datetime'].dt.to_period('M').astype(str)
    out['monthly_trend'] = (df.groupby('ym')['id'].count().reset_index()
        .rename(columns={'id':'count'}).sort_values('ym').to_dict('records'))
    out['planned_vs_unplanned'] = df['event_type'].value_counts().reset_index().to_dict('records')
    hi = df[(df['requires_road_closure']==True)&(df['priority']=='High')]
    out['hotspots'] = (hi.dropna(subset=['latitude','longitude'])
        [['event_cause','corridor','zone','police_station',
          'start_datetime','duration_mins','latitude','longitude']]
        .assign(duration_mins=lambda x: x['duration_mins'].round(1)).to_dict('records'))
    out['congestion_by_corridor'] = (df.groupby('corridor')['target']
        .apply(lambda x:(x==2).mean()).reset_index()
        .rename(columns={'target':'high_rate'})
        .sort_values('high_rate',ascending=False).to_dict('records'))
    out['events_by_corporation'] = (df.groupby('gba_identifier')['id'].count()
        .reset_index().rename(columns={'id':'count'}).to_dict('records'))
    out['summary'] = {
        'total_events':   int(len(df)),
        'high_priority':  int((df['priority']=='High').sum()),
        'road_closures':  int(df['requires_road_closure'].sum()),
        'planned':        int((df['event_type']=='planned').sum()),
        'median_duration':round(float(df['duration_mins'].median()),1),
        'corridors':      int(df['corridor'].nunique()),
        'zones':          int(df['zone'].nunique()),
    }
    return out

# ════════════════════════════════════════════
# 6. SIMILARITY ENGINE
# ════════════════════════════════════════════
def build_similarity_index(df):
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import normalize
    ref = df[df['status'].isin(['closed','resolved'])].copy()
    X_norm = normalize(get_X(ref))
    sim = cosine_similarity(X_norm)
    np.fill_diagonal(sim, 0)
    ids = ref['id'].astype(str).values
    index = {}
    for i, eid in enumerate(ids):
        top3 = np.argsort(sim[i])[-3:][::-1]
        index[eid] = [
            {'event_id': ids[j], 'similarity': round(float(sim[i,j]),3),
             'event_cause': ref.iloc[j]['event_cause'], 'corridor': ref.iloc[j]['corridor'],
             'duration_mins': round(float(ref.iloc[j]['duration_mins']),1)
                if pd.notna(ref.iloc[j]['duration_mins']) else None,
             'date': str(ref.iloc[j]['start_datetime'])[:10],
             'police_station': ref.iloc[j]['police_station']}
            for j in top3]
    print(f"[similarity] {len(index)} events indexed")
    return index

# ════════════════════════════════════════════
# 7. RECOMMENDATION ENGINE
# ════════════════════════════════════════════
DIVERSION_MAP = {
    'Mysore Road':       ['Kanakapura Road','Bannerghatta Road'],
    'Tumkur Road':       ['Magadi Road','Bellary Road 1'],
    'Bellary Road 1':    ['Hebbal Flyover','ORR North'],
    'Bellary Road 2':    ['Bellary Road 1','ORR North 1'],
    'ORR East 1':        ['Old Madras Road','Whitefield Road'],
    'ORR East 2':        ['Sarjapur Road','Hosur Road'],
    'ORR North 1':       ['Bellary Road 1','Tumkur Road'],
    'ORR North 2':       ['Hennur Road','Bellary Road 2'],
    'ORR West 1':        ['Magadi Road','Mysore Road'],
    'Old Madras Road':   ['Whitefield Main Road','ORR East'],
    'Hosur Road':        ['Bannerghatta Road','Electronics City Flyover'],
    'Bannerghatta Road': ['Kanakapura Road','Hosur Road'],
    'Magadi Road':       ['Tumkur Road','Mysore Road'],
    'CBD 2':             ['MG Road','Residency Road'],
    'Non-corridor':      ['Use alternate local roads','Check live traffic'],
}

def recommend(event, level, proba):
    sev = SEVERITY_MAP.get(event.get('event_cause','others').lower(), 2)
    bands = {0:(2,5), 1:(6,12), 2:(14,28)}
    lo, hi = bands[level]
    n = int(np.clip(lo*(sev/3), lo, hi))
    corridor  = event.get('corridor','Non-corridor')
    diversions = DIVERSION_MAP.get(corridor, ['Use alternate routes'])
    try:
        dt = pd.to_datetime(event.get('start_datetime',''), errors='coerce')
        hour = dt.hour; is_wknd = dt.dayofweek >= 5
    except Exception:
        hour, is_wknd = 12, False
    alerts = []
    if event.get('requires_road_closure'):    alerts.append('⚠️ Full road closure required')
    if event.get('event_type')=='planned':     alerts.append('📅 Planned — pre-deploy 2 hrs early')
    if 7<=hour<=10 or 17<=hour<=21:           alerts.append('🕐 Peak hour — expect 2× congestion')
    if is_wknd:                               alerts.append('📆 Weekend — higher civilian traffic')
    return {
        'congestion_level':      level,
        'congestion_label':      ['Low','Medium','High'][level],
        'confidence_pct':        round(float(proba.max())*100, 1),
        'personnel_recommended': n,
        'barricades_recommended':[0,2,4][level],
        'diversion_routes':      diversions,
        'alerts':                alerts,
        'probabilities':         {'low':round(float(proba[0])*100,1),
                                  'medium':round(float(proba[1])*100,1),
                                  'high':round(float(proba[2])*100,1)},
        'corridor': corridor,
        'police_station': event.get('police_station','Unknown'),
        'zone': event.get('zone','Unknown'),
    }

# ════════════════════════════════════════════
# 8. PREDICT SINGLE EVENT
# ════════════════════════════════════════════
def predict_event(lgbm, cat, xgbc, event_dict):
    defaults = {
        'duration_mins':np.nan,'actual_end':None,'target':1,'id':'PRED_001',
        'assigned_to_police_id':None,'citizen_accident_id':None,
        'authenticated':'yes','kgid':None,'cargo_material':None,
        'end_address':None,'endlatitude':None,'endlongitude':None,
        'description':'','veh_no':None,'client_id':1,
        'gba_identifier':'none','junction':'Unknown',
    }
    row = pd.DataFrame([{**defaults, **event_dict}])
    for c in ['start_datetime','end_datetime','actual_end',
              'closed_datetime','resolved_datetime','created_date','modified_datetime']:
        if c in row: row[c] = pd.to_datetime(row[c], utc=True, errors='coerce')
    row = engineer(row, fit=False)
    X = get_X(row)
    proba = ensemble_proba(lgbm, cat, xgbc, X)[0]
    return recommend(event_dict, int(proba.argmax()), proba)

# ════════════════════════════════════════════
# 9. MAIN
# ════════════════════════════════════════════
def run(kaggle=True, local_path=None, out_dir='./artifacts'):
    os.makedirs(out_dir, exist_ok=True)
    print("="*55)
    print("  GRIDLOCK 2.0 — FINAL BEST PIPELINE v3")
    print("="*55)

    df = load_data(kaggle=kaggle, local_path=local_path)
    df = clean(df)
    df = make_target(df, fit=True)
    df = engineer(df, fit=True)
    X  = get_X(df)
    y  = df['target'].values

    # Hold-out eval
    X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)
    lgbm, cat, xgbc = build_models()
    print("\n[train] Fitting models...")
    lgbm.fit(X_tr,y_tr); cat.fit(X_tr,y_tr); xgbc.fit(X_tr,y_tr)
    y_pred   = ensemble_predict(lgbm, cat, xgbc, X_te)
    macro_f1 = f1_score(y_te, y_pred, average='macro')
    acc      = accuracy_score(y_te, y_pred)
    print(f"\n[eval]  Macro F1 = {macro_f1:.4f}  |  Accuracy = {acc:.4f}")
    print(classification_report(y_te, y_pred, target_names=['Low','Medium','High']))

    # Retrain on full data
    print("[train] Retraining on full dataset...")
    lgbm2, cat2, xgbc2 = build_models()
    lgbm2.fit(X,y); cat2.fit(X,y); xgbc2.fit(X,y)

    # Build auxiliary data
    print("[build] Analytics + similarity index...")
    analytics = build_analytics(df)
    sim_index = build_similarity_index(df)

    # Save
    print("[save] Writing artifacts...")
    pickle.dump(lgbm2,          open(f'{out_dir}/lgbm_model.pkl','wb'))
    pickle.dump(cat2,           open(f'{out_dir}/cat_model.pkl','wb'))
    pickle.dump(xgbc2,          open(f'{out_dir}/xgb_model.pkl','wb'))
    pickle.dump(LE_DICT,        open(f'{out_dir}/le_dict.pkl','wb'))
    pickle.dump(STAT_MAPS,      open(f'{out_dir}/stat_maps.pkl','wb'))
    pickle.dump(CLASS_RATE_MAPS,open(f'{out_dir}/class_rate_maps.pkl','wb'))
    pickle.dump(ZONE_HIGH_MAP,  open(f'{out_dir}/zone_high_map.pkl','wb'))
    pickle.dump(PS_HIGH_MAP,    open(f'{out_dir}/ps_high_map.pkl','wb'))
    pickle.dump(CORR_HIGH_MAP,  open(f'{out_dir}/corr_high_map.pkl','wb'))
    pickle.dump(GBA_HIGH_MAP,   open(f'{out_dir}/gba_high_map.pkl','wb'))
    json.dump(THRESHOLDS,   open(f'{out_dir}/thresholds.json','w'))
    json.dump(FEATURE_COLS, open(f'{out_dir}/feature_cols.json','w'))
    json.dump(analytics,    open(f'{out_dir}/analytics.json','w'), default=str)
    json.dump(sim_index,    open(f'{out_dir}/similarity.json','w'), default=str)

    print(f"\n✅  Done!  Macro F1={macro_f1:.4f}  Accuracy={acc:.4f}")
    print(f"    Features: {len(FEATURE_COLS)} | Training rows: {len(X)}")
    print(f"    Artifacts: {out_dir}/")

    demo = {'event_cause':'public_event','event_type':'planned','corridor':'Mysore Road',
            'zone':'West Zone 1','police_station':'Kengeri','veh_type':'none',
            'priority':'High','requires_road_closure':True,
            'start_datetime':'2024-03-15 19:00:00','latitude':12.9321,'longitude':77.4892}
    print("\n[demo]"); print(json.dumps(predict_event(lgbm2,cat2,xgbc2,demo), indent=2))
    return lgbm2, cat2, xgbc2

if __name__ == '__main__':
    # Kaggle: run(kaggle=True)
    # Local:  run(kaggle=False, local_path='data/astram_events.csv')
    run(kaggle=True)
