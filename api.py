"""
Gridlock 2.0 — Flask REST API
Run: python api.py
Env: DATA_PATH=./data/astram_events.csv  ARTIFACTS_DIR=./artifacts  PORT=5000
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd, numpy as np, pickle, json, os
from datetime import datetime

from pipeline import (
    clean, make_target, engineer, get_X, predict_event,
    build_analytics, ensemble_proba,
    LE_DICT, STAT_MAPS, CLASS_RATE_MAPS, ZONE_HIGH_MAP,
    PS_HIGH_MAP, CORR_HIGH_MAP, GBA_HIGH_MAP,
    THRESHOLDS, FEATURE_COLS
)

app  = Flask(__name__)
CORS(app)

ART  = os.environ.get('ARTIFACTS_DIR', './artifacts')
DATA = os.environ.get('DATA_PATH', './data/astram_events.csv')

print("[startup] Loading artifacts...")
lgbm_model = pickle.load(open(f'{ART}/lgbm_model.pkl','rb'))
cat_model  = pickle.load(open(f'{ART}/cat_model.pkl', 'rb'))
xgb_model  = pickle.load(open(f'{ART}/xgb_model.pkl', 'rb'))
LE_DICT.update(pickle.load(open(f'{ART}/le_dict.pkl','rb')))
STAT_MAPS.update(pickle.load(open(f'{ART}/stat_maps.pkl','rb')))
CLASS_RATE_MAPS.update(pickle.load(open(f'{ART}/class_rate_maps.pkl','rb')))
ZONE_HIGH_MAP.update(pickle.load(open(f'{ART}/zone_high_map.pkl','rb')))
PS_HIGH_MAP.update(pickle.load(open(f'{ART}/ps_high_map.pkl','rb')))
CORR_HIGH_MAP.update(pickle.load(open(f'{ART}/corr_high_map.pkl','rb')))
GBA_HIGH_MAP.update(pickle.load(open(f'{ART}/gba_high_map.pkl','rb')))
THRESHOLDS.update(json.load(open(f'{ART}/thresholds.json')))
analytics_cache = json.load(open(f'{ART}/analytics.json'))
similarity_idx  = json.load(open(f'{ART}/similarity.json'))
df_raw = clean(pd.read_csv(DATA))
feedback_store  = []
print(f"[startup] {len(df_raw)} events loaded. Ready.")

def safe(v):
    if isinstance(v,(np.integer,)): return int(v)
    if isinstance(v,(np.floating,)): return None if np.isnan(v) else float(v)
    if isinstance(v, pd.Timestamp): return str(v)
    try:
        if pd.isna(v): return None
    except Exception: pass
    return v

def to_dict(row): return {k: safe(v) for k,v in row.items()}

@app.route('/health')
def health():
    return jsonify({'status':'ok','model':'LGBM+CatBoost+XGB ensemble (63 features)',
                    'total_events':len(df_raw),'timestamp':datetime.utcnow().isoformat()})

@app.route('/summary')
def summary():
    return jsonify({'success':True,'data':analytics_cache.get('summary',{})})

@app.route('/analytics')
def analytics():
    return jsonify({'success':True,'data':analytics_cache})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data: return jsonify({'success':False,'error':'No JSON body'}), 400
        if 'event_cause' not in data or 'start_datetime' not in data:
            return jsonify({'success':False,'error':'event_cause and start_datetime required'}), 400
        result = predict_event(lgbm_model, cat_model, xgb_model, data)
        return jsonify({'success':True,'input':data,'prediction':result,
                        'generated_at':datetime.utcnow().isoformat()})
    except Exception as e:
        return jsonify({'success':False,'error':str(e)}), 500

@app.route('/similar/<event_id>')
def similar(event_id):
    m = similarity_idx.get(event_id)
    if m is None: return jsonify({'success':False,'error':f'{event_id} not found'}), 404
    return jsonify({'success':True,'event_id':event_id,'similar_events':m})

@app.route('/events')
def events():
    try:
        page  = int(request.args.get('page',1))
        limit = int(request.args.get('limit',20))
        f = df_raw.copy()
        for field in ['event_type','event_cause','priority','zone','corridor','status']:
            val = request.args.get(field)
            if val: f = f[f[field]==val]
        total = len(f)
        cols  = ['id','event_type','event_cause','priority','status','requires_road_closure',
                 'corridor','zone','police_station','latitude','longitude',
                 'start_datetime','end_datetime','duration_mins','gba_identifier']
        cols  = [c for c in cols if c in f.columns]
        start = (page-1)*limit
        recs  = [to_dict(r) for _,r in f[cols].iloc[start:start+limit].iterrows()]
        return jsonify({'success':True,'total':total,'page':page,'limit':limit,
                        'pages':(total+limit-1)//limit,'events':recs})
    except Exception as e:
        return jsonify({'success':False,'error':str(e)}), 500

@app.route('/hotspots')
def hotspots():
    return jsonify({'success':True,
                    'hotspots':analytics_cache.get('hotspots',[]),
                    'congestion_by_corridor':analytics_cache.get('congestion_by_corridor',[])})

@app.route('/feedback', methods=['POST'])
def post_feedback():
    try:
        data = request.get_json()
        data['id'] = f'FB{len(feedback_store):04d}'
        data['submitted_at'] = datetime.utcnow().isoformat()
        feedback_store.append(data)
        correct = data.get('predicted_level') == data.get('actual_level')
        return jsonify({'success':True,'feedback_id':data['id'],
                        'prediction_correct':correct,
                        'message':'Feedback recorded. Improves future predictions.'})
    except Exception as e:
        return jsonify({'success':False,'error':str(e)}), 500

@app.route('/feedback', methods=['GET'])
def get_feedback():
    if not feedback_store:
        return jsonify({'success':True,'records':[],'accuracy':None,'total':0})
    correct  = sum(1 for f in feedback_store if f.get('predicted_level')==f.get('actual_level'))
    accuracy = round(correct/len(feedback_store)*100,1)
    return jsonify({'success':True,'total':len(feedback_store),
                    'accuracy':accuracy,'records':feedback_store})

@app.route('/meta/corridors')
def corridors(): return jsonify({'success':True,'corridors':sorted(df_raw['corridor'].dropna().unique().tolist())})

@app.route('/meta/zones')
def zones(): return jsonify({'success':True,'zones':sorted(df_raw['zone'].dropna().unique().tolist())})

@app.route('/meta/causes')
def causes(): return jsonify({'success':True,'causes':sorted(df_raw['event_cause'].dropna().unique().tolist())})

@app.route('/meta/corporations')
def corporations(): return jsonify({'success':True,'corporations':sorted(df_raw['gba_identifier'].dropna().unique().tolist())})

if __name__ == '__main__':
    port = int(os.environ.get('PORT',5000))
    print(f"[server] http://0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
