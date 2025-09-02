import os, json, joblib, numpy as np, pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "osteoporosis_screening_model.pkl")
EVAL_PATH  = os.path.join(SCRIPT_DIR, "eval_curves.json")
TRAIN_PKL  = os.path.join(SCRIPT_DIR, "clean_merged.pkl")

app = Flask(__name__)
CORS(app)

model = None
eval_data = None
train_df = None  # used to build cohort defaults

REQUIRED_FEATURES = [
    'WTMECPRP','DR1TSFAT','DR1TCHOL','DRD370M','BMXWAIST','DR1TVB12','DR1TCARB','DRD360',
    'SDMVPSU','BMIARML','DRD350C','DRQSDT5','DRD370Q','DRD370G','DMDYRUSZ','DR1TVB6',
    'DR1TP182','DRD370TQ','DR1TNUMF','RIDRETH1','DR1TLZ','DRD370A','DR1TP204','DR1LANG',
    'MIAPROXY','DRD350GQ','DMDBORN4','DRD370B','DRQSPREP','DR1TM181','DRD350F','INDFMPIR',
    'DR1TM201','DRD350FQ','SIAINTRP','DRD370NQ','DXXOSA','DR1TCAFF','BMXBMI','DRD370U',
    'DRD370D','RIDSTATR','DRQSDIET','DR1TVC','DXXL2A','DBD100','DRD370C','DRABF','SIAPROXY',
    'DR1TS040','DR1DAY','DMDEDUC2','DRD370OQ','DR1TP205','DRD370JQ','DR1TALCO','DR1TP183',
    'DRD350DQ','DRD350EQ','DXXL3A','DR1TACAR','DRD370DQ','RIDEXMON','DRD350D','DR1TPROT',
    'DRD370O','DR1_320Z','DRD350CQ','DRD370UQ','FIAPROXY','DR1TMOIS','DXXNKA','DRDINT',
    'SDDSRVYR','MIAINTRP','SDMVSTRA','BMXARML','WTDRD1PP','BMXHT','DR1TVD','DRD370AQ',
    'DR1TPOTA','DR1EXMER','DRD370MQ','DR1TS080','DRD350K','DRD370F','DR1TMAGN','DR1HELP',
    'DR1STY','DR1_330Z','DRQSDT12','DRD370LQ','DXXWDA','DR1TM161','DRD370HQ','DRQSDT91',
    'DXXOFA','SIALANG','BMIARMC','DR1TS060','AIALANGA','DR1DBIH','DRQSDT1','DRD350H',
    'DXXL1A','DRD370SQ','DR1TNIAC','DR1_300','DR1TCHL','DR1TZINC','DRD370R','BMIWT',
    'DR1TBCAR','DR1TP184','BMXHIP','DRD370BQ','DR1TIRON','WTINTPRP','BMIWAIST','DRD370RQ',
    'DR1TCOPP','DRD350B','DR1SKY','DR1TVB1','DR1TSELE','DR1TVK','DRD350E','DRD370KQ',
    'WTDR2DPP','DRD370P','DR1TP226','DR1TS180','DXXOSBCC','DR1TVB2','DR1TFOLA','DR1TFA',
    'DR1TVARA','DR1TWSZ','DRD350JQ','MIALANG','DR1DRSTZ','DRD370I','DRD370L','DR1TP225',
    'DRD370EQ','BMXWT','BMXLEG','DR1TTFAT','DRQSDT7','DRD370GQ','FIAINTRP','DR1TPFAT',
    'DR1TPHOS','DR1TSUGR','DMDMARTZ','DXXL2BCC','DRQSDT3','BMIHT','DRQSDT8','DRQSDT2',
    'DR1TM221','DR1TFF','DRD350BQ','DRQSDT11','BMXARMC','DR1TS120','DR1BWATZ','DRD350I',
    'DR1MRESP','DXXINA','RIAGENDR','DRD370PQ','DRD370CQ','DXXTRA','DRQSDT9','DXXL4A',
    'DRQSDT4','DR1TS100','DXXL3BCC','DRQSDT10','DRD370K','DR1TRET','DXXL4BCC','DR1TLYCO',
    'RIDAGEYR','DRD370H','DBQ095Z','DRD370E','DRD370J','DR1TMFAT','DR1TFIBE','DRD350A',
    'DRD370T','DRD370V','BMILEG','DR1TS160','DR1TATOC','BMIHIP','DRD350G','DRD370QQ',
    'FIALANG','RIDRETH3','DRD350HQ','DRQSDT6','DRD370FQ','DR1TB12A','DR1TKCAL','DR1TTHEO',
    'DRD350AQ','DR1TCALC','DRD350J','DRD370S','DXXFMBCC','DRD350IQ','DXXL1BCC','DR1TS140',
    'DRD370IQ','DR1TSODI','DR1TCRYP','DR1TFDFE','DRD370N','DRD340','DR1TATOA'
]

USER_TO_MODEL_MAP = {
    'age':     'RIDAGEYR',
    'gender':  'RIAGENDR',
    'bmi':     'BMXBMI',
    'alcohol': 'DR1TALCO',  # UI sends per-week; we convert to per-day
    'calcium': 'DR1TCALC',
    'fracture':'DRD350A'
}

RISK_BANDS = [
    ("Low Risk",      0.00, 0.10),
    ("Moderate Risk", 0.10, 0.30),
    ("High Risk",     0.30, 1.01),
]
def classify_risk(p):
    for label, lo, hi in RISK_BANDS:
        if lo <= p < hi:
            return label
    return "Low Risk"

def _cohort_defaults(age: float, gender: float, bmi: float) -> dict:
    if train_df is None:
        return {col: np.nan for col in REQUIRED_FEATURES}

    df = train_df
    windows = [
        dict(age=3, bmi=2),
        dict(age=5, bmi=3),
        dict(age=8, bmi=5),
        dict(age=None, bmi=None),  # global fallback
    ]
    subset = df
    for w in windows:
        subset = df
        if gender in (1, 2, 1.0, 2.0) and 'RIAGENDR' in subset.columns:
            subset = subset[subset['RIAGENDR'] == gender]
        if w['age'] is not None and 'RIDAGEYR' in subset.columns:
            subset = subset[(subset['RIDAGEYR'] >= age - w['age']) & (subset['RIDAGEYR'] <= age + w['age'])]
        if w['bmi'] is not None and 'BMXBMI' in subset.columns and not np.isnan(bmi):
            subset = subset[(subset['BMXBMI'] >= bmi - w['bmi']) & (subset['BMXBMI'] <= bmi + w['bmi'])]
        if len(subset) >= 80:
            break

    out = {}
    for col in REQUIRED_FEATURES:
        if col in subset.columns:
            s = subset[col]
            if pd.api.types.is_numeric_dtype(s):
                val = float(s.median()) if s.notna().any() else np.nan
            else:
                try:
                    val = s.mode(dropna=True).iloc[0]
                except Exception:
                    val = np.nan
        else:
            val = np.nan

        # global fallback
        if pd.isna(val) and (train_df is not None) and (col in train_df.columns):
            if pd.api.types.is_numeric_dtype(train_df[col]):
                g = train_df[col].dropna()
                val = float(g.median()) if len(g) else np.nan
            else:
                try:
                    val = train_df[col].mode(dropna=True).iloc[0]
                except Exception:
                    val = np.nan
        out[col] = val
    return out

def load_resources():
    global model, eval_data, train_df
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully!")
    if os.path.exists(EVAL_PATH):
        with open(EVAL_PATH, "r") as f:
            eval_data = json.load(f)
        print("✅ Evaluation data loaded successfully!")
    else:
        print(f"⚠️ {EVAL_PATH} not found.")

    # ---- FIX: build a UNIQUE column list before subsetting; then drop dupes defensively ----
    if os.path.exists(TRAIN_PKL):
        df = pd.read_pickle(TRAIN_PKL)
        keep_req = [c for c in REQUIRED_FEATURES if c in df.columns]
        extra = [c for c in ('RIDAGEYR','RIAGENDR','BMXBMI') if c in df.columns]
        # Use dict.fromkeys to preserve order and remove duplicates
        cols = list(dict.fromkeys(keep_req + extra))
        train_df_local = df[cols].copy()
        # If anything still duplicated, drop duplicated columns
        train_df_local = train_df_local.loc[:, ~train_df_local.columns.duplicated()]
        train_df = train_df_local
        print(f"✅ Training data loaded for cohort defaults (rows={len(train_df)})")
    else:
        print(f"⚠️ {TRAIN_PKL} not found; cohort defaults will be NaN (imputer will fill).")

load_resources()

@app.route("/predict_risk", methods=["POST"])
def predict_risk():
    if model is None:
        return jsonify({"status":"error","error":"Model not loaded."}), 500
    try:
        payload = request.get_json(silent=True) or {}

        age    = float(payload.get('age',    np.nan))
        gender = float(payload.get('gender', np.nan))
        bmi    = float(payload.get('bmi',    np.nan))

        row = _cohort_defaults(age, gender, bmi)

        if "alcohol" in payload:
            try:
                payload["alcohol"] = float(payload["alcohol"]) / 7.0  # week -> day
            except Exception:
                pass

        for user_key, model_key in USER_TO_MODEL_MAP.items():
            if user_key in payload:
                try:
                    row[model_key] = float(payload[user_key])
                except Exception:
                    row[model_key] = payload[user_key]

        final_df = pd.DataFrame([row], columns=REQUIRED_FEATURES)
        proba = float(model.predict_proba(final_df)[0][1])
        risk_text = classify_risk(proba)

        return jsonify({"status":"success","prediction_proba":proba,"risk_text":risk_text})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"status":"error","error":str(e)}), 400

@app.route("/get_evaluation_data", methods=["GET"])
def get_evaluation_data():
    if isinstance(eval_data, dict):
        return jsonify(eval_data)
    return jsonify({"status":"error","error":"Evaluation data not available."}), 500

@app.route("/debug_expected_row", methods=["POST"])
def debug_expected_row():
    payload = request.get_json(silent=True) or {}
    age    = float(payload.get('age', np.nan))
    gender = float(payload.get('gender', np.nan))
    bmi    = float(payload.get('bmi', np.nan))
    row = _cohort_defaults(age, gender, bmi)
    if 'alcohol' in payload:
        try: payload['alcohol'] = float(payload['alcohol'])/7.0
        except Exception: pass
    for k,mk in USER_TO_MODEL_MAP.items():
        if k in payload:
            try: row[mk] = float(payload[k])
            except Exception: row[mk] = payload[k]
    return jsonify({"row": row})

@app.route("/train_stats", methods=["GET"])
def train_stats():
    try:
        if train_df is None:
            return jsonify({"status":"error","error":"training df not loaded"}), 500
        full = pd.read_pickle(TRAIN_PKL)
        if "has_osteoporosis" not in full.columns:
            return jsonify({"status":"error","error":"'has_osteoporosis' not in training file"}), 500
        prev = float(full["has_osteoporosis"].mean())
        n = int(full.shape[0])
        return jsonify({"status":"success","prevalence":prev,"n":n,"bands":RISK_BANDS})
    except Exception as e:
        return jsonify({"status":"error","error":str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
