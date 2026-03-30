from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
import pandas as pd
import pickle
import hashlib
import json
import time
import os
import math
import random   # ✅ ADDED
import joblib

app = FastAPI(title="EEW Backend")

# -------------------------------
# CORS
# -------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# PATHS
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USER_DB = os.path.join(BASE_DIR, "users.json")
MODEL_PATH = os.path.join(BASE_DIR, "model_files", "japanmodel.pkl")
STATIONS_PATH = os.path.join(BASE_DIR, "model_files", "stations.csv")

# -------------------------------
# LOAD MODEL + STATIONS
# -------------------------------
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model file missing at: {MODEL_PATH}")

def load_model_compatibly(path):
    try:
        return joblib.load(path, mmap_mode=None)
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")

# Load the model with compatibility mode using pickle as a fallback
def load_model_with_pickle(path):
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load model with pickle: {e}")

try:
    model = load_model_compatibly(MODEL_PATH)
except RuntimeError:
    model = load_model_with_pickle(MODEL_PATH)

stations_df = pd.read_csv(STATIONS_PATH).set_index("station_code")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# -------------------------------
# USER SYSTEM
# -------------------------------
def hash_password(pw: str):
    return hashlib.sha256(pw.encode()).hexdigest()

def load_users():
    try:
        with open(USER_DB, "r") as f:
            return json.load(f)
    except:
        return {}

def save_users(data):
    with open(USER_DB, "w") as f:
        json.dump(data, f, indent=4)

def authenticate(email, password):
    users = load_users()
    hashed = hash_password(password)
    if email in users and users[email]["password"] == hashed:
        return users[email]
    return None

def make_token(email):
    return hashlib.sha256(f"{email}{time.time()}".encode()).hexdigest()

TOKENS = {}

# -------------------------------
# REGISTER
# -------------------------------
class RegisterRequest(BaseModel):
    name: str
    email: str
    city: str
    password: str

@app.post("/register")
def register(data: RegisterRequest):
    users = load_users()
    if data.email in users:
        raise HTTPException(400, "Email already registered")

    users[data.email] = {
        "name": data.name,
        "email": data.email,
        "city": data.city,
        "password": hash_password(data.password)
    }
    save_users(users)
    return {"message": "Registration successful!"}

# -------------------------------
# LOGIN
# -------------------------------
@app.post("/token")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate(form_data.username, form_data.password)
    if not user:
        raise HTTPException(401, "Invalid email or password")

    token = make_token(user["email"])
    TOKENS[token] = user["email"]
    return {"access_token": token, "token_type": "bearer"}

@app.get("/me")
def me(token: str = Depends(oauth2_scheme)):
    if token not in TOKENS:
        raise HTTPException(401, "Invalid token")
    users = load_users()
    return users[TOKENS[token]]

# -------------------------------
# EEW CORE
# -------------------------------
MAX_STATIONS = 15

DANGER_DISTANCE_KM = 300
MODERATE_DISTANCE_KM = 700

def build_features(st_codes, p_times):
    df = pd.DataFrame({"station_code": st_codes, "p_time": p_times})
    df = df.sort_values("p_time").reset_index(drop=True)

    t0 = df["p_time"].iloc[0]
    df["dt"] = df["p_time"] - t0

    df["sta_lat"] = df["station_code"].map(stations_df["sta_lat"])
    df["sta_lon"] = df["station_code"].map(stations_df["sta_lon"])

    if df["sta_lat"].isna().any():
        missing = df[df["sta_lat"].isna()]["station_code"].tolist()
        raise HTTPException(400, f"Unknown station(s): {missing}")

    row = {}
    for k in range(1, MAX_STATIONS + 1):
        if k <= len(df):
            row[f"dt_{k}"] = float(df.loc[k - 1, "dt"])
            row[f"sta_lat_{k}"] = float(df.loc[k - 1, "sta_lat"])
            row[f"sta_lon_{k}"] = float(df.loc[k - 1, "sta_lon"])
        else:
            row[f"dt_{k}"] = 0.0
            row[f"sta_lat_{k}"] = 0.0
            row[f"sta_lon_{k}"] = 0.0

    cols = []
    for k in range(1, MAX_STATIONS + 1):
        cols += [f"dt_{k}", f"sta_lat_{k}", f"sta_lon_{k}"]

    return pd.DataFrame([row])[cols]

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    a = (
        math.sin(dLat / 2) ** 2 +
        math.cos(math.radians(lat1)) *
        math.cos(math.radians(lat2)) *
        math.sin(dLon / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

# -------------------------------
# JAPAN DATA
# -------------------------------
JAPAN_CITIES = {
    "Tokyo": (35.6762, 139.6503),
    "Osaka": (34.6937, 135.5023),
    "Sapporo": (43.0618, 141.3545),
    "Sendai": (38.2682, 140.8694),
    "Nagoya": (35.1815, 136.9066),
    "Fukuoka": (33.5903, 130.4017),
    "Kyoto": (35.0116, 135.7681)
}

# -------------------------------
# RANDOM JAPAN EPICENTER (KEY FIX)
# -------------------------------
def random_japan_epicenter():
    # Japan seismic belt (Pacific side)
    lat = random.uniform(32.0, 41.5)
    lon = random.uniform(135.0, 145.5)
    return lat, lon

# -------------------------------
# PREDICT API
# -------------------------------
@app.post("/predict")
def predict(payload: dict):
    mode = payload.get("mode", "real")
    codes = payload["station_codes"]
    times = payload["arrival_times"]

    if mode == "japan_sample":
        # 🔥 RANDOMIZED EVERY REQUEST
        pred_lat, pred_lon = random_japan_epicenter()
        earliest_station = codes[0]
        t0 = 0.0
        now = 0.0
    else:
        X = build_features(codes, times)
        pred_lat, pred_lon = model.predict(X)[0]

        earliest_index = times.index(min(times))
        earliest_station = codes[earliest_index]

        sta_lat = stations_df.loc[earliest_station, "sta_lat"]
        sta_lon = stations_df.loc[earliest_station, "sta_lon"]

        Vp = 6.0
        dist_station = haversine(pred_lat, pred_lon, sta_lat, sta_lon)
        t0 = times[earliest_index] - (dist_station / Vp)
        now = times[earliest_index]

    Vs = 3.5
    warnings = {}

    for city, (clat, clon) in JAPAN_CITIES.items():
        d = haversine(pred_lat, pred_lon, clat, clon)
        s_arrival = t0 + d / Vs
        warn = s_arrival - now

        if d <= DANGER_DISTANCE_KM:
            level = "Danger"
            early_warning = True
        elif d <= MODERATE_DISTANCE_KM:
            level = "Moderate"
            early_warning = True
        else:
            level = "Safe"
            early_warning = False

        warnings[city] = {
            "distance_km": round(d, 2),
            "s_arrival": None if warn < 0 else round(s_arrival, 2),
            "warning_seconds": None if warn < 0 else round(warn, 2),
            "risk_level": level,
            "early_warning": early_warning
        }

    return {
        "mode": mode,
        "pred_lat": round(float(pred_lat), 4),
        "pred_lon": round(float(pred_lon), 4),
        "earliest_station": earliest_station,
        "t0": round(t0, 2),
        "warnings": warnings,
        "s_wave_ring_max_km": max(v["distance_km"] for v in warnings.values())
    }
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
