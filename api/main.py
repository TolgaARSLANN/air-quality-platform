import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

# ─────────────────────────────────────────────
# UYGULAMA TANIMI
# ─────────────────────────────────────────────
app = FastAPI(
    title="Hava Kalitesi Tahmin API",
    description="Global hava kirliliği verisiyle eğitilmiş ML modeli üzerinden AQI tahmini yapar.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Frontend bağlantısı için (prod'da kısıtla)
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# MODEL & ENCODER YÜKLEME
# ─────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH    = os.path.join(BASE_DIR, "ml", "model.pkl")
ENCODERS_PATH = os.path.join(BASE_DIR, "ml", "encoders.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "ml", "features.pkl")

try:
    model        = joblib.load(MODEL_PATH)
    encoders     = joblib.load(ENCODERS_PATH)
    feature_cols = joblib.load(FEATURES_PATH)
    print("✅ Model ve encoder'lar yüklendi.")
except Exception as e:
    print(f"❌ Model yüklenemedi: {e}")
    model = encoders = feature_cols = None

# ─────────────────────────────────────────────
# YARDIMCI FONKSİYONLAR
# ─────────────────────────────────────────────
def get_aqi_category(aqi: float) -> dict:
    """AQI değerinden kategori, renk ve sağlık önerisi döndürür."""
    if aqi <= 50:
        return {"category": "Good",                            "color": "#00E400", "health_tip": "Hava kalitesi iyi, dışarıda vakit geçirebilirsiniz."}
    elif aqi <= 100:
        return {"category": "Moderate",                        "color": "#FFFF00", "health_tip": "Hassas bireyler uzun süreli dış aktivitelerden kaçınsın."}
    elif aqi <= 150:
        return {"category": "Unhealthy for Sensitive Groups",  "color": "#FF7E00", "health_tip": "Astım ve kalp hastaları dikkatli olmalı."}
    elif aqi <= 200:
        return {"category": "Unhealthy",                       "color": "#FF0000", "health_tip": "Herkes uzun süreli dış aktivitelerden kaçınsın."}
    elif aqi <= 300:
        return {"category": "Very Unhealthy",                  "color": "#8F3F97", "health_tip": "Dışarı çıkmaktan kaçının, maske takın."}
    else:
        return {"category": "Hazardous",                       "color": "#7E0023", "health_tip": "Acil durum: Dışarı çıkmayın, pencere ve kapıları kapatın."}


def encode_label(encoder, value: str) -> int:
    """Bilinmeyen etiket gelirse -1 döndür (hata vermeden)."""
    try:
        return int(encoder.transform([value])[0])
    except ValueError:
        return -1

# ─────────────────────────────────────────────
# SCHEMA TANIMLARI
# ─────────────────────────────────────────────
class PredictRequest(BaseModel):
    country:         str   = Field(..., example="Turkey")
    city:            str   = Field(..., example="Istanbul")
    co_aqi_value:    float = Field(..., ge=0, example=1.2)
    ozone_aqi_value: float = Field(..., ge=0, example=32.0)
    no2_aqi_value:   float = Field(..., ge=0, example=18.5)
    pm25_aqi_value:  float = Field(..., ge=0, example=55.0)

class PredictResponse(BaseModel):
    city:            str
    country:         str
    predicted_aqi:   float
    aqi_category:    str
    color:           str
    health_tip:      str
    timestamp:       str

class CityStatsResponse(BaseModel):
    city:            str
    country:         str
    record_count:    int
    avg_aqi:         float
    min_aqi:         float
    max_aqi:         float
    dominant_category: str

# ─────────────────────────────────────────────
# ENDPOINT'LER
# ─────────────────────────────────────────────

@app.get("/", tags=["Info"])
def root():
    return {
        "message": "Hava Kalitesi Tahmin API'sine Hoş Geldiniz 🌍",
        "docs":    "/docs",
        "version": "1.0.0",
    }


@app.get("/health", tags=["Info"])
def health_check():
    """API ve model durumunu kontrol eder."""
    return {
        "status":       "ok",
        "model_loaded": model is not None,
        "timestamp":    datetime.utcnow().isoformat(),
    }


@app.post("/predict-aqi", response_model=PredictResponse, tags=["Prediction"])
def predict_aqi(req: PredictRequest):
    """
    Girilen hava kirliliği değerlerine göre AQI tahmini yapar.
    - co_aqi_value: Karbon monoksit AQI değeri
    - ozone_aqi_value: Ozon AQI değeri
    - no2_aqi_value: Nitrojen dioksit AQI değeri
    - pm25_aqi_value: PM2.5 AQI değeri
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model yüklenemedi, sunucu hatası.")

    country_enc = encode_label(encoders["country"], req.country)
    city_enc    = encode_label(encoders["city"],    req.city)

    input_data = pd.DataFrame([{
        "co_aqi_value":    req.co_aqi_value,
        "ozone_aqi_value": req.ozone_aqi_value,
        "no2_aqi_value":   req.no2_aqi_value,
        "pm2.5_aqi_value": req.pm25_aqi_value,
        "country_enc":     country_enc,
        "city_enc":        city_enc,
    }])[feature_cols]

    predicted_aqi = float(round(model.predict(input_data)[0], 2))
    predicted_aqi = max(0, predicted_aqi)   # Negatif değeri engelle

    info = get_aqi_category(predicted_aqi)

    return PredictResponse(
        city=req.city,
        country=req.country,
        predicted_aqi=predicted_aqi,
        aqi_category=info["category"],
        color=info["color"],
        health_tip=info["health_tip"],
        timestamp=datetime.utcnow().isoformat(),
    )


@app.get("/city-stats/{country}/{city}", response_model=CityStatsResponse, tags=["Statistics"])
def city_stats(country: str, city: str):
    """
    Belirtilen şehir için veri setindeki istatistikleri döndürür.
    Örnek: /city-stats/Turkey/Istanbul
    """
    try:
        data_path = os.path.join(BASE_DIR, "data", "processed", "clean_air.csv")
        df = pd.read_csv(data_path)
        df.columns = df.columns.str.lower().str.replace(" ", "_")

        filtered = df[
            (df["country"].str.lower() == country.lower()) &
            (df["city"].str.lower()    == city.lower())
        ]

        if filtered.empty:
            raise HTTPException(
                status_code=404,
                detail=f"'{city}, {country}' için veri bulunamadı."
            )

        dominant_cat = (
            filtered["aqi_category"].value_counts().idxmax()
            if "aqi_category" in filtered.columns else "N/A"
        )

        return CityStatsResponse(
            city=city,
            country=country,
            record_count=len(filtered),
            avg_aqi=round(float(filtered["aqi_value"].mean()), 2),
            min_aqi=round(float(filtered["aqi_value"].min()),  2),
            max_aqi=round(float(filtered["aqi_value"].max()),  2),
            dominant_category=dominant_cat,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/countries", tags=["Statistics"])
def list_countries():
    """Veri setindeki tüm ülkeleri listeler."""
    try:
        data_path = os.path.join(BASE_DIR, "data", "processed", "clean_air.csv")
        df = pd.read_csv(data_path)
        countries = sorted(df["country"].dropna().unique().tolist())
        return {"count": len(countries), "countries": countries}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/cities/{country}", tags=["Statistics"])
def list_cities(country: str):
    """Belirtilen ülkedeki şehirleri listeler."""
    try:
        data_path = os.path.join(BASE_DIR, "data", "processed", "clean_air.csv")
        df = pd.read_csv(data_path)
        filtered = df[df["country"].str.lower() == country.lower()]
        if filtered.empty:
            raise HTTPException(status_code=404, detail=f"'{country}' ülkesi bulunamadı.")
        cities = sorted(filtered["city"].dropna().unique().tolist())
        return {"country": country, "count": len(cities), "cities": cities}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))