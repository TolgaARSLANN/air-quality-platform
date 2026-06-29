import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ─────────────────────────────────────────────
# 1. VERİYİ YÜKLE
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "clean_air.csv")

df = pd.read_csv(DATA_PATH)
print(f"✅ Veri yüklendi: {df.shape[0]} satır, {df.shape[1]} sütun")
print(f"📋 Sütunlar: {list(df.columns)}\n")

# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────

# Kategorik sütunları encode et (Country, City)
le_country = LabelEncoder()
le_city    = LabelEncoder()

df["country_enc"] = le_country.fit_transform(df["country"].astype(str))
df["city_enc"]    = le_city.fit_transform(df["city"].astype(str))

# Modelde kullanılacak özellikler
# Not: Kaggle "Global Air Pollution Dataset" sütun adları kullanıldı.
# Veri setine göre bu listeyi güncelle.
FEATURE_COLS = [
    "co_aqi_value",
    "ozone_aqi_value",
    "no2_aqi_value",
    "pm2.5_aqi_value",
    "country_enc",
    "city_enc",
]

TARGET_COL = "aqi_value"

# Eksik feature sütunlarını kontrol et
available = [c for c in FEATURE_COLS if c in df.columns]
missing   = [c for c in FEATURE_COLS if c not in df.columns]

if missing:
    print(f"⚠️  Şu sütunlar veri setinde bulunamadı, atlanıyor: {missing}")

FEATURE_COLS = available

if TARGET_COL not in df.columns:
    raise ValueError(f"❌ Hedef sütun '{TARGET_COL}' bulunamadı. "
                     f"Mevcut sütunlar: {list(df.columns)}")

# NaN içeren satırları düşür
df_model = df[FEATURE_COLS + [TARGET_COL]].dropna()
print(f"📊 Model için kullanılan satır: {len(df_model)}")
print(f"🎯 Özellikler: {FEATURE_COLS}\n")

X = df_model[FEATURE_COLS]
y = df_model[TARGET_COL]

# ─────────────────────────────────────────────
# 3. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"🔀 Eğitim: {len(X_train)} satır | Test: {len(X_test)} satır\n")

# ─────────────────────────────────────────────
# 4. MODEL EĞİTİMİ
# ─────────────────────────────────────────────
print("🚀 Model eğitiliyor...")

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    n_jobs=-1,         # Tüm CPU çekirdeklerini kullan
    random_state=42,
)
model.fit(X_train, y_train)
print("✅ Eğitim tamamlandı!\n")

# ─────────────────────────────────────────────
# 5. PERFORMANS METRİKLERİ
# ─────────────────────────────────────────────
y_pred = model.predict(X_test)

mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print("=" * 40)
print("📈 MODEL PERFORMANSI")
print("=" * 40)
print(f"  MAE  (Ortalama Mutlak Hata) : {mae:.2f}")
print(f"  RMSE (Kök Ortalama Kare)   : {rmse:.2f}")
print(f"  R²   (Açıklanan Varyans)   : {r2:.4f}")
print("=" * 40)

if r2 >= 0.90:
    print("🟢 Mükemmel model performansı!")
elif r2 >= 0.75:
    print("🟡 İyi model performansı.")
else:
    print("🔴 Model iyileştirme gerekebilir.")

# Feature importance
print("\n🔍 Özellik Önem Sıralaması:")
importances = pd.Series(model.feature_importances_, index=FEATURE_COLS)
for feat, imp in importances.sort_values(ascending=False).items():
    bar = "█" * int(imp * 40)
    print(f"  {feat:<25} {bar} {imp:.3f}")

# ─────────────────────────────────────────────
# 6. MODELİ & ENCODER'LARI KAYDET
# ─────────────────────────────────────────────
MODEL_DIR = os.path.join(BASE_DIR, "ml")
os.makedirs(MODEL_DIR, exist_ok=True)

model_path      = os.path.join(MODEL_DIR, "model.pkl")
encoders_path   = os.path.join(MODEL_DIR, "encoders.pkl")
features_path   = os.path.join(MODEL_DIR, "features.pkl")

joblib.dump(model,                          model_path)
joblib.dump({"country": le_country, "city": le_city}, encoders_path)
joblib.dump(FEATURE_COLS,                   features_path)

print(f"\n💾 Kaydedilen dosyalar:")
print(f"  → {model_path}")
print(f"  → {encoders_path}")
print(f"  → {features_path}")
print("\n🎉 train.py başarıyla tamamlandı!")