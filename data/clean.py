import pandas as pd

df = pd.read_csv("data/raw/global_air_pollution.csv")

print(df.head())
print(df.info())
print(df.isnull().sum())

# Sütun adlarını standardize et
df.columns = df.columns.str.lower().str.replace(" ", "_")

# Eksik değerleri temizle
df = df.dropna(subset=["aqi_value", "country", "city"])

# AQI kategorilerini sayısala çevir (ML için)
aqi_map = {"Good": 0, "Moderate": 1, "Unhealthy for Sensitive Groups": 2,
           "Unhealthy": 3, "Very Unhealthy": 4, "Hazardous": 5}
df["aqi_category_num"] = df["aqi_category"].map(aqi_map)

df.to_csv("data/processed/clean_air.csv", index=False)
print(f"Temizlendi: {len(df)} satır kaldı")