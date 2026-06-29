# 🌍 Air Quality & Health Analysis Platform

A full-stack machine learning platform that predicts Air Quality Index (AQI) values using real-world pollution data. Built with FastAPI, scikit-learn, PostgreSQL, Docker, and React.

---

## 📸 Preview

> Real-time AQI prediction dashboard — enter pollutant values for any city and get an instant health assessment.

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| Machine Learning | Python, scikit-learn (Random Forest) |
| Backend API | FastAPI, Uvicorn |
| Database | PostgreSQL |
| Containerization | Docker, Docker Compose |
| Frontend | React (vanilla, no build step) |
| Data Processing | Pandas, NumPy |
| API Testing | Postman |
| DB Management | DBeaver |

---

## 📁 Project Structure

```
air-quality-platform/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── api/
│   └── main.py              # FastAPI application
├── ml/
│   ├── train.py             # Model training script
│   ├── model.pkl            # Trained Random Forest model
│   ├── encoders.pkl         # Label encoders
│   └── features.pkl         # Feature column names
├── data/
│   └── processed/
│       └── clean_air.csv    # Cleaned dataset
└── frontend/
    └── index.html           # React dashboard
```

---

## 🤖 Machine Learning Model

- **Dataset:** [Global Air Pollution Dataset](https://www.kaggle.com/datasets/hasibalmuchus/global-air-pollution-dataset) — 23,035 records
- **Algorithm:** Random Forest Regressor
- **Features:** PM2.5, Ozone, CO, NO₂ AQI values + encoded city/country
- **Target:** AQI value (regression)

| Metric | Score |
|---|---|
| R² (Explained Variance) | **0.9968** |
| MAE (Mean Absolute Error) | 0.30 |
| RMSE | 3.28 |

---

## 🚀 Getting Started

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- Python 3.11+

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/air-quality-platform.git
cd air-quality-platform
```

### 2. Train the model

```bash
pip install -r requirements.txt
python ml/train.py
```

### 3. Start all services

```bash
docker-compose up --build
```

This starts:
- **FastAPI** on `http://localhost:8000`
- **PostgreSQL** on `localhost:5432`

### 4. Open the dashboard

Open `frontend/index.html` in your browser.

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Welcome message |
| GET | `/health` | API & model status |
| POST | `/predict-aqi` | Predict AQI for a city |
| GET | `/city-stats/{country}/{city}` | Historical city statistics |
| GET | `/countries` | List all countries in dataset |
| GET | `/cities/{country}` | List cities for a country |

### Example Request

```bash
POST http://localhost:8000/predict-aqi
Content-Type: application/json

{
  "country": "Turkey",
  "city": "Istanbul",
  "co_aqi_value": 1.2,
  "ozone_aqi_value": 32.0,
  "no2_aqi_value": 18.5,
  "pm25_aqi_value": 55.0
}
```

### Example Response

```json
{
  "city": "Istanbul",
  "country": "Turkey",
  "predicted_aqi": 55.0,
  "aqi_category": "Moderate",
  "color": "#FFFF00",
  "health_tip": "Hassas bireyler uzun süreli dış aktivitelerden kaçınsın.",
  "timestamp": "2024-01-15T10:30:00"
}
```

---

## 🗄️ Database Schema

```sql
-- Historical air quality records
CREATE TABLE air_quality_records (
    id              SERIAL PRIMARY KEY,
    country         VARCHAR(100),
    city            VARCHAR(100),
    aqi_value       FLOAT,
    aqi_category    VARCHAR(50),
    co_aqi_value    FLOAT,
    ozone_aqi_value FLOAT,
    no2_aqi_value   FLOAT,
    pm25_aqi_value  FLOAT,
    recorded_at     TIMESTAMP DEFAULT NOW()
);

-- API prediction logs
CREATE TABLE prediction_logs (
    id            SERIAL PRIMARY KEY,
    country       VARCHAR(100),
    city          VARCHAR(100),
    input_pm25    FLOAT,
    predicted_aqi FLOAT,
    aqi_category  VARCHAR(50),
    created_at    TIMESTAMP DEFAULT NOW()
);
```

---

## 🎨 AQI Color Scale

| AQI Range | Category | Color |
|---|---|---|
| 0 – 50 | Good | 🟢 |
| 51 – 100 | Moderate | 🟡 |
| 101 – 150 | Unhealthy for Sensitive Groups | 🟠 |
| 151 – 200 | Unhealthy | 🔴 |
| 201 – 300 | Very Unhealthy | 🟣 |
| 301+ | Hazardous | ⚫ |

---

## 🗺️ Roadmap

- [ ] Add time-series forecasting (LSTM)
- [ ] City autocomplete from dataset
- [ ] Historical AQI chart per city
- [ ] Deploy to cloud (Railway / Render)

---

## 👤 Author

**Tolga** — Computer Engineering Graduate  
Building AI-powered full-stack applications.

---

## 📄 License

MIT License
