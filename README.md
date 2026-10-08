# 🇮🇳 Hybrid AI–NWP Multi-Model Weather Forecast Blending System

A production-oriented and research-ready weather forecast blending engine designed to dynamically combine Numerical Weather Prediction (NWP) models (**ECMWF**, **GFS**) with specialized **XGBoost AI Forecast Models** using a **LightGBM Dynamic Weighting Engine** and **Weather Regime Detection**.

---

## 🏗️ Architecture & Project Structure

The project is organized into an independent **Frontend** and **Backend** architecture:

```
Weather-Forecast/
├── frontend/                          # 🌐 Static SPA Dashboard (Optimized for Vercel)
│   ├── index.html                     # Interactive Leaflet & Chart.js dashboard
│   ├── config.js                      # Dynamic API routing configuration
│   ├── vercel.json                    # Vercel deployment configuration
│   ├── package.json                   # Frontend metadata
│   └── README.md                      # Vercel setup instructions
│
├── backend/                           # ⚙️ FastAPI & ML Engine (Optimized for Render)
│   ├── api/                           # Modular FastAPI REST API routes & schemas
│   ├── config/                        # Locations & system settings
│   ├── models/                        # ML base models, weights & regime engines
│   ├── pipeline/                      # Ingest, preprocess, feature & fusion pipelines
│   ├── storage/                       # DuckDB database engine & schema definitions
│   ├── tests/                         # Automated unit & integration tests
│   ├── requirements.txt               # Backend Python dependencies
│   ├── render.yaml                    # Render Blueprint deployment definition
│   ├── Procfile                       # Render Web process runner
│   ├── pytest.ini                     # Pytest configuration
│   └── README.md                      # Render setup instructions
│
├── .gitignore                         # Project-wide git exclusions
└── README.md                          # Main project documentation
```

---

## 🚀 Recommended Deployment Strategy

### Why Frontend on Vercel + Backend on Render?
- **Render for Backend**: Backend packages (`xgboost`, `lightgbm`, `scikit-learn`, `duckdb`) exceed Vercel's strict **250MB serverless bundle limit**. Render runs full persistent Python processes with no bundle size limits and supports live streaming/websockets.
- **Vercel for Frontend**: Instant global CDN edge delivery for static assets, automatic SSL, zero build timeouts, and lightning-fast load times.

---

### 1. Deploy Backend on Render
1. Create a free account on [Render](https://dashboard.render.com/).
2. Click **New +** -> **Web Service**.
3. Select your GitHub repository (`teamimposter01/Weather-Forecast`).
4. Set:
   - **Root Directory**: `backend`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn api.app:app --host 0.0.0.0 --port $PORT`
5. Click **Deploy Web Service**.
6. Copy your public service URL (e.g. `https://weather-forecast-backend.onrender.com`).

---

### 2. Deploy Frontend on Vercel
1. Create a free account on [Vercel](https://vercel.com/).
2. Click **Add New Project** and import `teamimposter01/Weather-Forecast`.
3. In **Root Directory**, select **`frontend`**.
4. Framework Preset: **Other**.
5. Click **Deploy**.
6. Once deployed, open your Vercel URL, click **⚙️ Backend API** in the navigation bar, enter your Render URL, and test the connection!

---

## 💻 Local Development

### 1. Run Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn api.app:app --reload --port 8000
```
- API Docs: `http://localhost:8000/docs`
- Health Check: `http://localhost:8000/api/health`

### 2. Run Automated Tests
```bash
cd backend
pytest
```

### 3. Open Frontend
Open `frontend/index.html` in your web browser or serve via any static server:
```bash
cd frontend
python -m http.server 3000
```
Open `http://localhost:3000`. It will automatically connect to your local backend at `http://localhost:8000`.
