# ⚙️ Weather Forecast System — Backend Engine & REST API

Production-ready weather forecasting, dynamic multi-model blending engine, and FastAPI backend service.

---

## 🚀 Deploy to Render (Web Service)

### Method 1: Using `render.yaml` Blueprint (Automated)
1. Push your repository to GitHub.
2. In [dashboard.render.com](https://dashboard.render.com/), click **"New"** -> **"Blueprint"**.
3. Connect your repository: `https://github.com/teamimposter01/Weather-Forecast.git`.
4. Render will automatically detect `backend/render.yaml` and configure the Web Service with the correct root directory, build command, and start command.
5. Click **Apply**.

---

### Method 2: Manual Web Service Setup
1. In [Render Dashboard](https://dashboard.render.com/), click **"New +"** -> **"Web Service"**.
2. Connect your GitHub repository.
3. Configure the following fields:
   - **Name**: `weather-forecast-backend`
   - **Root Directory**: `backend`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn api.app:app --host 0.0.0.0 --port $PORT`
   - **Plan**: `Free`
4. Click **Deploy Web Service**.

Once deployed, Render will provide your public backend URL (e.g., `https://weather-forecast-backend.onrender.com`).

---

## 🧪 Running Locally

```bash
cd backend
pip install -r requirements.txt
uvicorn api.app:app --reload --port 8000
```
- Interactive API Docs: `http://localhost:8000/docs`
- Health Endpoint: `http://localhost:8000/api/health`

### Run Test Suite
```bash
pytest
```
