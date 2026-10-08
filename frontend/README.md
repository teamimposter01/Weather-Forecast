# 🌐 Weather Forecast System — Frontend Dashboard

Interactive, responsive Scientific Weather Dashboard for the **Hybrid AI–NWP Weather Forecast Blending System for India & Global Locations**.

---

## 🚀 Deploy to Vercel (1-Click / Zero Config)

### Method 1: Via Vercel Web Dashboard (Recommended)
1. Push your repository to GitHub.
2. Go to [vercel.com](https://vercel.com/) and click **"Add New Project"**.
3. Import your repository: `https://github.com/teamimposter01/Weather-Forecast.git`.
4. In **Root Directory**, click **Edit** and select **`frontend`**.
5. Framework Preset: **Other** (Static HTML/JS).
6. Click **Deploy**!

### Method 2: Via Vercel CLI
```bash
cd frontend
vercel
```

---

## 🔗 Connecting to your Render Backend

Once your backend is live on Render (e.g. `https://weather-forecast-backend.onrender.com`):

1. **In the Web UI**:
   - Open your deployed Vercel site.
   - Click the **⚙️ Backend API** button in the header.
   - Paste your Render URL (`https://your-backend.onrender.com`).
   - Click **Test Connection**, then **Save & Connect**.
   - Your URL is saved in `localStorage` so it persists across refreshes!

2. **Or in `config.js`**:
   - Update `DEFAULT_API_URL` in `config.js` with your production Render URL.
