/**
 * Frontend Configuration for Weather Forecast System.
 * Automatically connects to local backend or production Render backend URL.
 */
const CONFIG = {
    // Default API URL (auto-detects local vs production)
    DEFAULT_API_URL: (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1')
        ? 'http://localhost:8000'
        : 'https://weather-forecast-4zdw.onrender.com', // Live Render Backend URL

    getApiBaseUrl() {
        const stored = localStorage.getItem('WEATHER_API_URL');
        if (stored && stored.trim().length > 0) {
            return stored.trim().replace(/\/$/, '');
        }
        return this.DEFAULT_API_URL;
    },

    setApiBaseUrl(url) {
        if (!url || url.trim().length === 0) {
            localStorage.removeItem('WEATHER_API_URL');
        } else {
            localStorage.setItem('WEATHER_API_URL', url.trim().replace(/\/$/, ''));
        }
    }
};

window.CONFIG = CONFIG;
