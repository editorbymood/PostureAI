/* =====================================================
   API Client — Backend Communication Module
   
   Handles communication with the FastAPI backend for
   ML-based posture classification (LSTM/MLP models).
   
   Falls back gracefully when backend is unavailable.
   ===================================================== */

const DEFAULT_URL = 'http://localhost:8000';

export class APIClient {
    constructor(baseUrl = DEFAULT_URL) {
        this.baseUrl = baseUrl;
        this.isConnected = false;
        this.latency = 0;
        this._checkInterval = null;
    }

    /** Check if backend is reachable */
    async checkConnection() {
        try {
            const start = performance.now();
            const res = await fetch(`${this.baseUrl}/health`, {
                method: 'GET',
                signal: AbortSignal.timeout(3000),
            });
            this.latency = Math.round(performance.now() - start);
            this.isConnected = res.ok;
            return this.isConnected;
        } catch {
            this.isConnected = false;
            return false;
        }
    }

    /** Start periodic health checks */
    startHealthCheck(intervalMs = 10000) {
        this.checkConnection();
        this._checkInterval = setInterval(() => this.checkConnection(), intervalMs);
    }

    stopHealthCheck() {
        if (this._checkInterval) clearInterval(this._checkInterval);
    }

    /**
     * Send features to backend for LSTM classification.
     * @param {number[][]} window - Sliding window of feature vectors
     * @param {Object|null} calibration - Calibration baseline
     * @returns {Promise<Object|null>} Prediction result or null
     */
    async classify(window, calibration = null) {
        if (!this.isConnected) return null;
        try {
            const start = performance.now();
            const res = await fetch(`${this.baseUrl}/predict`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    features: window,
                    calibration: calibration,
                    timestamp: Date.now(),
                }),
                signal: AbortSignal.timeout(5000),
            });
            this.latency = Math.round(performance.now() - start);
            if (!res.ok) return null;
            return await res.json();
        } catch {
            return null;
        }
    }

    /**
     * Send collected training data to backend.
     * @param {Object[]} samples - Array of {features, label} objects
     */
    async submitTrainingData(samples) {
        if (!this.isConnected) return false;
        try {
            const res = await fetch(`${this.baseUrl}/data/submit`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ samples }),
            });
            return res.ok;
        } catch {
            return false;
        }
    }

    /** Get model info from backend */
    async getModelInfo() {
        if (!this.isConnected) return null;
        try {
            const res = await fetch(`${this.baseUrl}/model/info`);
            return res.ok ? await res.json() : null;
        } catch {
            return null;
        }
    }

    getStatus() {
        return {
            connected: this.isConnected,
            url: this.baseUrl,
            latency: this.latency,
        };
    }
}
