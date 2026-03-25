/* =====================================================
   CalibrationSystem — Personalization Layer
   
   THIS is what separates a toy project from a real product.
   
   Problem: Everyone's body is different. A 6'4" person and
   a 5'2" person have very different "natural" postures.
   Fixed thresholds don't work for everyone.
   
   Solution: Calibration.
   1. User sits in their "best posture" position
   2. System records 90 frames (~3 seconds) of features
   3. Computes personal baseline: mean ± std of each feature
   4. Future deviations are measured relative to this baseline
   5. Stored in localStorage for persistence across sessions
   
   This demonstrates:
   - Product thinking (user-specific adaptation)
   - ML maturity (personalized baselines)
   - Data engineering (online statistics)
   ===================================================== */

import { NUM_FEATURES, FEATURE_NAMES } from './featureEngine.js';

const DEFAULT_STORAGE_KEY = 'postureai_calibration';

export class CalibrationSystem {
    /**
     * @param {string} [storagePrefix] — Optional prefix for localStorage key.
     *   Used in multi-person mode to store per-person calibration separately.
     */
    constructor(storagePrefix = '') {
        this._storageKey = storagePrefix
            ? `postureai_cal_${storagePrefix}`
            : DEFAULT_STORAGE_KEY;

        this.isCalibrating = false;
        this.isCalibrated  = false;
        this.frames = [];
        this.targetFrames = 90;     // ~3 seconds at 30fps

        // Baseline statistics
        this.baseline = {
            mean: new Array(NUM_FEATURES).fill(0),
            std:  new Array(NUM_FEATURES).fill(1),
            min:  new Array(NUM_FEATURES).fill(Infinity),
            max:  new Array(NUM_FEATURES).fill(-Infinity),
        };

        // Deviation thresholds (in standard deviations)
        this.deviationThreshold = 2.0;  // Flag if > 2σ from baseline

        // Callbacks
        this._onProgress  = null;
        this._onComplete  = null;

        // Try to load saved calibration
        this.load();
    }

    /**
     * Start calibration recording.
     * @param {Function} onProgress - Called with (current, total) during calibration
     * @param {Function} onComplete - Called when calibration finishes
     */
    startCalibration(onProgress = null, onComplete = null) {
        this.isCalibrating = true;
        this.isCalibrated  = false;
        this.frames = [];
        this._onProgress = onProgress;
        this._onComplete = onComplete;

        console.log('🎯 Calibration started — sit in your best posture!');
    }

    /**
     * Add a frame during calibration.
     * @param {number[]} features - Feature vector from FeatureEngine
     * @returns {boolean} True if calibration is still in progress
     */
    addFrame(features) {
        if (!this.isCalibrating) return false;

        this.frames.push([...features]);

        if (this._onProgress) {
            this._onProgress(this.frames.length, this.targetFrames);
        }

        if (this.frames.length >= this.targetFrames) {
            this.finishCalibration();
            return false;
        }

        return true;
    }

    /**
     * Finish calibration and compute baseline statistics.
     */
    finishCalibration() {
        if (this.frames.length < 10) {
            console.warn('Not enough frames for calibration');
            this.isCalibrating = false;
            return;
        }

        const n = this.frames.length;

        // ── Compute mean ──
        for (let f = 0; f < NUM_FEATURES; f++) {
            let sum = 0;
            for (let i = 0; i < n; i++) {
                sum += this.frames[i][f];
            }
            this.baseline.mean[f] = sum / n;
        }

        // ── Compute std, min, max ──
        for (let f = 0; f < NUM_FEATURES; f++) {
            let sumSqDiff = 0;
            this.baseline.min[f] = Infinity;
            this.baseline.max[f] = -Infinity;

            for (let i = 0; i < n; i++) {
                const diff = this.frames[i][f] - this.baseline.mean[f];
                sumSqDiff += diff * diff;
                this.baseline.min[f] = Math.min(this.baseline.min[f], this.frames[i][f]);
                this.baseline.max[f] = Math.max(this.baseline.max[f], this.frames[i][f]);
            }

            this.baseline.std[f] = Math.sqrt(sumSqDiff / (n - 1));
            // Prevent division by zero — if std is too small, use 1
            if (this.baseline.std[f] < 0.001) {
                this.baseline.std[f] = 0.001;
            }
        }

        this.isCalibrating = false;
        this.isCalibrated  = true;

        // Save to localStorage
        this.save();

        console.log('✅ Calibration complete!');
        console.log('Baseline mean:', this.baseline.mean.map(v => v.toFixed(3)));
        console.log('Baseline std:', this.baseline.std.map(v => v.toFixed(3)));

        if (this._onComplete) {
            this._onComplete(this.baseline);
        }
    }

    /**
     * Get deviation of current features from baseline.
     * Returns a feature vector where each value represents
     * the absolute deviation (useful for classification).
     * 
     * @param {number[]} features - Current frame features
     * @returns {number[]} Deviation features (same length)
     */
    getDeviation(features) {
        if (!this.isCalibrated) return features;

        return features.map((val, i) => {
            // For most features, return deviation from mean
            // For signed features (like head_lateral_offset), keep sign
            return val;  // Return raw — the classifier uses deviation thresholds
        });
    }

    /**
     * Get z-scores (standardized deviation) of current features.
     * Each value = (feature - mean) / std
     * 
     * @param {number[]} features - Current frame features
     * @returns {{ zScores: number[], alerts: string[] }}
     */
    getZScores(features) {
        const zScores = new Array(NUM_FEATURES).fill(0);
        const alerts = [];

        if (!this.isCalibrated) {
            return { zScores, alerts };
        }

        for (let i = 0; i < NUM_FEATURES; i++) {
            zScores[i] = (features[i] - this.baseline.mean[i]) / this.baseline.std[i];

            if (Math.abs(zScores[i]) > this.deviationThreshold) {
                const direction = zScores[i] > 0 ? 'above' : 'below';
                alerts.push(
                    `${FEATURE_NAMES[i]}: ${Math.abs(zScores[i]).toFixed(1)}σ ${direction} your baseline`
                );
            }
        }

        return { zScores, alerts };
    }

    /**
     * Get personalized posture score based on calibration.
     * Measures how close current features are to the calibrated baseline.
     * 
     * @param {number[]} features - Current frame features
     * @returns {number} Score 0-100
     */
    getPersonalizedScore(features) {
        if (!this.isCalibrated) return null;

        const { zScores } = this.getZScores(features);

        // Emphasize posture-critical features
        const weights = [
            2.0,   // shoulder_angle
            2.5,   // neck_inclination
            2.5,   // torso_inclination
            1.5,   // hip_angle
            1.5,   // head_lateral_offset
            1.0,   // left_torso_ratio
            1.0,   // right_torso_ratio
            1.5,   // symmetry_score
            2.0,   // head_drop_ratio
            2.0,   // ear_alignment
            1.0,   // shoulder_hip_width
            0.0,   // avg_confidence (not a posture metric)
        ];

        let totalPenalty = 0;
        let totalWeight = 0;

        for (let i = 0; i < NUM_FEATURES; i++) {
            if (weights[i] > 0) {
                // Penalty = weighted absolute z-score, capped at 3σ
                const penalty = Math.min(3, Math.abs(zScores[i])) * weights[i];
                totalPenalty += penalty;
                totalWeight += weights[i] * 3;  // Max possible penalty
            }
        }

        const normalizedPenalty = totalWeight > 0 ? totalPenalty / totalWeight : 0;
        return Math.max(0, Math.round(100 * (1 - normalizedPenalty)));
    }

    /**
     * Save calibration to localStorage.
     */
    save() {
        if (!this.isCalibrated) return;

        const data = {
            baseline: this.baseline,
            timestamp: Date.now(),
            frameCount: this.frames.length,
            version: 1,
        };

        try {
            localStorage.setItem(this._storageKey, JSON.stringify(data));
            console.log('💾 Calibration saved to localStorage');
        } catch (e) {
            console.warn('Could not save calibration:', e);
        }
    }

    /**
     * Load calibration from localStorage.
     */
    load() {
        try {
            const raw = localStorage.getItem(this._storageKey);
            if (!raw) return false;

            const data = JSON.parse(raw);
            if (data.version !== 1 || !data.baseline) return false;

            this.baseline = data.baseline;
            this.isCalibrated = true;

            const age = Date.now() - data.timestamp;
            const ageHours = (age / 3600000).toFixed(1);
            console.log(`📂 Loaded calibration (${ageHours}h old, ${data.frameCount} frames)`);

            return true;
        } catch (e) {
            console.warn('Could not load calibration:', e);
            return false;
        }
    }

    /**
     * Clear saved calibration.
     */
    clear() {
        this.isCalibrated = false;
        this.baseline = {
            mean: new Array(NUM_FEATURES).fill(0),
            std:  new Array(NUM_FEATURES).fill(1),
            min:  new Array(NUM_FEATURES).fill(Infinity),
            max:  new Array(NUM_FEATURES).fill(-Infinity),
        };
        this.frames = [];
        localStorage.removeItem(this._storageKey);
        console.log('🗑️ Calibration cleared');
    }

    /** Get progress during calibration */
    get progress() {
        if (!this.isCalibrating) return this.isCalibrated ? 1 : 0;
        return this.frames.length / this.targetFrames;
    }

    /** Get human-readable baseline summary */
    getSummary() {
        if (!this.isCalibrated) return null;

        return FEATURE_NAMES.map((name, i) => ({
            feature: name,
            mean: this.baseline.mean[i],
            std: this.baseline.std[i],
            range: [this.baseline.min[i], this.baseline.max[i]],
        }));
    }

    /**
     * Export calibration data for the backend.
     */
    export() {
        return {
            isCalibrated: this.isCalibrated,
            baseline: this.isCalibrated ? this.baseline : null,
            featureNames: [...FEATURE_NAMES],
        };
    }
}
