/* =====================================================
   TemporalBuffer — Sliding Window + Temporal Classification
   
   Most posture projects ignore TIME. This module treats
   posture as a time-series problem:
   
   - Maintains a sliding window of N frames of features
   - Provides the window to an LSTM or temporal classifier
   - Implements a rule-based temporal classifier as fallback
   - Smooths predictions using exponential moving average
   
   Why temporal modeling matters:
   ─────────────────────────────
   A single frame might show "bad posture" because the person
   is reaching for something. But 3 seconds of persistent
   slouching IS bad posture. Temporal modeling captures this.
   ===================================================== */

import { NUM_FEATURES, FEATURE_NAMES } from './featureEngine.js';

/**
 * Posture class labels — consistent between frontend and backend
 */
export const POSTURE_CLASSES = {
    GOOD:          0,
    FORWARD_LEAN:  1,
    LEFT_LEAN:     2,
    RIGHT_LEAN:    3,
    NECK_STRAIN:   4,
};

export const POSTURE_LABELS = [
    'Good Posture',
    'Forward Lean',
    'Left Lean',
    'Right Lean',
    'Neck Strain',
];

export const POSTURE_ICONS = ['✅', '😔', '↙️', '↘️', '🦒'];
export const POSTURE_COLORS = ['#22c55e', '#f59e0b', '#3b82f6', '#8b5cf6', '#ef4444'];

export class TemporalBuffer {
    /**
     * @param {number} windowSize - Number of frames in sliding window (default 30 ≈ 1 second at 30fps)
     * @param {number} smoothingFactor - EMA smoothing factor for predictions (0-1)
     */
    constructor(windowSize = 30, smoothingFactor = 0.15) {
        this.windowSize = windowSize;
        this.smoothingFactor = smoothingFactor;

        // Circular buffer for feature vectors
        this.buffer = [];
        this.maxSize = windowSize * 2;  // Keep extra for statistics

        // Prediction smoothing
        this.smoothedProbs = new Array(POSTURE_LABELS.length).fill(1 / POSTURE_LABELS.length);
        this.lastPrediction = null;

        // Temporal statistics
        this.stats = {
            mean: new Array(NUM_FEATURES).fill(0),
            std:  new Array(NUM_FEATURES).fill(1),
            updateCount: 0,
        };

        // ═══════════════════════════════════════════════════
        // CALIBRATED THRESHOLDS
        // These thresholds define where "good" ends and "bad" begins.
        // Based on ergonomic research:
        //   - A normal upright person has ~5-12° shoulder tilt
        //   - Neck naturally sits at 8-18° from vertical
        //   - Torso at 5-15° from vertical when seated
        //   - Head offset within 0.15 of shoulder width is normal
        // 
        // CRITICAL: These must be realistic to allow score=100
        // for truly good posture. Previous values (12°, 20°, 18°)
        // were too tight and penalized natural micro-movements.
        // ═══════════════════════════════════════════════════
        this.thresholds = {
            shoulderAngleMax:     15,    // degrees — up from 12
            neckInclinationMax:   25,    // degrees — up from 20
            torsoInclinationMax:  22,    // degrees — up from 18
            headOffsetMax:        0.35,  // normalized — up from 0.3
            headDropMax:          0.18,  // normalized — up from 0.15
            earAlignmentMax:      30,    // degrees — up from 25
            symmetryMin:          0.65,  // 0-1 — down from 0.7
        };
    }

    /**
     * Push a new frame's feature vector into the buffer.
     * @param {number[]} features - Feature vector of length NUM_FEATURES
     */
    push(features) {
        if (features.length !== NUM_FEATURES) {
            console.warn(`Expected ${NUM_FEATURES} features, got ${features.length}`);
            return;
        }

        this.buffer.push([...features]);

        // Trim to max size
        if (this.buffer.length > this.maxSize) {
            this.buffer.shift();
        }

        // Update running statistics (Welford's online algorithm)
        this._updateStats(features);
    }

    /**
     * Get the current sliding window.
     * @returns {number[][]} Array of feature vectors (most recent last)
     */
    getWindow() {
        const start = Math.max(0, this.buffer.length - this.windowSize);
        return this.buffer.slice(start);
    }

    /** Is the window full? */
    isFull() {
        return this.buffer.length >= this.windowSize;
    }

    /** Current buffer size */
    get size() {
        return this.buffer.length;
    }

    /**
     * Run the rule-based temporal classifier.
     * This is used when no ML backend is available.
     * 
     * Strategy:
     * 1. Compute mean features over the window (temporal averaging)
     * 2. Apply threshold-based rules on the averaged features
     * 3. Smooth predictions with EMA
     * 
     * @param {Object|null} calibration - Calibration baseline (if available)
     * @returns {{ classIndex: number, label: string, confidence: number, probabilities: number[], issues: string[] }}
     */
    classifyRuleBased(calibration = null) {
        const window = this.getWindow();
        if (window.length === 0) {
            return this._defaultPrediction();
        }

        // ── Step 1: Temporal averaging over window ──
        const avgFeatures = this._windowMean(window);

        // ── Step 2: Compute deviations ──
        let features = avgFeatures;
        if (calibration && calibration.isCalibrated) {
            features = calibration.getDeviation(avgFeatures);
        }

        // ── Step 3: Rule-based classification ──
        // IMPROVED: Start with high base and subtract only for
        // meaningful deviations above thresholds.
        const scores = new Array(POSTURE_LABELS.length).fill(0);
        const issues = [];

        // Good posture starts with a stronger base
        scores[POSTURE_CLASSES.GOOD] = 0.7;

        // ── Check Forward Lean ──
        const neckInc = features[1];    // neck_inclination
        const torsoInc = features[2];   // torso_inclination
        const headDrop = features[8];   // head_drop_ratio
        const earAlign = features[9];   // ear_alignment

        if (neckInc > this.thresholds.neckInclinationMax) {
            const severity = Math.min(1, (neckInc - this.thresholds.neckInclinationMax) / 25);
            scores[POSTURE_CLASSES.FORWARD_LEAN] += 0.4 * severity;
            scores[POSTURE_CLASSES.GOOD] -= 0.35 * severity;
            issues.push(`Neck leaning forward (${neckInc.toFixed(1)}°)`);
        }

        if (torsoInc > this.thresholds.torsoInclinationMax) {
            const severity = Math.min(1, (torsoInc - this.thresholds.torsoInclinationMax) / 20);
            scores[POSTURE_CLASSES.FORWARD_LEAN] += 0.35 * severity;
            scores[POSTURE_CLASSES.GOOD] -= 0.25 * severity;
            issues.push(`Torso leaning (${torsoInc.toFixed(1)}°)`);
        }

        if (headDrop > this.thresholds.headDropMax) {
            const severity = Math.min(1, (headDrop - this.thresholds.headDropMax) / 0.25);
            scores[POSTURE_CLASSES.FORWARD_LEAN] += 0.2 * severity;
            issues.push('Head dropping forward');
        }

        // ── Check Neck Strain ──
        if (earAlign > this.thresholds.earAlignmentMax) {
            const severity = Math.min(1, (earAlign - this.thresholds.earAlignmentMax) / 25);
            scores[POSTURE_CLASSES.NECK_STRAIN] += 0.5 * severity;
            scores[POSTURE_CLASSES.GOOD] -= 0.3 * severity;
            issues.push(`Neck strain detected (${earAlign.toFixed(1)}°)`);
        }

        // ── Check Left/Right Lean ──
        const headOffset = features[4];  // head_lateral_offset (signed)
        const shoulderAng = features[0]; // shoulder_angle
        const symmetry = features[7];    // symmetry_score

        if (Math.abs(headOffset) > this.thresholds.headOffsetMax || shoulderAng > this.thresholds.shoulderAngleMax) {
            const severity = Math.min(1, Math.max(
                (Math.abs(headOffset) - this.thresholds.headOffsetMax) / 0.3,
                (shoulderAng - this.thresholds.shoulderAngleMax) / 15
            ));
            if (severity > 0) {
                if (headOffset < -0.1) {
                    scores[POSTURE_CLASSES.LEFT_LEAN] += 0.4 * severity;
                } else if (headOffset > 0.1) {
                    scores[POSTURE_CLASSES.RIGHT_LEAN] += 0.4 * severity;
                } else if (shoulderAng > this.thresholds.shoulderAngleMax) {
                    // Shoulders uneven but head centered — check shoulder height diff
                    scores[POSTURE_CLASSES.LEFT_LEAN] += 0.2 * severity;
                    scores[POSTURE_CLASSES.RIGHT_LEAN] += 0.2 * severity;
                }
                scores[POSTURE_CLASSES.GOOD] -= 0.2 * severity;
                issues.push(`Shoulders uneven (${shoulderAng.toFixed(1)}°)`);
            }
        }

        if (symmetry < this.thresholds.symmetryMin) {
            const severity = Math.min(1, (this.thresholds.symmetryMin - symmetry) / 0.35);
            scores[POSTURE_CLASSES.GOOD] -= 0.1 * severity;
        }

        // ── Step 4: Normalize to probabilities ──
        // Softmax-like normalization
        const maxScore = Math.max(...scores);
        const expScores = scores.map(s => Math.exp(s - maxScore));
        const sumExp = expScores.reduce((a, b) => a + b, 0);
        const probabilities = expScores.map(s => s / sumExp);

        // ── Step 5: Smooth with EMA ──
        for (let i = 0; i < probabilities.length; i++) {
            this.smoothedProbs[i] += this.smoothingFactor * (probabilities[i] - this.smoothedProbs[i]);
        }

        // ── Step 6: Pick winner ──
        let maxProb = -1;
        let classIndex = 0;
        for (let i = 0; i < this.smoothedProbs.length; i++) {
            if (this.smoothedProbs[i] > maxProb) {
                maxProb = this.smoothedProbs[i];
                classIndex = i;
            }
        }

        return {
            classIndex,
            label: POSTURE_LABELS[classIndex],
            icon: POSTURE_ICONS[classIndex],
            color: POSTURE_COLORS[classIndex],
            confidence: maxProb,
            probabilities: [...this.smoothedProbs],
            issues,
            score: this._computePostureScore(avgFeatures, calibration),
        };
    }

    /**
     * Compute a 0-100 posture score from features.
     * 
     * IMPROVED SCORING:
     * - Only penalizes deviations ABOVE the thresholds
     * - Features below threshold → zero penalty → score reaches 100
     * - Smoother penalty curves prevent harsh drops
     * - Each penalty channel has a realistic max contribution
     */
    _computePostureScore(features, calibration) {
        let penalties = 0;

        // Shoulder penalty (max 15 points)
        // Under threshold → NO penalty at all
        const shoulderExcess = Math.max(0, features[0] - this.thresholds.shoulderAngleMax);
        penalties += Math.min(15, shoulderExcess * 1.2);

        // Neck penalty (max 25 points)
        const neckExcess = Math.max(0, features[1] - this.thresholds.neckInclinationMax);
        penalties += Math.min(25, neckExcess * 1.2);

        // Torso penalty (max 25 points)
        const torsoExcess = Math.max(0, features[2] - this.thresholds.torsoInclinationMax);
        penalties += Math.min(25, torsoExcess * 1.5);

        // Head offset penalty (max 15 points)
        const headExcess = Math.max(0, Math.abs(features[4]) - this.thresholds.headOffsetMax);
        penalties += Math.min(15, headExcess * 25);

        // Ear alignment penalty (max 10 points)
        const earExcess = Math.max(0, features[9] - this.thresholds.earAlignmentMax);
        penalties += Math.min(10, earExcess * 0.8);

        // Symmetry penalty (max 10 points)
        const symDeficit = Math.max(0, this.thresholds.symmetryMin - features[7]);
        penalties += Math.min(10, symDeficit * 30);

        // Final score: 100 minus penalties, floored at 0
        return Math.max(0, Math.round(100 - Math.min(penalties, 100)));
    }

    /**
     * Compute mean feature vector over the window.
     */
    _windowMean(window) {
        const mean = new Array(NUM_FEATURES).fill(0);
        for (const frame of window) {
            for (let i = 0; i < NUM_FEATURES; i++) {
                mean[i] += frame[i];
            }
        }
        for (let i = 0; i < NUM_FEATURES; i++) {
            mean[i] /= window.length;
        }
        return mean;
    }

    /**
     * Update running mean/std using Welford's online algorithm.
     */
    _updateStats(features) {
        this.stats.updateCount++;
        const n = this.stats.updateCount;

        for (let i = 0; i < NUM_FEATURES; i++) {
            const delta = features[i] - this.stats.mean[i];
            this.stats.mean[i] += delta / n;
            const delta2 = features[i] - this.stats.mean[i];
            this.stats.std[i] += delta * delta2;
        }
    }

    /**
     * Get computed standard deviations.
     */
    getStdDevs() {
        if (this.stats.updateCount < 2) return new Array(NUM_FEATURES).fill(1);
        return this.stats.std.map(v => Math.sqrt(v / (this.stats.updateCount - 1)));
    }

    /** Default prediction when buffer is empty */
    _defaultPrediction() {
        return {
            classIndex: POSTURE_CLASSES.GOOD,
            label: 'Waiting...',
            icon: '⏳',
            color: '#64748b',
            confidence: 0,
            probabilities: new Array(POSTURE_LABELS.length).fill(1 / POSTURE_LABELS.length),
            issues: [],
            score: 0,
        };
    }

    /** Clear the buffer */
    reset() {
        this.buffer = [];
        this.smoothedProbs = new Array(POSTURE_LABELS.length).fill(1 / POSTURE_LABELS.length);
        this.lastPrediction = null;
        this.stats = { mean: new Array(NUM_FEATURES).fill(0), std: new Array(NUM_FEATURES).fill(1), updateCount: 0 };
    }

    /**
     * Export the current buffer as a JSON-serializable object
     * (for sending to backend or saving)
     */
    exportWindow() {
        return {
            window: this.getWindow(),
            windowSize: this.windowSize,
            frameCount: this.buffer.length,
            featureNames: [...FEATURE_NAMES],
        };
    }
}
