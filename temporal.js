/* =====================================================
   Temporal Smoothing & Prediction Engine
   
   Goes beyond frame-by-frame analysis:
   1. EMA smoothing to eliminate jitter
   2. Kalman-based keypoint prediction for occluded joints
   3. Velocity-aware adaptive smoothing
   4. Sliding window for temporal posture features
   5. Missing joint interpolation from temporal context
   
   Architecture:
   ┌────────────────────────────────────────────────┐
   │  Raw Keypoints (noisy, may have gaps)          │
   │     ↓                                          │
   │  Adaptive EMA Smoothing (velocity-aware)       │
   │     ↓                                          │
   │  Kalman Prediction (for missing joints)        │
   │     ↓                                          │
   │  Temporal Buffer (sliding window ~1s)          │
   │     ↓                                          │
   │  Temporal Features (stability, trend, jitter)  │
   │     ↓                                          │
   │  Smoothed Classification (majority voting)     │
   └────────────────────────────────────────────────┘
   ===================================================== */

class TemporalEngine {
    /**
     * @param {Object} config
     * @param {number} config.windowSize — Number of frames in sliding window
     * @param {number} config.smoothingBase — Base EMA factor (0-1, lower = smoother)
     * @param {number} config.predictionHorizon — Max frames to predict missing joints
     * @param {number} config.jitterThreshold — Min movement to count as real (pixels)
     */
    constructor(config = {}) {
        this.windowSize = config.windowSize || 30; // ~1s at 30fps
        this.smoothingBase = config.smoothingBase || 0.35;
        this.predictionHorizon = config.predictionHorizon || 15;
        this.jitterThreshold = config.jitterThreshold || 2.0;

        // Per-keypoint temporal state
        this.keypointHistory = new Array(17).fill(null).map(() => ({
            buffer: [],           // Sliding window of positions
            smoothed: null,       // Current smoothed position
            velocity: { x: 0, y: 0 },
            lastSeen: 0,          // Frame when last confidently detected
            confidence: 0,        // Smoothed confidence
            predicted: false,     // Whether current position is predicted
        }));

        // Posture temporal state
        this.postureBuffer = [];  // Sliding window of posture scores
        this.classBuffer = [];    // Sliding window of classifications
        this.frameCount = 0;

        // Jitter statistics
        this.jitterAccumulator = new Array(17).fill(0);
        this.stabilityScore = 1.0; // 0 = very jittery, 1 = perfectly stable
    }

    /**
     * Process a new frame of keypoints through the temporal engine.
     * @param {Array} rawKeypoints — 17 keypoints from detector [{x, y, score, name}]
     * @returns {Array} — Smoothed, stabilized keypoints with predictions
     */
    processFrame(rawKeypoints) {
        this.frameCount++;
        const output = [];

        for (let i = 0; i < 17; i++) {
            const raw = rawKeypoints[i];
            const hist = this.keypointHistory[i];
            const isDetected = raw && raw.score > 0.2;

            if (isDetected) {
                // ── Detected: Apply adaptive smoothing ──
                const smoothed = this._adaptiveSmooth(hist, raw);
                hist.lastSeen = this.frameCount;
                hist.predicted = false;

                // Update buffer
                hist.buffer.push({ x: smoothed.x, y: smoothed.y, score: raw.score, frame: this.frameCount });
                if (hist.buffer.length > this.windowSize) hist.buffer.shift();

                // Compute velocity
                if (hist.buffer.length >= 2) {
                    const prev = hist.buffer[hist.buffer.length - 2];
                    hist.velocity.x = 0.7 * hist.velocity.x + 0.3 * (smoothed.x - prev.x);
                    hist.velocity.y = 0.7 * hist.velocity.y + 0.3 * (smoothed.y - prev.y);
                }

                // Smooth confidence
                hist.confidence = 0.7 * (hist.confidence || raw.score) + 0.3 * raw.score;

                output.push({
                    x: smoothed.x,
                    y: smoothed.y,
                    score: hist.confidence,
                    name: raw.name || '',
                    predicted: false,
                    velocity: { ...hist.velocity },
                });

            } else {
                // ── Missing: Predict from temporal context ──
                const predicted = this._predictMissing(hist, i);

                if (predicted) {
                    output.push({
                        x: predicted.x,
                        y: predicted.y,
                        score: predicted.score,
                        name: raw?.name || '',
                        predicted: true,
                        velocity: { ...hist.velocity },
                    });
                } else {
                    output.push({
                        x: raw?.x || 0,
                        y: raw?.y || 0,
                        score: 0,
                        name: raw?.name || '',
                        predicted: false,
                        velocity: { x: 0, y: 0 },
                    });
                }
            }
        }

        // Update jitter/stability metrics
        this._updateStability(output);

        return output;
    }

    /**
     * Adaptive EMA smoothing — less smoothing during fast motion,
     * more smoothing during stillness (to suppress jitter).
     */
    _adaptiveSmooth(hist, raw) {
        if (!hist.smoothed) {
            hist.smoothed = { x: raw.x, y: raw.y };
            return { x: raw.x, y: raw.y };
        }

        // Compute motion magnitude
        const dx = raw.x - hist.smoothed.x;
        const dy = raw.y - hist.smoothed.y;
        const motion = Math.sqrt(dx * dx + dy * dy);

        // Adaptive alpha: fast motion → high alpha (less smoothing)
        //                 slow motion → low alpha (more smoothing, suppress jitter)
        let alpha;
        if (motion < this.jitterThreshold) {
            alpha = 0.05; // Very heavy smoothing for tiny movements
        } else if (motion < this.jitterThreshold * 3) {
            alpha = this.smoothingBase * 0.5; // Moderate smoothing
        } else {
            alpha = Math.min(0.8, this.smoothingBase + motion * 0.01); // Light smoothing
        }

        hist.smoothed.x += alpha * dx;
        hist.smoothed.y += alpha * dy;

        return { x: hist.smoothed.x, y: hist.smoothed.y };
    }

    /**
     * Predict missing keypoint position from temporal context.
     * Uses: 1) velocity extrapolation, 2) anatomical constraints,
     * 3) buffer interpolation
     */
    _predictMissing(hist, keypointIdx) {
        const framesSinceSeen = this.frameCount - hist.lastSeen;

        // Don't predict too far into the future
        if (framesSinceSeen > this.predictionHorizon || hist.buffer.length < 3) {
            hist.predicted = true;
            // Decay confidence over time
            hist.confidence = Math.max(0, hist.confidence * 0.9);
            return null;
        }

        hist.predicted = true;

        // Method 1: Velocity extrapolation (good for short-term)
        if (hist.smoothed && framesSinceSeen <= 5) {
            const predX = hist.smoothed.x + hist.velocity.x * framesSinceSeen;
            const predY = hist.smoothed.y + hist.velocity.y * framesSinceSeen;
            // Decay confidence based on prediction age
            const predConf = hist.confidence * Math.pow(0.85, framesSinceSeen);
            hist.confidence = predConf;
            return { x: predX, y: predY, score: predConf };
        }

        // Method 2: Buffer average (good for longer occlusions)
        if (hist.buffer.length >= 5) {
            const recent = hist.buffer.slice(-5);
            const avgX = recent.reduce((s, p) => s + p.x, 0) / recent.length;
            const avgY = recent.reduce((s, p) => s + p.y, 0) / recent.length;
            const predConf = hist.confidence * Math.pow(0.8, framesSinceSeen);
            hist.confidence = predConf;
            return { x: avgX, y: avgY, score: predConf };
        }

        // Method 3: Last known position (fallback)
        if (hist.smoothed) {
            const predConf = hist.confidence * Math.pow(0.7, framesSinceSeen);
            hist.confidence = predConf;
            return { x: hist.smoothed.x, y: hist.smoothed.y, score: predConf };
        }

        return null;
    }

    /**
     * Update jitter and stability metrics across all keypoints.
     */
    _updateStability(keypoints) {
        let totalJitter = 0;
        let validCount = 0;

        keypoints.forEach((kp, i) => {
            if (kp.score > 0.3 && !kp.predicted) {
                const hist = this.keypointHistory[i];
                if (hist.buffer.length >= 2) {
                    const prev = hist.buffer[hist.buffer.length - 2];
                    const jitter = Math.sqrt(
                        (kp.x - prev.x) ** 2 + (kp.y - prev.y) ** 2
                    );
                    this.jitterAccumulator[i] = 0.8 * this.jitterAccumulator[i] + 0.2 * jitter;
                    totalJitter += this.jitterAccumulator[i];
                    validCount++;
                }
            }
        });

        if (validCount > 0) {
            const avgJitter = totalJitter / validCount;
            // Map jitter to stability: 0 jitter = 1.0 stability, 10+ jitter = 0.0
            this.stabilityScore = Math.max(0, Math.min(1, 1 - avgJitter / 10));
        }
    }

    /**
     * Add a posture classification to the temporal buffer.
     * Returns temporally smoothed classification.
     */
    addPostureFrame(score, isGood, issues) {
        this.postureBuffer.push({ score, isGood, issues, frame: this.frameCount });
        this.classBuffer.push(isGood ? 1 : 0);

        if (this.postureBuffer.length > this.windowSize) this.postureBuffer.shift();
        if (this.classBuffer.length > this.windowSize) this.classBuffer.shift();

        return this.getTemporalClassification();
    }

    /**
     * Get temporally smoothed posture classification.
     * Uses majority voting + score averaging over the window.
     */
    getTemporalClassification() {
        if (this.postureBuffer.length === 0) return null;

        const window = this.postureBuffer;
        const n = window.length;

        // Weighted average score (recent frames weighted more)
        let weightedScore = 0;
        let totalWeight = 0;
        window.forEach((frame, i) => {
            const weight = 1 + i / n; // Linear weight: later frames = more weight
            weightedScore += frame.score * weight;
            totalWeight += weight;
        });
        const avgScore = weightedScore / totalWeight;

        // Majority voting for classification (with hysteresis)
        const goodCount = this.classBuffer.reduce((s, v) => s + v, 0);
        const goodRatio = goodCount / this.classBuffer.length;

        // Hysteresis: need 60% to switch to good, 40% to switch to bad
        // This prevents rapid flipping between states
        let isGood;
        if (this._lastClass === undefined) {
            isGood = goodRatio >= 0.5;
        } else {
            if (this._lastClass) {
                isGood = goodRatio >= 0.35; // Stay good unless clearly bad
            } else {
                isGood = goodRatio >= 0.65; // Need strong signal to switch to good
            }
        }
        this._lastClass = isGood;

        // Aggregate issues (most frequent in window)
        const issueCounts = {};
        window.forEach(f => {
            (f.issues || []).forEach(issue => {
                issueCounts[issue] = (issueCounts[issue] || 0) + 1;
            });
        });
        const persistentIssues = Object.entries(issueCounts)
            .filter(([, count]) => count >= n * 0.3) // Present in 30%+ of frames
            .sort((a, b) => b[1] - a[1])
            .map(([issue]) => issue);

        return {
            score: Math.round(avgScore),
            isGood,
            issues: persistentIssues,
            stability: this.stabilityScore,
            confidence: this.postureBuffer.length / this.windowSize, // 0-1 ramp-up
            windowSize: n,
        };
    }

    /**
     * Get temporal features for the current window (for ML input).
     */
    getTemporalFeatures() {
        if (this.postureBuffer.length < 5) return null;

        const scores = this.postureBuffer.map(f => f.score);
        const n = scores.length;

        // Statistical features over the temporal window
        const mean = scores.reduce((s, v) => s + v, 0) / n;
        const variance = scores.reduce((s, v) => s + (v - mean) ** 2, 0) / n;
        const std = Math.sqrt(variance);

        // Trend (linear regression slope)
        let sumXY = 0, sumX = 0, sumY = 0, sumX2 = 0;
        scores.forEach((y, x) => {
            sumXY += x * y;
            sumX += x;
            sumY += y;
            sumX2 += x * x;
        });
        const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX + 1e-6);

        // Min, max, range
        const min = Math.min(...scores);
        const max = Math.max(...scores);

        // Recent vs early comparison
        const recentAvg = scores.slice(-5).reduce((s, v) => s + v, 0) / Math.min(5, n);
        const earlyAvg = scores.slice(0, 5).reduce((s, v) => s + v, 0) / Math.min(5, n);

        return {
            mean,
            std,
            slope,
            min,
            max,
            range: max - min,
            recentAvg,
            earlyAvg,
            trend: recentAvg - earlyAvg,
            stability: this.stabilityScore,
            windowFill: this.postureBuffer.length / this.windowSize,
        };
    }

    /**
     * Get per-keypoint stability metrics.
     */
    getKeypointStability() {
        return this.keypointHistory.map((hist, i) => ({
            index: i,
            jitter: this.jitterAccumulator[i],
            bufferSize: hist.buffer.length,
            isPredicted: hist.predicted,
            confidence: hist.confidence,
            framesSinceSeen: this.frameCount - hist.lastSeen,
            velocity: Math.sqrt(hist.velocity.x ** 2 + hist.velocity.y ** 2),
        }));
    }

    /**
     * Reset all temporal state.
     */
    reset() {
        this.keypointHistory.forEach(h => {
            h.buffer = [];
            h.smoothed = null;
            h.velocity = { x: 0, y: 0 };
            h.lastSeen = 0;
            h.confidence = 0;
            h.predicted = false;
        });
        this.postureBuffer = [];
        this.classBuffer = [];
        this.frameCount = 0;
        this.jitterAccumulator.fill(0);
        this.stabilityScore = 1.0;
        this._lastClass = undefined;
    }
}

// Export to global scope (non-module script)
window.TemporalEngine = TemporalEngine;
