/* =====================================================
   ByteTrack-Inspired Multi-Person Tracker
   
   Production-grade tracking system that:
   1. Assigns consistent IDs across frames
   2. Handles occlusions via Kalman prediction
   3. Re-identifies people after leaving frame
   4. Uses IoU + appearance matching (keypoint geometry)
   
   Architecture:
   ┌──────────────────────────────────────────────┐
   │  New Detections (from MoveNet)               │
   │     ↓                                        │
   │  Hungarian Assignment (IoU matrix)           │
   │     ↓                                        │
   │  High-confidence matches → Update tracks     │
   │  Unmatched detections  → Birth new tracks    │
   │  Unmatched tracks      → Kalman predict      │
   │     ↓                                        │
   │  Re-ID check (appearance similarity)         │
   │     ↓                                        │
   │  Dead tracks → Archive for re-identification │
   └──────────────────────────────────────────────┘
   ===================================================== */

// ─── Kalman Filter (2D point tracker) ──────────────────
class KalmanPoint {
    constructor(x, y) {
        // State: [x, y, vx, vy]
        this.state = [x, y, 0, 0];
        // Uncertainty covariance (diagonal)
        this.P = [100, 100, 50, 50];
        // Process noise
        this.Q = [1, 1, 0.5, 0.5];
        // Measurement noise
        this.R = [4, 4];
    }

    predict(dt = 1) {
        // State prediction: x += vx*dt, y += vy*dt
        this.state[0] += this.state[2] * dt;
        this.state[1] += this.state[3] * dt;
        // Covariance prediction
        this.P[0] += this.Q[0] + this.P[2] * dt * dt;
        this.P[1] += this.Q[1] + this.P[3] * dt * dt;
        this.P[2] += this.Q[2];
        this.P[3] += this.Q[3];
        return { x: this.state[0], y: this.state[1] };
    }

    update(x, y) {
        // Kalman gain
        const Kx = this.P[0] / (this.P[0] + this.R[0]);
        const Ky = this.P[1] / (this.P[1] + this.R[1]);
        // Velocity update from innovation
        const dx = x - this.state[0];
        const dy = y - this.state[1];
        this.state[2] = 0.7 * this.state[2] + 0.3 * dx; // Smooth velocity
        this.state[3] = 0.7 * this.state[3] + 0.3 * dy;
        // Position update
        this.state[0] += Kx * dx;
        this.state[1] += Ky * dy;
        // Covariance update
        this.P[0] *= (1 - Kx);
        this.P[1] *= (1 - Ky);
        return { x: this.state[0], y: this.state[1] };
    }

    get position() {
        return { x: this.state[0], y: this.state[1] };
    }

    get velocity() {
        return { vx: this.state[2], vy: this.state[3] };
    }
}


// ─── Bounding Box Kalman Filter ────────────────────────
class KalmanBBox {
    constructor(bbox) {
        // State: [cx, cy, w, h, vcx, vcy, vw, vh]
        this.cx = bbox.x + bbox.width / 2;
        this.cy = bbox.y + bbox.height / 2;
        this.w = bbox.width;
        this.h = bbox.height;
        this.vcx = 0; this.vcy = 0; this.vw = 0; this.vh = 0;
        this.age = 0;
    }

    predict() {
        this.cx += this.vcx;
        this.cy += this.vcy;
        this.w += this.vw;
        this.h += this.vh;
        this.w = Math.max(10, this.w);
        this.h = Math.max(10, this.h);
        this.age++;
        return this.toBBox();
    }

    update(bbox) {
        const ncx = bbox.x + bbox.width / 2;
        const ncy = bbox.y + bbox.height / 2;
        const alpha = 0.4; // Smoothing
        this.vcx = alpha * (ncx - this.cx) + (1 - alpha) * this.vcx;
        this.vcy = alpha * (ncy - this.cy) + (1 - alpha) * this.vcy;
        this.vw = alpha * (bbox.width - this.w) + (1 - alpha) * this.vw;
        this.vh = alpha * (bbox.height - this.h) + (1 - alpha) * this.vh;
        this.cx = 0.7 * ncx + 0.3 * this.cx;
        this.cy = 0.7 * ncy + 0.3 * this.cy;
        this.w = 0.7 * bbox.width + 0.3 * this.w;
        this.h = 0.7 * bbox.height + 0.3 * this.h;
        this.age = 0;
    }

    toBBox() {
        return {
            x: this.cx - this.w / 2,
            y: this.cy - this.h / 2,
            width: this.w,
            height: this.h,
        };
    }
}


// ─── Appearance Descriptor (keypoint geometry) ─────────
function computeAppearance(keypoints) {
    // Encode body proportions as a descriptor for re-identification
    // This is scale/translation invariant
    const valid = keypoints.filter(k => k.score > 0.3);
    if (valid.length < 4) return null;

    // Centroid
    const cx = valid.reduce((s, k) => s + k.x, 0) / valid.length;
    const cy = valid.reduce((s, k) => s + k.y, 0) / valid.length;

    // Scale (max distance from centroid)
    const scale = Math.max(1, ...valid.map(k =>
        Math.sqrt((k.x - cx) ** 2 + (k.y - cy) ** 2)
    ));

    // Normalized relative positions (scale-invariant descriptor)
    const descriptor = new Array(17 * 2).fill(0);
    keypoints.forEach((kp, i) => {
        if (kp.score > 0.3) {
            descriptor[i * 2] = (kp.x - cx) / scale;
            descriptor[i * 2 + 1] = (kp.y - cy) / scale;
        }
    });

    // Add limb ratios for more robust re-id
    const ratios = [];
    const pairs = [[5, 7], [7, 9], [6, 8], [8, 10], [5, 11], [6, 12]]; // arm & torso
    pairs.forEach(([a, b]) => {
        if (keypoints[a].score > 0.3 && keypoints[b].score > 0.3) {
            const d = Math.sqrt(
                (keypoints[a].x - keypoints[b].x) ** 2 +
                (keypoints[a].y - keypoints[b].y) ** 2
            );
            ratios.push(d / scale);
        } else {
            ratios.push(0);
        }
    });

    return { descriptor, ratios, valid: valid.length };
}

function appearanceSimilarity(a, b) {
    if (!a || !b) return 0;
    // Cosine similarity of descriptors
    let dot = 0, magA = 0, magB = 0;
    for (let i = 0; i < a.descriptor.length; i++) {
        dot += a.descriptor[i] * b.descriptor[i];
        magA += a.descriptor[i] ** 2;
        magB += b.descriptor[i] ** 2;
    }
    const cosSim = (magA > 0 && magB > 0) ? dot / (Math.sqrt(magA) * Math.sqrt(magB)) : 0;

    // Ratio similarity
    let ratioSim = 0, rCount = 0;
    for (let i = 0; i < a.ratios.length; i++) {
        if (a.ratios[i] > 0 && b.ratios[i] > 0) {
            ratioSim += 1 - Math.abs(a.ratios[i] - b.ratios[i]);
            rCount++;
        }
    }
    ratioSim = rCount > 0 ? ratioSim / rCount : 0;

    return 0.6 * Math.max(0, cosSim) + 0.4 * ratioSim;
}


// ─── IoU Computation ───────────────────────────────────
function computeIoU(boxA, boxB) {
    const xA = Math.max(boxA.x, boxB.x);
    const yA = Math.max(boxA.y, boxB.y);
    const xB = Math.min(boxA.x + boxA.width, boxB.x + boxB.width);
    const yB = Math.min(boxA.y + boxA.height, boxB.y + boxB.height);
    const inter = Math.max(0, xB - xA) * Math.max(0, yB - yA);
    const areaA = boxA.width * boxA.height;
    const areaB = boxB.width * boxB.height;
    return inter / (areaA + areaB - inter + 1e-6);
}


// ─── Hungarian-lite Assignment (greedy with scoring) ───
function greedyAssign(costMatrix, threshold) {
    const nRows = costMatrix.length;
    const nCols = costMatrix[0]?.length || 0;
    const assignments = [];
    const usedRows = new Set();
    const usedCols = new Set();

    // Collect all valid (row, col, score) pairs
    const candidates = [];
    for (let r = 0; r < nRows; r++) {
        for (let c = 0; c < nCols; c++) {
            if (costMatrix[r][c] >= threshold) {
                candidates.push({ r, c, score: costMatrix[r][c] });
            }
        }
    }

    // Sort by score descending (best matches first)
    candidates.sort((a, b) => b.score - a.score);

    for (const { r, c, score } of candidates) {
        if (!usedRows.has(r) && !usedCols.has(c)) {
            assignments.push({ trackIdx: r, detIdx: c, score });
            usedRows.add(r);
            usedCols.add(c);
        }
    }

    const unmatchedTracks = [];
    for (let r = 0; r < nRows; r++) {
        if (!usedRows.has(r)) unmatchedTracks.push(r);
    }

    const unmatchedDets = [];
    for (let c = 0; c < nCols; c++) {
        if (!usedCols.has(c)) unmatchedDets.push(c);
    }

    return { assignments, unmatchedTracks, unmatchedDets };
}


// ─── Track Object ──────────────────────────────────────
class Track {
    constructor(id, detection, color) {
        this.id = id;
        this.color = color;
        this.keypoints = detection.keypoints.map(kp => ({
            ...kp,
            kalman: new KalmanPoint(kp.x, kp.y),
        }));
        this.bbox = new KalmanBBox(this._computeBBox(detection.keypoints));
        this.appearance = computeAppearance(detection.keypoints);
        this.score = detection.score || 0;

        // Track lifecycle
        this.hits = 1;           // Consecutive hits
        this.misses = 0;         // Consecutive misses
        this.totalHits = 1;
        this.age = 0;            // Total frames since creation
        this.state = 'tentative'; // tentative → confirmed → lost → dead

        // Posture stats per-person
        this.postureScore = null;
        this.postureHistory = [];
        this.goodFrames = 0;
        this.totalAnalyzed = 0;
        this.lastPosture = null;
    }

    predict() {
        this.age++;
        this.bbox.predict();
        this.keypoints.forEach(kp => {
            const pred = kp.kalman.predict();
            kp.x = pred.x;
            kp.y = pred.y;
        });
    }

    update(detection) {
        this.hits++;
        this.totalHits++;
        this.misses = 0;
        this.score = detection.score || this.score;

        // Update each keypoint with Kalman filter
        detection.keypoints.forEach((newKP, i) => {
            if (newKP.score > 0.2 && i < this.keypoints.length) {
                const updated = this.keypoints[i].kalman.update(newKP.x, newKP.y);
                this.keypoints[i].x = updated.x;
                this.keypoints[i].y = updated.y;
                this.keypoints[i].score = newKP.score;
                this.keypoints[i].name = newKP.name;
            }
            // If new keypoint is low-confidence, keep Kalman prediction
        });

        // Update bounding box
        this.bbox.update(this._computeBBox(detection.keypoints));

        // Update appearance for re-id
        const newApp = computeAppearance(detection.keypoints);
        if (newApp && newApp.valid >= 4) {
            if (this.appearance) {
                // Running average of appearance
                for (let i = 0; i < this.appearance.descriptor.length; i++) {
                    this.appearance.descriptor[i] = 0.8 * this.appearance.descriptor[i] +
                        0.2 * newApp.descriptor[i];
                }
                for (let i = 0; i < this.appearance.ratios.length; i++) {
                    this.appearance.ratios[i] = 0.8 * this.appearance.ratios[i] +
                        0.2 * newApp.ratios[i];
                }
            } else {
                this.appearance = newApp;
            }
        }

        // State machine
        if (this.state === 'tentative' && this.hits >= 3) {
            this.state = 'confirmed';
        }
        if (this.state === 'lost') {
            this.state = 'confirmed'; // Re-confirmed!
        }
    }

    markMissed() {
        this.hits = 0;
        this.misses++;
        if (this.state === 'confirmed' && this.misses >= 3) {
            this.state = 'lost';
        }
        if (this.state === 'tentative' && this.misses >= 3) {
            this.state = 'dead';
        }
        if (this.state === 'lost' && this.misses >= 30) {
            this.state = 'dead'; // ~1 second at 30fps
        }
    }

    _computeBBox(keypoints) {
        const valid = keypoints.filter(k => k.score > 0.2);
        if (valid.length === 0) return { x: 0, y: 0, width: 50, height: 50 };
        const xs = valid.map(k => k.x);
        const ys = valid.map(k => k.y);
        const pad = 20;
        return {
            x: Math.min(...xs) - pad,
            y: Math.min(...ys) - pad,
            width: Math.max(...xs) - Math.min(...xs) + pad * 2,
            height: Math.max(...ys) - Math.min(...ys) + pad * 2,
        };
    }

    get predictedBBox() {
        return this.bbox.toBBox();
    }

    get isVisible() {
        return this.state === 'confirmed' || this.state === 'tentative';
    }

    get isOccluded() {
        return this.state === 'lost';
    }
}


// ─── Multi-Person Tracker (ByteTrack-inspired) ────────
const TRACK_COLORS = [
    { primary: '#22c55e', bg: 'rgba(34,197,94,0.15)',  label: 'S1' },
    { primary: '#3b82f6', bg: 'rgba(59,130,246,0.15)', label: 'S2' },
    { primary: '#f59e0b', bg: 'rgba(245,158,11,0.15)', label: 'S3' },
    { primary: '#ef4444', bg: 'rgba(239,68,68,0.15)',  label: 'S4' },
    { primary: '#a855f7', bg: 'rgba(168,85,247,0.15)', label: 'S5' },
    { primary: '#06b6d4', bg: 'rgba(6,182,212,0.15)',  label: 'S6' },
    { primary: '#ec4899', bg: 'rgba(236,72,153,0.15)', label: 'S7' },
    { primary: '#84cc16', bg: 'rgba(132,204,22,0.15)', label: 'S8' },
];

class MultiPersonTracker {
    constructor(config = {}) {
        this.tracks = [];
        this.deadTracks = [];  // Archive for re-identification
        this.nextId = 1;
        this.frameCount = 0;

        // Configurable thresholds
        this.iouThreshHigh = config.iouThreshHigh || 0.3;
        this.iouThreshLow = config.iouThreshLow || 0.1;
        this.reIdThresh = config.reIdThresh || 0.6;
        this.maxDeadAge = config.maxDeadAge || 90;  // ~3s at 30fps
        this.maxTracks = config.maxTracks || 8;
    }

    /**
     * Main tracking step — takes raw detections, returns tracked persons.
     * @param {Array} detections — from MoveNet [{keypoints, score, bbox}, ...]
     * @returns {Track[]} — active tracks with consistent IDs
     */
    update(detections) {
        this.frameCount++;

        // 1. Predict all existing tracks
        this.tracks.forEach(t => t.predict());

        if (detections.length === 0) {
            this.tracks.forEach(t => t.markMissed());
            this._cleanupTracks();
            return this.getActiveTracks();
        }

        // 2. Compute bounding boxes for detections
        detections.forEach(det => {
            if (!det.bbox) {
                det.bbox = this._computeBBox(det.keypoints);
            }
        });

        // 3. First association: IoU matching (high threshold)
        const iouMatrix = this._buildIoUMatrix(this.tracks, detections);
        const firstMatch = greedyAssign(iouMatrix, this.iouThreshHigh);

        // Apply first-round matches
        firstMatch.assignments.forEach(({ trackIdx, detIdx }) => {
            this.tracks[trackIdx].update(detections[detIdx]);
        });

        // 4. Second association: remaining tracks vs remaining detections (lower IoU)
        const remainTracks = firstMatch.unmatchedTracks;
        const remainDets = firstMatch.unmatchedDets;

        if (remainTracks.length > 0 && remainDets.length > 0) {
            const subTracks = remainTracks.map(i => this.tracks[i]);
            const subDets = remainDets.map(i => detections[i]);
            const subMatrix = this._buildIoUMatrix(subTracks, subDets);
            const secondMatch = greedyAssign(subMatrix, this.iouThreshLow);

            secondMatch.assignments.forEach(({ trackIdx, detIdx }) => {
                this.tracks[remainTracks[trackIdx]].update(detections[remainDets[detIdx]]);
            });

            // Mark truly unmatched tracks as missed
            secondMatch.unmatchedTracks.forEach(i => {
                this.tracks[remainTracks[i]].markMissed();
            });

            // 5. Re-ID check for unmatched detections against dead tracks
            const finalUnmatched = secondMatch.unmatchedDets.map(i => remainDets[i]);
            this._handleNewDetections(finalUnmatched.map(i => detections[i]));

        } else {
            remainTracks.forEach(i => this.tracks[i].markMissed());
            this._handleNewDetections(remainDets.map(i => detections[i]));
        }

        // 6. Cleanup dead tracks
        this._cleanupTracks();

        return this.getActiveTracks();
    }

    _handleNewDetections(newDets) {
        newDets.forEach(det => {
            const detApp = computeAppearance(det.keypoints);

            // Try re-identification against archived dead tracks
            let reIdTrack = null;
            let bestSim = this.reIdThresh;

            this.deadTracks.forEach(dead => {
                const sim = appearanceSimilarity(detApp, dead.appearance);
                if (sim > bestSim) {
                    bestSim = sim;
                    reIdTrack = dead;
                }
            });

            if (reIdTrack) {
                // Re-identified! Resurrect track with same ID
                reIdTrack.update(det);
                reIdTrack.state = 'confirmed';
                reIdTrack.misses = 0;
                this.tracks.push(reIdTrack);
                this.deadTracks = this.deadTracks.filter(t => t !== reIdTrack);
                console.log(`🔄 Re-identified person #${reIdTrack.id}`);
            } else if (this.tracks.length < this.maxTracks) {
                // Birth new track
                const color = TRACK_COLORS[(this.nextId - 1) % TRACK_COLORS.length];
                const track = new Track(this.nextId++, det, color);
                this.tracks.push(track);
            }
        });
    }

    _cleanupTracks() {
        const dead = this.tracks.filter(t => t.state === 'dead');
        dead.forEach(t => {
            // Archive for re-identification
            if (t.totalHits >= 10 && t.appearance) {
                this.deadTracks.push(t);
            }
        });

        // Keep only recent dead tracks
        this.deadTracks = this.deadTracks
            .filter(t => this.frameCount - t.age < this.maxDeadAge)
            .slice(-20); // Max 20 archived

        this.tracks = this.tracks.filter(t => t.state !== 'dead');
    }

    _buildIoUMatrix(tracks, detections) {
        return tracks.map(track => {
            const tBBox = track.predictedBBox;
            return detections.map(det => computeIoU(tBBox, det.bbox));
        });
    }

    _computeBBox(keypoints) {
        const valid = keypoints.filter(k => k.score > 0.2);
        if (valid.length === 0) return { x: 0, y: 0, width: 50, height: 50 };
        const xs = valid.map(k => k.x);
        const ys = valid.map(k => k.y);
        const pad = 20;
        return {
            x: Math.min(...xs) - pad,
            y: Math.min(...ys) - pad,
            width: Math.max(...xs) - Math.min(...xs) + pad * 2,
            height: Math.max(...ys) - Math.min(...ys) + pad * 2,
        };
    }

    getActiveTracks() {
        return this.tracks.filter(t => t.isVisible || t.isOccluded);
    }

    getConfirmedTracks() {
        return this.tracks.filter(t => t.state === 'confirmed');
    }

    getStats() {
        return {
            active: this.getActiveTracks().length,
            confirmed: this.getConfirmedTracks().length,
            lost: this.tracks.filter(t => t.state === 'lost').length,
            archived: this.deadTracks.length,
            totalTracked: this.nextId - 1,
            frameCount: this.frameCount,
        };
    }

    reset() {
        this.tracks = [];
        this.deadTracks = [];
        this.nextId = 1;
        this.frameCount = 0;
    }
}

// Export to global scope (non-module script)
window.MultiPersonTracker = MultiPersonTracker;
window.KalmanPoint = KalmanPoint;
