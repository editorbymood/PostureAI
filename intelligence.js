/* =====================================================
   Intelligence Layer — Action Recognition + Pose
   Correction + Rep Counter + Feature Extraction
   
   This is where pose estimation becomes AI intelligence.
   
   Systems:
   1. Feature Extraction — Joint angles, velocities, symmetry
   2. Action Recognition — ST-GCN-lite + rule-based classifier
   3. Pose Correction — Compare vs ideal + real-time feedback
   4. Rep Counter — Motion cycle detection + counting
   5. Attention Weighting — Focus on important joints
   
   Architecture:
   ┌──────────────────────────────────────────────┐
   │  3D Keypoints + Temporal Buffer              │
   │     ↓                                        │
   │  Feature Extraction (angles, velocity, etc.) │
   │     ↓                                        │
   │  ST-GCN-lite (graph-based reasoning)         │
   │     ↓                                        │
   │  Action Classification                       │
   │     ↓                                        │
   │  Pose Correction Engine (vs ideal poses)     │
   │     + Rep Counter (cycle detection)          │
   │     ↓                                        │
   │  Feedback Output (text + visual cues)        │
   └──────────────────────────────────────────────┘
   ===================================================== */

// ─── Keypoint Indices (same as PoseNet/MoveNet) ────────
const KPI = {
    NOSE: 0, L_EYE: 1, R_EYE: 2, L_EAR: 3, R_EAR: 4,
    L_SHOULDER: 5, R_SHOULDER: 6, L_ELBOW: 7, R_ELBOW: 8,
    L_WRIST: 9, R_WRIST: 10, L_HIP: 11, R_HIP: 12,
    L_KNEE: 13, R_KNEE: 14, L_ANKLE: 15, R_ANKLE: 16,
};

// ─── Graph Edges (body skeleton as a graph) ────────────
const BODY_GRAPH_EDGES = [
    [KPI.NOSE, KPI.L_EYE], [KPI.NOSE, KPI.R_EYE],
    [KPI.L_EYE, KPI.L_EAR], [KPI.R_EYE, KPI.R_EAR],
    [KPI.NOSE, KPI.L_SHOULDER], [KPI.NOSE, KPI.R_SHOULDER],
    [KPI.L_SHOULDER, KPI.R_SHOULDER],
    [KPI.L_SHOULDER, KPI.L_ELBOW], [KPI.L_ELBOW, KPI.L_WRIST],
    [KPI.R_SHOULDER, KPI.R_ELBOW], [KPI.R_ELBOW, KPI.R_WRIST],
    [KPI.L_SHOULDER, KPI.L_HIP], [KPI.R_SHOULDER, KPI.R_HIP],
    [KPI.L_HIP, KPI.R_HIP],
    [KPI.L_HIP, KPI.L_KNEE], [KPI.L_KNEE, KPI.L_ANKLE],
    [KPI.R_HIP, KPI.R_KNEE], [KPI.R_KNEE, KPI.R_ANKLE],
];


// ═══════════════════════════════════════════════════════
// 1. FEATURE EXTRACTION
// ═══════════════════════════════════════════════════════

class FeatureExtractor {
    constructor() {
        this.prevFeatures = null;
        this.featureHistory = [];
        this.maxHistory = 60; // ~2s at 30fps
    }

    /**
     * Extract comprehensive features from keypoints.
     * @param {Array} kps — Keypoints [{x, y, score, ...}]
     * @param {Object} angles3D — From Pose3DEstimator
     * @returns {Object} features
     */
    extract(kps, angles3D = {}) {
        const features = {};

        // ── Joint Angles (2D) ──
        features.angles = {
            leftElbow:    this._angle2D(kps, KPI.L_SHOULDER, KPI.L_ELBOW, KPI.L_WRIST),
            rightElbow:   this._angle2D(kps, KPI.R_SHOULDER, KPI.R_ELBOW, KPI.R_WRIST),
            leftShoulder: this._angle2D(kps, KPI.L_ELBOW, KPI.L_SHOULDER, KPI.L_HIP),
            rightShoulder:this._angle2D(kps, KPI.R_ELBOW, KPI.R_SHOULDER, KPI.R_HIP),
            leftHip:      this._angle2D(kps, KPI.L_SHOULDER, KPI.L_HIP, KPI.L_KNEE),
            rightHip:     this._angle2D(kps, KPI.R_SHOULDER, KPI.R_HIP, KPI.R_KNEE),
            leftKnee:     this._angle2D(kps, KPI.L_HIP, KPI.L_KNEE, KPI.L_ANKLE),
            rightKnee:    this._angle2D(kps, KPI.R_HIP, KPI.R_KNEE, KPI.R_ANKLE),
        };

        // ── 3D Angles (if available) ──
        features.angles3D = angles3D;

        // ── Body Symmetry ──
        features.symmetry = this._computeSymmetry(kps);

        // ── Center of Mass (approximation) ──
        features.centerOfMass = this._computeCOM(kps);

        // ── Velocities (if previous frame exists) ──
        features.velocities = this._computeVelocities(kps);

        // ── Body Orientation ──
        features.orientation = this._computeOrientation(kps);

        // ── Limb Positions (relative to torso) ──
        features.limbPositions = this._computeLimbPositions(kps);

        // ── Spatial Attention Weights ──
        features.attention = this._computeAttention(kps, features);

        // Store for temporal analysis
        this.prevFeatures = features;
        this.featureHistory.push({
            angles: { ...features.angles },
            com: { ...features.centerOfMass },
            orientation: features.orientation,
            timestamp: Date.now(),
        });
        if (this.featureHistory.length > this.maxHistory) {
            this.featureHistory.shift();
        }

        return features;
    }

    _angle2D(kps, a, b, c) {
        const pA = kps[a], pB = kps[b], pC = kps[c];
        if (!pA || !pB || !pC) return null;
        if (pA.score < 0.15 || pB.score < 0.15 || pC.score < 0.15) return null;
        const ax = (pA.x || pA.position?.x) - (pB.x || pB.position?.x);
        const ay = (pA.y || pA.position?.y) - (pB.y || pB.position?.y);
        const cx = (pC.x || pC.position?.x) - (pB.x || pB.position?.x);
        const cy = (pC.y || pC.position?.y) - (pB.y || pB.position?.y);
        const dot = ax * cx + ay * cy;
        const cross = ax * cy - ay * cx;
        return Math.abs(Math.atan2(cross, dot) * (180 / Math.PI));
    }

    _computeSymmetry(kps) {
        const pairs = [
            ['shoulder', KPI.L_SHOULDER, KPI.R_SHOULDER],
            ['elbow', KPI.L_ELBOW, KPI.R_ELBOW],
            ['wrist', KPI.L_WRIST, KPI.R_WRIST],
            ['hip', KPI.L_HIP, KPI.R_HIP],
            ['knee', KPI.L_KNEE, KPI.R_KNEE],
            ['ankle', KPI.L_ANKLE, KPI.R_ANKLE],
        ];

        const midX = this._midpointX(kps, KPI.L_SHOULDER, KPI.R_SHOULDER) ||
                      this._midpointX(kps, KPI.L_HIP, KPI.R_HIP) || 0;

        let totalSym = 0, count = 0;
        const details = {};

        pairs.forEach(([name, l, r]) => {
            const lp = kps[l], rp = kps[r];
            if (lp?.score > 0.2 && rp?.score > 0.2) {
                const lx = lp.x || lp.position?.x;
                const rx = rp.x || rp.position?.x;
                const ly = lp.y || lp.position?.y;
                const ry = rp.y || rp.position?.y;
                const lDist = Math.abs(lx - midX);
                const rDist = Math.abs(rx - midX);
                const yDiff = Math.abs(ly - ry);
                const xSym = 1 - Math.abs(lDist - rDist) / Math.max(lDist, rDist, 1);
                const ySym = 1 - yDiff / 100;
                details[name] = Math.max(0, (xSym + ySym) / 2);
                totalSym += details[name];
                count++;
            }
        });

        return {
            overall: count > 0 ? totalSym / count : 0,
            details,
        };
    }

    _midpointX(kps, a, b) {
        const pa = kps[a], pb = kps[b];
        if (pa?.score > 0.2 && pb?.score > 0.2) {
            return ((pa.x || pa.position?.x) + (pb.x || pb.position?.x)) / 2;
        }
        return null;
    }

    _computeCOM(kps) {
        // Weighted center of mass (approximation using joint mass fractions)
        const weights = {
            0: 0.08, 5: 0.12, 6: 0.12, 11: 0.15, 12: 0.15,
            7: 0.06, 8: 0.06, 9: 0.03, 10: 0.03,
            13: 0.08, 14: 0.08, 15: 0.02, 16: 0.02,
        };
        let cx = 0, cy = 0, totalW = 0;
        Object.entries(weights).forEach(([idx, w]) => {
            const kp = kps[parseInt(idx)];
            if (kp && kp.score > 0.15) {
                cx += (kp.x || kp.position?.x || 0) * w;
                cy += (kp.y || kp.position?.y || 0) * w;
                totalW += w;
            }
        });
        return totalW > 0 ? { x: cx / totalW, y: cy / totalW } : { x: 0, y: 0 };
    }

    _computeVelocities(kps) {
        if (!this.prevFeatures) return { overall: 0, perJoint: {} };
        const prevCOM = this.prevFeatures.centerOfMass;
        const com = this._computeCOM(kps);
        const overall = Math.sqrt((com.x - prevCOM.x) ** 2 + (com.y - prevCOM.y) ** 2);
        return { overall, x: com.x - prevCOM.x, y: com.y - prevCOM.y };
    }

    _computeOrientation(kps) {
        const ls = kps[KPI.L_SHOULDER], rs = kps[KPI.R_SHOULDER];
        const lh = kps[KPI.L_HIP], rh = kps[KPI.R_HIP];

        let facing = 'unknown';
        if (ls?.score > 0.3 && rs?.score > 0.3) {
            const shoulderW = Math.abs((ls.x || ls.position?.x) - (rs.x || rs.position?.x));
            const shoulderH = Math.abs((ls.y || ls.position?.y) - (rs.y || rs.position?.y));
            if (shoulderH > shoulderW * 0.5) {
                facing = 'side';
            } else {
                facing = 'front';
            }
        }

        let standing = 'unknown';
        if (ls?.score > 0.3 && lh?.score > 0.3) {
            const torsoLen = Math.abs((ls.y || ls.position?.y) - (lh.y || lh.position?.y));
            const torsoW = Math.abs((ls.x || ls.position?.x) - (lh.x || lh.position?.x));
            if (torsoW > torsoLen * 0.8) {
                standing = 'lying';
            } else if (torsoLen > 50) {
                standing = 'upright';
            }
        }

        return { facing, standing };
    }

    _computeLimbPositions(kps) {
        const getPos = (idx) => {
            const kp = kps[idx];
            if (!kp || kp.score < 0.15) return null;
            return { x: kp.x || kp.position?.x || 0, y: kp.y || kp.position?.y || 0 };
        };

        const shoulderMid = this._midpoint(kps, KPI.L_SHOULDER, KPI.R_SHOULDER);
        if (!shoulderMid) return {};

        const positions = {};
        const names = ['leftWrist', 'rightWrist', 'leftAnkle', 'rightAnkle'];
        const indices = [KPI.L_WRIST, KPI.R_WRIST, KPI.L_ANKLE, KPI.R_ANKLE];

        indices.forEach((idx, i) => {
            const pos = getPos(idx);
            if (pos) {
                positions[names[i]] = {
                    x: pos.x - shoulderMid.x,
                    y: pos.y - shoulderMid.y,
                    above: pos.y < shoulderMid.y,
                };
            }
        });

        return positions;
    }

    _midpoint(kps, a, b) {
        const pa = kps[a], pb = kps[b];
        if (pa?.score > 0.15 && pb?.score > 0.15) {
            return {
                x: ((pa.x || pa.position?.x) + (pb.x || pb.position?.x)) / 2,
                y: ((pa.y || pa.position?.y) + (pb.y || pb.position?.y)) / 2,
            };
        }
        return null;
    }

    /**
     * Spatial attention — weight keypoints by importance for current action.
     */
    _computeAttention(kps, features) {
        const weights = new Array(17).fill(1.0);
        
        // Base: confidence-weighted
        kps.forEach((kp, i) => {
            weights[i] = kp.score || 0;
        });

        // Boost joints with high velocity (moving joints are more important)
        if (this.prevFeatures) {
            // Movement-based attention
            kps.forEach((kp, i) => {
                if (this.prevFeatures._prevKps && this.prevFeatures._prevKps[i]) {
                    const prev = this.prevFeatures._prevKps[i];
                    const dx = (kp.x || 0) - (prev.x || 0);
                    const dy = (kp.y || 0) - (prev.y || 0);
                    const motion = Math.sqrt(dx * dx + dy * dy);
                    weights[i] *= (1 + motion * 0.01); // Boost moving joints
                }
            });
        }

        // Normalize
        const sum = weights.reduce((s, w) => s + w, 0);
        return weights.map(w => sum > 0 ? w / sum : 1 / 17);
    }

    reset() {
        this.prevFeatures = null;
        this.featureHistory = [];
    }
}


// ═══════════════════════════════════════════════════════
// 2. ACTION RECOGNITION (ST-GCN-lite + Rules)
// ═══════════════════════════════════════════════════════

// Action definitions with angle patterns
const ACTIONS = {
    standing: {
        label: '🧍 Standing',
        icon: '🧍',
        conditions: (f) => {
            const hL = f.angles.leftHip, hR = f.angles.rightHip;
            const kL = f.angles.leftKnee, kR = f.angles.rightKnee;
            return (hL > 150 || hR > 150) && (kL > 155 || kR > 155) &&
                   f.orientation.standing === 'upright';
        },
        priority: 1,
    },
    sitting: {
        label: '🪑 Sitting',
        icon: '🪑',
        conditions: (f) => {
            const hL = f.angles.leftHip, hR = f.angles.rightHip;
            const kL = f.angles.leftKnee, kR = f.angles.rightKnee;
            return (hL !== null && hL < 130) && (kL !== null && kL < 140) &&
                   f.orientation.standing === 'upright';
        },
        priority: 2,
    },
    squat: {
        label: '🏋️ Squat',
        icon: '🏋️',
        conditions: (f) => {
            const hL = f.angles.leftHip, hR = f.angles.rightHip;
            const kL = f.angles.leftKnee, kR = f.angles.rightKnee;
            return ((hL !== null && hL < 110) || (hR !== null && hR < 110)) &&
                   ((kL !== null && kL < 110) || (kR !== null && kR < 110)) &&
                   f.orientation.standing === 'upright';
        },
        priority: 5,
        repTrackJoint: 'leftKnee',
        repRange: [70, 170],
    },
    pushup: {
        label: '💪 Push-Up',
        icon: '💪',
        conditions: (f) => {
            const eL = f.angles.leftElbow, eR = f.angles.rightElbow;
            return f.orientation.standing !== 'upright' &&
                   ((eL !== null && eL < 130) || (eR !== null && eR < 130)) &&
                   f.orientation.facing !== 'front';
        },
        priority: 6,
        repTrackJoint: 'leftElbow',
        repRange: [60, 170],
    },
    armRaise: {
        label: '🙌 Arm Raise',
        icon: '🙌',
        conditions: (f) => {
            const sL = f.angles.leftShoulder, sR = f.angles.rightShoulder;
            const wristAbove = f.limbPositions?.leftWrist?.above || f.limbPositions?.rightWrist?.above;
            return wristAbove &&
                   ((sL !== null && sL > 140) || (sR !== null && sR > 140));
        },
        priority: 4,
    },
    lunge: {
        label: '🦵 Lunge',
        icon: '🦵',
        conditions: (f) => {
            const kL = f.angles.leftKnee, kR = f.angles.rightKnee;
            if (kL === null || kR === null) return false;
            return Math.abs(kL - kR) > 40 && (kL < 120 || kR < 120) &&
                   f.orientation.standing === 'upright';
        },
        priority: 5,
        repTrackJoint: 'leftKnee',
        repRange: [80, 170],
    },
    lying: {
        label: '🛌 Lying Down',
        icon: '🛌',
        conditions: (f) => f.orientation.standing === 'lying',
        priority: 0,
    },
};

class ActionRecognizer {
    constructor() {
        this.currentAction = null;
        this.actionHistory = [];
        this.confidence = 0;
        this.holdFrames = 0;
        this.minHoldFrames = 5; // Must hold for 5 frames to confirm
        this.graphFeatures = []; // For ST-GCN-lite
    }

    /**
     * Classify current action from extracted features.
     * Uses rule-based matching + temporal consistency.
     * @param {Object} features — from FeatureExtractor
     * @returns {Object} — {action, confidence, label, icon}
     */
    classify(features) {
        // ── Graph Feature Extraction (ST-GCN-lite) ──
        this._updateGraphFeatures(features);

        // ── Rule-based classification ──
        let bestAction = null;
        let bestPriority = -1;

        Object.entries(ACTIONS).forEach(([key, def]) => {
            try {
                if (def.conditions(features) && def.priority > bestPriority) {
                    bestAction = key;
                    bestPriority = def.priority;
                }
            } catch (e) {
                // Skip actions with missing data
            }
        });

        // ── Temporal consistency (hold detection) ──
        if (bestAction === this.currentAction) {
            this.holdFrames++;
            this.confidence = Math.min(1, this.holdFrames / 15);
        } else {
            this.holdFrames = 0;
        }

        // Only switch action after enough hold frames
        if (this.holdFrames >= this.minHoldFrames || !this.currentAction) {
            if (bestAction !== this.currentAction) {
                this.currentAction = bestAction;
                this.holdFrames = 0;
                this.confidence = 0.3;
            }
        }

        // Action history for temporal analysis
        this.actionHistory.push({
            action: this.currentAction,
            timestamp: Date.now(),
        });
        if (this.actionHistory.length > 120) this.actionHistory.shift();

        const actionDef = ACTIONS[this.currentAction] || { label: '❓ Unknown', icon: '❓' };

        return {
            action: this.currentAction || 'unknown',
            label: actionDef.label,
            icon: actionDef.icon,
            confidence: this.confidence,
            holdFrames: this.holdFrames,
            repTrackJoint: actionDef.repTrackJoint || null,
            repRange: actionDef.repRange || null,
        };
    }

    /**
     * ST-GCN-lite — extract graph-based spatial-temporal features.
     * Uses adjacency matrix to propagate features along skeleton edges.
     */
    _updateGraphFeatures(features) {
        // Build node features: [angle, symmetry_deviation, velocity_component]
        const nodeFeatures = new Array(17).fill(0);

        // Map angles to their respective joints
        const angleMap = {
            [KPI.L_ELBOW]: features.angles.leftElbow,
            [KPI.R_ELBOW]: features.angles.rightElbow,
            [KPI.L_SHOULDER]: features.angles.leftShoulder,
            [KPI.R_SHOULDER]: features.angles.rightShoulder,
            [KPI.L_HIP]: features.angles.leftHip,
            [KPI.R_HIP]: features.angles.rightHip,
            [KPI.L_KNEE]: features.angles.leftKnee,
            [KPI.R_KNEE]: features.angles.rightKnee,
        };

        Object.entries(angleMap).forEach(([idx, val]) => {
            if (val !== null) nodeFeatures[parseInt(idx)] = val / 180; // Normalize 0-1
        });

        // Graph convolution: propagate features along edges
        const output = [...nodeFeatures];
        BODY_GRAPH_EDGES.forEach(([a, b]) => {
            output[a] += nodeFeatures[b] * 0.3;
            output[b] += nodeFeatures[a] * 0.3;
        });

        this.graphFeatures.push(output);
        if (this.graphFeatures.length > 30) this.graphFeatures.shift();
    }

    reset() {
        this.currentAction = null;
        this.actionHistory = [];
        this.confidence = 0;
        this.holdFrames = 0;
        this.graphFeatures = [];
    }
}


// ═══════════════════════════════════════════════════════
// 3. POSE CORRECTION ENGINE
// ═══════════════════════════════════════════════════════

// Ideal pose definitions
const IDEAL_POSES = {
    standing: {
        leftKnee: { min: 165, max: 180, tip: 'Keep your knees straight' },
        rightKnee: { min: 165, max: 180, tip: 'Keep your knees straight' },
        leftHip: { min: 160, max: 180, tip: 'Stand upright — don\'t lean forward' },
        rightHip: { min: 160, max: 180, tip: 'Stand upright — don\'t lean forward' },
        leftShoulder: { min: 0, max: 45, tip: 'Relax your shoulders down' },
        rightShoulder: { min: 0, max: 45, tip: 'Relax your shoulders down' },
    },
    sitting: {
        leftHip: { min: 80, max: 100, tip: 'Sit at a 90° hip angle' },
        rightHip: { min: 80, max: 100, tip: 'Sit at a 90° hip angle' },
        leftKnee: { min: 80, max: 100, tip: 'Keep your knees at 90°' },
        rightKnee: { min: 80, max: 100, tip: 'Keep your knees at 90°' },
    },
    squat: {
        leftKnee: { min: 70, max: 100, tip: 'Go deeper — knees to 90°' },
        rightKnee: { min: 70, max: 100, tip: 'Go deeper — knees to 90°' },
        leftHip: { min: 60, max: 100, tip: 'Push your hips back more' },
        rightHip: { min: 60, max: 100, tip: 'Push your hips back more' },
        _globalTips: [
            { check: (f) => f.symmetry.overall < 0.7, tip: 'Keep your weight balanced on both sides' },
            { check: (f) => {
                const kL = f.angles.leftKnee, kR = f.angles.rightKnee;
                return kL && kR && Math.abs(kL - kR) > 15;
            }, tip: 'Your knees are uneven — match depth on both sides' },
        ],
    },
    pushup: {
        leftElbow: { min: 70, max: 100, tip: 'Go lower — elbows to 90°' },
        rightElbow: { min: 70, max: 100, tip: 'Go lower — elbows to 90°' },
        _globalTips: [
            { check: (f) => {
                const h = f.angles.leftHip || f.angles.rightHip;
                return h && h < 150;
            }, tip: 'Keep your body straight — don\'t sag your hips' },
        ],
    },
};

class PoseCorrectionEngine {
    constructor() {
        this.lastFeedback = [];
        this.feedbackCooldowns = {}; // Prevent spamming same tip
        this.overallAccuracy = null;
    }

    /**
     * Compare current pose against ideal and generate feedback.
     * @param {string} action — current detected action
     * @param {Object} features — from FeatureExtractor
     * @returns {Object} — {corrections, accuracy, tips}
     */
    evaluate(action, features) {
        const ideal = IDEAL_POSES[action];
        if (!ideal) return { corrections: [], accuracy: null, tips: [] };

        const corrections = [];
        let totalMatch = 0;
        let matchCount = 0;

        // ── Per-joint correction ──
        Object.entries(ideal).forEach(([joint, range]) => {
            if (joint.startsWith('_')) return; // Skip global tips

            const angle = features.angles[joint];
            if (angle === null || angle === undefined) return;

            matchCount++;
            if (angle >= range.min && angle <= range.max) {
                totalMatch++;
            } else {
                const deviation = angle < range.min
                    ? range.min - angle
                    : angle - range.max;

                if (deviation > 5 && !this._onCooldown(joint)) {
                    corrections.push({
                        joint,
                        currentAngle: Math.round(angle),
                        idealMin: range.min,
                        idealMax: range.max,
                        deviation: Math.round(deviation),
                        severity: deviation > 30 ? 'high' : deviation > 15 ? 'medium' : 'low',
                        tip: range.tip,
                        direction: angle < range.min ? 'increase' : 'decrease',
                    });
                    this._setCooldown(joint);
                }
            }
        });

        // ── Global tips ──
        const tips = [];
        if (ideal._globalTips) {
            ideal._globalTips.forEach(gt => {
                try {
                    if (gt.check(features) && !this._onCooldown(gt.tip)) {
                        tips.push(gt.tip);
                        this._setCooldown(gt.tip);
                    }
                } catch (e) { /* skip */ }
            });
        }

        // ── Overall accuracy ──
        const accuracy = matchCount > 0 ? Math.round((totalMatch / matchCount) * 100) : null;
        this.overallAccuracy = accuracy !== null
            ? (this.overallAccuracy === null ? accuracy : 0.8 * this.overallAccuracy + 0.2 * accuracy)
            : this.overallAccuracy;

        this.lastFeedback = corrections;

        return {
            corrections: corrections.slice(0, 3), // Max 3 corrections at once
            accuracy,
            smoothedAccuracy: Math.round(this.overallAccuracy || 0),
            tips,
            isIdeal: corrections.length === 0 && accuracy !== null,
        };
    }

    _onCooldown(key) {
        const cd = this.feedbackCooldowns[key];
        if (!cd) return false;
        return Date.now() - cd < 3000; // 3 second cooldown
    }

    _setCooldown(key) {
        this.feedbackCooldowns[key] = Date.now();
    }

    reset() {
        this.lastFeedback = [];
        this.feedbackCooldowns = {};
        this.overallAccuracy = null;
    }
}


// ═══════════════════════════════════════════════════════
// 4. REP COUNTER (Motion Cycle Detection)
// ═══════════════════════════════════════════════════════

class RepCounter {
    constructor() {
        this.counters = {}; // Per-exercise rep counts
        this.angleBuffers = {}; // Per-joint angle history
        this.phases = {}; // Current phase per exercise
        this.maxBufferSize = 90; // ~3s at 30fps
    }

    /**
     * Track repetitions for the current action.
     * Uses peak/valley detection on tracked joint angles.
     * @param {string} action — detected action
     * @param {Object} features — from FeatureExtractor
     * @param {Object} actionInfo — from ActionRecognizer (includes repTrackJoint, repRange)
     * @returns {Object} — {reps, phase, progress, isCountable}
     */
    track(action, features, actionInfo) {
        if (!actionInfo.repTrackJoint || !actionInfo.repRange) {
            return { reps: 0, phase: 'none', progress: 0, isCountable: false };
        }

        const joint = actionInfo.repTrackJoint;
        const [minAngle, maxAngle] = actionInfo.repRange;
        const angle = features.angles[joint];

        if (angle === null || angle === undefined) {
            return this._getState(action);
        }

        // Initialize per-exercise state
        if (!this.counters[action]) {
            this.counters[action] = 0;
            this.angleBuffers[action] = [];
            this.phases[action] = 'neutral'; // neutral → down → up → neutral (1 rep)
        }

        // Push to angle buffer
        this.angleBuffers[action].push(angle);
        if (this.angleBuffers[action].length > this.maxBufferSize) {
            this.angleBuffers[action].shift();
        }

        // ── Phase detection (state machine) ──
        const midAngle = (minAngle + maxAngle) / 2;
        const range = maxAngle - minAngle;
        const downThresh = minAngle + range * 0.3; // 30% from bottom
        const upThresh = maxAngle - range * 0.3;   // 30% from top

        const phase = this.phases[action];

        if (phase === 'neutral' && angle < midAngle) {
            this.phases[action] = 'going_down';
        } else if (phase === 'going_down' && angle <= downThresh) {
            this.phases[action] = 'at_bottom';
        } else if (phase === 'at_bottom' && angle > midAngle) {
            this.phases[action] = 'going_up';
        } else if (phase === 'going_up' && angle >= upThresh) {
            // Completed one rep!
            this.counters[action]++;
            this.phases[action] = 'neutral';
        }

        // Progress within the current rep
        const progress = Math.max(0, Math.min(1,
            (angle - minAngle) / (maxAngle - minAngle)
        ));

        return {
            reps: this.counters[action],
            phase: this.phases[action],
            progress: Math.round(progress * 100),
            isCountable: true,
            currentAngle: Math.round(angle),
            targetMin: minAngle,
            targetMax: maxAngle,
        };
    }

    _getState(action) {
        return {
            reps: this.counters[action] || 0,
            phase: this.phases[action] || 'none',
            progress: 0,
            isCountable: !!this.counters[action],
        };
    }

    reset(action) {
        if (action) {
            delete this.counters[action];
            delete this.angleBuffers[action];
            delete this.phases[action];
        } else {
            this.counters = {};
            this.angleBuffers = {};
            this.phases = {};
        }
    }

    getTotalReps() {
        return Object.values(this.counters).reduce((s, v) => s + v, 0);
    }

    getAllCounts() {
        return { ...this.counters };
    }
}


// ═══════════════════════════════════════════════════════
// 5. OCCLUSION RECOVERY (GNN-inspired)
// ═══════════════════════════════════════════════════════

class OcclusionRecovery {
    /**
     * Predict missing/low-confidence keypoints using:
     * 1. Symmetry assumptions
     * 2. Graph-based reasoning (adjacent joints)
     * 3. Anatomical constraints
     */
    static recover(keypoints) {
        const recovered = keypoints.map(kp => ({ ...kp, _recovered: false }));

        // ── Symmetry-based recovery ──
        const symPairs = [
            [KPI.L_SHOULDER, KPI.R_SHOULDER],
            [KPI.L_ELBOW, KPI.R_ELBOW],
            [KPI.L_WRIST, KPI.R_WRIST],
            [KPI.L_HIP, KPI.R_HIP],
            [KPI.L_KNEE, KPI.R_KNEE],
            [KPI.L_ANKLE, KPI.R_ANKLE],
        ];

        // Find body midline
        const midX = OcclusionRecovery._getMidline(keypoints);

        symPairs.forEach(([l, r]) => {
            const lp = recovered[l], rp = recovered[r];
            if (lp.score > 0.3 && rp.score < 0.15 && midX !== null) {
                // Mirror left to right
                const lx = lp.x || lp.position?.x;
                const ly = lp.y || lp.position?.y;
                const rx = 2 * midX - lx;
                rp.x = rx;
                if (rp.position) rp.position.x = rx;
                rp.y = ly;
                if (rp.position) rp.position.y = ly;
                rp.score = lp.score * 0.4;
                rp._recovered = true;
            } else if (rp.score > 0.3 && lp.score < 0.15 && midX !== null) {
                const rx = rp.x || rp.position?.x;
                const ry = rp.y || rp.position?.y;
                const lx = 2 * midX - rx;
                lp.x = lx;
                if (lp.position) lp.position.x = lx;
                lp.y = ry;
                if (lp.position) lp.position.y = ry;
                lp.score = rp.score * 0.4;
                lp._recovered = true;
            }
        });

        // ── Graph-based: interpolate from adjacent joints ──
        const adjacency = {
            [KPI.L_ELBOW]: [KPI.L_SHOULDER, KPI.L_WRIST],
            [KPI.R_ELBOW]: [KPI.R_SHOULDER, KPI.R_WRIST],
            [KPI.L_KNEE]: [KPI.L_HIP, KPI.L_ANKLE],
            [KPI.R_KNEE]: [KPI.R_HIP, KPI.R_ANKLE],
            [KPI.L_HIP]: [KPI.L_SHOULDER, KPI.L_KNEE],
            [KPI.R_HIP]: [KPI.R_SHOULDER, KPI.R_KNEE],
        };

        Object.entries(adjacency).forEach(([joint, neighbors]) => {
            const idx = parseInt(joint);
            const kp = recovered[idx];
            if (kp.score < 0.15 && !kp._recovered) {
                const validNeighbors = neighbors
                    .map(n => recovered[n])
                    .filter(n => n.score > 0.2);

                if (validNeighbors.length === 2) {
                    const [a, b] = validNeighbors;
                    const ax = a.x || a.position?.x || 0;
                    const ay = a.y || a.position?.y || 0;
                    const bx = b.x || b.position?.x || 0;
                    const by = b.y || b.position?.y || 0;
                    kp.x = (ax + bx) / 2;
                    kp.y = (ay + by) / 2;
                    if (kp.position) {
                        kp.position.x = kp.x;
                        kp.position.y = kp.y;
                    }
                    kp.score = Math.min(a.score, b.score) * 0.3;
                    kp._recovered = true;
                }
            }
        });

        return recovered;
    }

    static _getMidline(kps) {
        const pairs = [[KPI.L_SHOULDER, KPI.R_SHOULDER], [KPI.L_HIP, KPI.R_HIP]];
        for (const [l, r] of pairs) {
            if (kps[l]?.score > 0.3 && kps[r]?.score > 0.3) {
                return ((kps[l].x || kps[l].position?.x) + (kps[r].x || kps[r].position?.x)) / 2;
            }
        }
        return null;
    }
}


// ═══════════════════════════════════════════════════════
// 6. UNIFIED INTELLIGENCE ENGINE
// ═══════════════════════════════════════════════════════

class IntelligenceEngine {
    constructor(config = {}) {
        this.featureExtractor = new FeatureExtractor();
        this.actionRecognizer = new ActionRecognizer();
        this.poseCorrection = new PoseCorrectionEngine();
        this.repCounter = new RepCounter();
        this.pose3D = new Pose3DEstimator(config.pose3D || {});

        this.lastResult = null;
        this.frameCount = 0;
    }

    /**
     * Run full intelligence pipeline on keypoints.
     * @param {Array} keypoints — [{x, y, score}, ...] or [{position: {x,y}, score}, ...]
     * @returns {Object} — Complete intelligence output
     */
    process(keypoints) {
        this.frameCount++;

        // ── Normalize keypoint format ──
        const kps = keypoints.map(kp => ({
            x: kp.x ?? kp.position?.x ?? 0,
            y: kp.y ?? kp.position?.y ?? 0,
            score: kp.score ?? 0,
            name: kp.name ?? kp.part ?? '',
        }));

        // Step 0: Occlusion recovery
        const recoveredKps = OcclusionRecovery.recover(kps);

        // Step 1: 3D Pose estimation
        const pose3DResult = this.pose3D.estimate(recoveredKps);

        // Step 2: Feature extraction
        const features = this.featureExtractor.extract(recoveredKps, pose3DResult.angles3D);

        // Step 3: Action recognition
        const action = this.actionRecognizer.classify(features);

        // Step 4: Pose correction
        const correction = this.poseCorrection.evaluate(action.action, features);

        // Step 5: Rep counting
        const reps = this.repCounter.track(action.action, features, action);

        // Compile result
        this.lastResult = {
            pose3D: pose3DResult,
            features,
            action,
            correction,
            reps,
            recoveredJoints: recoveredKps.filter(k => k._recovered).length,
            symmetry: features.symmetry,
            frameCount: this.frameCount,
        };

        return this.lastResult;
    }

    reset() {
        this.featureExtractor.reset();
        this.actionRecognizer.reset();
        this.poseCorrection.reset();
        this.repCounter.reset();
        this.pose3D.reset();
        this.lastResult = null;
        this.frameCount = 0;
    }
}

// Export
window.IntelligenceEngine = IntelligenceEngine;
window.FeatureExtractor = FeatureExtractor;
window.ActionRecognizer = ActionRecognizer;
window.PoseCorrectionEngine = PoseCorrectionEngine;
window.RepCounter = RepCounter;
window.OcclusionRecovery = OcclusionRecovery;
window.ACTIONS = ACTIONS;
