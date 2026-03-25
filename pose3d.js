/* =====================================================
   Monocular 3D Pose Estimation Engine
   
   Estimates depth (Z-axis) from 2D keypoints using:
   1. Anatomical bone-length priors
   2. Perspective foreshortening analysis
   3. Cross-ratio depth estimation
   4. Symmetry-based depth inference
   
   Outputs a full 3D skeleton that can drive:
   - Real joint angle calculation (not just projected)
   - AR/VR avatar mapping
   - Motion quality analysis with depth context
   
   Architecture:
   ┌────────────────────────────────────────────────┐
   │  2D Keypoints (x, y, confidence)              │
   │     ↓                                         │
   │  Bone Length Analysis (foreshortening → depth) │
   │     ↓                                         │
   │  Cross-Ratio Depth (torso perspective)        │
   │     ↓                                         │
   │  Symmetry Constraint (left-right pairing)     │
   │     ↓                                         │
   │  Temporal Smoothing (Z stability)             │
   │     ↓                                         │
   │  3D Keypoints (x, y, z) + True Angles         │
   └────────────────────────────────────────────────┘
   ===================================================== */

// ─── Anatomical Constants (average adult, normalized) ──
const BONE_LENGTHS = {
    // Upper body (relative to shoulder width = 1.0)
    shoulderWidth:     1.0,
    upperArm:          0.85,   // shoulder → elbow
    forearm:           0.75,   // elbow → wrist
    neckToNose:        0.55,   // shoulder center → nose
    torso:             1.45,   // shoulder center → hip center
    // Lower body
    upperLeg:          1.2,    // hip → knee
    lowerLeg:          1.15,   // knee → ankle
    hipWidth:          0.75,   // left hip → right hip
    // Head
    eyeSpan:           0.25,   // left eye → right eye
    earSpan:           0.35,   // left ear → right ear
};

// Bone pair definitions for length measurement
const BONE_PAIRS = [
    { name: 'leftUpperArm',   a: 5,  b: 7,  refLength: BONE_LENGTHS.upperArm },
    { name: 'leftForearm',    a: 7,  b: 9,  refLength: BONE_LENGTHS.forearm },
    { name: 'rightUpperArm',  a: 6,  b: 8,  refLength: BONE_LENGTHS.upperArm },
    { name: 'rightForearm',   a: 8,  b: 10, refLength: BONE_LENGTHS.forearm },
    { name: 'leftTorso',      a: 5,  b: 11, refLength: BONE_LENGTHS.torso },
    { name: 'rightTorso',     a: 6,  b: 12, refLength: BONE_LENGTHS.torso },
    { name: 'leftUpperLeg',   a: 11, b: 13, refLength: BONE_LENGTHS.upperLeg },
    { name: 'leftLowerLeg',   a: 13, b: 15, refLength: BONE_LENGTHS.lowerLeg },
    { name: 'rightUpperLeg',  a: 12, b: 14, refLength: BONE_LENGTHS.upperLeg },
    { name: 'rightLowerLeg',  a: 14, b: 16, refLength: BONE_LENGTHS.lowerLeg },
];

// Symmetric joint pairs (left ↔ right)
const SYMMETRIC_PAIRS = [
    [1, 2],   // eyes
    [3, 4],   // ears
    [5, 6],   // shoulders
    [7, 8],   // elbows
    [9, 10],  // wrists
    [11, 12], // hips
    [13, 14], // knees
    [15, 16], // ankles
];


class Pose3DEstimator {
    constructor(config = {}) {
        this.focalLength = config.focalLength || 800;  // Approximate focal length in pixels
        this.calibrated = false;
        this.refShoulderWidth = null;  // Pixel width of shoulders at known depth
        this.refDepth = 1.0;           // Reference depth (normalized)
        
        // Per-keypoint Z state
        this.zHistory = new Array(17).fill(null).map(() => ({
            z: 0,
            smoothed: 0,
            velocity: 0,
            history: [],
        }));

        // Calibration data
        this.personScale = 1.0;
        this.frameCount = 0;
    }

    /**
     * Estimate 3D positions from 2D keypoints.
     * @param {Array} keypoints2D — [{x, y, score}, ...] (17 keypoints)
     * @returns {Object} — {keypoints3D, angles3D, depthMap, quality}
     */
    estimate(keypoints2D) {
        this.frameCount++;
        const kps = keypoints2D;

        // Step 1: Calibrate scale from shoulder width
        this._calibrateScale(kps);

        // Step 2: Estimate depth per keypoint
        const depthMap = this._estimateDepths(kps);

        // Step 3: Apply symmetry constraints
        this._applySymmetry(depthMap, kps);

        // Step 4: Temporal smoothing of Z values
        this._smoothDepths(depthMap);

        // Step 5: Build 3D keypoints
        const keypoints3D = kps.map((kp, i) => ({
            x: kp.x,
            y: kp.y,
            z: this.zHistory[i].smoothed,
            score: kp.score,
            name: kp.name || '',
            depthConfidence: depthMap[i]?.confidence || 0,
        }));

        // Step 6: Compute true 3D angles
        const angles3D = this._compute3DAngles(keypoints3D);

        // Step 7: Quality assessment
        const quality = this._assessQuality(keypoints3D, depthMap);

        return { keypoints3D, angles3D, depthMap, quality };
    }

    /**
     * Calibrate person scale from shoulder width.
     */
    _calibrateScale(kps) {
        const ls = kps[5], rs = kps[6];
        if (ls.score > 0.3 && rs.score > 0.3) {
            const sw = Math.sqrt((ls.x - rs.x) ** 2 + (ls.y - rs.y) ** 2);
            if (sw > 20) {
                if (!this.calibrated) {
                    this.refShoulderWidth = sw;
                    this.calibrated = true;
                } else {
                    // Running average
                    this.refShoulderWidth = 0.95 * this.refShoulderWidth + 0.05 * sw;
                }
                this.personScale = sw / (this.refShoulderWidth || sw);
            }
        }
    }

    /**
     * Estimate depth for each keypoint using bone foreshortening.
     */
    _estimateDepths(kps) {
        const depthMap = new Array(17).fill(null).map(() => ({
            z: 0,
            confidence: 0,
            method: 'none',
        }));

        if (!this.refShoulderWidth || this.refShoulderWidth < 20) return depthMap;

        const scale = this.refShoulderWidth; // Pixels per unit shoulder width

        // ── Method 1: Bone foreshortening ──
        // If a bone appears shorter than expected, the limb is pointing toward/away from camera
        BONE_PAIRS.forEach(bone => {
            const a = kps[bone.a], b = kps[bone.b];
            if (a.score > 0.2 && b.score > 0.2) {
                const apparent = Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2);
                const expected = bone.refLength * scale;
                
                if (expected > 0) {
                    const ratio = apparent / expected;
                    // ratio < 1 means foreshortened → bone pointing into/out of screen
                    // ratio ≈ 1 means bone is parallel to image plane (z ≈ 0)
                    let dz = 0;
                    if (ratio < 0.95) {
                        // Foreshortened: estimate Z from ratio
                        // apparent² + dz² ≈ expected² (Pythagorean)
                        const clampedRatio = Math.max(0.1, ratio);
                        dz = Math.sqrt(Math.max(0, 1 - clampedRatio * clampedRatio)) * expected;
                    }

                    // Determine direction: which joint is higher in Y is usually closer
                    const direction = (b.y < a.y) ? 1 : -1;
                    
                    // Update both endpoints
                    const conf = Math.min(a.score, b.score) * (1 - Math.abs(ratio - 0.5));
                    if (conf > depthMap[bone.b].confidence) {
                        depthMap[bone.b].z = depthMap[bone.a].z + direction * dz / scale;
                        depthMap[bone.b].confidence = conf;
                        depthMap[bone.b].method = 'foreshortening';
                    }
                }
            }
        });

        // ── Method 2: Torso perspective (cross-ratio) ──
        // Shoulder width vs hip width gives depth tilt
        const ls = kps[5], rs = kps[6], lh = kps[11], rh = kps[12];
        if (ls.score > 0.3 && rs.score > 0.3 && lh.score > 0.3 && rh.score > 0.3) {
            const shoulderW = Math.sqrt((ls.x - rs.x) ** 2 + (ls.y - rs.y) ** 2);
            const hipW = Math.sqrt((lh.x - rh.x) ** 2 + (lh.y - rh.y) ** 2);
            const expectedRatio = BONE_LENGTHS.hipWidth / BONE_LENGTHS.shoulderWidth;
            const actualRatio = hipW / shoulderW;
            
            // If hips appear narrower, they're further from camera (torso tilted)
            const torsoDzNorm = (actualRatio - expectedRatio) * 2;
            const torsoConf = Math.min(ls.score, rs.score, lh.score, rh.score);
            
            [11, 12].forEach(idx => {
                if (torsoConf > depthMap[idx].confidence * 0.8) {
                    depthMap[idx].z = torsoDzNorm;
                    depthMap[idx].confidence = Math.max(depthMap[idx].confidence, torsoConf * 0.7);
                    depthMap[idx].method = 'cross-ratio';
                }
            });
        }

        // ── Method 3: Nose depth from face size ──
        const nose = kps[0], le = kps[1], re = kps[2];
        if (nose.score > 0.3 && le.score > 0.3 && re.score > 0.3) {
            const eyeSpan = Math.sqrt((le.x - re.x) ** 2 + (le.y - re.y) ** 2);
            const expectedEyeSpan = BONE_LENGTHS.eyeSpan * scale;
            if (expectedEyeSpan > 5) {
                const faceScale = eyeSpan / expectedEyeSpan;
                // Larger face = closer to camera → negative Z
                depthMap[0].z = (1 - faceScale) * 0.5;
                depthMap[0].confidence = nose.score * 0.6;
                depthMap[0].method = 'face-scale';
            }
        }

        // ── Set base depth for torso (reference plane) ──
        [5, 6].forEach(idx => {
            if (depthMap[idx].confidence < 0.1) {
                depthMap[idx].z = 0; // Shoulders = reference depth
                depthMap[idx].confidence = kps[idx].score * 0.5;
                depthMap[idx].method = 'reference';
            }
        });

        return depthMap;
    }

    /**
     * Apply symmetry constraints between left/right pairs.
     */
    _applySymmetry(depthMap, kps) {
        SYMMETRIC_PAIRS.forEach(([l, r]) => {
            const dL = depthMap[l], dR = depthMap[r];
            if (kps[l].score > 0.3 && kps[r].score > 0.3) {
                // Both detected: verify they're at similar depth (symmetric assumption)
                if (dL.confidence > 0.1 && dR.confidence > 0.1) {
                    const avg = (dL.z + dR.z) / 2;
                    const diff = Math.abs(dL.z - dR.z);
                    // If difference is small, average them (symmetric pose)
                    if (diff < 0.3) {
                        dL.z = 0.7 * dL.z + 0.3 * avg;
                        dR.z = 0.7 * dR.z + 0.3 * avg;
                    }
                }
                // If only one has depth, copy to the other (mirrored)
                if (dL.confidence > 0.2 && dR.confidence < 0.1) {
                    dR.z = dL.z;
                    dR.confidence = dL.confidence * 0.5;
                    dR.method = 'symmetry';
                } else if (dR.confidence > 0.2 && dL.confidence < 0.1) {
                    dL.z = dR.z;
                    dL.confidence = dR.confidence * 0.5;
                    dL.method = 'symmetry';
                }
            }
        });
    }

    /**
     * Temporal smoothing of depth estimates.
     */
    _smoothDepths(depthMap) {
        depthMap.forEach((d, i) => {
            const hist = this.zHistory[i];
            
            if (d.confidence > 0.1) {
                // Adaptive smoothing based on confidence
                const alpha = 0.1 + d.confidence * 0.3;
                hist.velocity = 0.7 * hist.velocity + 0.3 * (d.z - hist.z);
                hist.z = d.z;
                hist.smoothed = hist.smoothed + alpha * (d.z - hist.smoothed);
                
                hist.history.push(d.z);
                if (hist.history.length > 15) hist.history.shift();
            } else {
                // Decay toward zero
                hist.smoothed *= 0.95;
                hist.velocity *= 0.9;
            }
        });
    }

    /**
     * Compute true 3D joint angles (using depth information).
     */
    _compute3DAngles(kps3D) {
        const angles = {};

        // Helper: 3D angle between vectors AB and BC
        const angle3D = (a, b, c) => {
            if (!a || !b || !c) return null;
            if (a.score < 0.2 || b.score < 0.2 || c.score < 0.2) return null;
            const ba = { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z };
            const bc = { x: c.x - b.x, y: c.y - b.y, z: c.z - b.z };
            const dot = ba.x * bc.x + ba.y * bc.y + ba.z * bc.z;
            const magBA = Math.sqrt(ba.x ** 2 + ba.y ** 2 + ba.z ** 2);
            const magBC = Math.sqrt(bc.x ** 2 + bc.y ** 2 + bc.z ** 2);
            if (magBA < 0.001 || magBC < 0.001) return null;
            const cosAngle = Math.max(-1, Math.min(1, dot / (magBA * magBC)));
            return Math.acos(cosAngle) * (180 / Math.PI);
        };

        // Key 3D angles
        angles.leftElbow = angle3D(kps3D[5], kps3D[7], kps3D[9]);
        angles.rightElbow = angle3D(kps3D[6], kps3D[8], kps3D[10]);
        angles.leftShoulder = angle3D(kps3D[7], kps3D[5], kps3D[11]);
        angles.rightShoulder = angle3D(kps3D[8], kps3D[6], kps3D[12]);
        angles.leftHip = angle3D(kps3D[5], kps3D[11], kps3D[13]);
        angles.rightHip = angle3D(kps3D[6], kps3D[12], kps3D[14]);
        angles.leftKnee = angle3D(kps3D[11], kps3D[13], kps3D[15]);
        angles.rightKnee = angle3D(kps3D[12], kps3D[14], kps3D[16]);
        angles.neck = angle3D(kps3D[0], this._midpoint3D(kps3D[5], kps3D[6]),
                              this._midpoint3D(kps3D[11], kps3D[12]));

        // Spine angle (3D — shoulder midpoint to hip midpoint vs vertical)
        const shoulderMid = this._midpoint3D(kps3D[5], kps3D[6]);
        const hipMid = this._midpoint3D(kps3D[11], kps3D[12]);
        if (shoulderMid && hipMid) {
            const spine = { x: shoulderMid.x - hipMid.x, y: shoulderMid.y - hipMid.y, z: shoulderMid.z - hipMid.z };
            const vertical = { x: 0, y: -1, z: 0 };
            const dot = spine.x * vertical.x + spine.y * vertical.y + spine.z * vertical.z;
            const mag = Math.sqrt(spine.x ** 2 + spine.y ** 2 + spine.z ** 2);
            if (mag > 0.001) {
                angles.spineVertical = Math.acos(Math.max(-1, Math.min(1, dot / mag))) * (180 / Math.PI);
            }
        }

        return angles;
    }

    _midpoint3D(a, b) {
        if (!a || !b || a.score < 0.15 || b.score < 0.15) return null;
        return {
            x: (a.x + b.x) / 2,
            y: (a.y + b.y) / 2,
            z: (a.z + b.z) / 2,
            score: Math.min(a.score, b.score),
        };
    }

    /**
     * Assess quality of the 3D estimation.
     */
    _assessQuality(kps3D, depthMap) {
        const confidentDepths = depthMap.filter(d => d.confidence > 0.2).length;
        const methodCounts = {};
        depthMap.forEach(d => {
            methodCounts[d.method] = (methodCounts[d.method] || 0) + 1;
        });

        return {
            depthCoverage: confidentDepths / 17,
            methods: methodCounts,
            isCalibrated: this.calibrated,
            personScale: this.personScale,
            reliability: confidentDepths > 8 ? 'high' : confidentDepths > 4 ? 'medium' : 'low',
        };
    }

    /**
     * Reset calibration state.
     */
    reset() {
        this.calibrated = false;
        this.refShoulderWidth = null;
        this.personScale = 1.0;
        this.frameCount = 0;
        this.zHistory.forEach(h => {
            h.z = 0;
            h.smoothed = 0;
            h.velocity = 0;
            h.history = [];
        });
    }
}

// Export
window.Pose3DEstimator = Pose3DEstimator;
window.BONE_PAIRS = BONE_PAIRS;
window.SYMMETRIC_PAIRS = SYMMETRIC_PAIRS;
