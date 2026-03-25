/* =====================================================
   FeatureEngine — Posture Feature Engineering Module
   
   Extracts 12 engineered features from raw keypoints.
   This is where real ML thinking happens:
   instead of feeding raw (x, y) coordinates, we compute
   meaningful geometric features that capture posture quality.
   
   Features are normalized and scale-invariant so they work
   across different camera distances and body sizes.
   ===================================================== */

/**
 * MoveNet / COCO keypoint indices
 * Both MoveNet and PoseNet use the COCO 17-keypoint topology
 */
export const KP_INDEX = {
    NOSE:            0,
    LEFT_EYE:        1,
    RIGHT_EYE:       2,
    LEFT_EAR:        3,
    RIGHT_EAR:       4,
    LEFT_SHOULDER:   5,
    RIGHT_SHOULDER:  6,
    LEFT_ELBOW:      7,
    RIGHT_ELBOW:     8,
    LEFT_WRIST:      9,
    RIGHT_WRIST:    10,
    LEFT_HIP:       11,
    RIGHT_HIP:     12,
    LEFT_KNEE:      13,
    RIGHT_KNEE:     14,
    LEFT_ANKLE:     15,
    RIGHT_ANKLE:    16,
};

/**
 * Feature names in order — used for logging, CSV export, and
 * ensuring consistency between frontend and backend
 */
export const FEATURE_NAMES = [
    'shoulder_angle',        // 0: Shoulder tilt from horizontal
    'neck_inclination',      // 1: Neck forward lean from vertical
    'torso_inclination',     // 2: Torso lean from vertical
    'hip_angle',             // 3: Hip tilt from horizontal
    'head_lateral_offset',   // 4: Head X offset, normalized
    'left_torso_ratio',      // 5: Left torso length, normalized
    'right_torso_ratio',     // 6: Right torso length, normalized
    'symmetry_score',        // 7: Left/Right body symmetry
    'head_drop_ratio',       // 8: Nose vertical drop, normalized
    'ear_alignment',         // 9: Ear-shoulder vertical alignment
    'shoulder_hip_width',    // 10: Shoulder/hip width ratio
    'avg_confidence',        // 11: Average keypoint confidence
];

export const NUM_FEATURES = FEATURE_NAMES.length;

export class FeatureEngine {
    /**
     * @param {number} minConfidence - Minimum keypoint confidence to use
     */
    constructor(minConfidence = 0.25) {
        this.minConfidence = minConfidence;
    }

    /**
     * Extract 12 engineered features from raw keypoints.
     * 
     * @param {Array} keypoints - Array of 17 keypoints, each with
     *   {x, y, score} (MoveNet) or {position: {x,y}, score} (PoseNet)
     * @returns {{ features: number[], valid: boolean, raw: Object }}
     */
    extract(keypoints) {
        const kps = this._normalizeFormat(keypoints);
        const result = {
            features: new Array(NUM_FEATURES).fill(0),
            valid: false,
            raw: {},             // Intermediate values for debugging
            keypointCount: 0,    // How many keypoints are confident
        };

        // Count confident keypoints
        const confidentKPs = kps.filter(kp => kp.score >= this.minConfidence);
        result.keypointCount = confidentKPs.length;

        // HIGH ACCURACY CHECK:
        // Must have at least 5 confident keypoints AND core upper body parts
        const hasCoreUpperBody = this._hasConfidence(kps, [
            KP_INDEX.NOSE,
            KP_INDEX.LEFT_SHOULDER,
            KP_INDEX.RIGHT_SHOULDER,
        ]);

        // Calculate average confidence for the whole pose
        const avgPoseConfidence = kps.reduce((s, kp) => s + kp.score, 0) / kps.length;

        if (!hasCoreUpperBody || result.keypointCount < 5 || avgPoseConfidence < 0.25) {
            result.valid = false;
            return result;
        }

        result.valid = true;

        // Helper: get keypoint position
        const p = (idx) => ({ x: kps[idx].x, y: kps[idx].y });
        const conf = (idx) => kps[idx].score >= this.minConfidence;

        // ── Compute reference measurements for normalization ──
        const lS = p(KP_INDEX.LEFT_SHOULDER);
        const rS = p(KP_INDEX.RIGHT_SHOULDER);
        const midShoulder = this._midpoint(lS, rS);
        const shoulderWidth = this._distance(lS, rS);

        // Estimate "body height" = distance from nose to midpoint of hips (or shoulders if hips not visible)
        const nose = p(KP_INDEX.NOSE);
        let bodyHeight = shoulderWidth * 2.5;  // Fallback estimate

        const hasHips = conf(KP_INDEX.LEFT_HIP) && conf(KP_INDEX.RIGHT_HIP);
        let midHip = null;
        if (hasHips) {
            const lH = p(KP_INDEX.LEFT_HIP);
            const rH = p(KP_INDEX.RIGHT_HIP);
            midHip = this._midpoint(lH, rH);
            bodyHeight = this._distance(nose, midHip);
        }

        // Avoid division by zero
        if (shoulderWidth < 1) return result;
        if (bodyHeight < 1) bodyHeight = shoulderWidth * 2.5;

        // ═══════════════════════════════════════════════════
        // Feature 0: SHOULDER ANGLE
        // Angle of the shoulder line from horizontal.
        // 0° = perfectly level shoulders
        // ═══════════════════════════════════════════════════
        const shoulderAngle = Math.abs(
            this._angleDeg(rS.x - lS.x, rS.y - lS.y)
        );
        result.features[0] = shoulderAngle;
        result.raw.shoulderAngle = shoulderAngle;

        // ═══════════════════════════════════════════════════
        // Feature 1: NECK INCLINATION
        // How far the head leans forward from vertical.
        // Measured as angle from the vertical axis between
        // shoulder midpoint and nose.
        // 0° = perfectly upright neck
        // ═══════════════════════════════════════════════════
        const neckVecX = nose.x - midShoulder.x;
        const neckVecY = midShoulder.y - nose.y;  // Flip Y (screen coords)
        const neckAngleFromVert = Math.abs(
            Math.atan2(neckVecX, neckVecY) * (180 / Math.PI)
        );
        result.features[1] = neckAngleFromVert;
        result.raw.neckInclination = neckAngleFromVert;

        // ═══════════════════════════════════════════════════
        // Feature 2: TORSO INCLINATION
        // Spine lean from vertical. Computed as angle of the
        // line from hip midpoint to shoulder midpoint.
        // 0° = perfectly upright torso
        // ═══════════════════════════════════════════════════
        if (midHip) {
            const torsoVecX = midShoulder.x - midHip.x;
            const torsoVecY = midHip.y - midShoulder.y;  // Flip Y
            const torsoAngle = Math.abs(
                Math.atan2(torsoVecX, torsoVecY) * (180 / Math.PI)
            );
            result.features[2] = torsoAngle;
            result.raw.torsoInclination = torsoAngle;
        }

        // ═══════════════════════════════════════════════════
        // Feature 3: HIP ANGLE
        // Tilt of the hip line from horizontal.
        // 0° = level hips
        // ═══════════════════════════════════════════════════
        if (hasHips) {
            const lH = p(KP_INDEX.LEFT_HIP);
            const rH = p(KP_INDEX.RIGHT_HIP);
            const hipAngle = Math.abs(
                this._angleDeg(rH.x - lH.x, rH.y - lH.y)
            );
            result.features[3] = hipAngle;
            result.raw.hipAngle = hipAngle;
        }

        // ═══════════════════════════════════════════════════
        // Feature 4: HEAD LATERAL OFFSET
        // How far the nose is from the shoulder center,
        // normalized by shoulder width. 
        // 0 = centered, positive = rightward
        // ═══════════════════════════════════════════════════
        const headOffset = (nose.x - midShoulder.x) / shoulderWidth;
        result.features[4] = headOffset;
        result.raw.headLateralOffset = headOffset;

        // ═══════════════════════════════════════════════════
        // Feature 5 & 6: LEFT/RIGHT TORSO LENGTH RATIO
        // Distance from each shoulder to corresponding hip,
        // normalized by body height. Asymmetry indicates lean.
        // ═══════════════════════════════════════════════════
        if (hasHips) {
            const lH = p(KP_INDEX.LEFT_HIP);
            const rH = p(KP_INDEX.RIGHT_HIP);
            result.features[5] = this._distance(lS, lH) / bodyHeight;
            result.features[6] = this._distance(rS, rH) / bodyHeight;
            result.raw.leftTorsoRatio = result.features[5];
            result.raw.rightTorsoRatio = result.features[6];
        }

        // ═══════════════════════════════════════════════════
        // Feature 7: SYMMETRY SCORE
        // Measures left/right body symmetry (0-1).
        // Uses mirrored keypoint distances from the vertical
        // center line. 1.0 = perfectly symmetric.
        // ═══════════════════════════════════════════════════
        result.features[7] = this._computeSymmetry(kps, midShoulder.x);
        result.raw.symmetryScore = result.features[7];

        // ═══════════════════════════════════════════════════
        // Feature 8: HEAD DROP RATIO
        // How far the nose has dropped below the "expected"
        // position. Expected = shoulder_midpoint_y - shoulderWidth.
        // Normalized by bodyHeight. Positive = dropped.
        // ═══════════════════════════════════════════════════
        const expectedNoseY = midShoulder.y - shoulderWidth * 0.8;
        const headDrop = (nose.y - expectedNoseY) / bodyHeight;
        result.features[8] = headDrop;
        result.raw.headDropRatio = headDrop;

        // ═══════════════════════════════════════════════════
        // Feature 9: EAR-SHOULDER ALIGNMENT
        // Angle between ear and shoulder on the visible side.
        // Indicates forward head posture.
        // ═══════════════════════════════════════════════════
        let earAlignAngle = 0;
        if (conf(KP_INDEX.LEFT_EAR)) {
            const ear = p(KP_INDEX.LEFT_EAR);
            const vecX = ear.x - lS.x;
            const vecY = lS.y - ear.y;
            earAlignAngle = Math.abs(
                Math.atan2(vecX, vecY) * (180 / Math.PI)
            );
        } else if (conf(KP_INDEX.RIGHT_EAR)) {
            const ear = p(KP_INDEX.RIGHT_EAR);
            const vecX = ear.x - rS.x;
            const vecY = rS.y - ear.y;
            earAlignAngle = Math.abs(
                Math.atan2(vecX, vecY) * (180 / Math.PI)
            );
        }
        result.features[9] = earAlignAngle;
        result.raw.earAlignment = earAlignAngle;

        // ═══════════════════════════════════════════════════
        // Feature 10: SHOULDER-HIP WIDTH RATIO
        // Ratio of shoulder width to hip width.
        // Indicates hunching (ratio decreases when shoulders
        // round forward).
        // ═══════════════════════════════════════════════════
        if (hasHips) {
            const lH = p(KP_INDEX.LEFT_HIP);
            const rH = p(KP_INDEX.RIGHT_HIP);
            const hipWidth = this._distance(lH, rH);
            result.features[10] = hipWidth > 0 ? shoulderWidth / hipWidth : 1.0;
            result.raw.shoulderHipWidthRatio = result.features[10];
        } else {
            result.features[10] = 1.0;
        }

        // ═══════════════════════════════════════════════════
        // Feature 11: AVERAGE CONFIDENCE
        // Mean confidence of all keypoints. Low confidence
        // indicates occlusion or poor lighting.
        // ═══════════════════════════════════════════════════
        const avgConf = kps.reduce((s, kp) => s + kp.score, 0) / kps.length;
        result.features[11] = avgConf;
        result.raw.avgConfidence = avgConf;

        return result;
    }

    // ─── Geometry Utilities ────────────────────────────

    /**
     * Compute angle in degrees from dx, dy components.
     * Returns angle from horizontal axis.
     */
    _angleDeg(dx, dy) {
        let angle = Math.atan2(dy, dx) * (180 / Math.PI);
        // Normalize to [-90, 90] from horizontal
        if (angle > 90) angle = 180 - angle;
        if (angle < -90) angle = -180 - angle;
        return angle;
    }

    /** Euclidean distance between two points */
    _distance(a, b) {
        return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2);
    }

    /** Midpoint of two points */
    _midpoint(a, b) {
        return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
    }

    /**
     * Compute angle at point B in triangle A-B-C (in degrees).
     * Uses dot product formula: cos(θ) = (BA·BC) / (|BA|·|BC|)
     */
    static angleBetween(a, b, c) {
        const ba = { x: a.x - b.x, y: a.y - b.y };
        const bc = { x: c.x - b.x, y: c.y - b.y };
        const dot = ba.x * bc.x + ba.y * bc.y;
        const magBA = Math.sqrt(ba.x ** 2 + ba.y ** 2);
        const magBC = Math.sqrt(bc.x ** 2 + bc.y ** 2);
        if (magBA === 0 || magBC === 0) return 0;
        const cosAngle = Math.max(-1, Math.min(1, dot / (magBA * magBC)));
        return Math.acos(cosAngle) * (180 / Math.PI);
    }

    /**
     * Compute left/right body symmetry score (0-1).
     * Compares mirrored keypoint pairs' distances from vertical center.
     */
    _computeSymmetry(kps, centerX) {
        const pairs = [
            [KP_INDEX.LEFT_SHOULDER, KP_INDEX.RIGHT_SHOULDER],
            [KP_INDEX.LEFT_HIP,      KP_INDEX.RIGHT_HIP],
            [KP_INDEX.LEFT_EYE,      KP_INDEX.RIGHT_EYE],
            [KP_INDEX.LEFT_EAR,      KP_INDEX.RIGHT_EAR],
        ];

        let totalDiff = 0;
        let validPairs = 0;

        for (const [li, ri] of pairs) {
            if (kps[li].score >= this.minConfidence &&
                kps[ri].score >= this.minConfidence) {
                const leftDist  = Math.abs(kps[li].x - centerX);
                const rightDist = Math.abs(kps[ri].x - centerX);
                const maxDist = Math.max(leftDist, rightDist, 1);
                const diff = Math.abs(leftDist - rightDist) / maxDist;
                totalDiff += diff;
                validPairs++;
            }
        }

        if (validPairs === 0) return 0.5;
        return Math.max(0, 1 - (totalDiff / validPairs));
    }

    /**
     * Check if specific keypoints meet minimum confidence.
     */
    _hasConfidence(kps, indices) {
        return indices.every(i => kps[i].score >= this.minConfidence);
    }

    /**
     * Normalize keypoint format between MoveNet and PoseNet.
     * MoveNet: { x, y, score, name }
     * PoseNet: { position: {x, y}, score, part }
     * 
     * Returns unified: { x, y, score }
     */
    _normalizeFormat(keypoints) {
        return keypoints.map(kp => {
            if (kp.position) {
                // PoseNet format
                return { x: kp.position.x, y: kp.position.y, score: kp.score };
            }
            // MoveNet format (already has x, y, score)
            return { x: kp.x, y: kp.y, score: kp.score };
        });
    }
}
