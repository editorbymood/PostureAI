/* =====================================================
   PersonTracker — Multi-Person Tracking & State Manager
   
   Assigns persistent IDs to detected people across frames
   using centroid-based nearest-neighbor matching.
   
   HUMAN VALIDATION:
   Before accepting a detection as a "person", this module
   validates that the keypoint layout matches human anatomy:
     - Minimum # of confident body keypoints
     - Head is above shoulders
     - Shoulders are above hips (if visible)
     - Body proportions are human-like
     - Overall pose score is high enough
   This prevents animals, objects, and ghost detections
   from being tracked.
   
   Each tracked person gets their own:
     - TemporalBuffer (sliding window)
     - FeatureEngine instance
     - CalibrationSystem
     - Session stats
     - Assigned UI color
   ===================================================== */

import { FeatureEngine, KP_INDEX } from './featureEngine.js';
import { TemporalBuffer, POSTURE_CLASSES } from './temporalBuffer.js';
import { CalibrationSystem } from './calibration.js';

// Color palette for up to 6 tracked people
const PERSON_COLORS = [
    { primary: '#6366f1', light: '#818cf8', rgb: '99,102,241',  name: 'Indigo' },
    { primary: '#ec4899', light: '#f472b6', rgb: '236,72,153',  name: 'Pink' },
    { primary: '#14b8a6', light: '#2dd4bf', rgb: '20,184,166',  name: 'Teal' },
    { primary: '#f59e0b', light: '#fbbf24', rgb: '245,158,11',  name: 'Amber' },
    { primary: '#8b5cf6', light: '#a78bfa', rgb: '139,92,246',  name: 'Violet' },
    { primary: '#06b6d4', light: '#22d3ee', rgb: '6,182,212',   name: 'Cyan' },
];

const MAX_PERSONS = 6;
const MATCH_DISTANCE_THRESHOLD = 150;    // px — beyond this = new person
const STALE_TIMEOUT = 2000;              // ms — remove if unseen for 2s

// ═══════════════════════════════════════════════════════
// HUMAN VALIDATION CONSTANTS
// These are the gate checks that prevent false positives
// from animals, inanimate objects, or noisy detections.
// ═══════════════════════════════════════════════════════
const HUMAN_VALIDATION = {
    MIN_POSE_SCORE:        0.20,   // MoveNet overall pose confidence
    MIN_CONFIDENT_KPS:     5,      // At least 5 keypoints above confidence threshold
    KP_CONFIDENCE:         0.30,   // Per-keypoint confidence threshold
    MIN_SHOULDER_WIDTH:    20,     // px — minimum shoulder width (rejects tiny ghost detections)
    MAX_SHOULDER_HEAD_RATIO: 4.0,  // Shoulders shouldn't be 4x wider than nose-to-shoulder distance
    MIN_SHOULDER_HEAD_RATIO: 0.15, // Head-shoulder distance shouldn't be negligibly small
    REQUIRED_BODY_PARTS: [         // Must have nose/eyes AND shoulders — core human layout
        KP_INDEX.LEFT_SHOULDER,
        KP_INDEX.RIGHT_SHOULDER,
    ],
};


export class PersonTracker {
    constructor() {
        this._nextId = 1;
        /** @type {Map<number, TrackedPerson>} */
        this.persons = new Map();
        this._usedColorIndices = new Set();
    }

    /**
     * Update tracker with new detections from this frame.
     * @param {Object[]} poses — Array of {keypoints, score} from PoseDetector
     * @returns {TrackedPerson[]} — Currently active tracked persons (humans only)
     */
    update(poses) {
        const now = Date.now();

        // ─── FILTER: Only accept validated human poses ───
        const humanPoses = poses.filter(pose => this._isHuman(pose));

        // 1. Compute centroid for each validated detection
        const detections = humanPoses.map((pose, idx) => ({
            index: idx,
            keypoints: pose.keypoints,
            score: pose.score,
            centroid: this._computeCentroid(pose.keypoints),
            matched: false,
        }));

        // 2. Match existing tracked persons to detections (greedy nearest)
        const matchedPersonIds = new Set();
        const existingPersons = [...this.persons.values()].sort((a, b) => a.id - b.id);

        // Build distance matrix
        const pairs = [];
        for (const person of existingPersons) {
            for (const det of detections) {
                if (det.matched) continue;
                const dist = this._dist(person.centroid, det.centroid);
                pairs.push({ person, det, dist });
            }
        }
        // Sort by distance — greedy assignment
        pairs.sort((a, b) => a.dist - b.dist);

        for (const { person, det, dist } of pairs) {
            if (matchedPersonIds.has(person.id) || det.matched) continue;
            if (dist < MATCH_DISTANCE_THRESHOLD) {
                // Matched!
                person.centroid = det.centroid;
                person.keypoints = det.keypoints;
                person.poseScore = det.score;
                person.lastSeen = now;
                person.framesSinceUpdate = 0;
                det.matched = true;
                matchedPersonIds.add(person.id);
            }
        }

        // 3. Create new tracked persons for unmatched detections
        for (const det of detections) {
            if (det.matched) continue;
            if (this.persons.size >= MAX_PERSONS) break;

            const colorIdx = this._nextColorIndex();
            const person = new TrackedPerson(
                this._nextId++,
                det.keypoints,
                det.centroid,
                det.score,
                colorIdx,
                now
            );
            this.persons.set(person.id, person);
            this._usedColorIndices.add(colorIdx);
        }

        // 4. Remove stale persons (not seen for STALE_TIMEOUT)
        for (const [id, person] of this.persons) {
            if (now - person.lastSeen > STALE_TIMEOUT) {
                this._usedColorIndices.delete(person.colorIndex);
                this.persons.delete(id);
            }
        }

        // Return active persons sorted by ID
        return [...this.persons.values()].sort((a, b) => a.id - b.id);
    }

    /** Get the number of actively tracked people */
    get count() {
        return this.persons.size;
    }

    /** Reset all tracking */
    reset() {
        this.persons.clear();
        this._usedColorIndices.clear();
        this._nextId = 1;
    }

    // ═══════════════════════════════════════════════════════
    // HUMAN VALIDATION — The key differentiator
    // ═══════════════════════════════════════════════════════

    /**
     * Validate that a detected pose is actually a human.
     * 
     * MoveNet can fire keypoints on random objects, animals,
     * or even patterns in furniture. This function checks:
     * 
     * 1. Minimum pose score (model's own confidence)
     * 2. Enough confident keypoints (at least 5)
     * 3. Required body parts present (shoulders)
     * 4. Anatomical layout:
     *    - Head is above shoulders
     *    - Shoulders are above hips (if visible)
     *    - Shoulder width is reasonable
     *    - Body proportions are human-like
     * 
     * @param {Object} pose - {keypoints, score}
     * @returns {boolean} True if this is likely a human
     */
    _isHuman(pose) {
        const kps = pose.keypoints;
        if (!kps || kps.length < 17) return false;

        // ── CHECK 1: Overall pose confidence ──
        if ((pose.score || 0) < HUMAN_VALIDATION.MIN_POSE_SCORE) {
            return false;
        }

        // ── CHECK 2: Minimum confident keypoints ──
        const confidentKPs = kps.filter(kp => (kp.score || 0) >= HUMAN_VALIDATION.KP_CONFIDENCE);
        if (confidentKPs.length < HUMAN_VALIDATION.MIN_CONFIDENT_KPS) {
            return false;
        }

        // ── CHECK 3: Required body parts (shoulders must be visible) ──
        for (const idx of HUMAN_VALIDATION.REQUIRED_BODY_PARTS) {
            if ((kps[idx].score || 0) < HUMAN_VALIDATION.KP_CONFIDENCE) {
                return false;
            }
        }

        // ── CHECK 4: Anatomical proportions ──
        const lS = kps[KP_INDEX.LEFT_SHOULDER];
        const rS = kps[KP_INDEX.RIGHT_SHOULDER];
        const shoulderWidth = Math.sqrt(
            (lS.x - rS.x) ** 2 + (lS.y - rS.y) ** 2
        );

        // Shoulders too close = ghost/noise detection
        if (shoulderWidth < HUMAN_VALIDATION.MIN_SHOULDER_WIDTH) {
            return false;
        }

        // ── CHECK 5: Head above shoulders (gravity check) ──
        // In humans, the nose/eyes are above the shoulders (lower Y in screen coords)
        const hasHead = (kps[KP_INDEX.NOSE].score || 0) >= HUMAN_VALIDATION.KP_CONFIDENCE;
        if (hasHead) {
            const nose = kps[KP_INDEX.NOSE];
            const midShoulderY = (lS.y + rS.y) / 2;

            // Nose should be ABOVE shoulders (lower Y value in screen coords)
            // Allow 20% of shoulderWidth tolerance (person could be slightly tilted)
            if (nose.y > midShoulderY + shoulderWidth * 0.5) {
                return false; // Head is below shoulders — not human-like
            }

            // Head-to-shoulder distance should be proportional
            const headShoulderDist = Math.abs(midShoulderY - nose.y);
            const ratio = headShoulderDist / shoulderWidth;

            // If head is essentially at the same position as shoulders, suspicious
            // (proper neck height creates some distance)
            // But also if head is impossibly far from shoulders
            if (ratio < HUMAN_VALIDATION.MIN_SHOULDER_HEAD_RATIO) {
                return false; // Head and shoulders overlapping — likely not a person
            }
            if (ratio > HUMAN_VALIDATION.MAX_SHOULDER_HEAD_RATIO) {
                return false; // Head way too far from shoulders
            }
        }

        // ── CHECK 6: Hips below shoulders (if visible) ──
        const hasHips = (kps[KP_INDEX.LEFT_HIP].score || 0) >= HUMAN_VALIDATION.KP_CONFIDENCE &&
                        (kps[KP_INDEX.RIGHT_HIP].score || 0) >= HUMAN_VALIDATION.KP_CONFIDENCE;
        if (hasHips) {
            const midHipY = (kps[KP_INDEX.LEFT_HIP].y + kps[KP_INDEX.RIGHT_HIP].y) / 2;
            const midShoulderY = (lS.y + rS.y) / 2;

            // Hips must be below shoulders (higher Y in screen coords)
            if (midHipY < midShoulderY - shoulderWidth * 0.3) {
                return false; // Hips above shoulders — anatomically impossible
            }

            // Hip width should be roughly proportional to shoulder width
            const hipWidth = Math.sqrt(
                (kps[KP_INDEX.LEFT_HIP].x - kps[KP_INDEX.RIGHT_HIP].x) ** 2 +
                (kps[KP_INDEX.LEFT_HIP].y - kps[KP_INDEX.RIGHT_HIP].y) ** 2
            );
            const shRatio = shoulderWidth / Math.max(hipWidth, 1);
            // Human shoulder-to-hip ratio is typically 0.5 to 2.5
            if (shRatio < 0.3 || shRatio > 4.0) {
                return false; // Proportions are non-human
            }
        }

        // ── CHECK 7: Average confidence of upper body ──
        // Upper body keypoints (eyes, ears, shoulders, nose) should have
        // decent average confidence for a real human detection
        const upperBodyIndices = [
            KP_INDEX.NOSE, KP_INDEX.LEFT_EYE, KP_INDEX.RIGHT_EYE,
            KP_INDEX.LEFT_SHOULDER, KP_INDEX.RIGHT_SHOULDER
        ];
        const upperBodyScores = upperBodyIndices.map(i => kps[i].score || 0);
        const avgUpperBody = upperBodyScores.reduce((a, b) => a + b, 0) / upperBodyScores.length;
        if (avgUpperBody < 0.25) {
            return false; // Upper body not confidently detected
        }

        return true; // Passed all human validation checks ✅
    }

    // ─── INTERNALS ─────────────────────────────────────

    _computeCentroid(keypoints) {
        let sumX = 0, sumY = 0, count = 0;
        for (const kp of keypoints) {
            if (kp.score > 0.2) {
                sumX += kp.x;
                sumY += kp.y;
                count++;
            }
        }
        return count > 0
            ? { x: sumX / count, y: sumY / count }
            : { x: 0, y: 0 };
    }

    _dist(a, b) {
        return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2);
    }

    _nextColorIndex() {
        for (let i = 0; i < PERSON_COLORS.length; i++) {
            if (!this._usedColorIndices.has(i)) return i;
        }
        return this.persons.size % PERSON_COLORS.length;
    }
}


/**
 * Represents a single tracked person with their own
 * feature engineering, temporal buffer, and stats.
 */
export class TrackedPerson {
    constructor(id, keypoints, centroid, poseScore, colorIndex, timestamp) {
        this.id = id;
        this.keypoints = keypoints;
        this.centroid = centroid;
        this.poseScore = poseScore;
        this.colorIndex = colorIndex;
        this.color = PERSON_COLORS[colorIndex];
        this.lastSeen = timestamp;
        this.firstSeen = timestamp;
        this.framesSinceUpdate = 0;

        // Per-person ML pipeline
        this.featureEngine = new FeatureEngine(0.25);
        this.temporal = new TemporalBuffer(30, 0.12);
        this.calibration = new CalibrationSystem(`person_${id}`);

        // Per-person state
        this.lastPrediction = null;
        this.lastExtraction = null;
        this.prevScore = null;
        this.totalFrames = 0;
        this.goodFrames = 0;
        this.alertCount = 0;
        this.scoreSum = 0;
        this.lastAlertTime = 0;
    }

    /** Process this person's keypoints through the full pipeline */
    process(calibrationOverride = null) {
        const extraction = this.featureEngine.extract(this.keypoints);
        if (!extraction.valid) return false;

        this.lastExtraction = extraction;
        this.temporal.push(extraction.features);

        // Classify
        const cal = calibrationOverride || (this.calibration.isCalibrated ? this.calibration : null);
        let prediction = this.temporal.classifyRuleBased(cal);

        // Blend personalized score
        if (this.calibration.isCalibrated) {
            const pScore = this.calibration.getPersonalizedScore(extraction.features);
            if (pScore !== null) {
                prediction.score = Math.round(prediction.score * 0.4 + pScore * 0.6);
            }
        }

        this.lastPrediction = prediction;

        // Update stats
        this.totalFrames++;
        if (prediction.classIndex === POSTURE_CLASSES.GOOD) this.goodFrames++;
        this.scoreSum += prediction.score;

        // Track alerts
        const now = Date.now();
        if (prediction.classIndex !== POSTURE_CLASSES.GOOD && now - this.lastAlertTime > 5000) {
            this.lastAlertTime = now;
            this.alertCount++;
        }

        return true;
    }

    /** Get good posture percentage */
    get goodPercentage() {
        return this.totalFrames > 0 ? Math.round((this.goodFrames / this.totalFrames) * 100) : 0;
    }

    /** Get average score */
    get averageScore() {
        return this.totalFrames > 0 ? Math.round(this.scoreSum / this.totalFrames) : 0;
    }

    /** Get duration in seconds since first seen */
    get durationSeconds() {
        return Math.round((Date.now() - this.firstSeen) / 1000);
    }
}

export { PERSON_COLORS, MAX_PERSONS };
