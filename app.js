/* =====================================================
   PostureAI — Main Application Script
   Real-time posture detection with PoseNet + ml5.js
   ===================================================== */

// ─── DOM References ────────────────────────────────────
const DOM = {
    loadingOverlay: document.getElementById('loading-overlay'),
    loaderBar:      document.getElementById('loader-bar'),
    startPrompt:    document.getElementById('start-prompt'),
    startButton:    document.getElementById('start-button'),
    statusDot:      document.querySelector('.status-dot'),
    statusText:     document.getElementById('status-text'),
    webcam:         document.getElementById('webcam'),
    canvas:         document.getElementById('pose-canvas'),
    postureBanner:  document.getElementById('posture-banner'),
    postureIcon:    document.getElementById('posture-icon'),
    postureMessage: document.getElementById('posture-message'),
    confBadge:      document.getElementById('confidence-badge'),
    confValue:      document.getElementById('confidence-value'),
    scoreRingFill:  document.getElementById('score-ring-fill'),
    scoreValue:     document.getElementById('score-value'),
    scoreLabel:     document.getElementById('score-label'),
    scoreTrend:     document.getElementById('score-trend'),
    metricShoulder: document.getElementById('metric-shoulder'),
    metricNeck:     document.getElementById('metric-neck'),
    metricSpine:    document.getElementById('metric-spine'),
    metricHead:     document.getElementById('metric-head'),
    barShoulder:    document.getElementById('bar-shoulder'),
    barNeck:        document.getElementById('bar-neck'),
    barSpine:       document.getElementById('bar-spine'),
    barHead:        document.getElementById('bar-head'),
    statDuration:   document.getElementById('stat-duration'),
    statGoodPct:    document.getElementById('stat-good-pct'),
    statAlerts:     document.getElementById('stat-alerts'),
    statAvgScore:   document.getElementById('stat-avg-score'),
    resetBtn:       document.getElementById('reset-btn'),
    tipsContent:    document.getElementById('tips-content'),
    alertSound:     document.getElementById('alert-sound'),
};

const ctx = DOM.canvas.getContext('2d');

// ─── Configuration ─────────────────────────────────────
const CONFIG = {
    // PoseNet options
    poseNet: {
        architecture:   'MobileNetV1',
        imageScaleFactor: 0.3,
        outputStride:   16,
        flipHorizontal: true,          // Mirror the webcam
        minConfidence:  0.2,
        maxPoseDetections: 1,
        scoreThreshold: 0.5,
        nmsRadius:      20,
        detectionType:  'single',
    },

    // Thresholds for posture classification
    posture: {
        shoulderTiltMax:    12,     // degrees — max tilt before warning
        neckAngleMin:       150,    // degrees — min angle before warning
        spineDeviationMax:  15,     // degrees — max lateral deviation
        headOffsetMax:      40,     // px — head offset from shoulder midpoint
    },

    // Skeleton drawing
    draw: {
        keypointRadius: 6,
        keypointColor:  '#6366f1',
        skeletonColor:  'rgba(129,140,248,0.5)',
        skeletonWidth:  2,
        goodColor:      '#22c55e',
        badColor:       '#ef4444',
    },

    // Scoring
    scoreSmoothing: 0.15,     // Exponential moving average factor
    alertCooldown:  5000,     // ms between bad-posture alerts
};

// ─── State ─────────────────────────────────────────────
const state = {
    poseNet:        null,
    currentPose:    null,
    isRunning:      false,
    score:          null,
    smoothedScore:  null,

    // Session stats
    sessionStart:   null,
    totalFrames:    0,
    goodFrames:     0,
    alertCount:     0,
    scoreSum:       0,
    lastAlertTime:  0,
    durationTimer:  null,

    // Previous score for trend
    prevScore:      null,
};

// ─── PoseNet Keypoint Index Map ────────────────────────
// PoseNet outputs 17 keypoints in this order:
const KP = {
    NOSE:           0,
    LEFT_EYE:       1,
    RIGHT_EYE:      2,
    LEFT_EAR:       3,
    RIGHT_EAR:      4,
    LEFT_SHOULDER:   5,
    RIGHT_SHOULDER:  6,
    LEFT_ELBOW:      7,
    RIGHT_ELBOW:     8,
    LEFT_WRIST:      9,
    RIGHT_WRIST:    10,
    LEFT_HIP:       11,
    RIGHT_HIP:      12,
    LEFT_KNEE:      13,
    RIGHT_KNEE:     14,
    LEFT_ANKLE:     15,
    RIGHT_ANKLE:    16,
};

// Skeleton connections (pairs of keypoint indices)
const SKELETON_CONNECTIONS = [
    [KP.LEFT_SHOULDER, KP.RIGHT_SHOULDER],
    [KP.LEFT_SHOULDER, KP.LEFT_ELBOW],
    [KP.LEFT_ELBOW,    KP.LEFT_WRIST],
    [KP.RIGHT_SHOULDER,KP.RIGHT_ELBOW],
    [KP.RIGHT_ELBOW,   KP.RIGHT_WRIST],
    [KP.LEFT_SHOULDER, KP.LEFT_HIP],
    [KP.RIGHT_SHOULDER,KP.RIGHT_HIP],
    [KP.LEFT_HIP,      KP.RIGHT_HIP],
    [KP.LEFT_HIP,      KP.LEFT_KNEE],
    [KP.LEFT_KNEE,     KP.LEFT_ANKLE],
    [KP.RIGHT_HIP,     KP.RIGHT_KNEE],
    [KP.RIGHT_KNEE,    KP.RIGHT_ANKLE],
    [KP.NOSE,          KP.LEFT_EYE],
    [KP.NOSE,          KP.RIGHT_EYE],
    [KP.LEFT_EYE,      KP.LEFT_EAR],
    [KP.RIGHT_EYE,     KP.RIGHT_EAR],
    [KP.NOSE,          KP.LEFT_SHOULDER],
    [KP.NOSE,          KP.RIGHT_SHOULDER],
];


/* =======================================================
   1. WEBCAM SETUP
   ======================================================= */

/**
 * Request webcam access and pipe the stream into the <video> element.
 */
async function setupCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480, facingMode: 'user' },
            audio: false,
        });
        DOM.webcam.srcObject = stream;

        return new Promise(resolve => {
            DOM.webcam.onloadedmetadata = () => {
                // Match canvas to video dimensions
                DOM.canvas.width  = DOM.webcam.videoWidth;
                DOM.canvas.height = DOM.webcam.videoHeight;
                DOM.webcam.classList.add('active');
                resolve();
            };
        });
    } catch (err) {
        setStatus('Camera denied', 'error');
        DOM.postureMessage.textContent = 'Camera access required';
        DOM.postureIcon.textContent     = '🚫';
        throw err;
    }
}


/* =======================================================
   2. POSENET MODEL LOADING
   ======================================================= */

/**
 * Load PoseNet through ml5.js.
 * Returns a promise that resolves when the model is ready.
 */
function loadPoseNet() {
    return new Promise((resolve, reject) => {
        simulateLoading();

        state.poseNet = ml5.poseNet(
            DOM.webcam,
            CONFIG.poseNet,
            () => {
                console.log('✅ PoseNet model loaded');
                resolve();
            }
        );

        // Attach the pose event listener
        state.poseNet.on('pose', onPoseResults);
    });
}

/** Fake progress bar animation while loading */
function simulateLoading() {
    let progress = 0;
    const interval = setInterval(() => {
        progress += Math.random() * 12;
        if (progress > 90) progress = 90;     // cap until real load
        DOM.loaderBar.style.width = progress + '%';
    }, 300);

    // Store interval id so we can clear it later
    state._loadingInterval = interval;
}


/* =======================================================
   3. POSE DETECTION CALLBACK
   ======================================================= */

/**
 * Called each time PoseNet returns results.
 * @param {Array} results — array of detected poses
 */
function onPoseResults(results) {
    if (!results || results.length === 0) {
        state.currentPose = null;
        return;
    }

    // We only care about the first (single) pose
    state.currentPose = results[0].pose;
}


/* =======================================================
   4. DRAWING FUNCTIONS
   ======================================================= */

/**
 * Main render loop — clears canvas, draws keypoints/skeleton,
 * runs posture analysis, and updates UI.
 */
function renderLoop() {
    if (!state.isRunning) return;

    // Clear canvas
    ctx.clearRect(0, 0, DOM.canvas.width, DOM.canvas.height);

    if (state.currentPose) {
        const kps = state.currentPose.keypoints;
        const posture = analyzePosture(kps);

        // Draw skeleton & keypoints
        drawSkeleton(kps, posture);
        drawKeypoints(kps, posture);

        // Update all UI panels
        updatePostureBanner(posture);
        updateScoreCard(posture.score);
        updateMetrics(posture);
        updateConfidence(kps);
        updateSessionStats(posture);
        updateTips(posture);
    }

    requestAnimationFrame(renderLoop);
}

/**
 * Draw circles at each detected keypoint.
 */
function drawKeypoints(keypoints, posture) {
    const color = posture.isGood ? CONFIG.draw.goodColor : CONFIG.draw.badColor;

    keypoints.forEach(kp => {
        if (kp.score > CONFIG.poseNet.minConfidence) {
            ctx.beginPath();
            ctx.arc(kp.position.x, kp.position.y, CONFIG.draw.keypointRadius, 0, Math.PI * 2);
            ctx.fillStyle = color;
            ctx.shadowColor = color;
            ctx.shadowBlur = 10;
            ctx.fill();
            ctx.shadowBlur = 0;

            // Outline
            ctx.strokeStyle = 'rgba(255,255,255,0.3)';
            ctx.lineWidth   = 1.5;
            ctx.stroke();
        }
    });
}

/**
 * Draw lines connecting keypoints to form the skeleton.
 */
function drawSkeleton(keypoints, posture) {
    const color = posture.isGood
        ? 'rgba(34,197,94,0.45)'
        : 'rgba(239,68,68,0.45)';

    SKELETON_CONNECTIONS.forEach(([iA, iB]) => {
        const a = keypoints[iA];
        const b = keypoints[iB];

        if (a.score > CONFIG.poseNet.minConfidence &&
            b.score > CONFIG.poseNet.minConfidence) {
            ctx.beginPath();
            ctx.moveTo(a.position.x, a.position.y);
            ctx.lineTo(b.position.x, b.position.y);
            ctx.strokeStyle = color;
            ctx.lineWidth   = CONFIG.draw.skeletonWidth;
            ctx.stroke();
        }
    });
}


/* =======================================================
   5. POSTURE ANALYSIS LOGIC 🧠
   ======================================================= */

/**
 * Analyze keypoints to determine posture quality.
 * Returns an object with detailed metrics and an overall score (0-100).
 */
function analyzePosture(keypoints) {
    const get = idx => keypoints[idx];
    const conf = idx => get(idx).score > CONFIG.poseNet.minConfidence;

    const result = {
        shoulderTilt: null,
        neckAngle:    null,
        spineAngle:   null,
        headOffset:   null,
        score:        0,
        isGood:       true,
        issues:       [],
    };

    // --- Shoulder Tilt ---
    // Measure the angle of the line connecting both shoulders relative to horizontal
    if (conf(KP.LEFT_SHOULDER) && conf(KP.RIGHT_SHOULDER)) {
        const ls = get(KP.LEFT_SHOULDER).position;
        const rs = get(KP.RIGHT_SHOULDER).position;
        const dy = rs.y - ls.y;
        const dx = rs.x - ls.x;
        result.shoulderTilt = Math.abs(Math.atan2(dy, dx) * (180 / Math.PI));
        // Normalize: 0° is perfectly horizontal
        result.shoulderTilt = Math.abs(result.shoulderTilt);
        if (result.shoulderTilt > 90) result.shoulderTilt = 180 - result.shoulderTilt;
    }

    // --- Neck Angle ---
    // Angle at the nose between left-shoulder→nose and right-shoulder→nose
    if (conf(KP.NOSE) && conf(KP.LEFT_SHOULDER) && conf(KP.RIGHT_SHOULDER)) {
        const nose = get(KP.NOSE).position;
        const ls   = get(KP.LEFT_SHOULDER).position;
        const rs   = get(KP.RIGHT_SHOULDER).position;
        const midShoulder = { x: (ls.x + rs.x) / 2, y: (ls.y + rs.y) / 2 };

        // Angle between vertical line from midShoulder and line to nose
        const vecX = nose.x - midShoulder.x;
        const vecY = nose.y - midShoulder.y;   // Usually negative (nose is above)
        // Angle from vertical axis (0° = perfectly upright)
        const angleFromVertical = Math.abs(Math.atan2(vecX, -vecY) * (180 / Math.PI));
        // We want "neck angle" where 180° is perfectly upright → subtract from 180
        result.neckAngle = 180 - angleFromVertical;
    }

    // --- Spine Alignment ---
    // Angle of the line from hip midpoint to shoulder midpoint relative to vertical
    if (conf(KP.LEFT_SHOULDER) && conf(KP.RIGHT_SHOULDER) &&
        conf(KP.LEFT_HIP) && conf(KP.RIGHT_HIP)) {
        const ls = get(KP.LEFT_SHOULDER).position;
        const rs = get(KP.RIGHT_SHOULDER).position;
        const lh = get(KP.LEFT_HIP).position;
        const rh = get(KP.RIGHT_HIP).position;
        const midS = { x: (ls.x + rs.x) / 2, y: (ls.y + rs.y) / 2 };
        const midH = { x: (lh.x + rh.x) / 2, y: (lh.y + rh.y) / 2 };
        // Angle from vertical
        const dx = midS.x - midH.x;
        const dy = midS.y - midH.y;  // Typically negative
        result.spineAngle = Math.abs(Math.atan2(dx, -dy) * (180 / Math.PI));
    }

    // --- Head Position (lateral offset) ---
    // How far the nose is from the midpoint of the shoulders (in pixels)
    if (conf(KP.NOSE) && conf(KP.LEFT_SHOULDER) && conf(KP.RIGHT_SHOULDER)) {
        const nose = get(KP.NOSE).position;
        const ls   = get(KP.LEFT_SHOULDER).position;
        const rs   = get(KP.RIGHT_SHOULDER).position;
        const midX = (ls.x + rs.x) / 2;
        result.headOffset = Math.abs(nose.x - midX);
    }

    // --- Calculate Score ---
    let penalties = 0;
    const maxPenalty = 100;

    // Shoulder tilt penalty
    if (result.shoulderTilt !== null) {
        const excess = Math.max(0, result.shoulderTilt - CONFIG.posture.shoulderTiltMax);
        const penalty = Math.min(30, excess * 2);
        penalties += penalty;
        if (excess > 0) result.issues.push('Uneven shoulders');
    }

    // Neck angle penalty
    if (result.neckAngle !== null) {
        const deficit = Math.max(0, CONFIG.posture.neckAngleMin - result.neckAngle);
        const penalty = Math.min(30, deficit * 1.5);
        penalties += penalty;
        if (deficit > 0) result.issues.push('Neck tilted forward');
    }

    // Spine deviation penalty
    if (result.spineAngle !== null) {
        const excess = Math.max(0, result.spineAngle - CONFIG.posture.spineDeviationMax);
        const penalty = Math.min(25, excess * 2);
        penalties += penalty;
        if (excess > 0) result.issues.push('Spine not straight');
    }

    // Head offset penalty
    if (result.headOffset !== null) {
        const excess = Math.max(0, result.headOffset - CONFIG.posture.headOffsetMax);
        const penalty = Math.min(15, excess * 0.5);
        penalties += penalty;
        if (excess > 0) result.issues.push('Head tilted sideways');
    }

    result.score  = Math.max(0, Math.round(100 - Math.min(penalties, maxPenalty)));
    result.isGood = result.score >= 65;

    // Smooth the score
    if (state.smoothedScore === null) {
        state.smoothedScore = result.score;
    } else {
        state.smoothedScore += CONFIG.scoreSmoothing * (result.score - state.smoothedScore);
    }
    result.score = Math.round(state.smoothedScore);

    return result;
}


/* =======================================================
   6. UI UPDATE FUNCTIONS
   ======================================================= */

/** Update the posture banner overlay */
function updatePostureBanner(posture) {
    DOM.postureBanner.classList.remove('good', 'bad');

    if (posture.isGood) {
        DOM.postureBanner.classList.add('good');
        DOM.postureIcon.textContent    = '✅';
        DOM.postureMessage.textContent = 'Good posture — keep it up!';
    } else {
        DOM.postureBanner.classList.add('bad');
        DOM.postureIcon.textContent    = '⚠️';
        const issue = posture.issues[0] || 'Adjust your posture';
        DOM.postureMessage.textContent = issue;

        // Alert cooldown
        const now = Date.now();
        if (now - state.lastAlertTime > CONFIG.alertCooldown) {
            state.lastAlertTime = now;
            state.alertCount++;
            DOM.statAlerts.textContent = state.alertCount;
            playAlert();
        }
    }
}

/** Update the circular score ring and value */
function updateScoreCard(score) {
    const circumference = 2 * Math.PI * 52;   // r=52 from SVG
    const offset = circumference * (1 - score / 100);
    DOM.scoreRingFill.style.strokeDashoffset = offset;

    DOM.scoreValue.textContent = score;

    // Color based on score
    let color, label;
    if (score >= 80) {
        color = '#22c55e'; label = 'Excellent';
    } else if (score >= 65) {
        color = '#84cc16'; label = 'Good';
    } else if (score >= 45) {
        color = '#f59e0b'; label = 'Fair';
    } else {
        color = '#ef4444'; label = 'Poor';
    }

    DOM.scoreRingFill.style.stroke = color;
    DOM.scoreValue.style.color     = color;
    DOM.scoreLabel.textContent     = label;

    // Trend badge
    if (state.prevScore !== null) {
        const diff = score - state.prevScore;
        if (diff > 2) {
            DOM.scoreTrend.textContent = '▲ Improving';
            DOM.scoreTrend.style.color = '#22c55e';
        } else if (diff < -2) {
            DOM.scoreTrend.textContent = '▼ Declining';
            DOM.scoreTrend.style.color = '#ef4444';
        } else {
            DOM.scoreTrend.textContent = '● Stable';
            DOM.scoreTrend.style.color = '#f59e0b';
        }
    }
    state.prevScore = score;
}

/** Update the body metrics panel */
function updateMetrics(posture) {
    // Helper: set value, bar width, and bar color class
    function setMetric(valueEl, barEl, value, unit, thresholds) {
        if (value === null) {
            valueEl.textContent = '--' + unit;
            barEl.style.width   = '0%';
            return;
        }
        valueEl.textContent = Math.round(value) + unit;

        // Normalize to 0-100%
        const pct = Math.min(100, (value / thresholds.max) * 100);
        barEl.style.width = pct + '%';

        barEl.classList.remove('good', 'warning', 'bad');
        if (value <= thresholds.good) barEl.classList.add('good');
        else if (value <= thresholds.warn) barEl.classList.add('warning');
        else barEl.classList.add('bad');
    }

    setMetric(DOM.metricShoulder, DOM.barShoulder, posture.shoulderTilt, '°',
        { max: 30, good: CONFIG.posture.shoulderTiltMax, warn: 20 });

    // For neck angle, higher is better, so invert the logic
    if (posture.neckAngle !== null) {
        DOM.metricNeck.textContent = Math.round(posture.neckAngle) + '°';
        const pct = Math.min(100, (posture.neckAngle / 180) * 100);
        DOM.barNeck.style.width = pct + '%';
        DOM.barNeck.classList.remove('good', 'warning', 'bad');
        if (posture.neckAngle >= CONFIG.posture.neckAngleMin) DOM.barNeck.classList.add('good');
        else if (posture.neckAngle >= 130) DOM.barNeck.classList.add('warning');
        else DOM.barNeck.classList.add('bad');
    }

    setMetric(DOM.metricSpine, DOM.barSpine, posture.spineAngle, '°',
        { max: 40, good: CONFIG.posture.spineDeviationMax, warn: 25 });

    setMetric(DOM.metricHead, DOM.barHead, posture.headOffset, 'px',
        { max: 80, good: CONFIG.posture.headOffsetMax, warn: 55 });
}

/** Update the confidence badge */
function updateConfidence(keypoints) {
    const validKps = keypoints.filter(kp => kp.score > CONFIG.poseNet.minConfidence);
    const avgConf  = validKps.reduce((s, kp) => s + kp.score, 0) / (validKps.length || 1);
    DOM.confValue.textContent = Math.round(avgConf * 100) + '%';
    DOM.confBadge.classList.add('visible');
}

/** Update session statistics */
function updateSessionStats(posture) {
    state.totalFrames++;
    if (posture.isGood) state.goodFrames++;
    state.scoreSum += posture.score;

    // Good posture %
    const goodPct = Math.round((state.goodFrames / state.totalFrames) * 100);
    DOM.statGoodPct.textContent = goodPct + '%';

    // Average score
    const avgScore = Math.round(state.scoreSum / state.totalFrames);
    DOM.statAvgScore.textContent = avgScore;
}

/** Update the tips card */
function updateTips(posture) {
    let html = '';

    if (posture.isGood) {
        html = `
            <div class="tip-item">
                <span>✅</span>
                <span>Great job! Your posture looks good. Keep your shoulders relaxed and back straight.</span>
            </div>`;
    } else {
        posture.issues.forEach(issue => {
            let tip = '';
            let cls = 'warning-tip';
            switch (issue) {
                case 'Uneven shoulders':
                    tip = 'Try to level your shoulders. Relax both sides equally and avoid leaning.';
                    break;
                case 'Neck tilted forward':
                    tip = 'Pull your chin back slightly. Imagine a string pulling the top of your head upward.';
                    cls = 'danger-tip';
                    break;
                case 'Spine not straight':
                    tip = 'Sit up straight. Engage your core muscles and align your spine.';
                    cls = 'danger-tip';
                    break;
                case 'Head tilted sideways':
                    tip = 'Center your head over your shoulders. Avoid tilting to one side.';
                    break;
                default:
                    tip = 'Adjust your posture and try to sit upright.';
            }
            html += `<div class="tip-item ${cls}"><span>💡</span><span>${tip}</span></div>`;
        });
    }

    DOM.tipsContent.innerHTML = html;
}

/** Play a subtle alert sound */
function playAlert() {
    try {
        DOM.alertSound.currentTime = 0;
        DOM.alertSound.volume = 0.3;
        DOM.alertSound.play().catch(() => {});
    } catch (e) { /* silent fail */ }
}

/** Update status indicator */
function setStatus(text, type) {
    DOM.statusText.textContent = text;
    DOM.statusDot.classList.remove('active', 'error');
    if (type === 'active') DOM.statusDot.classList.add('active');
    if (type === 'error')  DOM.statusDot.classList.add('error');
}

/** Format seconds to MM:SS */
function formatDuration(seconds) {
    const m = Math.floor(seconds / 60).toString().padStart(2, '0');
    const s = (seconds % 60).toString().padStart(2, '0');
    return `${m}:${s}`;
}


/* =======================================================
   7. SESSION MANAGEMENT
   ======================================================= */

function startSession() {
    state.sessionStart = Date.now();
    state.totalFrames  = 0;
    state.goodFrames   = 0;
    state.alertCount   = 0;
    state.scoreSum     = 0;
    state.smoothedScore = null;
    state.prevScore    = null;

    DOM.statAlerts.textContent  = '0';
    DOM.statGoodPct.textContent = '--%';
    DOM.statAvgScore.textContent = '--';

    // Duration timer
    let elapsed = 0;
    state.durationTimer = setInterval(() => {
        elapsed++;
        DOM.statDuration.textContent = formatDuration(elapsed);
    }, 1000);
}

function resetSession() {
    if (state.durationTimer) clearInterval(state.durationTimer);
    DOM.statDuration.textContent = '00:00';
    startSession();
}


/* =======================================================
   8. INITIALIZATION
   ======================================================= */

DOM.startButton.addEventListener('click', async () => {
    DOM.startPrompt.classList.add('hidden');
    setStatus('Starting camera...', '');

    try {
        await setupCamera();
        setStatus('Loading model...', '');
        await loadPoseNet();

        // Clear loading
        clearInterval(state._loadingInterval);
        DOM.loaderBar.style.width = '100%';
        setTimeout(() => {
            DOM.loadingOverlay.classList.add('hidden');
        }, 500);

        setStatus('Detecting', 'active');
        state.isRunning = true;
        startSession();
        renderLoop();
    } catch (err) {
        console.error('Initialization failed:', err);
        setStatus('Error — see console', 'error');
        DOM.loadingOverlay.classList.add('hidden');
    }
});

DOM.resetBtn.addEventListener('click', resetSession);

// Hide the loading overlay initially (show start prompt instead)
DOM.loadingOverlay.classList.add('hidden');
