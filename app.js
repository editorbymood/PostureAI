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

    // Action Intelligence
    actionStatus:   document.getElementById('action-status'),
    actionIcon:     document.getElementById('action-icon'),
    actionLabel:    document.getElementById('action-label'),
    actionConfPct:  document.getElementById('action-conf-pct'),
    actionConfRing: document.getElementById('action-conf-ring'),

    // Rep Counter
    repCard:        document.getElementById('rep-card'),
    repCountText:   document.getElementById('rep-count'),
    repPhase:       document.getElementById('rep-phase'),
    repProgress:    document.getElementById('rep-progress'),

    // 3D/Vision
    depthReliability:document.getElementById('depth-reliability'),
    depthCoverage:   document.getElementById('depth-coverage'),
    personScale:     document.getElementById('person-scale'),
    bodySymmetry:    document.getElementById('body-symmetry'),
    
    // Original trackers
    trackActive:    document.getElementById('track-active'),
    trackConfirmed: document.getElementById('track-confirmed'),
    trackLost:      document.getElementById('track-lost'),
    trackArchived:  document.getElementById('track-archived'),
    personList:     document.getElementById('person-list'),
    temporalStability: document.getElementById('temporal-stability'),
    barStability:   document.getElementById('bar-stability'),
    temporalTrend:  document.getElementById('temporal-trend'),
    temporalPredicted: document.getElementById('temporal-predicted'),
};

const ctx = DOM.canvas.getContext('2d');

// ─── Configuration ─────────────────────────────────────
const CONFIG = {
    // PoseNet options
    poseNet: {
        architecture:   'MobileNetV1',
        imageScaleFactor: 0.3,
        outputStride:   16,
        flipHorizontal: true,          // ml5.js PoseNet needs this to mirror keypoints
        minConfidence:  0.1,
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
        keypointRadius: 7,
        keypointColor:  '#6366f1',
        skeletonColor:  'rgba(129,140,248,0.5)',
        skeletonWidth:  2.5,
        goodColor:      '#22c55e',
        badColor:       '#ef4444',
        warningColor:   '#f59e0b',
        labelFont:      '600 10px "Inter", "SF Pro", system-ui, sans-serif',
        hudFont:        '500 11px "JetBrains Mono", "Fira Code", monospace',
    },

    // Scoring
    scoreSmoothing: 0.15,     // Exponential moving average factor
    alertCooldown:  5000,     // ms between bad-posture alerts
};

// ─── State ─────────────────────────────────────────────
const state = {
    poseNet:        null,      // Legacy PoseNet (fallback)
    detector:       null,      // MoveNet MultiPose detector
    tracker:        null,      // ByteTrack-inspired tracker
    temporalEngines: new Map(), // Per-person temporal engines (trackId → TemporalEngine)
    activeTracks:   [],        // Currently tracked persons
    currentPose:    null,      // Primary person's pose (for backward compat)
    isRunning:      false,
    useMoveNet:     false,     // Whether MoveNet loaded successfully
    score:          null,
    smoothedScore:  null,

    // Session stats
    sessionStart:   null,
    totalFrames:    0,
    goodFrames:     0,
    alertCount:     0,
    scoreSum:       0,
    intelligenceEngines: new Map(), // (trackId → IntelligenceEngine)
    lastAlertTime:  0,
    startTime:      null,
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
                // CRITICAL: Set video element width/height attributes explicitly.
                // ml5.js PoseNet reads these to compute keypoint pixel coordinates.
                // If they are 0 or unset, all keypoints will have (x:0, y:0).
                DOM.webcam.width  = DOM.webcam.videoWidth;
                DOM.webcam.height = DOM.webcam.videoHeight;

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
   2. MODEL LOADING — MoveNet MultiPose (primary) + PoseNet (fallback)
   ======================================================= */

/**
 * Attempt to load MoveNet MultiPose Lightning.
 * Falls back to ml5 PoseNet if TF.js libraries aren't available.
 */
async function loadDetector() {
    simulateLoading();

    // Try MoveNet MultiPose first
    try {
        if (typeof poseDetection !== 'undefined' && typeof tf !== 'undefined') {
            await tf.ready();
            console.log('🧠 TF.js backend:', tf.getBackend());

            const model = poseDetection.SupportedModels.MoveNet;
            state.detector = await poseDetection.createDetector(model, {
                modelType: poseDetection.movenet.modelType.MULTIPOSE_LIGHTNING,
                enableTracking: false, // We use our own ByteTrack tracker
                minPoseScore: 0.1,     // Lower threshold for varied poses
                maxPoses: 6,
            });

            state.useMoveNet = true;
            console.log('✅ MoveNet MultiPose Lightning loaded');

            // Initialize ByteTrack tracker
            state.tracker = new MultiPersonTracker({
                iouThreshHigh: 0.3,
                iouThreshLow: 0.1,
                reIdThresh: 0.55,
                maxDeadAge: 90,
                maxTracks: 8,
            });
            console.log('✅ ByteTrack tracker initialized');
            return;
        }
    } catch (err) {
        console.warn('⚠️ MoveNet failed, falling back to PoseNet:', err.message);
    }

    // Fallback to ml5 PoseNet
    return new Promise((resolve, reject) => {
        state.poseNet = ml5.poseNet(
            DOM.webcam,
            CONFIG.poseNet,
            () => {
                console.log('✅ PoseNet model loaded (fallback)');
                state.useMoveNet = false;
                // Still create tracker for single-person temporal smoothing
                state.tracker = new MultiPersonTracker({ maxTracks: 1 });
                resolve();
            }
        );
        state.poseNet.on('pose', onPoseResults);
    });
}

/** Fake progress bar animation while loading */
function simulateLoading() {
    let progress = 0;
    const interval = setInterval(() => {
        progress += Math.random() * 12;
        if (progress > 90) progress = 90;
        DOM.loaderBar.style.width = progress + '%';
    }, 300);
    state._loadingInterval = interval;
}


/* =======================================================
   3. DETECTION LOOP + TRACKING PIPELINE
   ======================================================= */

/**
 * MoveNet detection loop — runs async, feeds into ByteTrack tracker.
 */
async function detectLoop() {
    if (!state.isRunning) return;

    try {
        if (state.useMoveNet && state.detector) {
            const poses = await state.detector.estimatePoses(DOM.webcam, {
                flipHorizontal: true,
            });

            // Convert MoveNet format to our tracker format
            const detections = poses.map(pose => ({
                keypoints: pose.keypoints.map(kp => ({
                    x: kp.x,
                    y: kp.y,
                    score: kp.score,
                    name: kp.name || '',
                })),
                score: pose.score || pose.keypoints.reduce((s, k) => s + k.score, 0) / pose.keypoints.length,
            }));

            // Feed into ByteTrack tracker
            state.activeTracks = state.tracker.update(detections);

            // Run temporal smoothing for each tracked person
            state.activeTracks.forEach(track => {
                if (!state.temporalEngines.has(track.id)) {
                    state.temporalEngines.set(track.id, new TemporalEngine({
                        windowSize: 30,
                        smoothingBase: 0.35,
                        predictionHorizon: 15,
                    }));
                }
                const engine = state.temporalEngines.get(track.id);

                // Convert track keypoints to the format temporal engine expects
                const rawKPs = track.keypoints.map(kp => ({
                    x: kp.x,
                    y: kp.y,
                    score: kp.score,
                    name: kp.name || '',
                }));

                // Apply temporal smoothing
                track._smoothedKeypoints = engine.processFrame(rawKPs);
                track._temporalEngine = engine;
            });

            // Cleanup temporal engines for dead tracks
            const activeIds = new Set(state.activeTracks.map(t => t.id));
            for (const [id] of state.temporalEngines) {
                if (!activeIds.has(id)) {
                    // Keep for a bit in case of re-id, then delete
                    const engine = state.temporalEngines.get(id);
                    if (engine.frameCount > 0 && engine.postureBuffer.length === 0) {
                        state.temporalEngines.delete(id);
                    }
                }
            }

            // Set primary person for backward-compat UI
            if (state.activeTracks.length > 0) {
                const primary = state.activeTracks[0];
                state.currentPose = {
                    keypoints: (primary._smoothedKeypoints || primary.keypoints).map((kp, i) => ({
                        position: { x: kp.x, y: kp.y },
                        score: kp.score,
                        part: KP_NAMES[i] || '',
                    })),
                };
            } else {
                state.currentPose = null;
            }
        }
    } catch (err) {
        // Silently handle detection errors to keep loop running
        if (err.message && !err.message.includes('disposed')) {
            console.warn('Detection error:', err.message);
        }
    }

    requestAnimationFrame(detectLoop);
}

// Keypoint name lookup (PoseNet order)
const KP_NAMES = [
    'nose','leftEye','rightEye','leftEar','rightEar',
    'leftShoulder','rightShoulder','leftElbow','rightElbow',
    'leftWrist','rightWrist','leftHip','rightHip',
    'leftKnee','rightKnee','leftAnkle','rightAnkle',
];

/**
 * PoseNet fallback callback — wraps single-person detection into tracker.
 */
function onPoseResults(results) {
    if (!results || results.length === 0) {
        state.activeTracks = state.tracker ? state.tracker.update([]) : [];
        state.currentPose = null;
        return;
    }

    // Convert ml5 PoseNet format to tracker format
    const pose = results[0].pose;
    const detection = {
        keypoints: pose.keypoints.map(kp => ({
            x: kp.position.x,
            y: kp.position.y,
            score: kp.score,
            name: kp.part || '',
        })),
        score: pose.score || 0,
    };

    state.activeTracks = state.tracker.update([detection]);

    // Apply temporal smoothing
    state.activeTracks.forEach(track => {
        if (!state.temporalEngines.has(track.id)) {
            state.temporalEngines.set(track.id, new TemporalEngine());
        }
        const engine = state.temporalEngines.get(track.id);
        track._smoothedKeypoints = engine.processFrame(
            track.keypoints.map(kp => ({ x: kp.x, y: kp.y, score: kp.score, name: kp.name || '' }))
        );
        track._temporalEngine = engine;
    });

    // Backward compat
    state.currentPose = results[0].pose;
}


/* =======================================================
   4. DRAWING FUNCTIONS
   ======================================================= */

/**
 * Main render loop — clears canvas, draws all tracked persons,
 * runs per-person posture analysis with temporal smoothing.
 */
function renderLoop() {
    if (!state.isRunning) return;

    // Clear canvas
    ctx.clearRect(0, 0, DOM.canvas.width, DOM.canvas.height);

    const tracks = state.activeTracks || [];

    if (tracks.length > 0) {
        // Draw each tracked person
        tracks.forEach((track, idx) => {
            const smoothed = track._smoothedKeypoints || track.keypoints;

            // Convert to the format analyzePosture expects
            const kps = smoothed.map((kp, i) => ({
                position: { x: kp.x, y: kp.y },
                score: kp.score,
                part: KP_NAMES[i] || '',
            }));

            const posture = analyzePosture(kps);

            // Feed posture into temporal engine for smoothed classification
            if (track._temporalEngine) {
                const temporalResult = track._temporalEngine.addPostureFrame(
                    posture.score, posture.isGood, posture.issues
                );
                if (temporalResult) {
                    posture.score = temporalResult.score;
                    posture.isGood = temporalResult.isGood;
                    posture.issues = temporalResult.issues;
                    posture._stability = temporalResult.stability;
                    posture._temporalConfidence = temporalResult.confidence;
                    posture._trend = temporalResult.trend;
                }
            }

            // ── Intelligence Layer Integration ──
            if (!state.intelligenceEngines.has(track.id)) {
                state.intelligenceEngines.set(track.id, new IntelligenceEngine());
            }
            const intel = state.intelligenceEngines.get(track.id);
            const intelOutput = intel.process(smoothed);
            track._intelligence = intelOutput;

            // Store posture on track for per-person cards
            track.lastPosture = posture;
            track.postureScore = posture.score;
            if (posture.isGood) track.goodFrames++;
            track.totalAnalyzed++;

            // Draw skeleton & keypoints
            drawSkeleton(kps, { ...posture, _trackColor: track.color?.primary, _trackId: track.id, _isOccluded: track.isOccluded });
            drawKeypoints(kps, { ...posture, _trackColor: track.color?.primary, _isOccluded: track.isOccluded });
            drawTrackBBox(track, posture);
            drawIntelligence(track, intelOutput);
            
            // Primary person (idx 0) → update advanced sidebar UI
            if (idx === 0) {
                updateUI(posture, intelOutput);
            }
        });

        // Draw multi-person data HUD (focused on primary)
        const primaryTrack = tracks[0];
        const primaryKps = primaryTrack._smoothedKeypoints || primaryTrack.keypoints;
        const primaryPosture = primaryTrack.lastPosture || { score: 0, isGood: true };
        drawDataHUD(
            primaryKps.map((kp, i) => ({ score: kp.score, position: { x: kp.x, y: kp.y } })),
            primaryPosture
        );

        // Update tracking panels
        updateTrackingPanel();
        updateTemporalPanel(primaryTrack);

    } else if (state.currentPose) {
        // Fallback for PoseNet (backward compat)
        const kps = state.currentPose.keypoints;
        const posture = analyzePosture(kps);
        drawSkeleton(kps, posture);
        drawKeypoints(kps, posture);
        drawDataHUD(kps, posture);
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
 * Draw bounding box and ID label for a tracked person.
 */
function drawTrackBBox(track, posture) {
    const bbox = track.predictedBBox;
    if (!bbox || bbox.width < 5) return;

    const color = track.color?.primary || '#6366f1';
    const isOccluded = track.isOccluded;

    // Bounding box
    ctx.beginPath();
    if (isOccluded) ctx.setLineDash([6, 4]);
    ctx.rect(bbox.x, bbox.y, bbox.width, bbox.height);
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.globalAlpha = isOccluded ? 0.4 : 0.7;
    ctx.stroke();
    ctx.globalAlpha = 1;
    ctx.setLineDash([]);

    // ID label background
    const label = `#${track.id} ${posture.isGood ? '✓' : '✗'} ${posture.score}`;
    ctx.font = '600 12px "Inter", sans-serif';
    const tw = ctx.measureText(label).width + 12;
    const lh = 20;
    const lx = bbox.x;
    const ly = bbox.y - lh - 2;

    ctx.fillStyle = color;
    ctx.globalAlpha = 0.85;
    ctx.beginPath();
    const rr = 4;
    ctx.moveTo(lx + rr, ly);
    ctx.arcTo(lx + tw, ly, lx + tw, ly + lh, rr);
    ctx.arcTo(lx + tw, ly + lh, lx, ly + lh, rr);
    ctx.arcTo(lx, ly + lh, lx, ly, rr);
    ctx.arcTo(lx, ly, lx + tw, ly, rr);
    ctx.closePath();
    ctx.fill();
    ctx.globalAlpha = 1;

    // ID text
    ctx.fillStyle = '#ffffff';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillText(label, lx + 6, ly + lh / 2);

    // If occluded, draw ghost indicator
    if (isOccluded) {
        ctx.font = '500 10px "Inter", sans-serif';
        ctx.fillStyle = '#f59e0b';
        ctx.textAlign = 'center';
        ctx.fillText('OCCLUDED', bbox.x + bbox.width / 2, bbox.y + bbox.height / 2);
    }
}

/**
 * Update the multi-person tracking panel in the sidebar.
 */
function updateTrackingPanel() {
    if (!state.tracker) return;

    const stats = state.tracker.getStats();
    const el = (id) => document.getElementById(id);

    el('track-active').textContent = stats.active;
    el('track-confirmed').textContent = stats.confirmed;
    el('track-lost').textContent = stats.lost;
    el('track-archived').textContent = stats.archived;
    el('tracker-status').textContent = `${stats.active} person${stats.active !== 1 ? 's' : ''}`;

    // Build per-person cards
    const personList = el('person-list');
    if (!personList) return;

    const tracks = state.activeTracks || [];
    let html = '';
    tracks.forEach(track => {
        const color = track.color?.primary || '#6366f1';
        const score = track.lastPosture?.score ?? '--';
        const isGood = track.lastPosture?.isGood;
        const scoreColor = isGood === true ? 'var(--success)' : isGood === false ? 'var(--danger)' : 'var(--text-secondary)';
        const occClass = track.isOccluded ? ' occluded' : '';

        let badges = '';
        if (track.isOccluded) badges += '<span class="person-badge occluded">Occluded</span>';
        if (track.state === 'tentative') badges += '<span class="person-badge new">New</span>';
        if (track.age > 0 && track.hits === 1 && track.totalHits > 10) badges += '<span class="person-badge re-id">Re-ID</span>';

        const goodPct = track.totalAnalyzed > 0
            ? Math.round((track.goodFrames / track.totalAnalyzed) * 100) + '%'
            : '--%';

        html += `
            <div class="person-card${occClass}">
                <div class="person-color-dot" style="background:${color}; --dot-color:${color}"></div>
                <span class="person-id">#${track.id}</span>
                <div class="person-meta">${badges}<span>${goodPct} good</span></div>
                <span class="person-score" style="color:${scoreColor}">${score}</span>
            </div>
        `;
    });

    personList.innerHTML = html || '<div style="color:var(--text-muted); font-size:0.8rem; padding:8px;">No persons detected</div>';
}

/**
 * Update the temporal analysis panel.
 */
function updateTemporalPanel(primaryTrack) {
    if (!primaryTrack?._temporalEngine) return;

    const engine = primaryTrack._temporalEngine;
    const features = engine.getTemporalFeatures();
    const kpStability = engine.getKeypointStability();
    const el = (id) => document.getElementById(id);

    // Stability
    const stability = Math.round(engine.stabilityScore * 100);
    el('temporal-stability').textContent = stability + '%';
    const barStab = el('bar-stability');
    if (barStab) {
        barStab.style.width = stability + '%';
        barStab.style.background = stability > 70 ? 'var(--success)' : stability > 40 ? 'var(--warning)' : 'var(--danger)';
    }

    // Trend
    if (features) {
        const trend = features.slope;
        const trendEl = el('temporal-trend');
        if (trend > 0.5) {
            trendEl.textContent = '↗ Improving';
            trendEl.style.color = 'var(--success)';
        } else if (trend < -0.5) {
            trendEl.textContent = '↘ Declining';
            trendEl.style.color = 'var(--danger)';
        } else {
            trendEl.textContent = '→ Stable';
            trendEl.style.color = 'var(--text-secondary)';
        }
    }

    // Predicted joints count
    const predicted = kpStability.filter(ks => ks.isPredicted && ks.confidence > 0.1).length;
    el('temporal-predicted').textContent = `${predicted}/17`;

    // Window fill
    const windowFill = engine.postureBuffer.length;
    el('temporal-window').textContent = `${windowFill}/30`;
}

/**
 * Draw circles at each detected keypoint with confidence-based styling.
 * Professional ML visualization: glow effects, confidence rings, labels.
 */
function drawKeypoints(keypoints, posture) {
    const color = posture.isGood ? CONFIG.draw.goodColor : CONFIG.draw.badColor;
    const warnColor = CONFIG.draw.warningColor;
    const KP_LABELS = [
        'nose','lEye','rEye','lEar','rEar',
        'lShldr','rShldr','lElbow','rElbow',
        'lWrist','rWrist','lHip','rHip',
        'lKnee','rKnee','lAnkle','rAnkle'
    ];

    keypoints.forEach((kp, i) => {
        if (kp.score > CONFIG.poseNet.minConfidence) {
            const x = kp.position.x;
            const y = kp.position.y;
            const r = CONFIG.draw.keypointRadius;
            const conf = kp.score;

            // Outer confidence ring — radius scales with confidence
            const ringR = r + 4 + (conf * 6);
            ctx.beginPath();
            ctx.arc(x, y, ringR, 0, Math.PI * 2 * conf); // partial arc = confidence %
            ctx.strokeStyle = color;
            ctx.lineWidth = 1.5;
            ctx.globalAlpha = 0.4;
            ctx.stroke();
            ctx.globalAlpha = 1;

            // Keypoint glow
            ctx.beginPath();
            ctx.arc(x, y, r + 3, 0, Math.PI * 2);
            ctx.fillStyle = color;
            ctx.globalAlpha = 0.15;
            ctx.fill();
            ctx.globalAlpha = 1;

            // Main keypoint dot — gradient fill
            const grad = ctx.createRadialGradient(x - 1, y - 1, 0, x, y, r);
            grad.addColorStop(0, '#ffffff');
            grad.addColorStop(0.4, color);
            grad.addColorStop(1, 'rgba(0,0,0,0.3)');
            ctx.beginPath();
            ctx.arc(x, y, r, 0, Math.PI * 2);
            ctx.fillStyle = grad;
            ctx.fill();

            // Thin white ring
            ctx.strokeStyle = 'rgba(255,255,255,0.6)';
            ctx.lineWidth = 1;
            ctx.stroke();

            // Confidence label on key joints (shoulders, nose, hips)
            if ([0, 5, 6, 11, 12].includes(i)) {
                ctx.font = CONFIG.draw.labelFont;
                const label = `${KP_LABELS[i]} ${(conf * 100).toFixed(0)}%`;
                const tw = ctx.measureText(label).width + 8;

                // Label background
                ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
                ctx.beginPath();
                const lx = x + r + 6;
                const ly = y - 6;
                // rounded rect
                const rr = 3;
                ctx.moveTo(lx + rr, ly);
                ctx.arcTo(lx + tw, ly, lx + tw, ly + 14, rr);
                ctx.arcTo(lx + tw, ly + 14, lx, ly + 14, rr);
                ctx.arcTo(lx, ly + 14, lx, ly, rr);
                ctx.arcTo(lx, ly, lx + tw, ly, rr);
                ctx.closePath();
                ctx.fill();

                // Label text
                ctx.fillStyle = conf > 0.7 ? '#22c55e' : (conf > 0.4 ? '#f59e0b' : '#ef4444');
                ctx.textAlign = 'left';
                ctx.textBaseline = 'middle';
                ctx.fillText(label, lx + 4, ly + 7);
            }
        }
    });
}

/**
 * Draw lines connecting keypoints to form the skeleton
 * with gradient coloring and thickness variation.
 */
function drawSkeleton(keypoints, posture) {
    const goodClr = posture.isGood ? 'rgba(34,197,94,0.7)' : 'rgba(239,68,68,0.7)';
    const goodClrFaint = posture.isGood ? 'rgba(34,197,94,0.15)' : 'rgba(239,68,68,0.15)';

    SKELETON_CONNECTIONS.forEach(([iA, iB]) => {
        const a = keypoints[iA];
        const b = keypoints[iB];

        if (a.score > CONFIG.poseNet.minConfidence &&
            b.score > CONFIG.poseNet.minConfidence) {
            const ax = a.position.x, ay = a.position.y;
            const bx = b.position.x, by = b.position.y;

            // Glow line (thick, faint)
            ctx.beginPath();
            ctx.moveTo(ax, ay);
            ctx.lineTo(bx, by);
            ctx.strokeStyle = goodClrFaint;
            ctx.lineWidth = CONFIG.draw.skeletonWidth + 6;
            ctx.stroke();

            // Main bone line — gradient along the bone
            const lineGrad = ctx.createLinearGradient(ax, ay, bx, by);
            lineGrad.addColorStop(0, goodClr);
            lineGrad.addColorStop(1, goodClr);
            ctx.beginPath();
            ctx.moveTo(ax, ay);
            ctx.lineTo(bx, by);
            ctx.strokeStyle = lineGrad;
            ctx.lineWidth = CONFIG.draw.skeletonWidth;
            ctx.lineCap = 'round';
            ctx.stroke();
        }
    });

    // Draw angle visualizations on key joints
    drawAngleAnnotations(keypoints, posture);
}

/**
 * Draw angle arc annotations at key body joints — the hallmark of
 * professional ML pose estimation visualization.
 */
function drawAngleAnnotations(keypoints, posture) {
    const get = (i) => keypoints[i];
    const ok = (i) => get(i).score > CONFIG.poseNet.minConfidence;
    // Note: KP is already defined as a global constant

    // ── Neck angle arc (between ears and shoulders) ──
    if (ok(KP.LEFT_SHOULDER) && ok(KP.RIGHT_SHOULDER) && ok(KP.NOSE)) {
        const ls = get(KP.LEFT_SHOULDER).position;
        const rs = get(KP.RIGHT_SHOULDER).position;
        const nose = get(KP.NOSE).position;
        const midShoulder = { x: (ls.x + rs.x) / 2, y: (ls.y + rs.y) / 2 };

        // Draw neck line
        ctx.beginPath();
        ctx.setLineDash([4, 4]);
        ctx.moveTo(midShoulder.x, midShoulder.y);
        ctx.lineTo(nose.x, nose.y);
        ctx.strokeStyle = 'rgba(168,85,247,0.6)';
        ctx.lineWidth = 1.5;
        ctx.stroke();
        ctx.setLineDash([]);

        // Draw angle arc
        if (posture.neckAngle > 0) {
            const angle1 = Math.atan2(nose.y - midShoulder.y, nose.x - midShoulder.x);
            const vertAngle = -Math.PI / 2; // straight up
            drawAngleArc(midShoulder.x, midShoulder.y, 25, vertAngle, angle1,
                posture.neckAngle, posture.neckAngle >= CONFIG.posture.neckAngleMin ? '#a855f7' : '#ef4444');
        }
    }

    // ── Shoulder tilt line + angle ──
    if (ok(KP.LEFT_SHOULDER) && ok(KP.RIGHT_SHOULDER)) {
        const ls = get(KP.LEFT_SHOULDER).position;
        const rs = get(KP.RIGHT_SHOULDER).position;

        // Horizontal reference line
        ctx.beginPath();
        ctx.setLineDash([3, 3]);
        ctx.moveTo(ls.x - 30, ls.y);
        ctx.lineTo(rs.x + 30, rs.y);
        ctx.strokeStyle = 'rgba(59,130,246,0.4)';
        ctx.lineWidth = 1;
        ctx.stroke();

        // Horizontal reference
        const midX = (ls.x + rs.x) / 2;
        const midY = (ls.y + rs.y) / 2;
        ctx.moveTo(midX - 40, midY);
        ctx.lineTo(midX + 40, midY);
        ctx.strokeStyle = 'rgba(255,255,255,0.2)';
        ctx.stroke();
        ctx.setLineDash([]);

        // Tilt angle label
        if (posture.shoulderTilt !== null) {
            const tiltColor = posture.shoulderTilt <= CONFIG.posture.shoulderTiltMax ? '#3b82f6' : '#ef4444';
            drawMeasurementLabel(midX, midY - 18, `Tilt: ${posture.shoulderTilt.toFixed(1)}°`, tiltColor);
        }
    }

    // ── Head offset indicator ──
    if (ok(KP.NOSE) && ok(KP.LEFT_SHOULDER) && ok(KP.RIGHT_SHOULDER)) {
        const nose = get(KP.NOSE).position;
        const ls = get(KP.LEFT_SHOULDER).position;
        const rs = get(KP.RIGHT_SHOULDER).position;
        const midX = (ls.x + rs.x) / 2;

        // Draw vertical reference from shoulder midpoint
        ctx.beginPath();
        ctx.setLineDash([2, 4]);
        ctx.moveTo(midX, (ls.y + rs.y) / 2);
        ctx.lineTo(midX, nose.y - 20);
        ctx.strokeStyle = 'rgba(255,255,255,0.15)';
        ctx.lineWidth = 1;
        ctx.stroke();
        ctx.setLineDash([]);

        // Offset arrow
        const offset = nose.x - midX;
        if (Math.abs(offset) > 5) {
            ctx.beginPath();
            ctx.moveTo(midX, nose.y);
            ctx.lineTo(nose.x, nose.y);
            ctx.strokeStyle = Math.abs(offset) > CONFIG.posture.headOffsetMax ? '#ef4444' : '#22c55e';
            ctx.lineWidth = 1.5;
            ctx.setLineDash([]);
            ctx.stroke();
        }
    }
}

/**
 * Draw a measurement angle arc with degree label.
 */
function drawAngleArc(cx, cy, radius, startAngle, endAngle, degrees, color) {
    ctx.beginPath();
    ctx.arc(cx, cy, radius, startAngle, endAngle, degrees > 180);
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.globalAlpha = 0.8;
    ctx.stroke();
    ctx.globalAlpha = 1;

    // Degree label at mid-arc
    const midAngle = (startAngle + endAngle) / 2;
    const lx = cx + (radius + 14) * Math.cos(midAngle);
    const ly = cy + (radius + 14) * Math.sin(midAngle);
    drawMeasurementLabel(lx, ly, `${degrees.toFixed(0)}°`, color);
}

/**
 * Draw a small measurement label with background.
 */
function drawMeasurementLabel(x, y, text, color) {
    ctx.font = CONFIG.draw.labelFont;
    const tw = ctx.measureText(text).width + 8;
    const h = 16;

    ctx.fillStyle = 'rgba(0, 0, 0, 0.75)';
    ctx.beginPath();
    const rr = 3;
    ctx.moveTo(x - tw/2 + rr, y - h/2);
    ctx.arcTo(x + tw/2, y - h/2, x + tw/2, y + h/2, rr);
    ctx.arcTo(x + tw/2, y + h/2, x - tw/2, y + h/2, rr);
    ctx.arcTo(x - tw/2, y + h/2, x - tw/2, y - h/2, rr);
    ctx.arcTo(x - tw/2, y - h/2, x + tw/2, y - h/2, rr);
    ctx.closePath();
    ctx.fill();

    // Border
    ctx.strokeStyle = color;
    ctx.lineWidth = 1;
    ctx.globalAlpha = 0.6;
    ctx.stroke();
    ctx.globalAlpha = 1;

    ctx.fillStyle = color;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, x, y);
    ctx.textAlign = 'start';
}

/**
 * Draw a professional ML data HUD overlay — FPS, model info, frame count.
 */
function drawDataHUD(keypoints, posture) {
    // Frame counter for HUD
    if (!drawDataHUD._frameCount) drawDataHUD._frameCount = 0;
    if (!drawDataHUD._lastTime) drawDataHUD._lastTime = performance.now();
    if (!drawDataHUD._fps) drawDataHUD._fps = 0;
    drawDataHUD._frameCount++;
    const now = performance.now();
    if (now - drawDataHUD._lastTime >= 1000) {
        drawDataHUD._fps = drawDataHUD._frameCount;
        drawDataHUD._frameCount = 0;
        drawDataHUD._lastTime = now;
    }

    const padding = 12;
    const lineH = 16;
    const avgConf = keypoints.reduce((s, k) => s + k.score, 0) / keypoints.length;
    const visibleKps = keypoints.filter(k => k.score > CONFIG.poseNet.minConfidence).length;
    const trackerStats = state.tracker ? state.tracker.getStats() : null;
    const modelName = state.useMoveNet ? 'MoveNet MultiPose' : 'PoseNet MobileNetV1';

    const lines = [
        `MODEL: ${modelName}`,
        `FPS: ${drawDataHUD._fps}`,
        `PERSONS: ${trackerStats ? trackerStats.active : 1}`,
        `KEYPOINTS: ${visibleKps}/17`,
        `AVG CONF: ${(avgConf * 100).toFixed(1)}%`,
        `TRACKER: ByteTrack`,
        `TEMPORAL: ${state.temporalEngines.size > 0 ? 'Active' : 'Off'}`,
        `FRAME: ${state.totalFrames}`,
        `CLASS: ${posture.isGood ? 'GOOD_POSTURE' : 'BAD_POSTURE'}`,
        `SCORE: ${posture.score}/100`,
    ];

    const boxW = 220;
    const boxH = lines.length * lineH + padding * 2;
    const boxX = 10;
    const boxY = 10;

    // Background
    ctx.fillStyle = 'rgba(0, 0, 0, 0.75)';
    ctx.beginPath();
    const rr = 6;
    ctx.moveTo(boxX + rr, boxY);
    ctx.arcTo(boxX + boxW, boxY, boxX + boxW, boxY + boxH, rr);
    ctx.arcTo(boxX + boxW, boxY + boxH, boxX, boxY + boxH, rr);
    ctx.arcTo(boxX, boxY + boxH, boxX, boxY, rr);
    ctx.arcTo(boxX, boxY, boxX + boxW, boxY, rr);
    ctx.closePath();
    ctx.fill();

    // Border
    ctx.strokeStyle = posture.isGood ? 'rgba(34,197,94,0.4)' : 'rgba(239,68,68,0.4)';
    ctx.lineWidth = 1;
    ctx.stroke();

    // Text lines
    ctx.font = CONFIG.draw.hudFont;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    lines.forEach((line, i) => {
        const parts = line.split(': ');
        // Key
        ctx.fillStyle = 'rgba(148, 163, 184, 0.9)';
        ctx.fillText(parts[0] + ':', boxX + padding, boxY + padding + i * lineH);
        // Value
        if (parts[0] === 'CLASS') {
            ctx.fillStyle = posture.isGood ? '#22c55e' : '#ef4444';
        } else if (parts[0] === 'AVG CONF') {
            ctx.fillStyle = avgConf > 0.7 ? '#22c55e' : '#f59e0b';
        } else {
            ctx.fillStyle = '#e2e8f0';
        }
        const keyW = ctx.measureText(parts[0] + ': ').width;
        ctx.fillText(parts[1], boxX + padding + keyW, boxY + padding + i * lineH);
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
        // atan2 gives angle from horizontal; for level shoulders dx≫dy so angle≈0
        // We want the absolute deviation from horizontal (0°)
        let tiltAngle = Math.atan2(dy, dx) * (180 / Math.PI);
        // Normalize to range [-90, 90] from horizontal
        if (tiltAngle > 90) tiltAngle = 180 - tiltAngle;
        if (tiltAngle < -90) tiltAngle = -180 - tiltAngle;
        result.shoulderTilt = Math.abs(tiltAngle);
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
        await loadDetector();

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

        // Start MoveNet async detection loop (runs in parallel with renderLoop)
        if (state.useMoveNet) {
            detectLoop();
        }
    } catch (err) {
        console.error('Initialization failed:', err);
        setStatus('Error — see console', 'error');
        DOM.loadingOverlay.classList.add('hidden');
    }
});

DOM.resetBtn.addEventListener('click', resetSession);

// Hide the loading overlay initially (show start prompt instead)
DOM.loadingOverlay.classList.add('hidden');

/**
 * ── CENTRALIZED UI UPDATE ──
 * Orcherstrates all sidebar updates for the primary tracked person.
 */
function updateUI(posture, intel) {
    if (!posture) {
        DOM.postureBanner.classList.remove('good', 'bad', 'warning');
        DOM.postureMessage.innerText = 'Waiting for detection...';
        DOM.postureIcon.innerText = '⌛';
        if (DOM.statusDot) DOM.statusDot.classList.remove('active');
        if (DOM.statusText) DOM.statusText.innerText = 'Idle';
        return;
    }

    // Status
    if (DOM.statusDot) DOM.statusDot.classList.add('active');
    if (DOM.statusText) DOM.statusText.innerText = 'Detecting';

    // Core Posture UI
    updatePostureBanner(posture);
    updateScoreCard(posture.score);
    updateMetrics(posture);
    updateSessionStats(posture);
    
    // ── Action Intelligence UI ──
    if (intel) {
        const { action, reps, pose3D, symmetry } = intel;

        if (DOM.actionLabel) DOM.actionLabel.innerText = action.label.split(' ').slice(1).join(' ') || action.label;
        if (DOM.actionIcon) DOM.actionIcon.innerText  = action.icon;
        if (DOM.actionStatus) DOM.actionStatus.innerText = action.holdFrames > 8 ? 'Confirmed' : 'HOLD...';
        
        const conf = Math.round(action.confidence * 100);
        if (DOM.actionConfPct) DOM.actionConfPct.innerText = conf + '%';
        if (DOM.actionConfRing) {
             DOM.actionConfRing.style.background = `conic-gradient(var(--accent-primary) ${conf * 3.6}deg, rgba(99, 102, 241, 0.1) 0deg)`;
        }

        // Rep Counter
        if (reps && reps.isCountable) {
            if (DOM.repCard) DOM.repCard.style.display = 'block';
            if (DOM.repCountText) DOM.repCountText.innerText = reps.reps;
            if (DOM.repPhase) {
                DOM.repPhase.innerText = reps.phase.replace('_', ' ').toUpperCase();
                if (reps.phase === 'at_bottom') DOM.repPhase.style.background = 'var(--bad-color)';
                else if (reps.phase === 'going_up') DOM.repPhase.style.background = 'var(--warning-color)';
                else DOM.repPhase.style.background = 'rgba(99, 102, 241, 0.2)';
            }
            if (DOM.repProgress) DOM.repProgress.style.width = reps.progress + '%';
        } else {
            if (DOM.repCard) DOM.repCard.style.display = 'none';
        }

        // 3D Motion UI
        if (DOM.depthReliability) {
            DOM.depthReliability.innerText = pose3D.quality.reliability.toUpperCase();
            DOM.depthReliability.style.color = pose3D.quality.reliability === 'high' ? 'var(--good-color)' : 'var(--warning-color)';
        }
        if (DOM.depthCoverage) DOM.depthCoverage.innerText = Math.round(pose3D.quality.depthCoverage * 100) + '%';
        if (DOM.personScale) DOM.personScale.innerText = pose3D.quality.personScale.toFixed(2) + 'x';
        if (DOM.bodySymmetry) DOM.bodySymmetry.innerText = Math.round(symmetry.overall * 100) + '%';

        // Advanced Tips / Corrections
        if (intel.correction) {
            let tipHtml = '';
            // Show correction tips first
            intel.correction.corrections.forEach(c => {
                tipHtml += `<div class="tip-item danger-tip">🎯 <strong>Correction:</strong> ${c.tip}</div>`;
            });
            // Then general tips
            intel.correction.tips.forEach(tip => {
                tipHtml += `<div class="tip-item active">✨ ${tip}</div>`;
            });
            
            // If everything is good, add positive reinforcement
            if (intel.correction.isIdeal && posture.isGood) {
                tipHtml = `<div class="tip-item good-tip">✅ Perfect form maintained. Keep going!</div>` + tipHtml;
            }
            
            if (tipHtml) DOM.tipsContent.innerHTML = tipHtml;
            else updateTips(posture);
        }
    }

    // Tracking Stats
    if (DOM.trackActive) DOM.trackActive.innerText = state.activeTracks.length;
    if (DOM.trackConfirmed) DOM.trackConfirmed.innerText = state.tracker.tracks.filter(t => t.isConfirmed()).length;
    
    // Temporal Stability
    if (posture._stability && DOM.temporalStability) {
        const stab = Math.round(posture._stability * 100);
        DOM.temporalStability.innerText = stab + '%';
        if (DOM.barStability) DOM.barStability.style.width = stab + '%';
        if (DOM.temporalTrend) {
            DOM.temporalTrend.innerText = posture._trend === 'up' ? 'Improving ↗' : (posture._trend === 'down' ? 'Degrading ↘' : 'Stable →');
            DOM.temporalTrend.style.color = posture._trend === 'up' ? 'var(--good-color)' : (posture._trend === 'down' ? 'var(--bad-color)' : 'var(--text-muted)');
        }
    }
}

/**
 * ── ADVANCED DRAWING ──
 * Visualizes 3D depth, action labels, and rep progress rings.
 */
function drawIntelligence(track, intel) {
    if (!intel) return;
    const { pose3D, recoveredJoints, action, reps } = intel;
    const kps = track._smoothedKeypoints || track.keypoints;
    const color = track.color?.primary || CONFIG.draw.keypointColor;

    // 1. Draw Depth Indicators (Subtle rings)
    pose3D.keypoints3D.forEach((kp, i) => {
        if (kp.score > 0.2) {
            const zScale = 1 + (kp.z * 0.4); 
            ctx.beginPath();
            ctx.arc(kp.x, kp.y, 8 * zScale, 0, 2 * Math.PI);
            ctx.strokeStyle = `rgba(99, 102, 241, ${0.1 * zScale})`;
            ctx.setLineDash([2, 2]);
            ctx.stroke();
            ctx.setLineDash([]);
        }
    });

    // 2. Head Action Badge
    const nose = kps[0];
    if (nose && nose.score > 0.3) {
        ctx.save();
        ctx.font = '700 13px "Inter", sans-serif';
        const label = `${action.icon} ${action.label}`;
        const tw = ctx.measureText(label).width + 16;
        const th = 24;
        const tx = nose.x - tw/2;
        const ty = nose.y - 60;

        ctx.fillStyle = 'rgba(15, 23, 42, 0.9)';
        ctx.strokeStyle = color;
        ctx.lineWidth = 1;
        
        // Draw rounded rect
        const r = 6;
        ctx.beginPath();
        ctx.moveTo(tx + r, ty);
        ctx.lineTo(tx + tw - r, ty);
        ctx.quadraticCurveTo(tx + tw, ty, tx + tw, ty + r);
        ctx.lineTo(tx + tw, ty + th - r);
        ctx.quadraticCurveTo(tx + tw, ty + th, tx + tw - r, ty + th);
        ctx.lineTo(tx + r, ty + th);
        ctx.quadraticCurveTo(tx, ty + th, tx, ty + th - r);
        ctx.lineTo(tx, ty + r);
        ctx.quadraticCurveTo(tx, ty, tx + r, ty);
        ctx.fill();
        ctx.stroke();

        ctx.fillStyle = '#fff';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(label, nose.x, ty + th/2);
        ctx.restore();
    }

    // 3. Occlusion Markers
    kps.forEach(kp => {
        if (kp._recovered) {
            ctx.beginPath();
            ctx.arc(kp.x, kp.y, 10, 0, 2 * Math.PI);
            ctx.strokeStyle = 'var(--bad-color)';
            ctx.lineWidth = 1.5;
            ctx.setLineDash([2, 2]);
            ctx.stroke();
            ctx.setLineDash([]);
        }
    });

    // 4. Rep Count Progression Ring on Active Joint
    if (reps && reps.isCountable && reps.progress > 0) {
        // Find the joint index by name
        const jointName = reps.repTrackJoint.toUpperCase();
        // Intelligence.js uses KPI maps which match PoseNet/MoveNet indices.
        // We can just rely on the joint index if known, but let's be safe.
        const jointIdx = KPI ? KPI[jointName] : null;
        const joint = kps[jointIdx];

        if (joint && joint.score > 0.2) {
            ctx.save();
            ctx.translate(joint.x, joint.y);
            
            // Background ring
            ctx.beginPath();
            ctx.arc(0, 0, 28, 0, Math.PI * 2);
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
            ctx.lineWidth = 6;
            ctx.stroke();

            // Progress arc
            ctx.beginPath();
            ctx.arc(0, 0, 28, -Math.PI/2, (-Math.PI/2) + (Math.PI*2 * (reps.progress/100)));
            ctx.strokeStyle = '#22c55e';
            ctx.lineWidth = 6;
            ctx.lineCap = 'round';
            ctx.stroke();

            // Number in center
            ctx.font = '900 16px "Inter"';
            ctx.fillStyle = '#fff';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(reps.reps, 0, 0);
            ctx.restore();
        }
    }
}
