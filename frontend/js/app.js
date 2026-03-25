/* =====================================================
   PostureAI v4.0 — Precision Dashboard Controller
   Performance Optimization: Throttled UI & High Accuracy
   ===================================================== */

import { PoseDetector } from './poseDetector.js';
import { KP_INDEX } from './featureEngine.js';
import { POSTURE_CLASSES, POSTURE_LABELS, POSTURE_COLORS } from './temporalBuffer.js';
import { APIClient } from './apiClient.js';
import { PersonTracker, PERSON_COLORS } from './personTracker.js';

// ─── SKELETON CONNECTIONS ──────────────────────────────
const SKELETON = [
    [KP_INDEX.LEFT_SHOULDER, KP_INDEX.RIGHT_SHOULDER],
    [KP_INDEX.LEFT_SHOULDER, KP_INDEX.LEFT_ELBOW],
    [KP_INDEX.LEFT_ELBOW, KP_INDEX.LEFT_WRIST],
    [KP_INDEX.RIGHT_SHOULDER, KP_INDEX.RIGHT_ELBOW],
    [KP_INDEX.RIGHT_ELBOW, KP_INDEX.RIGHT_WRIST],
    [KP_INDEX.LEFT_SHOULDER, KP_INDEX.LEFT_HIP],
    [KP_INDEX.RIGHT_SHOULDER, KP_INDEX.RIGHT_HIP],
    [KP_INDEX.LEFT_HIP, KP_INDEX.RIGHT_HIP],
    [KP_INDEX.LEFT_HIP, KP_INDEX.LEFT_KNEE],
    [KP_INDEX.LEFT_KNEE, KP_INDEX.LEFT_ANKLE],
    [KP_INDEX.RIGHT_HIP, KP_INDEX.RIGHT_KNEE],
    [KP_INDEX.RIGHT_KNEE, KP_INDEX.RIGHT_ANKLE],
    [KP_INDEX.NOSE, KP_INDEX.LEFT_EYE],
    [KP_INDEX.NOSE, KP_INDEX.RIGHT_EYE],
    [KP_INDEX.LEFT_EYE, KP_INDEX.LEFT_EAR],
    [KP_INDEX.RIGHT_EYE, KP_INDEX.RIGHT_EAR],
    [KP_INDEX.NOSE, KP_INDEX.LEFT_SHOULDER],
    [KP_INDEX.NOSE, KP_INDEX.RIGHT_SHOULDER],
];

// ─── DOM REFERENCES ────────────────────────────────────
const $ = (id) => document.getElementById(id);
const DOM = {
    loadingOverlay: $('loading-overlay'),
    loaderBar: $('loader-bar'),
    loaderText: $('loader-text'),
    startPrompt: $('start-prompt'),
    startBtn: $('start-btn'),
    webcam: $('webcam'),
    canvas: $('pose-canvas'),
    statusDot: $('status-dot'),
    statusText: $('status-text'),
    modelBadge: $('model-badge'),
    backendDot: $('backend-dot'),
    backendText: $('backend-text'),
    postureBanner: $('posture-banner'),
    postureIcon: $('posture-icon'),
    postureMsg: $('posture-msg'),
    fpsBadge: $('fps-badge'),
    personCardsContainer: $('person-cards'),
    globalGoodPct: $('global-good-pct'),
    globalAvgScore: $('global-avg-score'),
    globalAlerts: $('global-alerts'),
    statDuration: $('stat-duration'),
    resetBtn: $('reset-btn'),
    tipsContent: $('tips-content'),
};

const ctx = DOM.canvas.getContext('2d');

// ─── CORE MODULES ──────────────────────────────────────
const detector = new PoseDetector();
const tracker = new PersonTracker();
const api = new APIClient('http://localhost:8000');

// ─── STATE ─────────────────────────────────────────────
const state = {
    isRunning: false,
    sessionStart: null,
    useAPI: false,
    lastUIUpdateTime: 0,
    uiUpdateInterval: 100, // Update DOM at 10 FPS to save performance
};

// ═══════════════════════════════════════════════════════
// WEBCAM SETUP
// ═══════════════════════════════════════════════════════
async function setupCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: 'user' },
        audio: false,
    });
    DOM.webcam.srcObject = stream;
    return new Promise((resolve) => {
        DOM.webcam.onloadedmetadata = () => {
            const w = DOM.webcam.videoWidth;
            const h = DOM.webcam.videoHeight;
            DOM.canvas.width = w;
            DOM.canvas.height = h;
            resolve();
        };
    });
}

// ═══════════════════════════════════════════════════════
// MAIN RENDER LOOP — Multi-Person (Optimized)
// ═══════════════════════════════════════════════════════
async function renderLoop() {
    if (!state.isRunning) return;

    ctx.clearRect(0, 0, DOM.canvas.width, DOM.canvas.height);

    // 1. Detect ALL poses
    const poses = await detector.detectAll();
    
    // 2. Track & Process (Human Validation inside tracker)
    const activePersons = tracker.update(poses || []);

    // 3. Render Canvas (High FPS)
    for (const person of activePersons) {
        person.process(); // Run ML pipeline
        if (person.lastPrediction) {
            drawSkeleton(person.keypoints, person);
            drawKeypoints(person.keypoints, person);
            drawPersonLabel(person);
        }
    }

    // 4. Update UI (Throttled to prevent freezing)
    const now = Date.now();
    if (now - state.lastUIUpdateTime > state.uiUpdateInterval) {
        updateDashboard(activePersons);
        state.lastUIUpdateTime = now;
    }

    // FPS Counter
    if (DOM.fpsBadge) DOM.fpsBadge.textContent = `${detector.fps} FPS`;

    requestAnimationFrame(renderLoop);
}

// ═══════════════════════════════════════════════════════
// DIAGNOSTIC DRAWING (Clinical Style)
// ═══════════════════════════════════════════════════════
function drawKeypoints(kps, person) {
    const isGood = person.lastPrediction?.classIndex === POSTURE_CLASSES.GOOD;
    const color = isGood ? '#10b981' : person.color.primary;

    kps.forEach(kp => {
        if (kp.score > 0.35) {
            ctx.beginPath();
            ctx.arc(kp.x, kp.y, 3, 0, Math.PI * 2);
            ctx.fillStyle = color;
            ctx.fill();
            ctx.strokeStyle = '#fff';
            ctx.lineWidth = 1;
            ctx.stroke();
        }
    });
}

function drawSkeleton(kps, person) {
    const isGood = person.lastPrediction?.classIndex === POSTURE_CLASSES.GOOD;
    const color = isGood ? 'rgba(16, 185, 129, 0.4)' : `rgba(${person.color.rgb}, 0.5)`;

    SKELETON.forEach(([iA, iB]) => {
        const a = kps[iA], b = kps[iB];
        if (a && b && a.score > 0.35 && b.score > 0.35) {
            ctx.beginPath();
            ctx.moveTo(a.x, a.y);
            ctx.lineTo(b.x, b.y);
            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.stroke();
        }
    });
}

function drawPersonLabel(person) {
    const nose = person.keypoints[KP_INDEX.NOSE];
    if (!nose || nose.score < 0.3) return;

    const label = `S#${person.id} • ${person.lastPrediction?.score || 0}%`;
    ctx.font = '700 11px Inter, sans-serif';
    const tw = ctx.measureText(label).width + 16;
    
    ctx.fillStyle = 'rgba(10, 12, 16, 0.85)';
    ctx.beginPath();
    // Polyfill for roundRect (not supported in all browsers)
    if (ctx.roundRect) {
        ctx.roundRect(nose.x - tw/2, nose.y - 40, tw, 20, 4);
    } else {
        const x = nose.x - tw/2, y = nose.y - 40, w = tw, h = 20, r = 4;
        ctx.moveTo(x + r, y);
        ctx.arcTo(x + w, y, x + w, y + h, r);
        ctx.arcTo(x + w, y + h, x, y + h, r);
        ctx.arcTo(x, y + h, x, y, r);
        ctx.arcTo(x, y, x + w, y, r);
        ctx.closePath();
    }
    ctx.fill();
    ctx.strokeStyle = person.color.primary;
    ctx.lineWidth = 1.5;
    ctx.stroke();

    ctx.fillStyle = '#fff';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(label, nose.x, nose.y - 30);
    ctx.textAlign = 'start';
}

// ═══════════════════════════════════════════════════════
// THROTTLED DASHBOARD UPDATES
// ═══════════════════════════════════════════════════════
function updateDashboard(persons) {
    updatePeopleStatus(persons);
    updatePersonCards(persons);
    updateGlobalStats(persons);
    updateDiagnostics(persons);
}

function updatePeopleStatus(persons) {
    if (persons.length === 0) {
        DOM.postureIcon.textContent = '📡';
        DOM.postureMsg.textContent = 'Neural Scan Ready';
        DOM.statusDot.className = 'dot dot-standby';
        return;
    }

    const badCount = persons.filter(p => p.lastPrediction?.classIndex !== 0).length;
    DOM.statusDot.className = 'dot dot-online';

    if (badCount === 0) {
        DOM.postureIcon.textContent = '🛡️';
        DOM.postureMsg.textContent = `Nominal Integrity (${persons.length} Subjects)`;
    } else {
        DOM.postureIcon.textContent = '⚠️';
        DOM.postureMsg.textContent = `${badCount} Deviations Detected`;
    }
}

function updatePersonCards(persons) {
    if (!DOM.personCardsContainer) return;

    if (persons.length === 0) {
        DOM.personCardsContainer.innerHTML = `
            <div style="padding: 40px 20px; text-align: center; border: 1px dashed var(--border-subtle); border-radius: 12px; color: var(--text-low);">
                <div style="font-size: 2rem; margin-bottom: 12px;">👥</div>
                <p style="font-size: 0.8rem; font-weight: 500;">No subjects active.<br/>Initiate scan to begin.</p>
            </div>`;
        return;
    }

    DOM.personCardsContainer.innerHTML = persons.map(p => {
        const pred = p.lastPrediction;
        if (!pred) return ''; // Safety: skip if no prediction yet
        const score = pred.score;
        const isGood = pred.classIndex === 0;

        return `
        <div class="person-card ${isGood ? '' : 'active'}" style="--person-color: ${p.color.primary};">
            <div class="card-header">
                <span class="id-badge">ID:${p.id}</span>
                <span class="human-check">✔️ HUMAN VALIDATED</span>
            </div>
            
            <div style="display: flex; justify-content: space-between; align-items: flex-end; margin-bottom: 12px;">
                <div>
                   <div style="font-size: 0.85rem; font-weight: 700;">${pred.label.toUpperCase()}</div>
                   <div style="font-size: 0.65rem; color: var(--text-mid);">CONFIDENCE: ${(pred.confidence * 100).toFixed(0)}%</div>
                </div>
                <div style="font-size: 1.25rem; font-weight: 700; color: ${score > 80 ? 'var(--success)' : 'var(--warning)'}">${score}</div>
            </div>

            <div class="prob-bar-container">
                <div class="prob-fill" style="width: ${score}%; background: ${score > 80 ? 'var(--success)' : 'var(--primary)'};"></div>
            </div>

            <div style="display: flex; gap: 12px; font-size: 0.65rem; color: var(--text-mid); font-family: 'JetBrains Mono', monospace;">
               <span>⏱ ${formatDuration(p.durationSeconds)}</span>
               <span>🔔 ${p.alertCount} ALERTS</span>
            </div>
        </div>`;
    }).join('');
}

function updateGlobalStats(persons) {
    if (persons.length === 0) return;
    const avgScore = persons.reduce((s, p) => s + (p.lastPrediction?.score || 0), 0) / persons.length;
    const totalAlerts = persons.reduce((s, p) => s + p.alertCount, 0);

    // Calculate good posture percentage across all persons
    const totalFrames = persons.reduce((s, p) => s + p.totalFrames, 0);
    const goodFrames = persons.reduce((s, p) => s + p.goodFrames, 0);
    const goodPct = totalFrames > 0 ? Math.round((goodFrames / totalFrames) * 100) : 0;

    if (DOM.globalGoodPct) DOM.globalGoodPct.textContent = goodPct + '%';
    if (DOM.globalAvgScore) DOM.globalAvgScore.textContent = Math.round(avgScore);
    if (DOM.globalAlerts) DOM.globalAlerts.textContent = totalAlerts;
}

function updateDiagnostics(persons) {
    if (!DOM.tipsContent) return;
    const bad = persons.find(p => p.lastPrediction?.classIndex !== 0);
    
    if (!bad) {
        DOM.tipsContent.innerHTML = 'System stabilized. All subjects maintaining nominal posture integrity.';
        return;
    }

    DOM.tipsContent.innerHTML = `<strong>Subject #${bad.id}</strong>: ${bad.lastPrediction.label} detected. Corrective posture required to maintain integrity.`;
}

function formatDuration(sec) {
    const m = Math.floor(sec / 60).toString().padStart(2, '0');
    const s = (sec % 60).toString().padStart(2, '0');
    return `${m}:${s}`;
}

// ═══════════════════════════════════════════════════════
// CORE INITIALIZATION
// ═══════════════════════════════════════════════════════
async function init() {
    api.startHealthCheck(15000);
    const apiState = await api.checkConnection();
    DOM.backendDot.className = apiState ? 'dot dot-online' : 'dot dot-standby';
    DOM.backendText.textContent = `AI Link: ${apiState ? 'Optimal' : 'Standalone'}`;
    state.useAPI = apiState;
}

DOM.startBtn.addEventListener('click', async () => {
    DOM.startPrompt.classList.add('hidden');
    DOM.loadingOverlay.classList.remove('hidden');
    
    try {
        await setupCamera();
        DOM.loaderText.textContent = 'Linking Neural Interface...';
        
        const m = await detector.init(DOM.webcam, (pct) => {
            DOM.loaderBar.style.width = pct + '%';
        });

        DOM.modelBadge.textContent = m.toUpperCase();
        DOM.loadingOverlay.classList.add('hidden');
        DOM.statusDot.className = 'dot dot-online';
        DOM.statusText.textContent = 'Analysis Active';
        
        state.isRunning = true;
        state.sessionStart = Date.now();
        setInterval(() => {
            const elapsed = Math.floor((Date.now() - state.sessionStart) / 1000);
            DOM.statDuration.textContent = formatDuration(elapsed);
        }, 1000);

        renderLoop();
    } catch (err) {
        console.error('Initialization failed:', err);
        DOM.loadingOverlay.classList.add('hidden');
        DOM.statusDot.className = 'dot dot-standby';
        DOM.statusText.textContent = 'Error — check camera';
        
        // Show error state in banner
        DOM.postureIcon.textContent = '🚫';
        DOM.postureMsg.textContent = err.name === 'NotAllowedError'
            ? 'Camera permission denied. Please allow camera access and refresh.'
            : 'Failed to initialize. Check camera and refresh.';
        
        // Re-show start prompt so user can retry
        DOM.startPrompt.classList.remove('hidden');
    }
});

DOM.resetBtn.addEventListener('click', () => { tracker.reset(); state.sessionStart = Date.now(); });

init();
