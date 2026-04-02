/**
 * SignLens — Client-side logic
 * Uses MediaPipe Holistic → fetch /predict
 */

// ═══════════════════════════════════════════
// ─── THEME ───
// ═══════════════════════════════════════════
(function applyStoredTheme() {
    const saved = localStorage.getItem('sl-theme') || 'dark';
    document.documentElement.setAttribute('data-theme', saved);
    _syncThemeIcon(saved);
})();

function _syncThemeIcon(theme) {
    const icon = theme === 'dark' ? '☀︎' : '☾';
    const el = document.getElementById('ribbon-theme-icon');
    if (el) el.textContent = icon;
}

function toggleTheme() {
    const cur  = document.documentElement.getAttribute('data-theme') || 'dark';
    const next = cur === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('sl-theme', next);
    _syncThemeIcon(next);
}

// ═══════════════════════════════════════════
// ─── LANDING ↔ APP ───
// ═══════════════════════════════════════════
function enterApp() {
    const landing = document.getElementById('landing');
    const app     = document.getElementById('app');
    landing.style.transition = 'opacity .4s ease';
    landing.style.opacity    = '0';
    setTimeout(() => {
        landing.style.display = 'none';
        app.style.display     = 'flex';
        app.style.flexDirection = 'column';
        app.style.opacity     = '0';
        app.style.transition  = 'opacity .4s ease';
        requestAnimationFrame(() => { app.style.opacity = '1'; });
    }, 400);
}

function goBack() {
    const landing = document.getElementById('landing');
    const app     = document.getElementById('app');
    app.style.transition = 'opacity .3s ease';
    app.style.opacity    = '0';
    setTimeout(() => {
        app.style.display     = 'none';
        landing.style.display = 'flex';
        landing.style.opacity = '0';
        landing.style.transition = 'opacity .4s ease';
        requestAnimationFrame(() => { landing.style.opacity = '1'; });
        stopAnimation(); // stop any running skeleton animation
    }, 300);
}

// ═══════════════════════════════════════════
// ─── GUIDE MODAL ───
// ═══════════════════════════════════════════
function openGuide(tab) {
    document.getElementById('guide-overlay').style.display = 'block';
    const modal = document.getElementById('guide-modal');
    modal.style.display = 'flex';
    modal.style.flexDirection = 'column';
    switchTab(tab || 'abc');
}

function closeGuide() {
    document.getElementById('guide-overlay').style.display = 'none';
    document.getElementById('guide-modal').style.display   = 'none';
}

function switchTab(tab) {
    // Update tab buttons
    ['abc','words','view'].forEach(t => {
        const btn = document.getElementById('gtab-' + t);
        const panel = document.getElementById('gpanel-' + t);
        if (!btn || !panel) return;
        const active = t === tab;
        btn.classList.toggle('active', active);
        btn.setAttribute('aria-selected', active);
        panel.style.display = active ? 'flex' : 'none';
    });
}

// Close on Escape
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeGuide(); });

// ═══════════════════════════════════════════
// ─── VIEWER FULLSCREEN ───
// ═══════════════════════════════════════════
function toggleFullscreen() {
    const card = document.getElementById('viewer-card');
    const expand = document.getElementById('fs-icon-expand');
    const shrink = document.getElementById('fs-icon-shrink');

    if (!document.fullscreenElement && !document.webkitFullscreenElement) {
        // Enter fullscreen
        const req = card.requestFullscreen || card.webkitRequestFullscreen;
        if (req) req.call(card);
    } else {
        // Exit fullscreen
        const exit = document.exitFullscreen || document.webkitExitFullscreen;
        if (exit) exit.call(document);
    }
}

// Sync icon when user exits fullscreen via Escape or browser chrome
function _onFullscreenChange() {
    const inFs = !!(document.fullscreenElement || document.webkitFullscreenElement);
    const expand = document.getElementById('fs-icon-expand');
    const shrink = document.getElementById('fs-icon-shrink');
    if (expand) expand.style.display = inFs ? 'none'  : 'inline';
    if (shrink) shrink.style.display = inFs ? 'inline' : 'none';
}
document.addEventListener('fullscreenchange',       _onFullscreenChange);
document.addEventListener('webkitfullscreenchange', _onFullscreenChange);


// ═══════════════════════════════════════════
// ─── VIEWER MODE: ABC vs WORDS ───
// ═══════════════════════════════════════════
let viewerSubMode = 'abc';    // 'abc' or 'words'
let alphabetListLoaded = false;

function setViewerMode(mode) {
    viewerSubMode = mode;
    // Toggle sub-toggle active state
    document.getElementById('subtog-abc').classList.toggle('active', mode === 'abc');
    document.getElementById('subtog-words').classList.toggle('active', mode === 'words');
    // Toggle control rows
    document.getElementById('viewer-controls').style.display = mode === 'words' ? 'flex' : 'none';
    document.getElementById('viewer-alpha-controls').style.display = mode === 'abc' ? 'flex' : 'none';
    // Show/hide scrubber (only for words)
    document.getElementById('viewer-scrubber').style.display = mode === 'words' ? 'flex' : 'none';
    // Frame info
    document.getElementById('viewer-frame-info').textContent = mode === 'words' ? 'Frame —' : '';
    // Load alphabet list on first switch
    if (mode === 'abc' && !alphabetListLoaded) {
        fetch('/alphabet_list').then(r => r.json()).then(letters => {
            const sel = document.getElementById('letter-select');
            letters.forEach(l => {
                const opt = document.createElement('option');
                opt.value = l; opt.textContent = l;
                sel.appendChild(opt);
            });
            alphabetListLoaded = true;
        });
    }
    // Clear canvas
    const c = document.getElementById('viz-canvas');
    const ctx = c.getContext('2d');
    ctx.clearRect(0, 0, c.width, c.height);
    document.getElementById('viewer-word-display').textContent = '';
    // Stop any word animation
    stopAnimation();
}

// ── Alphabet hand drawing ──
const VIEWER_HAND_CONNS = [
    [0,1],[1,2],[2,3],[3,4],      // thumb
    [0,5],[5,6],[6,7],[7,8],      // index
    [0,9],[9,10],[10,11],[11,12], // middle
    [0,13],[13,14],[14,15],[15,16], // ring
    [0,17],[17,18],[18,19],[19,20], // pinky
    [5,9],[9,13],[13,17]            // palm
];

function loadAlphabetDemo() {
    const letter = document.getElementById('letter-select').value;
    if (!letter) return;

    fetch(`/alphabet_demo?letter=${letter}`)
        .then(r => r.json())
        .then(data => {
            if (data.error) { alert(data.error); return; }
            drawHandLandmarks(data.landmarks, data.letter);
        });
}

function drawHandLandmarks(flat, letter) {
    const c = document.getElementById('viz-canvas');
    const ctx = c.getContext('2d');
    const W = c.width, H = c.height;
    ctx.clearRect(0, 0, W, H);

    // flat is 63 floats: [x0,y0,z0, x1,y1,z1, ... x20,y20,z20]
    // MediaPipe gives normalised [0,1] coords
    const pts = [];
    for (let i = 0; i < 21; i++) {
        pts.push({ x: flat[i*3], y: flat[i*3+1], z: flat[i*3+2] });
    }

    // Find bounding box and scale to fill canvas with padding
    let minX=Infinity, maxX=-Infinity, minY=Infinity, maxY=-Infinity;
    pts.forEach(p => { minX=Math.min(minX,p.x); maxX=Math.max(maxX,p.x); minY=Math.min(minY,p.y); maxY=Math.max(maxY,p.y); });
    const padFrac = 0.12;
    const rangeX = (maxX-minX) || 0.01;
    const rangeY = (maxY-minY) || 0.01;
    const scale = Math.min(W*(1-2*padFrac)/rangeX, H*(1-2*padFrac)/rangeY);
    const cx = W/2 - (minX+maxX)/2 * scale;
    const cy = H/2 - (minY+maxY)/2 * scale;

    function tx(p) { return p.x * scale + cx; }
    function ty(p) { return p.y * scale + cy; }

    // Draw connections
    ctx.strokeStyle = 'rgba(255,143,163,0.7)';
    ctx.lineWidth = 2.5;
    VIEWER_HAND_CONNS.forEach(([a,b]) => {
        ctx.beginPath();
        ctx.moveTo(tx(pts[a]), ty(pts[a]));
        ctx.lineTo(tx(pts[b]), ty(pts[b]));
        ctx.stroke();
    });

    // Draw joints
    pts.forEach((p, i) => {
        const r = i === 0 ? 6 : 4;
        ctx.beginPath();
        ctx.arc(tx(p), ty(p), r, 0, Math.PI*2);
        ctx.fillStyle = i <= 4 ? '#ff5f80' : '#ffb7c5';
        ctx.fill();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 1;
        ctx.stroke();
    });

    // Show letter label
    document.getElementById('viewer-word-display').textContent = letter;
}


// ═══════════════════════════════════════════
// ─── EVALUATION MODAL ───
// ═══════════════════════════════════════════
let evalData = null;        // cached after first fetch

function openEval() {
    document.getElementById('eval-overlay').style.display = 'block';
    const modal = document.getElementById('eval-modal');
    modal.style.display = 'flex'; modal.style.flexDirection = 'column';

    // Always fetch fresh data when opening the modal
    runEvaluation();
}

function closeEval() {
    document.getElementById('eval-overlay').style.display = 'none';
    document.getElementById('eval-modal').style.display   = 'none';
}

function switchEvalTab(tab) {
    ['alphabet', 'word', 'cnn_word'].forEach(t => {
        const btn = document.getElementById('etab-' + t);
        const panel = document.getElementById('epanel-' + t);
        if (!btn || !panel) return;
        const active = t === tab;
        btn.classList.toggle('active', active);
        btn.setAttribute('aria-selected', active);
        panel.style.display = active ? 'flex' : 'none';
    });
}

function runEvaluation() {
    const loading = document.getElementById('eval-loading');
    loading.style.display = 'flex';
    document.getElementById('epanel-alphabet').style.display = 'none';
    document.getElementById('epanel-word').style.display = 'none';
    document.getElementById('epanel-cnn_word').style.display = 'none';

    fetch('/evaluate?t=' + new Date().getTime())
        .then(r => r.json())
        .then(data => {
            evalData = data;
            loading.style.display = 'none';

            if (data.alphabet) populateEvalPanel('alphabet', data.alphabet);
            if (data.word)     populateEvalPanel('word', data.word);
            if (data.cnn_word) populateEvalPanel('cnn_word', data.cnn_word);

            switchEvalTab(data.alphabet ? 'alphabet' : 'word');
        })
        .catch(err => {
            loading.style.display = 'none';
            alert('Evaluation failed: ' + err);
        });
}

function _fmtNum(n) {
    if (n >= 1e6) return (n/1e6).toFixed(1) + 'M';
    if (n >= 1e3) return (n/1e3).toFixed(1) + 'K';
    return n.toString();
}

function populateEvalPanel(type, d) {
    const idMap = { alphabet: 'alpha-metrics', word: 'word-metrics', cnn_word: 'cnn_word-metrics' };
    const metricsEl = document.getElementById(idMap[type] || type + '-metrics');
    metricsEl.innerHTML = '';

    // Build metric cards
    const cards = [
        { val: d.accuracy + '%', label: 'Accuracy' },
        { val: d.f1_macro + '%', label: 'F1 Score (Macro)' },
    ];
    if (d.wer !== undefined) cards.push({ val: d.wer + '%', label: 'Word Error Rate' });
    if (d.mean_jaccard !== undefined) cards.push({ val: d.mean_jaccard + '%', label: 'Mean Jaccard Index' });
    if (d.precision_macro) cards.push({ val: d.precision_macro + '%', label: 'Precision' });
    if (d.recall_macro) cards.push({ val: d.recall_macro + '%', label: 'Recall' });
    if (d.avg_latency_ms) cards.push({ val: d.avg_latency_ms.toFixed(2) + 'ms', label: 'Avg Latency' });
    if (d.param_count) cards.push({ val: _fmtNum(d.param_count), label: 'Parameters' });
    if (d.flops) cards.push({ val: _fmtNum(d.flops), label: 'FLOPs' });
    if (d.total_samples) cards.push({ val: d.total_samples.toString(), label: 'Total Samples' });

    cards.forEach(c => {
        metricsEl.innerHTML += `<div class="metric-card"><span class="metric-val">${c.val}</span><span class="metric-label">${c.label}</span></div>`;
    });

    // Populate per-class table
    const tableMap = { alphabet: 'alpha-table', word: 'word-table', cnn_word: 'cnn_word-table' };
    const tbody = document.querySelector(`#${tableMap[type] || type + '-table'} tbody`);
    tbody.innerHTML = '';
    (d.per_class || []).forEach(row => {
        const f1cls = row.f1 >= 80 ? 'val-good' : row.f1 >= 50 ? 'val-ok' : 'val-bad';
        tbody.innerHTML += `<tr>
            <td><strong>${row.class}</strong></td>
            <td>${row.precision}%</td>
            <td>${row.recall}%</td>
            <td class="${f1cls}">${row.f1}%</td>
            <td>${row.support}</td>
        </tr>`;
    });
}

// Close eval on Escape
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeEval(); });



// ─── State ───
let currentMode = 'alphabet';
let boardText = '';

// ─── Alphabet stability ───
let potCandidate = '';
let stabilityCount = 0;
// Lower threshold on mobile: high-latency connections get fewer responses per second,
// so 10 consecutive hits is nearly impossible. 5 gives a good balance.
const isMobile = /Mobi|Android|iPhone|iPad/i.test(navigator.userAgent);
const STABILITY_THRESHOLD = isMobile ? 5 : 10;

// isPredicting flag for alphabet — same pattern as word mode.
// Prevents request pile-up on slow connections (ngrok, mobile data).
let isAlphaPredicting = false;

// ─── Word mode state ───
let sequence = [];
let isPredicting = false;
let lastWordPrediction = '';
let potWordCandidate = '';
let wordStabilityCount = 0;
const WORD_STABILITY_THRESHOLD = 4; // Number of consistent predictions needed

// ─── Word mode: confidence threshold ───
// 55 is significantly below the old hard-coded 70.
// Data shows this recovers 10-15% correct predictions for weak signs
// (like, white, person, fall, will, alligator) while the stability
// filter (4 consecutive matches) still prevents false commits.
const WORD_CONFIDENCE_THRESHOLD = 55;

// ─── DOM refs ───
const videoEl = document.getElementById('webcam');
const canvasEl = document.getElementById('overlay');
const ctx = canvasEl.getContext('2d');
const statusEl = document.getElementById('status-text');
const progressEl = document.getElementById('progress-fill');
const boardEl = document.getElementById('board-text');
const rawValEl = document.getElementById('raw-value');
const confValEl = document.getElementById('conf-value');

// ─── Selected face landmark indices (matches original app) ───
const FACE_SEL = [0, 4, 7, 8, 10, 13, 14, 17, 21, 33, 37, 39, 40, 46, 52, 53, 54, 55, 58, 61, 63, 65, 66, 67, 70, 78, 80, 81, 82, 84, 87, 88, 91, 93, 95, 103, 105, 107, 109, 127, 132, 133, 136, 144, 145, 146, 148, 149, 150, 152, 153, 154, 155, 157, 158, 159, 160, 161, 162, 163, 172, 173, 176, 178, 181, 185, 191, 234, 246, 249, 251, 263, 267, 269, 270, 276, 282, 283, 284, 285, 288, 291, 293, 295, 296, 297, 300, 308, 310, 311, 312, 314, 317, 318, 321, 323, 324, 332, 334, 336, 338, 356, 361, 362, 365, 373, 374, 375, 377, 378, 379, 380, 381, 382, 384, 385, 386, 387, 388, 389, 390, 397, 398, 400, 402, 405, 409, 415, 454, 466, 468, 473];

// ─── Mode switching ───
function setMode(mode) {
    currentMode = mode;
    document.getElementById('btn-alphabet').classList.toggle('active', mode === 'alphabet');
    document.getElementById('btn-word').classList.toggle('active', mode === 'word');
    document.getElementById('btn-view').classList.toggle('active', mode === 'view');

    const cameraCard = document.getElementById('camera-card');
    const boardCard = document.getElementById('board-card');
    const viewerCard = document.getElementById('viewer-card');

    if (mode === 'view') {
        cameraCard.style.display = 'none';
        boardCard.style.display = 'none';
        viewerCard.style.display = 'flex';
        viewerCard.style.flexDirection = 'column';
        populateWordList();
        // Always start in ABC sub-mode when entering View
        setViewerMode('abc');
    } else {
        cameraCard.style.display = '';
        boardCard.style.display = '';
        viewerCard.style.display = 'none';
        stopAnimation();
    }

    boardText = '';
    boardEl.textContent = '';
    rawValEl.textContent = '—';
    confValEl.textContent = '';
    potCandidate = '';
    stabilityCount = 0;
    sequence = [];
    lastWordPrediction = '';
    potWordCandidate = '';
    wordStabilityCount = 0;
    if (mode !== 'view') {
        statusEl.textContent = 'Switched to ' + (mode === 'alphabet' ? 'Alphabet' : 'Word') + ' mode';
    }
}

function clearBoard() {
    boardText = '';
    boardEl.textContent = '';
}

// ─── Hand connections for minimal overlay ───
const HAND_CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 4],
    [0, 5], [5, 6], [6, 7], [7, 8],
    [5, 9], [9, 10], [10, 11], [11, 12],
    [9, 13], [13, 14], [14, 15], [15, 16],
    [13, 17], [17, 18], [18, 19], [19, 20], [0, 17]
];

// ─── Draw minimal hand landmarks (live webcam overlay) ───
function drawHand(landmarks, color, W, H) {
    if (!landmarks) return;
    ctx.strokeStyle = color.replace(')', ', 0.25)').replace('rgb', 'rgba');
    ctx.lineWidth = 1;
    for (const [a, b] of HAND_CONNECTIONS) {
        ctx.beginPath();
        ctx.moveTo(landmarks[a].x * W, landmarks[a].y * H);
        ctx.lineTo(landmarks[b].x * W, landmarks[b].y * H);
        ctx.stroke();
    }
    for (const pt of landmarks) {
        ctx.beginPath();
        ctx.arc(pt.x * W, pt.y * H, 2, 0, 2 * Math.PI);
        ctx.fillStyle = color.replace(')', ', 0.5)').replace('rgb', 'rgba');
        ctx.fill();
    }
}

// ─── Extract xyz from landmarks for word mode ───
function getXYZ(landmarks) {
    if (!landmarks) return new Array(63).fill(0);
    return landmarks.map(p => [p.x, p.y, p.z]).flat();
}

// ─── Holistic results handler (matches old app's onResults exactly) ───
function onResults(res) {
    // Sync canvas pixel resolution to its current CSS display size so landmarks
    // coincide exactly with the video pixels (avoids scaling artefacts when the
    // container is wider/taller than 640×480).
    const rect = canvasEl.getBoundingClientRect();
    const dpr  = window.devicePixelRatio || 1;
    canvasEl.width  = rect.width  * dpr;
    canvasEl.height = rect.height * dpr;
    // Scale drawing context to match device pixel ratio
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, rect.width, rect.height);


    // Draw minimal landmarks — pass CSS display dimensions for pixel-perfect alignment
    drawHand(res.rightHandLandmarks, 'rgb(108, 99, 255)', rect.width, rect.height);
    drawHand(res.leftHandLandmarks,  'rgb(72, 207, 173)',  rect.width, rect.height);

    // No hands detected
    if (!res.leftHandLandmarks && !res.rightHandLandmarks) {
        statusEl.textContent = 'Show your hand';
        rawValEl.textContent = '—';
        confValEl.textContent = '';
        lastWordPrediction = '';
        potWordCandidate = '';
        wordStabilityCount = 0;
        progressEl.style.width = '0%';
        sequence = [];
        return;
    }

    // Build feature array: RIGHT hand first, then LEFT hand (matches old app)
    let feat = [];
    feat.push(...getXYZ(res.rightHandLandmarks), ...getXYZ(res.leftHandLandmarks));

    if (currentMode === 'alphabet') {
        // ── ALPHABET: uses 63 features (X,Y,Z of ONE hand)
        let alphaFeat = [];
        const getAlphaXYZ = (lms) => lms ? lms.map(p => [p.x, p.y, p.z]).flat() : new Array(63).fill(0);

        let handLms = res.rightHandLandmarks || res.leftHandLandmarks;
        if (handLms) {
            alphaFeat = getAlphaXYZ(handLms);
        } else {
            alphaFeat = new Array(63).fill(0);
        }

        // On mobile only: skip this frame if a request is already in-flight.
        // This prevents request pile-up over high-latency connections (ngrok, mobile data).
        // On desktop the guard is skipped — every frame fires as before.
        if (isMobile && isAlphaPredicting) return;
        if (isMobile) isAlphaPredicting = true;

        fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ landmarks: alphaFeat, mode: 'alphabet' })
        }).then(r => r.json()).then(data => {
            confValEl.textContent = data.confidence + '%';

            if (data.prediction) {
                rawValEl.textContent = data.prediction;
                statusEl.textContent = 'Detected: ' + data.prediction;
            }

            // Client-side stability (EXACT match of old app's logic)
            if (data.prediction !== '' && data.prediction !== 'ERROR') {
                if (data.prediction === potCandidate) {
                    stabilityCount++;
                } else {
                    potCandidate = data.prediction;
                    stabilityCount = 0;
                }
                progressEl.style.width = (stabilityCount / STABILITY_THRESHOLD) * 100 + '%';

                if (stabilityCount >= STABILITY_THRESHOLD) {
                    if (data.prediction === 'SPACE') {
                        boardText += ' ';
                    } else if (data.prediction === 'DEL') {
                        boardText = boardText.slice(0, -1);
                    } else {
                        boardText += data.prediction;
                    }
                    boardEl.textContent = boardText;
                    stabilityCount = 0;
                    statusEl.textContent = '✓ ' + data.prediction;
                }
            } else {
                progressEl.style.width = '0%';
            }
        }).catch(() => {}).finally(() => {
            if (isMobile) isAlphaPredicting = false;  // release lock (mobile only)
        });

    } else if (currentMode === 'word') {
        // ── WORD MODE: buffer 30 frames, then fetch ──
        // Build full 540 features per frame (X, Y, Z)
        [11, 12, 13, 14, 15, 16].forEach(i => {
            const p = res.poseLandmarks?.[i];
            feat.push(p?.x || 0, p?.y || 0, p?.z || 0);
        });
        FACE_SEL.forEach(i => {
            const p = res.faceLandmarks?.[i];
            feat.push(p?.x || 0, p?.y || 0, p?.z || 0);
        });

        // ── EXPLICIT FEATURE EXTRACTION: Hand-to-Face Distances ──
        // (Must strictly match data_prep.py implementation)
        const LIP_VAR = 14;
        const NOSE_VAR = 4;
        const TEMPLE_VAR = 67;

        const calcDist = (pointA, pointB) => {
            if (!pointA || !pointB) return 0.0;
            return Math.sqrt(
                Math.pow(pointA.x - pointB.x, 2) +
                Math.pow(pointA.y - pointB.y, 2) +
                Math.pow(pointA.z - pointB.z, 2)
            );
        };

        const rightIndex = res.rightHandLandmarks?.[8];
        const leftIndex = res.leftHandLandmarks?.[8];
        const lip = res.faceLandmarks?.[LIP_VAR];
        const nose = res.faceLandmarks?.[NOSE_VAR];
        const temple = res.faceLandmarks?.[TEMPLE_VAR];

        // 6 appended features: Right Lip, Right Nose, Right Temple, Left Lip, Left Nose, Left Temple
        feat.push(calcDist(rightIndex, lip));
        feat.push(calcDist(rightIndex, nose));
        feat.push(calcDist(rightIndex, temple));

        feat.push(calcDist(leftIndex, lip));
        feat.push(calcDist(leftIndex, nose));
        feat.push(calcDist(leftIndex, temple));

        sequence.push(feat);
        if (sequence.length > 60) sequence.shift();   // rolling 60-frame history

        // Dispatch once buffer holds 30+ frames and no request is in flight.
        // Sends the most recent 30 frames each time (sliding window).
        // This matches original dispatch frequency (every round-trip) while
        // ensuring the sign's peak is always near the centre of the window.
        if (sequence.length >= 30 && !isPredicting) {

            const window30 = sequence.slice(-30);

            // ── Guard: Zero-landmark check ───────────────────────────────────
            // Skip if >55% of right+left hand coordinate features are zero.
            // One-handed signs produce ~50% zeros (63/126 from untracked hand),
            // so threshold must be above 50%. Blocks only when both hands are
            // mostly/fully untracked — prevents garbage-input predictions.
            let zeroCount = 0;
            for (let f = 0; f < window30.length; f++) {
                for (let i = 0; i < 126; i++) {
                    if (window30[f][i] === 0) zeroCount++;
                }
            }
            const zeroFrac = zeroCount / (30 * 126);

            if (zeroFrac > 0.55) {
                statusEl.textContent = 'Keep hands in frame...';

            } else {
                // ── Dispatch to model ────────────────────────────────────────
                isPredicting = true;
                fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ landmarks: window30, mode: 'word' })
                }).then(r => r.json()).then(data => {
                    confValEl.textContent = data.confidence + '%';

                    if (data.prediction !== 'BUFFERING' &&
                        data.prediction !== 'ERROR'     &&
                        data.prediction !== '') {

                        rawValEl.textContent = data.prediction;
                        statusEl.textContent = 'Detected: ' + data.prediction;

                        // Uniform threshold for all signs — equal treatment.
                        if (data.confidence > WORD_CONFIDENCE_THRESHOLD) {
                            if (data.prediction === potWordCandidate) {
                                wordStabilityCount++;
                            } else {
                                potWordCandidate = data.prediction;
                                wordStabilityCount = 1;
                            }

                            progressEl.style.width =
                                (wordStabilityCount / WORD_STABILITY_THRESHOLD) * 100 + '%';

                            if (wordStabilityCount >= WORD_STABILITY_THRESHOLD) {
                                if (data.prediction !== lastWordPrediction) {
                                    boardText += (boardText ? ' ' : '') + data.prediction;
                                    boardEl.textContent = boardText;
                                    lastWordPrediction = data.prediction;
                                }
                                // Reset stability state only — keep rolling buffer.
                                wordStabilityCount = 0;
                                potWordCandidate   = '';
                                progressEl.style.width = '0%';
                            }
                        } else {
                            wordStabilityCount = 0;
                            progressEl.style.width = '0%';
                        }

                    } else {
                        if (data.prediction === '') { lastWordPrediction = ''; }
                        wordStabilityCount = 0;
                        potWordCandidate   = '';
                        progressEl.style.width = '0%';
                    }

                    isPredicting = false;
                }).catch(() => { isPredicting = false; });
            }

        } else if (sequence.length < 30) {
            statusEl.textContent = 'Buffering...';
            progressEl.style.width = (sequence.length / 30) * 100 + '%';
        }
    }
}

// ─── Initialize MediaPipe Holistic (same settings as old app) ───
const holistic = new Holistic({
    locateFile: (f) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${f}`
});

holistic.setOptions({
    // modelComplexity 0 on mobile: significantly faster (less CPU), still accurate.
    // On desktop complexity 1 gives marginally better landmarks at no UX cost.
    modelComplexity: isMobile ? 0 : 1,
    smoothLandmarks: true,
    enableSegmentation: false,
    smoothSegmentation: false,
    refineFaceLandmarks: true,
    minDetectionConfidence: isMobile ? 0.6 : 0.5,
    minTrackingConfidence: 0.5
});

holistic.onResults(onResults);

// ─── Start camera (using Camera utility, same as old app) ───
const camera = new Camera(videoEl, {
    onFrame: async () => { await holistic.send({ image: videoEl }); },
    width: 640,
    height: 480
});

camera.start().then(() => {
    statusEl.textContent = 'Camera ready';
}).catch(err => {
    statusEl.textContent = 'Camera access denied';
    console.error('Camera error:', err);
});

// ════════════════════════════════════════════
// ─── SIGN VISUALIZER ───
// ════════════════════════════════════════════

const vizCanvas = document.getElementById('viz-canvas');
const vizCtx = vizCanvas.getContext('2d');
const wordSelectEl = document.getElementById('word-select');
const frameInfoEl = document.getElementById('viewer-frame-info');
const wordDisplayEl = document.getElementById('viewer-word-display');
const btnPlay = document.getElementById('btn-play');
const btnPause = document.getElementById('btn-pause');
const frameSlider = document.getElementById('frame-slider');
const scrubLabel = document.getElementById('scrub-label');

let vizFrames = [];      // array of 30 frames, each flat [546]
let vizFrame = 0;
let vizRAF = null;
let vizLastTime = 0;
let vizPaused = false;
const VIZ_FPS = 10;      // playback speed (frames per second)

// Hand skeleton connections (same indices as live mode)
const VIZ_HAND_CONN = [
    [0,1],[1,2],[2,3],[3,4],
    [0,5],[5,6],[6,7],[7,8],
    [5,9],[9,10],[10,11],[11,12],
    [9,13],[13,14],[14,15],[15,16],
    [13,17],[17,18],[18,19],[19,20],[0,17]
];

// Pose arm/shoulder connections (within the 6-point group: 0=LS,1=RS,2=LE,3=RE,4=LW,5=RW)
const VIZ_POSE_CONN = [[0,1],[0,2],[2,4],[1,3],[3,5]];

function populateWordList() {
    if (wordSelectEl.options.length > 1) return; // already loaded
    fetch('/word_list').then(r => r.json()).then(words => {
        words.forEach(w => {
            const opt = document.createElement('option');
            opt.value = w;
            opt.textContent = w.charAt(0).toUpperCase() + w.slice(1);
            wordSelectEl.appendChild(opt);
        });
    });
}

function stopAnimation() {
    if (vizRAF) { cancelAnimationFrame(vizRAF); vizRAF = null; }
    vizPaused = false;
    if (btnPause) btnPause.textContent = '⏸ Pause';
}

function togglePause() {
    if (!vizFrames.length) return;
    vizPaused = !vizPaused;
    btnPause.textContent = vizPaused ? '▶ Resume' : '⏸ Pause';
    if (!vizPaused && !vizRAF) playAnimation(); // resume
}

// Called when user drags the frame slider
function onScrub(val) {
    vizPaused = true;
    if (btnPause) btnPause.textContent = '▶ Resume';
    stopAnimation();
    vizFrame = parseInt(val, 10);
    if (vizFrames.length) {
        drawSkeleton(vizFrames[vizFrame]);
        frameInfoEl.textContent = `Frame ${vizFrame + 1} / ${vizFrames.length}`;
        scrubLabel.textContent = vizFrame;
    }
}

function loadAndPlay() {
    const word = wordSelectEl.value;
    if (!word) return;

    stopAnimation();
    vizPaused = false;
    btnPlay.disabled = true;
    btnPause.disabled = true;
    btnPlay.textContent = '⏳ Loading…';
    wordDisplayEl.textContent = '';

    fetch(`/sign_demo?word=${encodeURIComponent(word)}`)
        .then(r => r.json())
        .then(data => {
            if (data.error) { alert(data.error); return; }
            vizFrames = data.frames;  // 30 × 546
            vizFrame = 0;
            wordDisplayEl.textContent = word.toUpperCase();
            // Set up slider range to match frame count
            const maxFrame = vizFrames.length - 1;
            frameSlider.max = maxFrame;
            frameSlider.value = 0;
            document.getElementById('scrub-max').textContent = maxFrame;
            scrubLabel.textContent = '0';
            btnPlay.textContent = '▶ Play';
            btnPlay.disabled = false;
            btnPause.disabled = false;
            playAnimation();
        })
        .catch(() => {
            btnPlay.textContent = '▶ Play';
            btnPlay.disabled = false;
            btnPause.disabled = false;
        });
}

function playAnimation() {
    stopAnimation();
    vizPaused = false;
    btnPause.textContent = '⏸ Pause';
    vizLastTime = 0;

    function step(ts) {
        if (!vizFrames.length) return;
        if (!vizPaused && ts - vizLastTime >= 1000 / VIZ_FPS) {
            drawSkeleton(vizFrames[vizFrame]);
            frameInfoEl.textContent = `Frame ${vizFrame + 1} / ${vizFrames.length}`;
            frameSlider.value = vizFrame;
            scrubLabel.textContent = vizFrame;
            vizFrame++;
            if (vizFrame >= vizFrames.length) {
                if (document.getElementById('chk-loop').checked) {
                    vizFrame = 0;
                } else {
                    stopAnimation();
                    return;
                }
            }
            vizLastTime = ts;
        }
        vizRAF = requestAnimationFrame(step);
    }
    vizRAF = requestAnimationFrame(step);
}

/**
 * Draw one frame of skeleton data onto #viz-canvas.
 * Feature layout (546 total per frame):
 *   [0..62]   → right hand  (21 pts × 3)
 *   [63..125] → left hand   (21 pts × 3)
 *   [126..143]→ pose 6 pts (LS,RS,LE,RE,LW,RW) × 3
 *   [144..539]→ face 132 pts × 3
 *   [540..545]→ 6 explicit distances (ignored here)
 */
function drawSkeleton(flat) {
    const W = vizCanvas.width;
    const H = vizCanvas.height;
    vizCtx.clearRect(0, 0, W, H);

    // subtle gradient bg
    const bg = vizCtx.createRadialGradient(W/2, H/2, 20, W/2, H/2, W*0.7);
    bg.addColorStop(0, 'rgba(108,99,255,0.04)');
    bg.addColorStop(1, 'rgba(0,0,0,0)');
    vizCtx.fillStyle = bg;
    vizCtx.fillRect(0, 0, W, H);

    // Helper: extract N points starting at offset `off` (x,y used; z scaled for dot size)
    function pts(off, n) {
        const out = [];
        for (let i = 0; i < n; i++) {
            const x = flat[off + i*3];
            const y = flat[off + i*3 + 1];
            // Skip zero/absent landmarks
            if (x === 0 && y === 0) { out.push(null); continue; }
            out.push({ x: x * W, y: y * H });
        }
        return out;
    }

    const rh = pts(0,   21);
    const lh = pts(63,  21);
    const po = pts(126,  6);
    const fc = pts(144, 132);

    // ── Face: soft dim dots only ──
    vizCtx.fillStyle = 'rgba(180,180,220,0.18)';
    for (const p of fc) {
        if (!p) continue;
        vizCtx.beginPath();
        vizCtx.arc(p.x, p.y, 1.2, 0, Math.PI*2);
        vizCtx.fill();
    }

    // ── Pose: bright white lines + dots ──
    vizCtx.strokeStyle = 'rgba(255,255,255,0.55)';
    vizCtx.lineWidth = 2;
    for (const [a, b] of VIZ_POSE_CONN) {
        if (!po[a] || !po[b]) continue;
        vizCtx.beginPath();
        vizCtx.moveTo(po[a].x, po[a].y);
        vizCtx.lineTo(po[b].x, po[b].y);
        vizCtx.stroke();
    }
    vizCtx.fillStyle = 'rgba(255,255,255,0.8)';
    for (const p of po) {
        if (!p) continue;
        vizCtx.beginPath();
        vizCtx.arc(p.x, p.y, 4, 0, Math.PI*2);
        vizCtx.fill();
    }

    // ── Hands: skeleton lines then glowing dots ──
    function drawVizHand(landmarks, boneColor, dotColor, glowColor) {
        // Bone lines
        vizCtx.strokeStyle = boneColor;
        vizCtx.lineWidth = 2;
        for (const [a, b] of VIZ_HAND_CONN) {
            if (!landmarks[a] || !landmarks[b]) continue;
            vizCtx.beginPath();
            vizCtx.moveTo(landmarks[a].x, landmarks[a].y);
            vizCtx.lineTo(landmarks[b].x, landmarks[b].y);
            vizCtx.stroke();
        }
        // Glow halo then filled dot
        landmarks.forEach((p, i) => {
            if (!p) return;
            const r = i === 0 ? 6 : 3.5; // wrist is larger
            // outer glow
            vizCtx.beginPath();
            vizCtx.arc(p.x, p.y, r + 3, 0, Math.PI*2);
            vizCtx.fillStyle = glowColor;
            vizCtx.fill();
            // core dot
            vizCtx.beginPath();
            vizCtx.arc(p.x, p.y, r, 0, Math.PI*2);
            vizCtx.fillStyle = dotColor;
            vizCtx.fill();
        });
    }

    drawVizHand(rh,
        'rgba(108,99,255,0.7)',
        'rgba(140,130,255,1)',
        'rgba(108,99,255,0.2)');

    drawVizHand(lh,
        'rgba(72,207,173,0.7)',
        'rgba(100,230,200,1)',
        'rgba(72,207,173,0.2)');
}
