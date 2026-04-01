/* ============================================================
   AERIS v0.3 — Voice + WebSocket + Character Engine
   Always-on hotword "Hey AERIS" → ring+particle UI
   ============================================================ */

'use strict';

// ── Config ────────────────────────────────────────────────────
const CFG = {
    wsUrl: `ws://${location.host}/ws`,
    hotword: 'hey aeris',
    greetings: [
        'Online and ready, sir.',
        'AERIS systems nominal. At your service.',
        'Good to hear from you. How can I assist?',
        'All systems online. What do you need?',
    ],
    ttsEnabled: true,
    micMuted: false,
    ttsRate: 0.93,
    ttsPitch: 0.82,
};

// ── State ─────────────────────────────────────────────────────
const ST = { IDLE: 'idle', LISTEN: 'listening', THINK: 'thinking', SPEAK: 'speaking', GREET: 'greeting', OFF: 'offline' };
let curState = ST.IDLE;
let aerisActive = false;    // Has hotword been detected → AERIS is awake
let ws = null;
let recognition = null;
let audioCtx = null;
let analyser = null;
let waveViz = null;
let particles = null;
let pendingConfirm = null;  // Resolver for in-flight confirmation requests

// ── DOM refs ──────────────────────────────────────────────────
const $body = document.body;
const $pCanvas = document.getElementById('particleCanvas');
const $wCanvas = document.getElementById('waveformCanvas');
const $statusDot = document.getElementById('statusDot');
const $statusTxt = document.getElementById('statusText');
const $tBody = document.getElementById('transcriptBody');
const $intent = document.getElementById('intentReadout');
const $conn = document.getElementById('connIndicator');
const $connLbl = $conn.querySelector('.conn-label');
const $debug = document.getElementById('debugPanel');
const $debugBody = document.getElementById('debugBody');
const $bootBar = document.getElementById('bootBarFill');
const $bootMsg = document.getElementById('bootMessage');
const $bootOvl = document.getElementById('bootOverlay');

// ── State Machine ─────────────────────────────────────────────
function setState(s) {
    curState = s;
    $body.dataset.state = s;

    const MAP = {
        [ST.IDLE]: { text: 'Say "Hey AERIS" to activate', dot: '#404060' },
        [ST.LISTEN]: { text: 'Listening…', dot: '#00cfff' },
        [ST.THINK]: { text: 'Processing…', dot: '#ffaa00' },
        [ST.SPEAK]: { text: 'Speaking…', dot: '#00ffb3' },
        [ST.GREET]: { text: 'AERIS Online', dot: '#ffffff' },
        [ST.OFF]: { text: 'Connection Lost', dot: '#ff3d6b' },
    };
    const cfg = MAP[s] || MAP[ST.IDLE];
    $statusTxt.textContent = cfg.text;
    $statusDot.style.background = cfg.dot;
    $statusDot.style.boxShadow = s !== ST.IDLE ? `0 0 10px ${cfg.dot}` : 'none';

    if (particles) particles.setSpeedMult({ idle: 1, listening: 2.5, thinking: 4.5, speaking: 2, greeting: 6, offline: 0.3 }[s] || 1);

    // Show waveform only when listening
    $wCanvas.style.opacity = s === ST.LISTEN ? '1' : '0';
    if (s === ST.LISTEN && waveViz) waveViz.start();
    else if (waveViz) waveViz.stop();
}

// ═══════════════════════════════════════════════════════════════
// PARTICLE SYSTEM
// ═══════════════════════════════════════════════════════════════
class ParticleSystem {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.cx = canvas.width / 2;
        this.cy = canvas.height / 2;
        this.speedMult = 1;
        this.raf = null;
        this.particles = [];
        this._init();
    }

    _init() {
        const rings = [
            { n: 14, r: 135, spd: 0.28, sz: 2.2, col: '#00cfff' },
            { n: 20, r: 172, spd: -0.18, sz: 1.5, col: '#0055ff' },
            { n: 9, r: 210, spd: 0.13, sz: 3.0, col: '#00cfff' },
            { n: 26, r: 230, spd: -0.09, sz: 1.1, col: '#0099ff' },
        ];
        rings.forEach(ring => {
            for (let i = 0; i < ring.n; i++) {
                this.particles.push({
                    angle: (Math.PI * 2 / ring.n) * i + Math.random() * 0.4,
                    r: ring.r + (Math.random() - 0.5) * 14,
                    spd: ring.spd * (0.75 + Math.random() * 0.5),
                    sz: ring.sz * (0.7 + Math.random() * 0.6),
                    col: ring.col,
                    alpha: 0.4 + Math.random() * 0.5,
                    pOff: Math.random() * Math.PI * 2,
                    trail: [],
                    trailLen: 4 + Math.floor(Math.random() * 5),
                });
            }
        });
    }

    setSpeedMult(m) { this.speedMult = m; }

    _tick(t) {
        const ctx = this.ctx;
        ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        this.particles.forEach(p => {
            // push to trail
            const px = this.cx + Math.cos(p.angle) * p.r;
            const py = this.cy + Math.sin(p.angle) * p.r;
            p.trail.push({ x: px, y: py });
            if (p.trail.length > p.trailLen) p.trail.shift();

            // advance
            p.angle += p.spd * 0.012 * this.speedMult;
            p.alpha = 0.35 + Math.sin(t * 0.0018 + p.pOff) * 0.3;

            // draw trail
            for (let i = 1; i < p.trail.length; i++) {
                const frac = i / p.trail.length;
                ctx.beginPath();
                ctx.setLineDash([]);
                const a = Math.floor(frac * p.alpha * 180).toString(16).padStart(2, '0');
                ctx.strokeStyle = p.col + a;
                ctx.lineWidth = p.sz * frac * 0.9;
                ctx.moveTo(p.trail[i - 1].x, p.trail[i - 1].y);
                ctx.lineTo(p.trail[i].x, p.trail[i].y);
                ctx.stroke();
            }

            // glow dot
            const g = ctx.createRadialGradient(px, py, 0, px, py, p.sz * 3.5);
            g.addColorStop(0, p.col + 'ff');
            g.addColorStop(0.4, p.col + '88');
            g.addColorStop(1, p.col + '00');
            ctx.beginPath();
            ctx.fillStyle = g;
            ctx.arc(px, py, p.sz * 3.5, 0, Math.PI * 2);
            ctx.fill();

            // core white
            ctx.beginPath();
            ctx.fillStyle = `rgba(255,255,255,${p.alpha * 0.9})`;
            ctx.arc(px, py, p.sz * 0.7, 0, Math.PI * 2);
            ctx.fill();
        });
    }

    start() {
        const loop = (t) => {
            this._tick(t);
            this.raf = requestAnimationFrame(loop);
        };
        this.raf = requestAnimationFrame(loop);
    }

    stop() { if (this.raf) cancelAnimationFrame(this.raf); }
}

// ═══════════════════════════════════════════════════════════════
// WAVEFORM VISUALIZER
// ═══════════════════════════════════════════════════════════════
class WaveformViz {
    constructor(canvas, analyser) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.analyser = analyser;
        this.raf = null;
        this.live = false;
    }

    start() {
        if (this.live) return;
        this.live = true;
        const buf = new Uint8Array(this.analyser.frequencyBinCount);
        const draw = () => {
            if (!this.live) return;
            this.raf = requestAnimationFrame(draw);
            this.analyser.getByteTimeDomainData(buf);
            const ctx = this.ctx, W = this.canvas.width, H = this.canvas.height;
            ctx.clearRect(0, 0, W, H);

            // mid line
            ctx.strokeStyle = 'rgba(0,207,255,0.15)';
            ctx.lineWidth = 1;
            ctx.setLineDash([4, 4]);
            ctx.beginPath(); ctx.moveTo(0, H / 2); ctx.lineTo(W, H / 2); ctx.stroke();
            ctx.setLineDash([]);

            // waveform
            ctx.lineWidth = 2.5;
            ctx.strokeStyle = '#00cfff';
            ctx.shadowBlur = 8;
            ctx.shadowColor = '#00cfff';
            ctx.beginPath();
            const sw = W / buf.length;
            let x = 0;
            for (let i = 0; i < buf.length; i++) {
                const y = (buf[i] / 128) * (H / 2);
                i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
                x += sw;
            }
            ctx.stroke();
            ctx.shadowBlur = 0;
        };
        draw();
    }

    stop() {
        this.live = false;
        if (this.raf) cancelAnimationFrame(this.raf);
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }
}

// ═══════════════════════════════════════════════════════════════
// WEBSOCKET
// ═══════════════════════════════════════════════════════════════
function connectWS() {
    try { ws = new WebSocket(CFG.wsUrl); } catch (e) { scheduleReconnect(); return; }

    ws.onopen = () => {
        $conn.classList.add('connected');
        $connLbl.textContent = 'Connected';
        if (curState === ST.OFF) setState(ST.IDLE);
    };

    ws.onmessage = (ev) => {
        try { handleMsg(JSON.parse(ev.data)); }
        catch (e) { console.error('[WS] parse error', e); }
    };

    ws.onclose = () => {
        $conn.classList.remove('connected');
        $connLbl.textContent = 'Reconnecting…';
        setState(ST.OFF);
        scheduleReconnect();
    };

    ws.onerror = () => ws.close();
}

function scheduleReconnect() { setTimeout(connectWS, 3000); }

function wsSend(obj) {
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj));
    else addEntry('system', 'Not connected to AERIS server.', 'error');
}

function handleMsg(msg) {
    if (msg.type === 'state') { setState(msg.state); return; }

    const resp = msg.response || '';

    if (msg.type === 'confirmation_required') {
        setState(ST.IDLE);
        addConfirmEntry(resp, msg.action, msg.params);
        if (CFG.ttsEnabled && resp) speak(resp);
        return;
    }

    if (msg.type === 'response' || msg.type === 'error') {
        const intent = msg.intent || '';
        $intent.textContent = intent ? `◈ ${intent}` : '';
        addEntry('aeris', resp, intent);

        if (msg.session_closed) { aerisActive = false; setState(ST.IDLE); return; }
        if (CFG.ttsEnabled && resp) speak(resp);
        else setState(aerisActive ? ST.LISTEN : ST.IDLE);
    }
}

function sendInput(text) {
    setState(ST.THINK);
    wsSend({ type: 'input', text });
}

// ═══════════════════════════════════════════════════════════════
// SPEECH RECOGNITION
// ═══════════════════════════════════════════════════════════════
function initSpeech() {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) {
        addEntry('system', 'Speech recognition requires Chrome or Edge.', 'error');
        return;
    }

    recognition = new SR();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = 'en-US';
    recognition.maxAlternatives = 1;

    let finalBuf = '';
    let sendTimer = null;

    recognition.onresult = (ev) => {
        let interim = '';
        for (let i = ev.resultIndex; i < ev.results.length; i++) {
            if (ev.results[i].isFinal) finalBuf += ev.results[i][0].transcript + ' ';
            else interim += ev.results[i][0].transcript;
        }

        const full = (finalBuf + interim).trim().toLowerCase();

        // ── Hotword detection ──────────────────────────────────
        if (!aerisActive) {
            let isHotword = false;
            const variants = [
                'hey aeris', 'hey aris', 'hey iris', 'hey ares', 'hey eris', 
                'hey areas', 'hey aeros', 'hey eric', 'hey erris', 'hey heris', 
                'hey aras', 'hey eros', 'hey heiress', 'aeris', 'hey harris'
            ];
            
            if (variants.some(v => full.includes(v))) {
                isHotword = true;
            } else if (['hey', 'hello', 'hi', 'wake up'].includes(full)) {
                // strict match if the STT cuts off the second word entirely
                isHotword = true;
            }

            if (isHotword) {
                aerisActive = true;
                finalBuf = '';
                greet();
                return;
            }
        }

        if (!aerisActive) return;

        // Show live transcript
        showInterim(finalBuf + interim);

        // Debounce: send after 800ms silence on final
        clearTimeout(sendTimer);
        if (finalBuf.trim()) {
            sendTimer = setTimeout(() => {
                const cmd = finalBuf.trim();
                finalBuf = '';
                clearInterim();
                addEntry('user', cmd);
                sendInput(cmd);
            }, 800);
        }
    };

    recognition.onend = () => {
        if (!CFG.micMuted) setTimeout(() => { try { recognition.start(); } catch (_) { } }, 150);
        // If we were listening, go back to idle briefly
        if (curState === ST.LISTEN) setState(aerisActive ? ST.LISTEN : ST.IDLE);
    };

    recognition.onerror = (e) => {
        if (!['no-speech', 'aborted'].includes(e.error)) console.warn('[SR]', e.error);
    };

    try { recognition.start(); } catch (_) { }
}

// ═══════════════════════════════════════════════════════════════
// TTS
// ═══════════════════════════════════════════════════════════════
function speak(text) {
    speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    u.rate = CFG.ttsRate;
    u.pitch = CFG.ttsPitch;
    u.volume = 1;

    // Try to pick a good voice
    const pick = () => {
        const voices = speechSynthesis.getVoices();
        return voices.find(v =>
            v.name.includes('David') ||
            v.name.includes('Mark') ||
            v.name.includes('Daniel') ||
            (v.lang.startsWith('en') && v.localService)
        ) || null;
    };

    const v = pick();
    if (v) u.voice = v;
    else speechSynthesis.onvoiceschanged = () => { const vv = pick(); if (vv) u.voice = vv; };

    setState(ST.SPEAK);
    u.onend = () => { setState(aerisActive ? ST.LISTEN : ST.IDLE); };
    u.onerror = () => { setState(aerisActive ? ST.LISTEN : ST.IDLE); };

    speechSynthesis.speak(u);
}

// ═══════════════════════════════════════════════════════════════
// GREETING
// ═══════════════════════════════════════════════════════════════
function greet() {
    setState(ST.GREET);
    const msg = CFG.greetings[Math.floor(Math.random() * CFG.greetings.length)];

    // typewriter status
    typeStatus('AERIS ONLINE', () => {
        setTimeout(() => {
            setState(ST.LISTEN);
            addEntry('aeris', msg, 'GREETING');
            speak(msg);
        }, 400);
    });
}

function typeStatus(text, cb) {
    let i = 0;
    $statusTxt.textContent = '';
    const iv = setInterval(() => {
        $statusTxt.textContent = text.slice(0, ++i);
        if (i >= text.length) { clearInterval(iv); if (cb) setTimeout(cb, 300); }
    }, 55);
}

// ═══════════════════════════════════════════════════════════════
// TRANSCRIPT
// ═══════════════════════════════════════════════════════════════
function clearEmpty() {
    const e = $tBody.querySelector('.transcript-empty');
    if (e) e.remove();
}

function addEntry(role, text, intent = '') {
    clearEmpty();
    clearInterim();
    const roleMap = { user: 'YOU', aeris: 'AERIS', system: 'SYS', error: 'ERR' };
    const div = document.createElement('div');
    div.className = `t-entry ${role}`;
    div.innerHTML = `
        <span class="t-role">${roleMap[role] || role.toUpperCase()}</span>
        <div class="t-bubble">
            ${intent && role === 'aeris' ? `<div class="t-badge">${intent}</div>` : ''}
            ${escHtml(text)}
        </div>`;
    $tBody.appendChild(div);
    $tBody.scrollTop = $tBody.scrollHeight;
}

function addConfirmEntry(text, action, params) {
    clearEmpty();
    clearInterim();
    const div = document.createElement('div');
    div.className = 't-entry aeris';

    const yesId = 'confirm-yes-' + Date.now();
    const noId = 'confirm-no-' + Date.now();

    div.innerHTML = `
        <span class="t-role">AERIS</span>
        <div class="t-bubble">
            <div class="t-badge">CONFIRMATION REQUIRED</div>
            ${escHtml(text)}
            <div class="confirm-bubble">
                <button class="confirm-yes" id="${yesId}">✓ Confirm</button>
                <button class="confirm-no"  id="${noId}">✕ Cancel</button>
            </div>
        </div>`;
    $tBody.appendChild(div);
    $tBody.scrollTop = $tBody.scrollHeight;

    document.getElementById(yesId).onclick = () => { wsSend({ confirmed: true }); div.remove(); };
    document.getElementById(noId).onclick = () => { wsSend({ confirmed: false }); div.remove(); addEntry('system', 'Action cancelled.'); };
}

let _interimEl = null;
function showInterim(text) {
    clearEmpty();
    if (!_interimEl) {
        _interimEl = document.createElement('div');
        _interimEl.className = 't-entry user interim';
        _interimEl.innerHTML = `<span class="t-role">YOU</span><div class="t-bubble"></div>`;
        $tBody.appendChild(_interimEl);
    }
    _interimEl.querySelector('.t-bubble').textContent = text;
    $tBody.scrollTop = $tBody.scrollHeight;
}
function clearInterim() { if (_interimEl) { _interimEl.remove(); _interimEl = null; } }

function escHtml(s) {
    return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

// ═══════════════════════════════════════════════════════════════
// AUDIO VISUALIZER (real mic input)
// ═══════════════════════════════════════════════════════════════
async function initAudio() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        analyser = audioCtx.createAnalyser();
        analyser.fftSize = 1024;
        audioCtx.createMediaStreamSource(stream).connect(analyser);
        waveViz = new WaveformViz($wCanvas, analyser);
    } catch (e) {
        console.warn('[AERIS] Mic not available:', e.message);
    }
}

// ═══════════════════════════════════════════════════════════════
// STARS BACKGROUND
// ═══════════════════════════════════════════════════════════════
function initStars() {
    const bg = document.getElementById('starsBg');
    for (let i = 0; i < 180; i++) {
        const s = document.createElement('div');
        s.className = 'star';
        s.style.cssText = `
            left:${Math.random() * 100}%;
            top:${Math.random() * 100}%;
            width:${Math.random() * 2 + 0.3}px;
            height:${Math.random() * 2 + 0.3}px;
            animation-delay:${Math.random() * 4}s;
            animation-duration:${2 + Math.random() * 3}s;
        `;
        bg.appendChild(s);
    }
}

// ═══════════════════════════════════════════════════════════════
// BOOT SEQUENCE
// ═══════════════════════════════════════════════════════════════
function runBoot(cb) {
    const messages = [
        'LOADING NEURAL MODULES…',
        'CALIBRATING INTENT CLASSIFIER…',
        'INITIALIZING VOICE ENGINE…',
        'CONNECTING TO AERIS CORE…',
        'SYSTEMS ONLINE.',
    ];
    let pct = 0;
    let mi = 0;

    const iv = setInterval(() => {
        pct = Math.min(pct + Math.random() * 4 + 1, 100);
        $bootBar.style.width = pct + '%';
        if (mi < messages.length && pct > (mi + 1) * (100 / messages.length)) {
            $bootMsg.textContent = messages[mi++];
        }
        if (pct >= 100) {
            clearInterval(iv);
            $bootMsg.textContent = 'READY.';
            setTimeout(() => {
                $bootOvl.classList.add('hidden');
                setTimeout(cb, 700);
            }, 600);
        }
    }, 60);
}

// ═══════════════════════════════════════════════════════════════
// CONTROLS
// ═══════════════════════════════════════════════════════════════
document.getElementById('muteBtn').addEventListener('click', () => {
    CFG.micMuted = !CFG.micMuted;
    const btn = document.getElementById('muteBtn');
    document.getElementById('muteIcon').textContent = CFG.micMuted ? '🔇' : '🎤';
    btn.querySelector('.ctrl-label').textContent = CFG.micMuted ? 'Mic Muted' : 'Mic Active';
    btn.classList.toggle('muted', CFG.micMuted);
    if (CFG.micMuted && recognition) recognition.stop();
    else if (recognition) { try { recognition.start(); } catch (_) { } }
});

document.getElementById('ttsBtn').addEventListener('click', () => {
    CFG.ttsEnabled = !CFG.ttsEnabled;
    document.getElementById('ttsIcon').textContent = CFG.ttsEnabled ? '🔊' : '🔇';
    document.getElementById('ttsBtn').querySelector('.ctrl-label').textContent = CFG.ttsEnabled ? 'Voice On' : 'Voice Off';
    if (!CFG.ttsEnabled) speechSynthesis.cancel();
});

document.getElementById('clearBtn').addEventListener('click', () => {
    $tBody.innerHTML = '<div class="transcript-empty"><span class="empty-icon">◎</span><span>Waiting for voice input…</span></div>';
    $intent.textContent = '';
});

document.getElementById('debugBtn').addEventListener('click', async () => {
    $debug.classList.toggle('visible');
    if ($debug.classList.contains('visible')) {
        try {
            const res = await fetch('/status');
            const d = await res.json();
            $debugBody.innerHTML = Object.entries(d)
                .map(([k, v]) => `<div><span style="color:rgba(0,207,255,0.6)">${k}</span>: ${JSON.stringify(v)}</div>`)
                .join('');
        } catch { $debugBody.textContent = 'Could not reach /status'; }
    }
});

document.getElementById('debugClose').addEventListener('click', () => $debug.classList.remove('visible'));

// ═══════════════════════════════════════════════════════════════
// INIT
// ═══════════════════════════════════════════════════════════════
async function init() {
    initStars();

    runBoot(async () => {
        // Particle system
        particles = new ParticleSystem($pCanvas);
        particles.start();

        setState(ST.IDLE);
        connectWS();

        // Audio + Speech
        await initAudio();
        initSpeech();
    });
}

document.addEventListener('DOMContentLoaded', init);
