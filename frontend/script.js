/* ═══════════════════════════════════════════════════════
   GarbageBot — Continuous-World Dashboard Logic
   Policy chain: Fine-tuned LLM → Q-table → BFS fallback
   ═══════════════════════════════════════════════════════ */

const API_BASE = "http://localhost:7861";

// ── DOM ───────────────────────────────────────────────────
const statusDot        = document.getElementById("status-dot");
const statusLabel      = document.getElementById("status-label");
const policyBadge      = document.getElementById("policy-badge");
const policyLabel      = document.getElementById("policy-label");
const taskSelect       = document.getElementById("task-select");
const speedSlider      = document.getElementById("speed-slider");
const speedVal         = document.getElementById("speed-val");
const resetBtn         = document.getElementById("reset-btn");
const autoBtn          = document.getElementById("auto-btn");
const manualBtn        = document.getElementById("manual-btn");
const clearLogBtn      = document.getElementById("clear-log");

const envGrid          = document.getElementById("env-grid");
const particleLayer    = document.getElementById("particle-layer");
const batteryProgress  = document.getElementById("battery-progress");
const batteryText      = document.getElementById("battery-text");
const scoreText        = document.getElementById("score-text");
const inventoryText    = document.getElementById("inventory-text");
const stepCounter      = document.getElementById("step-counter");
const episodeScoreChip = document.getElementById("episode-score-chip");
const logFeed          = document.getElementById("log-feed");
const rewardCanvas     = document.getElementById("reward-chart");

// ── State ─────────────────────────────────────────────────
let autoMode      = false;
let autoTimer     = null;
let currentState  = null;
let robotEntity   = null;
let stepCount     = 0;
let totalReward   = 0;
let rewardHistory = [];
let maxBattery    = 30;
let stepDelay     = 500;

// World dimensions (set on reset)
let WORLD_W = 5, WORLD_H = 5;
const CELL = 52;   // must match CSS --cell

// ── Speed slider ──────────────────────────────────────────
speedSlider.addEventListener("input", () => {
    stepDelay = parseInt(speedSlider.value);
    speedVal.textContent = `${stepDelay}ms`;
    // Update slider gradient fill
    const pct = ((stepDelay - 100) / 1400) * 100;
    speedSlider.style.background = `linear-gradient(90deg, var(--blue) ${pct}%, rgba(255,255,255,.15) ${pct}%)`;
    syncRobotTransition();
    if (autoMode) { clearInterval(autoTimer); autoTimer = setInterval(stepEnv, stepDelay); }
});

// Keep robot transition duration slightly under step delay
function syncRobotTransition() {
    if (!robotEntity) return;
    const dur = Math.min(Math.max(stepDelay * 0.72, 160), 480);
    envGrid.style.setProperty("--move-dur", `${dur}ms`);
}

// ── Log helpers ───────────────────────────────────────────
function addLog(msg, source = "sys") {
    const ph = logFeed.querySelector(".placeholder");
    if (ph) ph.remove();

    const entry  = document.createElement("div");
    entry.className = "log-entry";

    const badge = document.createElement("span");
    badge.className = `log-badge ${source === "q_table" ? "q-table" : source}`;
    badge.textContent = source.replace("_","-").toUpperCase();

    const text = document.createElement("span");
    text.textContent = msg;

    entry.append(badge, text);
    logFeed.prepend(entry);
    while (logFeed.children.length > 65) logFeed.removeChild(logFeed.lastChild);
}

clearLogBtn.addEventListener("click", () => {
    logFeed.innerHTML = `<p class="placeholder">Log cleared…</p>`;
});

// ── Mini reward chart ─────────────────────────────────────
function drawChart() {
    const ctx = rewardCanvas.getContext("2d");
    const W = rewardCanvas.width, H = rewardCanvas.height;
    ctx.clearRect(0, 0, W, H);
    if (rewardHistory.length < 2) return;

    const maxR = Math.max(...rewardHistory.map(Math.abs), .1);
    const step = W / (rewardHistory.length - 1);
    const pts  = rewardHistory.map((v, i) => [i * step, H - ((v + maxR) / (2 * maxR)) * H]);

    // Fill
    const grad = ctx.createLinearGradient(0, 0, 0, H);
    grad.addColorStop(0, "rgba(59,158,255,.5)");
    grad.addColorStop(1, "rgba(59,158,255,0)");
    ctx.beginPath();
    pts.forEach(([x, y], i) => i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y));
    ctx.lineTo(pts[pts.length-1][0], H);
    ctx.lineTo(0, H); ctx.closePath();
    ctx.fillStyle = grad; ctx.fill();

    // Line
    ctx.beginPath();
    pts.forEach(([x, y], i) => i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y));
    ctx.strokeStyle = "#3b9eff"; ctx.lineWidth = 2;
    ctx.lineJoin = "round"; ctx.stroke();

    // Latest dot
    const [lx, ly] = pts[pts.length-1];
    ctx.beginPath(); ctx.arc(lx, ly, 3.5, 0, Math.PI*2);
    ctx.fillStyle = "#a5c8ff"; ctx.fill();
}

// ── Particles ─────────────────────────────────────────────
function spawnParticles(px, py) {
    const colors = ["#c084fc","#818cf8","#3b9eff","#2dd4bf","#fbbf24"];
    for (let i = 0; i < 14; i++) {
        const p = document.createElement("div");
        p.className = "particle";
        const angle = (i / 14) * Math.PI * 2;
        const dist  = 28 + Math.random() * 42;
        const size  = 4 + Math.random() * 6;
        p.style.cssText = `
            left:${px}px; top:${py}px;
            width:${size}px; height:${size}px;
            background:${colors[i % colors.length]};
            box-shadow:0 0 6px ${colors[i%colors.length]};
            --tx:translate(${Math.cos(angle)*dist}px,${Math.sin(angle)*dist}px);
        `;
        particleLayer.appendChild(p);
        setTimeout(() => p.remove(), 780);
    }
}

// ── Trail ghost ───────────────────────────────────────────
function addTrail(left, top) {
    const g = document.createElement("div");
    g.className = "trail-ghost";
    g.style.left = `${left}px`;
    g.style.top  = `${top}px`;
    envGrid.appendChild(g);
    setTimeout(() => g.remove(), 1100);
}

// ── World coordinates ─────────────────────────────────────
// Grid origin is at top-left of envGrid; y-axis is flipped (0 = bottom)
function wx(x)     { return x * CELL; }
function wy(y, H)  { return (H - 1 - y) * CELL; }

// ── Direction → emoji ─────────────────────────────────────
const DIR_EMOJI = { UP:"🤖", DOWN:"🤖", LEFT:"🤖", RIGHT:"🤖", COLLECT:"🤖" };
// Could use arrows but robot emoji is cleaner; direction handled by trail

// ── Grid render ───────────────────────────────────────────
function renderGrid(obs, isReset = false) {
    const [W, H] = obs.grid_size;
    WORLD_W = W; WORLD_H = H;
    const worldPx = W * CELL;
    const worldPy = H * CELL;

    if (isReset) {
        envGrid.innerHTML = "";
        envGrid.style.width  = `${worldPx}px`;
        envGrid.style.height = `${worldPy}px`;
        envGrid.style.gridTemplateColumns = `repeat(${W}, ${CELL}px)`;
        envGrid.style.gridTemplateRows    = `repeat(${H}, ${CELL}px)`;

        // Background tile lines aligned to CELL size
        envGrid.style.backgroundSize = `${CELL}px ${CELL}px, ${CELL}px ${CELL}px, 100% 100%`;

        // Transparent click-target cells
        for (let y = H - 1; y >= 0; y--) {
            for (let x = 0; x < W; x++) {
                const cell = document.createElement("div");
                cell.className = "cell";
                cell.dataset.x = x; cell.dataset.y = y;
                cell.addEventListener("click", () => toggleGarbage(x, y));
                envGrid.appendChild(cell);
            }
        }

        // 3D obstacle walls
        obs.obstacle_positions.forEach(([x, y]) => {
            const el = document.createElement("div");
            el.className = "world-obstacle";
            el.style.left   = `${wx(x)}px`;
            el.style.top    = `${wy(y, H)}px`;
            el.style.width  = `${CELL}px`;
            el.style.height = `${CELL}px`;
            envGrid.appendChild(el);
        });

        // Robot entity
        robotEntity = document.createElement("div");
        robotEntity.className   = "robot-entity";
        robotEntity.textContent = "🤖";
        robotEntity.style.width  = `${CELL}px`;
        robotEntity.style.height = `${CELL}px`;
        robotEntity.style.left   = `${wx(obs.robot_position[0])}px`;
        robotEntity.style.top    = `${wy(obs.robot_position[1], H)}px`;
        envGrid.appendChild(robotEntity);
        syncRobotTransition();
    }

    // Smooth robot move
    if (robotEntity) {
        const nl = wx(obs.robot_position[0]);
        const nt = wy(obs.robot_position[1], H);
        const ol = parseInt(robotEntity.style.left) || nl;
        const ot = parseInt(robotEntity.style.top)  || nt;
        if (nl !== ol || nt !== ot) addTrail(ol, ot);
        robotEntity.style.left = `${nl}px`;
        robotEntity.style.top  = `${nt}px`;
    }

    // Re-render garbage
    document.querySelectorAll(".world-garbage").forEach(g => g.remove());
    obs.garbage_positions.forEach(([x, y]) => {
        const el = document.createElement("div");
        el.className = "world-garbage";
        el.style.left   = `${wx(x)}px`;
        el.style.top    = `${wy(y, H)}px`;
        el.style.width  = `${CELL}px`;
        el.style.height = `${CELL}px`;
        el.innerHTML    = `<span>🗑️</span>`;
        el.addEventListener("click", () => toggleGarbage(x, y));
        envGrid.appendChild(el);
    });
}

// ── Telemetry ─────────────────────────────────────────────
function updateTelemetry(obs, reward, done) {
    if (obs.battery_level > maxBattery) maxBattery = obs.battery_level;
    const pct = Math.max(0, (obs.battery_level / maxBattery) * 100);
    batteryProgress.style.width = `${pct}%`;
    batteryText.textContent     = `${obs.battery_level} / ${maxBattery}`;

    if      (pct > 55) batteryProgress.style.background = "#34d399";
    else if (pct > 25) batteryProgress.style.background = "#fbbf24";
    else               batteryProgress.style.background = "#fb7185";

    inventoryText.textContent = obs.inventory_count;

    if (reward !== undefined) {
        totalReward += reward;
        rewardHistory.push(totalReward);
        if (rewardHistory.length > 80) rewardHistory.shift();
        scoreText.textContent        = totalReward.toFixed(2);
        episodeScoreChip.textContent = `Score ${totalReward.toFixed(2)}`;
        drawChart();
    }

    stepCounter.textContent = `Step ${stepCount}`;
    addLog(obs.message, "sys");
}

// ── Policy badge ──────────────────────────────────────────
const POLICY_STYLES = {
    llm:     { color:"#3b9eff", border:"rgba(59,158,255,.6)" },
    bfs:     { color:"#2dd4bf", border:"rgba(45,212,191,.6)" },
    q_table: { color:"#fbbf24", border:"rgba(251,191,36,.6)" },
    sys:     { color:"#7ea8d8", border:"rgba(126,168,216,.3)" },
};
function showPolicy(source, action) {
    const s = POLICY_STYLES[source] || POLICY_STYLES.sys;
    policyLabel.textContent      = `${source.replace("_","-").toUpperCase()} → ${action}`;
    policyBadge.style.borderColor = s.border;
    policyBadge.style.color       = s.color;
    policyBadge.classList.add("active");
}

// ── BFS fallback ──────────────────────────────────────────
function bfsMove(rPos, target, obstacles, W, H) {
    if (rPos[0]===target[0] && rPos[1]===target[1]) return "COLLECT";
    const obs  = new Set(obstacles.map(([x,y]) => `${x},${y}`));
    const dirs = [["RIGHT",1,0],["LEFT",-1,0],["UP",0,1],["DOWN",0,-1]];
    const q    = [{pos:[...rPos], first:null}];
    const vis  = new Set([`${rPos[0]},${rPos[1]}`]);

    while (q.length) {
        const {pos, first} = q.shift();
        for (const [name, dx, dy] of dirs) {
            const nx = pos[0]+dx, ny = pos[1]+dy;
            if (nx<0||nx>=W||ny<0||ny>=H) continue;
            const key = `${nx},${ny}`;
            if (obs.has(key)||vis.has(key)) continue;
            const move = first||name;
            if (nx===target[0]&&ny===target[1]) return move;
            vis.add(key); q.push({pos:[nx,ny], first:move});
        }
    }
    return null;
}

function nnOrder(start, targets, obstacles, W, H) {
    function dist(a, b) {
        if (a[0]===b[0]&&a[1]===b[1]) return 0;
        const obs=new Set(obstacles.map(([x,y])=>`${x},${y}`));
        const dirs=[[1,0],[-1,0],[0,1],[0,-1]];
        const q=[{pos:[...a],d:0}];const vis=new Set([`${a[0]},${a[1]}`]);
        while(q.length){const{pos,d}=q.shift();for(const[dx,dy]of dirs){const nx=pos[0]+dx,ny=pos[1]+dy;if(nx<0||nx>=W||ny<0||ny>=H)continue;const k=`${nx},${ny}`;if(obs.has(k)||vis.has(k))continue;if(nx===b[0]&&ny===b[1])return d+1;vis.add(k);q.push({pos:[nx,ny],d:d+1});}}
        return Infinity;
    }
    let rem=[...targets],cur=[...start],ord=[];
    while(rem.length){
        let best=rem[0],bD=dist(cur,best);
        for(const t of rem){const d=dist(cur,t);if(d<bD){bD=d;best=t;}}
        ord.push(best);
        rem=rem.filter(t=>!(t[0]===best[0]&&t[1]===best[1]));
        cur=[...best];
    }
    return ord;
}

function localFallback(obs) {
    if (!obs.garbage_positions.length) return "UP";
    const r = obs.robot_position;
    if (obs.garbage_positions.some(([x,y]) => x===r[0]&&y===r[1])) return "COLLECT";
    const ordered = nnOrder(r, obs.garbage_positions, obs.obstacle_positions, obs.grid_size[0], obs.grid_size[1]);
    return bfsMove(r, ordered[0], obs.obstacle_positions, obs.grid_size[0], obs.grid_size[1]) || "RIGHT";
}

// ── Custom garbage toggle ─────────────────────────────────
async function toggleGarbage(x, y) {
    if (!currentState || autoMode) return;
    if (currentState.obstacle_positions.some(([ox,oy]) => ox===x&&oy===y)) return;
    if (currentState.robot_position[0]===x && currentState.robot_position[1]===y) return;

    const has  = currentState.garbage_positions.some(([gx,gy]) => gx===x&&gy===y);
    const next = has
        ? currentState.garbage_positions.filter(([gx,gy]) => !(gx===x&&gy===y))
        : [...currentState.garbage_positions, [x, y]];

    try {
        const res = await fetch(`${API_BASE}/configure`, {
            method: "POST", headers:{"Content-Type":"application/json"},
            body: JSON.stringify({task_id: taskSelect.value, garbage_positions: next})
        });
        const data = await res.json();
        currentState = data.observation;
        renderGrid(currentState);
        addLog(`Garbage ${has?"removed":"placed"} at (${x},${y})  ·  ${next.length} remaining`, "sys");
    } catch (e) { addLog(`Config error: ${e.message}`, "sys"); }
}

// ── Reset ─────────────────────────────────────────────────
async function resetEnv() {
    if (autoMode) toggleAutoMode();
    stepCount=0; totalReward=0; rewardHistory=[];
    scoreText.textContent        = "0.00";
    episodeScoreChip.textContent = "Score 0.00";
    stepCounter.textContent      = "Step 0";
    policyLabel.textContent      = "–";
    drawChart();

    try {
        const res  = await fetch(`${API_BASE}/reset`, {
            method:"POST", headers:{"Content-Type":"application/json"},
            body: JSON.stringify({task_id: taskSelect.value})
        });
        const data = await res.json();
        currentState = data.observation;
        maxBattery   = currentState.battery_level;
        logFeed.innerHTML = "";
        renderGrid(currentState, true);
        updateTelemetry(currentState);
        statusDot.className     = "pulse-dot online";
        statusLabel.textContent = "Connected";
    } catch (e) {
        statusDot.className     = "pulse-dot";
        statusLabel.textContent = "Offline";
        addLog(`Cannot reach ${API_BASE} — is app.py running?`, "sys");
    }
}

// ── Single step ───────────────────────────────────────────
async function stepEnv() {
    if (!currentState) return;
    stepCount++;

    // 1. Policy endpoint (LLM / Q-table on server)
    let action = null, source = "bfs";
    try {
        const pr = await fetch(`${API_BASE}/policy`, {
            method:"POST", headers:{"Content-Type":"application/json"},
            body: JSON.stringify({message: currentState.message})
        });
        if (pr.ok) { const pd = await pr.json(); action=pd.action; source=pd.source||"llm"; }
    } catch (_) {}

    // 2. Local BFS fallback
    if (!action) { action = localFallback(currentState); source = "bfs"; }

    showPolicy(source, action);

    // 3. Execute
    try {
        const res  = await fetch(`${API_BASE}/step`, {
            method:"POST", headers:{"Content-Type":"application/json"},
            body: JSON.stringify({command: action})
        });
        const data = await res.json();

        const wasCollect = action === "COLLECT";
        currentState = data.observation;
        renderGrid(currentState);
        updateTelemetry(currentState, data.reward, data.done);

        // Collect animation + particles
        if (wasCollect && robotEntity) {
            robotEntity.classList.add("collecting");
            setTimeout(() => robotEntity.classList.remove("collecting"), 440);
            const cx = parseInt(robotEntity.style.left)  + CELL/2;
            const cy = parseInt(robotEntity.style.top)   + CELL/2;
            spawnParticles(cx, cy);
        }

        const sign = data.reward >= 0 ? "+" : "";
        addLog(`${action}  ·  ${sign}${data.reward.toFixed(2)}`, source);

        if (data.done) {
            addLog(`🏁 Episode complete  ·  total ${totalReward.toFixed(2)}`, "sys");
            if (autoMode) toggleAutoMode();
        }
    } catch (e) {
        addLog(`Step error: ${e.message}`, "sys");
        if (autoMode) toggleAutoMode();
    }
}

// ── Auto mode ─────────────────────────────────────────────
function toggleAutoMode() {
    autoMode = !autoMode;
    if (autoMode) {
        autoBtn.textContent = "⏹  Stop";
        autoBtn.className   = "btn stop";
        autoTimer = setInterval(stepEnv, stepDelay);
    } else {
        autoBtn.textContent = "▶  Run Policy";
        autoBtn.className   = "btn primary";
        clearInterval(autoTimer);
    }
}

// ── Event listeners ───────────────────────────────────────
resetBtn .addEventListener("click",  resetEnv);
autoBtn  .addEventListener("click",  toggleAutoMode);
manualBtn.addEventListener("click",  stepEnv);
taskSelect.addEventListener("change", resetEnv);

// ── Boot ──────────────────────────────────────────────────
resetEnv();
