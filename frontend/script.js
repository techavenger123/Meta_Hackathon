/* ═══════════════════════════════════════════════════════
   GarbageBot — Dashboard Logic
   Policy chain: Q-table → Fine-tuned LLM → BFS fallback
   ═══════════════════════════════════════════════════════ */

const API_BASE = "http://localhost:7861";

// ── DOM ──────────────────────────────────────────────────
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

// ── State ────────────────────────────────────────────────
let autoMode       = false;
let autoTimer      = null;
let currentState   = null;
let robotEntity    = null;
let stepCount      = 0;
let totalReward    = 0;
let rewardHistory  = [];
let maxBattery     = 30;
let stepDelay      = 500;

// ── Speed slider ─────────────────────────────────────────
speedSlider.addEventListener("input", () => {
    stepDelay = parseInt(speedSlider.value);
    speedVal.textContent = `${stepDelay}ms`;
    if (autoMode) {
        clearInterval(autoTimer);
        autoTimer = setInterval(stepEnv, stepDelay);
    }
});

// ── Log helpers ───────────────────────────────────────────
function addLog(msg, source = "sys") {
    // Remove placeholder
    const ph = logFeed.querySelector(".placeholder");
    if (ph) ph.remove();

    const entry  = document.createElement("div");
    entry.className = "log-entry";

    const badge = document.createElement("span");
    badge.className = `log-badge ${source}`;
    badge.textContent = source.toUpperCase();

    const text = document.createElement("span");
    text.textContent = msg;

    entry.appendChild(badge);
    entry.appendChild(text);
    logFeed.prepend(entry);

    // Keep log tidy (max 60 entries)
    while (logFeed.children.length > 60) {
        logFeed.removeChild(logFeed.lastChild);
    }
}

clearLogBtn.addEventListener("click", () => {
    logFeed.innerHTML = `<p class="placeholder">Log cleared…</p>`;
});

// ── Mini reward chart ─────────────────────────────────────
function drawChart() {
    const ctx = rewardCanvas.getContext("2d");
    const W = rewardCanvas.width;
    const H = rewardCanvas.height;
    ctx.clearRect(0, 0, W, H);

    if (rewardHistory.length < 2) return;

    const maxR = Math.max(...rewardHistory.map(Math.abs), .1);
    const step = W / (rewardHistory.length - 1);

    // Gradient fill
    const grad = ctx.createLinearGradient(0, 0, 0, H);
    grad.addColorStop(0,   "rgba(99,102,241,.55)");
    grad.addColorStop(1,   "rgba(99,102,241,0)");

    ctx.beginPath();
    rewardHistory.forEach((v, i) => {
        const x = i * step;
        const y = H - ((v + maxR) / (2 * maxR)) * H;
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    // Close fill path
    ctx.lineTo((rewardHistory.length - 1) * step, H);
    ctx.lineTo(0, H);
    ctx.closePath();
    ctx.fillStyle = grad;
    ctx.fill();

    // Line
    ctx.beginPath();
    rewardHistory.forEach((v, i) => {
        const x = i * step;
        const y = H - ((v + maxR) / (2 * maxR)) * H;
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.strokeStyle = "#818cf8";
    ctx.lineWidth   = 1.8;
    ctx.lineJoin    = "round";
    ctx.stroke();
}

// ── Particles on COLLECT ──────────────────────────────────
function spawnParticles(px, py) {
    for (let i = 0; i < 10; i++) {
        const p    = document.createElement("div");
        p.className = "particle";
        const angle = (i / 10) * Math.PI * 2;
        const dist  = 30 + Math.random() * 35;
        p.style.left = `${px}px`;
        p.style.top  = `${py}px`;
        p.style.setProperty("--tx", `translate(${Math.cos(angle)*dist}px,${Math.sin(angle)*dist}px)`);
        p.style.background = ["#a855f7","#818cf8","#22d3ee","#f59e0b"][i % 4];
        particleLayer.appendChild(p);
        setTimeout(() => p.remove(), 750);
    }
}

// ── Robot trail ghost ─────────────────────────────────────
function addTrailGhost(leftPx, topPx) {
    const ghost = document.createElement("div");
    ghost.className = "trail-ghost";
    ghost.style.left = `${leftPx}px`;
    ghost.style.top  = `${topPx}px`;
    envGrid.appendChild(ghost);
    setTimeout(() => ghost.remove(), 900);
}

// ── Grid rendering ────────────────────────────────────────
const CELL_SIZE = 44;
const CELL_GAP  = 4;
const PAD       = 8;

function cellX(x) { return PAD + x * (CELL_SIZE + CELL_GAP); }
function cellY(y, height) { return PAD + ((height - 1) - y) * (CELL_SIZE + CELL_GAP); }

function renderGrid(obs, isReset = false) {
    const [W, H] = obs.grid_size;

    if (isReset) {
        envGrid.innerHTML = "";
        envGrid.style.gridTemplateColumns = `repeat(${W}, 1fr)`;

        // Base cells
        for (let y = H - 1; y >= 0; y--) {
            for (let x = 0; x < W; x++) {
                const cell = document.createElement("div");
                cell.className = "cell";
                cell.dataset.x = x; cell.dataset.y = y;
                cell.addEventListener("click", () => toggleGarbage(x, y));
                envGrid.appendChild(cell);
            }
        }

        // Obstacle cells
        obs.obstacle_positions.forEach(([x, y]) => {
            const el = document.createElement("div");
            el.className    = "cell obstacle";
            el.style.cssText = `position:absolute;left:${cellX(x)}px;top:${cellY(y,H)}px;`;
            envGrid.appendChild(el);
        });

        // Robot entity
        robotEntity = document.createElement("div");
        robotEntity.className   = "robot-entity";
        robotEntity.textContent = "🤖";
        envGrid.appendChild(robotEntity);
    }

    // Move robot (with trail)
    if (robotEntity) {
        const newLeft = cellX(obs.robot_position[0]);
        const newTop  = cellY(obs.robot_position[1], H);

        // Drop trail ghost at old position
        const oldLeft = parseInt(robotEntity.style.left) || newLeft;
        const oldTop  = parseInt(robotEntity.style.top)  || newTop;
        if (oldLeft !== newLeft || oldTop !== newTop) {
            addTrailGhost(oldLeft, oldTop);
        }

        robotEntity.style.left = `${newLeft}px`;
        robotEntity.style.top  = `${newTop}px`;
    }

    // Re-render garbage
    document.querySelectorAll(".cell.garbage").forEach(g => g.remove());
    obs.garbage_positions.forEach(([x, y]) => {
        const el = document.createElement("div");
        el.className    = "cell garbage";
        el.style.cssText = `position:absolute;left:${cellX(x)}px;top:${cellY(y,H)}px;z-index:5;`;
        el.addEventListener("click", () => toggleGarbage(x, y));
        envGrid.appendChild(el);
    });
}

// ── Telemetry update ──────────────────────────────────────
function updateTelemetry(obs, reward, done) {
    // Battery
    if (obs.battery_level > maxBattery) maxBattery = obs.battery_level;
    const pct = Math.max(0, (obs.battery_level / maxBattery) * 100);
    batteryProgress.style.width = `${pct}%`;
    batteryText.textContent = `${obs.battery_level} / ${maxBattery}`;

    if      (pct > 50) batteryProgress.style.background = "#22d3ee";
    else if (pct > 20) batteryProgress.style.background = "#f59e0b";
    else               batteryProgress.style.background = "#f43f5e";

    inventoryText.textContent = obs.inventory_count;

    // Reward
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

// ── Policy badge update ───────────────────────────────────
const policyColors = {
    llm:     { color: "#818cf8", border: "rgba(99,102,241,.5)" },
    bfs:     { color: "#14b8a6", border: "rgba(20,184,166,.5)" },
    "q-table":{ color: "#f59e0b", border: "rgba(245,158,11,.5)" },
    sys:     { color: "#64748b", border: "rgba(100,116,139,.3)" },
};
function showPolicy(source, action) {
    const c = policyColors[source] || policyColors.sys;
    policyLabel.textContent = `${source.toUpperCase()} → ${action}`;
    policyBadge.style.borderColor = c.border;
    policyBadge.style.color = c.color;
    policyBadge.classList.add("active");
}

// ── BFS fallback ──────────────────────────────────────────
function bfsNextMove(rPos, target, obstacles, gridW, gridH) {
    if (rPos[0] === target[0] && rPos[1] === target[1]) return "COLLECT";
    const obsSet = new Set(obstacles.map(([a,b]) => `${a},${b}`));
    const dirs   = [["RIGHT",1,0],["LEFT",-1,0],["UP",0,1],["DOWN",0,-1]];
    const queue  = [{ pos: [...rPos], first: null }];
    const visited= new Set([`${rPos[0]},${rPos[1]}`]);

    while (queue.length) {
        const { pos, first } = queue.shift();
        for (const [name, dx, dy] of dirs) {
            const nx = pos[0]+dx, ny = pos[1]+dy;
            if (nx < 0 || nx >= gridW || ny < 0 || ny >= gridH) continue;
            const key = `${nx},${ny}`;
            if (obsSet.has(key) || visited.has(key)) continue;
            const move = first || name;
            if (nx === target[0] && ny === target[1]) return move;
            visited.add(key);
            queue.push({ pos: [nx, ny], first: move });
        }
    }
    return null;
}

function nnOrder(start, targets, obstacles, gW, gH) {
    function dist(a, b) {
        // BFS distance (reuse bfsNextMove principle)
        if (a[0]===b[0]&&a[1]===b[1]) return 0;
        const obsSet = new Set(obstacles.map(([x,y])=>`${x},${y}`));
        const dirs = [[1,0],[-1,0],[0,1],[0,-1]];
        const q = [{pos:[...a],d:0}]; const vis=new Set([`${a[0]},${a[1]}`]);
        while(q.length){const{pos,d}=q.shift();for(const[dx,dy]of dirs){const nx=pos[0]+dx,ny=pos[1]+dy;if(nx<0||nx>=gW||ny<0||ny>=gH)continue;const k=`${nx},${ny}`;if(obsSet.has(k)||vis.has(k))continue;if(nx===b[0]&&ny===b[1])return d+1;vis.add(k);q.push({pos:[nx,ny],d:d+1});}}
        return Infinity;
    }
    let rem=[...targets], cur=[...start], ord=[];
    while(rem.length){let best=rem[0],bD=dist(cur,best);for(const t of rem){const d=dist(cur,t);if(d<bD){bD=d;best=t;}}ord.push(best);rem=rem.filter(t=>!(t[0]===best[0]&&t[1]===best[1]));cur=[...best];}
    return ord;
}

function localFallback(obs) {
    if (!obs.garbage_positions.length) return "UP";
    const r = obs.robot_position;
    if (obs.garbage_positions.some(([x,y])=>x===r[0]&&y===r[1])) return "COLLECT";
    const ordered = nnOrder(r, obs.garbage_positions, obs.obstacle_positions, obs.grid_size[0], obs.grid_size[1]);
    return bfsNextMove(r, ordered[0], obs.obstacle_positions, obs.grid_size[0], obs.grid_size[1]) || "RIGHT";
}

// ── Custom garbage toggle ─────────────────────────────────
async function toggleGarbage(x, y) {
    if (!currentState || autoMode) return;
    if (currentState.obstacle_positions.some(([ox,oy]) => ox===x && oy===y)) return;
    if (currentState.robot_position[0]===x && currentState.robot_position[1]===y) return;

    const has  = currentState.garbage_positions.some(([gx,gy]) => gx===x && gy===y);
    const next = has
        ? currentState.garbage_positions.filter(([gx,gy]) => !(gx===x && gy===y))
        : [...currentState.garbage_positions, [x, y]];

    try {
        const res = await fetch(`${API_BASE}/configure`, {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ task_id: taskSelect.value, garbage_positions: next })
        });
        const data = await res.json();
        currentState = data.observation;
        renderGrid(currentState);
        addLog(`Garbage ${has?"removed":"placed"} at (${x},${y}) — ${next.length} remaining`, "sys");
    } catch (e) { addLog(`Config error: ${e.message}`, "sys"); }
}

// ── Reset ─────────────────────────────────────────────────
async function resetEnv() {
    if (autoMode) toggleAutoMode();
    stepCount   = 0;
    totalReward = 0;
    rewardHistory = [];
    scoreText.textContent        = "0.00";
    episodeScoreChip.textContent = "Score 0.00";
    stepCounter.textContent      = "Step 0";
    drawChart();

    try {
        const res = await fetch(`${API_BASE}/reset`, {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ task_id: taskSelect.value })
        });
        const data = await res.json();
        currentState = data.observation;
        maxBattery   = currentState.battery_level;

        logFeed.innerHTML = "";
        renderGrid(currentState, true);
        updateTelemetry(currentState);

        statusDot.className = "pulse-dot online";
        statusLabel.textContent = "Connected";
        policyLabel.textContent = "–";
    } catch (e) {
        statusDot.className   = "pulse-dot";
        statusLabel.textContent = "Offline";
        addLog(`Cannot reach server at ${API_BASE}`, "sys");
    }
}

// ── Step ──────────────────────────────────────────────────
async function stepEnv() {
    if (!currentState) return;
    stepCount++;

    // 1. Ask /policy (fine-tuned LLM or Q-table on server)
    let action = null, source = "bfs";
    try {
        const pr = await fetch(`${API_BASE}/policy`, {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message: currentState.message })
        });
        if (pr.ok) { const pd = await pr.json(); action = pd.action; source = pd.source || "llm"; }
    } catch (_) {}

    // 2. Local BFS fallback
    if (!action) action = localFallback(currentState);

    // Show policy
    showPolicy(source, action);

    // 3. Execute action
    try {
        const res = await fetch(`${API_BASE}/step`, {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ command: action })
        });
        const data = await res.json();

        const wasCollect = action === "COLLECT";
        currentState = data.observation;
        renderGrid(currentState);
        updateTelemetry(currentState, data.reward, data.done);

        // Collect animation + particles
        if (wasCollect && robotEntity) {
            robotEntity.classList.add("collecting");
            setTimeout(() => robotEntity.classList.remove("collecting"), 420);
            const cx = parseInt(robotEntity.style.left) + CELL_SIZE / 2;
            const cy = parseInt(robotEntity.style.top)  + CELL_SIZE / 2;
            spawnParticles(cx, cy);
        }

        addLog(`${action}  ·  reward ${data.reward >= 0 ? "+" : ""}${data.reward.toFixed(2)}`, source);

        if (data.done) {
            addLog(`🏁 Episode done — score ${totalReward.toFixed(2)}`, "sys");
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

// ── Listeners ─────────────────────────────────────────────
resetBtn.addEventListener("click",  resetEnv);
autoBtn .addEventListener("click",  toggleAutoMode);
manualBtn.addEventListener("click", stepEnv);
taskSelect.addEventListener("change", resetEnv);

// ── Boot ──────────────────────────────────────────────────
resetEnv();
