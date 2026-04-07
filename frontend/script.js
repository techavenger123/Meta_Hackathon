const API_BASE = "http://localhost:7861";

// DOM Elements
const statusDot = document.getElementById("status-dot");
const taskSelect = document.getElementById("task-select");
const resetBtn = document.getElementById("reset-btn");
const autoBtn = document.getElementById("auto-btn");
const manualBtn = document.getElementById("manual-btn");

const envGrid = document.getElementById("env-grid");
const batteryProgress = document.getElementById("battery-progress");
const batteryText = document.getElementById("battery-text");
const scoreText = document.getElementById("score-text");
const inventoryText = document.getElementById("inventory-text");
const logFeed = document.getElementById("log-feed");

// State logic
let autoMode = false;
let autoInterval = null;
let currentState = null;

// Helper to log messages to UI
function addLog(msg) {
    const p = document.createElement("p");
    p.className = "log-entry";
    p.textContent = msg;
    logFeed.prepend(p);
}

let robotEntity = null;

// Draw the Grid
function renderGrid(obs, isReset = false) {
    const [width, height] = obs.grid_size;
    const CELL_SIZE = 45;
    const CELL_GAP = 4;
    const PADDING = 8;
    
    if (isReset) {
        envGrid.innerHTML = "";
        envGrid.style.gridTemplateColumns = `repeat(${width}, 1fr)`;
        
        // Render base grid
        for (let y = height - 1; y >= 0; y--) {
            for (let x = 0; x < width; x++) {
                const cell = document.createElement("div");
                cell.className = "cell";
                cell.dataset.x = x;
                cell.dataset.y = y;
                // Click to place/remove garbage dynamically
                cell.addEventListener("click", () => toggleGarbage(x, y));
                envGrid.appendChild(cell);
            }
        }
        
        // Render obstacles
        obs.obstacle_positions.forEach(pos => {
            const el = document.createElement("div");
            el.className = "cell obstacle";
            el.style.position = "absolute";
            el.style.left = `${PADDING + pos[0] * (CELL_SIZE + CELL_GAP)}px`;
            el.style.top = `${PADDING + ((height - 1) - pos[1]) * (CELL_SIZE + CELL_GAP)}px`;
            envGrid.appendChild(el);
        });
        
        // Initialize Robot entity smoothly
        robotEntity = document.createElement("div");
        robotEntity.className = "robot-entity";
        robotEntity.textContent = "🤖";
        envGrid.appendChild(robotEntity);
    }
    
    // Smoothly transition robot
    if (robotEntity) {
        robotEntity.style.left = `${PADDING + obs.robot_position[0] * (CELL_SIZE + CELL_GAP)}px`;
        robotEntity.style.top = `${PADDING + ((height - 1) - obs.robot_position[1]) * (CELL_SIZE + CELL_GAP)}px`;
    }
    
    // Clear and re-render garbage cleanly
    const oldGarbage = document.querySelectorAll(".cell.garbage");
    oldGarbage.forEach(g => g.remove());
    
    obs.garbage_positions.forEach(pos => {
        const el = document.createElement("div");
        el.className = "cell garbage";
        el.textContent = "🗑️";
        el.style.position = "absolute";
        el.style.left = `${PADDING + pos[0] * (CELL_SIZE + CELL_GAP)}px`;
        el.style.top = `${PADDING + ((height - 1) - pos[1]) * (CELL_SIZE + CELL_GAP)}px`;
        el.style.zIndex = "5"; 
        envGrid.appendChild(el);
    });
}

// Update Telemetry UI
function updateTelemetry(obs, reward, done) {
    if (!window.maxBattery || obs.battery_level > window.maxBattery) {
        window.maxBattery = obs.battery_level;
    }
    
    const pct = Math.max(0, (obs.battery_level / window.maxBattery) * 100);
    batteryProgress.style.width = `${pct}%`;
    batteryText.textContent = `${obs.battery_level} units`;
    
    if (pct > 50) batteryProgress.style.background = "var(--success)";
    else if (pct > 20) batteryProgress.style.background = "var(--warning)";
    else batteryProgress.style.background = "var(--danger)";

    inventoryText.textContent = obs.inventory_count;
    
    if (reward !== undefined) {
        let currentReward = parseFloat(scoreText.textContent) || 0;
        currentReward += reward;
        scoreText.textContent = currentReward.toFixed(1);
    }
    
    addLog(obs.message);
}

// API Calls
async function toggleGarbage(x, y) {
    if (!currentState || autoMode) return;
    
    // Ignore obstacles and robot starting position
    if (currentState.obstacle_positions.some(o => o[0] === x && o[1] === y)) return;
    if (currentState.robot_position[0] === x && currentState.robot_position[1] === y) return;

    let isGarbage = currentState.garbage_positions.some(g => g[0] === x && g[1] === y);
    let newPositions = [...currentState.garbage_positions];
    
    if (isGarbage) {
        newPositions = newPositions.filter(g => !(g[0] === x && g[1] === y));
    } else {
        newPositions.push([x, y]);
    }

    try {
        const res = await fetch(`${API_BASE}/configure`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ 
                task_id: taskSelect.value,
                garbage_positions: newPositions
            })
        });
        const data = await res.json();
        currentState = data.observation;
        renderGrid(currentState);
        updateTelemetry(currentState);
        addLog(`Custom garbage updated! Remaining: ${newPositions.length}`);
    } catch(err) {
        console.error("Config error", err);
        addLog("Failed to update custom garbage layout.");
    }
}

async function resetEnv() {
    try {
        const taskId = taskSelect.value;
        const res = await fetch(`${API_BASE}/reset`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ task_id: taskId })
        });
        const data = await res.json();
        
        currentState = data.observation;
        window.maxBattery = currentState.battery_level; // set new max battery
        scoreText.textContent = "0.0";
        logFeed.innerHTML = "";
        
        renderGrid(currentState, true);
        updateTelemetry(currentState);
        
        statusDot.className = "dot online";
        
        if (autoMode) toggleAutoMode(); // Stop auto if we reset manually
    } catch(err) {
        statusDot.className = "dot";
        addLog("Error connecting to server. Is app.py running?");
        console.error(err);
    }
}

// BFS + TSP Heuristic (Mirrors inference.py)
function bfsNextMove(rPos, target, obstacles, gridW, gridH) {
    if (rPos[0] === target[0] && rPos[1] === target[1]) return "COLLECT";
    const obsSet = new Set(obstacles.map(o => `${o[0]},${o[1]}`));
    const dirs = [
        {name: "RIGHT", dx: 1, dy: 0}, {name: "LEFT", dx: -1, dy: 0},
        {name: "UP", dx: 0, dy: 1}, {name: "DOWN", dx: 0, dy: -1}
    ];

    let queue = [ {pos: [...rPos], firstMove: null} ];
    let visited = new Set([`${rPos[0]},${rPos[1]}`]);

    while(queue.length > 0) {
        let current = queue.shift();
        for (let d of dirs) {
            let nx = current.pos[0] + d.dx;
            let ny = current.pos[1] + d.dy;
            if (nx < 0 || nx >= gridW || ny < 0 || ny >= gridH) continue;
            let nKey = `${nx},${ny}`;
            if (obsSet.has(nKey) || visited.has(nKey)) continue;
            let move = current.firstMove ? current.firstMove : d.name;
            if (nx === target[0] && ny === target[1]) return move;
            visited.add(nKey);
            queue.push({pos: [nx, ny], firstMove: move});
        }
    }
    return null;
}

function nearestNeighbourOrder(start, targets, obstacles, gridW, gridH) {
    function bfsDist(a, b) {
        if (a[0] === b[0] && a[1] === b[1]) return 0;
        const obsSet = new Set(obstacles.map(o => `${o[0]},${o[1]}`));
        const dirs = [[1,0], [-1,0], [0,1], [0,-1]];
        let queue = [ {pos: [...a], dist: 0} ];
        let visited = new Set([`${a[0]},${a[1]}`]);

        while (queue.length > 0) {
            let curr = queue.shift();
            for (let d of dirs) {
                let nx = curr.pos[0] + d[0];
                let ny = curr.pos[1] + d[1];
                if (nx < 0 || nx >= gridW || ny < 0 || ny >= gridH) continue;
                let nKey = `${nx},${ny}`;
                if (obsSet.has(nKey) || visited.has(nKey)) continue;
                if (nx === b[0] && ny === b[1]) return curr.dist + 1;
                visited.add(nKey);
                queue.push({pos: [nx, ny], dist: curr.dist + 1});
            }
        }
        return Infinity;
    }

    let remaining = [...targets];
    let ordered = [];
    let current = [...start];
    
    while(remaining.length > 0) {
        let nearest = remaining[0];
        let minDist = bfsDist(current, nearest);
        for (let i = 1; i < remaining.length; i++) {
            let d = bfsDist(current, remaining[i]);
            if (d < minDist) {
                minDist = d;
                nearest = remaining[i];
            }
        }
        ordered.push(nearest);
        remaining = remaining.filter(t => t !== nearest);
        current = [...nearest];
    }
    return ordered;
}

function getNextOptimalAction(obs) {
    if (obs.garbage_positions.length === 0) return "UP";
    const rPos = obs.robot_position;
    const targets = obs.garbage_positions;
    const obstacles = obs.obstacle_positions;
    const gridW = obs.grid_size[0];
    const gridH = obs.grid_size[1];

    if (targets.some(g => g[0] === rPos[0] && g[1] === rPos[1])) return "COLLECT";

    const ordered = nearestNeighbourOrder(rPos, targets, obstacles, gridW, gridH);
    const move = bfsNextMove(rPos, ordered[0], obstacles, gridW, gridH);
    return move ? move : "RIGHT";
}

async function stepEnv() {
    if (!currentState) return;
    
    // In a fully deployed setup, we would hit a `/policy` endpoint that queries the Llama-3 model.
    // For this dashboard demo, we'll replicate the naive deterministic fallback or random to visualize behavior live!
    const action = getNextOptimalAction(currentState);
    
    try {
        const res = await fetch(`${API_BASE}/step`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ command: action })
        });
        const data = await res.json();
        
        currentState = data.observation;
        renderGrid(currentState);
        updateTelemetry(currentState, data.reward, data.done);
        
        if (data.done) {
            addLog(`Episode Finished! Total Reward: ${scoreText.textContent}`);
            if (autoMode) toggleAutoMode(); // Stop auto
        }
    } catch(err) {
        console.error("Step error", err);
        if (autoMode) toggleAutoMode();
    }
}

function toggleAutoMode() {
    autoMode = !autoMode;
    if (autoMode) {
        autoBtn.textContent = "Stop Auto-RL";
        autoBtn.classList.remove("primary");
        autoBtn.classList.add("danger");
        autoBtn.style.background = "var(--danger)";
        
        autoInterval = setInterval(() => {
            stepEnv();
        }, 500); // Step every 500ms
    } else {
        autoBtn.textContent = "Start Auto-RL";
        autoBtn.classList.remove("danger");
        autoBtn.classList.add("primary");
        autoBtn.style.background = "var(--accent)";
        clearInterval(autoInterval);
    }
}

// Event Listeners
resetBtn.addEventListener("click", resetEnv);
manualBtn.addEventListener("click", stepEnv);
autoBtn.addEventListener("click", toggleAutoMode);
taskSelect.addEventListener("change", resetEnv);

// Initial ping
resetEnv();
