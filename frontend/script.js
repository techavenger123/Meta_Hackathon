const API_BASE = "http://localhost:7860";

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
    // We need max battery from somewhere. We can infer it if we know defaults, OR just use current level vs 100 for visual sake, 
    // but scenario sizes are 30, 50, 80. Let's dynamically calculate percentage based on max_seen_battery.
    if (!window.maxBattery || obs.battery_level > window.maxBattery) {
        window.maxBattery = obs.battery_level;
    }
    
    const pct = Math.max(0, (obs.battery_level / window.maxBattery) * 100);
    batteryProgress.style.width = `${pct}%`;
    batteryText.textContent = `${obs.battery_level} units`;
    
    // Color code battery
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

// The Naive Heuristic (same as inference.py fallback)
function getNextOptimalAction(obs) {
    if (obs.garbage_positions.length > 0) {
        const gPos = obs.garbage_positions[0];
        const rPos = obs.robot_position;
        if (gPos[0] === rPos[0] && gPos[1] === rPos[1]) return "COLLECT";
        if (gPos[0] > rPos[0]) return "RIGHT";
        if (gPos[0] < rPos[0]) return "LEFT";
        if (gPos[1] > rPos[1]) return "UP";
        if (gPos[1] < rPos[1]) return "DOWN";
    }
    return "LEFT"; // Default
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
