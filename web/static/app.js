/* deepMaze browser client.
 *
 * Modes:
 *   - Train: configure params, click Train -> live SSE
 *   - Pretrained: pick a model, pick maze source, click Watch -> live SSE
 *
 * Editor model:
 *   Maze always starts from a server-generated layout (POST /api/maze/generate).
 *   Paint actions modify that base; Reset edits restores it; Regenerate fetches
 *   a fresh one. No client-side maze generation.
 */

const API = (path) => (window.API_BASE_URL || "") + path;

const HOLE = 0, LAND = 1, START = 2, EXIT = 3, LAVA = 4;
const COLORS = {
  [HOLE]:  "#1a1a1a",
  [LAND]:  "#cfd0d3",
  [START]: "#6ec07b",
  [EXIT]:  "#e8c84a",
  [LAVA]:  "#e25555",
};
const AGENT_COLOR = "#4ea8ff";

const $ = (id) => document.getElementById(id);
const canvas = $("maze");
const ctx = canvas.getContext("2d");
const memCanvas = $("memory");
const memCtx = memCanvas.getContext("2d");

let W = +$("w").value, H = +$("h").value;
let maze = blankMaze(W, H);
let cachedGenerated = null;  // last server-generated layout (for Reset edits)
let agentPos = null;
let paintMode = LAND;
let paintHeld = false;

let frames = [];
let frameIdx = 0;
let playing = true;
let policyArrows = null;
let visitTrail = new Map();
let overlayArrows = true, overlayVisits = false;
let currentAgentType = null;
const AGENTS_WITH_MEMORY = ["drqn", "dtqn"];
let currentRunId = null;

function blankMaze(w, h) {
  // Walls-only placeholder used before first generator response.
  const m = Array.from({length: h}, () => Array.from({length: w}, () => HOLE));
  return m;
}

function cloneMaze(m) { return m.map(r => r.slice()); }
function cellSize() { return Math.floor(Math.min(canvas.width / W, canvas.height / H)); }

function draw() {
  const s = cellSize();
  ctx.fillStyle = "#000"; ctx.fillRect(0, 0, canvas.width, canvas.height);
  for (let i = 0; i < H; i++)
    for (let j = 0; j < W; j++) {
      ctx.fillStyle = COLORS[maze[i][j]] || COLORS[LAND];
      ctx.fillRect(j * s, i * s, s - 1, s - 1);
    }
  if (overlayVisits) {
    for (const [k, v] of visitTrail.entries()) {
      const [i, j] = k.split(",").map(Number);
      ctx.fillStyle = `rgba(78, 168, 255, ${0.25 * v})`;
      ctx.fillRect(j * s, i * s, s, s);
    }
  }
  if (overlayArrows && policyArrows) {
    ctx.strokeStyle = "#ffffffaa"; ctx.lineWidth = 1;
    for (let i = 0; i < H; i++)
      for (let j = 0; j < W; j++) {
        if (maze[i][j] === HOLE || maze[i][j] === LAVA) continue;
        const q = policyArrows[i] && policyArrows[i][j];
        if (q) drawArrow(j * s + s/2, i * s + s/2, argmax(q), s * 0.3);
      }
  }
  if (agentPos) {
    const [r, c] = agentPos;
    ctx.fillStyle = AGENT_COLOR;
    ctx.beginPath();
    ctx.arc(c * s + s/2, r * s + s/2, s * 0.36, 0, 2 * Math.PI);
    ctx.fill();
  }
}

function argmax(a) {
  let bi = 0; for (let i = 1; i < a.length; i++) if (a[i] > a[bi]) bi = i; return bi;
}
function drawArrow(cx, cy, action, len) {
  const d = [[0, -1], [1, 0], [0, 1], [-1, 0]][action];
  const x2 = cx + d[0] * len, y2 = cy + d[1] * len;
  ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(x2, y2); ctx.stroke();
  const ah = len * 0.35;
  ctx.beginPath();
  ctx.moveTo(x2, y2);
  ctx.lineTo(x2 - d[0]*ah + d[1]*ah*0.5, y2 - d[1]*ah - d[0]*ah*0.5);
  ctx.lineTo(x2 - d[0]*ah - d[1]*ah*0.5, y2 - d[1]*ah + d[0]*ah*0.5);
  ctx.closePath(); ctx.fillStyle = "#ffffffaa"; ctx.fill();
}

// --- paint editor (always operates on top of generator output) ----------
document.querySelectorAll(".paint-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".paint-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    paintMode = +btn.dataset.paint;
  });
});

function paintAt(e) {
  const rect = canvas.getBoundingClientRect();
  const s = cellSize();
  const j = Math.floor((e.clientX - rect.left) / s);
  const i = Math.floor((e.clientY - rect.top) / s);
  if (i <= 0 || j <= 0 || i >= H - 1 || j >= W - 1) return;
  if (paintMode === START) clearVal(START);
  maze[i][j] = paintMode;
  draw();
}
function clearVal(v) {
  for (let i = 0; i < H; i++) for (let j = 0; j < W; j++)
    if (maze[i][j] === v) maze[i][j] = LAND;
}
canvas.addEventListener("mousedown", (e) => { paintHeld = true; paintAt(e); });
canvas.addEventListener("mousemove", (e) => { if (paintHeld) paintAt(e); });
window.addEventListener("mouseup", () => { paintHeld = false; });

// --- generator (single source of mazes) ---------------------------------
async function generateMaze() {
  W = +$("w").value; H = +$("h").value;
  $("status").textContent = "generating…";
  try {
    const r = await fetch(API("/api/maze/generate"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        width: W, height: H,
        density: +$("density").value,
        generator: $("generator").value,
        n_lava: +$("n_lava").value || 0,
        n_treasures: +$("n_treasures").value || 1,
        seed: Math.floor(Math.random() * 1e6),
      }),
    });
    if (!r.ok) { $("status").textContent = "gen failed"; return; }
    const body = await r.json();
    cachedGenerated = body.maze;
    maze = cloneMaze(cachedGenerated);
    resetLive(); draw();
    $("status").textContent = "ready";
  } catch (e) {
    $("status").textContent = "gen error: " + e.message;
  }
}

$("regen").addEventListener("click", generateMaze);
$("resetEdits").addEventListener("click", () => {
  if (cachedGenerated) { maze = cloneMaze(cachedGenerated); resetLive(); draw(); }
});

$("ovArrows").addEventListener("change", e => { overlayArrows = e.target.checked; draw(); });
$("ovVisits").addEventListener("change", e => { overlayVisits = e.target.checked; draw(); });

// --- mode toggle --------------------------------------------------------
document.querySelectorAll("input[name=mode]").forEach(r => {
  r.addEventListener("change", async () => {
    const mode = r.value;
    $("trainPanel").style.display = mode === "train" ? "" : "none";
    $("inferencePanel").style.display = mode === "inference" ? "" : "none";
    if (mode === "inference") await populateModels();
  });
});

async function populateModels() {
  try {
    const r = await fetch(API("/api/models"));
    const body = r.ok ? await r.json() : { models: [] };
    const sel = $("model");
    sel.innerHTML = "";
    if (!body.models.length) {
      const o = document.createElement("option");
      o.textContent = "(no models found)"; o.disabled = true;
      sel.appendChild(o); return;
    }
    for (const m of body.models) {
      const o = document.createElement("option");
      o.value = `${m.source}/${m.name}`;
      o.textContent = `[${m.source}] ${m.name} (${m.agent_type || "?"})`;
      sel.appendChild(o);
    }
  } catch (e) {
    $("status").textContent = "models load failed";
  }
}

// --- charts -------------------------------------------------------------
const chartOpts = {
  animation: false, responsive: true, maintainAspectRatio: false,
  scales: { x: { ticks: { color: "#888" }, grid: { color: "#222" } },
            y: { ticks: { color: "#888" }, grid: { color: "#222" } } },
  plugins: { legend: { labels: { color: "#ccc", font: { size: 10 } } } },
  elements: { line: { tension: 0.2 }, point: { radius: 0 } },
};
function mkChart(id, label, color) {
  return new Chart($(id), {
    type: "line",
    data: { labels: [], datasets: [{ label, data: [], borderColor: color, borderWidth: 1.5 }] },
    options: chartOpts,
  });
}
const rewardChart = mkChart("rewardChart", "reward", "#4ea8ff");
const lengthChart = mkChart("lengthChart", "length", "#ffb479");
const lossChart   = mkChart("lossChart",   "loss",   "#e25555");
const epsChart    = mkChart("epsChart",    "epsilon","#6ec07b");
function pushChart(c, x, y) { c.data.labels.push(x); c.data.datasets[0].data.push(y); }
function flushCharts() { for (const c of [rewardChart, lengthChart, lossChart, epsChart]) c.update("none"); }
function resetCharts() {
  for (const c of [rewardChart, lengthChart, lossChart, epsChart]) {
    c.data.labels = []; c.data.datasets[0].data = []; c.update("none");
  }
}

// --- memory strip -------------------------------------------------------
function drawMemory(mem) {
  const w = memCanvas.width, h = memCanvas.height;
  memCtx.fillStyle = "#1c232c"; memCtx.fillRect(0, 0, w, h);
  if (!mem) {
    $("memLabel").textContent = AGENTS_WITH_MEMORY.includes(currentAgentType)
      ? "memory: (waiting for first step)"
      : "memory: n/a — this agent has no memory";
    return;
  }
  $("memLabel").textContent = mem.kind === "lstm_hidden"
    ? "memory: LSTM hidden state"
    : "memory: attention over past steps";
  const d = mem.data || [];
  if (!d.length) return;
  const mx = Math.max(...d.map(Math.abs)) || 1;
  const cw = w / d.length;
  for (let i = 0; i < d.length; i++) {
    const v = d[i] / mx;
    if (mem.kind === "attention_row") {
      const a = Math.max(0, Math.min(1, v));
      memCtx.fillStyle = `rgba(78, 168, 255, ${a})`;
    } else {
      const a = Math.min(1, Math.abs(v));
      memCtx.fillStyle = v >= 0 ? `rgba(78, 168, 255, ${a})` : `rgba(226, 85, 85, ${a})`;
    }
    memCtx.fillRect(i * cw, 0, cw + 1, h);
  }
}

// --- scrubber + frames --------------------------------------------------
function renderFrame(idx) {
  if (!frames[idx]) return;
  const f = frames[idx];
  maze = f.maze; agentPos = f.pos;
  drawMemory(f.memory); draw();
  $("frameLabel").textContent = `${idx + 1} / ${frames.length}`;
  $("scrub").max = frames.length - 1; $("scrub").value = idx;
}
function pushFrame(f) {
  frames.push(f);
  if (playing) { frameIdx = frames.length - 1; renderFrame(frameIdx); }
}
function resetLive() {
  frames = []; frameIdx = 0; agentPos = null; policyArrows = null;
  visitTrail.clear();
  $("scrub").max = 0; $("scrub").value = 0;
  $("frameLabel").textContent = "0 / 0";
  drawMemory(null);
}
$("playBtn").addEventListener("click", () => {
  playing = !playing; $("playBtn").textContent = playing ? "▶" : "⏸";
  if (playing && frames.length) { frameIdx = frames.length - 1; renderFrame(frameIdx); }
});
$("stepBtn").addEventListener("click", () => {
  playing = false; $("playBtn").textContent = "⏸";
  frameIdx = Math.min(frames.length - 1, frameIdx + 1); renderFrame(frameIdx);
});
$("scrub").addEventListener("input", () => {
  playing = false; $("playBtn").textContent = "⏸";
  frameIdx = +$("scrub").value; renderFrame(frameIdx);
});

// --- SSE ----------------------------------------------------------------
let es = null;
function startStream() {
  if (es) es.close();
  es = new EventSource(API("/api/events"));
  es.onmessage = (msg) => {
    const ev = JSON.parse(msg.data);
    if (ev.type === "step") {
      const baseMaze = ev.state.map(row => row.map(v => v >= 5 ? LAND : v));
      // Auto-sync the grid to the streamed maze — without this, watching a
      // model whose maze differs from the editor's left stale editor cells
      // on the canvas until the user hit Regenerate manually.
      if (baseMaze.length !== H || baseMaze[0].length !== W) {
        H = baseMaze.length; W = baseMaze[0].length;
        $("w").value = W; $("h").value = H;
        maze = cloneMaze(baseMaze);
        visitTrail.clear();
        draw();
      }
      pushFrame({ maze: baseMaze, pos: ev.position, memory: ev.memory });
      visitTrail.set(`${ev.position[0]},${ev.position[1]}`, 1);
    } else if (ev.type === "step_delta") {
      const baseMaze = frames.length ? frames[frames.length - 1].maze : maze;
      pushFrame({ maze: baseMaze, pos: ev.position, memory: ev.memory });
      for (const k of visitTrail.keys()) visitTrail.set(k, visitTrail.get(k) * 0.97);
      visitTrail.set(`${ev.position[0]},${ev.position[1]}`, 1);
    } else if (ev.type === "episode") {
      pushChart(rewardChart, ev.episode, ev.total_reward);
      pushChart(lengthChart, ev.episode, ev.length);
      if (ev.loss != null) pushChart(lossChart, ev.episode, ev.loss);
      pushChart(epsChart, ev.episode, ev.epsilon);
      if (rewardChart.data.labels.length % 5 === 0) flushCharts();
      $("status").textContent =
        `ep ${ev.episode} R=${ev.total_reward.toFixed(2)} len=${ev.length}`;
    } else if (ev.type === "run" && ev.kind === "end") {
      flushCharts(); $("status").textContent = "done";
      $("train").style.display = ""; $("stop").style.display = "none";
      currentRunId = null;
    }
  };
  es.onerror = () => { $("status").textContent = "stream closed"; };
}

// --- train --------------------------------------------------------------
$("train").addEventListener("click", async () => {
  W = +$("w").value; H = +$("h").value;
  $("status").textContent = "starting…";
  currentAgentType = $("agent").value;
  resetCharts(); resetLive(); draw();
  startStream();

  const partial = $("partial").value === "" ? null : +$("partial").value;
  const body = {
    agent_type: $("agent").value,
    width: W, height: H,
    density: +$("density").value,
    generator: $("generator").value,
    n_treasures: +$("n_treasures").value,
    num_episodes: +$("episodes").value,
    max_steps: +$("max_steps").value,
    n_lava: +$("n_lava").value,
    partial,
    maze,    // user-edited base
  };
  const r = await fetch(API("/api/runs"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) { $("status").textContent = "start failed"; return; }
  const j = await r.json();
  currentRunId = j.run_id;
  $("train").style.display = "none";
  $("stop").style.display = "";
});

$("stop").addEventListener("click", async () => {
  if (!currentRunId) return;
  await fetch(API(`/api/runs/${currentRunId}/cancel`), { method: "POST" });
  $("status").textContent = "stopping…";
});

// --- inference / Watch --------------------------------------------------
$("watch").addEventListener("click", async () => {
  const sel = $("model").value;
  if (!sel) return;
  const [source, ...rest] = sel.split("/");
  const name = rest.join("/");
  // infer agent type from the option label "[source] name (atype)"
  const label = $("model").options[$("model").selectedIndex].textContent;
  const m = label.match(/\(([^)]+)\)\s*$/);
  currentAgentType = m ? m[1] : null;
  $("status").textContent = "loading model…";
  resetCharts(); resetLive(); draw();
  startStream();

  const body = {
    source, name,
    episodes: +$("infEpisodes").value,
    maze_source: $("mazeSource").value,
    seed: Math.floor(Math.random() * 1e6),
  };
  if (body.maze_source === "custom") body.custom_maze = maze;

  const r = await fetch(API("/api/inference"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    const t = await r.text();
    $("status").textContent = "inference failed: " + t;
  }
});

// --- init ---------------------------------------------------------------
(async function() {
  await generateMaze();
  // ?inference=name auto-watch — set state directly to avoid the change-event
  // race where two populateModels() calls overlap before the dropdown settles.
  const params = new URLSearchParams(window.location.search);
  if (params.has("inference")) {
    $("trainPanel").style.display = "none";
    $("inferencePanel").style.display = "";
    document.querySelector('input[name=mode][value=inference]').checked = true;
    await populateModels();
    const want = params.get("inference");
    const sel = $("model");
    for (const opt of sel.options) {
      if (opt.value.endsWith("/" + want)) { sel.value = opt.value; break; }
    }
    $("watch").click();
  }
})();
