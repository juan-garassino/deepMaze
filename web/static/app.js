/* deepMaze browser client.
 * Renders maze + agent on <canvas>, draws-cells editor, SSE event consumer,
 * Chart.js live charts. No build step.
 */

const HOLE = 0, LAND = 1, START = 2, EXIT = 3;
const COLORS = {
  [HOLE]:  "#1a1a1a",
  [LAND]:  "#cfd0d3",
  [START]: "#3fa75a",
  [EXIT]:  "#e8c84a",
};
const AGENT_TINTS = ["#3a7db0", "#e25555", "#4ec07b", "#d28dff"];

const $ = (id) => document.getElementById(id);
const canvas = $("maze");
const ctx = canvas.getContext("2d");

let W = +$("w").value, H = +$("h").value;
let maze = newMaze(W, H, +$("density").value);
let agentPos = null;
let cycle = [LAND, HOLE, START, EXIT]; // toggle order on click

function newMaze(w, h, density) {
  const m = Array.from({length: h}, (_, i) =>
    Array.from({length: w}, (_, j) => {
      if (i === 0 || j === 0 || i === h - 1 || j === w - 1) return HOLE;
      return Math.random() < density ? HOLE : LAND;
    }),
  );
  m[1][1] = START;
  m[h - 2][w - 2] = EXIT;
  return m;
}

function cellSize() {
  return Math.floor(Math.min(canvas.width / W, canvas.height / H));
}

function draw() {
  const s = cellSize();
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  for (let i = 0; i < H; i++) {
    for (let j = 0; j < W; j++) {
      ctx.fillStyle = COLORS[maze[i][j]] || COLORS[LAND];
      ctx.fillRect(j * s, i * s, s - 1, s - 1);
    }
  }
  if (agentPos) {
    const [r, c] = agentPos;
    ctx.fillStyle = AGENT_TINTS[0];
    ctx.beginPath();
    ctx.arc(c * s + s / 2, r * s + s / 2, s * 0.35, 0, 2 * Math.PI);
    ctx.fill();
  }
}

canvas.addEventListener("click", (e) => {
  const rect = canvas.getBoundingClientRect();
  const s = cellSize();
  const j = Math.floor((e.clientX - rect.left) / s);
  const i = Math.floor((e.clientY - rect.top) / s);
  if (i <= 0 || j <= 0 || i >= H - 1 || j >= W - 1) return;
  const cur = maze[i][j];
  const next = cycle[(cycle.indexOf(cur) + 1) % cycle.length];
  // Ensure unique START and EXIT.
  if (next === START) clear(START);
  if (next === EXIT) clear(EXIT);
  maze[i][j] = next;
  draw();
});

function clear(v) {
  for (let i = 0; i < H; i++)
    for (let j = 0; j < W; j++)
      if (maze[i][j] === v) maze[i][j] = LAND;
}

$("regen").addEventListener("click", () => {
  W = +$("w").value; H = +$("h").value;
  maze = newMaze(W, H, +$("density").value);
  agentPos = null;
  draw();
});

// --- charts ---------------------------------------------------------------
const rewardChart = new Chart($("rewardChart"), {
  type: "line",
  data: { labels: [], datasets: [{ label: "reward", data: [], borderColor: "#7ec7ff", tension: 0.2, pointRadius: 0 }] },
  options: { animation: false, scales: { x: { ticks: { color: "#888" } }, y: { ticks: { color: "#888" } } }, plugins: { legend: { labels: { color: "#ccc" } } } },
});
const lengthChart = new Chart($("lengthChart"), {
  type: "line",
  data: { labels: [], datasets: [{ label: "length", data: [], borderColor: "#ffb479", tension: 0.2, pointRadius: 0 }] },
  options: { animation: false, scales: { x: { ticks: { color: "#888" } }, y: { ticks: { color: "#888" } } }, plugins: { legend: { labels: { color: "#ccc" } } } },
});

// --- SSE -----------------------------------------------------------------
let es = null;
function startStream() {
  if (es) es.close();
  es = new EventSource("/api/events");
  es.onmessage = (msg) => {
    const ev = JSON.parse(msg.data);
    if (ev.type === "step") {
      // Full payload at episode start: rebuild maze + place agent.
      const state = ev.state;
      H = state.length; W = state[0].length;
      maze = state.map(row => row.map(v => v >= 4 ? LAND : v));
      agentPos = ev.position;
      draw();
    } else if (ev.type === "step_delta") {
      agentPos = ev.position;
      draw();
    } else if (ev.type === "episode") {
      rewardChart.data.labels.push(ev.episode);
      rewardChart.data.datasets[0].data.push(ev.total_reward);
      lengthChart.data.labels.push(ev.episode);
      lengthChart.data.datasets[0].data.push(ev.length);
      if (rewardChart.data.labels.length % 5 === 0) {
        rewardChart.update("none");
        lengthChart.update("none");
      }
      $("status").textContent = `ep ${ev.episode} R=${ev.total_reward.toFixed(2)} len=${ev.length}`;
    } else if (ev.type === "run" && ev.kind === "end") {
      $("status").textContent = "done";
      rewardChart.update("none");
      lengthChart.update("none");
    }
  };
  es.onerror = () => { $("status").textContent = "stream closed"; };
}

$("train").addEventListener("click", async () => {
  W = +$("w").value; H = +$("h").value;
  // Re-sync size if changed.
  if (maze.length !== H || maze[0].length !== W) maze = newMaze(W, H, +$("density").value);
  $("status").textContent = "starting…";
  rewardChart.data.labels = []; rewardChart.data.datasets[0].data = [];
  lengthChart.data.labels = []; lengthChart.data.datasets[0].data = [];
  rewardChart.update("none"); lengthChart.update("none");

  startStream();
  await fetch("/api/runs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      agent_type: $("agent").value,
      width: W, height: H,
      density: +$("density").value,
      num_episodes: +$("episodes").value,
      max_steps: +$("max_steps").value,
      maze,
    }),
  });
});

// initial render
$("regen").click();
draw();
