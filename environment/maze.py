"""Maze environment + RenderMaze.

Ported from `001-maze-rl/003-environment/maze.py` with fixes:
- find_empty_cell allows the start cell (was buggy: maze[1,1]==2 so cell never
  matched maze==1 and the loop infinite-spun).
- step() reward shaping cleaner; off-grid prevented before lookup.
RenderMaze extended with:
- Q-value overlay (best=green, worst=red) ported from legacy generate_maze.py.
- save(path, fmt='gif'|'webp'|'mp4', frame_skip, max_frames).
- Per-agent tint for n_agents>1.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

HOLE, LAND, START, EXIT, LAVA = 0, 1, 2, 3, 4
AGENT_BASE = 5  # agents are encoded as AGENT_BASE + agent_index

# Length of the auxiliary feature vector appended to the observation when
# aux_features=True: [row, col, unit_dr, unit_dc, dist, remaining_frac].
AUX_DIM = 6
SPRITE_HOLE, SPRITE_LAND, SPRITE_LAVA, SPRITE_EXIT, SPRITE_AGENT = 0, 1, 2, 3, 4

# Per-agent tints for multi-agent renders; index 0 = no tint.
_AGENT_TINTS = [None, (255, 80, 80), (80, 255, 80), (80, 160, 255), (255, 200, 60)]


class MazeEnvironment:
    metadata = {"render.modes": ["human"]}

    def __init__(self, width: int = 10, height: int = 10, n_agents: int = 1,
                 density: float = 0.2, seed: int | None = None,
                 generator: str = "random", ensure_solvable: bool = True,
                 n_lava: int = 0, lava_reward: float = -1.0,
                 partial_view: int | None = None,
                 n_treasures: int = 1, collect_all: bool = False,
                 bump_penalty: float = -0.1,
                 aux_features: bool = False,
                 reward_shaping: bool = False,
                 shaping_gamma: float = 0.99,
                 shaping_coef: float = 0.01):
        if width < 5 or height < 5:
            raise ValueError("Maze must be at least 5x5")
        self.width = int(width)
        self.height = int(height)
        self.density = float(density)
        self.n_agents = int(n_agents)
        self.generator = generator
        self.ensure_solvable = bool(ensure_solvable)
        self.n_lava = int(n_lava)
        self.lava_reward = float(lava_reward)
        self.partial_view = None if partial_view is None else int(partial_view)
        self.n_treasures = max(1, int(n_treasures))
        self.collect_all = bool(collect_all)
        self.bump_penalty = float(bump_penalty)
        self.aux_features = bool(aux_features)
        self.reward_shaping = bool(reward_shaping)
        self.shaping_gamma = float(shaping_gamma)
        self.shaping_coef = float(shaping_coef)
        self._dist_map: np.ndarray | None = None
        self._nearest_src: np.ndarray | None = None
        self._rng = np.random.default_rng(seed)
        self.action_size = 4

        self._remaining: set[tuple[int, int]] = set()
        self.agent_positions: list[tuple[int, int]] = []
        self._build_layout()
        self.reset()

    def _build_layout(self) -> None:
        """(Re)build walls + treasures + lava + start. Uses current self._rng."""
        self.start_pos = (1, 1)
        # First treasure stays at the legacy corner; extras placed later.
        self.treasure_positions = [(self.height - 2, self.width - 2)]
        self.maze = self._generate_maze()
        if self.ensure_solvable:
            self._carve_path_if_needed()
        # markers may have been overwritten by carving — re-set
        self.maze[self.start_pos] = START
        for p in self.treasure_positions:
            self.maze[p] = EXIT
        if self.n_treasures > 1:
            self._place_extra_treasures(self.n_treasures - 1)
        if self.n_lava > 0:
            self._place_lava(self.n_lava)

    def regenerate(self, seed: int | None = None) -> np.ndarray:
        """Re-roll the maze layout. If seed is given, reseeds the rng first.
        Returns the new observation after a reset()."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._build_layout()
        return self.reset()

    # back-compat alias
    @property
    def treasure_pos(self) -> tuple[int, int]:
        return self.treasure_positions[0]

    # ------------------------------------------------------------------
    # generators
    # ------------------------------------------------------------------
    _GENERATORS = ("random", "dfs", "open")

    def _generate_maze(self) -> np.ndarray:
        # Comma-separated generators ("dfs,random") sample one per build —
        # with regenerate_every this trains across topologies, which is what
        # actually buys cross-maze generalization.
        choices = [g.strip() for g in self.generator.split(",") if g.strip()]
        unknown = [g for g in choices if g not in self._GENERATORS]
        if unknown or not choices:
            raise ValueError(f"Unknown generator: {self.generator!r}")
        gen = choices[int(self._rng.integers(len(choices)))] \
            if len(choices) > 1 else choices[0]
        self.generator_used = gen
        if gen == "random":
            return self._gen_random()
        if gen == "dfs":
            return self._gen_dfs()
        return self._gen_open()

    def _gen_random(self) -> np.ndarray:
        m = self._rng.choice([HOLE, LAND], size=(self.height, self.width),
                             p=[self.density, 1 - self.density]).astype(np.uint8)
        m[0, :] = m[-1, :] = m[:, 0] = m[:, -1] = HOLE
        m[self.start_pos] = START
        m[self.treasure_pos] = EXIT
        return m

    def _gen_open(self) -> np.ndarray:
        m = np.full((self.height, self.width), LAND, dtype=np.uint8)
        m[0, :] = m[-1, :] = m[:, 0] = m[:, -1] = HOLE
        m[self.start_pos] = START
        m[self.treasure_pos] = EXIT
        return m

    def _gen_dfs(self) -> np.ndarray:
        """Recursive-backtracker perfect maze on the odd-coord lattice.
        Cells on odd coords become rooms; even coords are walls/passages.
        Guarantees a single connected component covering all odd cells.
        """
        h, w = self.height, self.width
        m = np.full((h, w), HOLE, dtype=np.uint8)
        # snap start to nearest odd interior cell
        sr = 1 if h > 2 else 0
        sc = 1 if w > 2 else 0
        m[sr, sc] = LAND
        stack = [(sr, sc)]
        dirs = [(-2, 0), (2, 0), (0, -2), (0, 2)]
        while stack:
            r, c = stack[-1]
            cands = []
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                if 0 < nr < h - 1 and 0 < nc < w - 1 and m[nr, nc] == HOLE:
                    cands.append((nr, nc, dr, dc))
            if not cands:
                stack.pop()
                continue
            i = int(self._rng.integers(len(cands)))
            nr, nc, dr, dc = cands[i]
            m[r + dr // 2, c + dc // 2] = LAND  # knock wall between
            m[nr, nc] = LAND
            stack.append((nr, nc))
        # Goal may land on a wall for even-sized mazes; nudge it inward
        gr, gc = self.treasure_positions[0]
        if m[gr, gc] == HOLE:
            for ddr, ddc in [(0, 0), (-1, 0), (0, -1), (-1, -1)]:
                if m[gr + ddr, gc + ddc] == LAND:
                    self.treasure_positions[0] = (gr + ddr, gc + ddc)
                    break
            else:
                m[gr, gc] = LAND  # fallback
        m[self.start_pos] = START
        m[self.treasure_pos] = EXIT
        return m

    # ------------------------------------------------------------------
    # solvability
    # ------------------------------------------------------------------
    def _connected(self, src: tuple[int, int], dst: tuple[int, int]) -> bool:
        import collections
        h, w = self.height, self.width
        seen = {src}; q = collections.deque([src])
        while q:
            r, c = q.popleft()
            if (r, c) == dst:
                return True
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in seen \
                        and self.maze[nr, nc] != HOLE:
                    seen.add((nr, nc)); q.append((nr, nc))
        return False

    def _carve_path_if_needed(self) -> None:
        """Ensure connectivity from start to every treasure; carve walls
        with minimum breakage (Dijkstra) for each disconnected goal."""
        for target in self.treasure_positions:
            if self._connected(self.start_pos, target):
                continue
            self._carve_one(target)

    def _carve_one(self, target: tuple[int, int]) -> None:
        import heapq
        h, w = self.height, self.width
        dist = np.full((h, w), float("inf"))
        prev: dict = {}
        dist[self.start_pos] = 0.0
        pq = [(0.0, self.start_pos)]
        while pq:
            d, (r, c) = heapq.heappop(pq)
            if (r, c) == target:
                break
            if d > dist[r, c]:
                continue
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if not (0 < nr < h - 1 and 0 < nc < w - 1):
                    continue
                cost = 1.0 if self.maze[nr, nc] == HOLE else 0.0
                nd = d + cost
                if nd < dist[nr, nc]:
                    dist[nr, nc] = nd
                    prev[(nr, nc)] = (r, c)
                    heapq.heappush(pq, (nd, (nr, nc)))
        cur = target
        while cur != self.start_pos and cur in prev:
            if self.maze[cur] == HOLE:
                self.maze[cur] = LAND
            cur = prev[cur]

    def is_solvable(self) -> bool:
        return all(self._connected(self.start_pos, t)
                   for t in self.treasure_positions)

    # ------------------------------------------------------------------
    # lava placement
    # ------------------------------------------------------------------
    def _bfs_path(self, src: tuple[int, int], dst: tuple[int, int]) -> set:
        """Cells on a BFS shortest path src->dst (LAND/EXIT/START walkable)."""
        import collections
        h, w = self.height, self.width
        prev = {src: None}
        q = collections.deque([src])
        while q:
            r, c = q.popleft()
            if (r, c) == dst:
                break
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if (0 <= nr < h and 0 <= nc < w and (nr, nc) not in prev
                        and self.maze[nr, nc] != HOLE):
                    prev[(nr, nc)] = (r, c); q.append((nr, nc))
        path = set()
        cur = dst
        while cur is not None and cur in prev:
            path.add(cur); cur = prev[cur]
        return path

    def _place_extra_treasures(self, k: int) -> None:
        """Place k additional EXIT cells on LAND cells reachable from start."""
        h, w = self.height, self.width
        # BFS-reachable LAND cells
        import collections
        seen = {self.start_pos}
        q = collections.deque([self.start_pos])
        while q:
            r, c = q.popleft()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if (0 < nr < h - 1 and 0 < nc < w - 1
                        and (nr, nc) not in seen
                        and self.maze[nr, nc] != HOLE):
                    seen.add((nr, nc)); q.append((nr, nc))
        # exclude start + existing treasures
        seen.discard(self.start_pos)
        for p in self.treasure_positions:
            seen.discard(p)
        candidates = [c for c in seen if self.maze[c] == LAND]
        self._rng.shuffle(candidates)
        for p in candidates[:k]:
            self.maze[p] = EXIT
            self.treasure_positions.append(p)

    def _place_lava(self, n: int) -> None:
        """Drop n lava cells on LAND cells not on any start->treasure
        shortest path. Guarantees every treasure remains reachable."""
        h, w = self.height, self.width
        path = set()
        for t in self.treasure_positions:
            path |= self._bfs_path(self.start_pos, t)
        candidates = [(i, j) for i in range(1, h - 1) for j in range(1, w - 1)
                      if self.maze[i, j] == LAND and (i, j) not in path]
        self._rng.shuffle(candidates)
        for (i, j) in candidates[:n]:
            self.maze[i, j] = LAVA

    def _find_empty_cell(self, taken: Sequence[tuple[int, int]]) -> tuple[int, int]:
        candidates = [(i, j) for i in range(1, self.height - 1)
                      for j in range(1, self.width - 1)
                      if self.maze[i, j] in (LAND, START) and (i, j) not in taken]
        if not candidates:
            return self.start_pos
        return tuple(candidates[self._rng.integers(len(candidates))])

    def reset(self, at_start: bool = False) -> np.ndarray:
        # Restore any treasures consumed during the previous episode.
        for p in self.treasure_positions:
            self.maze[p] = EXIT
        self._remaining = set(self.treasure_positions) if self.collect_all else set()

        self.agent_positions = []
        if at_start:
            self.agent_positions.append(self.start_pos)
            for _ in range(self.n_agents - 1):
                self.agent_positions.append(self._find_empty_cell(self.agent_positions))
        else:
            for _ in range(self.n_agents):
                self.agent_positions.append(self._find_empty_cell(self.agent_positions))
        self._dist_map = None  # treasures restored — distances are stale
        return self.get_observation()

    # ------------------------------------------------------------------
    # shaping / aux features (multi-source BFS over remaining treasures)
    # ------------------------------------------------------------------
    @property
    def aux_dim(self) -> int:
        return AUX_DIM if self.aux_features else 0

    @property
    def grid_obs_shape(self) -> tuple[int, int]:
        """Shape of the GRID part of the observation (window or full)."""
        if self.partial_view is not None:
            size = 2 * self.partial_view + 1
            return (size, size)
        return (self.height, self.width)

    def split_observation(self, obs) -> tuple[np.ndarray, np.ndarray | None]:
        """(grid_2d, aux_vec | None). Accepts both the 2-D grid form and the
        flat grid+aux form; the grid comes back as integer cell labels."""
        obs = np.asarray(obs)
        if obs.ndim == 2:
            return obs, None
        gh, gw = self.grid_obs_shape
        grid = obs[:gh * gw].reshape(gh, gw)
        aux = obs[gh * gw:]
        return grid.astype(np.int64), (aux if aux.size else None)

    def _shaping_targets(self) -> set[tuple[int, int]]:
        if self.collect_all:
            return self._remaining
        return {p for p in self.treasure_positions if self.maze[p] == EXIT}

    def _get_dist_map(self) -> tuple[np.ndarray, np.ndarray]:
        """(dist, nearest_src): BFS distance to the nearest remaining
        treasure and that treasure's coordinates, per cell. Cached;
        invalidated on reset/regenerate/treasure consumption."""
        if self._dist_map is None:
            from collections import deque
            H, W = self.height, self.width
            dist = np.full((H, W), np.inf, dtype=np.float32)
            src = np.full((H, W, 2), -1, dtype=np.int16)
            q = deque()
            for (r, c) in self._shaping_targets():
                dist[r, c] = 0.0
                src[r, c] = (r, c)
                q.append((r, c))
            while q:
                r, c = q.popleft()
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < H and 0 <= nc < W
                            and self.maze[nr, nc] != HOLE
                            and not np.isfinite(dist[nr, nc])):
                        dist[nr, nc] = dist[r, c] + 1.0
                        src[nr, nc] = src[r, c]
                        q.append((nr, nc))
            self._dist_map, self._nearest_src = dist, src
        return self._dist_map, self._nearest_src

    def _phi(self, pos: tuple[int, int]) -> float:
        """Potential Φ(s) = -coef · BFS-dist to nearest remaining treasure.
        Function of (position, remaining-set) — both state — so shaping
        r += γΦ(s')-Φ(s) is policy-invariant (Ng-Harada-Russell)."""
        dist, _ = self._get_dist_map()
        d = float(dist[pos])
        if not np.isfinite(d):
            d = float(self.width + self.height)
        return -self.shaping_coef * d

    def _aux_vector(self) -> np.ndarray:
        r, c = self.agent_positions[0]
        dist, src = self._get_dist_map()
        d = float(dist[r, c])
        scale = float(self.width + self.height)
        if np.isfinite(d) and src[r, c][0] >= 0:
            tr, tc = int(src[r, c][0]), int(src[r, c][1])
            dr, dc = tr - r, tc - c
            norm = float(np.hypot(dr, dc))
            unit = (dr / norm, dc / norm) if norm > 0 else (0.0, 0.0)
            d_n = min(d / scale, 1.0)
        else:
            unit, d_n = (0.0, 0.0), 1.0
        remaining = (len(self._remaining) / self.n_treasures
                     if self.collect_all else 1.0)
        return np.array([
            r / max(self.height - 1, 1), c / max(self.width - 1, 1),
            unit[0], unit[1], d_n, remaining,
        ], dtype=np.float32)

    def get_observation(self) -> np.ndarray:
        """Full or partial (egocentric window) observation.

        Partial: returns a (2K+1)x(2K+1) crop centered on agent 0; cells
        outside the maze are HOLE. The agent's marker is always at center.
        """
        full = self.maze.copy()
        for i, pos in enumerate(self.agent_positions):
            full[pos] = AGENT_BASE + i
        if self.partial_view is None:
            return self._with_aux(full)
        K = self.partial_view
        size = 2 * K + 1
        out = np.full((size, size), HOLE, dtype=full.dtype)
        r, c = self.agent_positions[0]
        for di in range(-K, K + 1):
            for dj in range(-K, K + 1):
                rr, cc = r + di, c + dj
                if 0 <= rr < self.height and 0 <= cc < self.width:
                    out[di + K, dj + K] = full[rr, cc]
        return self._with_aux(out)

    def _with_aux(self, grid: np.ndarray) -> np.ndarray:
        if not self.aux_features:
            return grid
        return np.concatenate([grid.ravel().astype(np.float32),
                               self._aux_vector()])

    def step(self, actions: int | Sequence[int]):
        if self.n_agents == 1 and isinstance(actions, (int, np.integer)):
            actions = [int(actions)]
        actions = list(actions)
        assert len(actions) == self.n_agents

        rewards, dones = [], []
        for i, action in enumerate(actions):
            r, c = self.agent_positions[i]
            phi_s = self._phi((r, c)) if self.reward_shaping else 0.0
            nr, nc = r, c
            if action == 0:   nr = max(0, r - 1)
            elif action == 1: nc = min(self.width - 1, c + 1)
            elif action == 2: nr = min(self.height - 1, r + 1)
            elif action == 3: nc = max(0, c - 1)

            target = (nr, nc)
            cell = self.maze[target]
            blocked = cell == HOLE or target in self.agent_positions
            if blocked:
                target = (r, c)
                reward, done = self.bump_penalty, False
            elif cell == EXIT:
                reward = 1.0
                if self.collect_all:
                    # consume this treasure; done only when all collected
                    self.maze[target] = LAND
                    self._remaining.discard(target)
                    done = not self._remaining
                else:
                    done = True
                self._dist_map = None  # remaining-set changed
            elif cell == LAVA:
                reward, done = self.lava_reward, True
            else:
                reward, done = -0.01, False

            self.agent_positions[i] = target
            if self.reward_shaping:
                # Potential-based: r += γΦ(s') - Φ(s); Φ(terminal) = 0.
                phi_ns = 0.0 if done else self._phi(target)
                reward += self.shaping_gamma * phi_ns - phi_s
            rewards.append(reward)
            dones.append(done)

        info = {}
        if self.n_agents == 1:
            return self.get_observation(), rewards[0], dones[0], info
        return self.get_observation(), rewards, dones, info

    def get_possible_actions(self) -> list[int]:
        return [0, 1, 2, 3]

    def render(self, mode: str = "human"):
        if mode != "human":
            return self.get_observation()
        glyph = {HOLE: "█", LAND: "·", START: "S", EXIT: "T"}
        for i in range(self.height):
            row = []
            for j in range(self.width):
                if (i, j) in self.agent_positions:
                    row.append(f"A{self.agent_positions.index((i, j))}")
                else:
                    row.append(glyph.get(int(self.maze[i, j]), "?"))
            print(" ".join(row))
        print()


# Back-compat: RenderMaze historically lived in this module.
from render import RenderMaze  # noqa: E402, F401
