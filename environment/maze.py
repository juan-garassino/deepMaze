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

import os
from collections.abc import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HOLE, LAND, START, EXIT, LAVA = 0, 1, 2, 3, 4
AGENT_BASE = 5  # agents are encoded as AGENT_BASE + agent_index
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
                 bump_penalty: float = -0.1):
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
    def _generate_maze(self) -> np.ndarray:
        if self.generator == "random":
            return self._gen_random()
        if self.generator == "dfs":
            return self._gen_dfs()
        if self.generator == "open":
            return self._gen_open()
        raise ValueError(f"Unknown generator: {self.generator!r}")

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
        return self.get_observation()

    def get_observation(self) -> np.ndarray:
        """Full or partial (egocentric window) observation.

        Partial: returns a (2K+1)x(2K+1) crop centered on agent 0; cells
        outside the maze are HOLE. The agent's marker is always at center.
        """
        full = self.maze.copy()
        for i, pos in enumerate(self.agent_positions):
            full[pos] = AGENT_BASE + i
        if self.partial_view is None:
            return full
        K = self.partial_view
        size = 2 * K + 1
        out = np.full((size, size), HOLE, dtype=full.dtype)
        r, c = self.agent_positions[0]
        for di in range(-K, K + 1):
            for dj in range(-K, K + 1):
                rr, cc = r + di, c + dj
                if 0 <= rr < self.height and 0 <= cc < self.width:
                    out[di + K, dj + K] = full[rr, cc]
        return out

    def step(self, actions: int | Sequence[int]):
        if self.n_agents == 1 and isinstance(actions, (int, np.integer)):
            actions = [int(actions)]
        actions = list(actions)
        assert len(actions) == self.n_agents

        rewards, dones = [], []
        for i, action in enumerate(actions):
            r, c = self.agent_positions[i]
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
            elif cell == LAVA:
                reward, done = self.lava_reward, True
            else:
                reward, done = -0.01, False

            self.agent_positions[i] = target
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


# ---------------------------------------------------------------------------
# RenderMaze
# ---------------------------------------------------------------------------


class RenderMaze:
    """Assemble sprite-tiled frames of maze states and save as GIF/WebP/MP4.

    Frames are produced from (maze_state, agent_position_or_positions) tuples
    pushed via .add(). Q-values, if supplied, are overlaid on the frame.
    """

    def __init__(self, cropped_images: list[Image.Image]):
        if len(cropped_images) < 5:
            raise ValueError(
                f"Expected >=5 cropped sprites (HOLE,LAND,LAVA,EXIT,AGENT), "
                f"got {len(cropped_images)}"
            )
        if not all(isinstance(im, Image.Image) for im in cropped_images):
            raise TypeError("All sprites must be PIL Image instances")
        self.sprites = cropped_images
        self.frames: list[tuple[np.ndarray, object, np.ndarray | None]] = []

    def add(self, maze: np.ndarray, position, q_values: np.ndarray | None = None):
        self.frames.append((np.asarray(maze).copy(), position,
                            None if q_values is None else np.asarray(q_values).copy()))

    def __len__(self):
        return len(self.frames)

    def _tile(self, maze: np.ndarray, sprite_size: int) -> Image.Image:
        h, w = maze.shape
        canvas = Image.new("RGBA", (w * sprite_size, h * sprite_size))
        for i in range(h):
            for j in range(w):
                v = int(maze[i, j])
                if v >= AGENT_BASE: idx = SPRITE_AGENT
                elif v == HOLE:     idx = SPRITE_HOLE
                elif v == EXIT:     idx = SPRITE_EXIT
                elif v == LAVA:     idx = SPRITE_LAVA
                else:               idx = SPRITE_LAND
                sp = self.sprites[idx]
                if sp.size != (sprite_size, sprite_size):
                    sp = sp.resize((sprite_size, sprite_size), Image.LANCZOS)
                canvas.paste(sp.convert("RGBA"), (j * sprite_size, i * sprite_size))
                if v >= AGENT_BASE:
                    tint = _AGENT_TINTS[(v - AGENT_BASE) % len(_AGENT_TINTS)]
                    if tint is not None:
                        overlay = Image.new("RGBA", (sprite_size, sprite_size),
                                            tint + (90,))
                        canvas.alpha_composite(overlay, (j * sprite_size, i * sprite_size))
        return canvas

    def _overlay_q(self, frame: Image.Image, q: np.ndarray, sprite_size: int,
                   step: int):
        draw = ImageDraw.Draw(frame)
        try:
            font = ImageFont.truetype("Arial.ttf", max(10, sprite_size // 3))
        except OSError:
            font = ImageFont.load_default()
        labels = ["UP", "RT", "DN", "LF"]
        best = int(np.argmax(q))
        worst = int(np.argmin(q))
        x0 = 4
        y0 = 4
        for i, (lab, val) in enumerate(zip(labels, q)):
            color = (80, 220, 80) if i == best else (220, 80, 80) if i == worst else (240, 240, 240)
            draw.text((x0, y0 + i * (sprite_size // 3 + 2)),
                      f"{lab} {val:+.2f}", fill=color, font=font)
        draw.text((frame.size[0] - 60, 4), f"t={step}", fill=(255, 255, 255), font=font)

    def render_frames(self, sprite_size: int = 32,
                      frame_skip: int = 1, max_frames: int | None = None
                      ) -> list[Image.Image]:
        if not self.frames:
            return []
        keep_idx = self._frame_indices(frame_skip, max_frames)

        # If the underlying maze never changes across kept frames, we render
        # the static base once and only redraw the agent sprite per frame.
        bases = [self.frames[i][0] for i in keep_idx]
        first = bases[0]
        static = all(b.shape == first.shape and (b == first).all() for b in bases)

        base_rgba = None
        if static:
            base_rgba = self._tile(first, sprite_size)  # no agents

        out = []
        for idx in keep_idx:
            maze, pos, q = self.frames[idx]
            positions = pos if isinstance(pos, list) else [pos]
            if static and base_rgba is not None:
                img = base_rgba.copy()
                for ai, p in enumerate(positions):
                    if p is None:
                        continue
                    sp = self.sprites[SPRITE_AGENT]
                    if sp.size != (sprite_size, sprite_size):
                        sp = sp.resize((sprite_size, sprite_size), Image.LANCZOS)
                    img.paste(sp.convert("RGBA"),
                              (p[1] * sprite_size, p[0] * sprite_size))
                    tint = _AGENT_TINTS[(ai) % len(_AGENT_TINTS)]
                    if tint is not None:
                        overlay = Image.new("RGBA", (sprite_size, sprite_size),
                                            tint + (90,))
                        img.alpha_composite(overlay,
                                            (p[1] * sprite_size, p[0] * sprite_size))
                img = img.convert("RGB")
            else:
                m = maze.copy()
                for ai, p in enumerate(positions):
                    if p is not None:
                        m[p] = AGENT_BASE + ai
                img = self._tile(m, sprite_size).convert("RGB")
            if q is not None:
                self._overlay_q(img, q, sprite_size, step=idx)
            out.append(img)
        return out

    def _frame_indices(self, frame_skip: int, max_frames: int | None) -> list[int]:
        n = len(self.frames)
        idxs = list(range(0, n, max(1, int(frame_skip))))
        if idxs and idxs[-1] != n - 1:
            idxs.append(n - 1)
        if max_frames and len(idxs) > max_frames:
            stride = len(idxs) / max_frames
            idxs = [idxs[int(i * stride)] for i in range(max_frames)]
            if idxs[-1] != n - 1:
                idxs[-1] = n - 1
        return idxs

    def save(self, path: str, fmt: str = "webp", sprite_size: int = 32,
             frame_duration_ms: int = 80, frame_skip: int = 1,
             max_frames: int | None = None) -> str:
        """Render and write the animation to `path`. Returns the output path."""
        frames = self.render_frames(sprite_size, frame_skip, max_frames)
        if not frames:
            raise RuntimeError("No frames to save (call .add() first).")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fmt = fmt.lower()
        if fmt == "gif":
            frames[0].save(path, format="GIF", save_all=True,
                           append_images=frames[1:], duration=frame_duration_ms,
                           loop=0, optimize=True, disposal=2)
        elif fmt == "webp":
            frames[0].save(path, format="WEBP", save_all=True,
                           append_images=frames[1:], duration=frame_duration_ms,
                           loop=0, quality=70, method=4)
        elif fmt == "mp4":
            try:
                import imageio.v2 as imageio
            except Exception as e:
                raise RuntimeError(
                    "MP4 output requires `imageio[ffmpeg]`. "
                    "Install with `pip install imageio[ffmpeg]`."
                ) from e
            fps = max(1, int(round(1000 / max(1, frame_duration_ms))))
            with imageio.get_writer(path, fps=fps, codec="libx264",
                                    quality=8) as w:
                for f in frames:
                    w.append_data(np.asarray(f))
        else:
            raise ValueError(f"Unknown format: {fmt}")
        return path

    # Back-compat shim with old API (timestamped filename in cwd).
    def show(self, sprite_size: int = 32, gif_duration: int = 60) -> str:
        import datetime
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"moving_maze_{ts}.gif"
        return self.save(path, fmt="gif", sprite_size=sprite_size,
                         frame_duration_ms=gif_duration)

    @staticmethod
    def crop_images(input_path: str, image_names: list[str],
                    tile_h: int = 16, tile_w: int = 16, sprite_size: int = 32,
                    return_indexes: list[int] | None = None) -> list[Image.Image]:
        out: list[Image.Image] = []
        for name in image_names:
            with Image.open(os.path.join(input_path, name)) as im:
                iw, ih = im.size
                for y in range(0, ih, tile_h):
                    for x in range(0, iw, tile_w):
                        crop = im.crop((x, y, x + tile_w, y + tile_h))
                        out.append(crop.resize((sprite_size, sprite_size),
                                               Image.LANCZOS).copy())
        if return_indexes is not None:
            return [out[i] for i in return_indexes]
        return out

    @staticmethod
    def placeholder_sprites(sprite_size: int = 32) -> list[Image.Image]:
        """Fallback solid-color sprites when no sprite sheet is provided."""
        palette = {
            SPRITE_HOLE:  (30, 30, 30),
            SPRITE_LAND:  (200, 200, 200),
            SPRITE_LAVA:  (220, 80, 30),
            SPRITE_EXIT:  (240, 200, 40),
            SPRITE_AGENT: (60, 130, 220),
        }
        out = []
        for i in range(5):
            img = Image.new("RGBA", (sprite_size, sprite_size), palette[i] + (255,))
            d = ImageDraw.Draw(img)
            d.rectangle([0, 0, sprite_size - 1, sprite_size - 1],
                        outline=(0, 0, 0, 255), width=1)
            out.append(img)
        return out
