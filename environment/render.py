"""RenderMaze — sprite-tiled replay frames saved as WebP/GIF/MP4.

Split out of maze.py (which keeps the MazeEnvironment); maze.py re-exports
RenderMaze so existing `from maze import RenderMaze` imports keep working.
"""

from __future__ import annotations

import os

import numpy as np
from maze import (
    _AGENT_TINTS,  # per-agent tints (index 0 = no tint)
    AGENT_BASE,
    EXIT,
    HOLE,
    LAVA,
    SPRITE_AGENT,
    SPRITE_EXIT,
    SPRITE_HOLE,
    SPRITE_LAND,
    SPRITE_LAVA,
)
from PIL import Image, ImageDraw, ImageFont


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
