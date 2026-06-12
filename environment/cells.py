"""Cell-value and sprite vocabulary shared by MazeEnvironment and RenderMaze.

Lives in its own module so render.py does not need to import maze.py
(maze re-exports RenderMaze, which would otherwise be a cycle).
"""

HOLE, LAND, START, EXIT, LAVA = 0, 1, 2, 3, 4
AGENT_BASE = 5  # agents are encoded as AGENT_BASE + agent_index
SPRITE_HOLE, SPRITE_LAND, SPRITE_LAVA, SPRITE_EXIT, SPRITE_AGENT = 0, 1, 2, 3, 4

# Per-agent tints for multi-agent renders; index 0 = no tint.
AGENT_TINTS = [None, (255, 80, 80), (80, 255, 80), (80, 160, 255), (255, 200, 60)]
