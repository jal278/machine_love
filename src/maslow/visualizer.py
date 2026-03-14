"""
Pygame-based visualizer for Maslow gridworld simulations.

Shared infrastructure (Tileset, Tilemap, Game) used by both the base
gridworld renderer and the attachment gridworld renderer.

Icon tiles map to gridworld cell types:
  index 0 = empty (align-justify)
  index 1 = physiological / food (home)
  index 2 = safety / job (dollar)
  index 3 = belonging / friends (heart)
  index 4 = esteem / therapy (trophy)
  index 5 = self-actualization / poetry (music)
  index 6 = agent (user)
  index -1 = warning overlay (warning)
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pygame
import pygame.surfarray as surfarray
from pygame.locals import QUIT

# Default icon filenames in order matching gridworld cell values + 1
# (cell -1 → index 0, cell 0 → index 1, ..., cell 4 → index 5, agent → index 6)
DEFAULT_ICON_FILES = [
    "align-justify.png",  # 0: empty/floor
    "home.png",           # 1: physiological
    "dollar.png",         # 2: safety
    "heart.png",          # 3: belonging
    "trophy.png",         # 4: esteem
    "music.png",          # 5: self-actualization
    "user.png",           # 6: agent
    "warning.png",        # 7: warning overlay (last tile)
]


class TilesetBetter:
    """Loads a list of PNG files as individual tiles."""

    def __init__(self, files: list[str], size: tuple[int, int] = (85, 85)):
        self.files = files
        self.size = size
        self.images = [pygame.image.load(f).convert_alpha() for f in files]
        self.rect = self.images[0].get_rect()
        self.tiles = self.images

    def __str__(self) -> str:
        return f"{self.__class__.__name__} files:{self.files} tile:{self.size}"


class Tilemap:
    """Renders a 2-D numpy integer map using tiles from a TilesetBetter."""

    def __init__(self, tileset: TilesetBetter, size: tuple[int, int]):
        self.size = size
        self.tileset = tileset
        self.map = np.zeros(size, dtype=int)
        self.warning = np.zeros(size, dtype=float)
        self.img_w, self.img_h = tileset.size
        h, w = size
        self.image = pygame.Surface((self.img_w * w, self.img_h * h))
        self.rect = self.image.get_rect()

    def render(self):
        m, n = self.map.shape
        for i in range(m):
            for j in range(n):
                tile = self.tileset.tiles[self.map[i, j]]
                self.image.blit(tile, (j * self.img_w, i * self.img_h))
                if self.warning[i, j] == 1.0:
                    overlay = self.tileset.tiles[-1].copy()
                    overlay.fill((255, 255, 255, 128), None, pygame.BLEND_RGBA_MULT)
                    self.image.blit(overlay, (j * self.img_w, i * self.img_h))


class Game:
    """
    Pygame game loop for the Maslow gridworld.

    Works with both MaslowAgent (dict memory) and AttachmentAgent
    (Memory-object memory) by detecting the interface at runtime.

    Args:
        gridworld: a MaslowGridworld or AttachmentGridworld instance
        agent: the corresponding agent
        icons_dir: path to the directory containing the icon PNGs
        record: if True, save each frame to video/screen_NNNN.png
        step_delay: seconds to sleep between frames (default 0.1)
    """

    W = 1280
    H = 800

    def __init__(self, gridworld, agent, icons_dir: Path,
                 record: bool = False, step_delay: float = 0.1):
        pygame.init()
        self.gridworld = gridworld
        self.agent = agent
        self.record = record
        self.step_delay = step_delay

        self.screen = pygame.display.set_mode((self.W, self.H))
        pygame.display.set_caption("Maslow Gridworld")

        icon_paths = [str(icons_dir / f) for f in DEFAULT_ICON_FILES]
        self.tileset = TilesetBetter(icon_paths)
        self.tilemap = Tilemap(self.tileset, gridworld.map.shape)

        self.running = True

    # ------------------------------------------------------------------
    # Memory helpers — work with both dict and Memory-object APIs
    # ------------------------------------------------------------------

    def _memory_has_need(self, i: int) -> bool:
        mem = self.agent.memory
        if hasattr(mem, "need_in_memory"):
            return mem.need_in_memory(i)
        return i in mem

    def _memory_salience(self, i: int) -> float:
        mem = self.agent.memory
        if hasattr(mem, "get_max_salience"):
            return mem.get_max_salience(i)
        return mem[i][0]

    # ------------------------------------------------------------------
    # Draw helpers
    # ------------------------------------------------------------------

    def _draw_needs_bars(self):
        x_offset = 800 + self.tilemap.img_w
        y_offset = 200
        barsize = 20
        spacing = self.tilemap.img_w + 5

        for i in range(len(self.agent.needs)):
            col = (255, 0, 0)
            if self.agent.needs[i] > self.agent.unmet_threshold:
                col = (0, 255, 0)

            bg_rect = pygame.Rect(
                x_offset, y_offset + i * spacing + spacing / 2 - barsize / 2,
                100 * 2.0, barsize
            )
            fg_rect = pygame.Rect(
                x_offset, y_offset + i * spacing + spacing / 2 - barsize / 2,
                self.agent.needs[i] * 2.0, barsize
            )
            self.screen.blit(
                self.tilemap.tileset.tiles[i + 1],
                (x_offset - spacing, y_offset + i * spacing)
            )
            pygame.draw.rect(self.screen, (0, 0, 0), bg_rect)
            pygame.draw.rect(self.screen, col, fg_rect)

            # memory indicator dot
            dot_rect = pygame.Rect(i * 15, 400, 10, 10)
            if self._memory_has_need(i):
                dot_col = (255, 255, 0) if self._memory_salience(i) > 1.0 else (0, 255, 0)
            else:
                dot_col = (255, 0, 0)
            pygame.draw.rect(self.screen, dot_col, dot_rect)

    def _draw_attachment_bars(self):
        """Draw self-awareness and cycle phase bars (attachment agent only)."""
        if not hasattr(self.agent, "self_awareness"):
            return
        x_offset = 800 + self.tilemap.img_w
        y_offset = 200
        barsize = 20
        sa = max(0.0, self.agent.self_awareness * 10.0 * 2.0)
        cp = max(0.0, self.agent.cycle.phase * 10.0 * 2.0)
        pygame.draw.rect(self.screen, (0, 0, 0),
                         (x_offset, y_offset - 100, sa, barsize))
        pygame.draw.rect(self.screen, (0, 0, 0),
                         (x_offset, y_offset - 150, cp, barsize))

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        frame = 0
        while self.running:
            self.tilemap.map, self.tilemap.warning = self.gridworld.export_tilemap()
            self.tilemap.render()

            pygame.draw.rect(self.screen, (255, 255, 255),
                             pygame.Rect(0, 0, self.W, self.H))
            self.screen.blit(self.tilemap.image, self.tilemap.rect)

            self._draw_needs_bars()
            self._draw_attachment_bars()

            pygame.display.update()

            frame += 1
            if self.record:
                import os
                os.makedirs("video", exist_ok=True)
                surfarray.pixels_alpha(self.screen)[:] = 255
                pygame.image.save(self.screen, f"video/screen_{frame:04d}.png")

            time.sleep(self.step_delay)
            action = self.agent.update()
            self.gridworld.step(action)

            for event in pygame.event.get():
                if event.type == QUIT:
                    self.running = False

        pygame.quit()
