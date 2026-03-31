"""
RecurrentAI — Interactive Hopfield Network Visualizer
=====================================================
Draw patterns on a grid, store them as memories, corrupt them with noise,
and watch the network recall the original in real time.

Built on the Hopfield network logic from Hopfield_Nets_Noemia.py.

Controls:
  Left-click / drag   — paint cell ON
  Right-click / drag  — paint cell OFF
  [S] Store           — memorise current grid
  [N] Noise           — flip random bits (cycles 10%→25%→40%)
  [R] Recall          — auto-step until settled
  [SPACE] Step        — single async update
  [C] Clear grid      — blank the canvas
  [E] Erase memories  — forget all stored patterns
  [1-4] Presets       — load a built-in shape
  [UP/DOWN]           — resize grid (6×6 … 14×14)
  [Q / ESC] Quit
"""

import sys
import math
import random
import pygame

# ---------------------------------------------------------------------------
# Hopfield engine (adapted from Hoppynets in Hopfield_Nets_Noemia.py)
# ---------------------------------------------------------------------------

class HopfieldNet:
    """Hopfield network using 0/1 encoding and Hebbian learning."""

    def __init__(self, n):
        self.n = n
        self.q = 0.0
        self.state = [0] * n
        self.w = [[0.0] * n for _ in range(n)]
        self.patterns = []

    def erase_memory(self):
        for i in range(self.n):
            for j in range(self.n):
                self.w[i][j] = 0.0
        self.patterns.clear()

    def set_state(self, pattern):
        self.state = list(pattern)

    def train_one(self, pattern):
        self.patterns.append(list(pattern))
        for i in range(self.n):
            for j in range(i + 1, self.n):
                delta = 1.0 if pattern[i] == pattern[j] else -1.0
                self.w[i][j] += delta
                self.w[j][i] += delta

    def net_input(self, idx):
        total = 0.0
        for j in range(self.n):
            if j != idx:
                total += self.w[idx][j] * self.state[j]
        return total

    def energy(self):
        total = 0.0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                total += self.w[i][j] * self.state[i] * self.state[j]
        return -total

    def async_step(self, updates_per_step=None):
        """Update random neurons. Returns number of flips."""
        n_updates = updates_per_step or self.n
        indices = list(range(self.n))
        random.shuffle(indices)
        flipped = []
        for i in indices[:n_updates]:
            new_val = 1 if self.net_input(i) > self.q else 0
            if new_val != self.state[i]:
                flipped.append(i)
                self.state[i] = new_val
        return flipped

    def is_settled(self):
        for i in range(self.n):
            desired = 1 if self.net_input(i) > self.q else 0
            if desired != self.state[i]:
                return False
        return True

    def hamming(self, other):
        return sum(a != b for a, b in zip(self.state, other))

    @staticmethod
    def add_noise(pattern, p):
        out = list(pattern)
        for i in range(len(out)):
            if random.random() < p:
                out[i] = 1 - out[i]
        return out


# ---------------------------------------------------------------------------
# Preset patterns (for various grid sizes)
# ---------------------------------------------------------------------------

def make_presets(grid):
    """Return up to 4 preset pattern bitmaps for a grid×grid network."""
    n = grid * grid
    presets = {}

    # --- heart ---
    heart = [0] * n
    cx, cy = grid / 2, grid / 2
    for r in range(grid):
        for c in range(grid):
            x = (c - cx + 0.5) / (grid / 2)
            y = (r - cy + 0.5) / (grid / 2)
            y = -y + 0.2
            if (x * x + (y - abs(x) * 0.6) ** 2) < 0.45:
                heart[r * grid + c] = 1
    presets["heart"] = heart

    # --- cross ---
    cross = [0] * n
    lo, hi = grid // 4, grid - grid // 4
    for r in range(grid):
        for c in range(grid):
            if lo <= r < hi or lo <= c < hi:
                if lo <= r < hi and lo <= c < hi:
                    cross[r * grid + c] = 1
                elif lo <= r < hi:
                    cross[r * grid + c] = 1
                elif lo <= c < hi:
                    cross[r * grid + c] = 1
    presets["cross"] = cross

    # --- diamond ---
    diamond = [0] * n
    mid = grid // 2
    for r in range(grid):
        for c in range(grid):
            if abs(r - mid) + abs(c - mid) <= mid - 1:
                diamond[r * grid + c] = 1
    presets["diamond"] = diamond

    # --- checker ---
    checker = [0] * n
    block = max(1, grid // 4)
    for r in range(grid):
        for c in range(grid):
            if ((r // block) + (c // block)) % 2 == 0:
                checker[r * grid + c] = 1
    presets["checker"] = checker

    return presets


# ---------------------------------------------------------------------------
# Colours & drawing helpers
# ---------------------------------------------------------------------------

BG          = (18, 18, 24)
PANEL_BG    = (26, 26, 36)
CELL_ON     = (0, 224, 160)
CELL_OFF    = (38, 38, 52)
CELL_FLIP   = (255, 100, 80)
GRID_LINE   = (30, 30, 42)
TEXT_COL    = (200, 200, 210)
DIM_TEXT    = (100, 100, 120)
ACCENT      = (0, 180, 140)
ENERGY_COL  = (80, 180, 255)
ENERGY_LOW  = (0, 224, 160)
THUMB_ON    = (0, 160, 120)
THUMB_OFF   = (44, 44, 58)
BTN_BG      = (40, 40, 56)
BTN_HOVER   = (55, 55, 75)
BTN_TEXT    = (220, 220, 230)
NOISE_COLS  = [(255, 200, 60), (255, 140, 50), (255, 80, 60)]


def lerp_color(a, b, t):
    t = max(0, min(1, t))
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


# ---------------------------------------------------------------------------
# Button helper
# ---------------------------------------------------------------------------

class Button:
    def __init__(self, rect, label, key_hint="", color=BTN_BG):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.key_hint = key_hint
        self.color = color
        self.hover = False

    def draw(self, surface, font, small_font):
        col = BTN_HOVER if self.hover else self.color
        pygame.draw.rect(surface, col, self.rect, border_radius=6)
        pygame.draw.rect(surface, ACCENT, self.rect, 1, border_radius=6)
        lbl = font.render(self.label, True, BTN_TEXT)
        surface.blit(lbl, lbl.get_rect(center=self.rect.center))
        if self.key_hint:
            hint = small_font.render(self.key_hint, True, DIM_TEXT)
            surface.blit(hint, (self.rect.right - hint.get_width() - 4,
                                self.rect.bottom - hint.get_height() - 2))

    def hit(self, pos):
        return self.rect.collidepoint(pos)


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

NOISE_LEVELS = [0.10, 0.25, 0.40]
MIN_GRID = 6
MAX_GRID = 14
DEFAULT_GRID = 10
MAX_MEMORIES = 6
RECALL_SPEED = 8          # async steps per frame during auto-recall
FPS = 60
WIN_W, WIN_H = 900, 700


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("RecurrentAI — Hopfield Network Visualizer")
    clock = pygame.time.Clock()

    font = pygame.font.SysFont("Menlo,Consolas,monospace", 15)
    big_font = pygame.font.SysFont("Menlo,Consolas,monospace", 20, bold=True)
    small_font = pygame.font.SysFont("Menlo,Consolas,monospace", 11)
    title_font = pygame.font.SysFont("Menlo,Consolas,monospace", 24, bold=True)

    # --- state ---
    grid_size = DEFAULT_GRID
    net = HopfieldNet(grid_size * grid_size)
    presets = make_presets(grid_size)
    preset_keys = list(presets.keys())

    noise_idx = 0
    auto_recalling = False
    recall_steps = 0
    energy_history = []
    flash_cells = {}          # idx -> frames remaining
    settled_flash = 0
    message = ""
    message_timer = 0

    def rebuild_net():
        nonlocal net, presets, preset_keys, energy_history, flash_cells
        nonlocal auto_recalling, recall_steps, settled_flash
        net = HopfieldNet(grid_size * grid_size)
        presets = make_presets(grid_size)
        preset_keys = list(presets.keys())
        energy_history.clear()
        flash_cells.clear()
        auto_recalling = False
        recall_steps = 0
        settled_flash = 0

    def set_message(msg, frames=120):
        nonlocal message, message_timer
        message = msg
        message_timer = frames

    def do_step():
        nonlocal recall_steps
        flipped = net.async_step()
        for idx in flipped:
            flash_cells[idx] = 10
        energy_history.append(net.energy())
        if len(energy_history) > 300:
            energy_history.pop(0)
        recall_steps += 1
        return flipped

    # --- layout constants ---
    GRID_LEFT = 30
    GRID_TOP = 70
    GRID_AREA = 440
    PANEL_LEFT = GRID_LEFT + GRID_AREA + 30
    PANEL_W = WIN_W - PANEL_LEFT - 20
    ENERGY_TOP = WIN_H - 140
    ENERGY_H = 110

    # --- buttons ---
    bw = (PANEL_W - 10) // 2
    bx1 = PANEL_LEFT
    bx2 = PANEL_LEFT + bw + 10
    btn_y = GRID_TOP
    btn_h = 34
    gap = 42

    buttons = {
        "store":  Button((bx1, btn_y,           bw, btn_h), "Store",  "S"),
        "noise":  Button((bx2, btn_y,           bw, btn_h), "Noise",  "N"),
        "recall": Button((bx1, btn_y + gap,     bw, btn_h), "Recall", "R"),
        "step":   Button((bx2, btn_y + gap,     bw, btn_h), "Step",   "SPC"),
        "clear":  Button((bx1, btn_y + gap * 2, bw, btn_h), "Clear",  "C"),
        "erase":  Button((bx2, btn_y + gap * 2, bw, btn_h), "Erase",  "E"),
    }

    running = True
    mouse_painting = None     # 1 = paint on, 0 = paint off

    while running:
        dt = clock.tick(FPS)
        mx, my = pygame.mouse.get_pos()

        cell_size = GRID_AREA // grid_size
        grid_px = cell_size * grid_size

        # --- events ---
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False

            elif ev.type == pygame.MOUSEBUTTONDOWN:
                # grid painting
                gx = (ev.pos[0] - GRID_LEFT) // cell_size
                gy = (ev.pos[1] - GRID_TOP) // cell_size
                if 0 <= gx < grid_size and 0 <= gy < grid_size:
                    idx = gy * grid_size + gx
                    if ev.button == 1:
                        net.state[idx] = 1
                        mouse_painting = 1
                    elif ev.button == 3:
                        net.state[idx] = 0
                        mouse_painting = 0
                    auto_recalling = False

                # buttons
                for name, btn in buttons.items():
                    if btn.hit(ev.pos):
                        if name == "store":
                            if len(net.patterns) >= MAX_MEMORIES:
                                set_message(f"Max {MAX_MEMORIES} memories — erase first")
                            else:
                                net.train_one(list(net.state))
                                set_message(f"Stored pattern #{len(net.patterns)}")
                        elif name == "noise":
                            if not net.patterns:
                                set_message("Store at least one pattern first")
                            else:
                                nl = NOISE_LEVELS[noise_idx]
                                net.state = HopfieldNet.add_noise(net.state, nl)
                                noise_idx = (noise_idx + 1) % len(NOISE_LEVELS)
                                energy_history.clear()
                                set_message(f"Added {int(nl*100)}% noise")
                        elif name == "recall":
                            if not net.patterns:
                                set_message("No memories stored yet")
                            else:
                                auto_recalling = True
                                recall_steps = 0
                                energy_history.clear()
                                set_message("Recalling…")
                        elif name == "step":
                            if net.patterns:
                                do_step()
                            else:
                                set_message("No memories stored yet")
                        elif name == "clear":
                            net.state = [0] * net.n
                            energy_history.clear()
                            auto_recalling = False
                            flash_cells.clear()
                            set_message("Grid cleared")
                        elif name == "erase":
                            rebuild_net()
                            set_message("All memories erased")

            elif ev.type == pygame.MOUSEBUTTONUP:
                mouse_painting = None

            elif ev.type == pygame.MOUSEMOTION and mouse_painting is not None:
                gx = (ev.pos[0] - GRID_LEFT) // cell_size
                gy = (ev.pos[1] - GRID_TOP) // cell_size
                if 0 <= gx < grid_size and 0 <= gy < grid_size:
                    net.state[gy * grid_size + gx] = mouse_painting

            elif ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif ev.key == pygame.K_s:
                    if len(net.patterns) >= MAX_MEMORIES:
                        set_message(f"Max {MAX_MEMORIES} memories — erase first")
                    else:
                        net.train_one(list(net.state))
                        set_message(f"Stored pattern #{len(net.patterns)}")
                elif ev.key == pygame.K_n:
                    if not net.patterns:
                        set_message("Store at least one pattern first")
                    else:
                        nl = NOISE_LEVELS[noise_idx]
                        net.state = HopfieldNet.add_noise(net.state, nl)
                        noise_idx = (noise_idx + 1) % len(NOISE_LEVELS)
                        energy_history.clear()
                        set_message(f"Added {int(nl*100)}% noise")
                elif ev.key == pygame.K_r:
                    if not net.patterns:
                        set_message("No memories stored yet")
                    else:
                        auto_recalling = True
                        recall_steps = 0
                        energy_history.clear()
                        set_message("Recalling…")
                elif ev.key == pygame.K_SPACE:
                    if net.patterns:
                        do_step()
                    else:
                        set_message("No memories stored yet")
                elif ev.key == pygame.K_c:
                    net.state = [0] * net.n
                    energy_history.clear()
                    auto_recalling = False
                    flash_cells.clear()
                    set_message("Grid cleared")
                elif ev.key == pygame.K_e:
                    rebuild_net()
                    set_message("All memories erased")
                elif ev.key == pygame.K_UP:
                    if grid_size < MAX_GRID:
                        grid_size += 2
                        rebuild_net()
                        set_message(f"Grid: {grid_size}×{grid_size}")
                elif ev.key == pygame.K_DOWN:
                    if grid_size > MIN_GRID:
                        grid_size -= 2
                        rebuild_net()
                        set_message(f"Grid: {grid_size}×{grid_size}")
                elif ev.key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4):
                    pi = ev.key - pygame.K_1
                    if pi < len(preset_keys):
                        net.state = list(presets[preset_keys[pi]])
                        auto_recalling = False
                        energy_history.clear()
                        set_message(f"Loaded preset: {preset_keys[pi]}")

        # --- auto recall ---
        if auto_recalling:
            for _ in range(RECALL_SPEED):
                do_step()
                if net.is_settled():
                    auto_recalling = False
                    settled_flash = 30
                    best = None
                    best_hd = net.n
                    for i, p in enumerate(net.patterns):
                        hd = net.hamming(p)
                        if hd < best_hd:
                            best_hd = hd
                            best = i + 1
                    set_message(
                        f"Settled in {recall_steps} steps — "
                        f"closest to memory #{best} (hamming {best_hd})",
                        180
                    )
                    break

        # --- flash decay ---
        to_remove = []
        for idx in flash_cells:
            flash_cells[idx] -= 1
            if flash_cells[idx] <= 0:
                to_remove.append(idx)
        for idx in to_remove:
            del flash_cells[idx]
        if settled_flash > 0:
            settled_flash -= 1

        if message_timer > 0:
            message_timer -= 1

        # === DRAW ===
        screen.fill(BG)

        # title
        title = title_font.render("RecurrentAI", True, ACCENT)
        screen.blit(title, (GRID_LEFT, 18))
        sub = small_font.render("Hopfield Network Visualizer", True, DIM_TEXT)
        screen.blit(sub, (GRID_LEFT + title.get_width() + 12, 26))

        # grid size label
        gs_label = small_font.render(f"{grid_size}×{grid_size}  [↑↓]", True, DIM_TEXT)
        screen.blit(gs_label, (GRID_LEFT + grid_px - gs_label.get_width(), GRID_TOP - 16))

        # --- main grid ---
        settled_glow = settled_flash / 30.0
        for r in range(grid_size):
            for c in range(grid_size):
                idx = r * grid_size + c
                x = GRID_LEFT + c * cell_size
                y = GRID_TOP + r * cell_size
                rect = pygame.Rect(x + 1, y + 1, cell_size - 2, cell_size - 2)

                if idx in flash_cells:
                    t = flash_cells[idx] / 10.0
                    col = lerp_color(CELL_ON if net.state[idx] else CELL_OFF, CELL_FLIP, t)
                elif net.state[idx] == 1:
                    col = lerp_color(CELL_ON, (255, 255, 255), settled_glow * 0.3)
                else:
                    col = CELL_OFF

                pygame.draw.rect(screen, col, rect, border_radius=3)

        # grid border
        pygame.draw.rect(screen, GRID_LINE,
                         (GRID_LEFT - 1, GRID_TOP - 1, grid_px + 2, grid_px + 2), 1,
                         border_radius=2)

        # --- right panel ---
        # buttons
        for btn in buttons.values():
            btn.hover = btn.hit((mx, my))
            btn.draw(screen, font, small_font)

        # --- stored memories thumbnails ---
        mem_y = btn_y + gap * 3 + 16
        mem_label = font.render(f"Memories ({len(net.patterns)}/{MAX_MEMORIES})", True, TEXT_COL)
        screen.blit(mem_label, (PANEL_LEFT, mem_y))
        mem_y += 24

        thumb_size = min(50, (PANEL_W - 10) // 3)
        for i, pat in enumerate(net.patterns):
            tx = PANEL_LEFT + (i % 3) * (thumb_size + 8)
            ty = mem_y + (i // 3) * (thumb_size + 8)
            ts = thumb_size // grid_size or 1
            for r in range(grid_size):
                for c in range(grid_size):
                    cx = tx + c * ts
                    cy_pos = ty + r * ts
                    col = THUMB_ON if pat[r * grid_size + c] else THUMB_OFF
                    pygame.draw.rect(screen, col, (cx, cy_pos, ts, ts))
            pygame.draw.rect(screen, ACCENT, (tx - 1, ty - 1,
                             grid_size * ts + 2, grid_size * ts + 2), 1)
            num = small_font.render(f"#{i+1}", True, DIM_TEXT)
            screen.blit(num, (tx, ty + grid_size * ts + 2))

        # --- presets ---
        preset_y = mem_y + (MAX_MEMORIES // 3 + 1) * (thumb_size + 8) + 30
        pr_label = font.render("Presets [1-4]", True, TEXT_COL)
        screen.blit(pr_label, (PANEL_LEFT, preset_y))
        preset_y += 22
        for i, name in enumerate(preset_keys[:4]):
            pr_text = small_font.render(f"[{i+1}] {name}", True, DIM_TEXT)
            screen.blit(pr_text, (PANEL_LEFT, preset_y + i * 18))

        # --- energy graph ---
        pygame.draw.rect(screen, PANEL_BG,
                         (GRID_LEFT - 4, ENERGY_TOP - 4,
                          grid_px + 8, ENERGY_H + 8),
                         border_radius=6)
        e_label = small_font.render("Energy", True, DIM_TEXT)
        screen.blit(e_label, (GRID_LEFT, ENERGY_TOP - 2))

        if len(energy_history) > 1:
            e_min = min(energy_history)
            e_max = max(energy_history)
            e_range = e_max - e_min if e_max != e_min else 1.0
            graph_x = GRID_LEFT + 4
            graph_w = grid_px
            graph_y = ENERGY_TOP + 14
            graph_h = ENERGY_H - 20

            points = []
            for i, e in enumerate(energy_history):
                px = graph_x + int(i / max(1, len(energy_history) - 1) * graph_w)
                py = graph_y + graph_h - int((e - e_min) / e_range * graph_h)
                points.append((px, py))

            if len(points) >= 2:
                pygame.draw.lines(screen, ENERGY_COL, False, points, 2)
                # glow on last point
                last = points[-1]
                col = ENERGY_LOW if auto_recalling or settled_flash else ENERGY_COL
                pygame.draw.circle(screen, col, last, 4)

            # energy value
            ev_text = small_font.render(f"E = {energy_history[-1]:.1f}", True, ENERGY_COL)
            screen.blit(ev_text, (graph_x + graph_w - ev_text.get_width(), ENERGY_TOP - 2))

        # --- noise level indicator ---
        nl = NOISE_LEVELS[noise_idx]
        ni_col = NOISE_COLS[noise_idx]
        ni_text = small_font.render(f"noise: {int(nl*100)}%", True, ni_col)
        screen.blit(ni_text, (PANEL_LEFT + PANEL_W - ni_text.get_width(), GRID_TOP - 16))

        # --- status message ---
        if message_timer > 0:
            alpha = min(1.0, message_timer / 20.0)
            msg_col = lerp_color(BG, TEXT_COL, alpha)
            msg_surf = font.render(message, True, msg_col)
            screen.blit(msg_surf, (GRID_LEFT, WIN_H - 22))

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
