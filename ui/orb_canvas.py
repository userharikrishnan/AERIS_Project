"""
AERIS Orb Canvas
================
Animated ring + particle character drawn on a tkinter Canvas.
Runs its own 30 fps animation loop via canvas.after().

States: idle, listening, thinking, speaking, greeting, offline
"""
import tkinter as tk
import math
import random


# ── Per-state visual config ───────────────────────────────────────
_STATE_CFG = {
    "idle": {
        "core":       "#00cfff",
        "mid":        "#003399",
        "outer":      "#020820",
        "ring":       "#0077cc",
        "particle":   "#0099ee",
        "ring_speed": 1.0,
        "pulse_amp":  3,
    },
    "listening": {
        "core":       "#66eeff",
        "mid":        "#0055cc",
        "outer":      "#010a20",
        "ring":       "#00cfff",
        "particle":   "#00cfff",
        "ring_speed": 2.5,
        "pulse_amp":  6,
    },
    "thinking": {
        "core":       "#ffee00",
        "mid":        "#884400",
        "outer":      "#1a0500",
        "ring":       "#ffaa00",
        "particle":   "#ffcc44",
        "ring_speed": 4.5,
        "pulse_amp":  8,
    },
    "speaking": {
        "core":       "#00ffaa",
        "mid":        "#006633",
        "outer":      "#001209",
        "ring":       "#00dd99",
        "particle":   "#00ffaa",
        "ring_speed": 2.0,
        "pulse_amp":  5,
    },
    "greeting": {
        "core":       "#ffffff",
        "mid":        "#5588ff",
        "outer":      "#000820",
        "ring":       "#aaddff",
        "particle":   "#ffffff",
        "ring_speed": 6.0,
        "pulse_amp":  12,
    },
    "offline": {
        "core":       "#660011",
        "mid":        "#220008",
        "outer":      "#100004",
        "ring":       "#440011",
        "particle":   "#660022",
        "ring_speed": 0.3,
        "pulse_amp":  1,
    },
}


def _hex_blend(c1: str, c2: str, t: float) -> str:
    """Linearly blend two hex colours, t in [0,1] (0=c1, 1=c2)."""
    r1, g1, b1 = int(c1[1:3], 16), int(c1[3:5], 16), int(c1[5:7], 16)
    r2, g2, b2 = int(c2[1:3], 16), int(c2[3:5], 16), int(c2[5:7], 16)
    r = int(r1 + (r2 - r1) * t)
    g = int(g1 + (g2 - g1) * t)
    b = int(b1 + (b2 - b1) * t)
    return f"#{r:02x}{g:02x}{b:02x}"


class OrbCanvas(tk.Canvas):
    """Self-animating AERIS orb character."""

    ORB_R     = 52      # core orb radius (px)
    RINGS     = [80, 100, 122, 142]   # ring radii
    RING_W    = [2.5, 2, 1.5, 1]      # ring widths
    ARC_SEGS  = 7       # dashes around each ring
    FPS       = 30

    def __init__(self, parent, size: int = 320, **kw):
        super().__init__(
            parent,
            width=size, height=size,
            bg="#04060f",
            highlightthickness=0,
            **kw,
        )
        self.size = size
        self.cx   = size // 2
        self.cy   = size // 2

        self._state     = "idle"
        self._speed     = 1.0
        self._tick      = 0
        self._ring_a    = [0.0, 0.0, 0.0, 0.0]   # current angle per ring
        self._ring_dir  = [1, -1, 1, -1]           # rotation direction per ring
        self._particles = self._make_particles()
        self._running   = True
        self._after_id  = None

        self._schedule()

    # ── Public API ────────────────────────────────────────────────

    def set_state(self, state: str):
        self._state = state if state in _STATE_CFG else "idle"
        cfg = _STATE_CFG[self._state]
        self._speed = cfg["ring_speed"]

    def stop(self):
        self._running = False
        if self._after_id:
            self.after_cancel(self._after_id)

    # ── Particle init ─────────────────────────────────────────────

    def _make_particles(self):
        particles = []
        rings = [
            {"n": 11, "r": 95,  "spd":  0.30, "sz": 2.8},
            {"n": 16, "r": 118, "spd": -0.20, "sz": 2.0},
            {"n":  7, "r": 142, "spd":  0.14, "sz": 3.5},
            {"n": 20, "r": 160, "spd": -0.09, "sz": 1.3},
        ]
        for ring in rings:
            for i in range(ring["n"]):
                particles.append({
                    "angle": (360 / ring["n"]) * i + random.uniform(-5, 5),
                    "r":     ring["r"] + random.uniform(-8, 8),
                    "spd":   ring["spd"] * random.uniform(0.75, 1.25),
                    "sz":    ring["sz"]  * random.uniform(0.6, 1.4),
                    "phase": random.uniform(0, math.pi * 2),  # for alpha pulse
                })
        return particles

    # ── Animation loop ────────────────────────────────────────────

    def _schedule(self):
        if self._running:
            self._after_id = self.after(1000 // self.FPS, self._frame)

    def _frame(self):
        self._tick += 1
        self.delete("all")

        cfg = _STATE_CFG[self._state]
        pulse = math.sin(self._tick * 0.12) * cfg["pulse_amp"]

        self._draw_outer_glow(cfg, pulse)
        self._draw_rings(cfg, pulse)
        self._draw_particles(cfg)
        self._draw_orb(cfg, pulse)

        self._schedule()

    # ── Drawing primitives ────────────────────────────────────────

    def _draw_outer_glow(self, cfg: dict, pulse: float):
        """Concentric dim halos around the orb for glow illusion."""
        for i in range(6, 0, -1):
            r = self.ORB_R + i * 12 + pulse * 0.5
            shade = _hex_blend(cfg["outer"], cfg["ring"], i / 7)
            self.create_oval(
                self.cx - r, self.cy - r,
                self.cx + r, self.cy + r,
                fill=shade, outline="",
            )

    def _draw_rings(self, cfg: dict, pulse: float):
        """Rotating dashed rings at various radii."""
        for idx, (base_r, w) in enumerate(zip(self.RINGS, self.RING_W)):
            r = base_r + pulse * 0.35
            spd = self._speed * self._ring_dir[idx]
            self._ring_a[idx] = (self._ring_a[idx] + spd * 0.5) % 360

            # Draw ARC_SEGS visible arcs evenly spaced
            gap    = 360 / self.ARC_SEGS
            extent = gap * 0.55

            # Vary ring brightness by depth
            bright_t = 1.0 - idx * 0.15
            color = _hex_blend(cfg["mid"], cfg["ring"], bright_t)

            for s in range(self.ARC_SEGS):
                start = (self._ring_a[idx] + s * gap) % 360
                self.create_arc(
                    self.cx - r, self.cy - r,
                    self.cx + r, self.cy + r,
                    start=start, extent=extent,
                    outline=color, style=tk.ARC, width=w,
                )

    def _draw_particles(self, cfg: dict):
        """Small orbiting glowing dots."""
        for p in self._particles:
            p["angle"] = (p["angle"] + p["spd"] * self._speed) % 360
            rad = math.radians(p["angle"])
            px  = self.cx + math.cos(rad) * p["r"]
            py  = self.cy + math.sin(rad) * p["r"]
            sz  = p["sz"]
            # pulse alpha via phase
            alpha_t = 0.5 + 0.5 * math.sin(self._tick * 0.07 + p["phase"])
            color = _hex_blend(cfg["mid"], cfg["particle"], alpha_t)
            self.create_oval(px - sz, py - sz, px + sz, py + sz, fill=color, outline="")

    def _draw_orb(self, cfg: dict, pulse: float):
        """Central glowing orb with gradient layers."""
        r = self.ORB_R + pulse * 0.4
        steps = 10
        for i in range(steps, 0, -1):
            t    = i / steps
            cr   = int(r * t)
            # blend from outer → mid → core
            if t < 0.5:
                color = _hex_blend(cfg["outer"], cfg["mid"], t * 2)
            else:
                color = _hex_blend(cfg["mid"], cfg["core"], (t - 0.5) * 2)
            self.create_oval(
                self.cx - cr, self.cy - cr,
                self.cx + cr, self.cy + cr,
                fill=color, outline="",
            )

        # bright core highlight
        hc = int(r * 0.22)
        ho = int(r * 0.28)
        self.create_oval(
            self.cx - hc + ho // 2, self.cy - hc - ho // 2,
            self.cx + hc + ho // 2, self.cy + hc - ho // 2,
            fill="#ddf8ff", outline="",
        )

        # centre white dot
        cd = 5
        self.create_oval(
            self.cx - cd, self.cy - cd,
            self.cx + cd, self.cy + cd,
            fill="#ffffff", outline="",
        )
