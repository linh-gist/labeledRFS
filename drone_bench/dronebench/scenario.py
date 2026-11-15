"""Planar drone-like targets in a camera field of view, rendered as noisy images.

All units are pixels and frames. Noise sigma is 1, so blob amplitudes are SNRs
(peak amplitude / noise std).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

FWHM2SIG = 1.0 / 2.3548


def _rng_range(rng, v):
    """Scalar -> itself, (lo, hi) -> uniform sample."""
    if np.isscalar(v):
        return float(v)
    lo, hi = v
    return float(rng.uniform(lo, hi))


@dataclass
class ScenarioConfig:
    # camera / sequence
    width: int = 320
    height: int = 240
    n_frames: int = 80
    noise_sigma: float = 1.0           # keep 1.0 so amplitudes == SNR
    bg_level: float = 0.0
    bg_gradient: float = 2.0           # amplitude of a smooth sky/ground background ramp

    # targets (drones)
    n_targets_init: tuple | int = (3, 5)
    birth_rate: float = 0.03           # Poisson births per frame
    p_death: float = 0.004             # per-frame chance a drone disappears (lands / switches off)
    max_targets: int = 10
    spawn: str = "edge"                # 'edge' (fly in) or 'anywhere' (pop up)
    stay_in_view: bool = True          # soft boundary that steers drones back inside
    target_size_px: tuple = (1.0, 5.0) # blob FWHM in px
    snr: tuple | float = 3.0           # scalar or (lo, hi) per target
    snr_fluctuation: float = 0.15      # frame-to-frame relative brightness flicker (std)

    # drone behaviour
    mode_probs: dict = field(default_factory=lambda: {"hover": 0.3, "cruise": 0.25,
                                                      "accel": 0.25, "circle": 0.2})
    dwell_frames: tuple = (10, 30)
    v_max: float = 4.0                 # px/frame
    cruise_speed: tuple = (0.8, 3.0)
    accel: tuple = (0.08, 0.35)        # px/frame^2
    circle_radius: tuple = (8.0, 30.0)
    hover_jitter: float = 0.15         # px/frame velocity noise while hovering
    accel_noise: float = 0.05          # px/frame^2 process noise in all modes

    # clutter
    n_static_clutter: int = 60         # persistent blobs (ground features, trees, buildings)
    moving_clutter_frac: float = 0.0   # share of persistent clutter that drifts (birds, clouds)
    clutter_speed: float = 0.6         # px/frame
    transient_clutter_rate: float = 15.0  # Poisson blobs that appear for one frame (glints)
    clutter_size_px: tuple = (1.0, 25.0)  # FWHM
    clutter_snr: tuple = (0.5, 5.0)


MODES = ("hover", "cruise", "accel", "circle")


@dataclass
class Drone:
    id: int
    pos: np.ndarray
    vel: np.ndarray
    size: float
    snr: float
    mode: str = "cruise"
    dwell: int = 0
    acc: np.ndarray = field(default_factory=lambda: np.zeros(2))
    omega: float = 0.0
    hover_center: np.ndarray = field(default_factory=lambda: np.zeros(2))


@dataclass
class Sequence:
    cfg: ScenarioConfig
    frames: np.ndarray                 # (K, H, W) float32
    truth: list                        # per frame: dict(ids, pos(N,2), vel, mode, size, snr)
    clutter: list                      # per frame: (M,2) clutter centres (for display)

    @property
    def K(self):
        return len(self.truth)


class DroneSimulator:
    def __init__(self, cfg: ScenarioConfig, seed=0):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.next_id = 0
        self.drones: list[Drone] = []
        W, H = cfg.width, cfg.height
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        ang = self.rng.uniform(0, 2 * np.pi)
        self.bg = (cfg.bg_level + cfg.bg_gradient * (np.cos(ang) * xx / W + np.sin(ang) * yy / H)
                   ).astype(np.float32)
        self._init_clutter()

    # ---------------------------------------------------------------- clutter
    def _init_clutter(self):
        c, r = self.cfg, self.rng
        n = c.n_static_clutter
        self.cl_pos = np.column_stack([r.uniform(0, c.width, n), r.uniform(0, c.height, n)])
        self.cl_sig = np.array([_rng_range(r, c.clutter_size_px) for _ in range(n)]) * FWHM2SIG
        self.cl_amp = np.array([_rng_range(r, c.clutter_snr) for _ in range(n)])
        moving = r.random(n) < c.moving_clutter_frac
        ang = r.uniform(0, 2 * np.pi, n)
        self.cl_vel = np.column_stack([np.cos(ang), np.sin(ang)]) * c.clutter_speed * moving[:, None]

    def _step_clutter(self):
        c, r = self.cfg, self.rng
        moving = np.linalg.norm(self.cl_vel, axis=1) > 0
        self.cl_vel[moving] += r.normal(0, 0.05, (moving.sum(), 2))
        self.cl_pos += self.cl_vel
        self.cl_pos[:, 0] %= c.width
        self.cl_pos[:, 1] %= c.height
        nt = r.poisson(c.transient_clutter_rate)
        tp = np.column_stack([r.uniform(0, c.width, nt), r.uniform(0, c.height, nt)])
        ts = np.array([_rng_range(r, c.clutter_size_px) for _ in range(nt)]) * FWHM2SIG
        ta = np.array([_rng_range(r, c.clutter_snr) for _ in range(nt)])
        return (np.vstack([self.cl_pos, tp]), np.concatenate([self.cl_sig, ts]),
                np.concatenate([self.cl_amp, ta]))

    # ----------------------------------------------------------------- drones
    def _spawn(self):
        c, r = self.cfg, self.rng
        W, H = c.width, c.height
        speed = _rng_range(r, c.cruise_speed)
        if c.spawn == "edge":
            side = r.integers(4)
            pos = [np.array([0.0, r.uniform(0, H)]), np.array([W - 1.0, r.uniform(0, H)]),
                   np.array([r.uniform(0, W), 0.0]), np.array([r.uniform(0, W), H - 1.0])][side]
            to_c = np.array([W / 2, H / 2]) - pos
            ang = np.arctan2(to_c[1], to_c[0]) + r.normal(0, 0.5)
            mode = "cruise"
        else:
            pos = np.array([r.uniform(0.1 * W, 0.9 * W), r.uniform(0.1 * H, 0.9 * H)])
            ang = r.uniform(0, 2 * np.pi)
            mode = None
        d = Drone(id=self.next_id, pos=pos, vel=speed * np.array([np.cos(ang), np.sin(ang)]),
                  size=_rng_range(r, c.target_size_px), snr=_rng_range(r, c.snr))
        self.next_id += 1
        self._set_mode(d, mode)
        self.drones.append(d)

    def _set_mode(self, d: Drone, mode=None):
        c, r = self.cfg, self.rng
        if mode is None:
            names = list(c.mode_probs)
            p = np.array([c.mode_probs[k] for k in names], float)
            mode = names[r.choice(len(names), p=p / p.sum())]
        d.mode = mode
        d.dwell = int(r.integers(c.dwell_frames[0], c.dwell_frames[1] + 1))
        if mode == "hover":
            d.hover_center = d.pos + d.vel * 3.0      # decelerate into a hover point
        elif mode == "accel":
            # mostly along/against heading, sometimes sideways
            heading = np.arctan2(d.vel[1], d.vel[0]) if np.linalg.norm(d.vel) > 0.1 else r.uniform(0, 2 * np.pi)
            ang = heading + r.choice([0.0, np.pi, np.pi / 2, -np.pi / 2], p=[0.5, 0.2, 0.15, 0.15])
            d.acc = _rng_range(r, c.accel) * np.array([np.cos(ang), np.sin(ang)])
        elif mode == "circle":
            speed = max(np.linalg.norm(d.vel), _rng_range(r, c.cruise_speed))
            if np.linalg.norm(d.vel) < 0.1:
                a = r.uniform(0, 2 * np.pi)
                d.vel = np.array([np.cos(a), np.sin(a)])
            d.vel = d.vel / np.linalg.norm(d.vel) * speed
            d.omega = r.choice([-1, 1]) * speed / _rng_range(r, c.circle_radius)
        elif mode == "cruise":
            if np.linalg.norm(d.vel) < 0.3:
                a = r.uniform(0, 2 * np.pi)
                d.vel = _rng_range(r, c.cruise_speed) * np.array([np.cos(a), np.sin(a)])

    def _step_drone(self, d: Drone):
        c, r = self.cfg, self.rng
        if d.mode == "hover":
            # damped spring to the hover point + jitter
            d.vel = 0.75 * d.vel + 0.08 * (d.hover_center - d.pos) + r.normal(0, c.hover_jitter, 2)
        elif d.mode == "accel":
            d.vel = d.vel + d.acc
        elif d.mode == "circle":
            co, si = np.cos(d.omega), np.sin(d.omega)
            d.vel = np.array([co * d.vel[0] - si * d.vel[1], si * d.vel[0] + co * d.vel[1]])
        d.vel = d.vel + r.normal(0, c.accel_noise, 2)
        if c.stay_in_view:
            m = 0.12 * np.array([c.width, c.height])
            push = np.zeros(2)
            push[d.pos < m] = 0.25
            push[d.pos > np.array([c.width, c.height]) - m] = -0.25
            d.vel = d.vel + push
            if np.any(push != 0) and d.mode == "hover":
                d.hover_center = np.clip(d.hover_center, m, np.array([c.width, c.height]) - m)
        sp = np.linalg.norm(d.vel)
        if sp > c.v_max:
            d.vel *= c.v_max / sp
        d.pos = d.pos + d.vel
        d.dwell -= 1
        if d.dwell <= 0:
            self._set_mode(d)

    # -------------------------------------------------------------- rendering
    @staticmethod
    def render_blobs(img, pos, sig, amp):
        H, W = img.shape
        for (x, y), s, a in zip(pos, sig, amp):
            s = np.hypot(s, 0.5)               # minimum optical blur (PSF)
            rad = int(np.ceil(3.5 * s))
            x0, x1 = max(int(x) - rad, 0), min(int(x) + rad + 2, W)
            y0, y1 = max(int(y) - rad, 0), min(int(y) + rad + 2, H)
            if x0 >= x1 or y0 >= y1:
                continue
            gx = np.exp(-0.5 * ((np.arange(x0, x1) - x) / s) ** 2)
            gy = np.exp(-0.5 * ((np.arange(y0, y1) - y) / s) ** 2)
            img[y0:y1, x0:x1] += a * np.outer(gy, gx)
        return img

    def run(self) -> Sequence:
        c, r = self.cfg, self.rng
        for _ in range(int(_rng_range(r, c.n_targets_init) if not np.isscalar(c.n_targets_init)
                           else c.n_targets_init)):
            self._spawn()
        frames = np.empty((c.n_frames, c.height, c.width), np.float32)
        truth, clutter = [], []
        for k in range(c.n_frames):
            if k > 0:
                for d in self.drones:
                    self._step_drone(d)
                for _ in range(r.poisson(c.birth_rate)):
                    if len(self.drones) < c.max_targets:
                        self._spawn()
            # deaths: random, or left the image
            alive = []
            for d in self.drones:
                inside = (-2 <= d.pos[0] < c.width + 2) and (-2 <= d.pos[1] < c.height + 2)
                if inside and r.random() >= c.p_death:
                    alive.append(d)
            self.drones = alive

            img = self.bg + r.normal(0, c.noise_sigma, (c.height, c.width)).astype(np.float32)
            cpos, csig, camp = self._step_clutter()
            self.render_blobs(img, cpos, csig, camp * c.noise_sigma)
            vis = [d for d in self.drones if 0 <= d.pos[0] < c.width and 0 <= d.pos[1] < c.height]
            amps = [max(d.snr * (1 + c.snr_fluctuation * r.normal()), 0.0) * c.noise_sigma for d in vis]
            self.render_blobs(img, [d.pos for d in vis], [d.size * FWHM2SIG for d in vis], amps)
            frames[k] = img
            truth.append(dict(
                ids=np.array([d.id for d in vis], int),
                pos=np.array([d.pos for d in vis]).reshape(-1, 2),
                vel=np.array([d.vel for d in vis]).reshape(-1, 2),
                mode=[d.mode for d in vis],
                size=np.array([d.size for d in vis]),
                snr=np.array([d.snr for d in vis]),
            ))
            clutter.append(cpos)
        return Sequence(cfg=c, frames=frames, truth=truth, clutter=clutter)


def simulate(cfg: ScenarioConfig, seed=0) -> Sequence:
    return DroneSimulator(cfg, seed).run()
