# Hawk–Dove Spatial Moran Simulator with Simple UI (Streamlit)
# -----------------------------------------------------------
# Requirements satisfied (from user spec):
#  - Update rule: DB default, toggle BD
#  - Neighborhood: Moore default, toggle Von Neumann
#  - Mutation rate: default 0.0, changeable
#  - Boundary: torus (fixed)
#  - Fitness mapping: default 0-clipping; toggle to shift
#  - Initial condition: default random; toggle to patches
#  - Initial Hawk ratio: default 0.5, changeable
#  - Interaction: fixed "each cell vs all neighbors exactly once"
#  - Generations: default 200, changeable
#  - Logging: optional (CSV)
#  - Produce an animation preview and downloadable GIF
#
# To run locally:
#   pip install streamlit numpy matplotlib pillow
#   streamlit run app.py

import io
import csv
import time
from dataclasses import dataclass, asdict
from typing import Literal, Optional, Dict, Any, List, Tuple

import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

# -----------------------------
# Payoff Matrix & Config
# -----------------------------

@dataclass
class PayoffMatrix:
    # strategies index: 0=Hawk, 1=Dove
    payoffs: np.ndarray  # shape (2,2)

    @staticmethod
    def hawk_dove(V: float, C: float) -> "PayoffMatrix":
        return PayoffMatrix(
            payoffs=np.array([
                [(V - C) / 2.0, V],  # H vs H, H vs D
                [0.0, V / 2.0]       # D vs H, D vs D
            ], dtype=float)
        )

@dataclass
class SimConfig:
    # Grid & neighborhood
    L: int = 40
    neighborhood: Literal["moore", "vonneumann"] = "moore"  # default Moore
    # Update rule
    update_rule: Literal["DB", "BD"] = "DB"                  # default DB
    # Game parameters
    V: float = 2.0
    C: float = 4.0
    # Fitness mapping
    fitness_mapping: Literal["clip0", "shift"] = "clip0"    # default clip0
    shift_amount: float = 0.0                                   # used when mapping == "shift"
    selection_intensity: float = 1.0                            # f = 1 - w + w*payoff
    # Evolution details
    mutation_rate: float = 0.0                                   # default 0
    generations: int = 200                                       # default 200
    rng_seed: Optional[int] = 123
    # Initial condition
    init_mode: Literal["random", "patches"] = "random"         # default random
    initial_hawk_ratio: float = 0.5                              # default 0.5
    # Patch params (when init_mode="patches")
    num_patches: int = 3
    patch_radius: int = 5
    patch_strategy: Literal["hawk", "dove", "mixed"] = "hawk"
    # Diffusion (optional; default off)
    diffusion_rate: float = 0.0
    # Visualization / export
    frame_interval_ms: int = 80
    draw_skip: int = 2
    # Logging (optional)
    log_metrics: bool = False

# -----------------------------
# Helpers
# -----------------------------

def get_neighbor_offsets(neighborhood: str) -> np.ndarray:
    if neighborhood == "moore":
        return np.array([(dy, dx) for dy in (-1,0,1) for dx in (-1,0,1) if not (dy==0 and dx==0)], dtype=int)
    elif neighborhood == "vonneumann":
        return np.array([(-1,0),(1,0),(0,-1),(0,1)], dtype=int)
    else:
        raise ValueError("Unknown neighborhood")


def accumulate_payoffs(state: np.ndarray, P: PayoffMatrix, offsets: np.ndarray) -> np.ndarray:
    """Each cell plays once with all neighbors (fixed)."""
    L = state.shape[0]
    payoff = np.zeros((L, L), dtype=float)
    y = np.arange(L)[:, None]
    x = np.arange(L)[None, :]
    for (dy, dx) in offsets:
        ny = (y + dy) % L
        nx = (x + dx) % L
        a = state
        b = state[ny, nx]
        payoff += P.payoffs[a, b]
    return payoff


def fitness_from_payoff(payoff: np.ndarray, w: float, mapping: str, shift_amount: float) -> np.ndarray:
    f = 1.0 - w + w * payoff
    if mapping == "clip0":
        return np.clip(f, 0.0, None)
    elif mapping == "shift":
        return f + shift_amount
    else:
        raise ValueError("Unknown fitness mapping")


def init_state(cfg: SimConfig, rng: np.random.Generator) -> np.ndarray:
    L = cfg.L
    if cfg.init_mode == "random":
        hawk = rng.random((L, L)) < cfg.initial_hawk_ratio
        return np.where(hawk, 0, 1).astype(np.int8)
    # patches: start all Dove, stamp patches
    state = np.ones((L, L), dtype=np.int8)
    centers = rng.integers(low=0, high=L, size=(cfg.num_patches, 2))
    for (cy, cx) in centers:
        yy, xx = np.ogrid[:L, :L]
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= cfg.patch_radius ** 2
        if cfg.patch_strategy == "hawk":
            state[mask] = 0
        elif cfg.patch_strategy == "dove":
            state[mask] = 1
        else:  # mixed
            rnd = rng.random(mask.sum()) < cfg.initial_hawk_ratio
            block = np.where(rnd, 0, 1).astype(np.int8)
            state[mask] = block
    return state


def moran_DB_step(state: np.ndarray, fitness: np.ndarray, offsets: np.ndarray, rng: np.random.Generator, mu: float) -> np.ndarray:
    L = state.shape[0]
    y = rng.integers(0, L)
    x = rng.integers(0, L)
    ys = (y + offsets[:, 0]) % L
    xs = (x + offsets[:, 1]) % L
    neigh_fit = fitness[ys, xs].astype(float)
    if np.all(neigh_fit <= 0):
        probs = np.ones_like(neigh_fit) / len(neigh_fit)
    else:
        neigh_fit = np.clip(neigh_fit, 0.0, None)
        s = neigh_fit.sum()
        probs = neigh_fit / s if s > 0 else np.ones_like(neigh_fit) / len(neigh_fit)
    idx = rng.choice(len(ys), p=probs)
    new_strategy = state[ys[idx], xs[idx]]
    if rng.random() < mu:
        new_strategy = 1 - new_strategy
    new_state = state.copy()
    new_state[y, x] = new_strategy
    return new_state


def moran_BD_step(state: np.ndarray, fitness: np.ndarray, offsets: np.ndarray, rng: np.random.Generator, mu: float) -> np.ndarray:
    L = state.shape[0]
    fit = np.clip(fitness.astype(float), 0.0, None)
    S = fit.sum()
    if S <= 0:
        py = rng.integers(0, L); px = rng.integers(0, L)
    else:
        flat = rng.choice(L * L, p=(fit / S).ravel())
        py, px = divmod(flat, L)
    dy, dx = offsets[rng.integers(0, len(offsets))]
    ty, tx = (py + dy) % L, (px + dx) % L
    new_strategy = state[py, px]
    if rng.random() < mu:
        new_strategy = 1 - new_strategy
    new_state = state.copy()
    new_state[ty, tx] = new_strategy
    return new_state


def diffuse(state: np.ndarray, offsets: np.ndarray, m: float, rng: np.random.Generator) -> np.ndarray:
    if m <= 0:
        return state
    L = state.shape[0]
    new_state = state.copy()
    order = [(i, j) for i in range(L) for j in range(L)]
    rng.shuffle(order)
    for (i, j) in order:
        if rng.random() < m:
            k = rng.integers(0, len(offsets))
            di, dj = offsets[k]
            ni, nj = (i + di) % L, (j + dj) % L
            new_state[i, j], new_state[ni, nj] = new_state[ni, nj], new_state[i, j]
    return new_state


def run_simulation(cfg: SimConfig) -> Dict[str, Any]:
    rng = np.random.default_rng(cfg.rng_seed)
    offsets = get_neighbor_offsets(cfg.neighborhood)
    P = PayoffMatrix.hawk_dove(cfg.V, cfg.C)

    state = init_state(cfg, rng)
    history: List[np.ndarray] = []
    metrics = {"hawk_ratio": [], "avg_payoff": []}

    for t in range(cfg.generations):
        payoff = accumulate_payoffs(state, P, offsets)
        fitness = fitness_from_payoff(payoff, cfg.selection_intensity, cfg.fitness_mapping, cfg.shift_amount)

        if cfg.update_rule == "DB":
            state = moran_DB_step(state, fitness, offsets, rng, cfg.mutation_rate)
        else:
            state = moran_BD_step(state, fitness, offsets, rng, cfg.mutation_rate)

        state = diffuse(state, offsets, cfg.diffusion_rate, rng)

        history.append(state.copy())
        if cfg.log_metrics:
            metrics["hawk_ratio"].append(float(np.mean(state == 0)))
            metrics["avg_payoff"].append(float(np.mean(payoff)))

    return {"history": history, "metrics": metrics, "config": asdict(cfg)}


def grid_to_rgb(grid: np.ndarray) -> np.ndarray:
    L = grid.shape[0]
    rgb = np.zeros((L, L, 3), dtype=float)
    rgb[grid == 0] = (0.85, 0.1, 0.1)  # Hawk: red-ish
    rgb[grid == 1] = (0.1, 0.1, 0.85)  # Dove: blue-ish
    return (rgb * 255).astype(np.uint8)


def make_gif_from_history(history: List[np.ndarray], draw_skip: int, duration_ms: int) -> bytes:
    """Create a GIF from history using Pillow only (no imageio)."""
    frames: List[Image.Image] = []
    steps = max(1, len(history) // max(1, draw_skip))
    for i in range(steps):
        arr = grid_to_rgb(history[i * draw_skip])
        frames.append(Image.fromarray(arr))
    bio = io.BytesIO()
    if len(frames) == 1:
        frames[0].save(bio, format="GIF")
    else:
        frames[0].save(bio, format="GIF", save_all=True, append_images=frames[1:], loop=0, duration=duration_ms)
    return bio.getvalue()


def download_metrics_csv(metrics: Dict[str, list]) -> bytes:
    bio = io.StringIO()
    writer = csv.writer(bio)
    writer.writerow(["t", "hawk_ratio", "avg_payoff"])
    for t, (p, ap) in enumerate(zip(metrics.get("hawk_ratio", []), metrics.get("avg_payoff", []))):
        writer.writerow([t, p, ap])
    return bio.getvalue().encode("utf-8")

# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(page_title="Hawk–Dove Moran Simulator", layout="wide")
st.title("Hawk–Dove Spatial Moran Simulator (Moran DB/BD)")

with st.sidebar:
    st.header("Parameters")
    L = st.slider("Grid size L", 20, 100, 40, step=2)
    neighborhood = st.selectbox("Neighborhood", ["moore", "vonneumann"], index=0)
    update_rule = st.selectbox("Update rule", ["DB", "BD"], index=0)

    st.markdown("**Game (V,C)**")
    V = st.number_input("Resource value V", min_value=0.0, value=2.0, step=0.5)
    C = st.number_input("Conflict cost C", min_value=0.0, value=4.0, step=0.5)

    st.markdown("**Fitness mapping**")
    mapping = st.selectbox("Mapping", ["clip0", "shift"], index=0)
    shift_amount = st.number_input("Shift amount (if shift)", value=0.0, step=0.1)
    w = st.slider("Selection intensity w", 0.0, 1.0, 1.0, step=0.05)

    st.markdown("**Evolution**")
    mu = st.number_input("Mutation rate μ", min_value=0.0, max_value=1.0, value=0.0, step=0.005)
    gens = st.number_input("Generations", min_value=1, max_value=5000, value=200, step=50)
    diffusion = st.slider("Diffusion rate m", 0.0, 0.5, 0.0, step=0.01)

    st.markdown("**Initialization**")
    init_mode = st.selectbox("Init mode", ["random", "patches"], index=0)
    p0 = st.slider("Initial Hawk ratio p0", 0.0, 1.0, 0.5, step=0.05)
    num_patches = st.slider("# patches", 1, 10, 3)
    patch_radius = st.slider("Patch radius", 2, 15, 5)
    patch_strategy = st.selectbox("Patch strategy", ["hawk", "dove", "mixed"], index=0)

    st.markdown("**Visualization**")
    draw_skip = st.slider("Draw every k frames", 1, 10, 2)
    frame_interval = st.slider("Frame duration (ms)", 20, 200, 80, step=5)

    st.markdown("**Misc**")
    seed = st.number_input("Random seed", value=123, step=1)
    log_metrics = st.checkbox("Log metrics to CSV (hawk ratio & avg payoff)", value=False)

    run_btn = st.button("Run simulation", type="primary")

# Assemble config
cfg = SimConfig(
    L=L,
    neighborhood=neighborhood,
    update_rule=update_rule,
    V=V,
    C=C,
    fitness_mapping=mapping,
    shift_amount=shift_amount,
    selection_intensity=w,
    mutation_rate=mu,
    generations=int(gens),
    rng_seed=int(seed),
    init_mode=init_mode,
    initial_hawk_ratio=p0,
    num_patches=int(num_patches),
    patch_radius=int(patch_radius),
    patch_strategy=patch_strategy,
    diffusion_rate=diffusion,
    frame_interval_ms=int(frame_interval),
    draw_skip=int(draw_skip),
    log_metrics=log_metrics,
)

st.caption("Boundary is fixed to torus; each generation computes payoffs vs all neighbors exactly once.")

# Preview panel
col1, col2 = st.columns([3, 2], gap="large")

if run_btn:
    with st.spinner("Simulating..."):
        result = run_simulation(cfg)

    history = result["history"]
    frames = max(1, len(history) // max(1, cfg.draw_skip))

    with col1:
        st.subheader("Animation preview")
        # Build GIF bytes and preview
        gif_bytes = make_gif_from_history(history, cfg.draw_skip, cfg.frame_interval_ms)
        st.image(gif_bytes)
        st.download_button("Download GIF", data=gif_bytes, file_name="hawk_dove_moran.gif", mime="image/gif")

    with col2:
        st.subheader("Final state & Config")
        final_rgb = grid_to_rgb(history[-1])
        st.image(final_rgb, caption="Final grid (H:red, D:blue)")
        st.json(cfg.__dict__)

        if cfg.log_metrics:
            csv_bytes = download_metrics_csv(result["metrics"])
            st.download_button("Download CSV metrics", data=csv_bytes, file_name="metrics.csv", mime="text/csv")

else:
    st.info("Set parameters on the left and click **Run simulation**. Default settings meet the class spec.")
