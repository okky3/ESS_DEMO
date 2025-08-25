import time
import numpy as np
import streamlit as st

# Streamlit 1.30+ uses st.rerun(); older versions use _safe_rerun().
# Provide a backwards/forwards-compatible wrapper.

def _safe_rerun():
    try:
        # New API
        st.rerun()
    except Exception:
        # Fallback for older Streamlit
        try:
            _safe_rerun()
        except Exception:
            pass  # As a last resort, do nothing to avoid crashing

# -----------------------------
# Helpers
# -----------------------------

H, D = 0, 1  # Hawk=0 (red), Dove=1 (blue)

@st.cache_resource
def init_rng(seed: int):
    return np.random.default_rng(seed)


def torus_neighbors(i, j, L, neighborhood="Moore"):
    if neighborhood == "Moore":
        coords = [
            (i-1, j-1), (i-1, j), (i-1, j+1),
            (i, j-1),             (i, j+1),
            (i+1, j-1), (i+1, j), (i+1, j+1)
        ]
    else:  # Von Neumann
        coords = [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]
    # wrap
    return [((r % L), (c % L)) for (r, c) in coords]


def payoff_pair(a, b, V, C):
    # Returns (payoff_a, payoff_b) for one interaction
    if a == H and b == H:
        return (V - C) / 2.0, (V - C) / 2.0
    if a == H and b == D:
        return V, 0.0
    if a == D and b == H:
        return 0.0, V
    # D vs D
    return V / 2.0, V / 2.0


def compute_payoffs(grid, V, C, neighborhood):
    L = grid.shape[0]
    P = np.zeros_like(grid, dtype=float)
    # Each unordered neighbor pair should be counted once; easiest is to iterate all cells
    # and add pair payoffs only for a subset of neighbors to avoid double-counting.
    for i in range(L):
        for j in range(L):
            s_ij = grid[i, j]
            for (ni, nj) in torus_neighbors(i, j, L, neighborhood):
                # Only accumulate when (ni, nj) is lexicographically greater to avoid double count
                if (ni > i) or (ni == i and nj > j):
                    s_n = grid[ni, nj]
                    pa, pb = payoff_pair(s_ij, s_n, V, C)
                    P[i, j] += pa
                    P[ni, nj] += pb
    return P


def map_fitness(payoffs, w, mapping_mode, shift_amount):
    # base mapping
    f = 1.0 - w + w * payoffs
    if mapping_mode == "Clip to zero":
        f = np.clip(f, 0.0, None)
    else:  # Shift
        f = f + shift_amount
        # Keep fitness non-negative to avoid pathological probabilities
        f = np.clip(f, 0.0, None)
    return f


def death_birth_update(grid, rng, fitness, neighborhood, mu):
    L = grid.shape[0]
    # Pick a random site to die
    di, dj = int(rng.integers(L)), int(rng.integers(L))
    neighs = torus_neighbors(di, dj, L, neighborhood)
    # Choose reproducer among neighbors proportional to fitness
    fit_vals = np.array([fitness[i, j] for (i, j) in neighs], dtype=float)
    if fit_vals.sum() > 0:
        probs = fit_vals / fit_vals.sum()
        k = rng.choice(len(neighs), p=probs)
    else:
        k = rng.integers(len(neighs))
    pi, pj = neighs[k]
    offspring = grid[pi, pj]
    # mutation
    if rng.random() < mu:
        offspring = 1 - offspring
    new_grid = grid.copy()
    new_grid[di, dj] = offspring
    return new_grid


def birth_death_update(grid, rng, fitness, neighborhood, mu):
    L = grid.shape[0]
    # Choose reproducer anywhere proportional to fitness
    flat_fit = fitness.ravel()
    if flat_fit.sum() > 0:
        probs = flat_fit / flat_fit.sum()
        idx = int(rng.choice(L * L, p=probs))
    else:
        idx = int(rng.integers(L * L))
    bi, bj = divmod(idx, L)
    # Pick a random neighbor to die
    neighs = torus_neighbors(bi, bj, L, neighborhood)
    di, dj = neighs[int(rng.integers(len(neighs)))]
    offspring = grid[bi, bj]
    if rng.random() < mu:
        offspring = 1 - offspring
    new_grid = grid.copy()
    new_grid[di, dj] = offspring
    return new_grid


# -------------------------------------------------------------
# シミュレーション本体
# -------------------------------------------------------------
def run_simulation(params):
    L = params["L"]
    offsets = get_neighbor_offsets(params["neighborhood"])
    state, rng = init_state(
        L,
        params["init_mode"],
        params["p0"],
        params["num_patches"],
        params["patch_radius"],
        params["patch_strategy"],
        params["seed"],
    )
    history = [state.copy()]
    metrics = []
    payoffs = accumulate_payoffs(state, params["V"], params["C"], offsets)
    if params["log_metrics"]:
        metrics.append((0, np.mean(state == 0), payoffs.mean()))
    replace_steps = int(L * L * params["replace_rate"] / 100)
    for t in range(1, params["generations"] + 1):
        for _ in range(replace_steps):
            fitness = fitness_from_payoff(
                payoffs, params["w"], params["fitness_mapping"], params["shift_amount"]
            )
            if params["update_rule"] == "BD":
                moran_BD_step(state, fitness, offsets, params["mu"], rng)
            else:
                moran_DB_step(state, fitness, offsets, params["mu"], rng)
            if params["m"] > 0:
                diffuse(state, params["m"], offsets, rng)
            payoffs = accumulate_payoffs(state, params["V"], params["C"], offsets)
        if params["log_metrics"]:
            metrics.append((t, np.mean(state == 0), payoffs.mean()))
        history.append(state.copy())
    return history, metrics

# -------------------------------------------------------------
# グリッドをRGB画像へ変換
# -------------------------------------------------------------
def grid_to_rgb(state):
    L = state.shape[0]
    img = np.zeros((L, L, 3), dtype=np.uint8)
    img[grid == H] = np.array([220, 20, 60], dtype=np.uint8)   # crimson-ish
    img[grid == D] = np.array([65, 105, 225], dtype=np.uint8)  # royal blue-ish
    return img


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Hawk–Dove Spatial Moran (Animated)", layout="wide")
st.title("Hawk–Dove Game — Spatial Moran Process (In‑page Animation)")

with st.sidebar:
    st.header("設定")
    L = st.number_input("L (格子サイズ)", min_value=10, max_value=200, value=40, step=1)
    neighborhood = st.selectbox("近傍", ["moore", "vonneumann"])
    update_rule = st.selectbox("更新ルール", ["DB", "BD"])
    V = st.number_input("資源価値 V", value=2.0)
    C = st.number_input("闘争コスト C", value=4.0)
    fitness_mapping = st.selectbox("適応度マッピング", ["clip0", "shift"])
    shift_amount = st.number_input("シフト量", value=0.0)
    w = st.number_input("選択強度 w", value=1.0)
    mu = st.number_input("突然変異率 μ", value=0.0)
    generations = st.number_input("世代数", min_value=1, value=200, step=1)
    replace_rate = st.slider(
        "Replacement rate per generation (%)",
        min_value=0.0,
        max_value=100.0,
        value=10.0,
        step=1.0,
        help="1世代で全個体のうち何％を置き換えるか"
    )
    m = st.number_input("拡散確率 m", value=0.0)
    init_mode = st.selectbox("初期状態", ["random", "patches"])
    p0 = st.number_input("初期ホーク比 p0", min_value=0.0, max_value=1.0, value=0.5)
    num_patches = st.number_input("パッチ数", min_value=1, value=3, step=1)
    patch_radius = st.number_input("パッチ半径", min_value=1, value=5, step=1)
    patch_strategy = st.selectbox("パッチ戦略", ["hawk", "dove", "mixed"])
    draw_skip = st.number_input("描画間隔 (k世代ごと)", min_value=1, value=2, step=1)
    frame_duration = st.number_input("フレーム時間 (ms)", min_value=20, value=80, step=10)
    seed = st.number_input("乱数シード", value=0, step=1)
    log_metrics = st.checkbox("メトリクスを記録", value=False)

config = {
    "L": L,
    "neighborhood": neighborhood,
    "update_rule": update_rule,
    "V": V,
    "C": C,
    "fitness_mapping": fitness_mapping,
    "shift_amount": shift_amount,
    "w": w,
    "mu": mu,
    "generations": generations,
    "replace_rate": replace_rate,
    "m": m,
    "init_mode": init_mode,
    "p0": p0,
    "num_patches": num_patches,
    "patch_radius": patch_radius,
    "patch_strategy": patch_strategy,
    "draw_skip": draw_skip,
    "frame_duration": frame_duration,
    "seed": seed,
    "log_metrics": log_metrics,
}

    mapping_mode = st.radio("Fitness mapping", ["Clip to zero", "Shift"], index=0,
                            help="Base mapping: f = 1 - w + w * payoff. 'Clip' clamps negatives to 0. 'Shift' adds a constant shift.")
    shift_amount = st.number_input("Shift amount (if 'Shift')", value=0.0, step=0.1)
    mu = st.number_input("Mutation rate μ", min_value=0.0, max_value=1.0, value=0.0, step=0.001, format="%.3f")

    init_hawk_frac = st.slider("Initial Hawks fraction", 0.0, 1.0, 0.5, step=0.05)
    seed = st.number_input("Random seed", value=42, step=1)

    fps = st.slider("Animation FPS", 1, 30, 8)
    max_steps = st.number_input("Max steps while running", value=10_000, step=100)
    run_for = st.number_input("Run generations (on Start)", value=200, step=10)
    display_interval = st.number_input("Display every N generations", value=1, min_value=1, step=1)

# Session state setup
if "rng" not in st.session_state or st.session_state.get("seed", None) != seed:
    st.session_state["rng"] = init_rng(seed)
    st.session_state["seed"] = seed

if "grid" not in st.session_state or st.session_state.get("L", None) != L:
    rng = st.session_state["rng"]
    grid = (rng.random((L, L)) > init_hawk_frac).astype(int)  # Dove=1 if > frac, Hawk=0 otherwise
    st.session_state["grid"] = grid
    st.session_state["L"] = L
    st.session_state["step"] = 0

if "running" not in st.session_state:
    st.session_state["running"] = False
if "target_end" not in st.session_state:
    st.session_state["target_end"] = None

# Controls row
col1, col2, col3, col4, col5 = st.columns([1,1,1,1,2])
with col1:
    if st.button("Start ▶"):
        st.session_state.running = True
        # Set the target generation to stop at
        current = st.session_state.get("step", 0)
        st.session_state["target_end"] = min(current + int(run_for), int(max_steps))
with col2:
    if st.button("Stop ⏸"):
        st.session_state.running = False
        st.session_state["target_end"] = None
with col3:
    if st.button("Step Once ⏭"):
        st.session_state.running = False
        st.session_state["target_end"] = None
        st.session_state["step"] += 1
        # perform one update below
with col4:
    if st.button("Reset ♻"):
        rng = init_rng(seed)
        st.session_state["rng"] = rng
        grid = (rng.random((L, L)) > init_hawk_frac).astype(int)
        st.session_state["grid"] = grid
        st.session_state["step"] = 0
        st.session_state.running = False
        st.session_state["target_end"] = None

with col5:
    st.markdown(f"**Step:** {st.session_state['step']}")

# Compute payoffs and fitness for current grid
rng = st.session_state["rng"]
grid = st.session_state["grid"]
P = compute_payoffs(grid, V, C, neighborhood)
F = map_fitness(P, w, mapping_mode, shift_amount)

# Perform one update if stepping or running
if st.session_state.running or st.session_state.get("last_clicked") == "Step Once ⏭":
    if update_rule.startswith("Death"):
        new_grid = death_birth_update(grid, rng, F, neighborhood, mu)
    else:
        new_grid = birth_death_update(grid, rng, F, neighborhood, mu)
    st.session_state["grid"] = new_grid
    st.session_state["step"] += 1

# Render current grid
img = grid_to_image(st.session_state["grid"])  # RGB
canvas = st.empty()
canvas.image(img, caption=f"L={L}, Step={st.session_state['step']} (Hawk=red, Dove=blue)", use_container_width=True)

# Auto-refresh to animate when running
if st.session_state.running:
    # Determine the stopping point
    if st.session_state.get("target_end") is None:
        st.session_state["target_end"] = min(st.session_state.get("step", 0) + int(run_for), int(max_steps))
    target_end = int(st.session_state["target_end"]) if st.session_state["target_end"] is not None else int(max_steps)

    # Stop automatically once we reach the target generation or max_steps
    if st.session_state["step"] >= min(int(max_steps), target_end):
        st.session_state.running = False
        st.session_state["target_end"] = None
    else:
        # Control the frame rate and continue
        time.sleep(1.0 / max(1, fps))
        _safe_rerun()
ß