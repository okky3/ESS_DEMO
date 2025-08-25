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


def grid_to_image(grid):
    # Map Hawk=0 -> red, Dove=1 -> blue
    L = grid.shape[0]
    img = np.zeros((L, L, 3), dtype=np.uint8)
    img[grid == H] = np.array([220, 20, 60], dtype=np.uint8)   # crimson-ish
    img[grid == D] = np.array([65, 105, 225], dtype=np.uint8)  # royal blue-ish
    return img


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Hawk–Dove Spatial Moran (Animated)", layout="wide")
st.title("タカ派、ハト派（ゲーム理論）")

with st.sidebar:
    st.header("パラメータ")
    L = st.slider("グリッドの大きさ L × L", 20, 200, 40, step=5)
    V = st.number_input("資源の価値(V)", value=1.0, step=0.1)
    C = st.number_input("戦争のコスト(C)", value=2.0, step=0.1)
    w = st.slider("選択強度 (w)", 0.0, 1.0, 0.9, step=0.05)

    neighborhood = "Moore"
    update_rule = st.radio("更新ルール", ["Death–Birth (DB)", "Birth–Death (BD)"], index=0)

    mapping_mode = st.radio("適応度の計算方法", ["Clip to zero", "Shift"], index=0,
                            help="Base mapping: f = 1 - w + w * payoff. 'Clip' clamps negatives to 0. 'Shift' adds a constant shift.")
    shift_amount = st.number_input("シフト量（Shiftの場合）", value=0.0, step=0.1)
    mu = st.number_input("突然変異率 μ", min_value=0.0, max_value=1.0, value=0.0, step=0.001, format="%.3f")
    replacement_rate_pct = st.slider("世代交代率（%）", 1, 100, 10)
    replacement_rate = replacement_rate_pct / 100

    init_hawk_frac = st.slider("初期タカ割合", 0.0, 1.0, 0.5, step=0.05)
    seed = st.number_input("乱数シード", value=42, step=1)

    fps = st.slider("アニメーションFPS", 1, 30, 8)
    max_steps = st.number_input("世代数（最大）", value=10_000, step=100)
    display_interval = st.number_input("N世代ごとに表示", value=10, min_value=1, step=1)

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
if "do_step" not in st.session_state:
    st.session_state["do_step"] = False

# Controls row
col1, col2, col3, col4, col5 = st.columns([1,1,1,1,2])
with col1:
    if st.button("Start ▶"):
        st.session_state.running = True
with col2:
    if st.button("Stop ⏸"):
        st.session_state.running = False
with col3:
    if st.button("Step Once ⏭"):
        st.session_state.running = False
        st.session_state["do_step"] = True
with col4:
    if st.button("Reset ♻"):
        rng = init_rng(seed)
        st.session_state["rng"] = rng
        grid = (rng.random((L, L)) > init_hawk_frac).astype(int)
        st.session_state["grid"] = grid
        st.session_state["step"] = 0
        st.session_state.running = False

with col5:
    st.markdown(f"**Step:** {st.session_state['step']}")

# 更新処理
rng = st.session_state["rng"]
grid = st.session_state["grid"]
if (st.session_state.running or st.session_state.get("do_step", False)) \
        and st.session_state["step"] < int(max_steps):
    num_updates = max(1, int(replacement_rate * L * L))
    # 世代交代率に基づいて複数回更新
    for _ in range(num_updates):
        P = compute_payoffs(grid, V, C, neighborhood)
        F = map_fitness(P, w, mapping_mode, shift_amount)
        if update_rule.startswith("Death"):
            grid = death_birth_update(grid, rng, F, neighborhood, mu)
        else:
            grid = birth_death_update(grid, rng, F, neighborhood, mu)
        st.session_state["step"] += 1
        if st.session_state["step"] >= int(max_steps):
            st.session_state.running = False
            break
    st.session_state["grid"] = grid
    st.session_state["do_step"] = False

# Render current grid at specified intervals
col_img, _ = st.columns([7, 3])
if st.session_state["step"] % int(display_interval) == 0:
    img = grid_to_image(st.session_state["grid"])  # RGB
    col_img.image(
        img,
        caption=f"L={L}, Step={st.session_state['step']} (タカ派=赤, ハト派=青)",
        use_column_width=True,
    )

# Auto-refresh to animate when running
if st.session_state.running:
    # Stop automatically once we reach max_steps
    if st.session_state["step"] >= int(max_steps):
        st.session_state.running = False
    else:
        # Control the frame rate and continue
        time.sleep(1.0 / max(1, fps))
        _safe_rerun()

