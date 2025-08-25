# app.py
# 2次元ホーク・ダブゲームを空間モラン過程でシミュレートしGIFを生成するStreamlitアプリ
import streamlit as st
import numpy as np
from PIL import Image
import io
import json

# -------------------------------------------------------------
# 近傍オフセットを取得する
# -------------------------------------------------------------
def get_neighbor_offsets(neighborhood):
    if neighborhood == "vonneumann":
        return [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:  # "moore"
        return [(-1, -1), (-1, 0), (-1, 1),
                (0, -1),           (0, 1),
                (1, -1),  (1, 0),  (1, 1)]

# -------------------------------------------------------------
# 各セルの利得を近傍と1回ずつ対戦して加算する
# -------------------------------------------------------------
def accumulate_payoffs(state, V, C, offsets):
    L = state.shape[0]
    payoff_mat = np.array([
        [(V - C) / 2, V],
        [0, V / 2]
    ])
    payoffs = np.zeros((L, L), dtype=float)
    for dx, dy in offsets:
        neigh = np.roll(state, shift=dx, axis=0)
        neigh = np.roll(neigh, shift=dy, axis=1)
        payoffs += payoff_mat[state, neigh]
    return payoffs

# -------------------------------------------------------------
# 利得から適応度へマッピング
# -------------------------------------------------------------
def fitness_from_payoff(payoff, w, mapping, shift_amount):
    f = 1 - w + w * payoff
    if mapping == "clip0":
        f = np.clip(f, 0, None)
    else:  # "shift"
        f = f + shift_amount
    return f

# -------------------------------------------------------------
# 初期状態の生成
# -------------------------------------------------------------
def init_state(L, mode, p0, num_patches, patch_radius, patch_strategy, seed):
    rng = np.random.default_rng(seed)
    state = (rng.random((L, L)) >= p0).astype(np.int8)
    if mode == "patches":
        x = np.arange(L)
        y = np.arange(L)
        X, Y = np.meshgrid(x, y, indexing="ij")
        centers = rng.integers(0, L, size=(num_patches, 2))
        for cx, cy in centers:
            dx = np.minimum(np.abs(X - cx), L - np.abs(X - cx))
            dy = np.minimum(np.abs(Y - cy), L - np.abs(Y - cy))
            mask = dx**2 + dy**2 <= patch_radius**2
            if patch_strategy == "hawk":
                state[mask] = 0
            elif patch_strategy == "dove":
                state[mask] = 1
            else:  # mixed
                state[mask] = (rng.random(np.count_nonzero(mask)) >= p0).astype(np.int8)
    return state, rng

# -------------------------------------------------------------
# Death-Birth更新
# -------------------------------------------------------------
def moran_DB_step(state, fitness, offsets, mu, rng):
    L = state.shape[0]
    i = rng.integers(L)
    j = rng.integers(L)
    neigh_coords = [((i + dx) % L, (j + dy) % L) for dx, dy in offsets]
    neigh_fit = np.array([fitness[x, y] for x, y in neigh_coords])
    total = neigh_fit.sum()
    if total == 0:
        probs = np.ones(len(neigh_coords)) / len(neigh_coords)
    else:
        probs = neigh_fit / total
    idx = rng.choice(len(neigh_coords), p=probs)
    px, py = neigh_coords[idx]
    s = state[px, py]
    if rng.random() < mu:
        s = 1 - s
    state[i, j] = s
    return state

# -------------------------------------------------------------
# Birth-Death更新
# -------------------------------------------------------------
def moran_BD_step(state, fitness, offsets, mu, rng):
    L = state.shape[0]
    total = fitness.sum()
    if total == 0:
        idx = rng.integers(L * L)
    else:
        flat = (fitness / total).ravel()
        idx = rng.choice(L * L, p=flat)
    i, j = divmod(idx, L)
    dx, dy = offsets[rng.integers(len(offsets))]
    ni = (i + dx) % L
    nj = (j + dy) % L
    s = state[i, j]
    if rng.random() < mu:
        s = 1 - s
    state[ni, nj] = s
    return state

# -------------------------------------------------------------
# 拡散処理（近傍と入れ替え）
# -------------------------------------------------------------
def diffuse(state, m, offsets, rng):
    L = state.shape[0]
    cells = np.argwhere(rng.random((L, L)) < m)
    for i, j in cells:
        dx, dy = offsets[rng.integers(len(offsets))]
        ni = (i + dx) % L
        nj = (j + dy) % L
        state[i, j], state[ni, nj] = state[ni, nj], state[i, j]

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
    img[state == 0] = np.array([255, 0, 0], dtype=np.uint8)  # ホーク: 赤
    img[state == 1] = np.array([0, 0, 255], dtype=np.uint8)  # ダブ: 青
    return img

# -------------------------------------------------------------
# 履歴からGIFを作成
# -------------------------------------------------------------
def make_gif_from_history(history, duration, draw_skip):
    frames = [Image.fromarray(grid_to_rgb(g)) for g in history[::draw_skip]]
    buf = io.BytesIO()
    frames[0].save(
        buf,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
    )
    return buf.getvalue()

# -------------------------------------------------------------
# メトリクスCSVを生成
# -------------------------------------------------------------
def download_metrics_csv(metrics):
    buf = io.StringIO()
    buf.write("t,hawk_ratio,avg_payoff\n")
    for t, h, p in metrics:
        buf.write(f"{t},{h},{p}\n")
    return buf.getvalue().encode("utf-8")

# -------------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------------
st.title("ホーク・ダブ空間モランゲーム")

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

if st.button("Run simulation"):
    history, metrics = run_simulation(config)
    gif_bytes = make_gif_from_history(history, frame_duration, draw_skip)
    st.image(gif_bytes, caption="進化の様子", output_format="GIF")
    st.download_button(
        "Download GIF",
        data=gif_bytes,
        file_name="hawk_dove.gif",
        mime="image/gif",
    )
    st.image(grid_to_rgb(history[-1]), caption="最終グリッド")
    st.json(config)
    if log_metrics:
        csv_bytes = download_metrics_csv(metrics)
        st.download_button(
            "メトリクスCSVをダウンロード",
            data=csv_bytes,
            file_name="metrics.csv",
            mime="text/csv",
        )
