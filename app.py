import io

import numpy as np
from PIL import Image
import streamlit as st

# ========================= 基本的なヘルパー関数 =========================

def get_neighbor_offsets(name: str):
    """近傍タイプからオフセット一覧を取得する"""
    if name == "moore":
        offsets = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if not (dx == 0 and dy == 0)]
    else:  # von_neumann
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    return np.array(offsets, dtype=int)


def accumulate_payoffs(grid: np.ndarray, offsets: np.ndarray, V: float, C: float):
    """全セルの利得を近傍対戦で合計"""
    payoffs = np.zeros_like(grid, dtype=float)
    for dx, dy in offsets:
        neigh = np.roll(np.roll(grid, dx, axis=0), dy, axis=1)
        a = grid
        b = neigh
        inter = np.zeros_like(grid, dtype=float)
        hh = (a == 0) & (b == 0)
        hd = (a == 0) & (b == 1)
        dh = (a == 1) & (b == 0)
        dd = (a == 1) & (b == 1)
        inter[hh] = (V - C) / 2
        inter[hd] = V
        inter[dh] = 0
        inter[dd] = V / 2
        payoffs += inter
    return payoffs


def fitness_from_payoff(payoff: np.ndarray, w: float, mapping: str, shift_amount: float):
    """利得から適応度へ変換"""
    f = 1 - w + w * payoff
    if mapping == "clip0":
        f = np.clip(f, 0, None)
    else:  # shift
        f = f + shift_amount
    return f


def init_state(L: int, mode: str, p0: float, num_patches: int, patch_radius: int,
               patch_strategy: str, seed: int):
    """初期状態を生成"""
    rng = np.random.default_rng(seed)
    grid = np.where(rng.random((L, L)) < p0, 0, 1)
    if mode == "patches":
        for _ in range(num_patches):
            cx, cy = rng.integers(L, size=2)
            rr = patch_radius
            for dx in range(-rr, rr + 1):
                for dy in range(-rr, rr + 1):
                    if dx * dx + dy * dy <= rr * rr:
                        x = (cx + dx) % L
                        y = (cy + dy) % L
                        if patch_strategy == "hawk":
                            grid[x, y] = 0
                        elif patch_strategy == "dove":
                            grid[x, y] = 1
                        else:  # mixed
                            grid[x, y] = 0 if rng.random() < p0 else 1
    return grid


def moran_DB_step(grid: np.ndarray, fitness: np.ndarray, offsets: np.ndarray, mu: float, rng):
    """Death-Birth 更新"""
    L = grid.shape[0]
    x, y = rng.integers(L, size=2)
    neigh_coords = [( (x + dx) % L, (y + dy) % L) for dx, dy in offsets]
    neigh_fit = np.array([fitness[nx, ny] for nx, ny in neigh_coords])
    total = neigh_fit.sum()
    if total <= 0:
        idx = rng.integers(len(neigh_coords))
    else:
        idx = rng.choice(len(neigh_coords), p=neigh_fit / total)
    px, py = neigh_coords[idx]
    strategy = grid[px, py]
    if rng.random() < mu:
        strategy = 1 - strategy
    grid[x, y] = strategy


def moran_BD_step(grid: np.ndarray, fitness: np.ndarray, offsets: np.ndarray, mu: float, rng):
    """Birth-Death 更新"""
    L = grid.shape[0]
    flat_fit = fitness.ravel()
    total = flat_fit.sum()
    if total <= 0:
        idx = rng.integers(flat_fit.size)
    else:
        idx = rng.choice(flat_fit.size, p=flat_fit / total)
    x, y = divmod(idx, L)
    dx, dy = offsets[rng.integers(len(offsets))]
    nx, ny = (x + dx) % L, (y + dy) % L
    strategy = grid[x, y]
    if rng.random() < mu:
        strategy = 1 - strategy
    grid[nx, ny] = strategy


def diffuse(grid: np.ndarray, m: float, offsets: np.ndarray, rng):
    """拡散ステップ：確率mで近傍と交換"""
    if m <= 0:
        return grid
    L = grid.shape[0]
    for x in range(L):
        for y in range(L):
            if rng.random() < m:
                dx, dy = offsets[rng.integers(len(offsets))]
                nx, ny = (x + dx) % L, (y + dy) % L
                grid[x, y], grid[nx, ny] = grid[nx, ny], grid[x, y]
    return grid


def grid_to_rgb(grid: np.ndarray):
    """戦略グリッドをRGB画像へ"""
    rgb = np.zeros(grid.shape + (3,), dtype=np.uint8)
    rgb[grid == 0] = [255, 0, 0]
    rgb[grid == 1] = [0, 0, 255]
    return rgb


def make_gif_from_history(history, duration):
    """履歴からGIFを生成"""
    frames = [Image.fromarray(grid_to_rgb(h)) for h in history]
    buf = io.BytesIO()
    frames[0].save(buf, format="GIF", save_all=True, append_images=frames[1:],
                   duration=duration, loop=0)
    buf.seek(0)
    return buf


def download_metrics_csv(metrics):
    """メトリクスをCSV形式に変換"""
    lines = ["t,hawk_ratio,avg_payoff"]
    for t, hr, ap in metrics:
        lines.append(f"{t},{hr},{ap}")
    return "\n".join(lines).encode()


def run_simulation(params):
    """シミュレーションを実行"""
    L = params['L']
    rng = np.random.default_rng(params['seed'])
    offsets = get_neighbor_offsets(params['neighborhood'])
    grid = init_state(L, params['init_mode'], params['p0'], params['num_patches'],
                      params['patch_radius'], params['patch_strategy'], params['seed'])
    history = [grid.copy()]
    metrics = []
    for t in range(1, params['generations'] + 1):
        pay = accumulate_payoffs(grid, offsets, params['V'], params['C'])
        fit = fitness_from_payoff(pay, params['w'], params['fitness_mapping'],
                                  params['shift_amount'])
        if params['update_rule'] == 'DB':
            moran_DB_step(grid, fit, offsets, params['mu'], rng)
        else:
            moran_BD_step(grid, fit, offsets, params['mu'], rng)
        grid = diffuse(grid, params['m'], offsets, rng)
        # 更新後の状態に基づいて利得を再計算し平均利得を求める
        pay = accumulate_payoffs(grid, offsets, params['V'], params['C'])
        avg_payoff = pay.mean()
        if t % params['draw_skip'] == 0 or t == params['generations']:
            history.append(grid.copy())
        if params['log_metrics']:
            metrics.append((t, np.mean(grid == 0), avg_payoff))
    return history, grid, metrics

# ========================= Streamlit UI =========================

st.title("2D Hawk-Dove Moran Game")

with st.sidebar:
    st.header("設定")
    L = st.number_input("L", min_value=10, max_value=200, value=40)
    neighborhood = st.selectbox("近傍", ["moore", "von_neumann"], index=0)
    update_rule = st.selectbox("更新規則", ["DB", "BD"], index=0)
    V = st.number_input("V", value=2.0)
    C = st.number_input("C", value=4.0)
    fitness_mapping = st.selectbox("適応度マッピング", ["clip0", "shift"], index=0)
    shift_amount = st.number_input("shift_amount", value=0.0)
    w = st.number_input("選択圧 w", value=1.0)
    mu = st.number_input("突然変異率 μ", min_value=0.0, max_value=1.0, value=0.0)
    generations = st.number_input("世代数", min_value=1, value=200)
    m = st.number_input("拡散確率 m", min_value=0.0, max_value=1.0, value=0.0)
    init_mode = st.selectbox("初期状態", ["random", "patches"], index=0)
    p0 = st.number_input("初期タカ率 p0", min_value=0.0, max_value=1.0, value=0.5)
    num_patches = st.number_input("パッチ数", min_value=1, value=3)
    patch_radius = st.number_input("パッチ半径", min_value=1, value=5)
    patch_strategy = st.selectbox("パッチ戦略", ["hawk", "dove", "mixed"], index=0)
    draw_skip = st.number_input("描画間隔", min_value=1, value=2)
    frame_ms = st.number_input("フレーム時間(ms)", min_value=10, value=80)
    seed = st.number_input("乱数シード", value=0)
    log_metrics = st.checkbox("メトリクスを記録", value=False)
    run = st.button("Run simulation")

params = dict(L=L, neighborhood=neighborhood, update_rule=update_rule, V=V, C=C,
              fitness_mapping=fitness_mapping, shift_amount=shift_amount, w=w, mu=mu,
              generations=generations, m=m, init_mode=init_mode, p0=p0,
              num_patches=num_patches, patch_radius=patch_radius,
              patch_strategy=patch_strategy, draw_skip=draw_skip, seed=seed,
              log_metrics=log_metrics)

if run:
    history, final_grid, metrics = run_simulation(params)
    gif_buf = make_gif_from_history(history, duration=frame_ms)
    st.image(gif_buf.getvalue(), format="GIF")
    st.download_button("Download GIF", gif_buf.getvalue(), file_name="hawk_dove.gif")
    st.image(grid_to_rgb(final_grid))
    st.json(params)
    if log_metrics:
        csv_bytes = download_metrics_csv(metrics)
        st.download_button("Download metrics", csv_bytes, file_name="metrics.csv")
