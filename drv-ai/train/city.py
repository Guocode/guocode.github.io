"""CityDrive — 网格城市端到端驾驶环境（向量化 numpy）

世界
  * G×G 灯控路口组成的网格路网（节点 = 路口，边 = 双向两车道街道）
  * 右侧行驶：每条方向的车道中心在行进方向右侧 1.85 m
  * 车道线（±1.85）、路沿（±3.7）、人行道（3.7~7.0）、建筑（街区深处）、树木（人行道带上确定性分布）
  * 红绿灯：每路口 NS/EW 两相位轮换（各 8 s，末 2 s 黄灯按红处理）

导航引擎
  * Dijkstra 最优路径（按边长）→ 路线节点序列
  * 行进中若偏离路线 > OFF_ROUTE_DIST，从最近路口重新规划到终点（走错路重规划）
  * 到达终点 +10 并自动随机换一个新终点，本回合继续（无尽）

智能体
  * NPC 车辆：车道内巡游，红灯停车线等待，随机选择下一路段（禁止 U 掉头）
  * 行人：沿街区人行道往返；骑车人：贴路沿骑行
  * 本车：自行车模型，连续动作 (加速度 a ∈ [-3,2] m/s², 转向 s ∈ [-1,1])

观测（端到端）
  * 6 路 BEV 语义相机：朝向 [0,60,...,300]°，各 32×32、44.8 m 见方
    R = 道路结构（路面 1.0 / 人行道 0.55 / 街区 0.15 / 树 0.35）
    G = 动态智能体（车 1.0 / 骑车人 0.7 / 行人 0.45）
    B = 车道线 0.8 + 路沿 0.5 + 红绿灯状态光斑（红 1.0 / 黄 0.7 / 绿 0.35）
  * 导航结构化 8 维：下一转向 sin/cos、到下一路口距离、下下转向 sin/cos、
    剩余路径长度、本车速度、下一路口灯态
"""

import numpy as np

# ================================================================ 常量
G_GRID = 4               # 路口网格 4×4
B_LEN = 90.0             # 路口间距 m
R_HALF = 3.7             # 可行驶半宽（路沿处）
LANE_OFF = 1.85          # 车道中心偏移
CURB_W = 0.22            # 路沿带宽
LINE_W = 0.16            # 车道线宽
SW_IN, SW_OUT = 3.7, 7.0 # 人行道内外沿
BLD_IN = 8.5             # 建筑起始深度
LIGHT_CYCLE = 16.0       # 灯周期 s
GREEN_T = 6.0            # 绿灯时长（后 2 s 黄灯）
STOP_DIST = 7.0          # 停车线距路口中心
DT = 0.05                # 20 Hz
MAX_STEPS = 3600         # 训练回合上限（无尽，到达终点自动换终点）

N_CARS, N_PEDS, N_BIKES = 22, 12, 6
VIEW_RES = 32            # BEV 每视角 32×32
VIEW_EXT = 44.8          # 覆盖 44.8 m（1.4 m/格）
VIEW_YAWS = np.deg2rad(np.array([0.0, 60.0, 120.0, 180.0, 240.0, 300.0]))
N_VIEW = 6
N_NAV = 8
OFF_ROUTE_DIST = 10.0    # 偏航重规划阈值

W_PROGRESS = 1.0
W_GOAL = 10.0
W_COLLIDE = 10.0
W_OFFROAD = 5.0
W_RED = 2.0
W_REPLAN = 0.2
W_COMFORT_A, W_COMFORT_S = 0.02, 0.02

A_MIN, A_MAX = -4.5, 2.0   # 动作空间：急刹留裕度（跟车防追尾需要 > NPC 的 4 m/s²）
V_EGO_MAX = 14.0
WHEELBASE = 2.7
STEER_GAIN = 0.42        # s=1 → 前轮 0.42 rad

# ================================================================ 图 / 导航
def node_xy(i, j):
    return np.array([i * B_LEN, j * B_LEN])

def node_id(i, j):
    return i * G_GRID + j

def node_ij(nid):
    return nid // G_GRID, nid % G_GRID

_NODE_POS = np.stack([np.tile(np.arange(G_GRID) * B_LEN, G_GRID),
                      np.repeat(np.arange(G_GRID) * B_LEN, G_GRID)], axis=1)

def neighbors(nid):
    i, j = node_ij(nid)
    out = []
    if i > 0: out.append(node_id(i - 1, j))
    if i < G_GRID - 1: out.append(node_id(i + 1, j))
    if j > 0: out.append(node_id(i, j - 1))
    if j < G_GRID - 1: out.append(node_id(i, j + 1))
    return out

def dijkstra_route(src, dst):
    """最短路径（边长均匀）→ 节点序列 [src..dst]"""
    import heapq
    dist = {src: 0}
    prev = {}
    pq = [(0.0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, 1e18):
            continue
        if u == dst:
            break
        for v in neighbors(u):
            nd = d + B_LEN
            if nd < dist.get(v, 1e18):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    path = [dst]
    while path[-1] != src:
        path.append(prev[path[-1]])
    return path[::-1]

def route_polyline(route):
    pts = _NODE_POS[np.array(route)]
    seg = np.diff(pts, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    return pts, seg, seg_len

# 有向边表：相邻节点对 (f<t) 生成两个方向
EDGES = []
EDGE_ID = {}
for _i in range(G_GRID):
    for _j in range(G_GRID):
        for _n in neighbors(node_id(_i, _j)):
            if node_id(_i, _j) < _n:
                EDGES.append((node_id(_i, _j), _n))
                EDGES.append((_n, node_id(_i, _j)))
for _k, (_f, _t) in enumerate(EDGES):
    EDGE_ID[(_f, _t)] = _k
N_EDGES = len(EDGES)

# 有向边数组化（gather 用）
_EDGE_FROM = np.array([e[0] for e in EDGES])
_EDGE_DIR = np.stack([(_NODE_POS[e[1]] - _NODE_POS[e[0]]) / B_LEN for e in EDGES])
_EDGE_RIGHT = np.stack([_EDGE_DIR[:, 1], -_EDGE_DIR[:, 0]], axis=1)

def project_route(x, y, route):
    """返回 (progress 沿线累计, 离线距离, 最近节点)"""
    pts, seg, seg_len = route_polyline(route)
    if len(route) == 1:
        return 0.0, float(np.hypot(x - pts[0, 0], y - pts[0, 1])), route[0]
    best = (1e18, 0.0)
    cum = 0.0
    for k in range(len(seg_len)):
        p0, d = pts[k], seg[k]
        L2 = seg_len[k] ** 2
        t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((x - p0[0]) * d[0] + (y - p0[1]) * d[1]) / L2))
        px, py = p0[0] + t * d[0], p0[1] + t * d[1]
        dd = np.hypot(x - px, y - py)
        if dd < best[0]:
            best = (dd, cum + t * seg_len[k])
        cum += seg_len[k]
    total = seg_len.sum()
    return best[1], best[0], int(np.argmin(np.hypot(_NODE_POS[:, 0] - x, _NODE_POS[:, 1] - y)))

# ================================================================ 灯态
def light_state(t, travel_dir):
    """travel_dir: 0 = 南北向（|dy|>|dx|），1 = 东西向。返回 0 绿 1 黄 2 红"""
    ph = int((t % LIGHT_CYCLE) // (LIGHT_CYCLE / 2))   # 0: NS 绿, 1: EW 绿
    tt = (t % LIGHT_CYCLE) % (LIGHT_CYCLE / 2)
    yellow = tt >= GREEN_T
    my_green = (ph == 0 and travel_dir == 0) or (ph == 1 and travel_dir == 1)
    if my_green and not yellow:
        return 0
    if my_green and yellow:
        return 1
    return 2

# ================================================================ 环境
class CityEnv:
    """单实例城市驾驶环境（训练时并行跑多个实例）。"""

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.t = 0.0
        self.steps = 0
        self._npc_cache = None
        self._npc_dirty = True
        self._reset_world(seed)

    # ---------- 世界初始化 ----------
    def _reset_world(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        rng = self.rng
        # NPC 车辆：随机有向边 + 位置
        self.car_edge = rng.integers(0, N_EDGES, size=N_CARS)
        self.car_s = rng.uniform(4.0, B_LEN - 12.0, size=N_CARS)
        self.car_v = rng.uniform(4.0, 7.0, size=N_CARS)
        self.car_next = rng.integers(0, 3, size=N_CARS)   # 预选下一路口转向
        # 行人：随机街区的一条边，沿其往返
        self.ped_block = rng.integers(0, (G_GRID - 1) ** 2, size=N_PEDS)
        self.ped_side = rng.integers(0, 4, size=N_PEDS)
        self.ped_p = rng.uniform(4.0, B_LEN - 4.0, size=N_PEDS)
        self.ped_dir = rng.choice([-1.0, 1.0], size=N_PEDS)
        # 骑车人：贴右侧路沿
        self.bike_edge = rng.integers(0, N_EDGES, size=N_BIKES)
        self.bike_s = rng.uniform(4.0, B_LEN - 8.0, size=N_BIKES)
        self.bike_v = rng.uniform(2.5, 4.0, size=N_BIKES)
        # 本车（出生点避让 NPC：6 m 内有智能体则换路线重试）
        self._npc_dirty = True
        self.spawn_goal()

    def spawn_goal(self, keep_pos=False):
        rng = self.rng
        n_total = G_GRID * G_GRID
        if keep_pos:
            cur = self._nearest_node()
        else:
            cur = rng.integers(0, n_total)
        while True:
            goal = rng.integers(0, n_total)
            if goal != cur:
                break
        if not keep_pos:
            self.route = dijkstra_route(int(cur), int(goal))
            start_nbr = self.route[1]          # 出生航向与路线首边对齐
            p = _NODE_POS[cur]
            q = _NODE_POS[start_nbr]
            dvec = (q - p) / np.linalg.norm(q - p)
            sx = p[0] + dvec[0] * 6.0 + dvec[1] * LANE_OFF
            sy = p[1] + dvec[1] * 6.0 - dvec[0] * LANE_OFF
            # 出生点避让：附近有 NPC 则换终点重试
            for _try in range(24):
                ap, _acl = self.npc_positions()
                if np.min(np.hypot(ap[:, 0] - sx, ap[:, 1] - sy)) > 6.0:
                    break
                while goal == cur:
                    goal = int(rng.integers(0, n_total))
                self.route = dijkstra_route(int(cur), int(goal))
                start_nbr = self.route[1]
                q = _NODE_POS[start_nbr]
                dvec = (q - p) / np.linalg.norm(q - p)
                sx = p[0] + dvec[0] * 6.0 + dvec[1] * LANE_OFF
                sy = p[1] + dvec[1] * 6.0 - dvec[0] * LANE_OFF
            self.x, self.y = sx, sy
            self.psi = np.arctan2(dvec[1], dvec[0])
            self.v = 4.0
            self.a_cmd, self.s_cmd = 0.0, 0.0
        else:
            self.route = dijkstra_route(self._nearest_node(), int(goal))
        self.progress_prev, _, _ = project_route(self.x, self.y, self.route)
        self.prog = self.progress_prev
        self.dev = 0.0
        self.red_flag = True         # 出生即在线内，豁免闯红灯判（离开 12 m 后复位）
        self.replans = 0
        self.goals_done = 0
        self.collided = False
        self.offroaded = False

    def _nearest_node(self):
        return int(np.argmin(np.hypot(_NODE_POS[:, 0] - self.x, _NODE_POS[:, 1] - self.y)))

    # ---------- 静态语义查询（O(1)/点，网格解析式） ----------
    def static_sem(self, px, py):
        """向量化：返回 (road, sidewalk, lane_line, curb, tree) 布尔/浮点数组"""
        span = (G_GRID - 1) * B_LEN
        dxv = np.abs(px / B_LEN - np.round(px / B_LEN)) * B_LEN
        dyh = np.abs(py / B_LEN - np.round(py / B_LEN)) * B_LEN
        # 只有网格范围内才有路
        in_v = (py >= -R_HALF) & (py <= span + R_HALF)
        in_h = (px >= -R_HALF) & (px <= span + R_HALF)
        near_v = (dxv <= R_HALF) & in_v
        near_h = (dyh <= R_HALF) & in_h
        road = near_v | near_h
        sw = ((dxv > R_HALF) & (dxv <= SW_OUT) & in_v) | ((dyh > R_HALF) & (dyh <= SW_OUT) & in_h)
        lane = ((np.abs(dxv - LANE_OFF) <= LINE_W) & in_v) | ((np.abs(dyh - LANE_OFF) <= LINE_W) & in_h)
        curb = ((np.abs(dxv - R_HALF) <= CURB_W) & in_v) | ((np.abs(dyh - R_HALF) <= CURB_W) & in_h)
        tree = (sw & (((px % 15.0) < 1.2) | ((py % 15.0) < 1.2))
                & (np.abs((px % 15.0) - 0.6) < 1.0) & (np.abs((py % 15.0) - 0.6) < 1.0))
        return road, sw, lane, curb, tree

    # ---------- NPC 位置与步进 ----------
    def _edge_geom(self, eidx):
        """有向边索引 → (from_id, to_id, 单位方向, 长度)"""
        f, tt = EDGES[eidx]
        p, q = _NODE_POS[f], _NODE_POS[tt]
        d = (q - p) / B_LEN
        return f, tt, d, B_LEN

    def npc_positions(self):
        """所有 NPC 的位置与类别（向量化 + 缓存）。类别 0 车 1 骑车 2 行人"""
        if not self._npc_dirty and self._npc_cache is not None:
            return self._npc_cache
        pos = np.empty((N_CARS + N_BIKES + N_PEDS, 2))
        # 车：车道中心；骑车人：贴右路沿
        pos[:N_CARS] = (_NODE_POS[_EDGE_FROM[self.car_edge]]
                        + _EDGE_DIR[self.car_edge] * self.car_s[:, None]
                        + _EDGE_RIGHT[self.car_edge] * LANE_OFF)
        pos[N_CARS:N_CARS + N_BIKES] = (_NODE_POS[_EDGE_FROM[self.bike_edge]]
                                        + _EDGE_DIR[self.bike_edge] * self.bike_s[:, None]
                                        + _EDGE_RIGHT[self.bike_edge] * (R_HALF - 0.8))
        # 行人：街区人行道（距路中线 OFF_SW，四边往返）
        bi = self.ped_block
        bx = (bi // (G_GRID - 1)) * B_LEN
        by = (bi % (G_GRID - 1)) * B_LEN
        pp = self.ped_p
        off = 0.5 * (SW_IN + SW_OUT)          # 5.35 m
        pxs = np.where(self.ped_side == 0, bx + pp,
              np.where(self.ped_side == 1, bx + B_LEN - off,
              np.where(self.ped_side == 2, bx + pp, bx + off)))
        pys = np.where(self.ped_side == 0, by + off,
              np.where(self.ped_side == 1, by + pp,
              np.where(self.ped_side == 2, by + B_LEN - off, by + pp)))
        pos[N_CARS + N_BIKES:] = np.stack([pxs, pys], axis=1)
        cls = np.concatenate([np.zeros(N_CARS), np.ones(N_BIKES), np.full(N_PEDS, 2.0)])
        self._npc_cache = (pos, cls)
        self._npc_dirty = False
        return pos, cls

    def _step_npcs(self):
        self._npc_dirty = True             # NPC 已移动，位置缓存失效
        rng = self.rng
        t = self.t
        # --- 车辆 ---
        for k in range(N_CARS):
            f, tt, d, L = self._edge_geom(self.car_edge[k])
            dir_cls = 0 if abs(d[1]) > abs(d[0]) else 1
            # 前车距离（同边同向后车）
            gap = L
            for m in range(N_CARS):
                if m != k and self.car_edge[m] == self.car_edge[k]:
                    ds = self.car_s[m] - self.car_s[k]
                    if 0.5 < ds < gap:
                        gap = ds
            # 本车也纳入跟车避让（横偏 <2.2 m 视为同车道）
            rel_ego_x = self.x - _NODE_POS[f][0]
            rel_ego_y = self.y - _NODE_POS[f][1]
            ds_ego = rel_ego_x * d[0] + rel_ego_y * d[1]
            lat_ego = abs(rel_ego_x * d[1] - rel_ego_y * d[0])
            if lat_ego < 2.2 and 0.0 < ds_ego - self.car_s[k] < gap:
                gap = ds_ego - self.car_s[k]
            v_target = 7.0
            # 跟车安全速度：匀减速 2.2 m/s² 恰在 (gap-4) 处停住
            v_safe = np.sqrt(2.0 * 2.2 * max(gap - 4.0, 0.0))
            v_target = min(v_target, v_safe)
            stop_s = L - STOP_DIST
            if self.car_s[k] < stop_s:
                ls = light_state(t, dir_cls)
                if ls >= 1:
                    dd = stop_s - self.car_s[k]
                    if dd < 22.0:
                        v_stop = np.sqrt(max(0.0, 2.0 * 2.2 * max(dd, 0.0)))
                        v_target = min(v_target, v_stop)
            if self.car_s[k] >= stop_s - 0.5 and v_target < 1.0 and light_state(t, dir_cls) >= 1:
                self.car_v[k] = 0.0
            else:
                self.car_v[k] = np.clip(self.car_v[k] + np.clip(v_target - self.car_v[k], -4 * DT, 2 * DT), 0.0, 8.0)
            self.car_s[k] += self.car_v[k] * DT
            if self.car_s[k] >= L - 1.0 and light_state(t, dir_cls) == 0:
                # 绿灯：随机选下一路口（不回头）；目标边入口有车/本车则等待
                nbrs = [n for n in neighbors(tt) if n != f]
                nx = nbrs[int(self.car_next[k] * len(nbrs) / 3) % len(nbrs)]
                ne = self._edge_index(tt, nx)
                nd = _EDGE_DIR[ne]
                np_ = _NODE_POS[tt]
                blocked = False
                for m in range(N_CARS):
                    if m != k and self.car_edge[m] == ne and self.car_s[m] < 9.0:
                        blocked = True
                        break
                if not blocked:
                    rx, ry = self.x - np_[0], self.y - np_[1]
                    ds_e = rx * nd[0] + ry * nd[1]
                    lat_e = abs(rx * nd[1] - ry * nd[0])
                    if 0.0 < ds_e < 9.0 and lat_e < 2.2:
                        blocked = True
                if blocked:
                    self.car_s[k] = L - 1.0
                    self.car_v[k] = 0.0
                else:
                    self.car_next[k] = rng.integers(0, 3)
                    self.car_edge[k] = ne
                    self.car_s[k] = 0.5
        # --- 骑车人（贴路沿，不守灯，低速） ---
        for k in range(N_BIKES):
            self.bike_s[k] += self.bike_v[k] * DT
            if self.bike_s[k] >= B_LEN - 1.0:
                f, tt, d, L = self._edge_geom(self.bike_edge[k])
                nbrs = [n for n in neighbors(tt) if n != f]
                self.bike_edge[k] = self._edge_index(tt, nbrs[int(rng.integers(0, len(nbrs)))])
                self.bike_s[k] = 0.5
        # --- 行人 ---
        lim = B_LEN - 8.0
        self.ped_p += self.ped_dir * 1.3 * DT
        flip = (self.ped_p > lim) & (self.ped_dir > 0) | (self.ped_p < 4.0) & (self.ped_dir < 0)
        self.ped_dir = np.where(flip, -self.ped_dir, self.ped_dir)
        self.ped_p = np.clip(self.ped_p, 4.0, lim)

    @staticmethod
    def _edge_index(f, t):
        return EDGE_ID[(f, t)]

    # ---------- 导航 ----------
    def _nav_update(self):
        prog, dev, near_node = project_route(self.x, self.y, self.route)
        if dev > OFF_ROUTE_DIST:
            # 走错路：从最近路口重新规划到原终点
            self.route = dijkstra_route(near_node, self.route[-1])
            self.replans += 1
            self.replan_flag = True
            prog, dev, near_node = project_route(self.x, self.y, self.route)
        else:
            self.replan_flag = False
        pts, seg, seg_len = route_polyline(self.route)
        self.prog = prog
        self.dev = dev
        return prog

    def _nav_inputs(self):
        """结构化导航 8 维"""
        pts, seg, seg_len = route_polyline(self.route)
        total = seg_len.sum()
        remain = total - self.prog
        # 下一节点
        cum = 0.0
        nxt_k = len(seg) - 1
        for k in range(len(seg_len)):
            if self.prog <= cum + seg_len[k]:
                nxt_k = k
                break
            cum += seg_len[k]
        d_next = cum + seg_len[nxt_k] - self.prog
        node_next = self.route[nxt_k + 1]
        node_cur = self.route[nxt_k]
        # 下一转向（相对当前航向）
        d1 = _NODE_POS[node_next] - _NODE_POS[node_cur]
        d1 = d1 / (np.linalg.norm(d1) + 1e-9)
        h1 = np.arctan2(d1[1], d1[0])
        turn1 = ((h1 - self.psi + np.pi) % (2 * np.pi)) - np.pi
        # 下下转向
        if nxt_k + 2 <= len(seg) - 1:
            n2 = self.route[nxt_k + 2]
            d2 = _NODE_POS[n2] - _NODE_POS[node_next]
            d2 = d2 / (np.linalg.norm(d2) + 1e-9)
            h2 = np.arctan2(d2[1], d2[0])
            turn2 = ((h2 - h1 + np.pi) % (2 * np.pi)) - np.pi
        else:
            turn2 = 0.0
        # 下一路口灯态（本车行进方向）
        dcur = np.array([np.cos(self.psi), np.sin(self.psi)])
        dir_cls = 0 if abs(dcur[1]) > abs(dcur[0]) else 1
        ls = light_state(self.t, dir_cls)
        red_ahead = (ls >= 1) and (d_next < 30.0)
        return np.array([
            np.sin(turn1), np.cos(turn1), np.tanh(d_next / 30.0),
            np.sin(turn2), np.cos(turn2), np.tanh(remain / 400.0),
            self.v / V_EGO_MAX, 1.0 if red_ahead else 0.0,
        ], dtype=np.float32)

    # ---------- BEV 相机 ----------
    def render_views(self):
        """6 路 BEV 语义图，返回 (6,3,32,32) float32 ∈ [0,1]"""
        C, R = VIEW_RES, VIEW_RES
        cell = VIEW_EXT / C
        o = np.arange(C) * cell + cell / 2 - VIEW_EXT / 2        # 局部坐标
        gx, gy = np.meshgrid(o, o)                                # gy 向下
        # 本车相关灯态
        dcur = np.array([np.cos(self.psi), np.sin(self.psi)])
        dir_cls = 0 if abs(dcur[1]) > abs(dcur[0]) else 1
        ls = light_state(self.t, dir_cls)
        pts, seg, seg_len = route_polyline(self.route)
        # 下一路口位置（沿路线）
        cum = 0.0
        nxt_k = len(seg) - 1
        for k in range(len(seg_len)):
            if self.prog <= cum + seg_len[k]:
                nxt_k = k
                break
            cum += seg_len[k]
        node_next = self.route[nxt_k + 1]
        lp = _NODE_POS[node_next]
        d_next = np.linalg.norm(lp - np.array([self.x, self.y]))

        agent_pos, agent_cls = self.npc_positions()
        near = np.hypot(agent_pos[:, 0] - self.x, agent_pos[:, 1] - self.y) < 42.0
        agent_pos, agent_cls = agent_pos[near], agent_cls[near]

        out = np.zeros((N_VIEW, 3, C, R), dtype=np.float32)
        ca, sa = np.cos(self.psi), np.sin(self.psi)
        for vi in range(N_VIEW):
            yaw = self.psi + VIEW_YAWS[vi]
            cy, sy = np.cos(yaw), np.sin(yaw)
            # 视角局部 (gx 前, gy 右) → 世界
            wx = self.x + gx * cy - gy * sy
            wy = self.y + gx * sy + gy * cy
            road, sw, lane, curb, tree = self.static_sem(wx, wy)
            ch0 = np.where(road, 1.0, np.where(sw, 0.55, 0.15))
            ch0 = np.where(tree, 0.35, ch0)
            ch2 = np.where(lane, 0.8, np.where(curb, 0.5, 0.0))
            # 动态智能体
            if len(agent_pos):
                lx = (agent_pos[:, 0] - self.x) * cy + (agent_pos[:, 1] - self.y) * sy
                ly = -(agent_pos[:, 0] - self.x) * sy + (agent_pos[:, 1] - self.y) * cy
                ix = (lx + VIEW_EXT / 2) / cell
                iy = (ly + VIEW_EXT / 2) / cell
                for ag, cv in ((0, 1.0), (1, 0.7), (2, 0.45)):
                    sel = (agent_cls == ag) & (ix >= 1) & (ix < C - 1) & (iy >= 1) & (iy < C - 1)
                    if sel.any():
                        x0 = ix[sel].astype(int); y0 = iy[sel].astype(int)
                        ch1 = out[vi, 1]
                        for ddx in (-1, 0, 1):
                            for ddy in (-1, 0, 1):
                                w = 1.0 if (ddx == 0 and ddy == 0) else (0.55 if abs(ddx) + abs(ddy) == 1 else 0.3)
                                xi = np.clip(x0 + ddx, 0, C - 1); yi = np.clip(y0 + ddy, 0, R - 1)
                                ch1[yi, xi] = np.maximum(ch1[yi, xi], cv * w)
            # 红绿灯光斑（画在下一路口中心处）
            if d_next < 60.0:
                lx2 = (lp[0] - self.x) * cy + (lp[1] - self.y) * sy
                ly2 = -(lp[0] - self.x) * sy + (lp[1] - self.y) * cy
                ix = int((lx2 + VIEW_EXT / 2) / cell); iy = int((ly2 + VIEW_EXT / 2) / cell)
                if 1 <= ix < C - 1 and 1 <= iy < R - 1:
                    val = {0: 0.35, 1: 0.7, 2: 1.0}[ls]
                    out[vi, 2, iy - 1:iy + 2, ix - 1:ix + 2] = np.maximum(
                        out[vi, 2, iy - 1:iy + 2, ix - 1:ix + 2], val)
            out[vi, 0] = ch0
            out[vi, 2] = np.maximum(out[vi, 2], ch2)
        return out

    # ---------- 主 step ----------
    def obs(self):
        return self.render_views(), self._nav_inputs()

    def step(self, a_cmd, s_cmd):
        """连续动作：加速度 m/s²，转向 [-1,1]"""
        self.a_cmd, self.s_cmd = a_cmd, s_cmd
        steer = s_cmd * STEER_GAIN
        a = np.clip(a_cmd, A_MIN, A_MAX)
        self.v = np.clip(self.v + a * DT, 0.0, V_EGO_MAX)
        self.psi += self.v / WHEELBASE * np.tan(steer) * DT
        self.x += self.v * np.cos(self.psi) * DT
        self.y += self.v * np.sin(self.psi) * DT
        self.t += DT
        self.steps += 1
        self._step_npcs()
        prog_prev = self.progress_prev
        prog = self._nav_update()
        self.progress_prev = prog

        reward = W_PROGRESS * (prog - prog_prev) - W_COMFORT_A * abs(a) - W_COMFORT_S * s_cmd * s_cmd
        done = False
        info = {"event": ""}

        # 碰撞
        ap, acl = self.npc_positions()
        dd = np.hypot(ap[:, 0] - self.x, ap[:, 1] - self.y)
        hit_car = (acl == 0) & (dd < 2.4)
        hit_bike = (acl == 1) & (dd < 1.9)
        hit_ped = (acl == 2) & (dd < 1.3)
        if (hit_car | hit_bike | hit_ped).any():
            reward -= W_COLLIDE
            done = True
            info["event"] = "collision"
            self.collided = True
        # 冲出路面
        sem = self.static_sem(np.array([self.x]), np.array([self.y]))
        road_q = sem[0][0]
        if not road_q:
            reward -= W_OFFROAD
            done = True
            info["event"] = "offroad"
            self.offroaded = True
        # 闯红灯：接近下一路口且红灯时进入 STOP_DIST 内
        dcur = np.array([np.cos(self.psi), np.sin(self.psi)])
        dir_cls = 0 if abs(dcur[1]) > abs(dcur[0]) else 1
        pts, seg, seg_len = route_polyline(self.route)
        node_next = self.route[-1]
        # 找当前路段的下一节点
        cum = 0.0
        nxt_k = len(seg) - 1
        for k in range(len(seg_len)):
            if self.prog <= cum + seg_len[k]:
                nxt_k = k
                break
            cum += seg_len[k]
        node_next = self.route[nxt_k + 1]
        lp = _NODE_POS[node_next]
        d_next = np.linalg.norm(lp - np.array([self.x, self.y]))
        ls = light_state(self.t, dir_cls)
        if d_next < STOP_DIST and ls >= 1 and not self.red_flag:
            self.red_flag = True
            reward -= W_RED
            info["event"] = "red_light"
        if d_next > STOP_DIST + 5:
            self.red_flag = False
        # 到达终点
        remain = seg_len.sum() - self.prog
        if remain < 4.0:
            reward += W_GOAL
            self.goals_done += 1
            self.spawn_goal(keep_pos=True)     # 自动随机换新终点
            info["event"] = "goal"
        if self.replan_flag:
            reward -= W_REPLAN
        if self.steps >= MAX_STEPS:
            done = True
            info["event"] = "timeout"
        return self.obs(), float(reward), done, info

    # ---------- 启发式专家 ----------
    def expert(self):
        pts, seg, seg_len = route_polyline(self.route)
        total = seg_len.sum()
        # 前瞻目标点：距离随速度自适应（低速转弯用短前瞻防切弯）
        look = float(np.clip(0.7 * self.v + 3.5, 4.5, 10.0))
        target_s = min(self.prog + look, total - 0.1)
        cum = 0.0
        tp = pts[-1]
        for k in range(len(seg_len)):
            if target_s <= cum + seg_len[k]:
                t = (target_s - cum) / max(seg_len[k], 1e-6)
                tp = pts[k] + t * seg[k]
                dn = seg[k] / max(np.linalg.norm(seg[k]), 1e-9)
                tp = tp + np.array([dn[1], -dn[0]]) * LANE_OFF
                break
            cum += seg_len[k]
        dx, dy = tp[0] - self.x, tp[1] - self.y
        h_err = ((np.arctan2(dy, dx) - self.psi + np.pi) % (2 * np.pi)) - np.pi
        steer = np.clip(1.6 * h_err, -1.0, 1.0)
        # 速度目标
        v_t = 8.0
        nxt_k = len(seg) - 1
        cum = 0.0
        for k in range(len(seg_len)):
            if self.prog <= cum + seg_len[k]:
                nxt_k = k
                break
            cum += seg_len[k]
        node_next = self.route[nxt_k + 1]
        d_next = np.linalg.norm(_NODE_POS[node_next] - np.array([self.x, self.y]))
        # 转向前减速
        if nxt_k + 1 <= len(seg) - 1:
            a1 = _NODE_POS[node_next] - _NODE_POS[self.route[nxt_k]]
            a2 = _NODE_POS[self.route[nxt_k + 2]] - _NODE_POS[node_next] if nxt_k + 2 < len(self.route) else a1
            ang = np.arccos(np.clip(np.dot(a1, a2) / (np.linalg.norm(a1) * np.linalg.norm(a2) + 1e-9), -1, 1))
            if ang > 0.3 and d_next < 22.0:
                v_t = min(v_t, 3.5)
        # 转弯中减速：航向与当前路段方向偏差大时压低目标速度
        dn = seg[nxt_k] / max(np.linalg.norm(seg[nxt_k]), 1e-9)
        hdev = abs(((np.arctan2(dn[1], dn[0]) - self.psi + np.pi) % (2 * np.pi)) - np.pi)
        if hdev > 0.35:
            v_t = min(v_t, 3.0)
        # 红灯
        dcur = np.array([np.cos(self.psi), np.sin(self.psi)])
        dir_cls = 0 if abs(dcur[1]) > abs(dcur[0]) else 1
        ls = light_state(self.t, dir_cls)
        if ls >= 1 and d_next < 30.0:
            stop_at = d_next - STOP_DIST + 1.5
            v_stop = np.sqrt(max(0.0, 2.0 * 2.6 * max(stop_at, 0.0)))
            v_t = min(v_t, v_stop)
        # 前车 / 骑车人（骑车人贴路沿骑进本车道边缘，一并跟车）
        ap, acl = self.npc_positions()
        fwd = np.array([np.cos(self.psi), np.sin(self.psi)])
        rel = ap - np.array([self.x, self.y])
        along = rel @ fwd
        lat = np.abs(rel[:, 0] * fwd[1] - rel[:, 1] * fwd[0])
        ahead = (acl <= 1) & (along > 1.0) & (along < 20.0) & (lat < 2.0)
        if ahead.any():
            gap = along[ahead].min()
            # 安全速度曲线：按 4 m/s² 满舵计，前车急刹时留 6 m 静态缓冲
            v_t = min(v_t, np.sqrt(2.0 * 4.0 * max(gap - 6.0, 0.0)))
        a = np.clip(3.0 * (v_t - self.v), A_MIN, A_MAX)
        return float(a), float(steer)


# ================================================================ 测试
if __name__ == "__main__":
    import time
    env = CityEnv(seed=3)
    t0 = time.time()
    steps = 0
    events = {"goal": 0, "collision": 0, "offroad": 0, "red_light": 0}
    for ep in range(4):
        env = CityEnv(seed=ep)
        m = 0.0
        for i in range(2400):
            a, s = env.expert()
            (_, nav), r, done, info = env.step(a, s)
            m += r
            steps += 1
            if info["event"]:
                events[info["event"]] = events.get(info["event"], 0) + 1
            if done:
                break
        print(f"ep{ep}: reward={m:8.1f} goals={env.goals_done} replans={env.replans} steps={i+1} v_avg={env.v:.1f}")
    t = time.time() - t0
    print(f"events={events}")
    print(f"{steps} steps in {t:.1f}s -> {steps/t:.0f} steps/s")
    # 单帧渲染检查
    v, nav = env.obs()
    print("views", v.shape, "nav", np.round(nav, 2))
