from dataclasses import dataclass
import numpy as np
import math


def gravity_from_altitude(h_m: float) -> float:
    g0 = 9.80665
    Re = 6371000.0
    h = max(0.0, float(h_m))
    return g0 * (Re / (Re + h)) ** 2


def isa_pressure(alt_m: float, P_sea: float = 101325.0) -> float:
    return P_sea * (1.0 - 2.25577e-5 * alt_m) ** 5.25588


L_lapse = -0.0065
R_gas = 8.3144598
M_air = 0.0289644
g0 = 9.80665


T_TROPOPAUSE = 216.15


def get_atmosphere(current_alt: float, alt_gun: float, T0_K: float, P0_pa: float):
    h_diff = max(0.0, current_alt - alt_gun)

    if T0_K <= T_TROPOPAUSE:
        T_current = T_TROPOPAUSE
        P_current = P0_pa * np.exp(-g0 * M_air * h_diff / (R_gas * T_TROPOPAUSE))
    else:
        h_tropopause_from_gun = (T_TROPOPAUSE - T0_K) / L_lapse

        if h_diff <= h_tropopause_from_gun:
            T_current = T0_K + L_lapse * h_diff
            exponent = -(g0 * M_air) / (R_gas * L_lapse)
            P_current = P0_pa * (T_current / T0_K) ** exponent
        else:
            T_current = T_TROPOPAUSE
            exponent = -(g0 * M_air) / (R_gas * L_lapse)
            P_tropopause = P0_pa * (T_TROPOPAUSE / T0_K) ** exponent
            h_above = h_diff - h_tropopause_from_gun
            P_current = P_tropopause * np.exp(-g0 * M_air * h_above / (R_gas * T_TROPOPAUSE))

    rho_current = (P_current * M_air) / (R_gas * T_current)
    return rho_current, T_current


def calc_dynamic_cd_G7(v_rel: float, T_kelvin: float) -> float:
    c = 20.05 * np.sqrt(max(T_kelvin, 1.0))
    Ma = v_rel / max(c, 1e-6)

    cd_sub = 0.12
    cd_trans = 0.12 + (Ma - 0.9) * (0.28 / 0.30)
    cd_super = 0.40 * (1.2 / max(Ma, 1e-12)) ** 0.5

    w1 = 1.0 / (1.0 + np.exp(-(Ma - 0.9) / 0.03))
    w2 = 1.0 / (1.0 + np.exp(-(Ma - 1.2) / 0.03))

    return cd_sub * (1.0 - w1) + cd_trans * w1 * (1.0 - w2) + cd_super * w2


@dataclass
class ProjectileParams:
    mass: float = 0.04828
    caliber: float = 0.01295
    v0_base: float = 840.0
    T_powder_C: float = 15.0
    temp_coeff: float = 1.2
    wind_x: float = 0.0
    wind_y: float = 0.0
    wind_z: float = 0.0
    alt_gun: float = 0.0
    cant_angle_deg: float = 0.0
    T0_C: float = 15.0
    P0_Pa: float = 101325.0

    def area(self) -> float:
        return math.pi * (self.caliber / 2.0) ** 2

    @property
    def v0(self):
        return self.v0_base

    @v0.setter
    def v0(self, value):
        self.v0_base = value

    def get_actual_v0(self) -> float:
        return self.v0_base + self.temp_coeff * (self.T_powder_C - 15.0)


def get_dynamic_wind(current_alt: float, params: ProjectileParams):
    h_ref = 2.0
    z0 = 0.05
    h_above_ground = max(current_alt - params.alt_gun, z0 + 0.01)
    scale = np.log(h_above_ground / z0) / np.log(h_ref / z0)
    scale = max(0.0, min(scale, 5.0))
    return (
        params.wind_x * scale,
        params.wind_y,
        params.wind_z * scale,
    )


def dynamics(state: np.ndarray, params: ProjectileParams) -> np.ndarray:
    x, y, z, vx, vy, vz = state
    current_alt = params.alt_gun + y

    curr_wind_x, curr_wind_y, curr_wind_z = get_dynamic_wind(current_alt, params)
    vx_rel = vx - curr_wind_x
    vy_rel = vy - curr_wind_y
    vz_rel = vz - curr_wind_z
    v_rel = np.sqrt(vx_rel ** 2 + vy_rel ** 2 + vz_rel ** 2) + 1e-12

    T0_K = params.T0_C + 273.15
    current_rho, current_T_K = get_atmosphere(current_alt, params.alt_gun, T0_K, params.P0_Pa)
    current_g = gravity_from_altitude(current_alt)
    current_cd = calc_dynamic_cd_G7(v_rel, current_T_K)
    current_k = 0.5 * current_rho * current_cd * params.area()

    ax = -(current_k / params.mass) * v_rel * vx_rel
    ay = -current_g - (current_k / params.mass) * v_rel * vy_rel
    az = -(current_k / params.mass) * v_rel * vz_rel
    return np.array([vx, vy, vz, ax, ay, az], dtype=np.float64)


def rk4_step(state: np.ndarray, dt: float, params: ProjectileParams) -> np.ndarray:
    k1 = dynamics(state, params)
    k2 = dynamics(state + 0.5 * dt * k1, params)
    k3 = dynamics(state + 0.5 * dt * k2, params)
    k4 = dynamics(state + dt * k3, params)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def _make_initial_state(theta_deg: float, alpha_deg: float, params: ProjectileParams,
                        x0: float, y0: float, z0: float):
    theta = np.deg2rad(theta_deg)
    alpha = np.deg2rad(alpha_deg)
    phi = np.deg2rad(params.cant_angle_deg)
    v0_actual = params.get_actual_v0()

    vx0 = v0_actual * np.cos(theta) * np.cos(alpha)
    vy0 = v0_actual * (np.sin(theta) * np.cos(phi) - np.cos(theta) * np.sin(alpha) * np.sin(phi))
    vz0 = v0_actual * (np.sin(theta) * np.sin(phi) + np.cos(theta) * np.sin(alpha) * np.cos(phi))

    state0 = np.array([x0, y0, z0, vx0, vy0, vz0], dtype=np.float64)
    return state0, float(v0_actual)


def hit_at_x(theta_deg: float,
             params: ProjectileParams,
             x_target: float,
             alpha_deg: float = 0.0,
             dt: float = 0.01,
             t_max: float = 200.0,
             x0: float = 0.0,
             y0: float = 0.0,
             z0: float = 0.0) -> dict:
    """
    只积分到 x=x_target 截面，避免生成整条轨迹。
    返回:
      hit=True  -> 成功穿过目标截面，给出 t_hit/y_hit/z_hit
      hit=False -> 未到目标截面就落地或超时，给出落地点 range/z_land
    """
    if dt <= 0:
        raise ValueError("dt must be positive.")
    if t_max <= 0:
        raise ValueError("t_max must be positive.")

    state, v0_actual = _make_initial_state(theta_deg, alpha_deg, params, x0, y0, z0)
    if state[4] <= 0 and y0 <= 0:
        return {
            "hit": False,
            "t_hit": None,
            "y_hit": None,
            "z_hit": None,
            "range": float(x0),
            "z_land": float(z0),
            "theta_deg": float(theta_deg),
            "alpha_deg": float(alpha_deg),
            "v0_actual": float(v0_actual),
        }

    x_target = float(x_target)
    n_steps = int(np.ceil(t_max / dt))
    t_cur = 0.0

    for _ in range(n_steps):
        state_next = rk4_step(state, dt, params)
        t_next = t_cur + dt

        x_prev, y_prev, z_prev = state[0], state[1], state[2]
        x_cur, y_cur, z_cur = state_next[0], state_next[1], state_next[2]

        frac_x = None
        if (x_prev <= x_target <= x_cur) or (x_cur <= x_target <= x_prev):
            dx = x_cur - x_prev
            if abs(dx) < 1e-12:
                frac_x = 1.0
            else:
                frac_x = (x_target - x_prev) / dx
                if not (0.0 <= frac_x <= 1.0):
                    frac_x = None

        frac_ground = None
        if y_prev > 0.0 and y_cur <= 0.0:
            dy = y_cur - y_prev
            if abs(dy) < 1e-12:
                frac_ground = 1.0
            else:
                frac_ground = y_prev / (y_prev - y_cur)
                if not (0.0 <= frac_ground <= 1.0):
                    frac_ground = None

        if frac_x is not None and (frac_ground is None or frac_x <= frac_ground):
            y_hit = y_prev + frac_x * (y_cur - y_prev)
            z_hit = z_prev + frac_x * (z_cur - z_prev)
            t_hit = t_cur + frac_x * (t_next - t_cur)
            return {
                "hit": True,
                "t_hit": float(t_hit),
                "y_hit": float(y_hit),
                "z_hit": float(z_hit),
                "range": float(x_target),
                "z_land": float(z_hit),
                "theta_deg": float(theta_deg),
                "alpha_deg": float(alpha_deg),
                "v0_actual": float(v0_actual),
            }

        if frac_ground is not None:
            x_land = x_prev + frac_ground * (x_cur - x_prev)
            z_land = z_prev + frac_ground * (z_cur - z_prev)
            return {
                "hit": False,
                "t_hit": None,
                "y_hit": None,
                "z_hit": None,
                "range": float(x_land),
                "z_land": float(z_land),
                "theta_deg": float(theta_deg),
                "alpha_deg": float(alpha_deg),
                "v0_actual": float(v0_actual),
            }

        # 提前终止：若速度方向已经明显反向且还没到目标截面，没有继续积分价值
        if state_next[3] <= 0.0 and state_next[0] < x_target:
            return {
                "hit": False,
                "t_hit": None,
                "y_hit": None,
                "z_hit": None,
                "range": float(state_next[0]),
                "z_land": float(state_next[2]),
                "theta_deg": float(theta_deg),
                "alpha_deg": float(alpha_deg),
                "v0_actual": float(v0_actual),
            }

        state = state_next
        t_cur = t_next

    return {
        "hit": False,
        "t_hit": None,
        "y_hit": None,
        "z_hit": None,
        "range": float(state[0]),
        "z_land": float(state[2]),
        "theta_deg": float(theta_deg),
        "alpha_deg": float(alpha_deg),
        "v0_actual": float(v0_actual),
    }


def simulate_trajectory(
        theta_deg: float,
        params: ProjectileParams,
        alpha_deg: float = 0.0,
        dt: float = 0.01,
        t_max: float = 200.0,
        x0: float = 0.0,
        y0: float = 0.0,
        z0: float = 0.0,
) -> dict:
    if dt <= 0:
        raise ValueError("dt must be positive.")
    if t_max <= 0:
        raise ValueError("t_max must be positive.")

    state, v0_actual = _make_initial_state(theta_deg, alpha_deg, params, x0, y0, z0)

    if state[4] <= 0 and y0 <= 0:
        return {
            "t": np.array([0.0]), "x": np.array([x0]), "y": np.array([y0]), "z": np.array([z0]),
            "theta_deg": float(theta_deg), "alpha_deg": float(alpha_deg), "range": float(x0),
            "drift_z": float(z0), "v0_actual": float(v0_actual)
        }

    t_list, x_list, y_list, z_list = [0.0], [state[0]], [state[1]], [state[2]]
    n_steps = int(np.ceil(t_max / dt))

    for i in range(n_steps):
        t = (i + 1) * dt
        state_next = rk4_step(state, dt, params)

        t_list.append(t)
        x_list.append(state_next[0])
        y_list.append(state_next[1])
        z_list.append(state_next[2])

        if state_next[1] <= 0.0:
            y_prev, y_now = state[1], state_next[1]
            if (y_prev - y_now) != 0:
                frac = y_prev / (y_prev - y_now)
                x_land = state[0] + frac * (state_next[0] - state[0])
                z_land = state[2] + frac * (state_next[2] - state[2])
            else:
                x_land, z_land = state_next[0], state_next[2]

            x_list[-1] = x_land
            y_list[-1] = 0.0
            z_list[-1] = z_land
            break

        state = state_next

    return {
        "t": np.asarray(t_list, dtype=np.float64),
        "x": np.asarray(x_list, dtype=np.float64),
        "y": np.asarray(y_list, dtype=np.float64),
        "z": np.asarray(z_list, dtype=np.float64),
        "theta_deg": float(theta_deg),
        "alpha_deg": float(alpha_deg),
        "range": float(x_list[-1]),
        "drift_z": float(z_list[-1]),
        "v0_actual": float(v0_actual)
    }


def interpolate_y_on_xgrid(traj: dict, x_grid: np.ndarray) -> np.ndarray:
    y_grid, _ = interpolate_yz_on_xgrid(traj, x_grid)
    return y_grid


def interpolate_yz_on_xgrid(traj: dict, x_grid: np.ndarray):
    x, y, z = traj["x"], traj["y"], traj["z"]
    if x_grid.ndim != 1:
        raise ValueError("x_grid must be 1D array.")

    inc_mask = np.ones_like(x, dtype=bool)
    inc_mask[1:] = x[1:] > x[:-1]
    x_m, y_m, z_m = x[inc_mask], y[inc_mask], z[inc_mask]

    if len(x_m) < 2:
        return np.full_like(x_grid, np.nan, dtype=np.float64), np.full_like(x_grid, np.nan, dtype=np.float64)

    y_grid = np.full_like(x_grid, np.nan, dtype=np.float64)
    z_grid = np.full_like(x_grid, np.nan, dtype=np.float64)
    valid = (x_grid >= x_m[0]) & (x_grid <= x_m[-1])

    if np.any(valid):
        y_grid[valid] = np.interp(x_grid[valid], x_m, y_m)
        z_grid[valid] = np.interp(x_grid[valid], x_m, z_m)

    return y_grid, z_grid


def time_and_y_at_x(traj: dict, x_target: float):
    x, y, z, t = traj["x"], traj["y"], traj["z"], traj["t"]

    if x_target < x[0] or x_target > x[-1] + 1e-9:
        return None, None, None

    idx = np.searchsorted(x, x_target, side="left")
    if idx == 0:
        return float(t[0]), float(y[0]), float(z[0])
    if idx >= len(x):
        return None, None, None

    x0, x1 = x[idx - 1], x[idx]
    t0, t1 = t[idx - 1], t[idx]
    y0, y1 = y[idx - 1], y[idx]
    z0, z1 = z[idx - 1], z[idx]

    if abs(x1 - x0) < 1e-12:
        return float(t1), float(y1), float(z1)

    alpha = (x_target - x0) / (x1 - x0)
    t_hit = t0 + alpha * (t1 - t0)
    y_hit = y0 + alpha * (y1 - y0)
    z_hit = z0 + alpha * (z1 - z0)
    return float(t_hit), float(y_hit), float(z_hit)
