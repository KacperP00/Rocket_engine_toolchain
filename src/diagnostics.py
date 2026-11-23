import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.optimize import fsolve


def verify_mass_balance(workdir, log_file=None):
    # Check axisymmetric mass conservation across inlet, throat, and outlet sections
    flow_path = os.path.join(workdir, 'restart.csv')
    if not os.path.exists(flow_path):
        return None

    df = pd.read_csv(flow_path)
    df.columns = df.columns.str.strip().str.replace('"', '')

    x_in = df['x'].min()
    x_out = df['x'].max()

    inlet_nodes = df[np.abs(df['x'] - x_in) < 1e-5].sort_values('y')
    outlet_nodes = df[np.abs(df['x'] - x_out) < 1e-5].sort_values('y')

    def calc_mass_flow(nodes):
        # Integrate axisymmetric mass flux 2*pi*y*rho*u over radial coordinate
        y = nodes['y'].values
        mom_x = nodes['Momentum_x'].values
        return float(np.trapezoid(2.0 * np.pi * y * mom_x, y))

    m_in = calc_mass_flow(inlet_nodes)
    m_out = calc_mass_flow(outlet_nodes)

    # Interpolate axial momentum along exact vertical throat plane x = 0.0
    throat_band = df[np.abs(df['x']) < 0.015]
    if len(throat_band) > 10:
        surf_path = os.path.join(workdir, 'surface_flow.csv')
        if os.path.exists(surf_path):
            df_surf = pd.read_csv(surf_path)
            df_surf.columns = df_surf.columns.str.strip().str.replace('"', '')
            r_throat = float(df_surf['y'].min())
        else:
            r_throat = float(throat_band['y'].max())
        y_grid = np.linspace(0.0, r_throat, 200)
        pts = np.column_stack((throat_band['x'].values, throat_band['y'].values))
        mom_interp = griddata(pts, throat_band['Momentum_x'].values,
                              np.column_stack((np.zeros_like(y_grid), y_grid)),
                              method='linear')
        valid = ~np.isnan(mom_interp)
        m_throat = float(np.trapezoid(2.0 * np.pi * y_grid[valid] * mom_interp[valid], y_grid[valid])) if np.sum(valid) > 2 else 0.0
    else:
        m_throat = 0.0

    error = abs(m_in - m_out) / max(abs(m_in), 1e-12) * 100.0
    err_throat = abs(m_in - m_throat) / max(abs(m_in), 1e-12) * 100.0

    msg = (
        f"[DIAGNOSTICS] Mass balance -> Inlet: {m_in:.4f} kg/s | "
        f"Throat: {m_throat:.4f} kg/s (err {err_throat:.2f}%) | "
        f"Outlet: {m_out:.4f} kg/s | Outlet Error: {error:.2f}%"
    )
    print(msg)
    if log_file:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(msg + "\n")
    return m_in, m_out, error


def verify_y_plus(workdir, log_file=None, mu_val=3.34e-5):
    # Compute wall dimensionless distance y+ from first interior cell layer
    surf_path = os.path.join(workdir, 'surface_flow.csv')
    vol_path = os.path.join(workdir, 'restart.csv')
    if not os.path.exists(surf_path) or not os.path.exists(vol_path):
        return None

    df_surf = pd.read_csv(surf_path)
    df_surf.columns = df_surf.columns.str.strip().str.replace('"', '')

    if 'Y_Plus' in df_surf.columns:
        y_plus_vals = df_surf['Y_Plus'].values
    else:
        df_vol = pd.read_csv(vol_path)
        df_vol.columns = df_vol.columns.str.strip().str.replace('"', '')

        vel_mag = np.sqrt(df_vol['Momentum_x']**2 + df_vol['Momentum_y']**2) / df_vol['Density']
        interior = df_vol[vel_mag > 1e-3].copy()
        interior['U'] = vel_mag[vel_mag > 1e-3]

        y_plus_list = []
        int_x = interior['x'].values
        int_y = interior['y'].values
        int_u = interior['U'].values
        for _, w_row in df_surf.iterrows():
            dist = np.hypot(int_x - w_row['x'], int_y - w_row['y'])
            idx_min = np.argmin(dist)
            dn = dist[idx_min]
            u1 = int_u[idx_min]
            rho_w = w_row['Density']
            y_plus_list.append(np.sqrt(rho_w * u1 * dn / mu_val))
        y_plus_vals = np.array(y_plus_list)

    y_max = float(np.max(y_plus_vals))
    y_mean = float(np.mean(y_plus_vals))
    msg = f"[DIAGNOSTICS] Wall mesh -> y+ max: {y_max:.3f} | y+ mean: {y_mean:.3f}"

    if y_max > 5.0:
        msg += "\n[CRITICAL] y+ exceeds viscous sublayer (y+ > 5). Reduce bl_size in gmsh_generator.py."
    elif y_max > 1.0:
        msg += "\n[INFO] Local throat y+ slightly above 1.0 (still inside viscous sublayer y+ < 5)."

    print(msg)
    if log_file:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(msg + "\n")
    return y_max, y_mean


def plot_convergence(workdir, log_file=None):
    # Plot RMS residual convergence history from SU2 history.csv
    hist_path = os.path.join(workdir, 'history.csv')
    if not os.path.exists(hist_path):
        return

    df = pd.read_csv(hist_path)
    df.columns = df.columns.str.strip().str.replace('"', '')
    iter_col = 'Inner_Iter' if 'Inner_Iter' in df.columns else 'Iter'

    plt.figure(figsize=(10, 5.5))
    if 'rms[Rho]' in df.columns:
        plt.plot(df[iter_col], df['rms[Rho]'], label='rms[Rho]', color='blue', linewidth=1.5)
    if 'rms[RhoE]' in df.columns:
        plt.plot(df[iter_col], df['rms[RhoE]'], label='rms[RhoE]', color='red', linewidth=1.2, alpha=0.85)
    if 'rms[k]' in df.columns:
        plt.plot(df[iter_col], df['rms[k]'], label='rms[k]', color='orange', linewidth=1.1, alpha=0.8)
    if 'rms[w]' in df.columns:
        plt.plot(df[iter_col], df['rms[w]'], label='rms[w] (omega)', color='green', linewidth=1.1, alpha=0.8)
    elif 'rms[omega]' in df.columns:
        plt.plot(df[iter_col], df['rms[omega]'], label='rms[omega]', color='green', linewidth=1.1, alpha=0.8)

    plt.title('SU2 Solver Convergence History (RMS Residuals)')
    plt.xlabel('Inner Iteration')
    plt.ylabel('Log10(RMS Residual)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    plot_path = os.path.join(workdir, 'convergence.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    msg = f"[DIAGNOSTICS] Convergence plot saved: {plot_path}"
    print(msg)
    if log_file:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(msg + "\n")


def verify_momentum_balance(workdir, cantera_data, log_file=None):
    # Verify integral axial momentum balance between boundaries and nozzle wall
    flow_path = os.path.join(workdir, 'restart.csv')
    surf_path = os.path.join(workdir, 'surface_flow.csv')
    if not os.path.exists(flow_path) or not os.path.exists(surf_path):
        return None

    df_vol = pd.read_csv(flow_path)
    df_vol.columns = df_vol.columns.str.strip().str.replace('"', '')
    df_wall = pd.read_csv(surf_path)
    df_wall.columns = df_wall.columns.str.strip().str.replace('"', '')
    df_wall = df_wall.sort_values('x').reset_index(drop=True)

    gamma = cantera_data['gamma']

    def section_thrust(nodes):
        # Integrate axisymmetric stream thrust (rho*u^2 + p)*2*pi*y dy
        y = nodes['y'].values
        rho = nodes['Density'].values
        u = (nodes['Momentum_x'] / rho).values
        v = (nodes['Momentum_y'] / rho).values
        p = (gamma - 1.0) * (nodes['Energy'].values - 0.5 * rho * (u**2 + v**2))
        return float(np.trapezoid((rho * u**2 + p) * 2.0 * np.pi * y, y))

    x_in, x_out = df_vol['x'].min(), df_vol['x'].max()
    inlet_nodes = df_vol[np.abs(df_vol['x'] - x_in) < 1e-5].sort_values('y')
    outlet_nodes = df_vol[np.abs(df_vol['x'] - x_out) < 1e-5].sort_values('y')

    f_in = section_thrust(inlet_nodes)
    f_out = section_thrust(outlet_nodes)

    y_w = df_wall['y'].values
    rho_w = df_wall['Density'].values
    v2_w = (df_wall['Momentum_x'] / rho_w)**2 + (df_wall['Momentum_y'] / rho_w)**2
    p_w = (gamma - 1.0) * (df_wall['Energy'].values - 0.5 * rho_w * v2_w)
    f_wall_x = float(np.trapezoid(p_w * 2.0 * np.pi * y_w, y_w))

    f_control_vol = f_in + f_wall_x
    err_mom = abs(f_out - f_control_vol) / max(abs(f_out), 1e-12) * 100.0

    msg = (
        f"[CFD VALIDATION] Vacuum thrust (exit): {f_out/1000.0:.3f} kN | "
        f"Control volume (inlet + wall): {f_control_vol/1000.0:.3f} kN | "
        f"Momentum error: {err_mom:.2f}%"
    )
    print(msg)
    if log_file:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(msg + "\n")
    return f_out / 1000.0, f_control_vol / 1000.0, err_mom


def verify_heat_transfer(workdir, results_dir, cantera_data, wall_temp=None, log_file=None):
    # Compute boundary-layer total heat flux (laminar + SST turbulent) and validate against Bartz
    import config
    if wall_temp is None:
        wall_temp = getattr(config, 'WALL_TEMP', 800.0)

    surf_path = os.path.join(workdir, 'surface_flow.csv')
    vol_path = os.path.join(workdir, 'restart.csv')
    if not os.path.exists(surf_path) or not os.path.exists(vol_path):
        return None

    df_surf = pd.read_csv(surf_path)
    df_surf.columns = df_surf.columns.str.strip().str.replace('"', '')
    df_surf = df_surf.sort_values('x').reset_index(drop=True)

    df_vol = pd.read_csv(vol_path)
    df_vol.columns = df_vol.columns.str.strip().str.replace('"', '')

    gamma = cantera_data['gamma']
    R_gas = cantera_data['R']
    mu_c = cantera_data['mu']
    Pr_c = cantera_data['Pr']
    Pr_turb = 0.90
    T_c = cantera_data['T_c']
    p_c = cantera_data['p_c']
    cp_c = cantera_data.get('cp', gamma * R_gas / (gamma - 1.0))
    k_lam_wall = cantera_data.get('k_wall', mu_c * cp_c / Pr_c)

    rho_v = df_vol['Density'].values
    v2_v = (df_vol['Momentum_x'].values / rho_v)**2 + (df_vol['Momentum_y'].values / rho_v)**2
    p_v = (gamma - 1.0) * (df_vol['Energy'].values - 0.5 * rho_v * v2_v)
    df_vol['Temp'] = p_v / (rho_v * R_gas)
    df_vol['Temp_Tot'] = df_vol['Temp'] + 0.5 * v2_v / cp_c
    df_vol['mu_t'] = rho_v * df_vol['Turb_Kin_Energy'].values / np.maximum(df_vol['Omega'].values, 1e-10)

    vol_x = df_vol['x'].values
    vol_y = df_vol['y'].values

    q_cfd = []
    for _, w_row in df_surf.iterrows():
        xw, yw = w_row['x'], w_row['y']
        tw_act = (gamma - 1.0) * w_row['Energy'] / (w_row['Density'] * R_gas)

        dist = np.hypot(vol_x - xw, vol_y - yw)
        mask_near = (dist > 1e-7) & (dist < 0.0032)
        if not np.any(mask_near):
            q_cfd.append(0.0)
            continue

        near = df_vol[mask_near].copy()
        near['dn'] = dist[mask_near]
        near = near.sort_values('dn')
        r1 = near.iloc[0]
        nx = (r1['x'] - xw) / r1['dn']
        ny = (r1['y'] - yw) / r1['dn']

        # Extract boundary-layer nodes aligned along wall-normal ray
        cross = np.abs((near['x'] - xw) * ny - (near['y'] - yw) * nx)
        ray = near[cross < 1e-5].sort_values('dn').reset_index(drop=True)
        if len(ray) < 2:
            q_cfd.append(0.0)
            continue

        dn_ray = np.insert(ray['dn'].values, 0, 0.0)
        Ttot_ray = np.insert(ray['Temp_Tot'].values, 0, tw_act)
        mut_ray = np.insert(ray['mu_t'].values, 0, 0.0)

        keff_ray = k_lam_wall + mut_ray * cp_c / Pr_turb
        q_tot_ray = keff_ray * np.gradient(Ttot_ray, dn_ray)

        log_mask = (dn_ray >= 1.5e-4) & (dn_ray <= 2.5e-3)
        q_val = np.max(q_tot_ray[log_mask]) if np.any(log_mask) else np.max(q_tot_ray)
        q_cfd.append(max(float(q_val), 0.0))

    q_cfd = np.array(q_cfd)

    # Filter numerical thermal singularity at inlet-wall corner
    x_cutoff = df_surf['x'].min() + 0.025
    idx_ref = np.argmin(np.abs(df_surf['x'].values - x_cutoff))
    q_cfd[df_surf['x'].values < x_cutoff] = q_cfd[idx_ref]

    # 1D Bartz semi-empirical correlation along nozzle contour
    r_w = df_surf['y'].values
    r_t = float(np.min(r_w))
    D_t = 2.0 * r_t
    c_star = np.sqrt(R_gas * T_c / gamma) * ((gamma + 1.0) / 2.0)**((gamma + 1.0) / (2.0 * (gamma - 1.0)))
    r_rec = Pr_c**(1.0 / 3.0)

    def area_eq(M, ar):
        return (1.0 / M) * ((2.0 / (gamma + 1.0)) * (1.0 + 0.5 * (gamma - 1.0) * M**2))**((gamma + 1.0) / (2.0 * (gamma - 1.0))) - ar

    x_throat = df_surf.loc[df_surf['y'].idxmin(), 'x']
    q_bartz = []
    for _, row in df_surf.iterrows():
        ar = max((row['y'] / r_t)**2, 1.0)
        if ar < 1.0001:
            M_loc = 1.0
        else:
            guess = 2.5 if row['x'] > x_throat else 0.15
            M_loc = abs(fsolve(area_eq, guess, args=(ar,))[0])
        term = 1.0 + 0.5 * (gamma - 1.0) * M_loc**2
        sigma = (0.5 * (wall_temp / T_c) * term + 0.5)**(-0.68) * (term)**(-0.12)
        h_g = (0.026 / (D_t**0.2)) * ((mu_c**0.2) * cp_c / (Pr_c**0.6)) * ((p_c / c_star)**0.8) * ((1.0 / ar)**0.9) * sigma
        T_aw = T_c * (1.0 + r_rec * 0.5 * (gamma - 1.0) * M_loc**2) / term
        q_bartz.append(h_g * (T_aw - wall_temp))

    q_bartz = np.array(q_bartz)

    # Integrate total wall heat load Q_total [MW]
    ds = np.hypot(np.diff(df_surf['x'].values), np.diff(r_w))
    r_mid = 0.5 * (r_w[:-1] + r_w[1:])
    q_mid = 0.5 * (q_cfd[:-1] + q_cfd[1:])
    Q_total_MW = float(np.sum(q_mid * 2.0 * np.pi * r_mid * ds)) / 1e6

    df_surf['Heat_Flux_CFD'] = q_cfd
    df_surf['Heat_Flux_Bartz'] = q_bartz
    df_surf.to_csv(os.path.join(workdir, 'wall_thermal_loads.csv'), index=False)

    plt.figure(figsize=(10, 4.2))
    plt.plot(df_surf['x'], q_bartz / 1e6, 'k--', linewidth=1.6, label='Bartz Correlation (1D Analytical) [MW/m^2]')
    plt.plot(df_surf['x'], q_cfd / 1e6, 'r-', linewidth=2.0, label='SU2 RANS SST (Boundary Layer Total Flux) [MW/m^2]')
    plt.axvline(x=x_throat, color='gray', linestyle=':', label='Nozzle Throat')
    plt.xlabel('Axial Coordinate x [m]')
    plt.ylabel('Heat Flux [MW/m^2]')
    plt.title(f'Wall Convective Heat Transfer Validation (T_w = {wall_temp:.0f} K)')
    plt.grid(True, ls='--')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'plot_wall_heat_flux.png'), dpi=300)
    plt.close()

    throat_mask = np.abs(df_surf['x'].values) < 0.025
    q_th_cfd = float(np.max(q_cfd[throat_mask]))
    q_th_bartz = float(np.max(q_bartz[throat_mask]))
    err_q = abs(q_th_cfd - q_th_bartz) / q_th_bartz * 100.0
    msg = (
        f"[THERMAL VALIDATION] Throat peak heat flux -> CFD: {q_th_cfd/1e6:.2f} MW/m^2 | "
        f"Bartz: {q_th_bartz/1e6:.2f} MW/m^2 | Error: {err_q:.2f}% | Total Q: {Q_total_MW:.3f} MW"
    )
    print(msg)
    if log_file:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(msg + "\n")
    return q_cfd


def verify_flow_separation(workdir, results_dir, cantera_data, log_file=None):
    # Detect boundary-layer separation point at sea level and validate against Schmucker criterion
    import config
    p_amb = getattr(config, 'AMBIENT_PRESSURE', 0.0)
    if p_amb <= 1.0:
        return None

    surf_path = os.path.join(workdir, 'surface_flow.csv')
    if not os.path.exists(surf_path):
        return None

    df_surf = pd.read_csv(surf_path)
    df_surf.columns = df_surf.columns.str.strip().str.replace('"', '')
    df_surf = df_surf.sort_values('x').reset_index(drop=True)

    gamma = cantera_data['gamma']
    p_c = cantera_data['p_c']
    rho = df_surf['Density'].values
    v2 = (df_surf['Momentum_x'].values / rho)**2 + (df_surf['Momentum_y'].values / rho)**2
    p_wall = (gamma - 1.0) * (df_surf['Energy'].values - 0.5 * rho * v2)
    x_w = df_surf['x'].values
    r_w = df_surf['y'].values

    # Separation point in CFD: static pressure minimum upstream of shock pressure rise
    div_mask = x_w > 0.02
    idx_sep_cfd = int(np.argmin(p_wall[div_mask]) + np.sum(~div_mask))
    x_sep_cfd = float(x_w[idx_sep_cfd])
    p_sep_cfd = float(p_wall[idx_sep_cfd])

    # Schmucker empirical separation criterion: p_sep / p_a = (1.88 * M_sep - 1)^(-0.64)
    def area_ratio(M):
        return (1.0 / M) * ((2.0 / (gamma + 1.0)) * (1.0 + 0.5 * (gamma - 1.0) * M**2))**((gamma + 1.0) / (2.0 * (gamma - 1.0)))

    def p_iso(M):
        return p_c * (1.0 + 0.5 * (gamma - 1.0) * M**2)**(-gamma / (gamma - 1.0))

    M_sch = float(fsolve(lambda M: (p_iso(M) / p_amb) - (1.88 * M - 1.0)**(-0.64), 2.8)[0])
    ar_sch = area_ratio(M_sch)
    r_t = float(np.min(r_w))
    r_sch = r_t * np.sqrt(ar_sch)
    x_sep_sch = float(np.interp(r_sch, r_w[div_mask], x_w[div_mask]))

    plt.figure(figsize=(10, 4.2))
    plt.plot(x_w, p_wall / 1e5, 'b-', linewidth=2.0, label='CFD Wall Static Pressure [bar]')
    plt.axhline(y=p_amb / 1e5, color='k', linestyle=':', label='Ambient Sea-Level Pressure (1.013 bar)')
    plt.axvline(x=x_sep_cfd, color='r', linestyle='--', label=f'CFD Separation Point (x = {x_sep_cfd:.3f} m)')
    plt.axvline(x=x_sep_sch, color='g', linestyle='-.', label=f'Schmucker Criterion (x = {x_sep_sch:.3f} m)')
    plt.xlabel('Axial Coordinate x [m]')
    plt.ylabel('Wall Static Pressure [bar]')
    plt.title('Overexpanded Flow Separation & Incipient Shock Validation at Sea Level')
    plt.grid(True, ls='--')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'plot_flow_separation.png'), dpi=300)
    plt.close()

    err_sep = abs(x_sep_cfd - x_sep_sch) / x_sep_sch * 100.0
    msg = (
        f"[SEPARATION VALIDATION] Separation point -> CFD: x = {x_sep_cfd:.3f} m (p = {p_sep_cfd:.0f} Pa) | "
        f"Schmucker: x = {x_sep_sch:.3f} m | Position Error: {err_sep:.2f}%"
    )
    print(msg)
    if log_file:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(msg + "\n")
    return x_sep_cfd