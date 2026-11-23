import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.optimize import fsolve

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import config

def area_ratio(M, gamma):
    # Compute isentropic area ratio A/A* as a function of Mach number
    term1 = 2.0 / (gamma + 1.0)
    term2 = 1.0 + 0.5 * (gamma - 1.0) * M**2
    exponent = (gamma + 1.0) / (2.0 * (gamma - 1.0))
    return (1.0 / M) * (term1 * term2)**exponent

def solve_mach(A_ratio, gamma, supersonic=False):
    # Invert area-Mach relation numerically for subsonic or supersonic branch
    if A_ratio < 1.0001 or np.isnan(A_ratio):
        return 1.0
    guess = 2.5 if supersonic else 0.15
    func = lambda M: area_ratio(M, gamma) - A_ratio
    result = fsolve(func, guess)[0]
    return abs(result)

def generate_geometry_and_contour_plots(df_vol, df_wall, results_dir):
    # Generate 2D nozzle geometry, mesh topology, and flowfield contour plots
    xw = df_wall['x'].values
    yw = df_wall['y'].values

    triang = tri.Triangulation(df_vol['x'].values, df_vol['y'].values)
    tri_x = np.mean(df_vol['x'].values[triang.triangles], axis=1)
    tri_y = np.mean(df_vol['y'].values[triang.triangles], axis=1)
    wall_y_interp = np.interp(tri_x, xw, yw)
    triang.set_mask(tri_y > wall_y_interp + 1e-4)

    dr_dx = np.gradient(yw, xw)
    cos_a = 1.0 / np.sqrt(1.0 + dr_dx**2)
    sin_a = dr_dx * cos_a
    t_wall = getattr(config, 'LINER_THICKNESS', 0.001) + getattr(config, 'JACKET_THICKNESS', 0.003)
    xw_out = xw - t_wall * sin_a
    yw_out = yw + t_wall * cos_a

    # 1. Geometry and Mesh plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax1.fill_between(xw, 0, yw, color='#e6f2ff', alpha=0.6, label='Fluid Domain (CH4/O2 Products)')
    ax1.fill(np.concatenate([xw, xw_out[::-1]]), np.concatenate([yw, yw_out[::-1]]),
             color='#d95f02', alpha=0.85, label='Bimetallic Wall (CuCrZr 1.0 mm + Inconel 718 3.0 mm)')
    ax1.plot(xw, yw, 'r-', linewidth=2.0, label='MARKER_WALL (Isothermal T_w = 800 K)')
    ax1.plot([xw.min(), xw.min()], [0.0, yw[0]], 'g-', linewidth=3.0, label='MARKER_INLET')
    ax1.plot([xw.max(), xw.max()], [0.0, yw[-1]], 'm-', linewidth=3.0, label='MARKER_OUTLET')
    ax1.plot([xw.min(), xw.max()], [0.0, 0.0], 'k-.', linewidth=1.8, label='MARKER_SYM (Centerline)')
    ax1.axvline(x=0.0, color='gray', linestyle=':', label='Geometric Throat')
    ax1.set_ylabel('Radial Coordinate r [m]')
    ax1.set_title('Axisymmetric Rocket Nozzle Geometry, Boundary Markers, and Structural Shell')
    ax1.set_ylim(-0.01, yw.max() + 0.025)
    ax1.grid(True, ls='--', alpha=0.5)
    ax1.legend(loc='upper left', ncol=2, fontsize=8.5)

    ax2.triplot(triang, color='#2b5c8f', linewidth=0.3, alpha=0.7)
    ax2.plot(xw, yw, 'k-', linewidth=1.2)
    ax2.set_xlabel('Axial Coordinate x [m]')
    ax2.set_ylabel('Radial Coordinate r [m]')
    ax2.set_title(f'Computational Grid ({len(df_vol):,} Nodes, Structured Boundary Layer dy_1 = 0.35 um)')
    ax2.set_ylim(-0.005, yw.max() + 0.015)
    ax2.grid(True, ls='--', alpha=0.5)

    axins = inset_axes(ax2, width="32%", height="55%", loc='upper left', borderpad=1.8)
    throat_nodes = df_vol[(np.abs(df_vol['x']) < 0.008) & (df_vol['y'] > yw.min() - 0.0008)]
    axins.scatter(throat_nodes['x']*1000, throat_nodes['y']*1000, s=1.5, c='red')
    axins.plot(xw*1000, yw*1000, 'k-', linewidth=1.5)
    axins.set_xlim(-4.5, 4.5)
    axins.set_ylim((yw.min() - 0.0005)*1000, (yw.min() + 0.00015)*1000)
    axins.set_title('Throat Boundary Layer Zoom [mm]', fontsize=8)
    axins.tick_params(labelsize=7)
    axins.grid(True, ls=':', alpha=0.6)

    plt.savefig(os.path.join(results_dir, 'plot_geometry_mesh.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. 2D Mach and Temperature Contour plot
    fig, ax = plt.subplots(figsize=(12, 5.5))
    levels_mach = np.linspace(0.0, max(df_vol['Mach'].max(), 3.4), 36)
    cf1 = ax.tricontourf(triang, df_vol['Mach'].values, levels=levels_mach, cmap='viridis')

    triang_mirror = tri.Triangulation(df_vol['x'].values, -df_vol['y'].values, triang.triangles)
    triang_mirror.set_mask(tri_y > wall_y_interp + 1e-4)
    levels_temp = np.linspace(800, max(df_vol['Temperature'].max(), 3450), 36)
    cf2 = ax.tricontourf(triang_mirror, df_vol['Temperature'].values, levels=levels_temp, cmap='inferno')

    ax.plot(xw, yw, 'k-', linewidth=1.8)
    ax.plot(xw, -yw, 'k-', linewidth=1.8)
    ax.axhline(0.0, color='white', linestyle='--', linewidth=1.0)
    ax.axvline(0.0, color='white', linestyle=':', linewidth=0.8)

    cbar1 = fig.colorbar(cf1, ax=ax, location='right', pad=0.02, shrink=0.85)
    cbar1.set_label('Mach Number [-] (Upper Half)')
    cbar2 = fig.colorbar(cf2, ax=ax, location='left', pad=0.06, shrink=0.85)
    cbar2.set_label('Static Temperature [K] (Lower Half)')

    ax.set_xlabel('Axial Coordinate x [m]')
    ax.set_ylabel('Radial Coordinate r [m]')
    ax.set_title('2D Axisymmetric RANS Flowfield: Mach Number (Top) and Static Temperature (Bottom)')
    ax.set_ylim(-yw.max() - 0.01, yw.max() + 0.01)
    ax.grid(True, ls='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'plot_2d_contours.png'), dpi=300)
    plt.close()

def generate_validation_plots(workdir, results_dir, chamber_data):
    # Generate 1D/2D validation plots and convergence history in English
    print("--- [POSTPROCESSING MODULE] Generating multi-parameter validation plots ---")
    history_file = os.path.join(workdir, "history.csv")
    surface_file = os.path.join(workdir, "surface_flow.csv")
    volume_file = os.path.join(workdir, "restart.csv")

    if os.path.exists(history_file):
        df_hist = pd.read_csv(history_file)
        df_hist.columns = df_hist.columns.str.strip().str.replace('"', '')
        plt.figure(figsize=(10, 4))
        iter_col = "Inner_Iter" if "Inner_Iter" in df_hist.columns else "Iter"
        if "rms[Rho]" in df_hist.columns and iter_col in df_hist.columns:
            plt.plot(df_hist[iter_col], df_hist["rms[Rho]"], label="RMS[Rho]", color='black', linewidth=1.2)
        plt.title("Solver Convergence History")
        plt.xlabel("Iteration")
        plt.ylabel("Log10(RMS Residual)")
        plt.grid(True, which="both", ls="--")
        plt.legend()
        plt.savefig(os.path.join(results_dir, "plot_convergence.png"), dpi=300, bbox_inches='tight')
        plt.close()

    if os.path.exists(volume_file) and os.path.exists(surface_file):
        df_vol = pd.read_csv(volume_file)
        df_vol.columns = df_vol.columns.str.strip().str.replace('"', '')
        df_wall = pd.read_csv(surface_file)
        df_wall.columns = df_wall.columns.str.strip().str.replace('"', '')
        df_wall = df_wall.sort_values(by="x").reset_index(drop=True)

        gamma = chamber_data['gamma']
        p_c = chamber_data['p_c']
        T_c = chamber_data.get('T_c', 3434.81)
        R_gas = chamber_data.get('R', 405.87)
        rho_c = p_c / (R_gas * T_c)

        # Compute primitive flow variables across entire volume domain
        rho_v = df_vol['Density'].values
        u_v = df_vol['Momentum_x'].values / rho_v
        v_v = df_vol['Momentum_y'].values / rho_v
        v2_v = u_v**2 + v_v**2
        p_v = (gamma - 1.0) * (df_vol['Energy'].values - 0.5 * rho_v * v2_v)
        df_vol['Pressure'] = p_v
        df_vol['Temperature'] = p_v / (rho_v * R_gas)
        df_vol['Mach'] = np.sqrt(v2_v) / np.sqrt(np.maximum(gamma * p_v / rho_v, 1e-6))

        generate_geometry_and_contour_plots(df_vol, df_wall, results_dir)

        df_axis = df_vol[df_vol['y'] < 1e-4].sort_values(by="x").reset_index(drop=True)
        df_axis['r_wall'] = np.interp(df_axis['x'], df_wall['x'], df_wall['y'])

        idx_throat = df_axis['r_wall'].idxmin()
        x_throat = df_axis.loc[idx_throat, 'x']
        R_t_eff = df_axis['r_wall'].min()

        M_ana, p_ana, T_ana, rho_ana = [], [], [], []
        for _, row in df_axis.iterrows():
            r_x = row['r_wall']
            supersonic = row['x'] > x_throat
            A_ratio = max((r_x / R_t_eff)**2, 1.0)
            M_val = solve_mach(A_ratio, gamma, supersonic)
            term = 1.0 + 0.5 * (gamma - 1.0) * M_val**2
            M_ana.append(M_val)
            p_ana.append(p_c * term**(-gamma / (gamma - 1.0)))
            T_ana.append(T_c * term**(-1.0))
            rho_ana.append(rho_c * term**(-1.0 / (gamma - 1.0)))

        has_equil = 'exp_equil' in chamber_data
        if has_equil:
            eq = chamber_data['exp_equil']
            ith = eq['idx_throat']
            M_eq, p_eq, T_eq, rho_eq = [], [], [], []
            for _, row in df_axis.iterrows():
                ar = max((row['r_wall'] / R_t_eff)**2, 1.0)
                if row['x'] <= x_throat:
                    ar_sub = eq['A_ratio'][:ith+1][::-1]
                    M_eq.append(np.interp(ar, ar_sub, eq['M'][:ith+1][::-1]))
                    p_eq.append(np.interp(ar, ar_sub, eq['p'][:ith+1][::-1]))
                    T_eq.append(np.interp(ar, ar_sub, eq['T'][:ith+1][::-1]))
                    rho_eq.append(np.interp(ar, ar_sub, eq['rho'][:ith+1][::-1]))
                else:
                    ar_sup = eq['A_ratio'][ith:]
                    M_eq.append(np.interp(ar, ar_sup, eq['M'][ith:]))
                    p_eq.append(np.interp(ar, ar_sup, eq['p'][ith:]))
                    T_eq.append(np.interp(ar, ar_sup, eq['T'][ith:]))
                    rho_eq.append(np.interp(ar, ar_sup, eq['rho'][ith:]))

        fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharex=True)

        axs[0, 0].plot(df_axis["x"], df_axis["Mach"], 'r.', markersize=5, label="SU2 RANS (Centerline)")
        axs[0, 0].plot(df_axis["x"], M_ana, 'k-', linewidth=1.5, label="1D Isentropic (Frozen)")
        if has_equil:
            axs[0, 0].plot(df_axis["x"], M_eq, 'c--', linewidth=1.5, label="Cantera (Equilibrium)")
        axs[0, 0].set_ylabel("Mach Number [-]")
        axs[0, 0].grid(True, ls='--')
        axs[0, 0].axvline(x=x_throat, color='gray', linestyle='--', alpha=0.5)
        axs[0, 0].legend()

        axs[0, 1].plot(df_axis["x"], df_axis["Pressure"] / 1e5, 'b.', markersize=5, label="SU2 RANS (Centerline)")
        axs[0, 1].plot(df_axis["x"], np.array(p_ana) / 1e5, 'k-', linewidth=1.5, label="1D Isentropic (Frozen)")
        if has_equil:
            axs[0, 1].plot(df_axis["x"], np.array(p_eq) / 1e5, 'c--', linewidth=1.5, label="Cantera (Equilibrium)")
        axs[0, 1].set_ylabel("Static Pressure [bar]")
        axs[0, 1].grid(True, ls='--')
        axs[0, 1].axvline(x=x_throat, color='gray', linestyle='--', alpha=0.5)
        axs[0, 1].legend()

        axs[1, 0].plot(df_axis["x"], df_axis["Temperature"], 'g.', markersize=5, label="SU2 RANS (Centerline)")
        axs[1, 0].plot(df_axis["x"], T_ana, 'k-', linewidth=1.5, label="1D Isentropic (Frozen)")
        if has_equil:
            axs[1, 0].plot(df_axis["x"], T_eq, 'c--', linewidth=1.5, label="Cantera (Equilibrium)")
        axs[1, 0].set_ylabel("Static Temperature [K]")
        axs[1, 0].set_xlabel("Axial Coordinate x [m]")
        axs[1, 0].grid(True, ls='--')
        axs[1, 0].axvline(x=x_throat, color='gray', linestyle='--', alpha=0.5)
        axs[1, 0].legend()

        axs[1, 1].plot(df_axis["x"], df_axis["Density"], 'm.', markersize=5, label="SU2 RANS (Centerline)")
        axs[1, 1].plot(df_axis["x"], rho_ana, 'k-', linewidth=1.5, label="1D Isentropic (Frozen)")
        if has_equil:
            axs[1, 1].plot(df_axis["x"], rho_eq, 'c--', linewidth=1.5, label="Cantera (Equilibrium)")
        axs[1, 1].set_ylabel("Density [kg/m^3]")
        axs[1, 1].set_xlabel("Axial Coordinate x [m]")
        axs[1, 1].grid(True, ls='--')
        axs[1, 1].axvline(x=x_throat, color='gray', linestyle='--', alpha=0.5)
        axs[1, 1].legend()

        plt.tight_layout()
        out_path = os.path.join(results_dir, "plot_nozzle_flow_2x2.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Validation plots saved to: {results_dir}")