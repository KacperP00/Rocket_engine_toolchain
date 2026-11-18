import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import config

def area_ratio(M, gamma):
    # Rownanie izentropowe A/A* f(M, gamma)
    term1 = 2.0 / (gamma + 1.0)
    term2 = 1.0 + 0.5 * (gamma - 1.0) * M**2
    exponent = (gamma + 1.0) / (2.0 * (gamma - 1.0))
    return (1.0 / M) * (term1 * term2)**exponent

def solve_mach(A_ratio, gamma, supersonic=False):
    # Numeryczne odwrocenie A/A* do M z zabezpieczeniem osobliwosci w gardle
    if A_ratio < 1.0001 or np.isnan(A_ratio):
        return 1.0
    guess = 2.5 if supersonic else 0.15
    func = lambda M: area_ratio(M, gamma) - A_ratio
    result = fsolve(func, guess)[0]
    return abs(result)

def generate_validation_plots(workdir, results_dir, chamber_data):
    print("--- [MODUŁ POSTPROCESSING] Generowanie wykresów wieloparametrowych ---")
    
    history_file = os.path.join(workdir, "history.csv")
    surface_file = os.path.join(workdir, "surface_flow.csv")
    volume_file = os.path.join(workdir, "restart.csv")
    
    # 1. Wykres zbieznosci
    if os.path.exists(history_file):
        df_hist = pd.read_csv(history_file)
        df_hist.columns = df_hist.columns.str.strip().str.replace('"', '')
        
        plt.figure(figsize=(10, 4))
        iter_col = "Inner_Iter" if "Inner_Iter" in df_hist.columns else "Iter"
        if "rms[Rho]" in df_hist.columns and iter_col in df_hist.columns:
            plt.plot(df_hist[iter_col], df_hist["rms[Rho]"], label="RMS Rho", color='black', linewidth=1.2)
        plt.title("Historia zbieżności solvera")
        plt.xlabel("Iteracja")
        plt.ylabel("Log10(RMS Rho)")
        plt.grid(True, which="both", ls="--")
        plt.legend()
        plt.savefig(os.path.join(results_dir, "plot_convergence.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. Walidacja 1D na osi dyszy wzgledem teorii izentropowej (Panel 2x2)
    if os.path.exists(volume_file) and os.path.exists(surface_file):
        df_vol = pd.read_csv(volume_file)
        df_vol.columns = df_vol.columns.str.strip().str.replace('"', '')
        
        df_wall = pd.read_csv(surface_file)
        df_wall.columns = df_wall.columns.str.strip().str.replace('"', '')
        df_wall = df_wall.sort_values(by="x").reset_index(drop=True)
        
        # Filtracja wylacznie wezlow na osi symetrii (y == 0)
        df_axis = df_vol[df_vol['y'] < 1e-4].sort_values(by="x").reset_index(drop=True)
        df_axis['r_wall'] = np.interp(df_axis['x'], df_wall['x'], df_wall['y'])
        
        gamma = chamber_data['gamma']
        p_c = chamber_data['p_c']
        T_c = chamber_data.get('T_c', 3440.0)
        R_gas = chamber_data.get('R', 405.87)
        rho_c = p_c / (R_gas * T_c)
        
        u = df_axis['Momentum_x'] / df_axis['Density']
        v = df_axis['Momentum_y'] / df_axis['Density']
        V_sq = u**2 + v**2
        
        p_stat = (gamma - 1.0) * (df_axis['Energy'] - 0.5 * df_axis['Density'] * V_sq)
        a = np.sqrt(gamma * p_stat / df_axis['Density'])
        
        df_axis['Pressure_CFD'] = p_stat
        df_axis['Mach_CFD'] = np.sqrt(V_sq) / a
        df_axis['Temperature_CFD'] = p_stat / (df_axis['Density'] * R_gas)
        df_axis['Density_CFD'] = df_axis['Density']
        
        # Lokalizacja gardla na podstawie minimalnego promienia scianki r_wall
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
            
        df_axis['Mach_Ana'] = M_ana
        df_axis['Pressure_Ana'] = p_ana
        df_axis['Temperature_Ana'] = T_ana
        df_axis['Density_Ana'] = rho_ana
        
        fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
        
        axs[0, 0].plot(df_axis["x"], df_axis["Mach_CFD"], 'r.', markersize=5, label="SU2 (Oś symetrii)")
        axs[0, 0].plot(df_axis["x"], df_axis["Mach_Ana"], 'k-', linewidth=1.5, label="Izentropowy 1D")
        axs[0, 0].set_ylabel("Liczba Macha [-]")
        axs[0, 0].grid(True)
        axs[0, 0].axvline(x=x_throat, color='gray', linestyle='--', alpha=0.5)
        axs[0, 0].legend()
        
        axs[0, 1].plot(df_axis["x"], df_axis["Pressure_CFD"], 'b.', markersize=5, label="SU2 (Oś symetrii)")
        axs[0, 1].plot(df_axis["x"], df_axis["Pressure_Ana"], 'k-', linewidth=1.5, label="Izentropowy 1D")
        axs[0, 1].set_ylabel("Ciśnienie [Pa]")
        axs[0, 1].grid(True)
        axs[0, 1].axvline(x=x_throat, color='gray', linestyle='--', alpha=0.5)
        axs[0, 1].legend()

        axs[1, 0].plot(df_axis["x"], df_axis["Temperature_CFD"], 'g.', markersize=5, label="SU2 (Oś symetrii)")
        axs[1, 0].plot(df_axis["x"], df_axis["Temperature_Ana"], 'k-', linewidth=1.5, label="Izentropowy 1D")
        axs[1, 0].set_ylabel("Temperatura [K]")
        axs[1, 0].set_xlabel("Oś X [m]")
        axs[1, 0].grid(True)
        axs[1, 0].axvline(x=x_throat, color='gray', linestyle='--', alpha=0.5)
        axs[1, 0].legend()
        
        axs[1, 1].plot(df_axis["x"], df_axis["Density_CFD"], 'm.', markersize=5, label="SU2 (Oś symetrii)")
        axs[1, 1].plot(df_axis["x"], df_axis["Density_Ana"], 'k-', linewidth=1.5, label="Izentropowy 1D")
        axs[1, 1].set_ylabel("Gęstość [kg/m^3]")
        axs[1, 1].set_xlabel("Oś X [m]")
        axs[1, 1].grid(True)
        axs[1, 1].axvline(x=x_throat, color='gray', linestyle='--', alpha=0.5)
        axs[1, 1].legend()
        
        plt.tight_layout()
        out_path = os.path.join(results_dir, "plot_nozzle_flow_2x2.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Panel wykresów walidacyjnych (2x2) zapisano w: {out_path}")
        
    print("-------------------------------------------------------------\n")