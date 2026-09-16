import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import config

def area_ratio(M, gamma):
    # Równanie izentropowe A/A* f(M, gamma)
    term1 = 2.0 / (gamma + 1.0)
    term2 = 1.0 + 0.5 * (gamma - 1.0) * M**2
    exponent = (gamma + 1.0) / (2.0 * (gamma - 1.0))
    return (1.0 / M) * (term1 * term2)**exponent

def solve_mach(A_ratio, gamma, supersonic=False):
    # Numeryczne odwrócenie A/A* do M. Zabezpieczenie osobliwości w gardle.
    if A_ratio < 1.0001:
        return 1.0
    
    guess = 2.0 if supersonic else 0.2
    func = lambda M: area_ratio(M, gamma) - A_ratio
    result = fsolve(func, guess)[0]
    return abs(result)

def generate_validation_plots(workdir, results_dir, chamber_data):
    print("--- [MODUŁ POSTPROCESSING] Generowanie wykresów wieloparametrowych ---")
    
    history_file = os.path.join(workdir, "history.csv")
    surface_file = os.path.join(workdir, "surface_flow.csv")
    
    # 1. Wykres zbieżności
    if os.path.exists(history_file):
        df_hist = pd.read_csv(history_file)
        df_hist.columns = df_hist.columns.str.strip().str.replace('"', '')
        
        plt.figure(figsize=(10, 4))
        iter_col = "Inner_Iter" if "Inner_Iter" in df_hist.columns else "Iter"
        
        if "rms[Rho]" in df_hist.columns and iter_col in df_hist.columns:
            plt.plot(df_hist[iter_col], df_hist["rms[Rho]"], label="RMS Rho", color='black', linewidth=1)
            
        plt.title("Historia zbieżności solvera")
        plt.xlabel("Iteracja")
        plt.ylabel("Log10(RMS Rho)")
        plt.grid(True, which="both", ls="--")
        plt.savefig(os.path.join(results_dir, "plot_convergence.png"))
        plt.close()
    
    # 2. Walidacja 1D (Panel 2x2)
    if os.path.exists(surface_file):
        df_surf = pd.read_csv(surface_file)
        df_surf.columns = df_surf.columns.str.strip().str.replace('"', '')
        
        if "x" in df_surf.columns:
            df_surf = df_surf.sort_values(by="x").reset_index(drop=True)
            
        if "Energy" in df_surf.columns and "Density" in df_surf.columns:
            # Pobranie parametrów termodynamicznych komory
            gamma = chamber_data['gamma']
            p_c = chamber_data['p_c']
            T_c = chamber_data.get('T_c', 3000.0)
            R_gas = chamber_data.get('R', 287.05)
            rho_c = p_c / (R_gas * T_c)
            
            # Przeliczanie zmiennych stanu SU2 (CFD) na prymitywne
            u = df_surf['Momentum_x'] / df_surf['Density']
            v = df_surf['Momentum_y'] / df_surf['Density']
            V_sq = u**2 + v**2
            
            p_stat = (gamma - 1.0) * (df_surf['Energy'] - 0.5 * df_surf['Density'] * V_sq)
            a = np.sqrt(gamma * p_stat / df_surf['Density'])
            
            df_surf['Pressure_CFD'] = p_stat
            df_surf['Mach_CFD'] = np.sqrt(V_sq) / a
            df_surf['Temperature_CFD'] = p_stat / (df_surf['Density'] * R_gas)
            df_surf['Density_CFD'] = df_surf['Density']
            
            # Lokalizacja przewężenia (gardła)
            idx_throat = df_surf['y'].idxmin()
            x_throat = df_surf.loc[idx_throat, 'x']
            R_t_eff = df_surf['y'].min()
            
            M_ana, p_ana, T_ana, rho_ana = [], [], [], []
            
            # Obliczanie izentropowych profili teoretycznych
            for index, row in df_surf.iterrows():
                r_x = row['y']
                supersonic = row['x'] > x_throat
                A_ratio = max((r_x / R_t_eff)**2, 1.0)
                
                M_val = solve_mach(A_ratio, gamma, supersonic)
                term = 1.0 + 0.5 * (gamma - 1.0) * M_val**2
                
                M_ana.append(M_val)
                p_ana.append(p_c * term**(-gamma / (gamma - 1.0)))
                T_ana.append(T_c * term**(-1.0))
                rho_ana.append(rho_c * term**(-1.0 / (gamma - 1.0)))
                
            df_surf['Mach_Ana'] = M_ana
            df_surf['Pressure_Ana'] = p_ana
            df_surf['Temperature_Ana'] = T_ana
            df_surf['Density_Ana'] = rho_ana
            
            # Generowanie panelu wykresów 2x2
            fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
            
            # (0,0) Liczba Macha
            axs[0, 0].plot(df_surf["x"], df_surf["Mach_CFD"], 'r.', markersize=4, label="SU2 (CFD)")
            axs[0, 0].plot(df_surf["x"], df_surf["Mach_Ana"], 'k-', linewidth=1.5, label="Izentropowy")
            axs[0, 0].set_ylabel("Liczba Macha [-]")
            axs[0, 0].grid(True)
            axs[0, 0].axvline(x=x_throat, color='gray', linestyle='--', alpha=0.5)
            axs[0, 0].legend()
            
            # (0,1) Ciśnienie
            axs[0, 1].plot(df_surf["x"], df_surf["Pressure_CFD"], 'b.', markersize=4, label="SU2 (CFD)")
            axs[0, 1].plot(df_surf["x"], df_surf["Pressure_Ana"], 'k-', linewidth=1.5, label="Izentropowy")
            axs[0, 1].set_ylabel("Ciśnienie [Pa]")
            axs[0, 1].grid(True)
            axs[0, 1].axvline(x=x_throat, color='gray', linestyle='--', alpha=0.5)
            axs[0, 1].legend()

            # (1,0) Temperatura
            axs[1, 0].plot(df_surf["x"], df_surf["Temperature_CFD"], 'g.', markersize=4, label="SU2 (CFD)")
            axs[1, 0].plot(df_surf["x"], df_surf["Temperature_Ana"], 'k-', linewidth=1.5, label="Izentropowy")
            axs[1, 0].set_ylabel("Temperatura [K]")
            axs[1, 0].set_xlabel("Oś X [m]")
            axs[1, 0].grid(True)
            axs[1, 0].axvline(x=x_throat, color='gray', linestyle='--', alpha=0.5)
            axs[1, 0].legend()
            
            # (1,1) Gęstość
            axs[1, 1].plot(df_surf["x"], df_surf["Density_CFD"], 'm.', markersize=4, label="SU2 (CFD)")
            axs[1, 1].plot(df_surf["x"], df_surf["Density_Ana"], 'k-', linewidth=1.5, label="Izentropowy")
            axs[1, 1].set_ylabel("Gęstość [kg/m^3]")
            axs[1, 1].set_xlabel("Oś X [m]")
            axs[1, 1].grid(True)
            axs[1, 1].axvline(x=x_throat, color='gray', linestyle='--', alpha=0.5)
            axs[1, 1].legend()
            
            plt.tight_layout()
            out_path = os.path.join(results_dir, "plot_nozzle_flow_2x2.png")
            plt.savefig(out_path)
            plt.close()
            
            print(f"Panel wykresów walidacyjnych (2x2) zapisano w: {out_path}")
            
    print("-------------------------------------------------------------\n")