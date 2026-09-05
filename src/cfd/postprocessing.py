import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import math

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import config

def area_ratio(M, gamma):
    # Wyznaczanie stosunku A/A* na podstawie liczby Macha i gamma
    term1 = 2.0 / (gamma + 1.0)
    term2 = 1.0 + 0.5 * (gamma - 1.0) * M**2
    exponent = (gamma + 1.0) / (2.0 * (gamma - 1.0))
    return (1.0 / M) * (term1 * term2)**exponent

def solve_mach(A_ratio, gamma, supersonic=False):
    # Numeryczne rozwiązywanie równania izentropowego dla liczby Macha
    if A_ratio < 1.0001:
        return 1.0
    
    guess = 2.0 if supersonic else 0.2
    func = lambda M: area_ratio(M, gamma) - A_ratio
    return fsolve(func, guess)[0]

def generate_validation_plots(workdir, results_dir, chamber_data):
    print("--- [MODUŁ POSTPROCESSING] Generowanie wykresów walidacyjnych z benchmarkiem ---")
    
    history_file = os.path.join(workdir, "history.csv")
    surface_file = os.path.join(workdir, "surface_flow.csv")
    
    # Wykres zbieżności solvera
    if os.path.exists(history_file):
        df_hist = pd.read_csv(history_file)
        df_hist.columns = df_hist.columns.str.strip().str.replace('"', '')
        
        plt.figure(figsize=(10, 5))
        if "rms[Rho]" in df_hist.columns:
            plt.plot(df_hist["Inner_Iter"], df_hist["rms[Rho]"], label="RMS Rho")
            
        plt.title("Historia zbieżności")
        plt.xlabel("Iteracja")
        plt.ylabel("Residuum gęstości [Log10]")
        plt.grid(True, which="both", ls="--")
        plt.savefig(os.path.join(results_dir, "plot_convergence.png"))
        plt.close()
    
    # Wykresy przepływowe (Walidacja CFD vs Analityka)
    if os.path.exists(surface_file):
        df_surf = pd.read_csv(surface_file)
        df_surf.columns = df_surf.columns.str.strip().str.replace('"', '')
        
        if "x" in df_surf.columns:
            df_surf = df_surf.sort_values(by="x")
            
        # Obliczanie zmiennych prymitywnych z zachowawczych (SU2 default)
        if "Energy" in df_surf.columns and "Density" in df_surf.columns:
            gamma = chamber_data['gamma']
            
            # Wektory prędkości z wektorów pędu
            u = df_surf['Momentum_x'] / df_surf['Density']
            v = df_surf['Momentum_y'] / df_surf['Density']
            V_sq = u**2 + v**2
            
            # Przekształcenie równania stanu gazu doskonałego
            p_stat = (gamma - 1.0) * (df_surf['Energy'] - 0.5 * df_surf['Density'] * V_sq)
            a = np.sqrt(gamma * p_stat / df_surf['Density'])
            
            df_surf['Pressure'] = p_stat
            df_surf['Mach'] = np.sqrt(V_sq) / a
            
        # Generowanie rozwiązania analitycznego dla geometrii dyszy
        gamma = chamber_data['gamma']
        p_c = chamber_data['p_c']
        
        R_t = config.THROAT_RADIUS
        R_e = R_t * math.sqrt(config.EXPANSION_RATIO)
        R_c = R_t * 2.0
        L_c = 0.1
        L_d = config.NOZZLE_LENGTH
        
        x_ana = np.linspace(-L_c, L_d, 200)
        M_ana = []
        p_ana = []
        
        for x in x_ana:
            if x <= 0:
                r_x = R_t - (x / L_c) * (R_c - R_t)
                supersonic = False
            else:
                r_x = R_t + (x / L_d) * (R_e - R_t)
                supersonic = True
                
            A_ratio = (r_x / R_t)**2
            M_val = solve_mach(A_ratio, gamma, supersonic)
            p_val = p_c * (1.0 + 0.5 * (gamma - 1.0) * M_val**2)**(-gamma / (gamma - 1.0))
            
            M_ana.append(M_val)
            p_ana.append(p_val)
            
        # Zestawienie wyników na wykresach
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        if "Mach" in df_surf.columns:
            ax1.plot(df_surf["x"], df_surf["Mach"], 'ro', markersize=3, label="SU2 (CFD)")
            ax1.plot(x_ana, M_ana, 'k-', linewidth=2, label="Analityczny (Izentropowy)")
            ax1.set_ylabel("Liczba Macha [-]")
            ax1.grid(True)
            ax1.axvline(x=0, color='gray', linestyle='--')
            ax1.legend()
            
        if "Pressure" in df_surf.columns:
            ax2.plot(df_surf["x"], df_surf["Pressure"], 'bo', markersize=3, label="SU2 (CFD)")
            ax2.plot(x_ana, p_ana, 'k-', linewidth=2, label="Analityczny (Izentropowy)")
            ax2.set_ylabel("Ciśnienie statyczne [Pa]")
            ax2.set_xlabel("Współrzędna osiowa X [m]")
            ax2.grid(True)
            ax2.axvline(x=0, color='gray', linestyle='--')
            ax2.legend()
            
        plt.suptitle("Walidacja 1D przepływu izentropowego: Teoria vs CFD")
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "plot_nozzle_flow.png"))
        plt.close()
        
    print(f"Zapisano wykresy walidacyjne z naniesionym benchmarkiem w: {results_dir}")