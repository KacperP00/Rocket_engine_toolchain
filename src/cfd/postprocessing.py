import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

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
    print("--- [MODUŁ POSTPROCESSING] Generowanie wykresów i metryk błędów ---")
    
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
    
    if os.path.exists(surface_file):
        df_surf = pd.read_csv(surface_file)
        df_surf.columns = df_surf.columns.str.strip().str.replace('"', '')
        
        if "x" in df_surf.columns:
            df_surf = df_surf.sort_values(by="x").reset_index(drop=True)
            
        if "Energy" in df_surf.columns and "Density" in df_surf.columns:
            gamma = chamber_data['gamma']
            p_c = chamber_data['p_c']
            
            # Przeliczanie CFD na zmienne prymitywne
            u = df_surf['Momentum_x'] / df_surf['Density']
            v = df_surf['Momentum_y'] / df_surf['Density']
            V_sq = u**2 + v**2
            
            p_stat = (gamma - 1.0) * (df_surf['Energy'] - 0.5 * df_surf['Density'] * V_sq)
            a = np.sqrt(gamma * p_stat / df_surf['Density'])
            
            df_surf['Pressure_CFD'] = p_stat
            df_surf['Mach_CFD'] = np.sqrt(V_sq) / a
            
            # Lokalizacja faktycznego gardła w siatce (minimum y)
            idx_throat = df_surf['y'].idxmin()
            x_throat = df_surf.loc[idx_throat, 'x']
            R_t_eff = df_surf['y'].min()
            
            M_ana = []
            p_ana = []
            
            # Obliczanie profilu analitycznego bez założeń o x=0
            for index, row in df_surf.iterrows():
                r_x = row['y']
                
                # Ustalenie reżimu na podstawie położenia względem faktycznego gardła
                supersonic = True if row['x'] > x_throat else False
                
                # Zabezpieczenie przed błędami numerycznymi precyzji zmiennoprzecinkowej
                A_ratio = max((r_x / R_t_eff)**2, 1.0)
                
                M_val = solve_mach(A_ratio, gamma, supersonic)
                p_val = p_c * (1.0 + 0.5 * (gamma - 1.0) * M_val**2)**(-gamma / (gamma - 1.0))
                
                M_ana.append(M_val)
                p_ana.append(p_val)
                
            df_surf['Mach_Ana'] = M_ana
            df_surf['Pressure_Ana'] = p_ana
            
            # Obliczanie błędów względnych [%]
            df_surf['Error_Mach_[%]'] = (abs(df_surf['Mach_CFD'] - df_surf['Mach_Ana']) / df_surf['Mach_Ana']) * 100.0
            df_surf['Error_Pressure_[%]'] = (abs(df_surf['Pressure_CFD'] - df_surf['Pressure_Ana']) / df_surf['Pressure_Ana']) * 100.0
            
            # Zapis metryk do czystego pliku CSV dla użytkownika
            metrics_cols = ['x', 'y', 'Mach_CFD', 'Mach_Ana', 'Error_Mach_[%]', 'Pressure_CFD', 'Pressure_Ana', 'Error_Pressure_[%]']
            metrics_df = df_surf[metrics_cols]
            
            csv_out = os.path.join(results_dir, "validation_metrics.csv")
            metrics_df.to_csv(csv_out, index=False)
            
            # Zestawienie wyników na wykresach
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            
            ax1.plot(df_surf["x"], df_surf["Mach_CFD"], 'ro', markersize=3, label="SU2 (CFD)")
            ax1.plot(df_surf["x"], df_surf["Mach_Ana"], 'k-', linewidth=2, label="Analityczny (Izentropowy)")
            ax1.set_ylabel("Liczba Macha [-]")
            ax1.grid(True)
            ax1.axvline(x=x_throat, color='gray', linestyle='--', label="Rzeczywiste gardło")
            ax1.legend()
            
            ax2.plot(df_surf["x"], df_surf["Pressure_CFD"], 'bo', markersize=3, label="SU2 (CFD)")
            ax2.plot(df_surf["x"], df_surf["Pressure_Ana"], 'k-', linewidth=2, label="Analityczny (Izentropowy)")
            ax2.set_ylabel("Ciśnienie statyczne [Pa]")
            ax2.set_xlabel("Współrzędna osiowa X [m]")
            ax2.grid(True)
            ax2.axvline(x=x_throat, color='gray', linestyle='--')
            ax2.legend()
            
            plt.suptitle("Walidacja 1D przepływu izentropowego: Teoria vs CFD")
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, "plot_nozzle_flow.png"))
            plt.close()
            
    print(f"Zapisano metryki zbieżności do: {os.path.join(results_dir, 'validation_metrics.csv')}")
    print("-------------------------------------------------------------\n")