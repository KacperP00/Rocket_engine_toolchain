import os
import pandas as pd
import matplotlib.pyplot as plt

def generate_validation_plots(workdir, results_dir):
    print("--- [MODUŁ POSTPROCESSING] Generowanie wykresów walidacyjnych ---")
    
    history_file = os.path.join(workdir, "history.csv")
    surface_file = os.path.join(workdir, "surface_flow.csv")
    
    # Wykres zbieżności solvera
    if os.path.exists(history_file):
        df_hist = pd.read_csv(history_file)
        # Usunięcie białych znaków z nazw kolumn
        df_hist.columns = df_hist.columns.str.strip().str.replace('"', '')
        
        plt.figure(figsize=(10, 5))
        if "rms[Rho]" in df_hist.columns:
            plt.plot(df_hist["Inner_Iter"], df_hist["rms[Rho]"], label="RMS Rho")
            
        plt.yscale("log")
        plt.title("Historia zbieżności")
        plt.xlabel("Iteracja")
        plt.ylabel("Residuum gęstości (log)")
        plt.grid(True, which="both", ls="--")
        plt.savefig(os.path.join(results_dir, "plot_convergence.png"))
        plt.close()
    
    # Wykresy parametrów przepływu wzdłuż ścianki
    if os.path.exists(surface_file):
        df_surf = pd.read_csv(surface_file)
        df_surf.columns = df_surf.columns.str.strip().str.replace('"', '')
        
        # Sortowanie węzłów siatki wzdłuż osi X
        if "x" in df_surf.columns:
            df_surf = df_surf.sort_values(by="x")
        elif "PointX" in df_surf.columns:
            # Kompatybilność wsteczna z niektórymi wersjami SU2
            df_surf = df_surf.rename(columns={"PointX": "x", "PointY": "y"})
            df_surf = df_surf.sort_values(by="x")
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Profil liczby Macha
        if "Mach" in df_surf.columns:
            ax1.plot(df_surf["x"], df_surf["Mach"], color="red")
            ax1.set_ylabel("Liczba Macha [-]")
            ax1.grid(True)
            ax1.axvline(x=0, color='k', linestyle='--', label="Gardło dyszy (x=0)")
            ax1.legend()
            
        # Profil ciśnienia statycznego
        if "Pressure" in df_surf.columns:
            ax2.plot(df_surf["x"], df_surf["Pressure"], color="blue")
            ax2.set_ylabel("Ciśnienie statyczne [Pa]")
            ax2.set_xlabel("Współrzędna osiowa X [m]")
            ax2.grid(True)
            ax2.axvline(x=0, color='k', linestyle='--')
            
        plt.suptitle("Parametry przepływu na ściance dyszy")
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "plot_nozzle_flow.png"))
        plt.close()
        
    print(f"Zapisano wykresy walidacyjne w: {results_dir}")
    print("-------------------------------------------------------------\n")