import os
import shutil
import subprocess
import sys
import datetime
import pandas as pd
import diagnostics
import re

# Aktywacja trybu developerskiego (logowanie, zrzut parametrów punktowych)
DEVELOP_MODE = True

# Dodanie podkatalogów do ścieżek interpretera
SRC_DIR = os.path.abspath(os.path.dirname(__file__))
for root, dirs, files in os.walk(SRC_DIR):
    if root not in sys.path:
        sys.path.append(root)
        
import cantera_solver
import su2_runner

def log_message(msg, filepath=None):
    """Zapis komunikatu na konsolę oraz do pliku raportu (jesli podano)."""
    print(msg)
    if DEVELOP_MODE and filepath:
        with open(filepath, 'a', encoding='utf-8') as f:
            f.write(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")

def extract_probe_data(workdir, stage_name, report_path, gamma=1.2, R_gas=405.8):
    """Odczyt siatki i interpolacja parametrów przepływu w strategicznych przekrojach."""
    flow_path = os.path.join(workdir, 'flow.csv')
    if not os.path.exists(flow_path):
        log_message("Brak flow.csv. Próbkowanie pominięte.")
        return
        
    df = pd.read_csv(flow_path)
    df.columns = df.columns.str.strip()
    
    # Obliczenie zmiennych prymitywnych z wektora stanu zachowawczego
    if 'Pressure' not in df.columns:
        rho = df['Density']
        vx = df['Momentum_x'] / rho
        vy = df['Momentum_y'] / rho
        v2 = vx**2 + vy**2
        df['Pressure'] = (gamma - 1) * (df['Energy'] - 0.5 * rho * v2)
        df['Temperature'] = df['Pressure'] / (rho * R_gas)
        df['Mach'] = (v2**0.5) / ((gamma * df['Pressure'] / rho)**0.5)
        
    x_in = df['x'].min()
    x_out = df['x'].max()
    
    sections = [
        (x_in, "Wlot"),
        (0.0, "Zwezenie"),
        (x_out, "Wylot")
    ]
    
    results = []
    results.append(f"ETAP,PRZEKROJ,Y,MACH,CISNIENIE_PA,TEMPERATURA_K,PREDKOSC_X")
    
    for x_val, name in sections:
        # Ekstrakcja węzłów na danym przekroju (tolerancja numeryczna 1mm)
        slice_df = df[abs(df['x'] - x_val) < 1e-3]
        if slice_df.empty:
            continue
            
        y_max = slice_df['y'].max()
        heights = [0.0, y_max/2, y_max]
        
        for y_val in heights:
            idx = (abs(slice_df['y'] - y_val)).idxmin()
            row = slice_df.loc[idx]
            
            mach = row['Mach']
            press = row['Pressure']
            temp = row['Temperature']
            vx = row['Momentum_x'] / row['Density']
            
            results.append(f"{stage_name},{name},{y_val:.4f},{mach:.4f},{press:.2f},{temp:.2f},{vx:.2f}")
            
    with open(report_path, 'a', encoding='utf-8') as f:
        f.write("\n".join(results) + "\n")

def run_stage(command, workdir, stage_name, log_file):
    """Zarządzanie procesem podrzędnym z przekierowaniem strumieni do raportu."""
    log_message(f"=== Uruchamianie: {stage_name} ===", log_file)
    
    if DEVELOP_MODE:
        with open(log_file, 'a') as f:
            process = subprocess.Popen(command, cwd=workdir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in process.stdout:
                sys.stdout.write(line)
                f.write(line)
            process.wait()
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, command)
    else:
        subprocess.run(command, cwd=workdir, check=True)

def run_automated_toolchain():
    workdir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../workdir'))
    cfg_path = os.path.join(workdir, 'run.cfg')
    restart_file = os.path.join(workdir, 'restart.dat')
    solution_file = os.path.join(workdir, 'solution.dat')
    
    log_file = os.path.join(workdir, 'develop_report.log') if DEVELOP_MODE else None
    probe_file = os.path.join(workdir, 'develop_sim_out.csv') if DEVELOP_MODE else None
    
    if DEVELOP_MODE:
        open(log_file, 'w').close()
        open(probe_file, 'w').close()

    log_message("=== [ETAP 0] Termochemia i Generacja Konfiguracji ===", log_file)
    cantera_data = cantera_solver.calculate_chamber_conditions()
    su2_runner.prepare_su2_config(cantera_data, workdir)
    
    # Wymuszenie eksportu pełnej domeny do CSV dla próbników
    if DEVELOP_MODE:
        with open(cfg_path, 'r') as f:
            cfg_content = f.read()
        if "CSV" not in cfg_content.split("OUTPUT_FILES")[1].split("\n")[0]:
            cfg_content = cfg_content.replace("OUTPUT_FILES= (RESTART, PARAVIEW, SURFACE_CSV)", 
                                              "OUTPUT_FILES= (RESTART, PARAVIEW, SURFACE_CSV, CSV)")
            with open(cfg_path, 'w') as f:
                f.write(cfg_content)

    with open(cfg_path, 'r') as f:
        if "SOLVER= RANS" in f.read():
            raise RuntimeError("Szablon base_config.cfg musi mieć SOLVER= EULER na starcie.")

    run_stage(["SU2_CFD", "run.cfg"], workdir, "EULER", log_file)
    
    if DEVELOP_MODE:
        extract_probe_data(workdir, "EULER", probe_file)

    if os.path.exists(restart_file):
        shutil.copy(restart_file, solution_file)
        log_message("Plik restartu skopiowany.", log_file)
    else:
        raise FileNotFoundError("Brak pliku restart.dat po etapie EULER.")
        
    with open(cfg_path, 'r') as f:
        config = f.read()
        
    config = config.replace("SOLVER= EULER", "SOLVER= RANS\nKIND_TURB_MODEL= SST")
    # UWAGA: Zostawiamy MUSCL_FLOW= NO (1. rzad) na potrzeby amortyzacji szoku
    config = config.replace("MUSCL_FLOW= NO", "MUSCL_FLOW= NO\nRESTART_SOL= YES")
    config = config.replace("MARKER_EULER= ( MARKER_WALL )", "MARKER_HEATFLUX= ( MARKER_WALL, 0.0 )")
    # Mikroskopijna liczba Couranta i ograniczenie iteracji do szybkiego zbudowania warstwy
    config = config.replace("CFL_NUMBER= 1.0", "CFL_NUMBER= 0.01") 
    config = re.sub(r'ITER=\s*4000', 'ITER= 250', config) 
    
    with open(cfg_path, 'w') as f:
        f.write(config)
        
    run_stage(["SU2_CFD", "run.cfg"], workdir, "RANS_START", log_file)
    
    # Skopiowanie stabilnego pola predkosci jako bazy dla wlasciwych obliczen
    if os.path.exists(restart_file):
        shutil.copy(restart_file, solution_file)
        
    # --- [ETAP 2B] Wlasciwe obliczenia RANS (2. Rzad) ---
    with open(cfg_path, 'r') as f:
        config = f.read()
        
    # Wlaczenie precyzji 2. rzedu, zwiekszenie CFL i powrot do 4000 iteracji
    config = config.replace("MUSCL_FLOW= NO", "MUSCL_FLOW= YES")
    config = config.replace("CFL_NUMBER= 0.01", "CFL_NUMBER= 0.1") 
    config = re.sub(r'ITER=\s*250', 'ITER= 4000', config)
    
    with open(cfg_path, 'w') as f:
        f.write(config)
        
    run_stage(["SU2_CFD", "run.cfg"], workdir, "RANS_FULL", log_file)
    
    if DEVELOP_MODE:
        extract_probe_data(workdir, "RANS", probe_file)
        diagnostics.verify_mass_balance(workdir, log_file)
        diagnostics.verify_y_plus(workdir, log_file)
        diagnostics.plot_convergence(workdir, log_file)
        
    log_message("=== ZAKONCZONO POTOK OBLICZENIOWY ===", log_file)

if __name__ == "__main__":
    run_automated_toolchain()