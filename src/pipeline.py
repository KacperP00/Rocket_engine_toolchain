import os
import shutil
import subprocess
import sys
import datetime
import re
import pandas as pd

# Aktywacja trybu developerskiego (logi, probniki, diagnostyka)
DEVELOP_MODE = True

# Dodanie katalogu glownego oraz podkatalogow /src do sciezek interpretera przed importami
SRC_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SRC_DIR, '..'))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
for root, dirs, files in os.walk(SRC_DIR):
    if root not in sys.path:
        sys.path.append(root)

import cantera_solver
import gmsh_generator
import su2_runner
import diagnostics
import postprocessing
import calculix_runner

def log_message(msg, filepath=None):
    # Zapis komunikatu na konsole oraz do pliku raportu
    print(msg)
    if DEVELOP_MODE and filepath:
        with open(filepath, 'a', encoding='utf-8') as f:
            f.write(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")

def extract_probe_data(workdir, stage_name, report_path, cantera_data):
    # Odczyt pelnej domeny z restart.csv i ekstrakcja punktow kontrolnych
    flow_path = os.path.join(workdir, 'restart.csv')
    if not os.path.exists(flow_path):
        log_message(f"Brak pliku {flow_path}. Probkowanie pominiete.", None)
        return

    df = pd.read_csv(flow_path)
    df.columns = df.columns.str.strip().str.replace('"', '')

    gamma = cantera_data['gamma']
    R_gas = cantera_data['R']

    rho = df['Density']
    vx = df['Momentum_x'] / rho
    vy = df['Momentum_y'] / rho
    v2 = vx**2 + vy**2

    if 'Pressure' not in df.columns:
        df['Pressure'] = (gamma - 1.0) * (df['Energy'] - 0.5 * rho * v2)
    if 'Temperature' not in df.columns:
        df['Temperature'] = df['Pressure'] / (rho * R_gas)
    if 'Mach' not in df.columns:
        a = (gamma * df['Pressure'] / rho).clip(lower=1e-6)**0.5
        df['Mach'] = (v2**0.5) / a

    x_in = df['x'].min()
    x_out = df['x'].max()
    sections = [(x_in, "Wlot"), (0.0, "Zwezenie"), (x_out, "Wylot")]

    write_header = not os.path.exists(report_path) or os.path.getsize(report_path) == 0
    results = []
    if write_header:
        results.append("TIMESTAMP,ETAP,PRZEKROJ,Y,MACH,CISNIENIE_PA,TEMPERATURA_K,PREDKOSC_X,GESTOSC")

    now_str = datetime.datetime.now().strftime('%H:%M:%S')
    for x_val, name in sections:
        slice_df = df[abs(df['x'] - x_val) < 2e-3]
        if slice_df.empty:
            continue
        y_max = slice_df['y'].max()
        for y_val in [0.0, y_max / 2.0, y_max]:
            idx = (abs(slice_df['y'] - y_val)).idxmin()
            row = slice_df.loc[idx]
            vx_val = row['Momentum_x'] / row['Density']
            results.append(
                f"{now_str},{stage_name},{name},{row['y']:.5f},"
                f"{row['Mach']:.4f},{row['Pressure']:.2f},{row['Temperature']:.2f},"
                f"{vx_val:.2f},{row['Density']:.4f}"
            )

    with open(report_path, 'a', encoding='utf-8') as f:
        f.write("\n".join(results) + "\n")

def run_stage(command, workdir, stage_name, log_file):
    # Uruchomienie procesu z logowaniem wyjscia w czasie rzeczywistym
    log_message(f"=== Uruchamianie: {stage_name} ===", log_file)
    if DEVELOP_MODE and log_file:
        with open(log_file, 'a', encoding='utf-8') as f:
            process = subprocess.Popen(
                command, cwd=workdir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
            for line in process.stdout:
                sys.stdout.write(line)
                f.write(line)
            process.wait()
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, command)
    else:
        subprocess.run(command, cwd=workdir, check=True)

def run_automated_toolchain():
    workdir = os.path.join(ROOT_DIR, 'workdir')
    results_dir = os.path.join(ROOT_DIR, 'results')
    os.makedirs(workdir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    cfg_path = os.path.join(workdir, 'run.cfg')
    restart_file = os.path.join(workdir, 'restart.dat')
    solution_file = os.path.join(workdir, 'solution.dat')

    log_file = os.path.join(workdir, 'develop_report.log') if DEVELOP_MODE else None
    probe_file = os.path.join(workdir, 'develop_sim_out.csv') if DEVELOP_MODE else None

    if DEVELOP_MODE:
        open(log_file, 'w').close()
        open(probe_file, 'w').close()

    # 1. Obliczenia termochemiczne Cantera
    log_message("=== [ETAP 0] Termochemia (Cantera) i Siatkowanie (Gmsh) ===", log_file)
    cantera_data = cantera_solver.calculate_chamber_conditions()

    # 2. Generacja siatki jesli nie istnieje
    mesh_file = os.path.join(workdir, 'nozzle.su2')
    if not os.path.exists(mesh_file):
        gmsh_generator.generate_nozzle_mesh()

    # 3. Przygotowanie konfiguracji bazowej SU2
    su2_runner.prepare_su2_config(cantera_data, workdir)

    # 4. Etap 1: Przedmuch RANS 1. rzedu (budowa warstwy przysciennej i pol k-omega)
    run_stage(["SU2_CFD", "run.cfg"], workdir, "RANS_1ST_ORDER", log_file)

    if DEVELOP_MODE:
        extract_probe_data(workdir, "RANS_1ST_ORDER", probe_file, cantera_data)

    if not os.path.exists(restart_file):
        raise FileNotFoundError("Brak pliku restart.dat po etapie 1.")
    shutil.copy(restart_file, solution_file)
    log_message("Plik restart.dat skopiowany jako solution.dat.", log_file)

    # 5. Etap 2: Pelne obliczenia RANS 2. rzedu z pliku restartu
    with open(cfg_path, 'r') as f:
        config_text = f.read()

# W pliku src/pipeline.py (Etap 2):
    config_text = config_text.replace("MUSCL_FLOW= NO", "MUSCL_FLOW= YES\nRESTART_SOL= YES")
    config_text = config_text.replace("CFL_ADAPT= YES", "CFL_ADAPT= NO")
    config_text = config_text.replace("CFL_NUMBER= 0.5", "CFL_NUMBER= 1.0")
    config_text = re.sub(r'ITER=\s*\d+', 'ITER= 3000', config_text)

    with open(cfg_path, 'w') as f:
        f.write(config_text)

    run_stage(["SU2_CFD", "run.cfg"], workdir, "RANS_2ND_ORDER", log_file)

    

    if DEVELOP_MODE:
        extract_probe_data(workdir, "RANS_2ND_ORDER", probe_file, cantera_data)
        diagnostics.verify_mass_balance(workdir, log_file)
        diagnostics.verify_momentum_balance(workdir, cantera_data, log_file)
        diagnostics.verify_y_plus(workdir, log_file, mu_val=cantera_data['mu'])
        diagnostics.verify_heat_transfer(workdir, results_dir, cantera_data, log_file=log_file)
        diagnostics.plot_convergence(workdir, log_file)
        diagnostics.verify_flow_separation(workdir, results_dir, cantera_data, log_file=log_file)

    postprocessing.generate_validation_plots(workdir, results_dir, cantera_data)

    # 7. Analiza wytrzymalosciowo-termiczna MES (CalculiX)
    log_message("=== [ETAP 3] Analiza MES (CalculiX) ===", log_file)
    calculix_runner.run_fea_analysis(workdir, results_dir, cantera_data)

    log_message("=== ZAKONCZONO POTOK OBLICZENIOWY ===", log_file)

if __name__ == "__main__":
    run_automated_toolchain()