import os
import shutil
import subprocess
import sys
import datetime
import re
import pandas as pd

# Enable developer mode (logs, probes, diagnostics)
DEVELOP_MODE = True

# Add the root directory and /src subdirectories to the interpreter paths before imports
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
import cooling_1d

def log_message(msg, filepath=None):
    # Write the message to the console and report file
    print(msg)
    if DEVELOP_MODE and filepath:
        with open(filepath, 'a', encoding='utf-8') as f:
            f.write(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")

def extract_probe_data(workdir, stage_name, report_path, cantera_data):
    # Read the full domain from restart.csv and extract control points
    flow_path = os.path.join(workdir, 'restart.csv')
    if not os.path.exists(flow_path):
        log_message(f"Missing file {flow_path}. Sampling skipped.", None)
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
    sections = [(x_in, "Inlet"), (0.0, "Throat"), (x_out, "Outlet")]

    write_header = not os.path.exists(report_path) or os.path.getsize(report_path) == 0
    results = []
    if write_header:
        results.append("TIMESTAMP,STAGE,SECTION,Y,MACH,PRESSURE_PA,TEMPERATURE_K,VELOCITY_X,DENSITY")

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
    # Run the process and log output in real time
    log_message(f"=== Running: {stage_name} ===", log_file)
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

    # 1. Cantera thermochemical calculations
    log_message("=== [STAGE 0] Thermochemistry (Cantera) and Meshing (Gmsh) ===", log_file)
    cantera_data = cantera_solver.calculate_chamber_conditions()

    # 2. Generate the mesh if it does not exist
    mesh_file = os.path.join(workdir, 'nozzle.su2')
    if not os.path.exists(mesh_file):
        gmsh_generator.generate_nozzle_mesh()

    # 3. Prepare the base SU2 configuration
    su2_runner.prepare_su2_config(cantera_data, workdir)

    # 4. Stage 1: first-order RANS (build the boundary layer and k-omega fields)
    run_stage(["SU2_CFD", "run.cfg"], workdir, "RANS_1ST_ORDER", log_file)

    if DEVELOP_MODE:
        extract_probe_data(workdir, "RANS_1ST_ORDER", probe_file, cantera_data)

    if not os.path.exists(restart_file):
        raise FileNotFoundError("Missing restart.dat file after stage 1.")
    shutil.copy(restart_file, solution_file)
    log_message("Copied restart.dat to solution.dat.", log_file)

    # 5. Stage 2: full second-order RANS calculation from the restart file
    # Stage 2: 2nd-order RANS with fixed CFL and extended iterations for sublayer convergence
    with open(cfg_path, 'r') as f:
        config_text = f.read()

    config_text = config_text.replace("MUSCL_FLOW= NO", "MUSCL_FLOW= YES\nRESTART_SOL= YES")
    config_text = config_text.replace("CFL_ADAPT= YES", "CFL_ADAPT= NO")
    config_text = config_text.replace("CFL_NUMBER= 0.5", "CFL_NUMBER= 2.0")
    config_text = re.sub(r'ITER=\s*\d+', 'ITER= 6000', config_text)

    with open(cfg_path, 'w') as f:
        f.write(config_text)

    run_stage(["SU2_CFD", "run.cfg"], workdir, "RANS_2ND_ORDER", log_file)

    # Diagnostics and postprocessing
    m_in = 8.64
    if DEVELOP_MODE:
        extract_probe_data(workdir, "RANS_2ND_ORDER", probe_file, cantera_data)
        m_res = diagnostics.verify_mass_balance(workdir, log_file)
        if isinstance(m_res, tuple):
            m_in = m_res[0]
        diagnostics.verify_momentum_balance(workdir, cantera_data, log_file)
        diagnostics.verify_y_plus(workdir, log_file, mu_val=cantera_data['mu_wall'])
        diagnostics.verify_heat_transfer(workdir, results_dir, cantera_data, log_file=log_file)
        diagnostics.verify_flow_separation(workdir, results_dir, cantera_data, log_file=log_file)
        diagnostics.plot_convergence(workdir, log_file)

    postprocessing.generate_validation_plots(workdir, results_dir, cantera_data)

    # Stage 3: 1D Regenerative Cooling
    log_message("=== [STAGE 3] 1D Regenerative Cooling ===", log_file)
    cooling_1d.run_cooling_analysis(workdir, results_dir, m_dot_total=m_in)

    # Stage 4: Bimetallic Elastoplastic FEA (CalculiX)
    log_message("=== [STAGE 4] Bimetallic Elastoplastic FEA (CalculiX) ===", log_file)
    calculix_runner.run_fea_analysis(workdir, results_dir, cantera_data)

    log_message("=== COMPUTATIONAL TOOLCHAIN COMPLETED ===", log_file)

if __name__ == "__main__":
    run_automated_toolchain()