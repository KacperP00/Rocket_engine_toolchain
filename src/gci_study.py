import os
import sys
import shutil
import subprocess
import re
import numpy as np
import pandas as pd

SRC_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SRC_DIR, '..'))
for path in [ROOT_DIR, SRC_DIR,
             os.path.join(SRC_DIR, 'chemistry'),
             os.path.join(SRC_DIR, 'meshing'),
             os.path.join(SRC_DIR, 'cfd'),
             os.path.join(SRC_DIR, 'thermal'),
             os.path.join(SRC_DIR, 'fea')]:
    if path not in sys.path:
        sys.path.append(path)

import cantera_solver
import gmsh_generator
import su2_runner
import diagnostics
import postprocessing
import cooling_1d
import calculix_runner


def compute_asme_gci(f3, f2, f1, r32, r21):
    # Compute ASME V&V 20 (Celik 2008) apparent order p, convergence regime, and fine-grid GCI
    eps32 = f3 - f2
    eps21 = f2 - f1

    if abs(eps21) < 1e-12 or abs(eps32) < 1e-12:
        return {
            "R_ratio": 0.0, "regime": "Exact / Zero Change",
            "p": 2.0, "Fs": 1.25, "e_a21_pct": 0.0,
            "GCI_fine_pct": 0.0, "f_ext": f1
        }

    R_ratio = eps21 / eps32
    e_a21 = abs((f1 - f2) / f1)

    if 0.0 < R_ratio < 1.0:
        # Monotonic convergence: solve fixed-point equation for p with variable r21, r32
        s = np.sign(eps32 / eps21)
        p = abs(np.log(abs(eps32 / eps21)) / np.log(r21))
        for _ in range(50):
            q_p = np.log((r21**p - s) / (r32**p - s))
            p_new = (1.0 / np.log(r21)) * abs(np.log(abs(eps32 / eps21)) + q_p)
            if abs(p_new - p) < 1e-6:
                p = p_new
                break
            p = p_new
        p = float(np.clip(p, 0.5, 4.0))
        regime = "Monotonic"
        Fs = 1.25
    elif R_ratio >= 1.0:
        # Non-asymptotic monotonic step: use formal 2nd-order p=2.0 with conservative Fs=3.0
        p = 2.0
        regime = "Non-Asymptotic"
        Fs = 3.0
    elif -1.0 < R_ratio < 0.0:
        # Oscillatory convergence: use formal 2nd-order p=2.0 with conservative Fs=3.0
        p = 2.0
        regime = "Oscillatory Conv."
        Fs = 3.0
    else:
        # Oscillatory divergence: conservative Fs=3.0, formal p=2.0
        p = 2.0
        regime = "Oscillatory Non-Asymp."
        Fs = 3.0

    gci_fine = (Fs * e_a21) / (r21**p - 1.0) * 100.0
    f_ext = (r21**p * f1 - f2) / (r21**p - 1.0)

    return {
        "R_ratio": float(R_ratio),
        "regime": regime,
        "p": float(p),
        "Fs": float(Fs),
        "e_a21_pct": float(e_a21 * 100.0),
        "GCI_fine_pct": float(gci_fine),
        "f_ext": float(f_ext)
    }


def run_gci_study():
    # Execute 3-mesh grid convergence study with scaled iterations and full Fine-grid postprocessing
    workdir = os.path.join(ROOT_DIR, 'workdir')
    results_dir = os.path.join(ROOT_DIR, 'results')
    os.makedirs(workdir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    cantera_data = cantera_solver.calculate_chamber_conditions()

    # Grid sequence: (Label, lc_core [m], bl_size [m], Stage 2 iterations)
    meshes = [
        ("Coarse", 0.0045, 7.5e-7, 5000),
        ("Medium", 0.0030, 5.0e-7, 7000),
        ("Fine",   0.0020, 3.5e-7, 10000)
    ]

    records = []
    m_in_final = 8.64

    for label, lc_val, bl_val, n_iter in meshes:
        print(f"\n================ GCI RUN: {label} (lc={lc_val}, iter={n_iter}) ================")
        gmsh_generator.generate_nozzle_mesh(lc_core=lc_val, bl_size=bl_val)
        su2_runner.prepare_su2_config(cantera_data, workdir)

        # Stage 1: 1st-order RANS initialization
        subprocess.run(["SU2_CFD", "run.cfg"], cwd=workdir, check=True)
        shutil.copy(os.path.join(workdir, "restart.dat"), os.path.join(workdir, "solution.dat"))

        # Stage 2: 2nd-order RANS MUSCL continuation
        cfg_path = os.path.join(workdir, "run.cfg")
        with open(cfg_path, 'r') as f:
            txt = f.read()
        txt = txt.replace("MUSCL_FLOW= NO", "MUSCL_FLOW= YES\nRESTART_SOL= YES")
        txt = txt.replace("CFL_ADAPT= YES", "CFL_ADAPT= NO")
        txt = txt.replace("CFL_NUMBER= 0.5", "CFL_NUMBER= 1.5")
        txt = re.sub(r'ITER=\s*\d+', f'ITER= {n_iter}', txt)
        with open(cfg_path, 'w') as f:
            f.write(txt)

        subprocess.run(["SU2_CFD", "run.cfg"], cwd=workdir, check=True)

        # Extract node count and integral/local validation metrics
        df_vol = pd.read_csv(os.path.join(workdir, 'restart.csv'))
        n_nodes = len(df_vol)

        m_in, m_out, err_m = diagnostics.verify_mass_balance(workdir)
        f_vac, f_cv, err_mom = diagnostics.verify_momentum_balance(workdir, cantera_data)
        y_max, y_mean = diagnostics.verify_y_plus(workdir, mu_val=cantera_data['mu_wall'])
        q_arr = diagnostics.verify_heat_transfer(workdir, results_dir, cantera_data)
        x_sep = diagnostics.verify_flow_separation(workdir, results_dir, cantera_data)

        df_surf = pd.read_csv(os.path.join(workdir, 'wall_thermal_loads.csv'))
        xw = df_surf['x'].values
        rw = df_surf['y'].values
        ds = np.hypot(np.diff(xw), np.diff(rw))
        q_tot_MW = float(np.sum(0.5 * (q_arr[:-1] + q_arr[1:]) * 2.0 * np.pi * 0.5 * (rw[:-1] + rw[1:]) * ds)) / 1e6
        q_throat = float(np.max(q_arr[np.abs(xw) < 0.025])) / 1e6

        m_in_final = m_in
        records.append({
            "Grid": label,
            "lc_m": lc_val,
            "N_nodes": n_nodes,
            "y_plus_mean": y_mean,
            "m_dot_kg_s": m_in,
            "F_vac_kN": f_vac,
            "Q_total_MW": q_tot_MW,
            "q_throat_MW_m2": q_throat,
            "x_sep_m": x_sep if x_sep is not None else 0.0
        })

    df_gci = pd.DataFrame(records)
    print("\n=== GRID CONVERGENCE STUDY DATA ===")
    print(df_gci.to_string(index=False))

    # Effective 2D grid refinement ratios based on actual node counts
    r32 = np.sqrt(df_gci["N_nodes"].iloc[1] / df_gci["N_nodes"].iloc[0])
    r21 = np.sqrt(df_gci["N_nodes"].iloc[2] / df_gci["N_nodes"].iloc[1])
    print(f"\nEffective 2D refinement ratios: r32 = {r32:.3f}, r21 = {r21:.3f}")

    gci_rows = []
    metrics = ["m_dot_kg_s", "F_vac_kN", "Q_total_MW", "q_throat_MW_m2", "x_sep_m"]
    for col in metrics:
        f3, f2, f1 = df_gci[col].iloc[0], df_gci[col].iloc[1], df_gci[col].iloc[2]
        if f1 > 0.0:
            res = compute_asme_gci(f3, f2, f1, r32, r21)
            res["Metric"] = col
            res["Coarse"] = f3
            res["Medium"] = f2
            res["Fine"] = f1
            gci_rows.append(res)
            print(
                f"Metric {col:15s} | Regime: {res['regime']:22s} | "
                f"p={res['p']:.2f} (Fs={res['Fs']:.2f}) | e_a21={res['e_a21_pct']:.2f}% | "
                f"GCI_fine={res['GCI_fine_pct']:.2f}% | f_ext={res['f_ext']:.4f}"
            )

    df_gci.to_csv(os.path.join(results_dir, "gci_summary.csv"), index=False)
    pd.DataFrame(gci_rows).to_csv(os.path.join(results_dir, "gci_metrics.csv"), index=False)

    # Run consistent postprocessing, 1D cooling, and elastoplastic FEA on converged Fine mesh
    print("\n=== FINAL CONSISTENT POSTPROCESSING ON FINE MESH ===")
    diagnostics.plot_convergence(workdir)
    postprocessing.generate_validation_plots(workdir, results_dir, cantera_data)
    cooling_1d.run_cooling_analysis(workdir, results_dir, m_dot_total=m_in_final)
    calculix_runner.run_fea_analysis(workdir, results_dir, cantera_data)
    print("=== GCI STUDY AND FINE-MESH WORKFLOW COMPLETED ===")


if __name__ == "__main__":
    run_gci_study()