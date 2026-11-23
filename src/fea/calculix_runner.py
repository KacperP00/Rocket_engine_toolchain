import os
import sys
import subprocess
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import config

def parse_frd_results(frd_path, n_pts, nodes_per_sec):
    # Extract nodal stress tensors and PEEQ; extrapolate inner liner surface stress
    raw_tensors = {}
    raw_peeq = {}
    mode = None

    with open(frd_path, 'r') as f:
        for line in f:
            if line.startswith(" -4  STRESS"):
                mode = "STRESS"
                continue
            elif line.startswith(" -4  PE "):
                mode = "PEEQ"
                continue
            elif line.startswith(" -3"):
                mode = None
                continue

            if mode == "STRESS" and line.startswith(" -1"):
                nid = int(line[3:13])
                vals = np.array([float(line[13 + 12*k : 13 + 12*(k+1)]) for k in range(6)])
                raw_tensors[nid] = vals
            elif mode == "PEEQ" and line.startswith(" -1"):
                nid = int(line[3:13])
                raw_peeq[nid] = float(line[13:25])

    vm_surface = []
    peeq_surface = []
    for i in range(n_pts):
        nid_0 = i * nodes_per_sec + 1  # Inner surface node (layer 0 centroid)
        nid_1 = i * nodes_per_sec + 2  # Interface between layer 0 and layer 1
        if nid_0 in raw_tensors and nid_1 in raw_tensors:
            s_wall = raw_tensors[nid_0]
            sxx, syy, szz, sxy, syz, szx = s_wall
            vm = np.sqrt(0.5 * ((sxx - syy)**2 + (syy - szz)**2 + (szz - sxx)**2 +
                                6.0 * (sxy**2 + syz**2 + szx**2)))
            vm_surface.append(vm)
        else:
            vm_surface.append(np.nan)
        peeq_surface.append(raw_peeq.get(nid_0, 0.0))

    return np.array(vm_surface), np.array(peeq_surface)

def cucrzr_yield_strength(temp_k):
    # Interpolate temperature-dependent 0.2% yield strength of CuCrZr [Pa]
    temps = np.array([300.0, 600.0, 800.0, 1000.0])
    yields = np.array([310e6, 240e6, 165e6, 95e6])
    return np.interp(temp_k, temps, yields)

def run_fea_analysis(workdir, results_dir, cantera_data):
    # Generate bimetallic elastoplastic CalculiX deck, run ccx, and plot FoS/stresses
    thermal_file = os.path.join(workdir, "wall_thermal_loads.csv")
    if not os.path.exists(thermal_file):
        print("[FEA MODULE] Missing wall_thermal_loads.csv. Skipping FEA.")
        return

    df = pd.read_csv(thermal_file)
    df.columns = df.columns.str.strip().str.replace('"', '')
    df = df.sort_values(by="x").reset_index(drop=True)

    gamma = cantera_data['gamma']
    rho = df['Density'].values
    v2 = (df['Momentum_x'].values / rho)**2 + (df['Momentum_y'].values / rho)**2
    p_int = (gamma - 1.0) * (df['Energy'].values - 0.5 * rho * v2)
    p_amb = getattr(config, 'AMBIENT_PRESSURE', 0.0)
    p_diff = p_int - p_amb

    x_i = df['x'].values
    r_i = df['y'].values
    dr_dx = np.gradient(r_i, x_i)
    cos_a = 1.0 / np.sqrt(1.0 + dr_dx**2)
    sin_a = dr_dx * cos_a

    t_liner = getattr(config, 'LINER_THICKNESS', 0.0010)
    t_jacket = getattr(config, 'JACKET_THICKNESS', 0.0030)
    t_ref = 300.0

    # Use 1D cooling profiles if available, otherwise fallback to constant T_wall
    if 'T_w_hot' in df.columns and 'T_w_cold' in df.columns and 'T_cool' in df.columns:
        t_hot = df['T_w_hot'].values
        t_cold = df['T_w_cold'].values
        t_jkt = df['T_cool'].values
    else:
        q_wall = df['Heat_Flux_CFD'].values
        t_hot = np.full_like(x_i, getattr(config, 'WALL_TEMP', 800.0))
        t_cold = t_hot - (q_wall * t_liner) / getattr(config, 'WALL_K', 320.0)
        t_jkt = np.full_like(x_i, 300.0)

    # Discretization: 4 layers CuCrZr + 2 layers Inconel 718 = 6 layers (7 nodes/station)
    n_pts = len(x_i)
    n_cu = 4
    n_inc = 2
    n_tot = n_cu + n_inc
    nodes_per_sec = n_tot + 1

    # Cumulative radial offsets across the 6 layers
    layer_offsets = np.concatenate([
        np.linspace(0.0, t_liner, n_cu + 1),
        np.linspace(t_liner, t_liner + t_jacket, n_inc + 1)[1:]
    ])

    inp_path = os.path.join(workdir, "nozzle_fea.inp")
    with open(inp_path, 'w') as f:
        f.write("*HEADING\nBimetallic Elastoplastic Rocket Nozzle FEA (CuCrZr + Inconel 718)\n*NODE\n")
        for i in range(n_pts):
            for j in range(nodes_per_sec):
                off = layer_offsets[j]
                r_node = r_i[i] + off * cos_a[i]
                x_node = x_i[i] - off * sin_a[i]
                nid = i * nodes_per_sec + j + 1
                f.write(f"{nid}, {r_node:.6f}, {x_node:.6f}, 0.0\n")

        # CuCrZr inner liner elements
        f.write("*ELEMENT, TYPE=CAX4, ELSET=EL_CU\n")
        eid = 1
        inner_eids = []
        for i in range(n_pts - 1):
            for j in range(n_cu):
                n1 = i * nodes_per_sec + j + 1
                n2 = i * nodes_per_sec + (j + 1) + 1
                n3 = (i + 1) * nodes_per_sec + (j + 1) + 1
                n4 = (i + 1) * nodes_per_sec + j + 1
                f.write(f"{eid}, {n1}, {n2}, {n3}, {n4}\n")
                if j == 0:
                    inner_eids.append((eid, i))
                eid += 1

        # Inconel 718 outer structural jacket elements
        f.write("*ELEMENT, TYPE=CAX4, ELSET=EL_INC\n")
        for i in range(n_pts - 1):
            for j in range(n_cu, n_tot):
                n1 = i * nodes_per_sec + j + 1
                n2 = i * nodes_per_sec + (j + 1) + 1
                n3 = (i + 1) * nodes_per_sec + (j + 1) + 1
                n4 = (i + 1) * nodes_per_sec + j + 1
                f.write(f"{eid}, {n1}, {n2}, {n3}, {n4}\n")
                eid += 1

        # Material 1: Temperature-dependent elastoplastic CuCrZr
        f.write("*MATERIAL, NAME=CUCRZR\n*ELASTIC\n")
        f.write("128e9, 0.34, 300.0\n112e9, 0.34, 600.0\n98e9, 0.34, 800.0\n85e9, 0.34, 1000.0\n")
        f.write("*PLASTIC\n")
        f.write("310e6, 0.000, 300.0\n350e6, 0.050, 300.0\n")
        f.write("240e6, 0.000, 600.0\n270e6, 0.050, 600.0\n")
        f.write("165e6, 0.000, 800.0\n185e6, 0.050, 800.0\n")
        f.write("95e6,  0.000, 1000.0\n110e6, 0.050, 1000.0\n")
        f.write(f"*EXPANSION, ZERO={t_ref:.1f}\n{config.WALL_ALPHA:.6e}\n")

        # Material 2: Cold Inconel 718 structural jacket
        f.write("*MATERIAL, NAME=INCONEL718\n*ELASTIC\n")
        f.write(f"{config.JACKET_E:.1f}, {config.JACKET_NU:.3f}\n")
        f.write(f"*EXPANSION, ZERO={t_ref:.1f}\n{config.JACKET_ALPHA:.6e}\n")

        f.write("*SOLID SECTION, ELSET=EL_CU, MATERIAL=CUCRZR\n")
        f.write("*SOLID SECTION, ELSET=EL_INC, MATERIAL=INCONEL718\n")

        # Axial support at chamber inlet (radially free for thermal expansion)
        f.write("*BOUNDARY\n")
        for j in range(nodes_per_sec):
            f.write(f"{j+1}, 2, 2, 0.0\n")

        # Assembly reference temperature (zero thermal strain at 300 K)
        f.write("*INITIAL CONDITIONS, TYPE=TEMPERATURE\n")
        for nid in range(1, n_pts * nodes_per_sec + 1):
            f.write(f"{nid}, {t_ref:.1f}\n")

        # Nonlinear static step with incremental loading for plasticity convergence
        f.write("*STEP, INC=100\n*STATIC\n0.1, 1.0, 1e-4, 0.25\n*TEMPERATURE\n")
        for i in range(n_pts):
            for j in range(nodes_per_sec):
                if j <= n_cu:
                    frac = j / float(n_cu)
                    t_node = t_hot[i] + frac * (t_cold[i] - t_hot[i])
                else:
                    t_node = t_jkt[i]
                nid = i * nodes_per_sec + j + 1
                f.write(f"{nid}, {t_node:.2f}\n")

        f.write("*DLOAD\n")
        for inner_eid, i in inner_eids:
            p_elem = 0.5 * (p_diff[i] + p_diff[i+1])
            f.write(f"{inner_eid}, P4, {p_elem:.2f}\n")

        f.write("*NODE FILE\nU, S, NT, PEEQ\n*END STEP\n")

    frd_path = os.path.join(workdir, "nozzle_fea.frd")
    sigma_vm_ccx, peeq_ccx = None, None
    if shutil.which("ccx"):
        subprocess.run(["ccx", "nozzle_fea"], cwd=workdir, check=True)
        if os.path.exists(frd_path):
            sigma_vm_ccx, peeq_ccx = parse_frd_results(frd_path, n_pts, nodes_per_sec)

    yield_curve = cucrzr_yield_strength(t_hot)
    if sigma_vm_ccx is not None:
        valid = slice(4, -4)
        fos = yield_curve[valid] / np.maximum(sigma_vm_ccx[valid], 1.0)
        idx_min = np.argmin(fos) + 4
        max_peeq = np.nanmax(peeq_ccx) * 100.0
        print(f"[STRUCTURAL INTEGRITY] Max Inner VM Stress: {np.nanmax(sigma_vm_ccx)/1e6:.1f} MPa | "
              f"Min FoS: {np.min(fos):.2f} at x={x_i[idx_min]:.3f} m | Max PEEQ: {max_peeq:.3f}%")

    # Dual-panel plot: Stress vs Temperature-Dependent Yield Strength + Plastic Strain
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True)
    ax1.plot(x_i, yield_curve / 1e6, 'k--', linewidth=1.8, label='CuCrZr Yield Strength R_p0.2(T_w) [MPa]')
    if sigma_vm_ccx is not None:
        ax1.plot(x_i, sigma_vm_ccx / 1e6, 'r-', linewidth=2.0, label='CalculiX Elastoplastic VM Stress [MPa]')
    ax1.axvline(x=0.0, color='gray', linestyle=':', label='Nozzle Throat')
    ax1.set_ylabel('Stress [MPa]')
    ax1.set_title(f'Bimetallic Liner Stress Analysis (CuCrZr {t_liner*1000:.1f} mm + Inconel 718 {t_jacket*1000:.1f} mm)')
    ax1.grid(True, ls='--')
    ax1.legend()

    if peeq_ccx is not None:
        ax2.plot(x_i, peeq_ccx * 100.0, 'b-', linewidth=2.0, label='Equivalent Plastic Strain (PEEQ) [%]')
    ax2.axvline(x=0.0, color='gray', linestyle=':')
    ax2.set_xlabel('Axial Coordinate x [m]')
    ax2.set_ylabel('Plastic Strain [%]')
    ax2.set_title('Accumulated Equivalent Plastic Strain on Hot Gas Wall (Low-Cycle Fatigue Indicator)')
    ax2.grid(True, ls='--')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "plot_fea_stress.png"), dpi=300)
    plt.close()