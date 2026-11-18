import os
import sys
import subprocess
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import config

def parse_frd_surface_stresses(frd_path, n_pts, nodes_per_sec):
    # Odczyt tensora naprezen z wezlow j=0 oraz j=1 i ekstrapolacja liniowa na powierzchnie wewnetrzna
    raw_tensors = {}
    in_stress_block = False
    with open(frd_path, 'r') as f:
        for line in f:
            if line.startswith(" -4  STRESS"):
                in_stress_block = True
                continue
            if in_stress_block:
                if line.startswith(" -3"):
                    in_stress_block = False
                    continue
                if line.startswith(" -1"):
                    node_id = int(line[3:13])
                    vals = np.array([float(line[13 + 12*k : 13 + 12*(k+1)]) for k in range(6)])
                    raw_tensors[node_id] = vals

    vm_surface = []
    for i in range(n_pts):
        nid_0 = i * nodes_per_sec + 1  # Wezel na sciance (srodek warstwy 0: frac = 0.125)
        nid_1 = i * nodes_per_sec + 2  # Drugi wezel (styk warstwy 0 i 1: frac = 0.250)
        if nid_0 in raw_tensors and nid_1 in raw_tensors:
            # Ekstrapolacja liniowa tensora naprezen do powierzchni frac = 0.0
            s_wall = 2.0 * raw_tensors[nid_0] - 1.0 * raw_tensors[nid_1]
            sxx, syy, szz, sxy, syz, szx = s_wall
            vm = np.sqrt(0.5 * ((sxx - syy)**2 + (syy - szz)**2 + (szz - sxx)**2 + 6.0 * (sxy**2 + syz**2 + szx**2)))
            vm_surface.append(vm)
        else:
            vm_surface.append(np.nan)
    return np.array(vm_surface)

def run_fea_analysis(workdir, results_dir, cantera_data):
    # Wczytanie obciazen cisnieniowych i cieplnych
    thermal_file = os.path.join(workdir, "wall_thermal_loads.csv")
    surf_file = os.path.join(workdir, "surface_flow.csv")
    source_file = thermal_file if os.path.exists(thermal_file) else surf_file
    if not os.path.exists(source_file):
        print("[MODUL FEA] Brak danych powierzchniowych. Analiza pominieta.")
        return

    df = pd.read_csv(source_file)
    df.columns = df.columns.str.strip().str.replace('"', '')
    df = df.sort_values(by="x").reset_index(drop=True)

    gamma = cantera_data['gamma']
    rho = df['Density'].values
    v2 = (df['Momentum_x'].values / rho)**2 + (df['Momentum_y'].values / rho)**2
    p_internal = (gamma - 1.0) * (df['Energy'].values - 0.5 * rho * v2)
    p_amb = getattr(config, 'AMBIENT_PRESSURE', 0.0)
    # Cisnienie roznicowe dzialajace na powloke (ponizej 1 atm w srodku oznacza zgniatanie z zewnatrz)
    p = p_internal - p_amb

    q_wall = df['Heat_Flux_CFD'].values if 'Heat_Flux_CFD' in df.columns else np.zeros_like(p)

    t = getattr(config, 'WALL_THICKNESS', 0.004)
    k_wall = getattr(config, 'WALL_K', 320.0)
    E_mod = getattr(config, 'WALL_E', 128e9)
    nu = getattr(config, 'WALL_NU', 0.34)
    alpha_th = getattr(config, 'WALL_ALPHA', 17.0e-6)
    T_in = getattr(config, 'WALL_TEMP', 800.0)

    # Gradient temperatury na grubosci scianki z prawa Fouriera: dT = q * t / k_wall
    dT = (q_wall * t) / k_wall

    x_i = df['x'].values
    r_i = df['y'].values
    dr_dx = np.gradient(r_i, x_i)
    d2r_dx2 = np.gradient(dr_dx, x_i)
    cos_a = 1.0 / np.sqrt(1.0 + dr_dx**2)
    sin_a = dr_dx * cos_a

    # Wzorzec analityczny: powloka Laplace'a-Younga (cisnienie) + Lorenz-Timoshenko (termika)
    integrand_axial = -p * 2.0 * np.pi * r_i * dr_dx
    f_axial_cum = np.concatenate([[0.0], cumulative_trapezoid(integrand_axial, x_i)])
    f_section_x = f_axial_cum[-1] - f_axial_cum

    sigma_phi_mech = f_section_x / (2.0 * np.pi * r_i * t * cos_a)
    r1_inv = -d2r_dx2 * (cos_a**3)
    r2 = r_i / cos_a
    sigma_theta_mech = r2 * (p / t - sigma_phi_mech * r1_inv)
    sigma_r_mech = -0.5 * p

    # Naprezenia cieplne od zginania termicznego na wewnetrznej sciance (sciskanie dwuosiowe)
    sigma_th_b = - (E_mod * alpha_th * dT) / (2.0 * (1.0 - nu))
    sigma_theta_tot = sigma_theta_mech + sigma_th_b
    sigma_phi_tot = sigma_phi_mech + sigma_th_b

    sigma_vm_mech_ana = np.sqrt(0.5 * ((sigma_theta_mech - sigma_phi_mech)**2 + (sigma_phi_mech - sigma_r_mech)**2 + (sigma_r_mech - sigma_theta_mech)**2))
    sigma_vm_thermo_ana = np.sqrt(0.5 * ((sigma_theta_tot - sigma_phi_tot)**2 + (sigma_phi_tot - sigma_r_mech)**2 + (sigma_r_mech - sigma_theta_tot)**2))

    # Generacja 4-warstwowej siatki CAX4 w CalculiXie
    n_pts = len(x_i)
    n_layers = 4
    nodes_per_sec = n_layers + 1
    inner_nodes = [i * nodes_per_sec + 1 for i in range(n_pts)]

    inp_path = os.path.join(workdir, "nozzle_fea.inp")
    with open(inp_path, 'w') as f:
        f.write("*HEADING\nThermo-Mechanical Rocket Nozzle FEA\n*NODE\n")
        for i in range(n_pts):
            for j in range(nodes_per_sec):
                frac = j / float(n_layers)
                r_node = r_i[i] + frac * t * cos_a[i]
                x_node = x_i[i] - frac * t * sin_a[i]
                nid = i * nodes_per_sec + j + 1
                f.write(f"{nid}, {r_node:.6f}, {x_node:.6f}, 0.0\n")

        f.write("*ELEMENT, TYPE=CAX4, ELSET=EALL\n")
        elem_id = 1
        for i in range(n_pts - 1):
            for j in range(n_layers):
                n1 = i * nodes_per_sec + j + 1
                n2 = i * nodes_per_sec + (j + 1) + 1
                n3 = (i + 1) * nodes_per_sec + (j + 1) + 1
                n4 = (i + 1) * nodes_per_sec + j + 1
                f.write(f"{elem_id}, {n1}, {n2}, {n3}, {n4}\n")
                elem_id += 1

        f.write(f"*MATERIAL, NAME=CUCRZR\n*ELASTIC\n{E_mod:.1f}, {nu:.3f}\n")
        f.write(f"*EXPANSION, ZERO=0.0\n{alpha_th:.6e}\n")
        f.write("*SOLID SECTION, ELSET=EALL, MATERIAL=CUCRZR\n")
        # Podparcie przesuwne promieniowo na wlocie (swobodna ekspansja)
        f.write("*BOUNDARY\n")
        for j in range(nodes_per_sec):
            f.write(f"{j+1}, 2, 2, 0.0\n")

        f.write("*INITIAL CONDITIONS, TYPE=TEMPERATURE\n")
        for nid in range(1, n_pts * nodes_per_sec + 1):
            f.write(f"{nid}, 0.0\n")

        f.write("*STEP\n*STATIC\n*TEMPERATURE\n")
        for i in range(n_pts):
            for j in range(nodes_per_sec):
                frac = j / float(n_layers)
                # Rozklad odchylenia temperatury wzgledem sredniej temperatury przekroju
                t_rel = dT[i] * (0.5 - frac)
                nid = i * nodes_per_sec + j + 1
                f.write(f"{nid}, {t_rel:.3f}\n")

        f.write("*DLOAD\n")
        for i in range(n_pts - 1):
            p_elem = 0.5 * (p[i] + p[i+1])
            inner_eid = i * n_layers + 1
            f.write(f"{inner_eid}, P4, {p_elem:.2f}\n")
        f.write("*NODE FILE\nU, S, NT\n*END STEP\n")

    frd_path = os.path.join(workdir, "nozzle_fea.frd")
    sigma_vm_ccx = None
    if shutil.which("ccx"):
        subprocess.run(["ccx", "nozzle_fea"], cwd=workdir, check=True)
        if os.path.exists(frd_path):
            sigma_vm_ccx = parse_frd_surface_stresses(frd_path, n_pts, nodes_per_sec)
            valid = slice(5, -5)
            err_th = np.mean(np.abs(sigma_vm_ccx[valid] - sigma_vm_thermo_ana[valid]) / sigma_vm_thermo_ana[valid]) * 100.0
            print(f"[WALIDACJA TERMOMECHANICZNA FEA] Sredni blad CalculiX vs teoria Lorenza-Timoshenki: {err_th:.2f}%")

    plt.figure(figsize=(10, 5))
    plt.plot(x_i, sigma_vm_mech_ana / 1e6, 'b--', linewidth=1.4, label="Tylko ciśnienie (Laplace-Young) [MPa]")
    plt.plot(x_i, sigma_vm_thermo_ana / 1e6, 'g-', linewidth=1.8, label="Teoria termosprężysta (Lorenz-Timoshenko) [MPa]")
    if sigma_vm_ccx is not None:
        plt.plot(x_i, sigma_vm_ccx / 1e6, 'ro', markersize=3.5, label="CalculiX MES (nozzle_fea.frd) [MPa]")
    plt.xlabel("Oś X [m]")
    plt.ylabel("Naprężenie zredukowane von Misesa [MPa]")
    plt.title(f"Analiza termomechaniczna płaszcza CuCrZr (t = {t*1000:.1f} mm, T_w = {T_in:.0f} K)")
    plt.grid(True, ls="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "plot_fea_stress.png"), dpi=300)
    plt.close()