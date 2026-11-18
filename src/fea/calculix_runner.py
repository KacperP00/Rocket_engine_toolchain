import os
import subprocess
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

def parse_frd_stresses(frd_path, inner_node_ids):
    # Ekstrakcja 6 skladowych tensora naprezen dla wezlow wewnetrznych z pliku .frd
    stresses = {}
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
                    if node_id in inner_node_ids:
                        vals = [float(line[13 + 12*k : 13 + 12*(k+1)]) for k in range(6)]
                        sxx, syy, szz, sxy, syz, szx = vals
                        vm = np.sqrt(0.5 * ((sxx - syy)**2 + (syy - szz)**2 + (szz - sxx)**2 + 6.0 * (sxy**2 + syz**2 + szx**2)))
                        stresses[node_id] = vm
    return [stresses.get(nid, np.nan) for nid in inner_node_ids]

def run_fea_analysis(workdir, results_dir, cantera_data, wall_thickness=0.004):
    # Wczytanie rozkladu cisnienia na sciance z SU2
    surf_file = os.path.join(workdir, "surface_flow.csv")
    if not os.path.exists(surf_file):
        print("[MODUL FEA] Brak pliku surface_flow.csv. Analiza pominieta.")
        return

    df = pd.read_csv(surf_file)
    df.columns = df.columns.str.strip().str.replace('"', '')
    df = df.sort_values(by="x").reset_index(drop=True)

    gamma = cantera_data['gamma']
    rho = df['Density'].values
    v2 = (df['Momentum_x'].values / rho)**2 + (df['Momentum_y'].values / rho)**2
    p = (gamma - 1.0) * (df['Energy'].values - 0.5 * rho * v2)

    x_i = df['x'].values
    r_i = df['y'].values

    # Obliczenie nachylenia i krzywizn powloki osiowosymetrycznej
    dr_dx = np.gradient(r_i, x_i)
    d2r_dx2 = np.gradient(dr_dx, x_i)
    cos_a = 1.0 / np.sqrt(1.0 + dr_dx**2)
    sin_a = dr_dx * cos_a

    # Rzutowanie wezlow zewnetrznych wzdluz wektora normalnego (stala grubosc prostopadla t)
    r_o = r_i + wall_thickness * cos_a
    x_o = x_i - wall_thickness * sin_a

    # Wzorzec analityczny 1: klasyczny walec Lamego
    sigma_theta_lame = p * (r_i**2 + (r_i + wall_thickness)**2) / ((r_i + wall_thickness)**2 - r_i**2)
    sigma_vm_lame = np.sqrt(sigma_theta_lame**2 + sigma_theta_lame * p + p**2)

    # Wzorzec analityczny 2: teoria powlok o podwojnej krzywiznie Laplace'a-Younga
    integrand_axial = -p * 2.0 * np.pi * r_i * dr_dx
    f_axial_cum = np.concatenate([[0.0], cumulative_trapezoid(integrand_axial, x_i)])
    f_section_x = f_axial_cum[-1] - f_axial_cum

    sigma_phi = f_section_x / (2.0 * np.pi * r_i * wall_thickness * cos_a)
    r1_inv = -d2r_dx2 * (cos_a**3)
    r2 = r_i / cos_a
    sigma_theta_shell = r2 * (p / wall_thickness - sigma_phi * r1_inv)
    sigma_r_shell = -0.5 * p
    sigma_vm_shell = np.sqrt(0.5 * (
        (sigma_theta_shell - sigma_phi)**2 +
        (sigma_phi - sigma_r_shell)**2 +
        (sigma_r_shell - sigma_theta_shell)**2
    ))

    # Generacja pliku wejsciowego CalculiX (.inp)
    inp_path = os.path.join(workdir, "nozzle_fea.inp")
    n_pts = len(x_i)
    inner_nodes = [2*i + 1 for i in range(n_pts)]

    with open(inp_path, 'w') as f:
        f.write("*HEADING\nAxisymmetric Rocket Nozzle FEA\n*NODE\n")
        for i in range(n_pts):
            f.write(f"{2*i+1}, {r_i[i]:.6f}, {x_i[i]:.6f}, 0.0\n")
            f.write(f"{2*i+2}, {r_o[i]:.6f}, {x_o[i]:.6f}, 0.0\n")

        f.write("*ELEMENT, TYPE=CAX4, ELSET=EALL\n")
        for i in range(n_pts - 1):
            n1, n2 = 2*i + 1, 2*i + 2
            n3, n4 = 2*(i+1) + 2, 2*(i+1) + 1
            f.write(f"{i+1}, {n1}, {n2}, {n3}, {n4}\n")

        f.write("*MATERIAL, NAME=INCONEL718\n*ELASTIC\n200e9, 0.29\n")
        f.write("*SOLID SECTION, ELSET=EALL, MATERIAL=INCONEL718\n")
        f.write("*BOUNDARY\n1, 2, 2, 0.0\n2, 2, 2, 0.0\n")
        f.write("*STEP\n*STATIC\n*DLOAD\n")
        for i in range(n_pts - 1):
            p_elem = 0.5 * (p[i] + p[i+1])
            f.write(f"{i+1}, P4, {p_elem:.2f}\n")
        f.write("*NODE FILE\nU, S\n*END STEP\n")

    frd_path = os.path.join(workdir, "nozzle_fea.frd")
    sigma_vm_ccx = None
    if shutil.which("ccx"):
        subprocess.run(["ccx", "nozzle_fea"], cwd=workdir, check=True)
        if os.path.exists(frd_path):
            sigma_vm_ccx = np.array(parse_frd_stresses(frd_path, inner_nodes))
            valid = slice(5, -5)
            err_lame = np.mean(np.abs(sigma_vm_ccx[valid] - sigma_vm_lame[valid]) / sigma_vm_lame[valid]) * 100.0
            err_shell = np.mean(np.abs(sigma_vm_ccx[valid] - sigma_vm_shell[valid]) / sigma_vm_shell[valid]) * 100.0
            print(f"[WALIDACJA FEA] Blad wzgledem walca Lamego: {err_lame:.2f}% | Blad wzgledem powloki Laplace'a-Younga: {err_shell:.2f}%")

    plt.figure(figsize=(10, 5))
    plt.plot(x_i, sigma_vm_lame / 1e6, 'k--', linewidth=1.2, alpha=0.6, label="Walec Lamégo (1D) [MPa]")
    plt.plot(x_i, sigma_vm_shell / 1e6, 'g-', linewidth=1.8, label="Powłoka Laplace'a-Younga (2D) [MPa]")
    if sigma_vm_ccx is not None:
        plt.plot(x_i, sigma_vm_ccx / 1e6, 'ro', markersize=4, label="CalculiX MES (nozzle_fea.frd) [MPa]")
    plt.plot(x_i, p / 1e5, 'b:', linewidth=1.2, label="Ciśnienie statyczne [bar]")
    plt.xlabel("Oś X [m]")
    plt.ylabel("Wartość [MPa / bar]")
    plt.title(f"Walidacja MES (CalculiX vs Teoria Powłokowa, t = {wall_thickness*1000:.1f} mm)")
    plt.grid(True, ls="--")
    plt.legend()
    plt.tight_layout()
    fea_plot = os.path.join(results_dir, "plot_fea_stress.png")
    plt.savefig(fea_plot, dpi=300)
    plt.close()