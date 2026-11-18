import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def verify_mass_balance(workdir, log_file=None):
    # Wczytanie wolumenu przeplywu z pliku restart.csv
    flow_path = os.path.join(workdir, 'restart.csv')
    if not os.path.exists(flow_path):
        return None
        
    df = pd.read_csv(flow_path)
    df.columns = df.columns.str.strip().str.replace('"', '')

    # Ekstrakcja krawedzi domeny (wlot i wylot)
    x_in = df['x'].min()
    x_out = df['x'].max()

    inlet_nodes = df[abs(df['x'] - x_in) < 1e-3].sort_values('y')
    outlet_nodes = df[abs(df['x'] - x_out) < 1e-3].sort_values('y')

    def calc_mass_flow(nodes):
        # Calkowanie po promieniu dla domeny osiowosymetrycznej
        y = nodes['y'].values
        rho = nodes['Density'].values
        u = (nodes['Momentum_x'] / rho).values
        integrand = 2 * np.pi * y * rho * u
        return np.trapezoid(integrand, y)

    m_in = calc_mass_flow(inlet_nodes)
    m_out = calc_mass_flow(outlet_nodes)
    error = abs(m_in - m_out) / abs(m_in) * 100.0

    msg = f"[DIAGNOSTYKA] Bilans masy -> Wlot: {m_in:.4f} kg/s | Wylot: {m_out:.4f} kg/s | Blad: {error:.2f}%"
    print(msg)
    if log_file:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(msg + "\n")
    return error

def verify_y_plus(workdir, log_file=None, mu_val=1.0012e-4):
    # Obliczenie y+ na podstawie scianki (surface_flow.csv) i pierwszej warstwy siatki (restart.csv)
    surf_path = os.path.join(workdir, 'surface_flow.csv')
    vol_path = os.path.join(workdir, 'restart.csv')
    if not os.path.exists(surf_path) or not os.path.exists(vol_path):
        return
        
    df_surf = pd.read_csv(surf_path)
    df_surf.columns = df_surf.columns.str.strip().str.replace('"', '')
    
    if 'Y_Plus' in df_surf.columns:
        y_plus_vals = df_surf['Y_Plus'].values
    else:
        df_vol = pd.read_csv(vol_path)
        df_vol.columns = df_vol.columns.str.strip().str.replace('"', '')
        
        # Odfiltrowanie wezlow lezacych bezposrednio na sciance (gdzie predkosc wynosi 0)
        vel_mag = np.sqrt(df_vol['Momentum_x']**2 + df_vol['Momentum_y']**2) / df_vol['Density']
        interior = df_vol[vel_mag > 1e-3].copy()
        interior['U'] = vel_mag[vel_mag > 1e-3]
        
        y_plus_list = []
        for _, w_row in df_surf.iterrows():
            dx = interior['x'].values - w_row['x']
            dy = interior['y'].values - w_row['y']
            dist = np.sqrt(dx**2 + dy**2)
            idx_min = np.argmin(dist)
            
            dn = dist[idx_min]
            u1 = interior['U'].iloc[idx_min]
            rho_w = w_row['Density']
            
            # y+ = (rho_w * u_tau * dn) / mu, gdzie u_tau = sqrt(mu * u1 / (rho_w * dn))
            y_p = np.sqrt(rho_w * u1 * dn / mu_val)
            y_plus_list.append(y_p)
        y_plus_vals = np.array(y_plus_list)

    y_max = np.max(y_plus_vals)
    y_mean = np.mean(y_plus_vals)
    msg = f"[DIAGNOSTYKA] Siatka przy sciance -> y+ max: {y_max:.3f} | y+ srednie: {y_mean:.3f}"
    
    if y_max > 5.0:
        msg += "\n[KRYTYCZNE] Wartosc y+ drastycznie przekracza 1. Zagesc siatke w Gmsh (zmniejsz Size)."
    elif y_max > 1.0:
        msg += "\n[OSTRZEZENIE] Wartosc y+ lokalnie powyzej 1 (w gardle). Rozwaz zmniejszenie Size w Gmsh do 3e-6."
        
    print(msg)
    if log_file:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(msg + "\n")

def plot_convergence(workdir, log_file=None):
    # Generowanie wykresu zbieznosci z pliku history.csv
    hist_path = os.path.join(workdir, 'history.csv')
    if not os.path.exists(hist_path):
        return
        
    df = pd.read_csv(hist_path)
    df.columns = df.columns.str.strip().str.replace('"', '')
    
    iter_col = 'Inner_Iter' if 'Inner_Iter' in df.columns else 'Iter'
    
    plt.figure(figsize=(10, 6))
    if 'rms[Rho]' in df.columns:
        plt.plot(df[iter_col], df['rms[Rho]'], label='rms[Rho]', color='blue', linewidth=1.5)
    if 'rms[RhoE]' in df.columns:
        plt.plot(df[iter_col], df['rms[RhoE]'], label='rms[RhoE]', color='red', linewidth=1.2, alpha=0.8)
    if 'rms[k]' in df.columns:
        plt.plot(df[iter_col], df['rms[k]'], label='rms[k]', color='orange', alpha=0.7)
    if 'rms[w]' in df.columns:
        plt.plot(df[iter_col], df['rms[w]'], label='rms[w] (omega)', color='green', alpha=0.7)
    elif 'rms[omega]' in df.columns:
        plt.plot(df[iter_col], df['rms[omega]'], label='rms[omega]', color='green', alpha=0.7)
        
    plt.title('Zbieznosc symulacji (Residua)')
    plt.xlabel('Iteracja')
    plt.ylabel('Log10(Residuum)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    plot_path = os.path.join(workdir, 'convergence.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    msg = f"[DIAGNOSTYKA] Wykres zbieznosci wygenerowany: {plot_path}"
    print(msg)
    if log_file:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(msg + "\n")

def verify_momentum_balance(workdir, cantera_data, log_file=None):
    # Weryfikacja bilansu pedu (sily ciagu) miedzy przekrojami brzegowymi a scianka
    flow_path = os.path.join(workdir, 'restart.csv')
    surf_path = os.path.join(workdir, 'surface_flow.csv')
    if not os.path.exists(flow_path) or not os.path.exists(surf_path):
        return

    df_vol = pd.read_csv(flow_path)
    df_vol.columns = df_vol.columns.str.strip().str.replace('"', '')
    df_wall = pd.read_csv(surf_path)
    df_wall.columns = df_wall.columns.str.strip().str.replace('"', '')
    df_wall = df_wall.sort_values('x').reset_index(drop=True)

    gamma = cantera_data['gamma']

    def section_thrust(nodes):
        # Calkowanie strumienia pedu i cisnienia statycznego po promieniu: int (rho*u^2 + p) * 2*pi*y dy
        y = nodes['y'].values
        rho = nodes['Density'].values
        u = (nodes['Momentum_x'] / rho).values
        v = (nodes['Momentum_y'] / rho).values
        p = (gamma - 1.0) * (nodes['Energy'].values - 0.5 * rho * (u**2 + v**2))
        integrand = (rho * u**2 + p) * 2.0 * np.pi * y
        return np.trapezoid(integrand, y)

    x_in, x_out = df_vol['x'].min(), df_vol['x'].max()
    inlet_nodes = df_vol[abs(df_vol['x'] - x_in) < 1e-3].sort_values('y')
    outlet_nodes = df_vol[abs(df_vol['x'] - x_out) < 1e-3].sort_values('y')

    f_in = section_thrust(inlet_nodes)
    f_out = section_thrust(outlet_nodes)

    # Sila osiowa od cisnienia na sciance: int p * 2*pi*y dy (z uwzglednieniem znaku dy/dx)
    y_w = df_wall['y'].values
    rho_w = df_wall['Density'].values
    v2_w = (df_wall['Momentum_x'] / rho_w)**2 + (df_wall['Momentum_y'] / rho_w)**2
    p_w = (gamma - 1.0) * (df_wall['Energy'].values - 0.5 * rho_w * v2_w)
    f_wall_x = np.trapezoid(p_w * 2.0 * np.pi * y_w, y_w)

    f_control_vol = f_in + f_wall_x
    err_mom = abs(f_out - f_control_vol) / abs(f_out) * 100.0

    msg = (
        f"[WALIDACJA CFD] Ciag w prozni (wylot): {f_out/1000:.3f} kN | "
        f"Suma sil (wlot + scianka): {f_control_vol/1000:.3f} kN | Blad bilansu pedu: {err_mom:.2f}%"
    )
    print(msg)
    if log_file:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(msg + "\n")