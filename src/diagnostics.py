import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def verify_mass_balance(workdir, log_file=None):
    # Wczytanie wolumenu przeplywu
    flow_path = os.path.join(workdir, 'flow.csv')
    if not os.path.exists(flow_path):
        return None
        
    df = pd.read_csv(flow_path)
    df.columns = df.columns.str.strip()

    # Ekstrakcja krawedzi domeny (wlot i wylot)
    x_in = df['x'].min()
    x_out = df['x'].max()

    # Pobranie posortowanych wezlow na przekrojach
    inlet_nodes = df[abs(df['x'] - x_in) < 1e-3].sort_values('y')
    outlet_nodes = df[abs(df['x'] - x_out) < 1e-3].sort_values('y')

    def calc_mass_flow(nodes):
        # Calkowanie po promieniu dla domeny osiowosymetrycznej
        y = nodes['y'].values
        rho = nodes['Density'].values
        u = (nodes['Momentum_x'] / rho).values
        integrand = 2 * np.pi * y * rho * u
        return np.trapz(integrand, y)

    m_in = calc_mass_flow(inlet_nodes)
    m_out = calc_mass_flow(outlet_nodes)

    # Weryfikacja bledu numerycznego
    error = abs(m_in - m_out) / abs(m_in) * 100.0

    msg = f"[DIAGNOSTYKA] Bilans masy -> Wlot: {m_in:.4f} kg/s | Wylot: {m_out:.4f} kg/s | Blad: {error:.2f}%"
    print(msg)
    
    if log_file:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(msg + "\n")

    return error

def verify_y_plus(workdir, log_file=None):
    # Wczytanie danych ze scianki w celu oceny grubosci siatki
    surf_path = os.path.join(workdir, 'surface_flow.csv')
    if not os.path.exists(surf_path):
        return
        
    df = pd.read_csv(surf_path)
    df.columns = df.columns.str.strip()
    
    if 'Y_Plus' not in df.columns:
        msg = "[DIAGNOSTYKA] Brak kolumny Y_Plus. Sprawdz czy wlaczono solver RANS."
    else:
        y_max = df['Y_Plus'].max()
        y_mean = df['Y_Plus'].mean()
        
        msg = f"[DIAGNOSTYKA] Siatka przy sciance -> y+ max: {y_max:.3f} | y+ srednie: {y_mean:.3f}"
        
        # Krytyczna walidacja warunku brzegowego warstwy lepkiej
        if y_max > 5.0:
            msg += "\n[KRYTYCZNE] Wartosc y+ drastycznie przekracza 1. Zagesc siatke w Gmsh (zmniejsz Size) lub zweryfikuj predkosc."
        elif y_max > 1.0:
            msg += "\n[OSTRZEZENIE] Wartosc y+ lekko powyzej 1. Obliczenia tarcia i wymiany ciepla moga byc obarczone bledem."
            
    print(msg)
    if log_file:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(msg + "\n")

def plot_convergence(workdir, log_file=None):
    # Generowanie wykresu zbieznosci z przebiegu iteracji
    hist_path = os.path.join(workdir, 'history.csv')
    if not os.path.exists(hist_path):
        return
        
    df = pd.read_csv(hist_path)
    df.columns = df.columns.str.strip()
    
    iter_col = 'Inner_Iter' if 'Inner_Iter' in df.columns else 'Iter'
    
    plt.figure(figsize=(10, 6))
    if 'rms[Rho]' in df.columns:
        plt.plot(df[iter_col], df['rms[Rho]'], label='rms[Rho]', color='blue', linewidth=1.5)
        
    if 'rms[k]' in df.columns:
        plt.plot(df[iter_col], df['rms[k]'], label='rms[k]', color='orange', alpha=0.7)
        plt.plot(df[iter_col], df['rms[omega]'], label='rms[omega]', color='green', alpha=0.7)
        
    plt.title('Zbieznosc symulacji (Residua)')
    plt.xlabel('Iteracja')
    plt.ylabel('Log(Residuum)')
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