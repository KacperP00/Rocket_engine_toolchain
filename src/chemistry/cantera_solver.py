import cantera as ct
import numpy as np
import sys
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="cantera")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import config

def compute_isentropic_expansion(gas_init, p_c, h_0, s_0, equilibrium=True, n_points=80):
    # Obliczenie profilu ekspansji izentropowej (S = const) dla przeplywu rownowagowego lub zamrozonego
    pressures = np.logspace(np.log10(p_c * 0.999), np.log10(p_c * 0.005), n_points)
    Y_frozen = gas_init.Y.copy()

    p_list, T_list, rho_list, v_list, a_list, M_list, mflux_list = [], [], [], [], [], [], []

    for p in pressures:
        gas = ct.Solution('gri30.yaml')
        gas.SPX = s_0, p, gas_init.X
        if equilibrium:
            # Przesuniecie rownowagi chemicznej przy stalej entropii i cisnieniu (rekombinacja rodnikow)
            gas.equilibrate('SP')
        else:
            # Zamrozony sklad chemiczny z komory spalania
            gas.SPY = s_0, p, Y_frozen

        dh = max(h_0 - gas.enthalpy_mass, 0.0)
        v = np.sqrt(2.0 * dh)

        # Predkosc dzwieku i strumien masy rho * v
        gamma_loc = gas.cp_mass / gas.cv_mass
        r_loc = ct.gas_constant / gas.mean_molecular_weight
        a = np.sqrt(gamma_loc * r_loc * gas.T)
        mach = v / a
        mflux = gas.density * v

        p_list.append(p)
        T_list.append(gas.T)
        rho_list.append(gas.density)
        v_list.append(v)
        a_list.append(a)
        M_list.append(mach)
        mflux_list.append(mflux)

    mflux_arr = np.array(mflux_list)
    idx_throat = np.argmax(mflux_arr)
    mflux_max = mflux_arr[idx_throat]
    area_ratios = mflux_max / np.maximum(mflux_arr, 1e-6)

    return {
        "p": np.array(p_list),
        "T": np.array(T_list),
        "rho": np.array(rho_list),
        "v": np.array(v_list),
        "M": np.array(M_list),
        "A_ratio": area_ratios,
        "idx_throat": idx_throat,
        "mflux_throat": mflux_max
    }

def calculate_chamber_conditions():
    print("--- [MODUŁ CANTERA] Obliczanie parametrów termochemicznych, ściennych i ekspansji ---")
    gas = ct.Solution('gri30.yaml')
    gas.transport_model = 'mixture-averaged'

    mass_fractions = f"{config.FUEL}:1.0, {config.OXIDIZER}:{config.O_F_RATIO}"
    gas.TPY = config.INLET_TEMP, config.P_CHAMBER, mass_fractions
    gas.equilibrate('HP')

    T_c = gas.T
    p_c = gas.P
    h_0 = gas.enthalpy_mass
    s_0 = gas.entropy_mass
    gamma = gas.cp_mass / gas.cv_mass
    R_specific = ct.gas_constant / gas.mean_molecular_weight
    mu = gas.viscosity
    k_tc = gas.thermal_conductivity
    cp = gas.cp_mass
    prandtl = (mu * cp) / k_tc

    # Wlasciwosci transportowe gazu przy schlodzonej sciance (T_wall) dla prawa Fouriera
    wall_temp = getattr(config, 'WALL_TEMP', 800.0)
    gas_wall = ct.Solution('gri30.yaml')
    gas_wall.transport_model = 'mixture-averaged'
    gas_wall.TPY = wall_temp, p_c, gas.Y
    mu_wall = gas_wall.viscosity
    k_wall = gas_wall.thermal_conductivity

    # Obliczenie krzywych ekspansji 1D: Zamrozona (Frozen) vs Rownowagowa (Shifting Equilibrium)
    exp_frozen = compute_isentropic_expansion(gas, p_c, h_0, s_0, equilibrium=False)
    exp_equil = compute_isentropic_expansion(gas, p_c, h_0, s_0, equilibrium=True)

    print(f"Temperatura w komorze (Tc): {T_c:.2f} K | Ciśnienie: {p_c / 101325:.2f} atm")
    print(f"Gamma (komora): {gamma:.4f} | Stała gazowa (R): {R_specific:.2f} J/(kg*K)")
    print(f"Przewodność (komora {T_c:.0f} K): {k_tc:.4f} W/(m*K) | (ścianka {wall_temp:.0f} K): {k_wall:.4f} W/(m*K)")
    print("-------------------------------------------------------------\n")

    return {
        "T_c": T_c,
        "p_c": p_c,
        "gamma": gamma,
        "R": R_specific,
        "mu": mu,
        "k": k_tc,
        "cp": cp,
        "Pr": prandtl,
        "mu_wall": mu_wall,
        "k_wall": k_wall,
        "exp_frozen": exp_frozen,
        "exp_equil": exp_equil
    }

if __name__ == "__main__":
    calculate_chamber_conditions()