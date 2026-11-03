import cantera as ct
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import config

def calculate_chamber_conditions():
    print("--- [MODUŁ CANTERA] Obliczanie parametrów termochemicznych i transportowych ---")
    
    gas = ct.Solution('gri30.yaml')
    
    # Inicjalizacja modelu transportowego dla mieszanin
    gas.transport_model = 'Mix'
    
    mass_fractions = f"{config.FUEL}:1.0, {config.OXIDIZER}:{config.O_F_RATIO}"
    gas.TPY = config.INLET_TEMP, config.P_CHAMBER, mass_fractions
    
    gas.equilibrate('HP')
    
    T_c = gas.T
    p_c = gas.P
    gamma = gas.cp_mass / gas.cv_mass
    R_specific = ct.gas_constant / gas.mean_molecular_weight
    
    # Ekstrakcja właściwości transportowych dla solvera RANS
    mu = gas.viscosity            # Lepkość dynamiczna [Pa*s]
    k_tc = gas.thermal_conductivity # Przewodność cieplna [W/m/K]
    cp = gas.cp_mass              # Ciepło właściwe izobaryczne [J/kg/K]
    
    # Bezwymiarowa liczba Prandtla dla sprzężenia warstwy przyściennej
    prandtl = (mu * cp) / k_tc    
    
    print(f"Temperatura (Tc):      {T_c:.2f} K")
    print(f"Ciśnienie (pc):        {p_c / 101325:.2f} atm")
    print(f"Stosunek ciepeł (gamma): {gamma:.4f}")
    print(f"Stała gazowa (R):      {R_specific:.2f} J/(kg*K)")
    print(f"Lepkość dyn. (mu):     {mu:.6e} Pa*s")
    print(f"Przewodność (k):       {k_tc:.4f} W/(m*K)")
    print(f"Liczba Prandtla (Pr):  {prandtl:.4f}")
    print("-------------------------------------------------------------\n")
    
    return {
        "T_c": T_c,
        "p_c": p_c,
        "gamma": gamma,
        "R": R_specific,
        "mu": mu,
        "k": k_tc,
        "Pr": prandtl
    }

if __name__ == "__main__":
    calculate_chamber_conditions()