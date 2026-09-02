import cantera as ct
import sys
import os

# Dodajemy katalog nadrzędny do ścieżki, żeby Python widział plik config.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import config

def calculate_chamber_conditions():
    print("--- [MODUŁ CANTERA] Obliczanie parametrów komory spalania ---")
    
    # 1. Wczytanie mechanizmu reakcji
    # gri30.yaml to zoptymalizowany, wbudowany mechanizm dla metanu/gazu ziemnego
    gas = ct.Solution('gri30.yaml')
    
    # 2. Definicja ułamków masowych na podstawie O/F ratio z configu
    mass_fractions = f"{config.FUEL}:1.0, {config.OXIDIZER}:{config.O_F_RATIO}"
    
    # 3. Ustawienie stanu początkowego (Temperatura, Ciśnienie, Skład masowy Y)
    gas.TPY = config.INLET_TEMP, config.P_CHAMBER, mass_fractions
    
    # 4. Obliczenie równowagi chemicznej (stała entalpia H i ciśnienie P)
    # Odpowiada to idealnemu procesowi spalania w komorze rakietowej
    gas.equilibrate('HP')
    
    # 5. Ekstrakcja wyników
    T_c = gas.T                        # Temperatura [K]
    p_c = gas.P                        # Ciśnienie [Pa]
    gamma = gas.cp_mass / gas.cv_mass  # Stosunek ciepeł właściwych [-]
    
    # Specyficzna stała gazowa R = R_uni / Masa_molowa
    # cantera.gas_constant jest w [J/kmol/K], masa molowa w [kg/kmol]
    R_specific = ct.gas_constant / gas.mean_molecular_weight
    
    print(f"Temperatura (Tc):      {T_c:.2f} K")
    print(f"Ciśnienie (pc):        {p_c / 101325:.2f} atm")
    print(f"Stosunek ciepeł (gamma): {gamma:.4f}")
    print(f"Stała gazowa (R):      {R_specific:.2f} J/(kg*K)")
    print(f"Masa molowa spalin:    {gas.mean_molecular_weight:.2f} kg/kmol")
    print("-------------------------------------------------------------\n")
    
    # Zwracamy słownik, który później przekażemy do SU2 i Gmsh
    return {
        "T_c": T_c,
        "p_c": p_c,
        "gamma": gamma,
        "R": R_specific
    }

# Pozwala na testowanie samego tego modułu z konsoli
if __name__ == "__main__":
    calculate_chamber_conditions()