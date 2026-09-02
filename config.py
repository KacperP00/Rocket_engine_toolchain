# config.py

# ==========================================
# PARAMETRY TERMODYNAMICZNE (CANTERA)
# ==========================================
P_CHAMBER = 20.0 * 101325  # Ciśnienie w komorze spalania [Pa] (tutaj 20 atm)
FUEL = 'CH4'               # Paliwo (Metan)
OXIDIZER = 'O2'            # Utleniacz (Tlen)
O_F_RATIO = 3.2            # Stosunek masowy utleniacza do paliwa (O/F)
INLET_TEMP = 298.15        # Początkowa temperatura substratów [K]

# ==========================================
# PARAMETRY GEOMETRII DYSZY (GMSH)
# ==========================================
# Te wartości przydadzą się w następnym kroku
THROAT_RADIUS = 0.05       # Promień gardła dyszy [m]
EXPANSION_RATIO = 10.0     # Stosunek pola przekroju wylotu do gardła (Ae/At)
NOZZLE_LENGTH = 0.30       # Długość sekcji rozbieżnej [m]