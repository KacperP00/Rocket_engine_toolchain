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

# ==========================================
# PARAMETRY CHLODZENIA I MATERIALU SCIANKI (ETAP 1)
# ==========================================
WALL_TEMP = 800.0          # Temperatura wewnetrznej scianki dyszy [K]
WALL_THICKNESS = 0.004     # Grubosc plaszcza [m]
WALL_K = 320.0             # Przewodnosc cieplna stopu CuCrZr [W/(m*K)]
WALL_E = 128e9             # Modul Younga CuCrZr [Pa]
WALL_NU = 0.34             # Liczba Poissona CuCrZr [-]
WALL_ALPHA = 17.0e-6       # Wspolczynnik rozszerzalnosci cieplnej [1/K]

# ==========================================
# WARUNKI OTOCZENIA (ETAP 3: POZIOM MORZA / PROZNIA)
# ==========================================
AMBIENT_PRESSURE = 101325.0  # Cisnienie otoczenia [Pa] (101325.0 = poziom morza, 0.0 = proznia)