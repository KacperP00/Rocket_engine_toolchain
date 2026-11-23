# config.py

# Thermodynamic parameters (Cantera)
P_CHAMBER = 20.0 * 101325.0  # Chamber pressure [Pa]
FUEL = 'CH4'                 # Fuel species
OXIDIZER = 'O2'              # Oxidizer species
O_F_RATIO = 3.2              # Oxidizer-to-fuel mass ratio [-]
INLET_TEMP = 298.15          # Propellant inlet temperature [K]

# Nozzle geometry parameters (Rao 80% bell contour)
THROAT_RADIUS = 0.05         # Throat radius R_t [m]
EXPANSION_RATIO = 10.0       # Exit-to-throat area ratio Ae/At [-]
RAO_BELL_PERCENT = 80.0      # Bell length relative to 15-deg cone [%]
THETA_CONV_DEG = 30.0        # Convergent cone half-angle [deg]
THETA_N_DEG = 26.0           # Initial divergent inflection angle [deg]
THETA_E_DEG = 11.0           # Nozzle exit lip angle [deg]

# Ambient operating conditions
AMBIENT_PRESSURE = 101325.0  # Ambient pressure [Pa] (101325.0 = sea level, 0.0 = vacuum)

# Wall thermal and regenerative cooling parameters (Stage 3)
WALL_TEMP = 800.0              # Reference isothermal CFD wall temperature [K]
COOLANT_INLET_TEMP = 120.0     # Supercritical CH4 inlet temperature at exit [K]
COOLANT_INLET_PRESSURE = 60e5  # Supercritical CH4 inlet manifold pressure [Pa]
COOLANT_CHANNELS = 80          # Number of regenerative cooling channels [-]
CHANNEL_HEIGHT_THROAT = 0.0016 # Channel height at nozzle throat [m]
CHANNEL_HEIGHT_CONV = 0.0022   # Channel height at chamber inlet [m]
CHANNEL_HEIGHT_BELL = 0.0022   # Channel height at nozzle exit [m]
RIB_PITCH_FRACTION = 0.42      # Rib width fraction of local circumferential pitch [-]
MAX_CONV_CHANNEL_WIDTH = 0.0036 # Maximum channel width in convergent chamber [m]
CHANNEL_ROUGHNESS = 5.0e-6     # Milled CuCrZr channel surface roughness [m]

# Structural bimetallic wall parameters (Stage 4: CuCrZr liner + Inconel 718 jacket)
LINER_THICKNESS = 0.0010     # Inner CuCrZr hot-wall thickness [m]
JACKET_THICKNESS = 0.0030    # Outer Inconel 718 structural jacket thickness [m]
WALL_K = 320.0               # CuCrZr thermal conductivity [W/(m*K)]
WALL_E = 128e9               # CuCrZr room-temperature Young's modulus [Pa]
WALL_NU = 0.34               # CuCrZr Poisson's ratio [-]
WALL_ALPHA = 17.0e-6         # CuCrZr thermal expansion coefficient [1/K]
JACKET_E = 200e9             # Inconel 718 Young's modulus [Pa]
JACKET_NU = 0.29             # Inconel 718 Poisson's ratio [-]
JACKET_ALPHA = 13.0e-6       # Inconel 718 thermal expansion coefficient [1/K]