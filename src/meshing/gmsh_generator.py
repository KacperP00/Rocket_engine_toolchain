import gmsh
import math
import sys
import os

# Podłączenie pliku konfiguracyjnego
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import config

def generate_nozzle_mesh():
    print("--- [MODUŁ GMSH] Generowanie siatki CFD 2D ---")
    
    # Upewniamy się, że folder na pliki tymczasowe istnieje
    workdir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../workdir'))
    os.makedirs(workdir, exist_ok=True)
    
    # 1. Wyliczenie brakujących parametrów geometrii
    # Pole przekroju jest proporcjonalne do kwadratu promienia (Ae = At * exp_ratio -> Re = Rt * sqrt(exp_ratio))
    R_t = config.THROAT_RADIUS
    R_e = R_t * math.sqrt(config.EXPANSION_RATIO)
    L_d = config.NOZZLE_LENGTH
    
    # Arbitralne parametry sekcji zbieżnej (dla uproszczenia pierwszego modelu)
    R_c = R_t * 2.0  # Promień komory (wlotu)
    L_c = 0.1        # Długość sekcji zbieżnej
    
    # Inicjalizacja Gmsh
    gmsh.initialize()
    gmsh.model.add("Rocket_Nozzle")
    
    # Globalny rozmiar oczka siatki (im mniej, tym gęstsza)
    lc = 0.005 

    # 2. Definiowanie punktów (x, y, z, rozmiar_oczka)
    # Oś symetrii (dół)
    p1 = gmsh.model.geo.addPoint(-L_c, 0, 0, lc)  # Wlot - środek
    p2 = gmsh.model.geo.addPoint(0, 0, 0, lc/2)   # Gardło - środek (zagęszczamy lc/2)
    p3 = gmsh.model.geo.addPoint(L_d, 0, 0, lc)   # Wylot - środek
    
    # Profil ścianki (góra)
    p4 = gmsh.model.geo.addPoint(L_d, R_e, 0, lc) # Wylot - ścianka
    p5 = gmsh.model.geo.addPoint(0, R_t, 0, lc/4) # Gardło - ścianka (bardzo gęsto, duże gradienty w CFD)
    p6 = gmsh.model.geo.addPoint(-L_c, R_c, 0, lc)# Wlot - ścianka
    
    # 3. Łączenie punktów w linie
    l_inlet = gmsh.model.geo.addLine(p1, p6)
    l_wall_conv = gmsh.model.geo.addLine(p6, p5)
    l_wall_div = gmsh.model.geo.addLine(p5, p4)
    l_outlet = gmsh.model.geo.addLine(p4, p3)
    l_sym_div = gmsh.model.geo.addLine(p3, p2)
    l_sym_conv = gmsh.model.geo.addLine(p2, p1)
    
    # 4. Tworzenie powierzchni płynu
    loop = gmsh.model.geo.addCurveLoop([l_inlet, l_wall_conv, l_wall_div, l_outlet, l_sym_div, l_sym_conv])
    surface = gmsh.model.geo.addPlaneSurface([loop])
    
    # Synchronizacja geometrii przed tworzeniem grup fizycznych
    gmsh.model.geo.synchronize()
    
    # 5. Grupy fizyczne - KLUCZOWE DLA SU2
    # Gmsh nada nazwy krawędziom, które potem wczyta SU2 do narzucenia warunków brzegowych
    gmsh.model.addPhysicalGroup(1, [l_inlet], name="MARKER_INLET")
    gmsh.model.addPhysicalGroup(1, [l_outlet], name="MARKER_OUTLET")
    gmsh.model.addPhysicalGroup(1, [l_wall_conv, l_wall_div], name="MARKER_WALL")
    gmsh.model.addPhysicalGroup(1, [l_sym_div, l_sym_conv], name="MARKER_SYM")
    gmsh.model.addPhysicalGroup(2, [surface], name="FLUID_DOMAIN")
    
    # 6. Generowanie siatki 2D i eksport
    gmsh.model.mesh.generate(2)
    
    mesh_path = os.path.join(workdir, "nozzle.su2")
    gmsh.write(mesh_path)
    
    print(f"Siatka wygenerowana: {mesh_path}")
    print("-------------------------------------------------------------\n")
    
    # Możesz to odkomentować, żeby aplikacja otworzyła się w oknie graficznym
    gmsh.fltk.run() 
    
    gmsh.finalize()
    
    return mesh_path

if __name__ == "__main__":
    generate_nozzle_mesh()