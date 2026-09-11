import gmsh
import math
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import config

def generate_nozzle_mesh():
    print("--- [MODUŁ GMSH] Generowanie wygładzonej siatki CFD 2D ---")
    
    workdir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../workdir'))
    os.makedirs(workdir, exist_ok=True)
    
    R_t = config.THROAT_RADIUS
    R_e = R_t * math.sqrt(config.EXPANSION_RATIO)
    L_d = config.NOZZLE_LENGTH
    
    R_c = R_t * 2.0
    L_c = 0.1
    
    gmsh.initialize()
    gmsh.model.add("Rocket_Nozzle_Smooth")
    
    lc = 0.004  # Nieco gęstsza siatka dla lepszej rozdzielczości gardła

    # Oś symetrii (dół)
    p1 = gmsh.model.geo.addPoint(-L_c, 0, 0, lc)
    p2 = gmsh.model.geo.addPoint(0, 0, 0, lc/2)
    p3 = gmsh.model.geo.addPoint(L_d, 0, 0, lc)
    
    # Profil ścianki górnej z punktami pośrednimi dla gładkiego przejścia (Spline)
    p_inlet = gmsh.model.geo.addPoint(-L_c, R_c, 0, lc)
    p_mid_conv = gmsh.model.geo.addPoint(-0.05, 0.5 * (R_c + R_t), 0, lc)
    p_throat = gmsh.model.geo.addPoint(0, R_t, 0, lc/4)  # Gardło - zagęszczenie
    p_mid_div = gmsh.model.geo.addPoint(L_d * 0.5, 0.5 * (R_t + R_e), 0, lc)
    p_outlet = gmsh.model.geo.addPoint(L_d, R_e, 0, lc)
    
    # Tworzenie gładkiej krzywej ścianki za pomocą B-Spline
    l_wall = gmsh.model.geo.addSpline([p_inlet, p_mid_conv, p_throat, p_mid_div, p_outlet])
    
    # Domknięcie pozostałych krawędzi domeny
    l_inlet = gmsh.model.geo.addLine(p1, p_inlet)
    l_outlet = gmsh.model.geo.addLine(p_outlet, p3)
    l_sym_div = gmsh.model.geo.addLine(p3, p2)
    l_sym_conv = gmsh.model.geo.addLine(p2, p1)
    
    # Pętla i powierzchnia płynu
    loop = gmsh.model.geo.addCurveLoop([l_inlet, l_wall, l_outlet, l_sym_div, l_sym_conv])
    surface = gmsh.model.geo.addPlaneSurface([loop])
    
    gmsh.model.geo.synchronize()
    
    # Grupy fizyczne dla SU2
    gmsh.model.addPhysicalGroup(1, [l_inlet], name="MARKER_INLET")
    gmsh.model.addPhysicalGroup(1, [l_outlet], name="MARKER_OUTLET")
    gmsh.model.addPhysicalGroup(1, [l_wall], name="MARKER_WALL")
    gmsh.model.addPhysicalGroup(1, [l_sym_div, l_sym_conv], name="MARKER_SYM")
    gmsh.model.addPhysicalGroup(2, [surface], name="FLUID_DOMAIN")
    
    gmsh.model.mesh.generate(2)
    
    mesh_path = os.path.join(workdir, "nozzle.su2")
    gmsh.write(mesh_path)
    
    print(f"Wygładzona siatka wygenerowana: {mesh_path}")
    print("-------------------------------------------------------------\n")
    
    gmsh.finalize()
    return mesh_path

if __name__ == "__main__":
    generate_nozzle_mesh()