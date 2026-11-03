import gmsh
import math
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import config

def generate_nozzle_mesh():
    print("--- [MODUŁ GMSH] Generowanie siatki z warstwą przyścienną (RANS) ---")
    
    workdir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../workdir'))
    os.makedirs(workdir, exist_ok=True)
    
    R_t = config.THROAT_RADIUS
    R_e = R_t * math.sqrt(config.EXPANSION_RATIO)
    L_d = config.NOZZLE_LENGTH
    R_c = R_t * 2.0
    L_c = 0.1
    
    gmsh.initialize()
    gmsh.model.add("Rocket_Nozzle_RANS")
    
    # Bazowy rozmiar elementu dla rdzenia przepływu
    lc = 0.004
    
    p_sym_in = gmsh.model.geo.addPoint(-L_c, 0, 0, lc)
    p_sym_th = gmsh.model.geo.addPoint(0, 0, 0, lc/2)
    p_sym_out = gmsh.model.geo.addPoint(L_d, 0, 0, lc)
    
    wall_points = []
    num_points_conv = 20
    num_points_div = 40
    
    for i in range(num_points_conv):
        x = -L_c + (L_c / num_points_conv) * i
        y = R_t + 0.5 * (R_c - R_t) * (1.0 - math.cos(math.pi * abs(x) / L_c))
        wall_points.append(gmsh.model.geo.addPoint(x, y, 0, lc))
        
    wall_points.append(gmsh.model.geo.addPoint(0, R_t, 0, lc/4))
    
    for i in range(1, num_points_div + 1):
        x = (L_d / num_points_div) * i
        y = R_t + 0.5 * (R_e - R_t) * (1.0 - math.cos(math.pi * x / L_d))
        wall_points.append(gmsh.model.geo.addPoint(x, y, 0, lc))
        
    l_wall = gmsh.model.geo.addSpline(wall_points)
    
    l_inlet = gmsh.model.geo.addLine(p_sym_in, wall_points[0])
    l_outlet = gmsh.model.geo.addLine(wall_points[-1], p_sym_out)
    l_sym_div = gmsh.model.geo.addLine(p_sym_out, p_sym_th)
    l_sym_conv = gmsh.model.geo.addLine(p_sym_th, p_sym_in)
    
    loop = gmsh.model.geo.addCurveLoop([l_inlet, l_wall, l_outlet, l_sym_div, l_sym_conv])
    surface = gmsh.model.geo.addPlaneSurface([loop])
    
    gmsh.model.geo.synchronize()
    
    # Pobranie ID punktów narożnych (styk ścianki z wlotem i wylotem)
    p_inlet_top = wall_points[0]
    p_outlet_top = wall_points[-1]

    # Wymuszenie czworokątnej siatki przyściennej dla warunku y+ <= 1
    bl_field = gmsh.model.mesh.field.add("BoundaryLayer")
    gmsh.model.mesh.field.setNumbers(bl_field, "CurvesList", [l_wall])
    
    # KLUCZOWE: Wskazanie punktów końcowych warstwy.
    # To komenda, która mówi Gmshowi: "Zanim zaczniesz budować 2D, weź krzywe
    # dotykające tych punktów (wlot i wylot) i zagęść je od rozmiaru 1e-5 
    # ze współczynnikiem 1.2, aby warstwa miała się do czego przyczepić".
    gmsh.model.mesh.field.setNumbers(bl_field, "PointsList", [p_inlet_top, p_outlet_top])
    
    gmsh.model.mesh.field.setNumber(bl_field, "Size", 1e-5)       # Grubość pierwszej komórki
    gmsh.model.mesh.field.setNumber(bl_field, "Ratio", 1.2)       # Przyrost objętości
    gmsh.model.mesh.field.setNumber(bl_field, "Thickness", 0.005) # Całkowita grubość wytłoczenia
    gmsh.model.mesh.field.setNumber(bl_field, "Quads", 1)
    
    # Wyłączenie generowania "wachlarzy" w narożnikach
    gmsh.option.setNumber("Mesh.BoundaryLayerFanElements", 0)
    
    # Ustawiamy pole TYLKO jako warstwę przyścienną (nie jako tło!)
    gmsh.model.mesh.field.setAsBoundaryLayer(bl_field)
    
    # Zmiana algorytmu 2D na Frontal-Delaunay (najskuteczniej łączy quads z trójkątami)
    gmsh.option.setNumber("Mesh.Algorithm", 8)
    
    gmsh.model.addPhysicalGroup(1, [l_inlet], name="MARKER_INLET")
    gmsh.model.addPhysicalGroup(1, [l_outlet], name="MARKER_OUTLET")
    gmsh.model.addPhysicalGroup(1, [l_wall], name="MARKER_WALL")
    gmsh.model.addPhysicalGroup(1, [l_sym_div, l_sym_conv], name="MARKER_SYM")
    gmsh.model.addPhysicalGroup(2, [surface], name="FLUID_DOMAIN")
    
    gmsh.model.mesh.generate(2)
    mesh_path = os.path.join(workdir, "nozzle.su2")
    gmsh.write(mesh_path)
    
    print(f"Siatka RANS wygenerowana: {mesh_path}")
    gmsh.finalize()
    return mesh_path

if __name__ == "__main__":
    generate_nozzle_mesh()