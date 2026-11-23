import gmsh
import math
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import config

def generate_nozzle_mesh(lc_core=0.0030, bl_size=5.0e-7):
    # Generate 2D axisymmetric hybrid mesh using exact analytical Rao curves (no spline ringing)
    print(f"--- [GMSH MODULE] Generating exact Rao bell mesh (lc={lc_core:.4f} m, dy1={bl_size:.2e} m) ---")
    workdir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../workdir'))
    os.makedirs(workdir, exist_ok=True)

    R_t = config.THROAT_RADIUS
    R_e = R_t * math.sqrt(config.EXPANSION_RATIO)
    R_c = R_t * 2.0

    theta_conv = math.radians(getattr(config, 'THETA_CONV_DEG', 30.0))
    theta_n = math.radians(getattr(config, 'THETA_N_DEG', 26.0))
    theta_e = math.radians(getattr(config, 'THETA_E_DEG', 11.0))
    bell_frac = getattr(config, 'RAO_BELL_PERCENT', 80.0) / 100.0

    # Key geometric coordinates of exact Rao 80% contour
    R_u = 1.5 * R_t
    x_u_start = -R_u * math.sin(theta_conv)
    r_u_start = R_t + R_u * (1.0 - math.cos(theta_conv))

    R_f = 1.0 * R_t
    dr_f = R_f * (1.0 - math.cos(theta_conv))
    dx_f = R_f * math.sin(theta_conv)
    r_cone_start = R_c - dr_f
    dx_cone = (r_cone_start - r_u_start) / math.tan(theta_conv)
    x_cone_start = x_u_start - dx_cone
    x_fillet_start = x_cone_start - dx_f
    x_in = x_fillet_start - 0.02

    R_d = 0.382 * R_t
    x_N = R_d * math.sin(theta_n)
    r_N = R_t + R_d * (1.0 - math.cos(theta_n))

    L_15deg = (R_e - R_t) / math.tan(math.radians(15.0))
    x_out = bell_frac * L_15deg
    m_N = math.tan(theta_n)
    m_E = math.tan(theta_e)
    x_Q = (R_e - r_N + m_N * x_N - m_E * x_out) / (m_N - m_E)
    r_Q = r_N + m_N * (x_Q - x_N)

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("Rocket_Nozzle_Rao_Exact")

    # Centerline points
    p_sym_in = gmsh.model.geo.addPoint(x_in, 0.0, 0.0, lc_core)
    p_sym_th = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, lc_core * 0.5)
    p_sym_out = gmsh.model.geo.addPoint(x_out, 0.0, 0.0, lc_core * 0.6)

    # Exact wall anchor points and arc centers
    p_w0 = gmsh.model.geo.addPoint(x_in, R_c, 0.0, lc_core)
    p_w1 = gmsh.model.geo.addPoint(x_fillet_start, R_c, 0.0, lc_core)
    p_c1 = gmsh.model.geo.addPoint(x_fillet_start, R_c - R_f, 0.0, lc_core)
    p_w2 = gmsh.model.geo.addPoint(x_cone_start, r_cone_start, 0.0, lc_core)
    p_w3 = gmsh.model.geo.addPoint(x_u_start, r_u_start, 0.0, lc_core * 0.5)
    p_c2 = gmsh.model.geo.addPoint(0.0, R_t + R_u, 0.0, lc_core)
    p_w4 = gmsh.model.geo.addPoint(0.0, R_t, 0.0, lc_core * 0.45)
    p_c3 = gmsh.model.geo.addPoint(0.0, R_t + R_d, 0.0, lc_core)
    p_w5 = gmsh.model.geo.addPoint(x_N, r_N, 0.0, lc_core * 0.45)
    p_Q  = gmsh.model.geo.addPoint(x_Q, r_Q, 0.0, lc_core)
    p_w6 = gmsh.model.geo.addPoint(x_out, R_e, 0.0, lc_core * 0.6)

    # Exact piecewise C1-continuous nozzle wall primitives
    l_stub = gmsh.model.geo.addLine(p_w0, p_w1)
    l_fillet = gmsh.model.geo.addCircleArc(p_w1, p_c1, p_w2)
    l_cone = gmsh.model.geo.addLine(p_w2, p_w3)
    l_arc_up = gmsh.model.geo.addCircleArc(p_w3, p_c2, p_w4)
    l_arc_down = gmsh.model.geo.addCircleArc(p_w4, p_c3, p_w5)
    l_bell = gmsh.model.geo.addBezier([p_w5, p_Q, p_w6])
    wall_curves = [l_stub, l_fillet, l_cone, l_arc_up, l_arc_down, l_bell]

    l_inlet = gmsh.model.geo.addLine(p_sym_in, p_w0)
    l_outlet = gmsh.model.geo.addLine(p_w6, p_sym_out)
    l_sym_div = gmsh.model.geo.addLine(p_sym_out, p_sym_th)
    l_sym_conv = gmsh.model.geo.addLine(p_sym_th, p_sym_in)

    loop = gmsh.model.geo.addCurveLoop([l_inlet] + wall_curves + [l_outlet, l_sym_div, l_sym_conv])
    surface = gmsh.model.geo.addPlaneSurface([loop])
    gmsh.model.geo.synchronize()

    # Structured quadrilateral boundary layer field
    bl_field = gmsh.model.mesh.field.add("BoundaryLayer")
    gmsh.model.mesh.field.setNumbers(bl_field, "CurvesList", wall_curves)
    gmsh.model.mesh.field.setNumbers(bl_field, "PointsList", [p_w0, p_w6])
    gmsh.model.mesh.field.setNumber(bl_field, "Size", bl_size)
    gmsh.model.mesh.field.setNumber(bl_field, "Ratio", 1.15)
    gmsh.model.mesh.field.setNumber(bl_field, "Thickness", 0.003)
    gmsh.model.mesh.field.setNumber(bl_field, "Quads", 1)
    gmsh.option.setNumber("Mesh.BoundaryLayerFanElements", 0)
    gmsh.model.mesh.field.setAsBoundaryLayer(bl_field)

    # Local refinement box in divergent shock/separation zone
    box_field = gmsh.model.mesh.field.add("Box")
    gmsh.model.mesh.field.setNumber(box_field, "VIn", lc_core * 0.55)
    gmsh.model.mesh.field.setNumber(box_field, "VOut", lc_core)
    gmsh.model.mesh.field.setNumber(box_field, "XMin", 0.12)
    gmsh.model.mesh.field.setNumber(box_field, "XMax", x_out + 0.01)
    gmsh.model.mesh.field.setNumber(box_field, "YMin", 0.0)
    gmsh.model.mesh.field.setNumber(box_field, "YMax", R_e + 0.01)
    gmsh.model.mesh.field.setNumber(box_field, "Thickness", 0.02)

    # Throat curvature refinement box
    throat_box = gmsh.model.mesh.field.add("Box")
    gmsh.model.mesh.field.setNumber(throat_box, "VIn", lc_core * 0.45)
    gmsh.model.mesh.field.setNumber(throat_box, "VOut", lc_core)
    gmsh.model.mesh.field.setNumber(throat_box, "XMin", -0.03)
    gmsh.model.mesh.field.setNumber(throat_box, "XMax", 0.03)
    gmsh.model.mesh.field.setNumber(throat_box, "YMin", 0.0)
    gmsh.model.mesh.field.setNumber(throat_box, "YMax", 0.07)
    gmsh.model.mesh.field.setNumber(throat_box, "Thickness", 0.02)

    min_field = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", [box_field, throat_box])
    gmsh.model.mesh.field.setAsBackgroundMesh(min_field)

    gmsh.option.setNumber("Mesh.Algorithm", 8)

    gmsh.model.addPhysicalGroup(1, [l_inlet], name="MARKER_INLET")
    gmsh.model.addPhysicalGroup(1, [l_outlet], name="MARKER_OUTLET")
    gmsh.model.addPhysicalGroup(1, wall_curves, name="MARKER_WALL")
    gmsh.model.addPhysicalGroup(1, [l_sym_div, l_sym_conv], name="MARKER_SYM")
    gmsh.model.addPhysicalGroup(2, [surface], name="FLUID_DOMAIN")

    gmsh.model.mesh.generate(2)
    mesh_path = os.path.join(workdir, "nozzle.su2")
    gmsh.write(mesh_path)

    print(f"Exact Rao mesh generated: {mesh_path}")
    gmsh.finalize()
    return mesh_path

if __name__ == "__main__":
    generate_nozzle_mesh()