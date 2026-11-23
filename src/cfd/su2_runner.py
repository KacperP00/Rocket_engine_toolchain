import os
import sys
import subprocess

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import config

def prepare_su2_config(cantera_data, workdir):
    # Populate SU2 configuration template with Cantera and config parameters
    print("--- [SU2 MODULE] Generating CFD configuration file ---")
    template_path = os.path.join(os.path.dirname(__file__), 'templates', 'base_config.cfg')
    output_path = os.path.join(workdir, 'run.cfg')

    with open(template_path, 'r') as f:
        cfg = f.read()

    cfg = cfg.replace('{GAMMA}', f"{cantera_data['gamma']:.4f}")
    cfg = cfg.replace('{R}', f"{cantera_data['R']:.2f}")
    cfg = cfg.replace('{T_C}', f"{cantera_data['T_c']:.2f}")
    cfg = cfg.replace('{P_C}', f"{cantera_data['p_c']:.2f}")
    cfg = cfg.replace('{MU}', f"{cantera_data['mu']:.8e}")
    cfg = cfg.replace('{PRANDTL}', f"{cantera_data['Pr']:.4f}")
    cfg = cfg.replace('{SUTHERLAND_S}', f"{cantera_data['sutherland_s']:.2f}")
    cfg = cfg.replace('{T_WALL}', f"{config.WALL_TEMP:.2f}")

    # Switch outlet boundary condition to static back-pressure for atmospheric operation
    p_amb = getattr(config, 'AMBIENT_PRESSURE', 0.0)
    if p_amb > 1.0:
        cfg = cfg.replace(
            "MARKER_SUPERSONIC_OUTLET= ( MARKER_OUTLET )",
            f"MARKER_OUTLET= ( MARKER_OUTLET, {p_amb:.2f} )"
        )
        cfg = cfg.replace(
            "FREESTREAM_PRESSURE= 10000.0",
            f"FREESTREAM_PRESSURE= {p_amb:.2f}"
        )

    with open(output_path, 'w') as f:
        f.write(cfg)

    print(f"SU2 config saved to: {output_path}")
    return output_path

def run_su2(workdir):
    # Execute SU2_CFD solver in working directory
    print("--- [SU2 MODULE] Running SU2_CFD ---")
    subprocess.run(["SU2_CFD", "run.cfg"], cwd=workdir, check=True)
    print("CFD simulation completed.\n")