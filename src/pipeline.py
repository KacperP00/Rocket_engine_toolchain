import os
import sys
import csv
import time
import datetime
from datetime import datetime
from cfd.postprocessing import generate_validation_plots

start_time = time.time()

# Ścieżki do importów
src_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(src_dir)

root_dir = os.path.abspath(os.path.join(src_dir, '..'))
sys.path.append(root_dir)

import config
from chemistry.cantera_solver import calculate_chamber_conditions
from meshing.gmsh_generator import generate_nozzle_mesh
from cfd.su2_runner import prepare_su2_config, run_su2

def log_run_data(chamber_data, workdir):
    # Zapis parametrów wejściowych i termodynamicznych do pliku CSV
    results_dir = os.path.abspath(os.path.join(root_dir, 'results'))
    os.makedirs(results_dir, exist_ok=True)
    
    csv_path = os.path.join(results_dir, 'run_history.csv')
    file_exists = os.path.isfile(csv_path)
    
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # Generowanie nagłówków dla nowego pliku
        if not file_exists:
            writer.writerow([
                "Data", "Paliwo", "Utleniacz", "O/F", 
                "P_komory[Pa]", "T_komory[K]", "Gamma", "R[J/kgK]"
            ])
            
        # Zapis parametrów bieżącej symulacji
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            config.FUEL,
            config.OXIDIZER,
            config.O_F_RATIO,
            config.P_CHAMBER,
            round(chamber_data['T_c'], 2),
            round(chamber_data['gamma'], 4),
            round(chamber_data['R'], 2)
        ])
        
    print(f"--- [LOG] Zapisano parametry do: {csv_path} ---")

def main():
    print("=== ROZPOCZĘCIE SYMULACJI TOOLCHAINA ===")
    
    chamber_data = calculate_chamber_conditions()
    
    mesh_path = generate_nozzle_mesh()
    workdir = os.path.dirname(mesh_path)
    
    log_run_data(chamber_data, workdir)
    
    prepare_su2_config(chamber_data, workdir)
    run_su2(workdir)

    # 4. Postprocessing
    results_dir = os.path.abspath(os.path.join(root_dir, 'results'))
    generate_validation_plots(workdir, results_dir)
    
    print("=== SYMULACJA ZAKOŃCZONA ===")
    print(f"Wyniki CFD (.vtk) znajdują się w folderze: {workdir}")

    elapsed_time = time.time() - start_time
    formatted_time = str(datetime.timedelta(seconds=int(elapsed_time)))
    print(f"--- Czas całkowity: {formatted_time} ---")

if __name__ == "__main__":
    main()