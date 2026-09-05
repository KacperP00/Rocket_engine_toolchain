import os
import subprocess

def prepare_su2_config(cantera_data, workdir):
    # Generuje plik .cfg dla SU2 na podstawie danych z Cantery
    print("--- [MODUŁ SU2] Generowanie pliku konfiguracyjnego CFD ---")
    
    template_path = os.path.join(os.path.dirname(__file__), 'templates', 'base_config.cfg')
    output_path = os.path.join(workdir, 'run.cfg')
    
    with open(template_path, 'r') as file:
        config_content = file.read()
    
    config_content = config_content.replace('{GAMMA}', f"{cantera_data['gamma']:.4f}")
    config_content = config_content.replace('{R}', f"{cantera_data['R']:.2f}")
    config_content = config_content.replace('{T_C}', f"{cantera_data['T_c']:.2f}")
    config_content = config_content.replace('{P_C}', f"{cantera_data['p_c']:.2f}")
    
    with open(output_path, 'w') as file:
        file.write(config_content)
        
    print(f"Zapisano gotową konfigurację do: {output_path}")
    return output_path

def run_su2(workdir):
    # Uruchamia proces SU2_CFD w podanym folderze roboczym
    print("--- [MODUŁ SU2] Uruchamianie solvera SU2_CFD ---")
    
    # Zakładamy, że komenda SU2_CFD jest dostępna w zmiennej PATH systemu.
    # Używamy cwd=workdir, aby SU2 wygenerowało pliki wynikowe w odpowiednim folderze.
    try:
        subprocess.run(["SU2_CFD", "run.cfg"], cwd=workdir, check=True)
        print("Obliczenia CFD zakończone sukcesem.")
    except subprocess.CalledProcessError as e:
        print(f"Błąd podczas uruchamiania SU2: {e}")
        print("Upewnij się, że SU2_CFD jest w zmiennej PATH środowiska.")
    print("-------------------------------------------------------------\n")