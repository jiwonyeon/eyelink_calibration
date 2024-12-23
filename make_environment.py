import os
import subprocess
import sys

def create_virtual_environment(env_name, requirements_file):
    # Create a virtual environment
    print(f"Creating virtual environment: {env_name}...")
    subprocess.run([sys.executable, "-m", "venv", env_name], check=True)
    
    # Activate the virtual environment
    if os.name == 'nt':  # Windows
        activate_script = os.path.join(env_name, "Scripts", "activate.bat")
    else:  # macOS/Linux
        activate_script = os.path.join(env_name, "bin", "activate")
    
    print(f"Virtual environment created. Activate it using: {activate_script}")
    
    # Install requirements
    print("Installing dependencies from requirements.txt...")
    pip_executable = os.path.join(env_name, "Scripts", "pip") if os.name == 'nt' else os.path.join(env_name, "bin", "pip")
    subprocess.run([pip_executable, "install", "-r", requirements_file], check=True)
    print("Dependencies installed successfully.")

if __name__ == "__main__":
    env_name = "jiwon"  # You can change the default environment name
    requirements_file = "requirements.txt"  # Ensure this file exists in your current directory
    
    # Check if the requirements.txt file exists
    if not os.path.exists(requirements_file):
        print(f"Error: {requirements_file} not found!")
        sys.exit(1)
    
    create_virtual_environment(env_name, requirements_file)