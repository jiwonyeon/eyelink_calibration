import os
import shutil
import sys

def remove_virtual_environment(env_name):
    """
    Remove the specified virtual environment directory.
    """
    if os.path.exists(env_name):
        print(f"Removing virtual environment: {env_name}...")
        shutil.rmtree(env_name)
        print(f"Virtual environment '{env_name}' has been removed.")
    else:
        print(f"Virtual environment '{env_name}' does not exist.")

def remove_python(python_path):
    """
    Remove Python installation directory.
    """
    if os.path.exists(python_path):
        print(f"Removing Python from: {python_path}...")
        shutil.rmtree(python_path)
        print("Python installation has been removed.")
    else:
        print(f"Python directory '{python_path}' does not exist.")

if __name__ == "__main__":
    # Define virtual environment name
    env_name = input("Enter the name of the virtual environment to remove: ").strip()
    remove_virtual_environment(env_name)
    
    # Ask if the user wants to remove Python
    remove_python_choice = input("Do you want to remove Python from the system? (yes/no): ").strip().lower()
    if remove_python_choice == "yes":
        python_path = input("Enter the path to the Python installation directory: ").strip()
        remove_python(python_path)
    else:
        print("Python installation has not been removed.")