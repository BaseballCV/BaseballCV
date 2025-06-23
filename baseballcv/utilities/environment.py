# baseballcv/utilities/environment.py
import subprocess
import sys
import importlib.util

def check_import(package_name: str) -> bool:
    """
    Checks if a package is installed without importing it.
    Returns True if installed, False otherwise.
    """
    try:
        if importlib.util.find_spec(package_name) is None:
            # Handle cases where install name differs from import name (e.g., scikit-learn vs sklearn)
            if '-' in package_name:
                package_name = package_name.replace('-', '_')
            if importlib.util.find_spec(package_name) is None:
                return False
        return True
    except ImportError:
        return False
    
def check_import(install_path: str, package_name: str) -> bool:
    """
    Checks if a package is installed and attempts to install it if not found.

    Args:
        install_path (str): The path to the package to check.
        package_name (str): The name of the package to check.

    Returns:
        bool: True if the package is installed, False otherwise.
    """
    try:
        __import__(package_name)
        return True
    except ImportError:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", install_path])
            return True
        except subprocess.CalledProcessError as e:
            print(f"Failed to install {package_name}: {str(e)}")
            raise
    