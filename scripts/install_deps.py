import importlib
import subprocess
import sys

# import name -> pip name
required_packages = {
    "numpy": "numpy",
    "matplotlib": "matplotlib",
    "cartopy": "cartopy",
    "requests": "requests",
    "PyQt6": "PyQt6",
    "pyqtgraph": "pyqtgraph",
    "h5py": "h5py",
    "pyproj": "pyproj",
    "scipy": "scipy",
    "sgp4": "sgp4",
}


def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install {package}. Please check your pip setup.")


def main():
    print("📦 Checking and installing required dependencies...\n")
    for module, package in required_packages.items():
        try:
            importlib.import_module(module)
            print(f"✅ {package} is already installed.")
        except ImportError:
            print(f"⬇️ {package} not found. Installing...")
            install_package(package)
    print("\n🎉 Dependency check complete!")


if __name__ == "__main__":
    main()
