import importlib
import subprocess
import sys

# List of required packages
required_packages = ["numpy", "matplotlib", "cartopy"]

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install {package}. Please check your pip setup.")

def main():
    print("📦 Checking and installing required dependencies...\n")
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package} is already installed.")
        except ImportError:
            print(f"⬇️ {package} not found. Installing...")
            install_package(package)
    print("\n🎉 Dependency check complete!")

if __name__ == "__main__":
    main()
