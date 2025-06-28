import subprocess
import platform


def install_dependencies():
    # Upgrade pip
    subprocess.run(['pip', 'install', '--upgrade', 'pip'])

    # Install requirements from requirements.txt file
    try:
        with open('./../environment/requirements.txt', 'r') as f:
            requirements = [line.strip() for line in f
                            if line.strip() and not line.startswith('#')]

        if requirements:
            subprocess.run(['pip', 'install'] + requirements)
        else:
            print("No requirements found in requirements.txt")
    except FileNotFoundError:
        print("requirements.txt file not found")

    # Check if the OS is Linux
    if platform.system() == 'Linux':
        # Install required apt packages
        apt_packages = ['libgl1-mesa-glx', 'libglib2.0-0']
        subprocess.run(['sudo', 'apt-get', 'update'])
        subprocess.run(['sudo', 'apt-get', 'install', '-y'] + apt_packages)


if __name__ == "__main__":
    install_dependencies()