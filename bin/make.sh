#!/usr/bin/bash

# Update package list and install necessary dependencies
apt-get update && apt-get install -y build-essential libpq-dev libssl-dev openssl libffi-dev sqlite3 libsqlite3-dev libbz2-dev zlib1g-dev libxerces-c-dev libfox-1.6-dev libgdal-dev libproj-dev libgl2ps-dev git g++ cmake wget

# Download and install Python 3.10.12
PYTHON_VERSION=3.10.12
wget https://www.python.org/ftp/python/$PYTHON_VERSION/Python-$PYTHON_VERSION.tar.xz
tar xvf Python-$PYTHON_VERSION.tar.xz
cd Python-$PYTHON_VERSION
./configure
make
make altinstall
cd ..
rm -rv Python-$PYTHON_VERSION.tar.xz Python-$PYTHON_VERSION

# Create a virtual environment with Python 3.10.12
mkdir venv
python3.10 -m venv venv/

# Activate the virtual environment
source venv/bin/activate

# Set temporary directory and install Python packages from requirements.txt
export TMPDIR='/var/tmp'
pip install --no-cache-dir -r requirements.txt

# Deactivate the virtual environment
deactivate

# Clone the SUMO repository and remove .git files
cd venv/
git clone --recursive https://github.com/eclipse/sumo
rm -rv $(find sumo/ -iname "*.git*")

# Build SUMO
mkdir sumo/build/cmake-build
cd sumo/build/cmake-build
cmake ../..
make -j$(nproc)

# Exit the script
exit
