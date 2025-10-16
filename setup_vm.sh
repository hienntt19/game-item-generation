#!/bin/bash

# Exit on error
set -e

# Log file for debugging
LOG_FILE="server_setup.log"
echo "Starting server setup at $(date)" | tee -a $LOG_FILE

# Update and upgrade system
echo "Updating system packages..." | tee -a $LOG_FILE
apt update >> $LOG_FILE 2>&1
apt upgrade -y >> $LOG_FILE 2>&1

# Install required packages
echo "Installing build-essential, dkms, wget, libxm12, and libgl1-mesa-glx..." | tee -a $LOG_FILE
apt install -y build-essential dkms wget libxml2 libgl1-mesa-glx >> $LOG_FILE 2>&1

# Download cuda 11.8 installer
echo "Downloading CUDA 11.8 installer..." | tee -a $LOG_FILE
wget -q https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run -O cuda_11.8.0_520.61.05_linux.run >> $LOG_FILE 2>&1

# Make cuda installer executable
chmod +x cuda_11.8.0_520.61.05_linux.run

# Install cuda 11.8 (toolkit only, no driver or documentation)
echo "Installing CUDA 11.8 (toolkit only)..." | tee -a $LOG_FILE
./cuda_11.8.0_520.61.05_linux.run --silent --toolkit --no-man-page --no-drm >> $LOG_FILE 2>&1

# Verify cuda installation
echo "Checking cuda version..." | tee -a $LOG_FILE
if command -v nvcc > /dev/null 2>&1; then
    nvcc --version | tee -a $LOG_FILE
else
    echo "ERROR: CUDA installation failed, nvcc not found. " | tee -a $LOG_FILE
    exit 1
fi

# Download anaconda installer
echo "Downloading anaconda installer..." | tee -a $LOG_FILE
wget -q https://repo.anaconda.com/archive/Anaconda3-2023.03-Linux-x86_64.sh -O Anaconda3-2023.03-Linux-x86_64.sh >> $LOG_FILE 2>&1

# Install anaconda in batch mode
echo "Installing anaconda..." | tee -a $LOG_FILE
bash Anaconda3-2023.03-Linux-x86_64.sh -b -p $HOME/anaconda3 >> $LOG_FILE 2>&1

# Update path in ~/.bashrc
echo "Updating PATH in ~/.bashrc..." | tee -a $LOG_FILE
echo 'export PATH="$HOME/anaconda3/bin:$PATH"' >> ~/.bashrc

# Clean up downloaded files
# echo "Cleaning up installer files..." | tee -a $LOG_FILE
# rm cuda_11.8.0_520.61.05_linux.run Anaconda3-2023.03-Linux-x86_64.sh

echo "Server setup completed successfully at $(date)" | tee -a $LOG_FILE
echo "Please run 'source ~/.bashrc' or start a new terminal to apply environment changes." | tee -a $LOG_FILE