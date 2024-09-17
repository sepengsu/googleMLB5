# Project

#To install the gemma-2-2b-it model please follow this instruction on linux virtual environment

Follow the steps below to set up a Linux virtual environment and install the gemma-2-2b-it model.

Prerequisites
Linux system (Ubuntu or similar distribution)
Python 3.6 or later installed
Git installed
Virtual environment module (venv) installed
Step-by-Step Installation
Step 1: Update and Upgrade System Packages
Before setting up the virtual environment, make sure your system packages are updated. Open your terminal and run the following commands:

###   sudo apt update
###   sudo apt upgrade

###   sudo apt install python3 python3-venv python3-pip
###   cd ~/workspace/personal_practice  # or any directory where you want to install
###   python3 -m venv gemma-venv  # Creates a virtual environment named "gemma-venv"
###   source gemma-venv/bin/activate
###   pip install -r requirements.txt
###   pip install torch transformers
###   git clone https://github.com/EHKY12/gemma-2-2b-it.git
###   cd gemma-2-2b-it
###   # Download model weights (if applicable)
###   wget http://model-download-link.com/gemma-2-2b-it-weights.bin

# Or install additional dependencies
###   pip install -r additional-requirements.txt



  
