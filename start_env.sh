#!/bin/bash

# Script: start_env.sh
# Author: bmalbusca
# Year: 2023
# Description: Set up and activate a Python virtual environment.

# Default values
CMD=$1
DIR=".msenv"

# Check if a command-line argument is provided to set the environment name
if [[ -n "$CMD" ]]; then
    DIR=$CMD
    echo $DIR
fi

# Set the path to the virtual environment
PWD=$(pwd)/$DIR
echo "Activating python environment."

# Check if the environment directory already exists
if [ -d "$DIR" ]; then
    echo "The $DIR exists."
else
    # If not, create the environment directory and set up a virtual environment
    echo "Creating $DIR environment directory."
    python3 -m venv $DIR # Virtual Environments on Python 3.5+
fi

# Activate the virtual environment
source $DIR/bin/activate

# Check if the virtual environment activation was successful
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "Not able to start the python environment."
else 
    echo "Activated."
fi
