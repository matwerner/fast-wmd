#!/bin/bash
# Download and install all dependencies to the project
#
# Instructions:
# sudo chmod 700 install_dependencies.sh
# sh install_dependencies.sh

# Variables
PROJ_DIR=${pwd}
DEPENDENCIES_DIR="/usr/"
LOCAL_DEPENDENCIES_DIR="/usr/local/"

# OR-Tools - Combinatorial Optimization Library

if ! [ -f ${LOCAL_DEPENDENCIES_DIR}/lib/libortools.so ]
then
    OLD_ORTOOLS_FILE="or-tools_Ubuntu-16.04-64bit_v6.7.4973"
    NEW_ORTOOLS_FILE="or-tools"
    EXT=".tar.gz"

    sudo mkdir TEMP
    cd TEMP

    sudo apt-get install build-essential
    curl -LJO "https://github.com/google/or-tools/releases/download/v6.7.1/${OLD_ORTOOLS_FILE}${EXT}"

    tar -zxf ${OLD_ORTOOLS_FILE}${EXT}
    sudo mv -v ${OLD_ORTOOLS_FILE}/lib/libortools.so ${LOCAL_DEPENDENCIES_DIR}/lib/libortools.so
    sudo mv -v ${OLD_ORTOOLS_FILE}/include/* ${LOCAL_DEPENDENCIES_DIR}/include/

    cd ..
    sudo rm -r TEMP
    cd ${PROJ_DIR}
fi


# Eigen - Linear Algebra Library - Only headers project
# It will be installed under usr/include/eigen3

dpkg --list | grep libeigen3-dev > /dev/null
if ! [ $? -eq  0 ]
then
    sudo apt-get install libeigen3-dev
fi