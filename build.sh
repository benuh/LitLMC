#!/bin/bash

echo "Building LitLM - Literature Language Model"

# Create build directory if it doesn't exist
if [ ! -d "build" ]; then
    mkdir build
fi

# Navigate to build directory
cd build

# Run cmake to configure the project
echo "Configuring project with CMake..."
cmake ..

# Check if cmake was successful
if [ $? -ne 0 ]; then
    echo "CMake configuration failed!"
    exit 1
fi

# Build the project
echo "Building project..."
make

# Check if build was successful
if [ $? -eq 0 ]; then
    echo "Build successful!"
    echo "Run './LitLM' to start the program"
else
    echo "Build failed!"
    exit 1
fi