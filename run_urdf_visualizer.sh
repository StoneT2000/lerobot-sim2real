#!/bin/bash

# URDF Visualizer Runner
# Simple script to launch the URDF visualizer with dependency checks

echo "🤖 Starting URDF Visualizer..."

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed or not in PATH"
    exit 1
fi

# Check/install required packages
echo "📦 Checking dependencies..."

# Function to check if a Python package is installed
check_package() {
    python3 -c "import $1" 2>/dev/null
    return $?
}

# List of required packages
REQUIRED_PACKAGES=("gradio" "plotly" "numpy" "trimesh" "urchin")
MISSING_PACKAGES=()

for package in "${REQUIRED_PACKAGES[@]}"; do
    if ! check_package "$package"; then
        MISSING_PACKAGES+=("$package")
    fi
done

# Install missing packages
if [ ${#MISSING_PACKAGES[@]} -ne 0 ]; then
    echo "⬇️  Installing missing packages: ${MISSING_PACKAGES[*]}"
    uv add "${MISSING_PACKAGES[@]}"
    
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install required packages. Please install manually:"
        echo "uv add ${MISSING_PACKAGES[*]}"
        exit 1
    fi
fi

# Launch the visualizer
echo "🚀 Launching URDF Visualizer on http://localhost:1337"
echo "Press Ctrl+C to stop"

cd "$(dirname "$0")" || exit 1
python3 urdf_visualizer.py

echo "👋 URDF Visualizer stopped"
