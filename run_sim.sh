#!/bin/bash
#SBATCH --job-name=Muscl
#SBATCH --output=logs/sycl_%j.out
#SBATCH --error=logs/sycl_%j.err
#SBATCH --partition=astro_gpu_interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00

cd "$SLURM_SUBMIT_DIR"        # directory where you ran sbatch

SRC_DIR="$SLURM_SUBMIT_DIR"
BUILD_DIR="$SRC_DIR/build"

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# 1) Configure only if not configured yet
if [ ! -f CMakeCache.txt ]; then
    echo "Configuring CMake..."
    cmake -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx "$SRC_DIR" || exit 1
else
    echo "CMake already configured – skipping configure step."
fi

# 2) Build (will be a no-op if everything is up to date)
echo "Building project..."
cmake --build . --config Release -j || exit 1

# 3) Run Python only if build succeeded
cd "$SRC_DIR"
export PYTHONPATH="$PWD/build:$PYTHONPATH"
python All-steps.py
