#!/bin/bash

# Test script for flore_gpu.cu performance evaluation
# This script compiles and runs the CUDA implementation with various batch sizes
# to evaluate performance characteristics

echo "Starting performance evaluation of flore_gpu.cu"
echo "Timestamp: $(date)"
echo "=============================================="

# Define batch sizes to test
batch_sizes=(100000 200000 300000 400000 500000 600000 700000 800000 900000 1000000)

# Check if source file exists
if [ ! -f "flore_gpu.cu" ]; then
    echo "Error: flore_gpu.cu not found in current directory!"
    exit 1
fi

# Check if nvcc is available
if ! command -v nvcc &> /dev/null; then
    echo "Error: nvcc (NVIDIA CUDA Compiler) is not installed or not in PATH!"
    exit 1
fi

echo "Found CUDA compiler:"
nvcc --version | head -n 4
echo ""

# Create results directory if it doesn't exist
mkdir -p results

# Loop through different batch sizes
for batch_size in "${batch_sizes[@]}"; do
    echo "Testing with BATCH_SIZE=$batch_size"
    
    # Compile the CUDA program
    nvcc flore_gpu.cu -DBATCH_SIZE=$batch_size -std=c++11 -Wno-deprecated-gpu-targets -o flore_gpu_${batch_size}
    
    if [ $? -ne 0 ]; then
        echo "Compilation failed for BATCH_SIZE=$batch_size"
        continue
    fi
    
    # Run the compiled program and capture output
    echo "Running test..."
    output=$(./flore_gpu_${batch_size} 2>&1)
    exit_status=$?
    
    if [ $exit_status -eq 0 ]; then
        echo "Completed successfully"
        echo "$output"
        
        # Save results to a file
        echo "# Results for BATCH_SIZE=$batch_size" > "Results/result_${batch_size}.txt"
        echo "# Timestamp: $(date)" >> "Results/result_${batch_size}.txt"
        echo "$output" >> "Results/result_${batch_size}.txt"
    else
        echo "Execution failed for BATCH_SIZE=$batch_size"
        echo "Error output: $output"
    fi
    
    echo "----------------------------------------------"
    
    # Clean up executable
    rm -f "flore_gpu_${batch_size}"
done

echo "Performance evaluation completed!"
echo "Results saved to Results/ directory"
echo "Final timestamp: $(date)"