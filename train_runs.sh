#!/bin/bash

# Simple training script runner
echo "Starting training experiments..."
echo "================================"

# Create logs directory
mkdir -p logs

# List of experiments
experiments=(
    "python rainbow_dqn.py --mode train --timesteps 1000000 --grid_size 20 20 --seed 42 --dirt_num 0 --eval_episodes 5"
    "python rainbow_dqn.py --mode train --timesteps 1000000 --grid_size 20 20 --seed 42 --dirt_num 0 --eval_episodes 5 --wall_mode random"
    "python rainbow_dqn.py --mode train --timesteps 1000000 --grid_size 20 20 --seed 42 --dirt_num 5 --eval_episodes 5 --wall_mode random"
    "python rainbow_dqn.py --mode train --timesteps 1000000 --grid_size 40 30 --seed 42 --dirt_num 0 --eval_episodes 5 --wall_mode hardcoded"
)

total=${#experiments[@]}
success=0
failed=0

# Run each experiment
for i in "${!experiments[@]}"; do
    exp_num=$((i + 1))
    echo ""
    echo "[$exp_num/$total] Starting experiment $exp_num..."
    echo "Command: ${experiments[$i]}"
    
    start_time=$(date +%s)
    
    # Run experiment and capture output
    if ${experiments[$i]} 2>&1 | tee "logs/experiment_${exp_num}.log"; then
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        echo "✓ Experiment $exp_num completed in ${duration}s"
        success=$((success + 1))
    else
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        echo "✗ Experiment $exp_num failed after ${duration}s"
        failed=$((failed + 1))
        echo "  Check logs/experiment_${exp_num}.log for details"
    fi
done

# Summary
echo ""
echo "Summary:"
echo "=========="
echo "Total: $total"
echo "Successful: $success"
echo "Failed: $failed"

if [ $failed -eq 0 ]; then
    echo "All experiments completed successfully!"
else
    echo "Some experiments failed - check log files"
fi
