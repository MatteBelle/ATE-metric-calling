#!/bin/bash
set -e

# Default to the standard MODEL_SERVER_URL if not specified
#export MODEL_SERVER_URL="${MODEL_SERVER_URL:-http://0.0.0.0:8000/generate}"
export MODEL_SERVER_URL="${MODEL_SERVER_URL:-http://llm-server-comparison:8000/generate}"

if [ "$1" = "main" ]; then
    echo "Running ATE main script..."
    python3 -u /home/belletti/ATE-metric-calling/ATE/main.py "${@:2}"
elif [ "$1" = "main_answer" ]; then
    echo "Running ATE main script..."
    python3 -u /home/belletti/ATE-metric-calling/ATE/test_compare/main_answer_only.py "${@:2}"
elif [ "$1" = "compare" ]; then
    echo "Running model comparison..."
    python3 -u /home/belletti/ATE-metric-calling/ATE/test_compare/compare_models.py "${@:2}"
elif [ "$1" = "evaluate" ]; then
    echo "Running single model evaluation..."
    python3 -u /home/belletti/ATE-metric-calling/ATE/test_compare/evaluate_single_model.py "${@:2}"
elif [ "$1" = "compare_saved" ]; then
    echo "Running saved results comparison..."
    python3 -u /home/belletti/ATE-metric-calling/ATE/test_compare/compare_saved_results.py "${@:2}"
else
    echo "Unknown command: $1"
    echo "Available commands: main, compare, evaluate, compare_saved"
    exit 1
fi