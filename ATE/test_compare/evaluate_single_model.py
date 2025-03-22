"""
Script for evaluating a single model and saving its results.
"""

import os
import sys
import fire
import json
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ""))

# Import project modules
from STE.test_compare.config.constants import (
    TEST_DATASET_PATH, 
    MAX_TURNS,
    DEFAULT_OUTPUT_DIR
)
from STE.test_compare.utility.dataset import load_dataset, load_api_metadata
from STE.test_compare.evaluation.model_runner import evaluate_test_set

def main(
    model_name: str,
    test_dataset_path: str = TEST_DATASET_PATH,
    max_turns: int = MAX_TURNS,
    output_dir: str = DEFAULT_OUTPUT_DIR
):
    """
    Evaluate a single model and save its results.
    
    Args:
        model_name: Name or path of the model to evaluate
        test_dataset_path: Path to the test dataset
        max_turns: Maximum number of turns for each query
        output_dir: Directory to save the results to
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load test dataset
    test_dataset = load_dataset(test_dataset_path)
    if not test_dataset:
        print("No data to evaluate. Exiting.")
        return
        
    # Load API metadata
    api_descriptions, api_list = load_api_metadata()
    
    # Get timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Evaluate model
    print(f"\nEvaluating model: {model_name}")
    model_results = evaluate_test_set(model_name, test_dataset, api_descriptions, api_list, max_turns)
    
    # Create a model identifier for the filename (strip invalid chars)
    model_id = model_name.replace('/', '_').replace('\\', '_')
    
    # Save results to file
    result_file = os.path.join(output_dir, f"evaluation_{model_id}_{timestamp}.json")
    with open(result_file, 'w') as f:
        json.dump(model_results, f, indent=2)
    
    print(f"\nModel evaluation completed.")
    print(f"Results saved to: {result_file}")
    
    # Print summary metrics
    metrics = model_results["metrics"]
    print("\n=== Model Evaluation Summary ===")
    print(f"Model: {model_name}")
    print(f"Successful queries: {metrics['successful_queries']}/{metrics['total_queries']} ({metrics['success_rate']:.2%})")
    print(f"Average turns for successful queries: {metrics['avg_turns_successful']:.2f}")

if __name__ == "__main__":
    fire.Fire(main)