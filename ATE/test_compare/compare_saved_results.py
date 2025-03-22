"""
Script for comparing two previously saved model evaluation results.
"""

import os
import sys
import fire
import json
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ""))

# Import project modules
from STE.test_compare.utility.metrics import calculate_comparison_metrics
from STE.test_compare.reporting.formatters import print_comparison_summary

def main(
    base_model_results_path: str,
    finetuned_model_results_path: str,
    output_dir: str = None
):
    """
    Compare two previously saved model evaluation results.
    
    Args:
        base_model_results_path: Path to the base model results JSON file
        finetuned_model_results_path: Path to the fine-tuned model results JSON file
        output_dir: Directory to save the comparison results (defaults to directory of base model file)
    """
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = os.path.dirname(base_model_results_path)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load evaluation results
    try:
        with open(base_model_results_path, 'r') as f:
            base_model_results = json.load(f)
        
        with open(finetuned_model_results_path, 'r') as f:
            finetuned_model_results = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: Could not find results file - {e}")
        return
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in results file - {e}")
        return
    
    print(f"\nComparing models:")
    print(f"Base model: {base_model_results['model']}")
    print(f"Fine-tuned model: {finetuned_model_results['model']}")
    
    # Calculate comparison metrics
    comparison_metrics = calculate_comparison_metrics(base_model_results, finetuned_model_results)
    
    # Prepare comparison results
    comparison_results = {
        "base_model": base_model_results,
        "finetuned_model": finetuned_model_results,
        "comparison": comparison_metrics,
        "timestamp": datetime.now().isoformat()
    }
    
    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    result_file = os.path.join(output_dir, f"comparison_results_{timestamp}.json")
    with open(result_file, 'w') as f:
        json.dump(comparison_results, f, indent=2)
    
    # Print summary
    print_comparison_summary(comparison_results, result_file)

if __name__ == "__main__":
    fire.Fire(main)