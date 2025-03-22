"""
Constants used throughout the project.
"""

# Model settings
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
# BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
#FINETUNED_MODEL_PATH = "/home/belletti/ATE-metric-calling/finetuning/outputs/metric_evaluation_assistant/final_model"
# FINETUNED_MODEL_PATH = "/home/belletti/ATE-metric-calling/ATE/finetuning/outputs/metric_evaluation_assistant/checkpoint-603"
# FINETUNED_MODEL_PATH = "MatteBelle/llama-3.1-8b-instruct-finetunedevaluator"
FINETUNED_MODEL_PATH = "MatteBelle/llama-3.1-8b-instruct-finetunedevaluator"
TEST_DATASET_PATH = "ATE/test_compare/test_dataset.json"
TEMPERATURE = 0.6
MAX_TURNS = 3

# Output settings
DEFAULT_OUTPUT_DIR = "ATE/test_compare/outputs"

# System message defaults
DEFAULT_SYSTEM_MSG = "You are a bot that creates and responds to evaluation queries."