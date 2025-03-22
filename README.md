# Metric Evaluation Assistant

A framework for dynamic, structured text - evaluation via metrics and dataset creation.
A fine-tuned model for processing and responding to metric evaluation queries for natural language processing tasks.

## Project Overview
Automated Trial and Error (ATE)

This project develops a framework to build a specialized assistant that processes natural language queries about evaluation metrics like ROUGE, BLEU, BERTScore, etc., and formulates proper metric calls with appropriate parameters.

The system:
1. Asks the chosen model to generate a user query that exploits an extracted subgroup of metrics
2. Asks the model to answer the user query in a structured json format, given metric docs
3. Identifies which metrics are needed based on the query
4. Generates proper Action/Action Input calls with correct parameters
5. Returns formatted responses that can be used programmatically
6. Cycle this process to create a dataset
7. Provide modules for fine-tuning and inference

## Repository Structure

```
ATE-metric-calling/
├── ATE/
│   ├── tool_metadata/
│   │   ├── API_list.json        # List of supported metrics
│   │   └── API_descriptions.json # Detailed descriptions of each metric
│   ├── finetuning/              # Fine-tuning related files
│   │   ├── dataset.json         # Dataset for fine-tuning
│   │   └── run_finetuning.sh    # Script to run fine-tuning
│   ├── test_compare/            # Testing and comparison tools
│   │   ├── Dockerfile.server.comparison # Docker configuration for server comparison
│   │   ├── utility/             # Utility functions for testing
│   │   └── outputs/             # Storage for comparison results
│   ├── dataset_refinement/      # Dataset refinement tools
│   │   ├── Dockerfile.refine_dataset # Docker configuration for dataset refinement
│   │   └── full_refined_dataset.json # Refined evaluation dataset
│   ├── results/                 # Evaluation results storage
│   │   ├── intermediate_results/ # Temporary results during evaluation
│   │   └── final_results/       # Final aggregated results
│   ├── llm_server.py           # Server for LLM inference
│   ├── my_llm.py               # LLM interaction utilities
│   ├── main.py                 # Main ATE implementation
│   ├── main_answer_only.py     # Alternative implementation focusing on answers
│   └── Dockerfile.main         # Docker configuration for main application
├── docker-compose.yml          # Docker composition for standard setup
├── docker-compose.comparison.yml # Docker composition for comparison testing
├── docker-compose-answer-only.yml # Docker composition for answer-only mode
├── extract-dataset.py          # Script to extract training data
└── README.md                   # This file
```

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 2.2.0+
- Transformers 4.38.0+
- Hugging Face account (for model access)

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/mattebelle/ATE-metric-calling.git
   cd ATE-metric-calling
   ```

2. Install the required dependencies (it's preferred to use the Docker):
   ```
   pip install -r ATE/requirements.txt
   pip install -r ATE/finetuning/requirements.finetuning.txt
   ```

3. Ensure you have the necessary API keys:
   - For Hugging Face access to Meta-Llama models (go to the model page on HF, ask for access)

## Usage

### Running Evaluation

To test the fine-tuned model against the base model:

```bash
python ATE/test_compare/run_comparison.py --dataset TEST_DATASET_PATH
```

### Fine-tuning the Model

The model can be fine-tuned on additional data:
1) Write you HuggingFace token in the hf_token.txt
2) Change the file (ATE/finetuning/finetuner.py) to choose the model to finetune
3) Create a model card on you HuggingFace profile, use it (nickname/modelname) in ATE/finetuning/finetuner.py in the push function
4) Run the following command:
```bash
cd ATE/finetuning
./run_finetuning.sh
```

### Creating a New Dataset

To extract and format new training data:

```bash
python extract-dataset.py
```

## The Model

This project creates a framework with prompting and dynamic import of tool_metadata. The objective is to create a dataset for metric (tool) calling. 
---
The workflow of the dataset creation (docker-compose.yml) consists of:
0) For each metric in API_list.json:
1) Prompting a model to create user queries, given some metrics (extracted as subgroups of a main metric).
2) Asking the model to answer such query in a structured manner.
3) Parse metric name and parameters, returns eventual errors to the model and asks for correction.
4) Calls the evaluate function for the parsed metric(s) and parameters, returns eventual errors to the model and asks for correction.
5) If call is returned with no errors, then response is correctly structured (structure correcteness): ask the model if it thinks the query is solved (proxy of precise correcteness).
6) Loops for each of the short-term-memory slots, asking for new query-answers with In-Context-Learning (restarting from point 1 with a follow-up prompt).
7) Finished the STM slots, restarts from point 1 a new session with the same metric (for NUMSESSION).
8) Saves final dataset.
---
The workflow of finetuning:
0) Merge all datasets keeping only the desired data.
1) Finetunes the model, using Unsloth for better performance, pushing it on HuggingFace. Requires setting necessary HF token, Base model path (i.e., from HF).
---
The workflow of evaluation:
0) Extract couples query - answer from the dataset.
1) User sets the desired model and hf token.
2) Tests fine-tuned model and save results.
3) Tests base model and save results.
--- ---

The fine-tuned, specialized evaluation assistant:
- Understands metric documentation
- Formats parameters correctly
- Handles multiple metric calls in one query
- Provides consistently formatted outputs

## License

This project is modified from original work under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This project is built upon the STE (Simulated Trial and Error) framework, expanded with metric calls, multi-tool calls, fine-tuning, open-source model handling, a distinct Server - Main docker architecture.
- Original paper: "LLMs in the Imaginarium: Tool Learning through Simulated Trial and Error"
- Uses the Meta-Llama/Llama-3.1-8B-Instruct model architecture