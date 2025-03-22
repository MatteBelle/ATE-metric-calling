import json
import os
import fire
from utils import parse_response, delete_intermediate_subfolder, sanitize_for_json
from my_llm import chat_my
from datetime import datetime
import evaluate

METRIC_CACHE = {}
TEMPERATURE = 0.6

def main(
    max_turn: int = 5,
    intermediate_dir_write: str = "STE/results/intermediate_results/",
    final_dir_write: str = "STE/results/final_results/",
    test_dataset_path: str = "STE/test_dataset.json",
    if_visualize: bool = True,
):
    data_dict = dict()
    hyperparameters = {
        "Temperature": TEMPERATURE,
        "max_turn": max_turn,
    }
    data_dict = {"Hyperparameters": hyperparameters}

    os.makedirs(final_dir_write, exist_ok=True)
    os.makedirs(intermediate_dir_write, exist_ok=True)
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    subfolder_path = os.path.join(intermediate_dir_write, f"run_{run_timestamp}")
    os.makedirs(subfolder_path, exist_ok=True)
    subfolder_info_path = os.path.join(intermediate_dir_write, "latest_run_subfolder.txt")
    with open(subfolder_info_path, "w") as f:
        f.write(subfolder_path)
        
    print(f"DEBUG: Intermediate results will be stored in: {subfolder_path}", flush=True)
    print("DEBUG: Entering main function with parameters:", flush=True)
    print(f"max_turn={max_turn}, final_dir_write={final_dir_write}, test_dataset_path={test_dataset_path}, if_visualize={if_visualize}", flush=True)

    # Load API descriptions and API list.
    with open("STE/tool_metadata/API_descriptions.json", "r", encoding='utf-8') as f:
        API_descriptions = json.load(f)
    with open("STE/tool_metadata/API_list.json", "r", encoding='utf-8') as f:
        API_list = json.load(f)
    data_dict["ALL_METRICS"] = API_list
    data_dict["ALL_METRICS_DESCRIPTIONS"] = API_descriptions

    # Load the online prompt template
    # with open("STE/prompts/prompt_online.txt", "r") as f:
    #     prompt_template = f.read().strip()
    
    # Load test dataset
    with open(test_dataset_path, "r", encoding='utf-8') as f:
        test_dataset = json.load(f)
    
    print(f"DEBUG: Loaded test dataset with {len(test_dataset)} entries", flush=True)
    
    results = []
    
    for entry_id, entry in enumerate(test_dataset):
        print(f"DEBUG: Processing dataset entry {entry_id}", flush=True)
        
        user_query = entry["query"]
        expected_answer = entry.get("answer", "")  # Get expected answer if available
        
        # Determine relevant metrics from the query
        # This is a simplification - in practice you might want to analyze the query
        # to determine which metrics are being requested
        relevant_metrics = []
        # for api in API_list:
        #     if api.lower() in user_query.lower():
        #         relevant_metrics.append(api)
        
        if not relevant_metrics:
            relevant_metrics = API_list  # Fallback to all metrics
        
        # # Format the prompt template with the user query
        # prompt = prompt_template.format(
        #     query=user_query,
        #     metric_name=", ".join(relevant_metrics)
        # )
        
        messages = [
            {"role": "system", "content": entry.get("system", "You are a bot that responds to evaluation queries.")}
        ]
        
        # print("DEBUG: Processing query using chat_my.", flush=True)
        # response, is_response_empty = chat_my(messages, prompt,
        #                    temp=TEMPERATURE, stop="Evaluation Result:", visualize=if_visualize, max_tokens=720)
        response, is_response_empty = chat_my(messages, user_query,
                           temp=TEMPERATURE, stop="Evaluation Result:", visualize=if_visualize, max_tokens=720)
        temp = response[-1]['content']
        
        #print(f"DEBUG USER QUERY {entry_id}: \n{prompt}\n\n\n\n", flush=True)
        print(f"DEBUG FIRST RESPONSE {entry_id}: \n{temp}", flush=True)
        print("DEBUG END: --------------------------------------------------------------------------------------------------------------\n\n\n\n\n\n\n\n", flush=True)
        
        entry_result = {
            "query": user_query,
            "expected_answer": expected_answer,
            "chains": []
        }
        
        parsed_response = parse_response(temp, relevant_metrics, API_descriptions)
        
        for n_turn in range(max_turn):
            if not parsed_response['parse_successful']:
                evaluation_result = parsed_response['parse_error_msg']
            else:
                if parsed_response.get('finish', False):
                    print(f"DEBUG: Final answer reached for entry {entry_id} at turn {n_turn}", flush=True)
                    parsed_response['evaluation_result'] = ""  # No evaluation needed for final answer
                    entry_result["chains"].append(parsed_response)
                    break
                else:
                    evaluation_result = ""
                    # Loop over each extracted action and call run_evaluation.
                    for act in parsed_response['actions']:
                        evaluation_result += run_evaluation(act['action'], act['action_input'], API_list, API_descriptions) + "\n"
            
            parsed_response['evaluation_result'] = evaluation_result
            entry_result["chains"].append(parsed_response)
            
            if parsed_response.get('finish', False):
                break
                
            messages, is_response_empty = chat_my(messages, 'Evaluation Result: ' + evaluation_result,
                               temp=TEMPERATURE, stop="Evaluation Result:", visualize=if_visualize, max_tokens=720)
            temp = messages[-1]['content']
            
            print(f"DEBUG QUERY {entry_id}, TURN {n_turn}, EVALUATION RESULT: \n{evaluation_result}\n\n\n\n", flush=True)
            print(f"DEBUG RESPONSE {entry_id}, TURN {n_turn}: \n{temp}", flush=True)
            print("DEBUG END: --------------------------------------------------------------------------------------------------------------\n\n\n\n\n\n\n\n", flush=True)
            
            parsed_response = parse_response(temp, relevant_metrics, API_descriptions)
        
        # If we didn't get a final answer or if chains is empty, add the last parsed response
        if len(entry_result["chains"]) == 0 or not entry_result["chains"][-1].get('finish', False):
            if 'parsed_response' in locals() and parsed_response:
                entry_result["chains"].append(parsed_response)
        
        # Determine if the query was solved successfully
        if len(entry_result["chains"]) > 0 and entry_result["chains"][-1].get('finish', False):
            entry_result["solved_at_turn"] = n_turn
        else:
            entry_result["solved_at_turn"] = -1  # Set to -1 if never solved
            
        results.append(entry_result)
        
        # Save intermediate results after each entry
        save_intermediate_results(entry_id, results[-1], subfolder_path)
    
    data_dict["results"] = results
    
    # Save final results
    final_data_path = os.path.join(final_dir_write, f"data_dict_{run_timestamp}.json")
    data_dict = sanitize_for_json(data_dict)
    with open(final_data_path, "w", encoding='utf-8') as f:
        json.dump(data_dict, f)
    
    delete_intermediate_subfolder(subfolder_path)
    print(f"DEBUG: Final data saved to {final_data_path}", flush=True)
    print("DEBUG: Finished main function.", flush=True)

def run_evaluation(metric_name, args, API_list, API_descriptions, truncate=False):
    """
    Execute an evaluation metric from Hugging Face evaluate.
    For the custom metric 'llm_judge', use the LLM to judge text quality.
    """
    global METRIC_CACHE
    if metric_name not in API_list:
        raise ValueError(f"Metric '{metric_name}' is not supported. Supported metrics are: {API_list}")
    
    # First, normalize the arguments.
    try:
        print(f"DEBUG: Normalizing evaluation arguments for metric '{metric_name}'", flush=True)
        print(f"DEBUG: Arguments before normalization: {args}", flush=True)
        # Note: We're still calling normalize_evaluation_args from metric_utils
        from metric_utils import normalize_evaluation_args
        normalized_args = normalize_evaluation_args(metric_name, args, API_descriptions)
        print("DEBUG: NORMALIZED ARGS = " + str(normalized_args), flush=True)
    except Exception as e:
        return f"Normalization error: {str(e)}"
    
    # If the metric is our custom llm_judge, handle it separately.
    if metric_name == "llm_judge":
        return run_llm_judge_evaluation(normalized_args, API_descriptions)
    
    # For all other metrics, use the standard evaluate.load mechanism.
    try:
        if metric_name not in METRIC_CACHE:
            METRIC_CACHE[metric_name] = evaluate.load(metric_name)
        metric = METRIC_CACHE[metric_name]
        
        # Special cases: adjust parameters if needed.
        if metric_name == "bertscore":
            print("DEBUG: Overriding model_type for bertscore to 'google/bert_uncased_L-2_H-128_A-2'", flush=True)
            normalized_args["model_type"] = "google/bert_uncased_L-2_H-128_A-2"
        if metric_name == "perplexity":
            print("DEBUG: Overriding model_id for perplexity to 'gpt-2'", flush=True)
            normalized_args["model_id"] = "gpt2"
        
        result = metric.compute(**normalized_args)
        result_str = json.dumps(result)
        if truncate:
            result_str = result_str[:truncate]
        return result_str
    except Exception as e:
        print("DEBUG ERROR: " + str(e), flush=True)
        return f"ERROR: The Action or Action Input is incorrect: {str(e)}. Fix it and provide new Action or Action input. When the Action or Action Input will be correct, immediately use \nThought: I now know the final answer\nFinal Answer:"

def run_llm_judge_evaluation(normalized_args, API_descriptions, temp=TEMPERATURE, max_tokens=720):
    """
    Custom handler for the 'llm_judge' metric. This function constructs a prompt
    to use the LLM as a judge for evaluating a candidate text based on multiple quality criteria.
    
    Expected keys in normalized_args for llm_judge:
      - candidate_texts (LIST of STRING): The text to evaluate.
      - quality_criteria (LIST of STRING): Aspects to consider (e.g., coherence, creativity).
      - scale_max (NUMBER): The top of the evaluation scale.
      - explanation_required (BOOLEAN, optional): Whether to output an explanation.
      - evaluation_type (STRING, optional): e.g., 'numeric' (default) or others.
      - prompt_template (STRING, optional): A custom prompt snippet to guide evaluation.
    
    Returns:
        JSON string containing the evaluation result (e.g., {"score": 8, "explanation": "..."})
    """
    # Ensure JSON-safe formatting for normalized_args
    json_args = json.dumps(normalized_args, ensure_ascii=False, indent=2)
    # Prepare a basic system message.
    messages = [{"role": "system", "content": "You are an LLM acting as an evaluation function with this documentation: " + json.dumps(API_descriptions["llm_judge"], ensure_ascii=False)}]
    
    prompt = (
        f"Your input parameters:\n```json\n{json_args}\n```\n\n"
        "You are an LLM Judge evaluating text quality.\n"
        "**Your ONLY task** is to output a JSON response formatted like this:\n\n"
        "```json\n"
        "{\n"
        '  "scores": {\n'
        '    "metric1": [score1, score2, ..., scoreN],\n'
        '    "metric...": [score1, score2, ..., scoreN],\n'
        '    "metricJ": [score1, score2, ..., scoreN]\n'
        "  },\n"
        '  "scale_max": <integer>, # scale_max\n'
        '  "explanation": "<text>"  # Only if explanation_required=true\n'
        "}\n"
        "```\n"
        "**STRICT INSTRUCTIONS:**\n"
        "- Respond ONLY with a JSON object.\n"
        "- Do NOT include any extra text before or after the JSON, except for exactly the text `'Evaluation Ends' right after the json.\n"
    )
    try:
        response, _ = chat_my(messages, prompt, temp=temp, stop="Evaluation Ends", visualize=False, max_tokens=max_tokens)
        response = response[-1]['content']
        return response
    except Exception as e:
        # If parsing fails, return an error message.
        return json.dumps({
            "error": f"Failed to process llm_judge response: {str(e)}",
            "raw_response": response if 'response' in locals() else ""
        })

def save_intermediate_results(entry_id, entry_result, subfolder_path):
    """
    Saves intermediate results in the dynamically created subfolder inside intermediate_dir_write.
    """
    try:
        # Generate a timestamp for each entry
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        intermediate_filename = os.path.join(subfolder_path, f"intermediate_entry_{entry_id}_{timestamp}.json")

        with open(intermediate_filename, "w", encoding="utf-8") as f:
            json.dump({
                "entry_id": entry_id,
                "entry_result": entry_result
            }, f, ensure_ascii=False, indent=2)

        print(f"DEBUG: Intermediate results saved to {intermediate_filename}", flush=True)
    except Exception as e:
        print(f"DEBUG: Error saving intermediate results: {e}", flush=True)

if __name__ == '__main__':
    fire.Fire(main)