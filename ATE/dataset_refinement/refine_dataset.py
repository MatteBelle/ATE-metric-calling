import json
import requests
import re
import os
from tqdm import tqdm
import time

# Path to your dataset
INPUT_FILE = "/home/belletti/STE-model-calling/STE/full_dataset.json"
OUTPUT_FILE = "/home/belletti/STE-model-calling/STE/full_corrected_dataset.json"
# Checkpoint file pattern
CHECKPOINT_FILE_PATTERN = "/home/belletti/STE-model-calling/STE/checkpoint_dataset_{}.json"
# Checkpoint frequency (save every N entries)
CHECKPOINT_FREQUENCY = 500

SERVER_URL = os.environ.get("MODEL_SERVER_URL", "http://localhost:8000")

# Function to extract metrics listed in the "Reminder" section
def extract_allowed_metrics(query):
    reminder_section = re.search(r"1\) The only values that should follow \"Action:\" are: (.+?)\n", query)
    if reminder_section:
        metrics_text = reminder_section.group(1)
        # Handle cases where metrics are listed with commas or other separators
        metrics = re.findall(r'[\w_]+', metrics_text)
        return metrics
    return []

# Function to extract the corrected answer from the LLM response
def extract_answer(response_text):
    # Look for regular pattern first
    pattern = r"Action: [\w_]+"
    action_matches = re.findall(pattern, response_text)
    
    if action_matches:
        start_idx = response_text.index(action_matches[0])
        content = response_text[start_idx:]
        
        # Find all complete action blocks
        all_blocks = []
        current_block = []
        lines = content.split('\n')
        
        in_action_block = False
        for line in lines:
            if re.match(r"Action: [\w_]+", line.strip()):
                if in_action_block and current_block:
                    all_blocks.append('\n'.join(current_block))
                    current_block = []
                in_action_block = True
                current_block.append(line.strip())
            elif in_action_block:
                current_block.append(line.strip())
                # Check if this completes a JSON object
                if line.strip() == '}':
                    has_action_input = any("Action Input:" in l for l in current_block)
                    if has_action_input:
                        all_blocks.append('\n'.join(current_block))
                        current_block = []
                        in_action_block = False
        
        # Add the last block if there's any
        if in_action_block and current_block:
            all_blocks.append('\n'.join(current_block))
            
        if all_blocks:
            return '\n\n'.join(all_blocks)
    
    return None

# Function to extract the corrected answer from the LLM response
def check_answer(system_prompt, query, current_answer):
    # Extract allowed metrics to include in the prompt
    allowed_metrics = extract_allowed_metrics(query)
    metrics_str = ", ".join(allowed_metrics) if allowed_metrics else "the metrics listed in the query"
    
    # Use a unique marker for easy parsing
    RESPONSE_MARKER = "###RESPONSE_BEGINS###"
    
    prompt = f"""
I need you to verify and correct an answer to an evaluation query.

The query specifies these available metrics: {metrics_str}

Here is the complete query with all metric documentation:
{query}

Current answer:
{current_answer}

Check if the current answer correctly addresses all the required metrics and parameters mentioned in the query. 
If the answer is correct, provide exactly the same answer.
If the answer is incorrect or incomplete, provide a fully corrected answer.

Provide ONLY the corrected answer without any explanations or additional text.
Start your response with this marker: {RESPONSE_MARKER}
"""

    # Prepare the request
    request_data = {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 2048,
        "temperature": 0.1
    }
    
    try:
        # Send the request to the server
        response = requests.post(f"{SERVER_URL}/generate", json=request_data)
        response.raise_for_status()
        full_response = response.json()["response"]
        
        # Extract only what comes after the marker
        if RESPONSE_MARKER in full_response:
            return full_response.split(RESPONSE_MARKER, 1)[1].strip()
        return full_response
        
    except Exception as e:
        print(f"Error sending request: {e}")
        return None

# Function to save checkpoint
def save_checkpoint(data, checkpoint_number):
    checkpoint_file = CHECKPOINT_FILE_PATTERN.format(checkpoint_number)
    with open(checkpoint_file, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\nCheckpoint saved: {checkpoint_file}")

# Main processing function
def process_dataset():
    # Check if there's a checkpoint to resume from
    checkpoint_files = sorted([f for f in os.listdir(os.path.dirname(CHECKPOINT_FILE_PATTERN.format(0))) 
                              if f.startswith('checkpoint_dataset_') and f.endswith('.json')])
    
    data = None
    start_index = 0
    
    if checkpoint_files:
        latest_checkpoint = os.path.join(os.path.dirname(CHECKPOINT_FILE_PATTERN.format(0)), checkpoint_files[-1])
        print(f"Found checkpoint: {latest_checkpoint}")
        try:
            with open(latest_checkpoint, 'r') as f:
                data = json.load(f)
            # Find the last processed index
            for i, entry in enumerate(data):
                if "corrected_answer" in entry:
                    start_index = i + 1
            print(f"Resuming from entry {start_index}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            data = None
    
    # If no valid checkpoint, load the original dataset
    if data is None:
        try:
            with open(INPUT_FILE, 'r') as f:
                data = json.load(f)
            print(f"Loaded original dataset with {len(data)} entries")
        except Exception as e:
            print(f"Error loading input file: {e}")
            return
    
    # Process each entry
    for i in tqdm(range(start_index, len(data)), initial=start_index, total=len(data), desc="Processing entries"):
        entry = data[i]
        if "query" in entry and "answer" in entry:
            system_prompt = entry.get("system", "You are a bot that creates and responds to evaluation queries.")
            
            # Check and correct the answer
            corrected_response = check_answer(system_prompt, entry["query"], entry["answer"])
            
            if corrected_response:
                # Extract the formatted answer from the response
                formatted_answer = extract_answer(corrected_response)
                if formatted_answer:
                    entry["corrected_answer"] = formatted_answer
                    print(f"Entry {i} processed.")
                    print(f"Original answer: {entry['answer'][:100]}...")
                    print(f"Corrected answer: {formatted_answer[:300]}...")
                    print("-" * 50)
                else:
                    print(f"Failed to extract answer from response for entry {i}")
                    print(f"Raw response: {corrected_response[:200]}...")
                    print("-" * 50)
            else:
                print(f"Failed to get a response for entry {i}")
            
            # Save checkpoint every CHECKPOINT_FREQUENCY entries
            if (i + 1) % CHECKPOINT_FREQUENCY == 0:
                checkpoint_number = (i + 1) // CHECKPOINT_FREQUENCY
                save_checkpoint(data, checkpoint_number)
            
            # Add a small delay to avoid overloading the server
            time.sleep(1)
    
    # Save the final dataset
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Processing complete. Updated dataset saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    process_dataset()