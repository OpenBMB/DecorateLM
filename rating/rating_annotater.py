# Pairwise Comparison using LLM
# This script reads a JSONL file with data, randomly pairs instances for comparison, 
# and uses an API call to determine the winner between each pair based on specific tasks.

import argparse
import json
import os
import re
import concurrent.futures
from tqdm import tqdm
from apicall import call_api

# List of tasks for pairwise comparison
pairwise_instance_tasks = [
    "fact&trivia", "expertise", "educational_value", "scarcity",
    "reasoning_level", "structural_format", "story-like", "subjectivity"
]

def get_raw_data(args):
    """
    Load and preprocess raw data from a JSONL file.
    Each line in the input file should be a JSON object.
    
    Args:
        args: Command line arguments including the input file path and keys to extract.
    
    Returns:
        List of concatenated text content based on specified keys.
    """
    raw_data = []
    keys = args.keys.split(',')

    # Read and process each line in the input JSONL file
    with open(args.input_file, "r") as f:
        for line in tqdm(f, desc="Reading .jsonl file"):
            d = json.loads(line.strip())
            cur_content = ""
            # Concatenate values from specified keys, separated by newline
            for key in keys:
                if key in d:
                    if len(cur_content) > 0:
                        cur_content += '\n'
                    cur_content += d[key]
            if len(cur_content) == 0:
                print("Raw data content is empty")
            raw_data.append(cur_content)

    return raw_data

def get_prompts(args):
    """
    Load prompt templates for each task from the specified path.
    
    Args:
        args: Command line arguments including prompts path.
    
    Returns:
        Dictionary mapping task names to prompt templates.
    """
    ret = {}
    for task in pairwise_instance_tasks:
        cur_path = os.path.join(args.prompts_path, f"{task}.txt")
        if os.path.exists(cur_path):
            with open(cur_path, 'r', encoding='utf-8') as fin:
                ret[task] = fin.read()
    return ret

def batch_random_sample_pair(raw_data, max_id, batch_size=10):
    """
    Generates batches of random pairs from raw data for comparison.
    
    Args:
        raw_data: List of processed text data.
        max_id: The maximum ID of pairs that have been processed.
        batch_size: Number of pairs per batch.
    
    Yields:
        List of tuples for each batch, where each tuple includes:
        - Pair ID
        - IDs of the two instances in the pair
        - Text content of the two instances
    """
    with open(args.pairs_path, 'r') as fin:
        random_pairs_json = json.loads(fin.read())
    
    if random_pairs_json['n'] != len(raw_data):
        print("The number of pairs does not match the number of instances")
        return
    pairs = random_pairs_json['pairs']
    total_pairs = len(pairs)
    total_run = total_pairs // batch_size + (1 if total_pairs % batch_size != 0 else 0)
    
    for i in range(total_run):
        batch = []
        for j in range(batch_size):
            cur_id = i * batch_size + j
            if cur_id <= max_id or cur_id >= total_pairs:
                continue
            id1, id2 = map(int, pairs[cur_id])
            batch.append((cur_id, (id1, id2), (raw_data[id1], raw_data[id2])))
        yield batch

if __name__ == "__main__":
    # Argument parser for command line inputs
    parser = argparse.ArgumentParser(description="Run pairwise comparison using LLM")
    parser.add_argument("--input_file", type=str, help="Path to the source jsonl file")
    parser.add_argument("--keys", type=str, help="Keys to extract content")
    parser.add_argument("--task", type=str, default="fact&trivia", help="Task to perform")
    parser.add_argument("--pairs_path", type=str, help="Path to the file containing the random pairs")
    parser.add_argument("--prompts_path", type=str, default="./rating/prompts", help="Path to the prompts")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for parallel call API")
    parser.add_argument("--output_path", type=str, default="", help="Path to save the output")
    args = parser.parse_args()

    print("Start GPT Annotation...")

    data = get_raw_data(args)
    prompts = get_prompts(args)
    pairwise_instance_task = args.task
    output_path = args.output_path

    # Resume from the last annotated pair, if any
    current_max_id = -1
    if os.path.exists(output_path):
        with open(output_path, "r") as fin:
            done = [json.loads(line) for line in fin if line.strip()]
            current_max_id = max(entry['pair_id'] for entry in done)

    # Process data in batches
    sample_gen = iter(batch_random_sample_pair(data, max_id=current_max_id, batch_size=args.batch_size))
    for batchid, batch in enumerate(sample_gen):
        print(f"Processing batch {batchid}...")
        
        # Parallel API calls to get comparison results
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.batch_size) as executor:
            tasks = [
                prompts[pairwise_instance_task].format(text_1=instance1, text_2=instance2)
                for _, instanceids, (instance1, instance2) in batch
            ]
            futures = [executor.submit(call_api, task) for task in tasks]

            # Process results and write to output
            results = [future.result() for future in futures]
            for i, reply in enumerate(results):
                cur_id, instanceids, _ = batch[i]
                instance1id, instance2id = instanceids

                # Parse the response to determine the winning choice and reasoning
                win_pattern = r'Choice: ([12])'
                why_pattern = r'Why: (.*)'

                reply = reply.strip()
                matches = re.findall(win_pattern, reply)

                if len(matches) == 1:  # Valid response
                    cur_why = re.search(why_pattern, reply).group(1) if re.search(why_pattern, reply) else ""
                    with open(output_path, "a") as f:
                        f.write(json.dumps({
                            "pair_id": cur_id,
                            "pair_ids": [instance1id, instance2id],
                            "win": matches[0],
                            "why": cur_why
                        }) + "\n")
                else:
                    print(reply)
                    print("No matches found. Skipping write.")
    print("End GPT Annotation.")
    print('*' * 50)