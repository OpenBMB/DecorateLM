import argparse
import random
import json

def random_sample_pair(ids, args):
    """
    Generates random pairs of integers for comparison purposes.
    
    Each integer ID will be paired with others to meet the minimum comparison count specified.
    Some IDs may have slightly more comparisons than others due to the sampling process.
    
    Parameters:
    - ids (set): A set of integer IDs to sample from.
    - args (Namespace): Parsed arguments from the command line containing settings for the sampling process.
    """
    # Set the random seed for reproducibility
    random.seed(args.seed)
    
    # Sort and store IDs for consistent access and tracking comparison counts
    list_ids = sorted(list(ids))
    output = []  # Store generated pairs
    count = [0] * (max(ids) + 1)  # Track comparison counts for each ID
    
    # Generate initial random pairs
    for _ in range(args.n_pairs):
        # Sample two unique IDs for pairing
        indices = random.sample(list_ids, 2)
        count[indices[0]] += 1
        count[indices[1]] += 1
        output.append(tuple(indices))

    added_cnt = 0

    # Ensure each ID meets the minimum comparison count
    for i in range(len(count)):
        if i not in ids:
            continue
        while count[i] < args.compare:
            # Sample a new ID for additional pairing with 'i'
            chosen = random.choice(list_ids)
            if chosen == i:
                continue  # Avoid self-pairing
            
            # Update comparison counts
            count[i] += 1
            count[chosen] += 1
            # Randomize the order in the tuple
            cur_tuple = (i, chosen) if random.random() < 0.5 else (chosen, i)
            output.append(tuple(cur_tuple))
            added_cnt += 1
    
    output_json = {
        "n": args.n,
        "n_pairs": len(output),
        "compare": args.compare,
        "pairs": output
    }

    print(output)
    
    # Write the generated pairs to the specified output file
    with open(args.output_path, 'w') as fout:
        fout.write(json.dumps(output_json, indent=4))
    
    # Display total number of pairs generated and output path
    print(f"Total number of pairs: {len(output)}")
    print(f"Total number of pairs added: {added_cnt}")
    print(f"Output pairs to {args.output_path}")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Generate random integer pairs for comparison sampling.")
    parser.add_argument("--n", type=int, required=True, help="Total number of unique integer IDs to sample from.")
    parser.add_argument("--n_pairs", type=int, required=True, help="Initial number of random pairs to generate.")
    parser.add_argument("--compare", type=int, default=5, help="Minimum required comparisons for each integer ID.")
    parser.add_argument("--output_path", type=str, help="Path to save the output json file with generated pairs.")
    parser.add_argument("--seed", type=int, default=2024, help="Seed for random number generator.")
    args = parser.parse_args()

    # Validate input arguments
    if 2 * args.n_pairs < args.n * args.compare:
        raise ValueError("Insufficient pairs for the minimum required comparisons per ID. Increase the 'pairs' count.")
    
    # Create set of IDs based on specified 'n' value
    ids = set(range(args.n))

    # Generate and save the random pairs
    print("Starting random pair generation...")
    random_sample_pair(ids, args=args)
    print("End of random pair generation.")
    print('*' * 50)
