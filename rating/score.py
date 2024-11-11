# Bradley-Terry Model for Pairwise Comparison Scoring
# This script calculates scores based on pairwise comparisons using the Bradley-Terry model.

import numpy as np
import json
import argparse
from tqdm import tqdm
import pandas as pd

def log_likelihood(params, matches):
    """
    Calculate the negative log-likelihood for the Bradley-Terry model.
    
    Args:
        params (np.array): Array of parameters (scores) for each item.
        matches (list of tuples): Each tuple contains two indices representing a match (winner, loser).
        
    Returns:
        float: Negative log-likelihood value.
    """
    matches = np.array(matches)
    probs = 1 / (1 + np.exp(params[matches[:, 1]] - params[matches[:, 0]]))
    ll = 0
    for w, l in matches:
        ll += np.log(probs[w])
    return -ll

def compute_gradient(params, matches):
    """
    Compute the gradient of the log-likelihood function.
    
    Args:
        params (np.array): Array of parameters (scores) for each item.
        matches (list of tuples): Each tuple contains two indices representing a match (winner, loser).
        
    Returns:
        np.array: Gradient for each parameter.
    """
    gradient = np.zeros_like(params)
    for winner, loser in matches:
        winner_idx, loser_idx = winner, loser
        p_win = 1 / (1 + np.exp(params[loser_idx] - params[winner_idx]))
        gradient[winner_idx] += (1 - p_win)
        gradient[loser_idx] -= (1 - p_win)
    return -gradient

def score_calculation(args, pairwise_scores):
    """
    Main function to calculate and output scores using the Bradley-Terry model.
    
    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        pairwise_scores (list): List of pairwise comparisons.
    """
    # Convert pairwise scores to a DataFrame
    df = pd.DataFrame(pairwise_scores)

    # Identify winners and losers based on the 'win' column
    df['winner'] = df.apply(lambda x: x.pair_ids[0] if x.win == '1' else x.pair_ids[1], axis=1)
    df['loser'] = df.apply(lambda x: x.pair_ids[1] if x.win == '1' else x.pair_ids[0], axis=1)

    # Map items to indices
    items = list(set(df['winner']).union(set(df['loser'])))
    item_to_index = {item: index for index, item in enumerate(items)}
    index_to_item = {index: item for item, index in item_to_index.items()}
    df['winner_idx'] = df['winner'].map(item_to_index)
    df['loser_idx'] = df['loser'].map(item_to_index)
    matches = list(zip(df['winner_idx'], df['loser_idx']))

    # Initialize model parameters and settings
    total_item = len(items)
    params = np.zeros(total_item)  # Initial scores for each item
    learning_rate = args.lr
    num_iterations = args.iterations

    # Optimize parameters using gradient descent
    for iteration in tqdm(range(num_iterations)):
        gradient = compute_gradient(params, matches)
        params -= learning_rate * gradient
        if iteration % 1000 == 0:
            loss = log_likelihood(params, matches)
            print(f"Iteration {iteration}, Loss: {loss}")

    # Sort results by scores in descending order
    argsort = np.argsort(-params)

    # Generate scores based on the specified scoring method
    if args.score_type == "bradley_terry":
        sorted_score = -np.sort(-params)
    elif args.score_type == "uniform":
        sorted_score = np.linspace(100 - 1, 1, total_item)
    else:
        raise ValueError("Invalid score type")

    # Prepare results for output
    sorted_tuples = sorted(list(zip(argsort, sorted_score)), key=lambda x: x[0])

    # Write results to the specified output file
    output_file_path = args.output_path
    with open(output_file_path, "w") as f:
        for item in sorted_tuples:
            output_tuple = (index_to_item[item[0]], item[1])
            output_json = {
                "id": output_tuple[0],
                "score": output_tuple[1]
            }
            f.write(json.dumps(output_json) + "\n")

if __name__ == "__main__":
    # Argument parsing for configurable settings
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate for gradient descent")
    parser.add_argument("--iterations", type=int, default=200000, help="Number of iterations for optimization")
    parser.add_argument("--score_type", type=str, default="bradley_terry", help="Type of scoring method")
    parser.add_argument("--gpt_annotations", type=str, required=True, help="Path to input file with pairwise comparisons")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save output scores")
    args = parser.parse_args()

    print("Starting Score Calculation...")

    # Load pairwise comparison data
    with open(args.gpt_annotations, "r") as f:
        pairwise_scores = [json.loads(line) for line in f]

    # Run the score calculation
    score_calculation(args=args, pairwise_scores=pairwise_scores)

    print("Score Calculation Finished!")
