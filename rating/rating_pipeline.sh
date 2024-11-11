#!/bin/bash

# Random Pair Sampler configuration
N=200                   # Number of items to sample
N_PAIRS=1000             # Number of pairs to sample
COMPARE=6               # Minimum number of comparisons for each item
RANDOM_SEED=20241109

# GPT Annotation configuration
KEYS="before"           # Key in JSON that contains the content, separated by commas if multiple
BATCH_SIZE=25           # Number of items in each batch for annotation
PROMPTS_PATH="/Users/edy/Desktop/DecorateLM/open_source/DecorateLM/rating/prompts"
TASK="educational_value" # Rating criteria to use

# Bradley-Terry Model Hyperparameters
LR=0.05                 # Learning rate for gradient descent
ITERATIONS=5000         # Number of iterations for model training
SCORE_TYPE="uniform"    # Scoring method type

# Define input, output, and intermediate file paths
INPUT_PATH="/Users/edy/Desktop/DecorateLM/open_source/test_data/test.jsonl"
PAIRS_IDX_PATH="test/random.json"
GPT_ANNOTATION_PATH=test/winners_${TASK}.jsonl
FINAL_OUTPUT_PATH=test/scores_${TASK}.jsonl

echo "Starting Random Pair Sampling..."
python rating/random_pairs_sampler.py \
    --n ${N} \
    --n_pairs ${N_PAIRS} \
    --compare ${COMPARE} \
    --output_path ${PAIRS_IDX_PATH} \
    --seed ${RANDOM_SEED}
echo "Random Pair Sampling completed. Output saved to ${PAIRS_IDX_PATH}"

echo "Starting GPT Annotation..."
python rating/rating_annotater.py \
    --input_file ${INPUT_PATH} \
    --keys ${KEYS} \
    --task ${TASK} \
    --pairs_path ${PAIRS_IDX_PATH} \
    --prompts_path ${PROMPTS_PATH} \
    --batch_size ${BATCH_SIZE} \
    --output_path ${GPT_ANNOTATION_PATH}
echo "GPT Annotation completed. Output saved to ${GPT_ANNOTATION_PATH}"

echo "Starting Bradley-Terry Scoring..."
python rating/score.py \
    --lr ${LR} \
    --iterations ${ITERATIONS} \
    --gpt_annotations ${GPT_ANNOTATION_PATH} \
    --score_type ${SCORE_TYPE} \
    --output_path ${FINAL_OUTPUT_PATH}
echo "Bradley-Terry Scoring completed. Final scores saved to ${FINAL_OUTPUT_PATH}"
