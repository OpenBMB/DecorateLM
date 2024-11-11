# DecorateLM

**DecorateLM** is an open-source data engineering toolkit for enhancing the quality of pretraining corpora used in Large Language Models (LLMs). By addressing the need for structured, high-quality data, DecorateLM applies advanced data processing techniques to improve model performance on diverse text data.

## Overview

DecorateLM is designed to:

- **Rate**: Assess texts against predefined quality criteria, helping filter out lower-quality data.
- **Tag**: Organize data with hierarchical labels for better dataset categorization and structured training.
- **Edit**: Standardize and refine text format to ensure consistency and clarity.

By systematically rating, tagging, and editing the corpus, DecorateLM aims to maximize the utility of pretraining data, boosting the model’s robustness and adaptability across tasks.

The repository provides open-source code for each annotation phase—rating, tagging, and editing. The DecorateLM model will soon be available on Hugging Face, allowing researchers and developers to seamlessly integrate high-quality data refinement into their LLM training pipelines.

## Annotation

### Rating

To run the complete scoring pipeline, execute the main Bash script:

``` bash
bash run_scoring_pipeline.sh
```

Each step in the pipeline performs a specific function, as outlined below.

#### Scripts

- Random Pair Sampling (`random_pairs_sampler.py`): This script generates random pairs for comparison from the input data file.
- GPT Annotation (`rating_annotater.py`): This script uses GPT to perform pairwise comparison to each data, generating a winner for each data.
- Bradley-Terry Scoring (`score.py`): This script calculates the scores for each item based on the Bradley-Terry model.

#### Configuration Options

The run_scoring_pipeline.sh script contains the following configurable options:

- Input and Output Paths
  - `INPUT_PATH`: Path to the input JSONL file containing items for comparison.
  - `PAIRS_IDX_PATH`: Path to save the sampled pairs.
  - `GPT_ANNOTATION_PATH`: Path to save the GPT-generated annotations.
  - `FINAL_OUTPUT_PATH`: Path to save the final scores.

- Random Pair Sampler Parameters
  - `N`: Number of items to sample from the input data.
  - `N_PAIRS`: Number of pairs to generate for comparison.
  - `COMPARE`: Minimum number of comparisons per item.
  - `RANDOM_SEED`: Seed for reproducibility.

- GPT Annotation Parameters

  - `KEYS`: Key(s) in JSON for content to be annotated, separated by commas.

  - `BATCH_SIZE`: Number of pairs in each batch for annotation.

  - `PROMPTS_PATH`: Path to the directory containing annotation prompts.

  - `TASK`: Rating criterion used in annotation (e.g., educational_value, expertise, etc.).

- Bradley-Terry Model Parameters
  - `LR`: Learning rate for the scoring model.
  - `ITERATIONS`: Number of iterations for optimization.
  - `SCORE_TYPE`: Method for score calculation (bradley_terry or uniform).

#### Available Rating Criteria

The following rating criteria are supported:

- fact&trivia: Rates items based on factual accuracy and trivia quality.
- expertise: Rates items on the level of expertise required or demonstrated.
- educational_value: Rates items based on their educational worth.
- scarcity: Rates items based on rarity or scarcity of information.
- reasoning_level: Rates items based on the complexity of reasoning.
- structural_format: Rates items based on their structural clarity and format.
- story-like: Rates items based on their narrative or story-like quality.
- subjectivity: Rates items based on subjectivity or personal opinion.

To use a rating criterion, set the `TASK` variable in the Bash script to one of the criteria listed above.