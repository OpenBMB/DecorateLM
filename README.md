# DecorateLM

This is the official repository for **[DecorateLM: Data Engineering through Corpus Rating, Tagging, and Editing with Language Models](https://arxiv.org/pdf/2410.05639)**, presented at the Main Conference at EMNLP 2024 (Miami). DecorateLM offers a systematic approach to improve pretraining corpora for Large Language Models (LLMs) through innovative data engineering techniques.

**DecorateLM** is an open-source toolkit designed to:

- **Rate**: Assess texts against predefined quality criteria, filtering lower-quality data.  
- **Tag**: Organize data with hierarchical labels for better categorization and structured training.  
- **Edit**: Refine and standardize text formats to ensure consistency and clarity.  

By optimizing the quality of pretraining data, DecorateLM enhances model robustness and adaptability across diverse tasks.

The repository provides open-source implementations for each annotation phase—rating, tagging, and editing. The DecorateLM model will soon be available on Hugging Face, empowering researchers and developers to seamlessly integrate high-quality data refinement into their LLM training pipelines.

## Annotation

### Rating

To run the complete scoring pipeline, execute the main Bash script:

``` bash
bash ./annotating/rating/run_scoring_pipeline.sh
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

### Tagging

During the tagging phase, begin by summarizing the data using the script:

```bash
bash ./annotating/tagging/scripts/run_summary.sh
```

This initial step helps facilitate subsequent annotation tasks.

Next, annotate the first level labels using the script:

```bash
bash ./annotating/tagging/scripts/run_tagging_first_level.sh
```

Finally, proceed to label the second and third-level tags with the script:

```bash
bash ./annotating/tagging/scripts/run_tagging_second_third_level.sh
```

#### Configuration Options

- `filepath`: Path to the input JSONL file prepared for tagging.
- `task`: Choose From `['summary' , 'tag_first_level', 'tag_second_third_level']`
- `totalinstance`: Total number of items to be tagged.
- `batchsize`: Number of items in each batch for annotation.
- `start`: Supports resuming tagging after interruption.

### Editing 

[Details coming soon.]

## Sampling

After obtaining the entire Decorated Corpus, data sampling can be performed based on predefined heuristic rules to obtain the filtered data used for training.

Specifically, start by generating a UUID for each data entry using the script located at `./sampling/generate_uuid.py`. 

Then, taking 'rating' as an example, use predefined heuristic rules to calculate the sampling probability for each data entry with the script `./sampling/rating/count_prob.py`. 

Next, perform sampling without replacement using the script `./sampling/tagging/sample_data.py`. 

Finally, complete the join operation using `./sampling/sample_by_uuid.py` to retrieve the complete data set based on the sampled UUIDs.

## Annotated Corpus

[Details coming soon]

## Decorated Corpus

[Details coming soon]

## DecorateLM model

[Details coming soon]

## Citation

If you find DecorateLM helpful in your research, please cite our paper:
``` bibtex
@inproceedings{zhao2024decoratelm,
  title={DecorateLM: Data Engineering through Corpus Rating, Tagging, and Editing with Language Models},
  author={Zhao, Ranchi and Thai, Zhen and Zhang, Yifan and Hu, Shengding and Zhou, Jie and Ba, Yunqi and Cai, Jie and Liu, Zhiyuan and Sun, Maosong},
  booktitle={Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing},
  pages={1401--1418},
  year={2024}
}
```