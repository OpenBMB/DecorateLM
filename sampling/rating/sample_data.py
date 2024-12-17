import json
import numpy as np
import sys,os,glob
from tqdm import tqdm
import pandas as pd

def sample_with_weights_no_return(data, weights, n_samples):
    df = pd.DataFrame({
        'uuid': data,
        'weights': weights
    })
    df['weights'] = df['weights'].astype('float64')
    sampled_df = df.sample(n=n_samples, weights='weights', replace=False)
    return sampled_df['uuid'].tolist()

def get_ids(json_file):
    ids = []
    probabilities = []

    sample_probs = {
        "educational_value": 0.2, 
        "expertise": 0.2,
        "fact&trivia": 0.2,
        "obscurity": 0.05,
        "reasoning_level": 0.2, 
        "story-like": 0.05,
        "structural_format": 0.05,
        "subjectivity": 0.05
    }

    with open(json_file[0], 'r') as file:
        for line in tqdm(file):
            da = json.loads(line)
            ids.append(da['uuid'])

            total_weighted_probability = 0.0
            for key, sample_value in sample_probs.items():
                total_weighted_probability += da[key] * sample_value
            probabilities.append(total_weighted_probability)

    return ids, probabilities
                    
if __name__ == '__main__':
    file_name="5merge1"
    datasets = ["baike_chinese_new_all", "c4_dedup", "en_dolma_dedup", "pilve_v4_dedup", "zh_common_crawl"]
    base_path = '/mnt/data/user/tc_agi/zyf/DecorateLLM/rt/rating_probability_0521/'
    json_files = []
    for dataset in datasets:
        file_path = os.path.join(base_path, dataset)
        json_files.extend(glob.glob(os.path.join(file_path, '*.json')))

    ids, probabilities = get_ids(json_files)
    lens = {
            "baike_chinese_new_all": 6807862,
            "c4_dedup": 19085597,
            "en_dolma_dedup": 26195918,
            "pilve_v4_dedup": 10104909,
            "zh_common_crawl": 72974345
    }
    sample_size = sum(lens.values())
    sampled_ids = sample_with_weights_no_return(ids, probabilities, sample_size)

    save_dir = f"/data/checkpoints/zyf/DecorateLLM/rating_uuid"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, file_name)

    fo = open(output_path + "_uuids.jsonl", "w")
    for item in tqdm(sampled_ids):
        tmp_id = {"uuid": item}
        fo.write(json.dumps(tmp_id, ensure_ascii=False) + "\n")