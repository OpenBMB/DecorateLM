import json
import numpy as np
import sys,os,glob
from tqdm import tqdm
import pandas as pd

def sample_with_weights_no_return(data, weights, n_samples):
    # weights = np.array(weights, dtype=np.float64)
    df = pd.DataFrame({
        'uuid': data,
        'weights': weights
    })
    df['weights'] = df['weights'].astype('float64')
    sampled_df = df.sample(n=n_samples, weights='weights', replace=False)
    return sampled_df['uuid'].tolist()
    
def adjust_probability(first_tag, second_tag, third_tag, probability):
    if first_tag in tag_list:
        return probability * 2
    elif second_tag in tag_list:
        return probability * 3
    elif third_tag in tag_list:
        return probability * 4
    return probability
    
def get_ids(json_file):
    ids = []
    probabilities = []
    with open(json_file[0], 'r') as file:
        for line in tqdm(file):
            da = json.loads(line)
            ids.append(da['uuid'])
            adjusted_prob = adjust_probability(data['first_level_tag'], data['second_level_tag'], data['third_level_tag'], data['probability'])
            # 添加 id 和调整后的 probability
            probabilities.append(adjusted_prob)
            probabilities.append(da['probability']) 
    return ids, probabilities
                    
if __name__ == '__main__':
    # 文件路径
    file_name = sys.argv[1]
    file_path = '/mnt/data/user/tc_agi/zyf/DecorateLLM/rt/tagging_probability_new/' + file_name
    json_file = glob.glob(os.path.join(file_path, '*.json'))
    # 读取文件
    #data = []
    #for json_file in json_files:
    #    with open(json_file, 'r') as file:
    #        lines = file.readlines()
    #    for line in lines:
    #        da = json.loads(line)
    #        data.append(da)
    # 解析每行数据
    #data = [json.loads(line) for line in tqdm(lines)]
    #ids = [item['uuid'] for item in data]
    #probabilities = [item['probability'] for item in data]
    ids, probabilities = get_ids(json_file)
    lens = {
            "baike_chinese_new_all": 8850221,
            "c4_dedup": 24811277,
            "en_dolma_dedup": 34054694,
            "pilve_v4_dedup": 13136382,
            "zh_common_crawl": 94866649
    }
    sample_size = lens[file_name]
    sampled_ids = sample_with_weights_no_return(ids, probabilities, sample_size)
    # print(sampled_ids)
    save_dir = f"/data/checkpoints/zyf/DecorateLLM/tagging_uuid"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, file_name)
    fo = open(output_path + "_uuids.jsonl", "w")
    # fo = open(file_path.split("/")[-2] + "_output.jsonl", "w") 
    for item in tqdm(sampled_ids):
        tmp_id = {"uuid": item}
        fo.write(json.dumps(tmp_id, ensure_ascii=False) + "\n")