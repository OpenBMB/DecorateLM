      

import json
from apicall import api_call
import os
from tqdm import tqdm
from datetime import datetime


def get_raw_data(file_path, total_instance, start=0):
    _, file_extension = os.path.splitext(file_path)
    raw_data = []

    if file_extension == '.jsonl':
        with open(file_path, "r") as f:
            for i, line in enumerate(tqdm(f, desc="Processing .jsonl file")):
                if len(raw_data) >= total_instance:
                    break
                d = json.loads(line.strip())
                if "instance_id" in d:
                    if d["instance_id"] < start:
                        continue
                else:
                    if i < start:
                        continue
                if 'original_content' in d and isinstance(d['original_content'], str):
                    d['original_content'] = json.loads(d['original_content'])
                    if 'clean_text' in d['original_content']:
                        d['original_content']['raw_content'] = d['original_content'].pop('clean_text')
                elif 'content' in d and isinstance(d['content'], str):
                    d['original_content'] = {}
                    d['original_content']['raw_content'] = d['content']
                    del d['content']
                elif 'summary' in d and isinstance(d['summary'], str):
                    d['original_content'] = {}
                    d['original_content']['raw_content'] = d['summary']
                raw_data.append(d)

    elif file_extension == '.md':
        with open(file_path, "r") as f:
            sample_content = []
            sample_count = -1  # Initialize to -1 because the counter updates at the start
            for line in tqdm(f, desc="Processing .md file"):
                if line.startswith('# --------------------------- sample'):
                    if sample_content:
                        sample_count += 1
                        if sample_count < start:
                            sample_content = []
                            continue
                        if len(raw_data) >= total_instance:
                            break
                        full_content = '\n'.join(sample_content).strip()
                        raw_data.append({'original_content': {"raw_content": full_content}})
                        sample_content = []
                        continue
                sample_content.append(line.strip())

            # For the last sample in the file
            if sample_content and len(raw_data) < total_instance and sample_count >= start - 1:
                full_content = '\n'.join(sample_content).strip()
                raw_data.append({'original_content': {"raw_content": full_content}})

    elif file_extension == '.txt':
        with open(file_path, "r") as f:
            for i, line in enumerate(tqdm(f, desc="Processing .txt file")):
                if i < start:
                    continue
                if len(raw_data) >= total_instance:
                    break
                d = json.loads(line.strip())
                d['original_content'] = json.loads(d['original_content'])
                raw_data.append(d)

    return raw_data

def randomsampler(raw_data, k):
    return random.sample(raw_data, k)

def random_sample_single(raw_data):
    import random
    indices = list(range(len(raw_data)))
    random.shuffle(indices)
    for index in indices:
        selected_item = raw_data[index]
        yield (index, selected_item)

def batch_random_sample_single(raw_data, batch_size=10):
    iterator = iter(random_sample_single(raw_data))
    total = len(raw_data)
    batch_count = (total + batch_size - 1) // batch_size

    for i in range(batch_count):
        batch = []
        for j in range(min(batch_size, total - i * batch_size)):
            batch.append(next(iterator))
        yield batch

def seqencesampler(raw_data, k):
    """
    Returns the first k elements from the raw_data.
    """
    return raw_data[:k]

def seqence_sample_single(raw_data):
    """
    Yields data instances one by one in order without shuffling.
    """
    for index, selected_item in enumerate(raw_data):
        yield (index, selected_item)

def batch_seqence_sample_single(raw_data, batch_size=10):
    """
    Yields batches of data instances in order without shuffling.
    """
    total = len(raw_data)
    batch_count = (total + batch_size - 1) // batch_size

    for i in range(batch_count):
        batch_start = i * batch_size
        batch_end = min(batch_start + batch_size, total)
        batch = [(index, raw_data[index]) for index in range(batch_start, batch_end)]
        yield batch

def remove_code_markers(s):
    start_marker = "```json"
    end_marker = "```"

    if s.startswith(start_marker) and s.endswith(end_marker):
        new_s = s[len(start_marker):].lstrip()
        new_s = new_s[:-len(end_marker)].rstrip()
        return new_s
    return s

prompts = {
"tag": 
"""
You are a sophisticated tagging system tasked with generating multi-level hierarchical tags to deepen an AI's comprehension and classification of text content.
Your objective is to scrutinize the provided text passage: [begin] {instance} [end] , and assign nested tags that effectively encapsulate its key themes across multiple layers.

Your tags must adhere to a three-level hierarchy:
- The first level must select from predefined categories: Natural Sciences, Humanities and Social Sciences, Industrial Manufacturing, Medical and Health, Agriculture and Forestry, Energy and Mining, Finance and Real Estate, Education, Transportation, Technology and Internet, Law, Military, Travel and Tourism, Entertainment, Arts and Culture, Emotional Psychology, Fashion and Beauty, Sports, Home and Lifestyle, Public Administration, and Social Events.
- The second level should narrow down within the first level's domain, capturing the text's principal theme or sector without delving into overly specific details.
- The third level introduces a general aspect or theme related to the second level, offering insight into broader topics or trends without getting lost in minute particulars.

Aim for a bird's-eye view in your tagging, ensuring that even the third-level tags remain relatively broad and thematic. 
The tags should akin to structures like 'Entertainment > Music > Drumming' or 'Natural Sciences > Meteorology > Cyclones'.
The tags at the second and the third level should not be unique to the analyzed text but should represent broad categories that can be applied to a wide range of texts. These two categories should be universal enough to facilitate the categorization of various texts under common themes and aspects.
Your tags should discerningly reflect the essence and primary focus of the text, avoiding overly niche or specific classifications to ensure relevance across a broader spectrum of texts within similar themes.
Your comprehensive output should mirror this structure: [{{"tag": "Parent Tag > Child Tag > Sub-child Tag", "explanation": "Provide a detailed explanation in English"}}].
""",
"summary":
"""
Your objective is to summarize the provided text: [begin] {instance} [end] , within 100 words，include the relevant information forthe use case in the summary as much as possible.
The summary will represent the input data for clustering in the nextstep.
Be concise and clear.
Do not add phrases like "This is the summary of" or "Summarized text:"...
Do not include any line breaks in the summary.
Provide your answer in English only.
Your comprehensive output should mirror this structure: {{"summary": ""}}.
""",
"tag_first_level":
"""
You are an advanced tagging system designed to identify the most pertinent theme within a given text passage: [begin] {instance} [end].
Your role is to analyze the text meticulously and choose the most fitting tag from the predefined list: Natural Sciences, Humanities and Social Sciences, Industrial Manufacturing, Medical and Health, Agriculture and Forestry, Energy and Mining, Finance and Real Estate, Education, Transportation, Technology and Internet, Law, Military, Travel and Tourism, Entertainment, Arts and Culture, Emotional Psychology, Fashion and Beauty, Sports, Home and Lifestyle, Public Administration, and Social Events.
Your task is to determine the single most relevant tag that encapsulates the primary theme of the text. Your selection should be substantiated with a detailed explanation, elucidating why this tag is the most accurate representation of the text's central subject matter.
Your output should follow this structure: {{"tag": "Selected Tag", "explanation": "Provide a detailed explanation in English  on why this is the most fitting tag."}}.
""",
"tag_second_third_level":
"""
You are an advanced tagging system designed to categorize a given text passage related to the first level tag "{first_level_tag}" into specific second and third-level tags within a predefined hierarchy.
Here is the tag hierarchy for the "{first_level_tag}" category in json format:
{tag_tree}
Here is the given text passage:
[begin] {instance} [end].
Your task is to analyze the text snippet above and assign the most fitting second-level and third-level tags, ensuring both tags align within the same hierarchical path.
The output should precisely reflect the main focus of the text, justifying why these tags are the most suitable choices.
Your output should follow this structure: {{"second_level_tag": "Selected Second Level Tag", "third_level_tag": "Selected Third Level Tag", "explanation": "Provide a detailed explanation in English on why these tags accurately represent the text's core content."}}.
"""
}

valid_parent_tags = [
    "Natural Sciences", "Humanities and Social Sciences", "Industrial Manufacturing",
    "Medical and Health", "Agriculture and Forestry", "Energy and Mining", 
    "Finance and Real Estate", "Education", "Transportation", "Technology and Internet",
    "Law", "Military", "Travel and Tourism", "Entertainment", "Arts and Culture",
    "Emotional Psychology", "Fashion and Beauty", "Sports", "Home and Lifestyle",
    "Public Administration", "Social Events"
]

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", type=str, default="data/deduped_redpajama_en_all_security_2024-1-8_new_0113.txt", help="Path to the source file")
    parser.add_argument("--task", type=str, default="tag", help="Task to perform")
    parser.add_argument("--totalinstance", type=int, default=10, help="Total number of instances in comparison or annotation")
    parser.add_argument("--batchsize", type=int, default=1, help="Batch size for parallel call chatgpt api")
    parser.add_argument("--start", type=int, default=0, help="Start id for data file")
    args = parser.parse_args()
    

    print("task: ",args.task)
    print("totalinstance: ",args.totalinstance)
    print("batchsize: ",args.batchsize)

    fraction_data = get_raw_data(args.filepath, args.totalinstance, args.start)
    print("fraction_data sample:\n",fraction_data[0])

    if args.batchsize == 1:
        single_instance_task = args.task
        prompt = prompts[single_instance_task]
        base_name = os.path.basename(args.filepath)
        file_name_without_extension = os.path.splitext(base_name)[0]
        today_date = datetime.now().strftime("%Y%m%d")
        output_file_name = f"tags_{file_name_without_extension}_{single_instance_task}.{args.totalinstance}.{today_date}.jsonl"
        output_path = os.path.join("/data/Decorate/data", output_file_name)
        
        for instanceid, instance in enumerate(fraction_data):
            task = prompt.format(instance=instance['original_content']['raw_content'][:3000])
            reply = api_call(task)
            try:
                print("*" * 16)
                print("reply: ", reply)
                reply_corrected = json.dumps(eval(reply))
                reply = json.loads(reply_corrected.replace('\n', ''))
                with open(output_path, "a") as f:
                    f.write(json.dumps({"instance_id": instanceid + args.start, "tags": reply}, ensure_ascii=False) + "\n")
            except:
                continue
                
    elif args.batchsize > 1:
        import concurrent.futures
        single_instance_task = args.task

        prompt = prompts[single_instance_task]
        sample_gen = iter(batch_seqence_sample_single(fraction_data, batch_size=args.batchsize))
        for batchid, batch in enumerate(sample_gen):
            print(f"Processing batch {batchid}...")
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.batchsize) as executor:
                if single_instance_task == "tag_second_third_level":
                    with open('/data/Decorate/data/tree/new_3_level_tag_tree.json', 'r', encoding='utf-8') as f:
                        tag_tree = json.load(f)
                    first_level_tags = []
                    valid_second_tags = []
                    valid_third_tags = []
                    for instanceids, instance in batch:
                        first_level_tags.append(instance["first_level_tag"])
                        valid_second_tags.append(list(tag_tree[instance["first_level_tag"].lower()].keys()))
                        valid_third_tag = []
                        for i in list(tag_tree[instance["first_level_tag"].lower()].values()):
                            valid_third_tag.extend(i.keys())
                        valid_third_tags.append(valid_third_tag)
                        # print(prompt.format(instance=instance['original_content']['raw_content'][:2500], first_level_tag=instance["first_level_tag"], tag_tree=tag_tree[instance["first_level_tag"].lower()]))
                        tasks = [prompt.format(instance=instance['original_content']['raw_content'][:2500], first_level_tag=instance["first_level_tag"], tag_tree=tag_tree[instance["first_level_tag"].lower()]) for instanceids, instance in batch]
                else:
                    tasks = [prompt.format(instance=instance['original_content']['raw_content'][:2500]) for instanceids, instance in batch]
                futures = [executor.submit(api_call, task) for task in tasks]

                results = [future.result() for future in futures]
                import json
                import re

                for i, reply in enumerate(results):
                    try:
                        print("*" * 16)
                        print("reply: ", reply)
                        reply = remove_code_markers(reply)
                        if "summary" in batch[i][1]:
                            summary = batch[i][1]["summary"]
                        if "first_level_explanation" in batch[i][1]:
                            first_level_explanation = batch[i][1]["first_level_explanation"]
                        if "first_level_tag" in batch[i][1]:
                            first_level_tag = batch[i][1]["first_level_tag"]
                        if "instance_id" in batch[i][1]:
                            instanceid = batch[i][1]["instance_id"]
                            _, instance = batch[i]
                        else:
                            instanceid, instance = batch[i]
                        base_name = os.path.basename(args.filepath)
                        file_name_without_extension = os.path.splitext(base_name)[0]
                        today_date = datetime.now().strftime("%Y%m%d")
                        today_date = "20240423"
                        output_file_name = f"tags_{file_name_without_extension}_{single_instance_task}.{args.totalinstance}.{today_date}.jsonl"
                        output_path = os.path.join("/data/Decorate/data", output_file_name)
                        reply_corrected = json.dumps(eval(reply))
                        reply = json.loads(reply_corrected.replace('\n', ''))
                        if single_instance_task=="tag":
                            is_valid_reply = all('tag' in tag and 'explanation' in tag for tag in reply)
                            if not is_valid_reply:
                                print("no tag/explanation")
                                continue
                            filtered_reply = [tag for tag in reply if len(tag['tag'].split('>')) == 3 and tag['tag'].split('>')[0].strip() in valid_parent_tags]
                            is_valid_reply = len(filtered_reply) > 0
                            if not is_valid_reply:
                                print("not 3 level tags or wrong first-level tag")
                                continue
                            with open(output_path, "a") as f:
                                f.write(json.dumps({"instance_id": instanceid + args.start, "tags": filtered_reply}, ensure_ascii=False) + "\n")
                        elif single_instance_task=="summary":
                            is_valid_reply = 'summary' in reply
                            if not is_valid_reply:
                                print("no summary")
                                continue
                            filtered_reply = reply
                            with open(output_path, "a") as f:
                                f.write(json.dumps({"instance_id": instanceid + args.start, "summary": filtered_reply['summary']}, ensure_ascii=False) + "\n")
                        elif single_instance_task=="tag_first_level":
                            is_valid_reply = 'tag' in reply and 'explanation' in reply
                            if not is_valid_reply:
                                print("no tag or explanation")
                                continue
                            is_valid_reply = reply["tag"] in valid_parent_tags
                            if not is_valid_reply:
                                print("wrong first-level tag")
                                continue
                            filtered_reply = reply
                            with open(output_path, "a") as f:
                                f.write(json.dumps({"instance_id": instanceid, "summary": summary, "first_level_tag": filtered_reply['tag'].lower(), "first_level_explanation": filtered_reply["explanation"]}, ensure_ascii=False) + "\n")
                        elif single_instance_task=="tag_second_third_level":
                            is_valid_reply = 'second_level_tag' in reply and 'third_level_tag' in reply and 'explanation' in reply
                            if not is_valid_reply:
                                print("no tag or explanation")
                                continue
                            is_valid_reply = reply["second_level_tag"] in valid_second_tags[i] and reply["third_level_tag"] in valid_third_tags[i] and reply["third_level_tag"] in tag_tree[first_level_tag.lower()][reply["second_level_tag"]].keys()
                            if not is_valid_reply:
                                print("wrong second and third level tag")
                                continue
                            filtered_reply = reply
                            with open(output_path, "a") as f:
                                f.write(json.dumps({"instance_id": instanceid, "summary": summary, "first_level_tag": first_level_tag.lower(), "second_level_tag": filtered_reply['second_level_tag'].lower(), "third_level_tag": filtered_reply['third_level_tag'].lower(), "first_explanation": first_level_explanation, "second_third_explanation": filtered_reply['explanation']}, ensure_ascii=False) + "\n")

                    except:
                        continue

    