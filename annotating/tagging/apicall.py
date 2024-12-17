      
import json
import threading
from openai import OpenAI
import os
import time, random

from concurrent.futures import ThreadPoolExecutor, as_completed

api_keys = [
    {'api_key' : 'xxxxx'}
]

invalid_keys = set()

def select_api_key():
    valid_indexes = [i for i in range(len(api_keys)) if i not in invalid_keys]
    if not valid_indexes:
        raise ValueError("All API keys are marked as invalid.")
    rand_index = random.choice(valid_indexes)
    return rand_index, api_keys[rand_index]


def api_call(prompt="Hello"):
    retry_limit = 3000
    done = False
    retry_count = 0

    while not done:
        try:
            key_index, api_key = select_api_key()
            print(f"using {key_index}.......")
            client = OpenAI(api_key=api_key['api_key'],base_url="https://yeysai.com/v1")
            model = "gpt-4-0125-preview"
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": "You are GPT-4, a large language model trained by OpenAI. Knowledge cutoff: 2021-09， Current date: 2024-04-04"}, {"role": "user", "content": prompt}],
                temperature=0.2
            )
            done = True
            reply = response.choices[0].message.content
            return reply
        except Exception as error:
            if "Please retry after" in str(error):
                timeout = int(str(error).split("Please retry after ")[1].split(" second")[0])
                print(f"Wait {timeout}s before OpenAI API retry ({error})")
                time.sleep(timeout)
            elif retry_count < retry_limit:
                print(f"OpenAI API retry for {retry_count} times ({error})")
                time.sleep(2)
                retry_count += 1
            else:
                print(f"OpenAI API failed for {retry_count} times ({error})")
                invalid_keys.add(key_index)
                print(f"API key {api_key['api_key']} marked as invalid.")
                return "<<FAILED>>"

            

    