import json
import os
import random
import re

import pandas as pd
from PIL import Image
from datasets import Dataset, Features, Value, Image
from tqdm import tqdm

random.seed(42)


def format_math_explanation(text: str):
    answer_match = re.search(r'Answer:\s*([A-Z])', text)
    if answer_match:
        answer = answer_match.group(1)
    else:
        raise ValueError("No answer found in the text")

    text = re.sub(r'Answer:\s*[A-Z]', '', text).strip()
    formatted_text = f"<think>{text}</think> <answer>{answer}</answer>"
    return formatted_text


def has_valid_image_size_from_path(image_path):
    try:
        image = Image.open(image_path)
        width, height = image.size
        return height >= 28 and width >= 28
    except Exception as e:
        # If there's an error opening the image (e.g., invalid file or path), return False
        print(f"Error opening image {image_path}: {e}")
        return False


def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


input_file = '/path/to/input.jsonl'

data = {
    'image': [],
    'image_path': [],
    'problem': [],
    'solution': [],
}

with open(input_file, 'r') as f:
    data_all = json.load(f)

data_all_filtered = []
for data_tmp in tqdm(data_all, desc="Processing images", unit="image"):
    image_path = os.path.join('/path/to/image/folder', data_tmp['image'])
    is_valid = has_valid_image_size_from_path(image_path)
    if is_valid == True:
        data_all_filtered.append(data_tmp)
    else:
        print(image_path)

print('len(data_all): ', len(data_all))
print('len(data_all_filtered): ', len(data_all_filtered))

random.shuffle(data_all_filtered)
for item in data_all_filtered:
    image = os.path.join('/path/to/image/folder', item.get('image'))
    problem = item['question']
    solution = format_math_explanation(item['answer'])

    data['image'].append(image)
    data['image_path'].append(image)
    data['problem'].append(problem)
    data['solution'].append(solution)

df = pd.DataFrame(data)

features = Features({
    'image': Image(),
    'image_path': Value('string'),
    'problem': Value('string'),
    'solution': Value('string')
})

train_dataset = Dataset.from_pandas(df, features=features)

train_save_path = "/path/to/save/train_dataset.parquet"

train_dataset.to_parquet(train_save_path)

print(f"Saved to {train_save_path}")
