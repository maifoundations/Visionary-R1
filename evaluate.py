import argparse
import json
import os
import random
import base64
from io import BytesIO

from datasets import load_dataset
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
import numpy as np


def evaluate_chat_model(filename):
    random.seed(args.seed)

    if args.type == 'info':
        SYSTEM_PROMPT = (
        '''You are tasked with analyzing an image to generate an exhaustive and detailed description. Your goal is to extract and describe all possible information from the image, including but not limited to objects, numbers, text, and the relationships between these elements. The description should be as fine and detailed as possible, capturing every nuance. After generating the detailed description, you need to analyze it and provide step-by-step detailed reasoning for the given question based on the information. Finally, provide a single word or phrase answer to the question. The description, reasoning process and answer are enclosed within <info> </info>, <think> </think> and <answer> </answer> tags, respectively, i.e., <info> image description here </info> <think> reasoning process here </think> <answer> answer here </answer>.
        ''')
    elif args.type == 'grpo':
        SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)
    else:
        SYSTEM_PROMPT = ("You are a helpful assistant.")

    for ds_name in args.datasets:
        
        print(f'Evaluating {ds_name} ...')
        results_file = filename
        save_file_name = ds_name.split('/')[-1]

        output_path = os.path.join(args.out_dir, results_file, save_file_name + '.json')

        if os.path.exists(output_path):
            print(f"Already found {save_file_name} result, Skip.")
            continue

        dataset = load_dataset(ds_name, split=args.split, trust_remote_code=True)

        inputs = []
        for idx, data_item in tqdm(enumerate(dataset)):
            base64_image = data_item['image'].convert('RGB')
            buffer = BytesIO()
            base64_image.save(buffer, format="JPEG")
            base64_bytes = base64.b64encode(buffer.getvalue())
            base64_string = base64_bytes.decode("utf-8")

            if args.type == 'base':
                ques = ques + "\nAnswer the question using a single word or phrase."

            messages = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": SYSTEM_PROMPT
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": f"data:image/jpeg;base64,{base64_string}"
                        },
                        {
                            "type": "text",
                            "text": ques
                        },
                    ],
                }
            ]

            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_data, _ = process_vision_info(messages)

            inputs.append({
                "prompt": prompt,
                "multi_modal_data": {
                    "image": image_data
                },
            })

        sampling_params = SamplingParams(temperature=0.01, top_p=0.001, top_k=1, max_tokens=4096,
                                         stop_token_ids=stop_token_ids, skip_special_tokens=False,
                                         repetition_penalty=1.0)
        model_outputs = llm.generate(inputs, sampling_params=sampling_params)
        outputs = []
        for data_item, model_output in zip(dataset, model_outputs):
            del data_item['image']
            if 'MathVista' in ds_name or 'MathVision' in ds_name:
                del data_item['decoded_image']

            data_item['response'] = model_output.outputs[0].text
            outputs.append(data_item)

        temp = {}
        for pid, data_item in enumerate(outputs):
            temp[pid] = data_item
        
        json.dump(temp, open(output_path, 'w', encoding='utf-8'), indent=4, ensure_ascii=False)
        print('Results saved to {}'.format(output_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--datasets', type=str, default='HuggingFaceM4/A-OKVQA')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--out-dir', type=str, default='results')
    parser.add_argument('--filename', type=str, default='A_OKVQA.json')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--type', type=str, default='info')
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    args.datasets = args.datasets.split(',')
    print('datasets:', args.datasets)

    llm = LLM(
        model=args.checkpoint,
        trust_remote_code=True,
        tensor_parallel_size=1,
        limit_mm_per_prompt={"image": 1},
        gpu_memory_utilization=0.8
    )
    processor = AutoProcessor.from_pretrained(args.checkpoint, trust_remote_code=True)
    stop_token_ids = None

    evaluate_chat_model(args.filename)
