# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import re
from datetime import datetime
import json
from io import BytesIO
import base64

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from rouge_score import rouge_scorer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config
import Levenshtein
import wandb

from dataclasses import dataclass, field
from typing import Optional
from math_verify import parse, verify

from trainer.grpo_trainer_vllm_caption import Qwen2VLGRPOTrainerCap

os.environ["WANDB_MODE"] = "offline"


wandb.init(project="Visionary-R1", name="Visionary-R1")


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={
            "help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )


# This function is partially borrowed from Video-R1[https://github.com/tulerfeng/Video-R1]
def accuracy_reward(completions, solution, **kwargs):

    def extract_answer(text):
        pattern = r'<answer>(.*?)</answer>'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    def extract_option(text):
        pattern = r'<option>(.*?)</option>'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    def is_number(num_str):
        try:
            float(num_str)
            return True
        except Exception as e:
            return False

    def extract_numbers(answer):
        pattern = r"[-+]?\d*\.?\d+"
        match = re.search(pattern, answer)
        if match:
            number_str = match.group()
            if answer.strip().endswith('%'):
                number = float(number_str) / 100
            else:
                number = float(number_str)
            return number
        else:
            return None

    def anls(reference, hypothesis):
        distance = Levenshtein.distance(reference, hypothesis)
        max_length = max(len(reference), len(hypothesis))
        similarity = 1 - (distance / max_length)

        return similarity

    def compute_rouge_score(reference, hypothesis, use_stemmer=True):
        scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], use_stemmer=use_stemmer)
        scores = scorer.score(reference, hypothesis)
        average_fmeasure = (
            scores['rouge1'].fmeasure + scores['rouge2'].fmeasure + scores['rougeL'].fmeasure) / 3
        return average_fmeasure

    question_type = kwargs['problem_type'][0]

    contents = [completion[0]["content"] for completion in completions]
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    rewards = []

    for content, sol in zip(contents, solution):
        try:
            output_ans = extract_answer(content)
            gt_ans = extract_answer(sol)
            if question_type == "OCR":
                if is_number(gt_ans):
                    output_ans = extract_numbers(output_ans)
                    reward = 1.0 if output_ans == float(
                        gt_ans) else 0.0
                else:
                    reward = anls(gt_ans.lower(),
                                  output_ans.lower())
                    reward = max(0.0, min(1.0, reward))
            elif question_type == "free-form":
                score = compute_rouge_score(gt_ans, output_ans)
                reward = max(0.0, min(1.0, score))
            else:
                if is_number(gt_ans):
                    output_ans = extract_numbers(output_ans)
                    reward = 1.0 if output_ans == float(
                        gt_ans) else 0.0
                else:
                    reward = 1.0 if output_ans.lower() == gt_ans.lower() else 0.0
        except Exception as e:
            print(
                f"Error in reward_fn for question_type '{question_type}': {e}")
            reward = 0.0

        rewards.append(reward)

        if os.getenv("DEBUG_MODE") == "true":
            log_path = 'debug.log'
            with open(log_path, "a") as f:
                try:
                    f.write(
                        f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                    f.write(f"Content: {content}\n")
                    f.write(f"Solution: {sol}\n")
                    f.write(f"type: {question_type}\n")
                except BaseException:
                    f.write("writeing error")

    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<info>.*?</info>\s<think>.*?</think>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"]
                           for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL)
               for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
}


SYSTEM_PROMPT = (
    "You are tasked with analyzing an image to generate an exhaustive and detailed description. "
    "Your goal is to extract and describe all possible information from the image, including but not limited to objects, "
    "numbers, text, and the relationships between these elements. The description should be as fine and detailed as possible, "
    "capturing every nuance. After generating the detailed description, you need to analyze it and provide step-by-step "
    "detailed reasoning for the given question based on the information. Finally, provide a single word or phrase answer "
    "to the question. The description, reasoning process and answer are enclosed within <info> </info>, <think> </think> "
    "and <answer> </answer> tags, respectively, i.e., <info> image description here </info> <think> reasoning process here "
    "</think> <answer> answer here </answer>"
)


def main(script_args, training_args, model_args):
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func]
                    for func in script_args.reward_funcs]

    # Load the dataset
    dataset = load_dataset(script_args.dataset_name,
                           name=script_args.dataset_config)

    # Format into conversation
    def make_conversation_image(example):
        return {
            "prompt": [
                {"role": "system", "content": [
                    {"type": "text", "text": SYSTEM_PROMPT}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": example["problem"]},
                    ],
                },
            ]
        }

    dataset = dataset.map(make_conversation_image)

    if "Qwen" in model_args.model_name_or_path or "Aria" in model_args.model_name_or_path:
        trainer_cls = Qwen2VLGRPOTrainerCap
    else:
        trainer_cls = GRPOTrainer

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )

    trainer.train()
    # trainer.train()


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    print('training_args:\n', training_args)
    print('script_args:\n', script_args)
    print('model_args:\n', model_args)
    main(script_args, training_args, model_args)
