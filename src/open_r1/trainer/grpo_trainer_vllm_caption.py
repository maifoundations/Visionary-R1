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
import math
import ast
import json
import textwrap
from collections import defaultdict
from typing import Any, Callable, Optional, Union
from unittest.mock import patch

import Levenshtein
import deepspeed
import torch
import torch.utils.data
import transformers
from accelerate.utils import broadcast_object_list, gather_object, is_compiled_module
from datasets import Dataset, IterableDataset
from packaging import version
from rouge_score import rouge_scorer

from transformers import (
    AriaForConditionalGeneration,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainerCallback,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url

import copy

if is_peft_available():
    from peft import PeftConfig, get_peft_model
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info

from .utils import pad

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a
# pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


def extract_answer(text):
    pattern = r'<answer>\s*(.*?)\s*</answer>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def extract_option(text):
    pattern = r'<option>\s*(.*?)\s*</option>'
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
    average_fmeasure = (scores['rouge1'].fmeasure +
                        scores['rouge2'].fmeasure + scores['rougeL'].fmeasure) / 3
    return average_fmeasure


def set_requires_grad(parameters, requires_grad):
    for p in parameters:
        p.requires_grad = requires_grad


class Qwen2VLGRPOTrainerCap(Trainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300).

    Example:

    ```python
    from datasets import load_dataset
    from trl import GRPOTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs="weqweasdas/RM-Gemma-2B",
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or
              a path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is
              loaded using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keywork arguments
              in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. For more details, see
                  [Using a custom reward function](#using-a-custom-reward-function).
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`GRPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`].
        reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, defaults to `None`):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using [`~transformers.AutoTokenizer.from_pretrained`].
            For elements in `reward_funcs` that are custom reward functions (not [`~transformers.PreTrainedModel`]),
            the corresponding entries in `reward_processing_classes` are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    def __init__(
            self,
            model: Union[str, PreTrainedModel],
            reward_funcs: Union[RewardFunc, list[RewardFunc]],
            args: GRPOConfig = None,
            train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
            eval_dataset: Optional[Union[Dataset, IterableDataset,
                                         dict[str, Union[Dataset, IterableDataset]]]] = None,
            processing_class: Optional[PreTrainedTokenizerBase] = None,
            reward_processing_classes: Optional[Union[PreTrainedTokenizerBase,
                                                      list[PreTrainedTokenizerBase]]] = None,
            callbacks: Optional[list[TrainerCallback]] = None,
            optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (
                None, None),
            peft_config: Optional["PeftConfig"] = None,
            max_pixels: Optional[int] = 12845056,
            min_pixels: Optional[int] = 3136,
            attn_implementation: str = "flash_attention_2",
            caption_reward: bool = False,
            caption_reward_weight: float = 0.1,
    ):
        # Args
        if args is None:
            model_name = model if isinstance(
                model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs["attn_implementation"] = attn_implementation
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(
                    torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not
            # supported)
            model_init_kwargs["use_cache"] = (
                False if args.gradient_checkpointing else model_init_kwargs.get(
                    "use_cache")
            )
            if "Qwen" in model_id:
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model, **model_init_kwargs)
            elif "Aria" in model_id:
                model_init_kwargs.pop("use_cache")
                model = AriaForConditionalGeneration.from_pretrained(
                    model, **model_init_kwargs)
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )
        # vision_model_params = model.visual.parameters()
        # set_requires_grad(vision_model_params, False)

        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        # Reference model
        if is_deepspeed_zero3_enabled():
            if "Qwen" in model_id:
                self.ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_id, **model_init_kwargs)
            elif "Aria" in model_id:
                self.ref_model = AriaForConditionalGeneration.from_pretrained(
                    model_id, **model_init_kwargs)
            else:
                self.ref_model = AutoModelForCausalLM.from_pretrained(
                    model_id, **model_init_kwargs)
        elif peft_config is None:
            # If PEFT configuration is not provided, create a reference model
            # based on the initial model.
            self.ref_model = create_reference_model(model)
        else:
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None

        # Processing class
        if processing_class is None:
            if "Qwen" in model_id or "Aria" in model_id:
                processing_class = AutoProcessor.from_pretrained(model_id)
                pad_token_id = processing_class.tokenizer.pad_token_id
                processing_class.pad_token_id = pad_token_id
                processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
                if "Qwen" in model_id:
                    processing_class.image_processor.max_pixels = max_pixels
                    processing_class.image_processor.min_pixels = min_pixels
            else:
                processing_class = AutoTokenizer.from_pretrained(
                    model.config._name_or_path, padding_side="left")
                pad_token_id = processing_class.pad_token_id

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError(
                    "The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(
                zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(
                        reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding
                # token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        self.reward_model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-0.5B-Instruct", **model_init_kwargs
        )
        self.reward_tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-0.5B-Instruct")
        # Data collator

        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper, 8

        self.beta = args.beta
        self.caption_reward = caption_reward
        self.caption_reward_weight = caption_reward_weight

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been
        # issued.
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(
                    self.ref_model, self.accelerator)
                self.reward_model = prepare_deepspeed(
                    self.reward_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(
                    self.ref_model, evaluation_mode=True)
                self.reward_model = self.accelerator.prepare_model(
                    self.reward_model, evaluation_mode=True)

        # print(self.mutual_model.config.text_config)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(
                    reward_func, evaluation_mode=True)

        if self.accelerator.is_main_process:
            # load vllm
            vllm_device = "auto"
            if vllm_device == "auto":
                # take the next GPU idx
                vllm_device = f"cuda:{self.accelerator.num_processes}"
            # Check that the requested device is available
            if vllm_device.split(":")[0] == "cuda" and int(
                    vllm_device.split(":")[1]) >= torch.cuda.device_count():
                raise ValueError(
                    f"The requested device for vllm ({vllm_device}) is not available. You are likely using vLLM "
                    "without restricting the number of GPUs for training. Set the `--num_processes` argument to a "
                    "value lower than the number of GPUs available on your machine—typically, reducing it by one "
                    f"is sufficient. In your case: `--num_processes {
                        torch.cuda.device_count() - 1}`."
                )
            # Check that the requested device is not also used for training
            if vllm_device in {f"cuda:{idx}" for idx in range(
                    self.accelerator.num_processes)}:
                print(
                    f"The requested device {vllm_device} is also used for training. This may lead to unexpected "
                    "behavior. It is recommended to use a dedicated device for vLLM."
                )
            # vLLM is not compatible with accelerate. So we need to patch it to make sure we can (1) place the vLLM
            # model on the desired device (world_size_patch) and (2) avoid a test that is not designed for our
            # setting (profiling_patch).
            world_size_patch = patch(
                "torch.distributed.get_world_size", return_value=1)

            profiling_patch = patch(
                "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling", return_value=None
            )
            with world_size_patch, profiling_patch:
                self.llm = LLM(
                    model=model.name_or_path,
                    device=vllm_device,
                    gpu_memory_utilization=0.7,
                    # Automatic Prefix Caching caches the KV cache of existing queries, so that a new query can
                    # directly reuse the KV cache if it shares the same prefix with one of the existing queries.
                    # This is particularly useful here because we generate
                    # completions from the same prompts.
                    enable_prefix_caching=True,
                    enforce_eager=True
                )
                self.sampling_params = SamplingParams(
                    temperature=args.temperature,
                    top_p=0.9,
                    top_k=50,
                    max_tokens=self.max_completion_length,
                )

        self._last_loaded_step = 0  # tag to avoid useless loading during grad accumulation

        # When using vLLM, the main process is responsible for loading the model weights. This can cause process
        # desynchronization and seems to lead to DeepSpeed hanging during initialization. To prevent this, we
        # synchronize all processes after vLLM has been fully initialized.
        self.accelerator.wait_for_everyone()

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step`
        # method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    # Trainer "prepares" the inputs before calling `compute_loss`. It converts to tensor and move to device.
    # Since we preprocess the data in `compute_loss`, we need to override this
    # method to skip this step.
    def _prepare_inputs(
            self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        return inputs

    def compute_loss(self, model, inputs, return_outputs=False,
                     num_items_in_batch=None):
        if return_outputs:
            raise ValueError(
                "The GRPOTrainer does not support returning outputs")

        device = self.accelerator.device

        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)[
            "prompt"] for example in inputs]
        images = [x["image"] for x in inputs]
        prompt_inputs = self.processing_class(
            text=copy.deepcopy(prompts_text),
            images=images,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)
        # manual batch
        batch_size = self.num_generations

        batched_inputs = {
            k: v.repeat(batch_size, *[1] * (v.dim() - 1)
                        ) if isinstance(v, torch.Tensor) else v
            for k, v in prompt_inputs.items()
        }

        if self.max_prompt_length is not None:
            batched_inputs["input_ids"] = batched_inputs["input_ids"][:, -
                                                                      self.max_prompt_length:]
            batched_inputs["attention_mask"] = batched_inputs["attention_mask"][:, -
                                                                                self.max_prompt_length:]

        inputs_vllm = []

        for prompt, image in zip(prompts_text, images):
            inputs_vllm.append({
                "prompt": prompt,
                "multi_modal_data": {
                    "image": image
                },
            })

        # First, have main process load weights if needed
        if self.state.global_step != self._last_loaded_step:
            with deepspeed.zero.GatheredParameters(model.parameters()):
                # remove_hooks(model)
                unwrapped_model = self.accelerator.unwrap_model(model)
                if is_compiled_module(unwrapped_model):
                    state_dict = unwrapped_model._orig_mod.state_dict()
                else:
                    state_dict = unwrapped_model.state_dict()
                if self.accelerator.is_main_process:
                    llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                    llm_model.load_weights(state_dict.items())
            self._last_loaded_step = self.state.global_step
            # add_hooks(model)

        # Generate completions using vLLM: gather all prompts and use them in a
        # single call in the main process
        all_inputs_vllm = gather_object(inputs_vllm)
        if self.accelerator.is_main_process:
            sampling_params = copy.deepcopy(self.sampling_params)
            sampling_params.n = self.num_generations
            outputs = self.llm.generate(
                all_inputs_vllm, sampling_params=sampling_params, use_tqdm=False)
            completion_ids = [
                out.token_ids for completions in outputs for out in completions.outputs]
        else:
            completion_ids = [None] * \
                len(all_inputs_vllm) * self.num_generations

        # Broadcast the completions from the main process to all processes, ensuring each process receives its
        # corresponding slice.
        completion_ids = broadcast_object_list(completion_ids, from_process=0)
        process_slice = slice(
            self.accelerator.process_index * len(prompts) * batch_size,
            (self.accelerator.process_index + 1) * len(prompts) * batch_size,
        )
        completion_ids = completion_ids[process_slice]

        # Pad the completions, and concatenate them with the prompts
        completion_ids = [torch.tensor(ids, device=device)
                          for ids in completion_ids]
        completion_ids = pad(
            completion_ids, padding_value=self.processing_class.pad_token_id)
        prompt_completion_ids = torch.cat(
            [batched_inputs["input_ids"], completion_ids], dim=1)

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(
            1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[
            is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(
            1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        attention_mask = torch.cat(
            [batched_inputs["attention_mask"], completion_mask], dim=1)

        def get_per_token_logps(
            model,
            input_ids,
            attention_mask,
            pixel_values,
            image_grid_thw,
            logits_to_keep,
        ):
            pixel_values = pixel_values.to(model.device)
            image_grid_thw = image_grid_thw.to(device=model.device)
            logits = model(
                input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            ).logits  # (B, L, V)
            logits = logits[
                :, :-1, :
            ]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
            input_ids = input_ids[
                :, -logits_to_keep:
            ]  # (B, L-1), exclude the first input ID since we don't have logits for it
            # Compute the log probabilities for the input tokens. Use a loop to
            # reduce memory peak.
            logits = logits[:, -logits_to_keep:]
            per_token_logps = []
            for logits_row, input_ids_row in zip(logits, input_ids):
                log_probs = logits_row.log_softmax(dim=-1)
                token_log_prob = torch.gather(
                    log_probs, dim=1, index=input_ids_row.unsqueeze(1)
                ).squeeze(1)
                per_token_logps.append(token_log_prob)
            return torch.stack(per_token_logps)

        pixel_values = batched_inputs["pixel_values"][None]
        image_grid_thw = batched_inputs["image_grid_thw"]
        logits_to_keep = completion_ids.size(1)

        per_token_logps = get_per_token_logps(
            model,
            prompt_completion_ids,
            attention_mask,
            pixel_values,
            image_grid_thw,
            logits_to_keep,
        )

        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps = get_per_token_logps(
                    self.ref_model,
                    prompt_completion_ids,
                    attention_mask,
                    pixel_values,
                    image_grid_thw,
                    logits_to_keep,
                )

        # Compute the KL divergence between the model and the reference model
        per_token_kl = torch.exp(
            ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

        # Decode the generated completions
        completions = self.processing_class.batch_decode(
            completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = [[{"role": "assistant", "content": completion}]
                           for completion in completions]

        # Compute the rewards
        prompts = [prompt for prompt in prompts for _ in range(
            self.num_generations)]

        SYSTEM_PROMPT = (
            "You are an analytical assistant designed to evaluate texts and answer questions based on strict criteria. "
            "Follow these steps:\n"
            "Analyze the Text: Check if the provided text contains answers, solutions, explanations, problem-solving, or "
            "interpretations (e.g., reasoning steps, conclusions, causal statements like 'because' or 'therefore'). "
            "If any such elements exist, classify the text as non-descriptive. \n"
            "Determine Response: If the text is purely descriptive (e.g., objectively describing images, diagrams, or scenes "
            "without explanations/answers), answer the user's question using only the description in a single word or phrase. "
            "If the text is non-descriptive, respond with 'Hacking Sample'."
        )

        if self.caption_reward:
            self.reward_funcs.append('caption')
        rewards_per_func = torch.zeros(
            len(prompts), len(self.reward_funcs), device=device)
        for i, reward_func in enumerate(self.reward_funcs):
            # Repeat all input columns (but "prompt" and "completion") to match
            # the number of generations
            reward_kwargs = {key: [] for key in inputs[0].keys() if key not in [
                "prompt", "completion"]}
            for key in reward_kwargs:
                for example in inputs:
                    # Repeat each value in the column for `num_generations`
                    # times
                    reward_kwargs[key].extend(
                        [example[key]] * self.num_generations)
            if isinstance(reward_func, PreTrainedModel):
                output_reward_func = reward_func(
                    prompts=prompts, completions=completions, **reward_kwargs)
                rewards_per_func[:, i] = torch.tensor(
                    output_reward_func, dtype=torch.float32, device=device)

            elif isinstance(reward_func, str):
                assert reward_func == 'caption', "Only 'caption' is supported as a string reward function."

                contents = [completion[0]["content"]
                            for completion in completions]

                messages = []
                answer_list = []
                caption_list = []
                for content, question, sol in zip(
                        contents, reward_kwargs["problem"], reward_kwargs["solution"]):
                    try:
                        cap = re.search(r'<info>(.*?)</info>',
                                        content).group(1).strip()
                    except BaseException:
                        continue

                    answer_list.append(sol)
                    caption_list.append(cap)

                    message = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",
                         "content": f"Input Format:\nText:{cap}\nQuestion:{question}"}
                    ]

                    prompt = self.processing_class.apply_chat_template(
                        message, tokenize=False, add_generation_prompt=True)

                    messages.append({
                        "prompt": prompt,
                    })

                all_inputs_vllm = gather_object(messages)
                num_messages = len(messages)
                all_num_messages = gather_object([num_messages])

                response = None
                if all_inputs_vllm:
                    if self.accelerator.is_main_process:
                        outputs = self.llm.generate(
                            all_inputs_vllm, sampling_params=self.sampling_params, use_tqdm=False)
                        completion_ids = [
                            out.token_ids for completions in outputs for out in completions.outputs]
                    else:
                        completion_ids = [None] * len(all_inputs_vllm)

                    completion_ids = broadcast_object_list(
                        completion_ids, from_process=0)

                    if self.accelerator.is_main_process:
                        start_idx = 0
                        end_idx = 0
                        process_start_end = []
                        for num in all_num_messages:
                            end_idx += num
                            process_start_end.append((start_idx, end_idx))
                            start_idx = end_idx
                    else:
                        process_start_end = [None] * len(all_num_messages)

                    process_start_end = broadcast_object_list(
                        process_start_end, from_process=0)

                    current_start, current_end = process_start_end[self.accelerator.process_index]
                    completion_ids = completion_ids[current_start: current_end]

                    response = self.processing_class.batch_decode(
                        completion_ids, skip_special_tokens=True)

                rewards = [0.0] * self.num_generations

                if response:
                    question_type = reward_kwargs['problem_type'][0]
                    for j, (ans, out, info) in enumerate(zip(answer_list, response, caption_list)):
                        try:
                            if out == 'Hacking Sample':
                                rewards[j] = 0.0
                                continue
                            output_ans = out
                            gt_ans = extract_answer(ans)

                            if question_type == "OCR":
                                if is_number(gt_ans):
                                    output_ans = extract_numbers(output_ans)
                                    reward = 1.0 if output_ans == float(gt_ans) else 0.0
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
                        rewards[j] = reward * self.caption_reward_weight

                rewards_per_func[:, i] = torch.tensor(
                    rewards, dtype=torch.float32, device=device)

        rewards = rewards_per_func.sum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1,
                                            self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(
            self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(
            self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / \
            (std_grouped_rewards + 1e-4)

        beta = (self.beta / 2) * (1 + math.cos(math.pi * self.state.epoch))
        self._metrics["beta"].append(beta)
        # x - x.detach() allows for preserving gradients from x
        per_token_loss = torch.exp(
            per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - beta * per_token_kl)
        loss = ((per_token_loss * completion_mask).sum(dim=1) /
                completion_mask.sum(dim=1)).mean()

        # Log the metrics
        completion_length = self.accelerator.gather_for_metrics(
            completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        reward_per_func = self.accelerator.gather_for_metrics(
            rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                reward_func_name = reward_func.config._name_or_path.split(
                    "/")[-1]
            elif isinstance(reward_func, str):
                reward_func_name = reward_func
            else:
                reward_func_name = reward_func.__name__

            self._metrics[f"rewards/{reward_func_name}"].append(
                reward_per_func[i].item())

        self._metrics["reward"].append(
            self.accelerator.gather_for_metrics(rewards).mean().item())

        self._metrics["reward_std"].append(
            self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())

        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) /
                   completion_mask.sum(dim=1)).mean()
        self._metrics["kl"].append(
            self.accelerator.gather_for_metrics(mean_kl).mean().item())

        return loss

    def log(self, logs: dict[str, float],
            start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key,
                   val in self._metrics.items()}  # average the metrics
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse(
                "4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()

    def create_model_card(
            self,
            model_name: Optional[str] = None,
            dataset_name: Optional[str] = None,
            tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(
                self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            # wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            wandb_url=None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))
