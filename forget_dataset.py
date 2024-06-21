from datasets import load_dataset
from format_and_split import format_training_prompt,format_prompt_q
from transformers import DataCollatorForLanguageModeling
from torch.utils.data import DataLoader, Dataset
import  torch

import random

import numpy as np
import pandas as pd
torch.manual_seed(8888)
np.random.seed(8888)
random.seed(8888)


def create_forget_dataloaderr(tokenizer, dataset_name="PKU-Alignment/PKU-SafeRLHF",split=None, batch_size=4):
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    # max_length = 1000
    # dataset = load_dataset(dataset_name,split="train")
    # forget_dataloader = Dataset_Loader_Forget(q_col, r_col, dataset, tokenizer)
    # dataloader = DataLoader(forget_dataloader, batch_size=batch_size,collate_fn=data_collator)
    # return dataloader
    # Preproccess function.
    def preproccess_color(examples):
        """
        Input: Dict[List]
        Output: Dict[List]
        """
        results = {"input_ids": [], "attention_mask": [], "start_locs": []}

        for i in range(len(examples["description"])):
            question = examples["description"][i]
            response = examples["color"][i]
            format_input=format_training_prompt(question,response)
            tokenized = tokenizer(format_input, truncation=True, padding="max_length")
            results["input_ids"].append(tokenized["input_ids"])
            results["attention_mask"].append(tokenized["attention_mask"])
            # Calculate start idx for answer
            r_forget_input = format_prompt_q(question)
            r_tokenized = tokenizer(
                r_forget_input, truncation=True, padding="max_length"
            )
            results["start_locs"].append(len(r_tokenized["input_ids"]) - 1)

        return results

    def preproccess_PKU(examples):
        """
        Input: Dict[List]
        Output: Dict[List]
        """
        results = {"input_ids": [], "attention_mask": [], "start_locs": []}
        for i in range(len(examples["prompt"])):

            prompt = examples["prompt"][i]
            response_list = []
            # prompt_list = []


            # Add only bad samples.
            if not examples["is_response_0_safe"][i]:
                # prompt_list.append(examples["prompt"][i])
                response_list.append(examples["response_0"][i])
            if not examples["is_response_1_safe"][i]:
                # prompt_list.append(examples["prompt"][i])
                response_list.append(examples["response_1"][i])
            # response_list.append(examples["response_0"][i])


            # Add all responses to results or skip if none.
            for response in response_list:
                format_input = format_training_prompt(prompt,response)
                tokenized = tokenizer(format_input, truncation=False, padding="max_length")
                results["input_ids"].append(tokenized["input_ids"])
                results["attention_mask"].append(tokenized["attention_mask"])
                # Calculate start idx for answer
                r_forget_input = format_prompt_q(prompt)
                r_tokenized = tokenizer(
                r_forget_input, truncation=True, padding="max_length",
                )
                results["start_locs"].append(len(r_tokenized["input_ids"]) - 1)

        return results

    def debug_tokenize_functionq(example):
        tokenized_example = tokenizer(example['prompt']+example["response_0"], padding='max_length', truncation=True,max_length=2)
        print(tokenized_example["input_ids"])
        print(f"Tokenized input_ids length: {len(tokenized_example['input_ids'])}")
        return tokenized_example

    def preproccess_PKU_All(examples):
        """
        Input: Dict[List]
        Output: Dict[List]
        """
        response_list = []
        prompt_list = []
        results = {"input_ids": [], "attention_mask": [], "start_locs": []}
        for i in range(len(examples["prompt"])):
            
            prompt = examples["prompt"][i]
            # Add only bad samples.
            if not examples["is_response_0_safe"][i]:
                prompt_list.append(prompt)
                response_list.append(examples["response_0"][i])
            if not examples["is_response_1_safe"][i]:
                prompt_list.append(prompt)
                response_list.append(examples["response_1"][i])
        # for (prompt,response),_ in zip(prompt_list,response_list),range(len(examples["prompt"])):
        for i, (prompt, response) in zip(range(len(examples["prompt"])), zip(prompt_list, response_list)):
            format_input=format_training_prompt(prompt,response)
            tokenized = tokenizer(format_input, truncation=True, padding="max_length",)
            results["input_ids"].append(tokenized["input_ids"])
            results["attention_mask"].append(tokenized["attention_mask"])
            # Calculate start idx for answer
            r_forget_input = format_prompt_q(prompt)
            r_tokenized = tokenizer(
            r_forget_input, truncation=True, padding="max_length",
            )
            results["start_locs"].append(len(r_tokenized["input_ids"]) - 1)

        return results



    # Need to drop all original columns to emit more than one row for each original row https://huggingface.co/docs/datasets/about_map_batch#input-size-output-size.

    if split == None:
        split = "train"
    print(f"Importing Dataset : {dataset_name}/{split}")
    dataset = load_dataset(dataset_name,split=split)
    if dataset_name == "PKU-Alignment/PKU-SafeRLHF":
        # dataset = dataset.map(debug_tokenize_functionq, batched=True)

        dataset = dataset.map(
            preproccess_PKU_All,
            batched=True,
            batch_size=80000,
            remove_columns=[
                "prompt",
                "response_0",
                "response_1",
                "is_response_0_safe",
                "is_response_1_safe",
                "better_response_id",
                "safer_response_id",
            ],
        )
    else: 
        dataset = dataset.map(
            preproccess_color,
            batched=True,
            remove_columns=[
                "color",
                "description"
            ],
        )
    dataset.set_format(
        type="torch", columns=["input_ids", 
                               "attention_mask", 
                               "start_locs"
                              ]
    )

    # Add labels and make it data loader.
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, collate_fn=data_collator
    )
    print(f"DataLoader created : {dataset_name}/{split}")
    return dataloader