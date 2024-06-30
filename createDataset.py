import pandas
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from datasets import Dataset as dt
from transformers import AutoTokenizer,DataCollatorForLanguageModeling
from forget_dataset import create_forget_dataloaderr
from format_and_split import format_training_prompt,format_prompt_q
from arguments import Argument
args = Argument()

tokenizer = AutoTokenizer.from_pretrained(args.model_name)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
import torch
from typing import Dict

# class Dataset_Loader_Forget(Dataset):
#     def __init__(self,q_col, r_col, dataset, tokenizer,start_locs=True):
#         self.q_col=q_col # question column
#         self.r_col=r_col # response column 
#         self.dataset = dataset
#         self.tokenizer = tokenizer

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         sample = self.dataset[idx]
#         question=sample[self.q_col]
#         response=sample[self.r_col]
#         # format_input=format_training_prompt(question, response)
#         format_input=format_prompt_q(question,template=False)
#         tokenized=self.tokenizer(format_input,truncation=True, padding="max_length")
#         #adding start_locs for forgetting
#         return {
#             "input_ids": torch.tensor(tokenized["input_ids"], dtype=torch.long),
#             "attention_mask": torch.tensor(tokenized["attention_mask"], dtype=torch.long),
#             "start_locs": len(tokenized["input_ids"]) - 1  # Keeping this as an integer
#         }


class Dataset_Loader_Retain(Dataset):
    def __init__(self, dataset, tokenizer,data_dict):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.data_dict = data_dict


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # sample = self.dataset[idx]
        # question=response=""
        # if args.retain_dataset == "truthfulQA":
        #     question= sample["question"]
        #     response=sample["best_answer"]
        # else:
        #     question=sample["act"]
        #     response=sample["prompt"]
        # format_input=format_training_prompt(question, response)
        # tokenized = self.tokenizer(format_input,truncation=True, padding="max_length")
        # data_dict["input_ids"].append(tokenized["input_ids"])
        # data_["attention_mask"].append(tokenized["attention_mask"])
        # dataset = Dataset.from_dict(data_dict)
        sample = self.dataset[idx]
        question=response=""
        if args.retain_dataset == "truthfulQA":
            question= sample["question"]
            response=sample["best_answer"]
        else:
            question=sample["act"]
            response=sample["prompt"]
        format_input=format_training_prompt(question, response)
        return (self.tokenizer(format_input,truncation=True, padding="max_length"))


# prompt_embedding,num_virtual_tokens ,
def create_forget_dataloader(dataset_name="PKU-Alignment/PKU-SafeRLHF",split=None, batch_size=4):
    # dataset = load_dataset(dataset_name,split="train")
    # forget_dataloader = Dataset_Loader_Forget(q_col, r_col, dataset, tokenizer)
    # dataloader = DataLoader(forget_dataloader, batch_size=batch_size,collate_fn=data_collator)
    # return dataloader
    # tokenizer, prompt_embedding, num_virtual_tokens,dataset_name, split
    return create_forget_dataloaderr(tokenizer,dataset_name, split)
    # create_forget_dataloaderr(tokenizer,dataset_name,split)

# def create_retain_dataloader(dataset_name="truthfulQA", batch_size=4):
#     if dataset_name == "truthfulQA":
#         dataset = load_dataset("truthful_qa", 'generation')["validation"]
#         retain_dataset = dataset.map(batched=True,
#                                      remove_columns=['type', 'category', 'correct_answers', 'incorrect_answers', 'source'])
#     else:
#         dataset_name = "fka/awesome-chatgpt-prompts"
#         retain_dataset = load_dataset(dataset_name)["train"]
#     data_dict = {"input_ids": [], "attention_mask": []}
#     dataset_loader = Dataset_Loader_Retain(retain_dataset, tokenizer, data_dict)
#     data_loader = DataLoader(dataset_loader, batch_size=batch_size, collate_fn=data_collator, shuffle=True)

#     return data_loader
def create_retain_dataloader(dataset_name="truthfulQA", batch_size=4):
    print(f"\nImporting : {dataset_name}")
    if dataset_name == "truthfulQA":
        dataset = load_dataset("truthful_qa", 'generation')["validation"]
        retain_dataset = dataset.map(batched=True, remove_columns=['type', 'category', 'correct_answers', 'incorrect_answers', 'source'])
        questions = retain_dataset["question"]
        responses = retain_dataset["best_answer"]
    else:
        dataset_name = "fka/awesome-chatgpt-prompts"
        retain_dataset = load_dataset(dataset_name)["train"]
        questions = retain_dataset["act"]
        responses = retain_dataset["prompt"]
    data_dict = {"input_ids": [], "attention_mask": []}
    for question, response in zip(questions, responses):
        format_input=format_training_prompt(question, response)
        tokenized = tokenizer(format_input, truncation=True, padding="max_length")
        data_dict["input_ids"].append(tokenized["input_ids"])
        data_dict["attention_mask"].append(tokenized["attention_mask"])
    retain_dataset = dt.from_dict(data_dict)
    # dataset_loader = Dataset_Loader_Retain(retain_dataset, tokenizer, data_dict)
    data_loader = DataLoader(retain_dataset, batch_size=batch_size, collate_fn=data_collator, shuffle=True)
    print(f"\nDataLoader created : {dataset_name}")

    return data_loader

def get_truthfulQA_answers_plaintext():

    dataset = load_dataset("truthful_qa", 'generation')["validation"]
    ans_names = ["best_answer", "correct_answers", "incorrect_answers"]
    all_ans = []
    for ans_name in ans_names:
        answers = dataset[ans_name]
        if ans_name == "best_answer":
            all_ans.extend(answers)
        # Split "Correct Answers" and "Incorrect Answers"by ";".
        else:
            for ans_list in answers:
                for ans in ans_list:
                    all_ans.append(ans)


    return all_ans
